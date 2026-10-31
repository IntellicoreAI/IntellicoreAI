
import json

import pytest

from app.monitoring import alert_engine


def make_reports(
    tmp_path,
    *,
    data_drift=False,
    prediction_drift=False,
    performance_degradation=False,
    data_quality_status="passed",
    evaluation_context=None,
):
    """Create isolated report files for alert-engine tests."""

    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir()

    drift_report = {
        "overall_summary": {
            "overall_status": (
                "drift_detected"
                if data_drift
                else "no_significant_drift"
            ),
            "drifted_features": 1 if data_drift else 0,
            "total_features": 10,
            "drift_rate": 0.1 if data_drift else 0.0,
            "drifted_feature_names": (
                ["age"] if data_drift else []
            ),
        }
    }

    prediction_report = {
        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },
        "comparison": {
            "drift_detected": prediction_drift,
            "ks_statistic": 0.2 if prediction_drift else 0.0,
            "p_value": 0.01 if prediction_drift else 0.5,
            "positive_prediction_rate_difference": 0.05,
        },
    }

    performance_report = {
        "summary": {
            "performance_degradation_detected": (
                performance_degradation
            ),
            "degraded_metric_count": (
                1 if performance_degradation else 0
            ),
            "total_metrics_evaluated": 8,
        },
        "degraded_metrics": (
            ["recall"] if performance_degradation else []
        ),
    }

    if evaluation_context is not None:
        performance_report["evaluation_context"] = evaluation_context

    quality_checks = {
        "schema": {"status": "passed"},
        "numeric_values": {"status": "passed"},
        "missing_values": {"status": "passed"},
        "duplicates": {"status": "passed"},
        "freshness": {
            "status": (
                "failed"
                if data_quality_status == "failed"
                else "warning"
                if data_quality_status == "warning"
                else "passed"
            )
        },
    }

    quality_report = {
        "overall_status": data_quality_status,
        "checks": quality_checks,
    }

    report_paths = {
        "DRIFT_REPORT_PATH": monitoring_dir / "drift_report.json",
        "PREDICTION_DRIFT_REPORT_PATH": (
            monitoring_dir / "prediction_drift_report.json"
        ),
        "PERFORMANCE_REPORT_PATH": (
            monitoring_dir / "performance_degradation_report.json"
        ),
        "DATA_QUALITY_REPORT_PATH": (
            monitoring_dir / "data_quality_report.json"
        ),
        "OUTPUT_PATH": monitoring_dir / "unified_alert.json",
    }

    reports = {
        "DRIFT_REPORT_PATH": drift_report,
        "PREDICTION_DRIFT_REPORT_PATH": prediction_report,
        "PERFORMANCE_REPORT_PATH": performance_report,
        "DATA_QUALITY_REPORT_PATH": quality_report,
    }

    for name, report in reports.items():
        report_paths[name].write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    # Point the module at temporary files rather than project data.
    for name, path in report_paths.items():
        setattr(alert_engine, name, path)

    return report_paths


def test_data_quality_failure_triggers_alert(tmp_path, monkeypatch):
    make_reports(
        tmp_path,
        data_quality_status="failed",
    )

    alert = alert_engine.generate_unified_alert()

    assert alert["overall_status"] == "alert"
    assert alert["summary"]["data_quality_failed"] is True
    assert alert["severity"] == "MEDIUM"

    quality_alert = next(
        item
        for item in alert["alerts"]
        if item["type"] == "data_quality"
    )

    assert quality_alert["status"] == "failed"
    assert quality_alert["failed_checks"] == ["freshness"]


def test_data_quality_warning_does_not_trigger_alert(tmp_path):
    make_reports(
        tmp_path,
        data_quality_status="warning",
    )

    alert = alert_engine.generate_unified_alert()

    assert alert["overall_status"] == "no_alert"
    assert alert["summary"]["data_quality_warning"] is True
    assert alert["summary"]["data_quality_failed"] is False

    quality_alert = next(
        item
        for item in alert["alerts"]
        if item["type"] == "data_quality"
    )

    assert quality_alert["status"] == "warning"
    assert quality_alert["warning_checks"] == ["freshness"]


def test_existing_drift_alerts_are_preserved(tmp_path):
    make_reports(
        tmp_path,
        data_drift=True,
        prediction_drift=True,
    )

    alert = alert_engine.generate_unified_alert()

    assert alert["overall_status"] == "alert"
    assert alert["severity"] == "HIGH"
    assert alert["summary"]["data_drift_detected"] is True
    assert alert["summary"]["prediction_drift_detected"] is True


def test_performance_degradation_remains_high_severity(tmp_path):
    make_reports(
        tmp_path,
        performance_degradation=True,
    )

    alert = alert_engine.generate_unified_alert()

    assert alert["overall_status"] == "alert"
    assert alert["severity"] == "HIGH"
    assert (
        alert["summary"]["performance_degradation_detected"]
        is True
    )


def test_controlled_degradation_is_not_an_operational_alert(tmp_path):
    make_reports(
        tmp_path,
        performance_degradation=True,
        evaluation_context="controlled_degradation_test",
    )

    alert = alert_engine.generate_unified_alert()

    assert alert["overall_status"] == "no_alert"
    assert alert["severity"] == "INFO"
    assert alert["summary"]["performance_degradation_detected"] is False
    assert (
        alert["summary"]["performance_degradation_report_detected"]
        is True
    )
    assert (
        alert["summary"]["performance_degradation_controlled_test"]
        is True
    )
    assert alert["summary"]["total_alert_conditions"] == 0

    performance_alert = next(
        item
        for item in alert["alerts"]
        if item["type"] == "performance_degradation"
    )

    assert performance_alert["status"] == "test_detected"
    assert performance_alert["detected_in_report"] is True
    assert performance_alert["degraded_metric_count"] == 1


def test_all_clear_returns_no_alert(tmp_path):
    make_reports(tmp_path)

    alert = alert_engine.generate_unified_alert()

    assert alert["overall_status"] == "no_alert"
    assert alert["severity"] == "INFO"
    assert alert["summary"]["total_alert_conditions"] == 0

    assert len(alert["alerts"]) == 4


def test_missing_report_raises_clear_error(tmp_path):
    paths = make_reports(tmp_path)

    paths["DATA_QUALITY_REPORT_PATH"].unlink()

    with pytest.raises(
        FileNotFoundError,
        match="data_quality_report.json",
    ):
        alert_engine.generate_unified_alert()
