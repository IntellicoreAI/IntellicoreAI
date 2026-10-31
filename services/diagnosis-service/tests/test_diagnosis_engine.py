
from app.diagnosis_engine import generate_diagnosis


def make_alert(alerts):
    return {
        "alert_id": "test-alert-001",
        "severity": "HIGH",
        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },
        "alerts": alerts,
    }


def test_undetected_data_drift_is_not_diagnosed():
    alert = make_alert([
        {
            "type": "data_drift",
            "status": "not_detected",
            "drifted_feature_count": 0,
            "total_feature_count": 10,
            "drift_rate": 0.0,
            "drifted_features": [],
        }
    ])

    result = generate_diagnosis(alert)

    assert result["overall_status"] == "no_issue_detected"
    assert result["diagnoses"] == []


def test_prediction_drift_is_diagnosed():
    alert = make_alert([
        {
            "type": "prediction_drift",
            "status": "detected",
            "ks_statistic": 0.92,
            "p_value": 0.001,
            "positive_prediction_rate_difference": 0.04,
        }
    ])

    result = generate_diagnosis(alert)

    assert result["overall_status"] == "issues_detected"
    assert len(result["diagnoses"]) == 1
    assert result["diagnoses"][0]["issue"] == (
        "prediction_distribution_change"
    )


def test_data_quality_failure_includes_failed_checks():
    alert = make_alert([
        {
            "type": "data_quality",
            "status": "failed",
            "failed_checks": ["freshness"],
            "warning_checks": [],
            "checks": {
                "freshness": {
                    "status": "failed",
                    "issues": ["data_stale"],
                }
            },
        }
    ])

    result = generate_diagnosis(alert)

    diagnosis = result["diagnoses"][0]

    assert result["overall_status"] == "issues_detected"
    assert diagnosis["issue"] == "telemetry_data_quality_issue"
    assert diagnosis["status"] == "failed"
    assert diagnosis["evidence"]["failed_checks"] == ["freshness"]


def test_data_quality_warning_is_not_a_confirmed_failure():
    alert = make_alert([
        {
            "type": "data_quality",
            "status": "warning",
            "failed_checks": [],
            "warning_checks": ["duplicates"],
            "checks": {},
        }
    ])

    result = generate_diagnosis(alert)

    assert result["overall_status"] == "no_issue_detected"
    assert result["diagnoses"][0]["status"] == "warning"


def test_empty_alert_produces_no_diagnoses():
    result = generate_diagnosis(make_alert([]))

    assert result["overall_status"] == "no_issue_detected"
    assert result["diagnoses"] == []


def test_performance_degradation_is_diagnosed():
    alert = make_alert([
        {
            "type": "performance_degradation",
            "status": "detected",
            "degraded_metric_count": 2,
            "total_metrics_evaluated": 8,
            "degraded_metrics": ["recall", "f1_score"],
        }
    ])

    result = generate_diagnosis(alert)

    assert result["overall_status"] == "issues_detected"
    assert result["diagnoses"][0]["issue"] == (
        "model_performance_degradation"
    )
    assert result["diagnoses"][0]["evidence"]["degraded_metric_count"] == 2
