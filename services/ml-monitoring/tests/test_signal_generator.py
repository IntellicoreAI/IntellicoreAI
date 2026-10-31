import pytest

from app.signals.generator import generate_signals


def make_diagnosis(issue, status="detected", evidence=None):
    """Create a sample diagnosis for isolated signal tests."""

    return {
        "diagnosis_id": "diagnosis-test-001",
        "source_alert_id": "alert-test-001",
        "alert_severity": "MEDIUM",
        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },
        "diagnoses": [
            {
                "issue": issue,
                "status": status,
                "investigation_area": "model_monitoring",
                "evidence": evidence or {"test_metric": 0.85},
            }
        ],
    }


@pytest.mark.parametrize(
    "issue, expected_signal_type",
    [
        ("telemetry_data_quality_issue", "data_quality"),
        ("data_distribution_change", "data_drift"),
        ("prediction_distribution_change", "prediction_drift"),
        ("model_performance_degradation", "performance_degradation"),
    ],
)
def test_signal_type_mapping(issue, expected_signal_type):
    diagnosis = make_diagnosis(issue)

    result = generate_signals(diagnosis)

    assert result["signal_count"] == 1
    assert result["signals"][0]["signal_type"] == expected_signal_type
    assert result["signals"][0]["issue"] == issue


def test_signal_contains_required_key_value_pairs():
    diagnosis = make_diagnosis("telemetry_data_quality_issue")

    result = generate_signals(diagnosis)
    signal = result["signals"][0]

    assert signal["schema_version"] == "1.0"
    assert signal["signal_id"].startswith("ml-signal-")
    assert signal["source"] == "ml_monitoring.diagnosis_engine"
    assert signal["status"] == "detected"
    assert signal["project_id"] == "intellicore-credit-risk"
    assert signal["environment"] == "development"
    assert signal["model"]["model_name"] == "xgboost_credit_risk"
    assert signal["model"]["model_version"] == "1.0"
    assert signal["diagnosis_id"] == "diagnosis-test-001"
    assert signal["source_alert_id"] == "alert-test-001"


def test_evidence_is_preserved():
    evidence = {
        "failed_checks": ["freshness"],
        "age_seconds": 2493,
        "max_age_seconds": 900,
    }

    diagnosis = make_diagnosis(
        "telemetry_data_quality_issue",
        evidence=evidence,
    )

    result = generate_signals(diagnosis)

    assert result["signals"][0]["evidence"] == evidence


def test_warning_signal_has_low_severity():
    diagnosis = make_diagnosis(
        "telemetry_data_quality_issue",
        status="warning",
    )

    result = generate_signals(diagnosis)
    signal = result["signals"][0]

    assert signal["status"] == "warning"
    assert signal["severity"] == "LOW"


def test_performance_degradation_has_high_severity():
    diagnosis = make_diagnosis("model_performance_degradation")

    result = generate_signals(diagnosis)

    assert result["signals"][0]["severity"] == "HIGH"


def test_root_cause_is_not_assumed():
    diagnosis = make_diagnosis("telemetry_data_quality_issue")

    result = generate_signals(diagnosis)

    assert result["signals"][0]["root_cause"] is None


def test_empty_diagnoses_generate_zero_signals():
    diagnosis = make_diagnosis("telemetry_data_quality_issue")
    diagnosis["diagnoses"] = []

    result = generate_signals(diagnosis)

    assert result["signal_count"] == 0
    assert result["signals"] == []


def test_missing_diagnosis_id_is_rejected():
    diagnosis = make_diagnosis("telemetry_data_quality_issue")
    diagnosis.pop("diagnosis_id")

    with pytest.raises(ValueError, match="diagnosis_id"):
        generate_signals(diagnosis)


def test_invalid_diagnosis_object_is_rejected():
    with pytest.raises(ValueError):
        generate_signals(None)


def test_unknown_issue_uses_other_category():
    diagnosis = make_diagnosis("unknown_test_issue")

    result = generate_signals(diagnosis)

    assert result["signals"][0]["signal_type"] == "other"


def test_multiple_diagnoses_generate_multiple_signals():
    diagnosis = make_diagnosis("telemetry_data_quality_issue")

    diagnosis["diagnoses"].append({
        "issue": "data_distribution_change",
        "status": "detected",
        "investigation_area": "feature_monitoring",
        "evidence": {"drifted_features": ["age"]},
    })

    result = generate_signals(diagnosis)

    assert result["signal_count"] == 2
    assert result["signals"][0]["signal_type"] == "data_quality"
    assert result["signals"][1]["signal_type"] == "data_drift"
