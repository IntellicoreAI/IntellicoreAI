import pytest
from pydantic import ValidationError

from app.signals.schemas import (
    MLDiagnosisSignal,
    MLDiagnosisSignalDocument,
)


def make_signal():
    """Return a valid ML diagnosis signal for testing."""

    return {
        "schema_version": "1.0",
        "signal_id": "ml-signal-test-001",
        "created_at": "2026-10-31T10:00:00+00:00",
        "source": "ml_monitoring.diagnosis_engine",
        "signal_type": "data_quality",
        "status": "failed",
        "severity": "MEDIUM",
        "project_id": "intellicore-credit-risk",
        "environment": "development",
        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },
        "issue": "telemetry_data_quality_issue",
        "investigation_area": "prediction_telemetry",
        "evidence": {
            "failed_checks": ["freshness"],
            "max_age_seconds": 900,
        },
        "source_alert_id": "alert-test-001",
        "diagnosis_id": "diagnosis-test-001",
        "root_cause": None,
    }


def make_document(signals=None):
    """Return a valid signal document."""

    if signals is None:
        signals = [make_signal()]

    return {
        "schema_version": "1.0",
        "generated_at": "2026-10-31T10:00:00+00:00",
        "source_diagnosis_id": "diagnosis-test-001",
        "signal_count": len(signals),
        "signals": signals,
    }


def test_valid_signal_is_accepted():
    signal = MLDiagnosisSignal.model_validate(make_signal())

    assert signal.signal_type == "data_quality"
    assert signal.status == "failed"
    assert signal.severity == "MEDIUM"


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("signal_type", "invalid_type"),
        ("status", "unknown_status"),
        ("severity", "EXTREME"),
        ("environment", "invalid_environment"),
        ("schema_version", "2.0"),
    ],
)
def test_invalid_signal_values_are_rejected(field, invalid_value):
    signal = make_signal()
    signal[field] = invalid_value

    with pytest.raises(ValidationError):
        MLDiagnosisSignal.model_validate(signal)


def test_missing_required_field_is_rejected():
    signal = make_signal()
    signal.pop("signal_id")

    with pytest.raises(ValidationError):
        MLDiagnosisSignal.model_validate(signal)


def test_unexpected_signal_field_is_rejected():
    signal = make_signal()
    signal["unexpected_field"] = "unexpected"

    with pytest.raises(ValidationError):
        MLDiagnosisSignal.model_validate(signal)


def test_invalid_model_information_is_rejected():
    signal = make_signal()
    signal["model"]["model_name"] = ""

    with pytest.raises(ValidationError):
        MLDiagnosisSignal.model_validate(signal)


def test_evidence_must_be_a_dictionary():
    signal = make_signal()
    signal["evidence"] = ["invalid", "evidence"]

    with pytest.raises(ValidationError):
        MLDiagnosisSignal.model_validate(signal)


def test_nullable_root_cause_is_accepted():
    signal = make_signal()
    signal["root_cause"] = None

    validated = MLDiagnosisSignal.model_validate(signal)

    assert validated.root_cause is None


def test_valid_signal_document_is_accepted():
    document = MLDiagnosisSignalDocument.model_validate(
        make_document()
    )

    assert document.signal_count == 1
    assert len(document.signals) == 1


def test_mismatched_signal_count_is_rejected():
    document = make_document()
    document["signal_count"] = 5

    with pytest.raises(ValidationError):
        MLDiagnosisSignalDocument.model_validate(document)


def test_empty_signal_document_is_accepted():
    document = make_document(signals=[])

    validated = MLDiagnosisSignalDocument.model_validate(
        document
    )

    assert validated.signal_count == 0
    assert validated.signals == []


def test_unexpected_document_field_is_rejected():
    document = make_document()
    document["extra_information"] = "unexpected"

    with pytest.raises(ValidationError):
        MLDiagnosisSignalDocument.model_validate(document)


def test_invalid_signal_inside_document_is_rejected():
    signal = make_signal()
    signal["severity"] = "INVALID"

    document = make_document(signals=[signal])

    with pytest.raises(ValidationError):
        MLDiagnosisSignalDocument.model_validate(document)
