
from app.data_quality.telemetry_structure import (
    validate_telemetry_structure,
)


def test_valid_event_passes():
    result = validate_telemetry_structure([
        {"input": {"features": {"age": 25}}}
    ])

    assert result["status"] == "passed"
    assert result["malformed_event_count"] == 0


def test_missing_input_is_detected():
    result = validate_telemetry_structure([{}])

    assert result["status"] == "failed"
    assert "input_missing" in result["issues"][0]


def test_non_object_input_is_detected():
    result = validate_telemetry_structure([
        {"input": "invalid"}
    ])

    assert result["status"] == "failed"
    assert "input_must_be_object" in result["issues"][0]


def test_missing_features_is_detected():
    result = validate_telemetry_structure([
        {"input": {}}
    ])

    assert result["status"] == "failed"
    assert "features_missing" in result["issues"][0]


def test_non_object_features_are_detected():
    result = validate_telemetry_structure([
        {"input": {"features": ["invalid"]}}
    ])

    assert result["status"] == "failed"
    assert "features_must_be_object" in result["issues"][0]


def test_multiple_malformed_events_are_counted():
    result = validate_telemetry_structure([
        {},
        {"input": "invalid"},
        {"input": {"features": {"age": 25}}},
    ])

    assert result["status"] == "failed"
    assert result["event_count"] == 3
    assert result["malformed_event_count"] == 2
