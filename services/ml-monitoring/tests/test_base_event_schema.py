
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "schemas"
    / "base-event.schema.json"
)


@pytest.fixture
def event_schema():
    with SCHEMA_PATH.open("r", encoding="utf-8-sig") as file:
        schema = json.load(file)

    Draft202012Validator.check_schema(schema)
    return schema


@pytest.fixture
def valid_event():
    return {
        "schema_version": "1.0",
        "event_id": "evt_001",
        "event_type": "prediction",
        "occurred_at": "2026-10-26T10:30:00Z",
        "project_id": "proj_123",
        "environment": "development",
        "model": {
            "model_name": "credit_risk_model",
            "model_version": "1.0",
        },
        "data": {
            "prediction": 0,
            "confidence": 0.91,
        },
    }


def test_schema_is_valid(event_schema):
    Draft202012Validator.check_schema(event_schema)


def test_valid_event_passes(event_schema, valid_event):
    validator = Draft202012Validator(
        event_schema,
        format_checker=FormatChecker(),
    )
    assert list(validator.iter_errors(valid_event)) == []


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "event_id",
        "event_type",
        "occurred_at",
        "project_id",
        "environment",
        "model",
        "data",
    ],
)
def test_required_fields_cannot_be_missing(
    event_schema, valid_event, field
):
    event = valid_event.copy()
    del event[field]

    validator = Draft202012Validator(
        event_schema,
        format_checker=FormatChecker(),
    )
    assert list(validator.iter_errors(event))


@pytest.mark.parametrize(
    "event_type",
    ["prediction", "monitoring_result", "alert"],
)
def test_supported_event_types_pass(event_schema, valid_event, event_type):
    event = valid_event.copy()
    event["event_type"] = event_type

    validator = Draft202012Validator(
        event_schema,
        format_checker=FormatChecker(),
    )
    assert list(validator.iter_errors(event)) == []


def test_invalid_environment_fails(event_schema, valid_event):
    event = valid_event.copy()
    event["environment"] = "unknown"

    validator = Draft202012Validator(event_schema)
    assert list(validator.iter_errors(event))


def test_empty_project_id_fails(event_schema, valid_event):
    event = valid_event.copy()
    event["project_id"] = ""

    validator = Draft202012Validator(event_schema)
    assert list(validator.iter_errors(event))


def test_invalid_timestamp_fails(event_schema, valid_event):
    event = valid_event.copy()
    event["occurred_at"] = "not-a-timestamp"

    validator = Draft202012Validator(
        event_schema,
        format_checker=FormatChecker(),
    )
    assert list(validator.iter_errors(event))


def test_unknown_top_level_field_fails(event_schema, valid_event):
    event = valid_event.copy()
    event["unexpected_field"] = "not allowed"

    validator = Draft202012Validator(event_schema)
    assert list(validator.iter_errors(event))


def test_missing_model_version_fails(event_schema, valid_event):
    event = valid_event.copy()
    del event["model"]["model_version"]

    validator = Draft202012Validator(event_schema)
    assert list(validator.iter_errors(event))
