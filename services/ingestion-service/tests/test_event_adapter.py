import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "services" / "ingestion-service"

sys.path.insert(0, str(SERVICE_ROOT))

from app.event_adapter import (
    adapt_prediction_event,
    adapt_unified_alert,
)


SCHEMA_PATH = ROOT / "schemas" / "base-event.schema.json"


def load_schema():
    with SCHEMA_PATH.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def assert_valid_event(event):
    validator = Draft202012Validator(
        load_schema(),
        format_checker=FormatChecker(),
    )
    errors = list(validator.iter_errors(event))
    assert errors == [], [error.message for error in errors]


def test_actual_prediction_telemetry_adapts():
    telemetry_path = ROOT / "data" / "telemetry" / "prediction_events.jsonl"

    with telemetry_path.open("r", encoding="utf-8") as file:
        source_event = json.loads(next(file))

    adapted = adapt_prediction_event(
        source_event,
        project_id="proj_test",
        environment="development",
    )

    assert adapted["event_id"] == source_event["event_id"]
    assert adapted["occurred_at"] == source_event["timestamp"]
    assert adapted["data"]["prediction"] == source_event["prediction"]
    assert adapted["data"]["explanation"] == source_event["explanation"]

    assert_valid_event(adapted)


def test_actual_unified_alert_adapts_with_provenance():
    alert_path = ROOT / "data" / "artifacts" / "monitoring" / "unified_alert.json"
    performance_path = (
        ROOT
        / "data"
        / "artifacts"
        / "monitoring"
        / "performance_degradation_report.json"
    )

    with alert_path.open("r", encoding="utf-8") as file:
        source_alert = json.load(file)

    with performance_path.open("r", encoding="utf-8") as file:
        performance_report = json.load(file)

    provenance = {
        "performance_evaluation_context": performance_report.get(
            "evaluation_context",
            "unspecified",
        )
    }

    adapted = adapt_unified_alert(
        source_alert,
        project_id="proj_test",
        environment="development",
        provenance=provenance,
    )

    assert adapted["event_id"] == source_alert["alert_id"]
    assert adapted["occurred_at"] == source_alert["generated_at"]
    assert adapted["data"]["alerts"] == source_alert["alerts"]
    assert (
        adapted["data"]["provenance"]["performance_evaluation_context"]
        == provenance["performance_evaluation_context"]
    )

    assert_valid_event(adapted)


def test_prediction_adapter_rejects_empty_project_id():
    try:
        adapt_prediction_event(
            {
                "event_id": "evt_1",
                "timestamp": "2026-10-26T10:30:00Z",
                "model": {
                    "model_name": "demo",
                    "model_version": "1.0",
                },
                "prediction": {"predicted_class": 0},
            },
            project_id="",
            environment="development",
        )
    except ValueError:
        return

    raise AssertionError("Expected an empty project_id to be rejected")


def test_alert_adapter_rejects_unknown_environment():
    try:
        adapt_unified_alert(
            {
                "alert_id": "alert_1",
                "generated_at": "2026-10-26T10:30:00Z",
                "model": {
                    "model_name": "demo",
                    "model_version": "1.0",
                },
                "overall_status": "alert",
            },
            project_id="proj_test",
            environment="unknown",
        )
    except ValueError:
        return

    raise AssertionError("Expected an unknown environment to be rejected")
