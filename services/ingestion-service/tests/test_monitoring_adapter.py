import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "services" / "ingestion-service"
sys.path.insert(0, str(SERVICE_ROOT))

from app.monitoring_adapter import adapt_drift_report


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def test_real_drift_report_adapts_to_base_event_schema():
    report = load_json(
        ROOT / "data" / "artifacts" / "monitoring" / "drift_report.json"
    )
    config = load_json(
        ROOT / "data" / "artifacts" / "monitoring" / "monitoring_config.json"
    )
    schema = load_json(ROOT / "schemas" / "base-event.schema.json")

    event = adapt_drift_report(
        report=report,
        model=config["model"],
        project_id="proj_test",
        environment="development",
        event_id="drift-test-001",
        occurred_at=datetime.now(timezone.utc).isoformat(),
    )

    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )
    errors = list(validator.iter_errors(event))

    assert errors == [], [error.message for error in errors]
    assert event["event_type"] == "monitoring_result"
    assert event["data"]["signal_type"] == "data_drift"
    assert (
        event["data"]["report"]["overall_summary"]["overall_status"]
        == "no_significant_drift"
    )
    assert event["data"]["report"]["overall_summary"]["drifted_features"] == 0


def test_adapter_rejects_empty_project_id():
    report = {"overall_summary": {}}

    try:
        adapt_drift_report(
            report=report,
            model={"model_name": "demo", "model_version": "1.0"},
            project_id="",
            environment="development",
            event_id="drift-test-002",
            occurred_at="2026-10-27T12:00:00Z",
        )
    except ValueError:
        return

    raise AssertionError("Expected an empty project_id to be rejected")


def test_adapter_rejects_unknown_environment():
    try:
        adapt_drift_report(
            report={"overall_summary": {}},
            model={"model_name": "demo", "model_version": "1.0"},
            project_id="proj_test",
            environment="unknown",
            event_id="drift-test-003",
            occurred_at="2026-10-27T12:00:00Z",
        )
    except ValueError:
        return

    raise AssertionError("Expected an unknown environment to be rejected")
