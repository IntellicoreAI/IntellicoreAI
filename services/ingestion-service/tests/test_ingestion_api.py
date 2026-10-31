
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "services" / "ingestion-service"
sys.path.insert(0, str(SERVICE_ROOT))

import app.main as ingestion_main
from app.main import app
from app.monitoring_adapter import adapt_drift_report


client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_supabase(monkeypatch):
    """Prevent tests from writing to the real Supabase database."""
    mock_client = MagicMock()

    monkeypatch.setattr(
        ingestion_main,
        "get_supabase_client",
        lambda: mock_client,
    )

    return mock_client


def valid_event():
    """Return a valid IntelliCore base event."""
    return {
        "schema_version": "1.0",
        "event_id": "evt_test_001",
        "event_type": "prediction",
        "occurred_at": "2026-10-26T10:30:00Z",
        "project_id": "proj_test",
        "environment": "development",
        "model": {
            "model_name": "demo_model",
            "model_version": "1.0",
        },
        "data": {
            "prediction": {
                "predicted_class": 1,
                "probability": 0.91,
            }
        },
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_event_is_persisted(mock_supabase):
    event = valid_event()

    response = client.post("/events", json=event)

    assert response.status_code == 202
    assert response.json() == {
        "status": "accepted",
        "event_id": "evt_test_001",
        "event_type": "prediction",
        "project_id": "proj_test",
        "persisted": True,
    }

    mock_supabase.table.assert_called_once_with("ml_events")

    inserted_event = (
        mock_supabase.table.return_value
        .insert.call_args.args[0]
    )

    assert inserted_event == {
        "event_id": "evt_test_001",
        "project_id": "proj_test",
        "environment": "development",
        "event_type": "prediction",
        "occurred_at": "2026-10-26T10:30:00Z",
        "model_name": "demo_model",
        "model_version": "1.0",
        "event_data": {
            "schema_version": "1.0",
            "prediction": {
                "predicted_class": 1,
                "probability": 0.91,
            },
        },
    }

    mock_supabase.table.return_value.insert.return_value.execute.assert_called_once()


def test_missing_required_field_is_rejected(mock_supabase):
    event = valid_event()
    del event["model"]

    response = client.post("/events", json=event)

    assert response.status_code == 422
    mock_supabase.table.assert_not_called()


def test_invalid_timestamp_is_rejected(mock_supabase):
    event = valid_event()
    event["occurred_at"] = "not-a-timestamp"

    response = client.post("/events", json=event)

    assert response.status_code == 422
    mock_supabase.table.assert_not_called()


def test_unknown_top_level_field_is_rejected(mock_supabase):
    event = valid_event()
    event["unexpected_field"] = "not allowed"

    response = client.post("/events", json=event)

    assert response.status_code == 422
    mock_supabase.table.assert_not_called()


def test_invalid_environment_is_rejected(mock_supabase):
    event = valid_event()
    event["environment"] = "unknown"

    response = client.post("/events", json=event)

    assert response.status_code == 422
    mock_supabase.table.assert_not_called()


def test_non_object_json_is_rejected(mock_supabase):
    response = client.post(
        "/events",
        json=["not", "an", "object"],
    )

    assert response.status_code == 422
    mock_supabase.table.assert_not_called()


def test_malformed_json_is_rejected(mock_supabase):
    response = client.post(
        "/events",
        content=b'{"event_id":',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    mock_supabase.table.assert_not_called()


def test_real_drift_report_is_persisted(mock_supabase):
    report_path = (
        ROOT
        / "data"
        / "artifacts"
        / "monitoring"
        / "drift_report.json"
    )
    config_path = (
        ROOT
        / "data"
        / "artifacts"
        / "monitoring"
        / "monitoring_config.json"
    )

    report = json.loads(
        report_path.read_text(encoding="utf-8-sig")
    )
    config = json.loads(
        config_path.read_text(encoding="utf-8-sig")
    )

    event = adapt_drift_report(
        report=report,
        model=config["model"],
        project_id="proj_test",
        environment="development",
        event_id="drift-integration-test-001",
        occurred_at=datetime.now(timezone.utc).isoformat(),
    )

    response = client.post("/events", json=event)

    assert response.status_code == 202
    assert response.json()["event_id"] == "drift-integration-test-001"
    assert response.json()["event_type"] == "monitoring_result"
    assert response.json()["persisted"] is True

    mock_supabase.table.assert_called_once_with("ml_events")

    inserted_event = (
        mock_supabase.table.return_value
        .insert.call_args.args[0]
    )

    assert inserted_event["event_type"] == "monitoring_result"
    assert inserted_event["project_id"] == "proj_test"
    assert inserted_event["event_data"]["signal_type"] == "data_drift"


def test_database_api_error_returns_500(monkeypatch):
    from postgrest.exceptions import APIError

    mock_client = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.side_effect = (
        APIError(
            {
                "message": "Database unavailable",
                "code": "TEST_ERROR",
                "hint": None,
                "details": None,
            }
        )
    )

    monkeypatch.setattr(
        ingestion_main,
        "get_supabase_client",
        lambda: mock_client,
    )

    response = client.post("/events", json=valid_event())

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to persist event to the database"
    )
    assert "Database unavailable" not in response.text
