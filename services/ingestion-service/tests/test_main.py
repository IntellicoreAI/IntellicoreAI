
import psycopg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import main


client = TestClient(app)

VALID_EVENT = {
    "event_id": "123e4567-e89b-12d3-a456-426614174000",
    "event_type": "monitoring_result",
    "timestamp": "2026-11-19T10:30:00Z",
    "source": "agentic",
    "schema_version": "1.0",
    "data": {
        "project_id": "project-01",
        "model_id": "agent-01",
        "trace_id": "trace-123",
        "span_id": "span-456",
        "execution_status": "success",
        "execution_time_ms": 120,
        "input": {"prompt": "Analyse this task"},
        "output": {"result": "Completed"},
        "error": None,
    },
}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "central-ingestion",
    }


def test_accepts_valid_agentic_event(monkeypatch):
    monkeypatch.setattr(
        main,
        "store_event",
        lambda event: True,
    )

    response = client.post("/events", json=VALID_EVENT)

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert response.json()["duplicate"] is False
    assert response.json()["source"] == "agentic"
    assert response.json()["system_type"] == "AGENT"


@pytest.mark.parametrize(
    ("source", "expected_system_type"),
    [
        ("agentic", "AGENT"),
        ("llm", "LLM"),
        ("ml", "ML"),
    ],
)
def test_source_to_system_type_mapping(
    monkeypatch,
    source,
    expected_system_type,
):
    monkeypatch.setattr(
        main,
        "store_event",
        lambda event: True,
    )

    event = VALID_EVENT.copy()
    event["source"] = source

    response = client.post("/events", json=event)

    assert response.status_code == 202
    assert response.json()["system_type"] == expected_system_type


def test_duplicate_event_is_handled(monkeypatch):
    monkeypatch.setattr(
        main,
        "store_event",
        lambda event: False,
    )

    response = client.post("/events", json=VALID_EVENT)

    assert response.status_code == 202
    assert response.json()["duplicate"] is True
    assert response.json()["status"] == "accepted"


def test_rejects_invalid_event_id(monkeypatch):
    event = VALID_EVENT.copy()
    event["event_id"] = "not-a-valid-uuid"

    response = client.post("/events", json=event)

    assert response.status_code == 422


def test_rejects_invalid_event_type(monkeypatch):
    event = VALID_EVENT.copy()
    event["event_type"] = "unknown_event"

    response = client.post("/events", json=event)

    assert response.status_code == 422


def test_rejects_invalid_source(monkeypatch):
    event = VALID_EVENT.copy()
    event["source"] = "python"

    response = client.post("/events", json=event)

    assert response.status_code == 422


def test_rejects_missing_required_field():
    event = VALID_EVENT.copy()
    del event["schema_version"]

    response = client.post("/events", json=event)

    assert response.status_code == 422


def test_rejects_extra_envelope_field():
    event = VALID_EVENT.copy()
    event["unexpected_field"] = "not allowed"

    response = client.post("/events", json=event)

    assert response.status_code == 422


def test_rejects_invalid_schema_version():
    event = VALID_EVENT.copy()
    event["schema_version"] = "2.0"

    response = client.post("/events", json=event)

    assert response.status_code == 422


def test_database_failure_returns_503(monkeypatch):
    def raise_database_error(event):
        raise psycopg.OperationalError("Database unavailable")

    monkeypatch.setattr(
        main,
        "store_event",
        raise_database_error,
    )

    response = client.post("/events", json=VALID_EVENT)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Could not store event in the database"
    )
