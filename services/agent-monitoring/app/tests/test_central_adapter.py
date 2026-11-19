
"""Tests for the Agentic AI central event adapter."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.signals.central_adapter import (
    normalize_agentic_event,
    normalize_timestamp,
)


VALID_EVENT_ID = "123e4567-e89b-12d3-a456-426614174000"


def build_event(**overrides):
    event = {
        "event_id": VALID_EVENT_ID,
        "timestamp": "2026-11-19T10:30:00+00:00",
        "system_type": "AGENT",
        "project_id": "test-project",
        "model_id": "test-agent",
        "trace_id": "trace-123",
        "span_id": "span-123",
        "event_type": "agent_execution",
        "input": {"prompt": "hello"},
        "output": {"result": "world"},
        "latency": 125.5,
        "status": "success",
        "error": None,
        "metadata": {"agent_name": "assistant"},
    }

    event.update(overrides)
    return event


def test_monitoring_result_uses_common_event_envelope():
    payload = normalize_agentic_event(build_event())

    assert set(payload) == {
        "event_id",
        "event_type",
        "timestamp",
        "source",
        "schema_version",
        "data",
    }

    assert payload["event_id"] == VALID_EVENT_ID
    assert payload["event_type"] == "monitoring_result"
    assert payload["source"] == "agentic"
    assert payload["schema_version"] == "1.0"
    assert payload["data"]["system_type"] == "AGENT"
    assert payload["data"]["agent_name"] == "assistant"


def test_alert_uses_a_valid_deterministic_uuid():
    first = normalize_agentic_event(build_event(), "alert")
    second = normalize_agentic_event(build_event(), "alert")

    assert first["event_id"] == second["event_id"]
    assert first["event_id"] != VALID_EVENT_ID
    assert str(UUID(first["event_id"])) == first["event_id"]
    assert first["event_type"] == "alert"


def test_monitoring_result_preserves_original_uuid():
    payload = normalize_agentic_event(build_event())

    assert payload["event_id"] == str(UUID(VALID_EVENT_ID))


def test_invalid_event_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported event type"):
        normalize_agentic_event(
            build_event(),
            event_type="unknown",
        )


def test_missing_event_id_is_rejected():
    with pytest.raises(ValueError, match="Event ID is required"):
        normalize_agentic_event(build_event(event_id=None))


def test_invalid_event_id_is_rejected():
    with pytest.raises(ValueError, match="valid UUID"):
        normalize_agentic_event(build_event(event_id="not-a-uuid"))


def test_timestamp_is_normalized_to_utc():
    result = normalize_timestamp("2026-11-19T16:00:00+05:30")

    assert result == "2026-11-19T10:30:00Z"


def test_naive_timestamp_is_treated_as_utc():
    result = normalize_timestamp("2026-11-19T10:30:00")

    assert result == "2026-11-19T10:30:00Z"


def test_datetime_timestamp_is_normalized_to_utc():
    value = datetime(
        2026, 11, 19, 16, 0,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )

    assert normalize_timestamp(value) == "2026-11-19T10:30:00Z"


def test_missing_timestamp_is_rejected():
    with pytest.raises(ValueError, match="Timestamp is required"):
        normalize_agentic_event(build_event(timestamp=None))


def test_invalid_timestamp_is_rejected():
    with pytest.raises(ValueError, match="Invalid timestamp"):
        normalize_agentic_event(build_event(timestamp="not-a-date"))


def test_non_dictionary_event_is_rejected():
    with pytest.raises(ValueError, match="Event must be a dictionary"):
        normalize_agentic_event(None)
