
"""ML Monitoring Adapter.

Normalizes native ML monitoring reports and alerts
into the IntelliCore common event contract.
"""

import uuid
from datetime import datetime, timezone
from typing import Any


SUPPORTED_EVENT_TYPES = {"monitoring_result", "alert"}

SUPPORTED_ENVIRONMENTS = {
    "development",
    "staging",
    "production",
}


def create_ml_event(
    *,
    event_type: str,
    signal_type: str,
    report: dict[str, Any],
    model: dict[str, str],
    project_id: str,
    environment: str,
) -> dict[str, Any]:
    """Convert an ML monitoring report into a common event."""

    if event_type not in SUPPORTED_EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type}")

    if not isinstance(signal_type, str) or not signal_type.strip():
        raise ValueError("signal_type must be a non-empty string.")

    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary.")

    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id must be a non-empty string.")

    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            "environment must be development, staging, or production."
        )

    if not isinstance(model, dict):
        raise ValueError("model must be a dictionary.")

    model_name = model.get("model_name")
    model_version = model.get("model_version")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string.")

    event_data = {"signal_type": signal_type}

    if event_type == "alert":
        event_data["alert"] = report
    else:
        event_data["report"] = report

    return {
        "schema_version": "1.0",
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "environment": environment,
        "model": {
            "model_name": model_name,
            "model_version": model_version,
        },
        "data": event_data,
    }
