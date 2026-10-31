from typing import Any


def adapt_drift_report(
    report: dict[str, Any],
    model: dict[str, str],
    project_id: str,
    environment: str,
    event_id: str,
    occurred_at: str,
) -> dict[str, Any]:
    """Wrap an existing drift report in the IntelliCore base-event envelope."""

    if not isinstance(report, dict):
        raise ValueError("report must be an object")

    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id must be a non-empty string")

    if environment not in {"development", "staging", "production"}:
        raise ValueError(f"Unsupported environment: {environment}")

    for field in ("model_name", "model_version"):
        value = model.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"model.{field} must be a non-empty string")

    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")

    if not isinstance(occurred_at, str) or not occurred_at.strip():
        raise ValueError("occurred_at must be a non-empty string")

    if not isinstance(report.get("overall_summary"), dict):
        raise ValueError("report.overall_summary must be an object")

    return {
        "schema_version": "1.0",
        "event_id": event_id,
        "event_type": "monitoring_result",
        "occurred_at": occurred_at,
        "project_id": project_id,
        "environment": environment,
        "model": {
            "model_name": model["model_name"],
            "model_version": model["model_version"],
        },
        "data": {
            "signal_type": "data_drift",
            "report": report,
        },
    }
