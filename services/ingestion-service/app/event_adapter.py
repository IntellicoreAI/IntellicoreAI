from typing import Any


SUPPORTED_ENVIRONMENTS = {
    "development",
    "staging",
    "production",
}


def _validate_identity(project_id: str, environment: str) -> None:
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id must be a non-empty string")

    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"Unsupported environment: {environment}")


def _validate_model(model: Any) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ValueError("model must be an object")

    for field in ("model_name", "model_version"):
        value = model.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"model.{field} must be a non-empty string")

    return {
        "model_name": model["model_name"],
        "model_version": model["model_version"],
    }


def adapt_prediction_event(
    source_event: dict[str, Any],
    project_id: str,
    environment: str,
) -> dict[str, Any]:
    _validate_identity(project_id, environment)

    for field in ("event_id", "timestamp", "model", "prediction"):
        if field not in source_event:
            raise ValueError(f"Prediction event is missing '{field}'")

    if not isinstance(source_event["event_id"], str):
        raise ValueError("event_id must be a string")

    if not isinstance(source_event["timestamp"], str):
        raise ValueError("timestamp must be a string")

    data = {
        key: source_event[key]
        for key in ("input", "prediction", "explanation")
        if key in source_event
    }

    return {
        "schema_version": "1.0",
        "event_id": source_event["event_id"],
        "event_type": "prediction",
        "occurred_at": source_event["timestamp"],
        "project_id": project_id,
        "environment": environment,
        "model": _validate_model(source_event["model"]),
        "data": data,
    }


def adapt_unified_alert(
    source_alert: dict[str, Any],
    project_id: str,
    environment: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_identity(project_id, environment)

    for field in ("alert_id", "generated_at", "model"):
        if field not in source_alert:
            raise ValueError(f"Unified alert is missing '{field}'")

    if not isinstance(source_alert["alert_id"], str):
        raise ValueError("alert_id must be a string")

    if not isinstance(source_alert["generated_at"], str):
        raise ValueError("generated_at must be a string")

    event_type = (
        "alert"
        if source_alert.get("overall_status") == "alert"
        else "monitoring_result"
    )

    data = {
        key: value
        for key, value in source_alert.items()
        if key not in {"alert_id", "generated_at", "model"}
    }

    if provenance is not None:
        data["provenance"] = provenance

    return {
        "schema_version": "1.0",
        "event_id": source_alert["alert_id"],
        "event_type": event_type,
        "occurred_at": source_alert["generated_at"],
        "project_id": project_id,
        "environment": environment,
        "model": _validate_model(source_alert["model"]),
        "data": data,
    }
