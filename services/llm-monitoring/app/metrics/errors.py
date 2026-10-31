def error_metrics(status: str, error: str | None) -> dict:
    failed = status.strip().lower() not in {"success", "ok", "completed", "200"}
    return {"is_error": failed or bool(error), "error_type": type_name(error) if error else ("request_failed" if failed else None),
            "error_message": error}
def type_name(error: str) -> str:
    first = error.strip().split(":", 1)[0]
    return first[:100] or "unknown_error"
