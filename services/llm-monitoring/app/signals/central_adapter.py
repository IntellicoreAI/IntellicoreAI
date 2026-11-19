"""Central adapter: converts native LLM monitoring events to the common event envelope.

Every adapter emits exactly six top-level fields:
  timestamp, source, schema_version, event_id, event_type, data

LLM events use source="llm". The agentic adapter mirrors this module's structure.
Forwarding is best-effort and runs in a background thread so a slow or unavailable
central ingestion service never delays or fails an LLM request.
"""
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_URL

logger = logging.getLogger(__name__)

SOURCE = "llm"
SCHEMA_VERSION = "1.0"
EVENT_TYPES = ("monitoring_result", "alert")
TIMEOUT_SECONDS = 10
MAX_RETRIES = 3


def central_url() -> str:
    return os.getenv("CENTRAL_INGESTION_URL", "http://127.0.0.1:8000/events")


def forwarding_enabled() -> bool:
    return os.getenv("CENTRAL_INGESTION_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _iso_utc(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            value = None
    if not isinstance(value, datetime):
        value = datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def alert_event_id(event_id: str) -> str:
    """Deterministic UUID for the alert derived from a monitoring event (idempotent on retry)."""
    return str(uuid5(NAMESPACE_URL, f"intellicore-llm-alert:{event_id}"))


def normalize_llm_event(event: dict[str, Any], event_type: str = "monitoring_result",
                        alert: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a native LLM event (as stored in the events table) to the common envelope."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type}")
    event_id = event.get("event_id")
    if not event_id:
        raise ValueError("LLM event must contain an event_id")
    event_id = str(UUID(str(event_id)))

    md = event.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}
    evaluations = md.get("evaluations") or {}
    metrics = md.get("metrics") or {}
    scores = {name: (ev or {}).get("score") for name, ev in evaluations.items() if isinstance(ev, dict)}
    labels = {name: (ev or {}).get("label") for name, ev in evaluations.items() if isinstance(ev, dict)}
    numeric = [s for s in scores.values() if isinstance(s, (int, float))]

    data: dict[str, Any] = {
        "system_type": "llm",
        "project_id": event.get("project_id"),
        "model_id": event.get("model_id"),
        "trace_id": event.get("trace_id"),
        "span_id": event.get("span_id"),
        "parent_span_id": md.get("parent_span_id"),
        "trace_source": md.get("trace_source"),
        "llm_event_type": event.get("event_type"),
        "execution_status": event.get("status"),
        "execution_time_ms": event.get("latency"),
        "error": event.get("error"),
        "input": event.get("input"),
        "output": event.get("output"),
        "metrics": {
            "input_tokens": metrics.get("input_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "total_tokens": metrics.get("total_tokens"),
            "cost_usd": metrics.get("cost_usd"),
            "latency_band": metrics.get("latency_band"),
        },
        # LLM evaluation results, carried as evidence for the Decision Pipeline.
        "evaluation": {
            "scores": scores,
            "labels": labels,
            "overall_heuristic_score": round(sum(numeric) / len(numeric), 4) if numeric else None,
            "risk_flags": md.get("risk_flags", []),
            "human_review_recommended": md.get("human_review_recommended"),
            "fintech_compliance": md.get("fintech_compliance"),
        },
        "diagnosis": md.get("diagnosis"),
        "recommendation": md.get("recommendation"),
        "metadata": {k: md.get(k) for k in ("provider", "environment", "user_id", "session_id") if md.get(k) is not None},
    }
    if event_type == "alert":
        data["alert"] = alert or {}
    return {
        "timestamp": _iso_utc(event.get("timestamp")),
        "source": SOURCE,
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id if event_type == "monitoring_result" else alert_event_id(event_id),
        "event_type": event_type,
        "data": data,
    }


def publish_central_event(payload: dict[str, Any]) -> dict[str, Any]:
    """POST a normalized event to central ingestion, retrying transient failures."""
    request = urllib.request.Request(central_url(), data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8")
                logger.info("Central event forwarded: %s", payload.get("event_id"))
                return {"status_code": response.status, "response": body}
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429) and not (500 <= exc.code <= 599):
                logger.error("Central ingestion rejected event: HTTP %s", exc.code)
                raise
            logger.warning("Temporary central API error: HTTP %s", exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Central ingestion request failed: %s", exc)
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError("Central ingestion failed after all retry attempts")


def _send_safely(payload: dict[str, Any]) -> None:
    try:
        publish_central_event(payload)
    except Exception:
        logger.exception("Failed to forward LLM event to central ingestion: %s", payload.get("event_id"))


def forward_llm_event(event: dict[str, Any], event_type: str = "monitoring_result",
                      alert: dict[str, Any] | None = None, wait: bool = False) -> dict[str, Any] | None:
    """Normalize and forward in the background. Never raises into the request path."""
    if not forwarding_enabled():
        return None
    try:
        payload = normalize_llm_event(event, event_type, alert)
    except Exception:
        logger.exception("Could not normalize LLM event for central ingestion")
        return None
    thread = threading.Thread(target=_send_safely, args=(payload,), daemon=True, name="llm-central-forward")
    thread.start()
    if wait:
        thread.join()
    return payload
