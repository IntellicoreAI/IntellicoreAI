
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

CENTRAL_INGESTION_URL = os.getenv(
    "CENTRAL_INGESTION_URL",
    "http://127.0.0.1:8000/events"
)

TIMEOUT_SECONDS = 10
MAX_RETRIES = 3


def normalize_agentic_event(
    event: dict[str, Any],
    event_type: str = "monitoring_result"
) -> dict[str, Any]:
    """Convert a native Agentic AI event to the common event envelope."""

    if event_type not in ("monitoring_result", "alert"):
        raise ValueError(f"Unsupported event type: {event_type}")

    data = {
        "agent_name": event.get("metadata", {}).get(
            "agent_name", "agent"
        ),
        "execution_status": event.get("status"),
        "execution_time_ms": event.get("latency_ms"),
        "input": event.get("input"),
        "output": event.get("output"),
        "error": event.get("error"),
        "trace_id": event.get("trace_id"),
        "span_id": event.get("span_id"),
        "system_type": "agentic_ai",
        "metadata": event.get("metadata", {}),
    }

    return {
        "event_id": (
            event.get("event_id")
            if event_type == "monitoring_result"
            else f"{event.get('event_id')}-alert"
        ),
        "event_type": event_type,
        "timestamp": event.get("timestamp"),
        "source": "agentic_ai",
        "schema_version": "1.0",
        "data": data,
    }


def publish_central_event(
    payload: dict[str, Any]
) -> dict[str, Any]:
    """Forward a normalized event to central ingestion."""

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        CENTRAL_INGESTION_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(
                request, timeout=TIMEOUT_SECONDS
            ) as response:
                response_body = response.read().decode("utf-8")
                return {
                    "status_code": response.status,
                    "response": response_body,
                }

        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                logger.error(
                    "Central ingestion rejected event: HTTP %s",
                    exc.code,
                )
                raise

            logger.warning(
                "Central ingestion server error: HTTP %s",
                exc.code,
            )

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Central ingestion request failed: %s", exc)

        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        "Central ingestion failed after all retry attempts"
    )
