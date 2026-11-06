
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
    "http://127.0.0.1:8000/events",
)

TIMEOUT_SECONDS = 10
MAX_RETRIES = 3


def normalize_agentic_event(
    event: dict[str, Any],
    event_type: str = "monitoring_result",
) -> dict[str, Any]:
    """Convert a native Agentic AI event to the common event envelope."""

    if event_type not in ("monitoring_result", "alert"):
        raise ValueError(f"Unsupported event type: {event_type}")

    event_id = event.get("event_id")
    if not event_id:
        raise ValueError("Agentic event must contain an event_id")

    metadata = event.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    data = {
        "agent_name": (
            metadata.get("agent_name")
            or event.get("model_id")
            or "agent"
        ),
        "project_id": event.get("project_id"),
        "model_id": event.get("model_id"),
        "execution_status": event.get("status"),
        "execution_time_ms": event.get(
            "latency",
            event.get("latency_ms"),
        ),
        "input": event.get("input"),
        "output": event.get("output"),
        "error": event.get("error"),
        "trace_id": event.get("trace_id"),
        "span_id": event.get("span_id"),
        "system_type": "agentic_ai",
        "metadata": metadata,
    }

    return {
        "event_id": (
            str(event_id)
            if event_type == "monitoring_result"
            else f"{event_id}-alert"
        ),
        "event_type": event_type,
        "timestamp": event.get("timestamp"),
        "source": "agentic_ai",
        "schema_version": "1.0",
        "data": data,
    }


def publish_central_event(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward a normalized event to central ingestion with retries."""

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
                request,
                timeout=TIMEOUT_SECONDS,
            ) as response:

                response_body = response.read().decode("utf-8")

                logger.info(
                    "Central event forwarded successfully: %s",
                    payload.get("event_id"),
                )

                return {
                    "status_code": response.status,
                    "response": response_body,
                }

        except urllib.error.HTTPError as exc:
            # Retry rate limits, request timeouts and server errors.
            if exc.code not in (408, 429) and not (
                500 <= exc.code <= 599
            ):
                logger.error(
                    "Central ingestion rejected event: HTTP %s",
                    exc.code,
                )
                raise

            logger.warning(
                "Temporary central API error: HTTP %s",
                exc.code,
            )

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "Central ingestion request failed: %s",
                exc,
            )

        if attempt < MAX_RETRIES - 1:
            delay = 2 ** attempt
            logger.info(
                "Retrying central ingestion in %s seconds",
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        "Central ingestion failed after all retry attempts"
    )
