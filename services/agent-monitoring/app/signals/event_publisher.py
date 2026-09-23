import json
import urllib.request
from typing import Any


EVENT_API_URL = "http://127.0.0.1:8000/events"


def publish_event(event: dict[str, Any]) -> dict[str, Any]:
    """Publish an IntelliCore event to the Agent Monitoring API."""

    payload = json.dumps(event).encode("utf-8")

    request = urllib.request.Request(
        EVENT_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=5) as response:
        response_body = response.read().decode("utf-8")

    return json.loads(response_body)