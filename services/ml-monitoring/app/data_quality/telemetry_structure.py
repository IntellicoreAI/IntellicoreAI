
"""Validate the structure of ML prediction telemetry events."""

from typing import Any


def validate_telemetry_structure(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect malformed event, input, and feature structures."""

    malformed_events = []
    issues = []

    for index, event in enumerate(events):
        event_issues = []

        if not isinstance(event, dict):
            event_issues.append("event_must_be_object")
        elif "input" not in event:
            event_issues.append("input_missing")
        elif not isinstance(event["input"], dict):
            event_issues.append("input_must_be_object")
        elif "features" not in event["input"]:
            event_issues.append("features_missing")
        elif not isinstance(event["input"]["features"], dict):
            event_issues.append("features_must_be_object")

        if event_issues:
            malformed_events.append({
                "event_index": index,
                "issues": event_issues,
            })
            issues.extend(
                f"event_{index}:{issue}"
                for issue in event_issues
            )

    return {
        "check": "telemetry_structure",
        "status": "failed" if malformed_events else "passed",
        "event_count": len(events),
        "malformed_event_count": len(malformed_events),
        "malformed_events": malformed_events,
        "issues": issues,
    }
