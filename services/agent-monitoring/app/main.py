from fastapi import FastAPI

from .repositories.event_repository import get_all_events, insert_event
from .signals.event_schema import AgenticEvent

app = FastAPI(
    title="IntelliCore Agent Monitoring Service",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """Check whether the Agent Monitoring service is running."""

    return {
        "status": "healthy",
        "service": "agent-monitoring",
    }


@app.post("/events")
def ingest_event(event: AgenticEvent):
    """Receive, validate, and persist an Agentic telemetry event."""

    insert_event(event.model_dump(mode="json"))

    return {
        "status": "accepted",
        "event_id": event.event_id,
    }


@app.get("/events")
def get_events():
    """Return all persisted Agentic telemetry events."""

    events = get_all_events()

    return {
        "count": len(events),
        "events": events,
    }