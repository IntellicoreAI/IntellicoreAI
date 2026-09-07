from fastapi import FastAPI

from .signals.event_schema import AgenticEvent

app = FastAPI(
    title="IntelliCore Agent Monitoring Service",
    version="0.1.0",
)


# Temporary in-memory event store.
# Database persistence will be added in a later step.
events: list[AgenticEvent] = []


@app.get("/health")
def health_check():
    """Check whether the Agent Monitoring service is running."""

    return {
        "status": "healthy",
        "service": "agent-monitoring",
    }


@app.post("/events")
def ingest_event(event: AgenticEvent):
    """Receive and validate an Agentic telemetry event."""

    events.append(event)

    return {
        "status": "accepted",
        "event_id": event.event_id,
    }


@app.get("/events")
def get_events():
    """Return all received Agentic telemetry events."""

    return {
        "count": len(events),
        "events": [event.model_dump(mode="json") for event in events],
    }