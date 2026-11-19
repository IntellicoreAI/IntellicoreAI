
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .execution_monitor.graph import agent_graph
from .repositories.event_repository import get_all_events, insert_event
from .signals.event_schema import AgenticEvent

app = FastAPI(
    title="IntelliCore Agent Monitoring Service",
    version="0.1.0",
)


class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)


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


@app.post("/run")
def run_agent(request: AgentRunRequest):
    """Run the Agentic workflow and return its monitoring results."""

    task = request.task.strip()
    if not task:
        raise HTTPException(
            status_code=422,
            detail="Task cannot be empty.",
        )

    trace_id = str(uuid4())

    result = agent_graph.invoke({
        "task": task,
        "trace_id": trace_id,
        "attempt_count": 0,
        "max_attempts": 3,
        "trajectory": [],
        "tool_calls": [],
        "tool_results": [],
        "telemetry_events": [],
        "memory": [],
    })

    return {
        "status": result.get("status", "unknown"),
        "trace_id": result.get("trace_id", trace_id),
        "output": result.get("output"),
        "diagnosis": result.get("diagnosis", {}),
        "recommendation": result.get("recommendation", {}),
        "analytics": result.get("analytics", {}),
        "event": result.get("event"),
    }
