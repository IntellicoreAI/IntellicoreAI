from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
import os

import psycopg
from psycopg.types.json import Jsonb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def get_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "intellicore"),
        user=os.getenv("POSTGRES_USER", "intellicore"),
        password=os.getenv("POSTGRES_PASSWORD", "intellicore_dev"),
    )


class EventEnvelope(BaseModel):
    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    schema_version: str = "1.0"
    data: dict[str, Any] = Field(default_factory=dict)


def normalize_event(payload: EventEnvelope) -> dict[str, Any]:
    data = payload.data

    system_type = str(data.get("system_type", "")).upper()
    if system_type == "AGENTIC_AI":
        system_type = "AGENT"

    if system_type not in {"AGENT", "LLM", "ML"}:
        raise ValueError("Unsupported system_type")

    raw_event_id = payload.event_id
    # Alert IDs use a suffix; derive a stable UUID for the database.
    try:
        event_id = UUID(raw_event_id)
    except ValueError:
        event_id = UUID(bytes=__import__("hashlib").sha256(
            raw_event_id.encode("utf-8")
        ).digest()[:16])

    latency = data.get("execution_time_ms")
    if latency is not None:
        latency = float(latency)
        if latency < 0:
            raise ValueError("execution_time_ms cannot be negative")

    return {
        "event_id": event_id,
        "timestamp": payload.timestamp,
        "system_type": system_type,
        "project_id": str(data.get("project_id") or "intellicore"),
        "model_id": data.get("model_id"),
        "trace_id": str(data.get("trace_id") or "unknown"),
        "span_id": str(data.get("span_id") or "unknown"),
        "event_type": payload.event_type,
        "input": data.get("input"),
        "output": data.get("output"),
        "latency": latency,
        "status": str(data.get("execution_status") or "unknown"),
        "error": data.get("error"),
        "metadata": {
            **(data.get("metadata") or {}),
            "source": payload.source,
            "schema_version": payload.schema_version,
            "original_event_id": raw_event_id,
            "agent_name": data.get("agent_name"),
        },
    }


def insert_event(event: dict[str, Any]) -> None:
    query = """
        INSERT INTO events (
            event_id, timestamp, system_type, project_id, model_id,
            trace_id, span_id, event_type, input, output, latency,
            status, error, metadata
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (event_id) DO NOTHING
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (
                event["event_id"], event["timestamp"], event["system_type"],
                event["project_id"], event["model_id"], event["trace_id"],
                event["span_id"], event["event_type"],
                Jsonb(event["input"]) if event["input"] is not None else None,
                Jsonb(event["output"]) if event["output"] is not None else None,
                event["latency"], event["status"], event["error"],
                Jsonb(event["metadata"]),
            ))


def fetch_events(limit: int = 100) -> list[dict[str, Any]]:
    query = """
        SELECT event_id, timestamp, system_type, project_id, model_id,
               trace_id, span_id, event_type, input, output, latency,
               status, error, metadata
        FROM events
        ORDER BY timestamp DESC
        LIMIT %s
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

    columns = [
        "event_id", "timestamp", "system_type", "project_id", "model_id",
        "trace_id", "span_id", "event_type", "input", "output", "latency",
        "status", "error", "metadata",
    ]
    return [
        {
            **dict(zip(columns, row)),
            "event_id": str(row[0]),
            "timestamp": row[1].isoformat(),
        }
        for row in rows
    ]


app = FastAPI(title="IntelliCore Central Ingestion", version="1.0.0")


@app.get("/health")
def health():
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1")
        return {"status": "healthy", "service": "ingestion-service"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


@app.post("/events")
def receive_event(payload: EventEnvelope):
    try:
        event = normalize_event(payload)
        insert_event(event)
        return {"status": "accepted", "event_id": payload.event_id}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Event persistence failed") from exc


@app.get("/events")
def list_events(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    try:
        events = fetch_events(limit)
        return {"count": len(events), "events": events}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not retrieve events") from exc
