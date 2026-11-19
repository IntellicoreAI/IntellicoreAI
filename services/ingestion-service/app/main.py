
import logging
import os
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import psycopg
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IntelliCore Central Ingestion Service",
    version="1.0.0",
    description="Validates and stores normalized ML, LLM, and Agentic AI events.",
)

SOURCE_TO_SYSTEM_TYPE = {
    "agentic": "AGENT",
    "llm": "LLM",
    "ml": "ML",
}


class CentralEvent(BaseModel):
    """The common six-field IntelliCore event envelope."""

    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    event_type: Literal["monitoring_result", "alert"]
    timestamp: datetime
    source: Literal["ml", "llm", "agentic"]
    schema_version: Literal["1.0"]
    data: dict[str, Any]


def get_database_connection():
    """Create a connection using the repository's PostgreSQL settings."""

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB", "intellicore"),
        user=os.getenv("POSTGRES_USER", "intellicore"),
        password=os.getenv("POSTGRES_PASSWORD", "intellicore_dev"),
        connect_timeout=5,
    )


def store_event(event: CentralEvent) -> bool:
    """
    Store a normalized event in the existing events table.

    Returns True for a new event and False if its event_id already exists.
    """

    data = event.data
    system_type = SOURCE_TO_SYSTEM_TYPE[event.source]

    # These fields are required by the existing database table.
    # Use the event ID as a safe fallback when trace identifiers are absent.
    project_id = str(data.get("project_id") or "unknown")
    trace_id = str(data.get("trace_id") or event.event_id)
    span_id = str(data.get("span_id") or event.event_id)

    event_status = str(
        data.get("execution_status")
        or data.get("status")
        or "unknown"
    )

    latency = data.get(
        "execution_time_ms",
        data.get("latency"),
    )

    if latency is not None:
        latency = float(latency)
        if latency < 0:
            raise ValueError("Event latency cannot be negative")

    query = """
        INSERT INTO events (
            event_id,
            timestamp,
            system_type,
            project_id,
            model_id,
            trace_id,
            span_id,
            event_type,
            input,
            output,
            latency,
            status,
            error,
            metadata
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (event_id) DO NOTHING
        RETURNING event_id
    """

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    event.event_id,
                    event.timestamp,
                    system_type,
                    project_id,
                    data.get("model_id"),
                    trace_id,
                    span_id,
                    event.event_type,
                    Jsonb(data.get("input")),
                    Jsonb(data.get("output")),
                    latency,
                    event_status,
                    data.get("error"),
                    Jsonb(data),
                ),
            )
            inserted = cursor.fetchone() is not None

    return inserted


@app.get("/health")
def health_check():
    """Check whether the ingestion API is running."""

    return {
        "status": "healthy",
        "service": "central-ingestion",
    }


@app.post("/events", status_code=status.HTTP_202_ACCEPTED)
def ingest_event(event: CentralEvent):
    """Validate and persist one normalized event."""

    try:
        inserted = store_event(event)

        return {
            "status": "accepted",
            "duplicate": not inserted,
            "event_id": str(event.event_id),
            "source": event.source,
            "system_type": SOURCE_TO_SYSTEM_TYPE[event.source],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except psycopg.Error as exc:
        logger.exception("Database error while storing central event")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not store event in the database",
        ) from exc
