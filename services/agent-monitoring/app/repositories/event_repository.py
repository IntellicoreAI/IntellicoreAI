from typing import Any

from psycopg.types.json import Jsonb

from ..database.connection import get_database_connection


def insert_event(event: dict[str, Any]) -> None:
    """Persist an Agentic telemetry event in PostgreSQL."""

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
    """

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    event["event_id"],
                    event["timestamp"],
                    event["system_type"],
                    event["project_id"],
                    event.get("model_id"),
                    event["trace_id"],
                    event["span_id"],
                    event["event_type"],
                    Jsonb(event.get("input")),
                    Jsonb(event.get("output")),
                    event.get("latency"),
                    event["status"],
                    event.get("error"),
                    Jsonb(event.get("metadata", {})),
                ),
            )

        connection.commit()


def get_all_events() -> list[dict[str, Any]]:
    """Retrieve all Agentic telemetry events from PostgreSQL."""

    query = """
        SELECT
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
        FROM events
        ORDER BY timestamp ASC
    """

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    columns = [
        "event_id",
        "timestamp",
        "system_type",
        "project_id",
        "model_id",
        "trace_id",
        "span_id",
        "event_type",
        "input",
        "output",
        "latency",
        "status",
        "error",
        "metadata",
    ]

    return [
        dict(zip(columns, row))
        for row in rows
    ]