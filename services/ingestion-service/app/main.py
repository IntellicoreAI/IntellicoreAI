
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from jsonschema import Draft202012Validator, FormatChecker
from postgrest.exceptions import APIError

from app.supabase_client import get_supabase_client


SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SERVICE_ROOT.parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "base-event.schema.json"


def load_event_validator() -> Draft202012Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8-sig") as file:
        schema = json.load(file)

    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )


event_validator = load_event_validator()

app = FastAPI(
    title="IntelliCore OS Ingestion Service",
    version="0.1.0",
    description="Validates and receives normalized IntelliCore events.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingestion-service"}


@app.post("/events", status_code=202)
async def ingest_event(request: Request) -> dict[str, Any]:
    try:
        event = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Request body must be valid JSON",
        )

    if not isinstance(event, dict):
        raise HTTPException(
            status_code=422,
            detail="Event must be a JSON object",
        )

    errors = sorted(
        event_validator.iter_errors(event),
        key=lambda error: list(map(str, error.absolute_path)),
    )

    if errors:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "path": ".".join(str(part) for part in error.absolute_path),
                    "message": error.message,
                }
                for error in errors
            ],
        )

    model = event["model"]

    database_event = {
        "event_id": event["event_id"],
        "project_id": event["project_id"],
        "environment": event["environment"],
        "event_type": event["event_type"],
        "occurred_at": event["occurred_at"],
        "model_name": model["model_name"],
        "model_version": model["model_version"],
        "event_data": {
            "schema_version": event["schema_version"],
            **event["data"],
        },
    }

    try:
        client = get_supabase_client()
        client.table("ml_events").insert(database_event).execute()
    except APIError:
        raise HTTPException(
            status_code=500,
            detail="Failed to persist event to the database",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Ingestion service could not persist the event",
        )

    return {
        "status": "accepted",
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "project_id": event["project_id"],
        "persisted": True,
    }
