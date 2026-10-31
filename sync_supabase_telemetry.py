import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INGESTION_SERVICE = ROOT / "services" / "ingestion-service"
OUTPUT_PATH = ROOT / "data" / "telemetry" / "supabase_prediction_events.jsonl"

# Reuse the existing Supabase client and its environment configuration.
sys.path.insert(0, str(INGESTION_SERVICE))
from app.supabase_client import get_supabase_client


def main():
    project_id = os.getenv(
        "INTELLICORE_PROJECT_ID", "intellicore-credit-risk"
    )
    environment = os.getenv(
        "INTELLICORE_ENVIRONMENT", "development"
    )
    model_name = os.getenv(
        "INTELLICORE_MODEL_NAME", "xgboost_credit_risk"
    )
    model_version = os.getenv(
        "INTELLICORE_MODEL_VERSION", "1.0"
    )
    limit = int(os.getenv("INTELLICORE_TELEMETRY_LIMIT", "1000"))

    client = get_supabase_client()

    response = (
        client.table("ml_events")
        .select(
            "event_id,occurred_at,project_id,environment,"
            "model_name,model_version,event_data"
        )
        .eq("event_type", "prediction")
        .eq("project_id", project_id)
        .eq("environment", environment)
        .eq("model_name", model_name)
        .eq("model_version", model_version)
        .order("occurred_at", desc=True)
        .limit(limit)
        .execute()
    )

    rows = response.data or []
    events = []

    for row in rows:
        payload = row.get("event_data") or {}
        input_data = payload.get("input")
        prediction = payload.get("prediction")

        if not isinstance(input_data, dict):
            continue
        if not isinstance(prediction, dict):
            continue

        event = {
            "event_id": row["event_id"],
            "timestamp": row["occurred_at"],
            "model": {
                "model_name": row["model_name"],
                "model_version": row["model_version"],
            },
            "input": input_data,
            "prediction": prediction,
        }

        if isinstance(payload.get("explanation"), dict):
            event["explanation"] = payload["explanation"]

        events.append(event)

    # Restore chronological order after fetching the newest records.
    events.sort(key=lambda event: event["timestamp"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"Project: {project_id}")
    print(f"Environment: {environment}")
    print(f"Model: {model_name} v{model_version}")
    print(f"Supabase rows fetched: {len(rows)}")
    print(f"Valid prediction events written: {len(events)}")
    print(f"Output: {OUTPUT_PATH.relative_to(ROOT)}")

    if events:
        print(f"Earliest event: {events[0]['timestamp']}")
        print(f"Latest event:   {events[-1]['timestamp']}")
    else:
        print(
            "No matching prediction events were written. "
            "Check the filters and Supabase records."
        )


if __name__ == "__main__":
    main()
