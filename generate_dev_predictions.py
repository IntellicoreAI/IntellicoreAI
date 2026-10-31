import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "artifacts" / "xgboost" / "model.json"
FEATURES_PATH = ROOT / "data" / "artifacts" / "xgboost" / "feature_names.json"
DATA_PATH = ROOT / "data" / "processed" / "test.csv"

INGESTION_URL = os.getenv(
    "INTELLICORE_INGESTION_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
PROJECT_ID = os.getenv(
    "INTELLICORE_PROJECT_ID",
    "intellicore-credit-risk",
)
ENVIRONMENT = "development"
BATCH_SIZE = 10


def main():
    sys.path.insert(0, str(ROOT / "services" / "ingestion-service"))
    from app.event_adapter import adapt_prediction_event

    with FEATURES_PATH.open(encoding="utf-8") as file:
        feature_names = json.load(file)

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    dataset = pd.read_csv(DATA_PATH)
    X = dataset[feature_names].head(BATCH_SIZE)

    if len(X) < BATCH_SIZE:
        raise RuntimeError("Not enough rows in the test dataset.")

    if X.isnull().any().any():
        raise RuntimeError("Input rows contain missing feature values.")

    predictions = model.predict(
        xgb.DMatrix(X, feature_names=feature_names)
    )

    health = requests.get(f"{INGESTION_URL}/health", timeout=5)
    health.raise_for_status()
    print("Ingestion service health:", health.json())

    accepted = 0

    for index, probability in enumerate(predictions):
        row = X.iloc[index]

        source_event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": {
                "model_name": "xgboost_credit_risk",
                "model_version": "1.0",
            },
            "input": {
                "features": {
                    name: float(row[name])
                    for name in feature_names
                },
                "provenance": {
                    "source": "processed_test_dataset",
                    "mode": "development_batch",
                    "row_index": index,
                    "note": (
                        "New inference on a held-out test row; "
                        "not live production traffic."
                    ),
                },
            },
            "prediction": {
                "prediction_probability": float(probability),
                "predicted_class": int(probability >= 0.5),
            },
        }

        event = adapt_prediction_event(
            source_event,
            project_id=PROJECT_ID,
            environment=ENVIRONMENT,
        )

        response = requests.post(
            f"{INGESTION_URL}/events",
            json=event,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        if (
            result.get("status") != "accepted"
            or result.get("persisted") is not True
            or result.get("event_id") != event["event_id"]
        ):
            raise RuntimeError(
                f"Event was not confirmed as persisted: {result}"
            )

        accepted += 1
        print(
            f"[{accepted}/{BATCH_SIZE}] "
            f"event_id={event['event_id']} "
            f"probability={probability:.6f} "
            f"persisted=True"
        )

    print(
        f"\nCompleted: {accepted}/{BATCH_SIZE} development "
        "prediction events confirmed persisted."
    )
    print("Historical telemetry files were not modified.")


if __name__ == "__main__":
    main()
