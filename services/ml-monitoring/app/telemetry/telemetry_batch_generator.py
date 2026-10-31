from pathlib import Path
import json
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
)

OUTPUT_PATH = OUTPUT_DIR / "prediction_events.jsonl"

# Number of telemetry events to generate
NUM_EVENTS = 1000

# Random seed for reproducibility
RANDOM_STATE = 42


# ============================================================
# IMPORT PREDICTOR
# ============================================================

APP_ROOT = Path(__file__).resolve().parents[1]

if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from prediction.predictor_02 import XGBoostPredictor


# ============================================================
# TELEMETRY BATCH GENERATOR
# ============================================================

class TelemetryBatchGenerator:
    """
    Generates multiple ML prediction telemetry events.

    Each event contains:

    - unique event ID
    - timestamp
    - model information
    - input features
    - prediction probability
    - predicted class
    - local SHAP explanation
    """

    def __init__(self):
        print("Initializing Telemetry Batch Generator...")

        # Load the existing prediction component
        self.predictor = XGBoostPredictor()

        # Load processed test data
        if not TEST_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Test dataset not found at: {TEST_DATA_PATH}"
            )

        self.test_data = pd.read_csv(TEST_DATA_PATH)

        print(
            f"Test dataset loaded successfully: "
            f"{self.test_data.shape}"
        )

        # Create output directory if necessary
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Telemetry output directory: {OUTPUT_DIR}"
        )

    # ========================================================
    # GENERATE SINGLE EVENT
    # ========================================================

    def generate_event(self, features):
        """
        Generate one telemetry event.
        """

        # Generate prediction and local SHAP explanation
        prediction_result = self.predictor.predict(features)

        # Generate unique event ID
        event_id = str(uuid.uuid4())

        # Generate UTC timestamp
        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        telemetry_event = {
            "event_id": event_id,

            "timestamp": timestamp,

            "model": {
                "model_name": "xgboost_credit_risk",
                "model_version": "1.0"
            },

            "input": {
                "features": features
            },

            "prediction": {
                "prediction_probability": (
                    prediction_result[
                        "prediction_probability"
                    ]
                ),

                "predicted_class": (
                    prediction_result[
                        "predicted_class"
                    ]
                )
            },

            "explanation": {
                "method": "SHAP",

                "feature_contributions": (
                    prediction_result[
                        "explanation"
                    ]
                )
            }
        }

        return telemetry_event

    # ========================================================
    # GENERATE BATCH
    # ========================================================

    def generate_batch(self, num_events=NUM_EVENTS):
        """
        Generate multiple telemetry events
        from the processed test dataset.
        """

        print("\n" + "=" * 60)
        print("GENERATING TELEMETRY BATCH")
        print("=" * 60)

        # Make sure we don't request more rows
        # than available in the test dataset.
        num_events = min(
            num_events,
            len(self.test_data)
        )

        print(
            f"Requested events: {num_events}"
        )

        # Sample test records
        sampled_data = self.test_data.sample(
            n=num_events,
            random_state=RANDOM_STATE
        )

        events = []

        for index, (_, row) in enumerate(
            sampled_data.iterrows(),
            start=1
        ):

            # Remove target column from input
            features = row.drop(
                labels=["SeriousDlqin2yrs"]
            ).to_dict()

            # Convert numeric values into
            # standard Python floats/ints
            cleaned_features = {}

            for feature, value in features.items():

                if pd.isna(value):
                    cleaned_features[feature] = None

                else:
                    cleaned_features[feature] = float(
                        value
                    )

            # Generate telemetry event
            event = self.generate_event(
                cleaned_features
            )

            events.append(event)

            # Progress display
            if index % 100 == 0 or index == num_events:
                print(
                    f"Generated {index}/{num_events} events"
                )

        return events

    # ========================================================
    # SAVE JSONL
    # ========================================================

    def save_events(self, events):
        """
        Save telemetry events as JSON Lines.

        One event is stored per line.
        """

        print("\nSaving telemetry events...")

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            for event in events:

                file.write(
                    json.dumps(event)
                    + "\n"
                )

        print(
            f"Telemetry events saved to:\n"
            f"{OUTPUT_PATH}"
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self, num_events=NUM_EVENTS):

        events = self.generate_batch(
            num_events
        )

        self.save_events(events)

        return events


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INTELLICORE AI - BATCH TELEMETRY GENERATOR")
    print("=" * 60)

    generator = TelemetryBatchGenerator()

    events = generator.run(
        num_events=NUM_EVENTS
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("TELEMETRY GENERATION SUMMARY")
    print("=" * 60)

    print(
        f"Total events generated: {len(events)}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    if events:

        print("\nFirst event preview:")

        print(
            json.dumps(
                events[0],
                indent=4
            )
        )

    print("\n" + "=" * 60)
    print("BATCH TELEMETRY GENERATION COMPLETED")
    print("=" * 60)