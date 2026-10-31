"""
Controlled Prediction Drift Test Generator
-------------------------------------------

Creates a separate telemetry file with intentionally shifted
model prediction probabilities.

The original prediction_events.jsonl is NOT modified.

Output:
    data/telemetry/prediction_drift_test_events.jsonl
"""

from pathlib import Path
import json


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "prediction_events.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "prediction_drift_test_events.jsonl"
)


# ============================================================
# DRIFT SETTINGS
# ============================================================

PROBABILITY_SHIFT = 0.25


# ============================================================
# CREATE CONTROLLED DRIFT
# ============================================================

def create_prediction_drift_test():

    print("\n" + "=" * 70)
    print("CREATING CONTROLLED PREDICTION DRIFT TEST")
    print("=" * 70)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source telemetry file not found:\n{SOURCE_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_events = 0
    modified_events = 0

    with open(
        SOURCE_PATH,
        "r",
        encoding="utf-8",
    ) as source_file, open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for line in source_file:

            line = line.strip()

            if not line:
                continue

            event = json.loads(line)

            prediction = event.get("prediction", {})

            original_probability = prediction.get(
                "prediction_probability"
            )

            if original_probability is not None:

                original_probability = float(
                    original_probability
                )

                # Shift probability upward.
                shifted_probability = min(
                    original_probability
                    + PROBABILITY_SHIFT,
                    0.99,
                )

                prediction[
                    "prediction_probability"
                ] = shifted_probability

                # Recalculate predicted class using
                # the same 0.50 threshold used by the model.
                prediction["predicted_class"] = int(
                    shifted_probability >= 0.50
                )

                event["prediction"] = prediction

                modified_events += 1

            output_file.write(
                json.dumps(event) + "\n"
            )

            total_events += 1

    print(f"Original events: {total_events:,}")
    print(f"Modified events: {modified_events:,}")

    print(
        f"Probability shift: +{PROBABILITY_SHIFT}"
    )

    print(
        "\nOriginal telemetry was NOT modified."
    )

    print(
        f"\nSaved controlled prediction-drift test to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("CONTROLLED PREDICTION DRIFT TEST CREATED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_prediction_drift_test()