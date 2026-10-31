from pathlib import Path
import json

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

NORMAL_TELEMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "prediction_events.jsonl"
)

DRIFT_TELEMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "drift_test_events.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Features whose distributions will deliberately be shifted.
DRIFT_FEATURES = {
    "age": {
        "operation": "shift",
        "value": 10,
    },

    "MonthlyIncome": {
        "operation": "scale",
        "value": 1.8,
    },

    "DebtRatio": {
        "operation": "scale",
        "value": 1.8,
    },

    "RevolvingUtilizationOfUnsecuredLines": {
        "operation": "scale",
        "value": 1.8,
    },
}


# ============================================================
# LOAD TELEMETRY
# ============================================================

def load_telemetry():

    if not NORMAL_TELEMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Normal telemetry file not found:\n"
            f"{NORMAL_TELEMETRY_PATH}"
        )

    events = []

    with open(
        NORMAL_TELEMETRY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                events.append(json.loads(line))

    if not events:
        raise ValueError(
            "Normal telemetry contains no events."
        )

    return events


# ============================================================
# APPLY DRIFT
# ============================================================

def apply_drift(events):

    drifted_events = []

    for event in events:

        # Make a deep copy so the original event
        # is never modified.
        drifted_event = json.loads(
            json.dumps(event)
        )

        features = (
            drifted_event
            .get("input", {})
            .get("features", {})
        )

        # ----------------------------------------------------
        # Age
        # ----------------------------------------------------

        if "age" in features:

            age = features["age"]

            if age is not None:

                features["age"] = (
                    float(age) + 10
                )

        # ----------------------------------------------------
        # Monthly Income
        # ----------------------------------------------------

        if "MonthlyIncome" in features:

            income = features["MonthlyIncome"]

            if income is not None:

                features["MonthlyIncome"] = (
                    float(income) * 1.8
                )

        # ----------------------------------------------------
        # Debt Ratio
        # ----------------------------------------------------

        if "DebtRatio" in features:

            debt_ratio = features["DebtRatio"]

            if debt_ratio is not None:

                features["DebtRatio"] = (
                    float(debt_ratio) * 1.8
                )

        # ----------------------------------------------------
        # Revolving Utilization
        # ----------------------------------------------------

        if (
            "RevolvingUtilizationOfUnsecuredLines"
            in features
        ):

            utilization = features[
                "RevolvingUtilizationOfUnsecuredLines"
            ]

            if utilization is not None:

                features[
                    "RevolvingUtilizationOfUnsecuredLines"
                ] = (
                    float(utilization) * 1.8
                )

        # ----------------------------------------------------
        # Add monitoring metadata
        # ----------------------------------------------------

        drifted_event[
            "monitoring_test"
        ] = {
            "type": "controlled_drift",
            "purpose": (
                "Validate that the ML drift "
                "detector identifies "
                "distribution changes."
            ),
            "modified_features": list(
                DRIFT_FEATURES.keys()
            ),
        }

        drifted_events.append(
            drifted_event
        )

    return drifted_events


# ============================================================
# SAVE DRIFTED TELEMETRY
# ============================================================

def save_telemetry(events):

    DRIFT_TELEMETRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        DRIFT_TELEMETRY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for event in events:

            file.write(
                json.dumps(event)
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def create_drift_test():

    print("=" * 60)
    print("CONTROLLED DRIFT TEST DATA GENERATOR")
    print("=" * 60)

    print("\nLoading normal telemetry...")

    events = load_telemetry()

    print(
        f"Normal events loaded: {len(events)}"
    )

    print("\nApplying controlled distribution shifts...")

    drifted_events = apply_drift(events)

    print("\nModified features:")

    for feature, configuration in DRIFT_FEATURES.items():

        operation = configuration["operation"]
        value = configuration["value"]

        print(
            f"  - {feature}: "
            f"{operation} by {value}"
        )

    print("\nSaving drifted telemetry...")

    save_telemetry(drifted_events)

    print("\n" + "=" * 60)
    print("CONTROLLED DRIFT DATA CREATED")
    print("=" * 60)

    print(
        f"Events generated: "
        f"{len(drifted_events)}"
    )

    print(
        "\nSaved to:"
    )

    print(
        DRIFT_TELEMETRY_PATH
    )

    print(
        "\nOriginal telemetry was NOT modified."
    )

    print(
        "\nNext step:"
    )

    print(
        "Run the drift detector against "
        "drift_test_events.jsonl."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    create_drift_test()