
"""Generate and validate structured ML diagnosis signals."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.signals.schemas import (
    SIGNAL_SCHEMA_VERSION,
    MLDiagnosisSignal,
    MLDiagnosisSignalDocument,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "diagnosis"
    / "diagnosis_result.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "diagnosis"
    / "ml_diagnosis_signals.json"
)

DEFAULT_PROJECT_ID = "intellicore-credit-risk"
DEFAULT_ENVIRONMENT = "development"


# Map diagnosis issues to their respective signal categories.
ISSUE_SIGNAL_TYPES = {
    "telemetry_data_quality_issue": "data_quality",
    "data_distribution_change": "data_drift",
    "prediction_distribution_change": "prediction_drift",
    "model_performance_degradation": "performance_degradation",
}

# Default severity for each diagnosis issue.
ISSUE_SEVERITY = {
    "telemetry_data_quality_issue": "MEDIUM",
    "data_distribution_change": "MEDIUM",
    "prediction_distribution_change": "MEDIUM",
    "model_performance_degradation": "HIGH",
}


def load_diagnosis():
    """Load and validate the diagnosis result JSON."""

    if not DIAGNOSIS_PATH.exists():
        raise FileNotFoundError(
            f"Diagnosis file not found: {DIAGNOSIS_PATH}"
        )

    with DIAGNOSIS_PATH.open("r", encoding="utf-8-sig") as file:
        diagnosis = json.load(file)

    if not isinstance(diagnosis, dict):
        raise ValueError("Diagnosis result must be a JSON object.")

    if not isinstance(diagnosis.get("diagnoses"), list):
        raise ValueError(
            "Diagnosis result must contain a diagnoses list."
        )

    return diagnosis


def get_severity(issue, status, alert_severity=None):
    """Determine signal severity based on issue and alert severity."""

    if status == "warning":
        return "LOW"

    severity = ISSUE_SEVERITY.get(issue, "MEDIUM")

    if (
        severity == "MEDIUM"
        and alert_severity in {"HIGH", "CRITICAL"}
    ):
        return alert_severity

    return severity


def generate_signals(
    diagnosis,
    project_id=DEFAULT_PROJECT_ID,
    environment=DEFAULT_ENVIRONMENT,
):
    """Convert diagnosis findings into validated ML signals."""

    if not isinstance(diagnosis, dict):
        raise ValueError("Diagnosis must be a JSON object.")

    if not isinstance(diagnosis.get("diagnoses"), list):
        raise ValueError(
            "Diagnosis result must contain a diagnoses list."
        )

    diagnosis_id = diagnosis.get("diagnosis_id")

    if not diagnosis_id:
        raise ValueError("Missing diagnosis_id.")

    model = diagnosis.get("model") or {}

    if not isinstance(model, dict):
        raise ValueError(
            "Model information must be a JSON object."
        )

    created_at = datetime.now(timezone.utc).isoformat()

    signals = []

    for item in diagnosis["diagnoses"]:

        if not isinstance(item, dict):
            continue

        status = item.get("status")

        if status not in {"detected", "failed", "warning"}:
            continue

        issue = item.get("issue")

        if not issue:
            continue

        signal = {
            "schema_version": SIGNAL_SCHEMA_VERSION,
            "signal_id": f"ml-signal-{uuid4().hex}",
            "created_at": created_at,
            "source": "ml_monitoring.diagnosis_engine",

            "signal_type": ISSUE_SIGNAL_TYPES.get(
                issue, "other"
            ),

            "status": status,

            "severity": get_severity(
                issue,
                status,
                diagnosis.get("alert_severity"),
            ),

            "project_id": project_id,
            "environment": environment,

            "model": {
                "model_name": str(
                    model.get("model_name", "unknown")
                ),
                "model_version": str(
                    model.get("model_version", "unknown")
                ),
            },

            "issue": issue,

            "investigation_area": item.get(
                "investigation_area",
                "unspecified",
            ),

            "evidence": item.get("evidence") or {},

            "source_alert_id": diagnosis.get(
                "source_alert_id"
            ),

            "diagnosis_id": diagnosis_id,

            # Root cause is not assumed without sufficient evidence.
            "root_cause": None,
        }

        # Validate every individual signal.
        validated_signal = MLDiagnosisSignal.model_validate(
            signal
        )

        signals.append(
            validated_signal.model_dump(mode="json")
        )

    # Construct the complete signal document.
    signal_document = {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "generated_at": created_at,
        "source_diagnosis_id": diagnosis_id,
        "signal_count": len(signals),
        "signals": signals,
    }

    # Validate the complete document, including signal_count.
    validated_document = (
        MLDiagnosisSignalDocument.model_validate(
            signal_document
        )
    )

    return validated_document.model_dump(mode="json")


def save_signals(signal_document):
    """Save validated diagnosis signals as formatted JSON."""

    # Validate again before writing to disk.
    validated_document = (
        MLDiagnosisSignalDocument.model_validate(
            signal_document
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            validated_document.model_dump(mode="json"),
            file,
            indent=4,
        )

    print(f"Signals saved to: {OUTPUT_PATH}")


def main():
    """Execute the complete signal-generation pipeline."""

    diagnosis = load_diagnosis()

    signal_document = generate_signals(diagnosis)

    save_signals(signal_document)

    print("\nINTELLICORE OS - ML DIAGNOSIS SIGNAL GENERATOR")
    print("---------------------------------------------")

    print(
        f"Diagnosis ID: "
        f"{signal_document['source_diagnosis_id']}"
    )

    print(
        f"Signals generated: "
        f"{signal_document['signal_count']}"
    )

    print("Schema validation: PASSED")

    for signal in signal_document["signals"]:
        print(
            f"- {signal['signal_type']}: "
            f"{signal['issue']} "
            f"({signal['status']}) "
            f"[{signal['severity']}]"
        )


if __name__ == "__main__":
    main()
