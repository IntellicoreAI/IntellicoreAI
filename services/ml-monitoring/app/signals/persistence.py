"""Persistence layer for ML diagnosis signals."""

from typing import Any

from supabase import Client

from app.signals.schemas import MLDiagnosisSignalDocument
from app.supabase_client import get_supabase_client


TABLE_NAME = "ml_diagnosis_signals"


def map_signal_to_database(signal: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated signal into a database row."""

    return {
        "signal_id": signal["signal_id"],
        "schema_version": signal["schema_version"],
        "created_at": signal["created_at"],
        "source": signal["source"],
        "signal_type": signal["signal_type"],
        "status": signal["status"],
        "severity": signal["severity"],
        "project_id": signal["project_id"],
        "environment": signal["environment"],
        "model_name": signal["model"]["model_name"],
        "model_version": signal["model"]["model_version"],
        "issue": signal["issue"],
        "investigation_area": signal["investigation_area"],
        "evidence": signal["evidence"],
        "source_alert_id": signal["source_alert_id"],
        "diagnosis_id": signal["diagnosis_id"],
        "root_cause": signal["root_cause"],
    }


def persist_signal_document(
    signal_document: dict[str, Any],
    client: Client | None = None,
) -> dict[str, Any]:
    """
    Validate and persist ML diagnosis signals.

    Existing signal IDs are ignored, making retries safe.
    """

    validated_document = MLDiagnosisSignalDocument.model_validate(
        signal_document
    )

    if not validated_document.signals:
        return {
            "success": True,
            "signal_count": 0,
            "message": "No signals to persist.",
        }

    supabase = client or get_supabase_client()

    rows = [
        map_signal_to_database(
            signal.model_dump(mode="json")
        )
        for signal in validated_document.signals
    ]

    response = (
        supabase.table(TABLE_NAME)
        .upsert(
            rows,
            on_conflict="signal_id",
            ignore_duplicates=True,
        )
        .execute()
    )

    return {
        "success": True,
        "signal_count": len(rows),
        "response_count": len(response.data or []),
        "message": "Signal persistence request completed.",
    }


def main():
    from pathlib import Path
    import json

    project_root = Path(__file__).resolve().parents[4]

    signal_path = (
        project_root
        / "data"
        / "artifacts"
        / "diagnosis"
        / "ml_diagnosis_signals.json"
    )

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Signal document not found: {signal_path}"
        )

    with signal_path.open("r", encoding="utf-8-sig") as file:
        signal_document = json.load(file)

    result = persist_signal_document(signal_document)

    print("\nINTELLICORE OS - ML SIGNAL PERSISTENCE")
    print("-------------------------------------")
    print(f"Success: {result['success']}")
    print(f"Signals processed: {result['signal_count']}")
    print(result["message"])


if __name__ == "__main__":
    main()
