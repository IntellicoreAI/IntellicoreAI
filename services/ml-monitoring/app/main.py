
"""Run IntelliCore OS monitoring detectors and publish their reports."""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.monitoring.alert_engine import (
    DATA_QUALITY_REPORT_PATH,
    DRIFT_REPORT_PATH,
    MONITORING_DIR,
    OUTPUT_PATH,
    PERFORMANCE_REPORT_PATH,
    PREDICTION_DRIFT_REPORT_PATH,
    generate_unified_alert,
)


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = MONITORING_DIR / "monitoring_config.json"

INGESTION_URL = os.getenv(
    "INTELLICORE_INGESTION_URL",
    "http://127.0.0.1:8000/events",
)
PROJECT_ID = os.getenv(
    "INTELLICORE_PROJECT_ID",
    "intellicore-credit-risk",
)
ENVIRONMENT = os.getenv("INTELLICORE_ENVIRONMENT", "development")
HTTP_TIMEOUT_SECONDS = 15
TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "telemetry" / "supabase_prediction_events.jsonl"
)
SYNC_SCRIPT = PROJECT_ROOT / "sync_supabase_telemetry.py"



# Run order: data quality, feature drift, prediction drift,
# and controlled performance degradation evaluation.
MONITORING_DETECTORS = [
    ("Data quality", "app.data_quality.data_quality_monitor"),
    ("Data drift", "app.monitoring.drift_detector"),
    ("Prediction drift", "app.monitoring.prediction_drift"),
    (
        "Performance degradation",
        "app.monitoring.performance_degradation_detector",
    ),
]


# ============================================================
# SUPABASE TELEMETRY SYNC
# ============================================================

def sync_supabase_telemetry() -> None:
    """Refresh the generated JSONL file from persisted prediction events."""

    print("\\nSyncing prediction telemetry from Supabase...")

    if not SYNC_SCRIPT.exists():
        raise FileNotFoundError(f"Telemetry sync script not found: {SYNC_SCRIPT}")

    subprocess.run(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        check=True,
        env={
            **os.environ,
            "INTELLICORE_PROJECT_ID": PROJECT_ID,
            "INTELLICORE_ENVIRONMENT": ENVIRONMENT,
        },
    )

    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Synced telemetry file was not created: {TELEMETRY_PATH}"
        )


# ============================================================
# DETECTOR ORCHESTRATION
# ============================================================

def run_monitoring_detectors() -> None:
    """Generate fresh reports before publishing any monitoring events."""

    print("\n" + "=" * 70)
    print("INTELLICORE OS - MONITORING PIPELINE")
    print("=" * 70)

    for name, module in MONITORING_DETECTORS:
        print(f"\nRunning {name}...")

        subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(SERVICE_ROOT),
            check=True,
            env={
                **os.environ,
                "INTELLICORE_TELEMETRY_PATH": str(TELEMETRY_PATH),
            },
        )

        print(f"{name} completed successfully.")

    print("\nAll monitoring detectors completed successfully.")


# ============================================================
# JSON AND MODEL CONFIGURATION
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def get_model(config: dict[str, Any]) -> dict[str, str]:
    """Read and validate the monitored model identity."""

    model = config.get("model")

    if not isinstance(model, dict):
        raise ValueError(
            "Monitoring configuration must define a model object."
        )

    model_name = model.get("model_name")
    model_version = model.get("model_version")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model.model_name must be a non-empty string.")

    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError(
            "model.model_version must be a non-empty string."
        )

    return {
        "model_name": model_name,
        "model_version": model_version,
    }


# ============================================================
# EVENT CREATION
# ============================================================

def build_event(
    *,
    event_type: str,
    signal_type: str,
    report: dict[str, Any],
    model: dict[str, str],
) -> dict[str, Any]:
    """Wrap a monitoring artifact in the base-event envelope."""

    if event_type not in {"monitoring_result", "alert"}:
        raise ValueError(f"Unsupported event type: {event_type}")

    if ENVIRONMENT not in {"development", "staging", "production"}:
        raise ValueError(
            "INTELLICORE_ENVIRONMENT must be development, staging, "
            "or production."
        )

    event_data: dict[str, Any] = {"signal_type": signal_type}

    if event_type == "alert":
        event_data["alert"] = report
    else:
        event_data["report"] = report

    return {
        "schema_version": "1.0",
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "project_id": PROJECT_ID,
        "environment": ENVIRONMENT,
        "model": model,
        "data": event_data,
    }


# ============================================================
# INGESTION API
# ============================================================

def publish_event(event: dict[str, Any]) -> dict[str, Any]:
    """Submit one event and verify that ingestion persisted it."""

    body = json.dumps(event, allow_nan=False).encode("utf-8")

    request = urllib.request.Request(
        INGESTION_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"Ingestion API returned HTTP {exc.code}: {error_body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach ingestion API at {INGESTION_URL}: "
            f"{exc.reason}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "Ingestion API returned an invalid response."
        )

    if (
        result.get("status") != "accepted"
        or result.get("persisted") is not True
        or result.get("event_id") != event["event_id"]
    ):
        raise RuntimeError(
            "Ingestion did not confirm persistence for event "
            f"{event['event_id']}."
        )

    return result


# ============================================================
# REPORT PUBLISHING
# ============================================================

def save_unified_alert(alert: dict[str, Any]) -> None:
    """Persist the newly generated unified alert."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            alert,
            file,
            indent=4,
            allow_nan=False,
        )


def run_publishing_workflow() -> list[dict[str, Any]]:
    """Publish four monitoring reports and one unified alert."""

    config = load_json(CONFIG_PATH)
    model = get_model(config)

    report_paths = [
        ("data_drift", DRIFT_REPORT_PATH),
        ("prediction_drift", PREDICTION_DRIFT_REPORT_PATH),
        ("performance_degradation", PERFORMANCE_REPORT_PATH),
        ("data_quality", DATA_QUALITY_REPORT_PATH),
    ]

    # All detectors must have finished before these files are read.
    loaded_reports = [
        (signal_type, load_json(path))
        for signal_type, path in report_paths
    ]

    # Generate the alert from the freshly generated reports.
    alert = generate_unified_alert()
    save_unified_alert(alert)

    events = [
        build_event(
            event_type="monitoring_result",
            signal_type=signal_type,
            report=report,
            model=model,
        )
        for signal_type, report in loaded_reports
    ]

    events.append(
        build_event(
            event_type="alert",
            signal_type="unified_alert",
            report=alert,
            model=model,
        )
    )

    results = []

    for index, event in enumerate(events, start=1):
        response = publish_event(event)
        results.append(response)

        print(
            f"[{index}/{len(events)}] Persisted "
            f"{event['data']['signal_type']}: "
            f"{response['event_id']}"
        )

    return results


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """Run detectors, generate the alert, and publish events."""

    print("IntelliCore OS - End-to-End Monitoring")
    print(f"Ingestion endpoint: {INGESTION_URL}")
    print(f"Project: {PROJECT_ID}")
    print(f"Environment: {ENVIRONMENT}")

    # 1. Refresh prediction telemetry from Supabase.
    sync_supabase_telemetry()

    # 2. Generate fresh monitoring reports.
    run_monitoring_detectors()

    # 3. Publish the four reports and unified alert.
    results = run_publishing_workflow()

    print(
        "\nPipeline completed successfully. "
        f"{len(results)} events persisted."
    )


if __name__ == "__main__":
    main()
