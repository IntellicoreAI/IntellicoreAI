
"""Run and aggregate data-quality checks for IntelliCore AI ML monitoring."""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.data_quality.schema_check import (
    validate_schema,
    validate_numeric_features,
    validate_domain_rules,
)
from app.data_quality.telemetry_structure import (
    validate_telemetry_structure,
)
from app.data_quality.missing_values import check_missing_values
from app.data_quality.duplicates import check_duplicates
from app.data_quality.freshness import check_freshness


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MONITORING_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "monitoring"
)

CONFIG_PATH = MONITORING_DIR / "monitoring_config.json"
BASELINE_PATH = MONITORING_DIR / "baseline.json"

TELEMETRY_PATH = Path(
    os.getenv(
        "INTELLICORE_TELEMETRY_PATH",
        str(PROJECT_ROOT / "data" / "telemetry" / "prediction_events.jsonl"),
    )
)
if not TELEMETRY_PATH.is_absolute():
    TELEMETRY_PATH = PROJECT_ROOT / TELEMETRY_PATH

OUTPUT_PATH = MONITORING_DIR / "data_quality_report.json"


# ============================================================
# CONFIGURATION
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and ensure its root is an object."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in file: {path}")

    return data


def load_monitoring_config() -> dict[str, Any]:
    """Load and validate data-quality monitoring settings."""

    config = load_json(CONFIG_PATH)

    try:
        data_quality_config = config["data_quality"]
        max_missing_rate = data_quality_config["max_missing_rate"]
        max_event_age = data_quality_config["max_event_age"]
        duplicate_policy = data_quality_config[
            "duplicate_event_id_policy"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Configuration must define data_quality.max_missing_rate, "
            "data_quality.max_event_age, and "
            "data_quality.duplicate_event_id_policy."
        ) from exc

    if isinstance(max_missing_rate, bool):
        raise ValueError(
            "max_missing_rate must be a number between 0 and 1."
        )

    try:
        max_missing_rate = float(max_missing_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_missing_rate must be a number between 0 and 1."
        ) from exc

    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be between 0 and 1.")

    if (
        not isinstance(max_event_age, str)
        or not max_event_age.strip()
    ):
        raise ValueError(
            "max_event_age must be a non-empty duration string."
        )

    if duplicate_policy not in {"warning", "failed"}:
        raise ValueError(
            "duplicate_event_id_policy must be 'warning' or 'failed'."
        )

    return {
        **config,
        "data_quality": {
            **data_quality_config,
            "max_missing_rate": max_missing_rate,
            "max_event_age": max_event_age.strip(),
            "duplicate_event_id_policy": duplicate_policy,
        },
    }


# ============================================================
# TELEMETRY LOADING
# ============================================================

def load_events() -> list[dict[str, Any]]:
    """Load prediction events from the JSONL telemetry file."""

    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {TELEMETRY_PATH}"
        )

    events: list[dict[str, Any]] = []

    with TELEMETRY_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at telemetry line {line_number}."
                ) from exc

            if not isinstance(event, dict):
                raise ValueError(
                    f"Telemetry line {line_number} must contain "
                    "a JSON object."
                )

            events.append(event)

    if not events:
        raise ValueError("Telemetry file contains no events.")

    return events


# ============================================================
# TELEMETRY TRANSFORMATION
# ============================================================

def build_feature_dataframe(
    events: list[dict[str, Any]],
) -> pd.DataFrame:
    """Extract feature dictionaries for feature-level checks."""

    rows = []

    for event in events:
        input_data = event.get("input")

        if not isinstance(input_data, dict):
            input_data = {}

        features = input_data.get("features")

        if not isinstance(features, dict):
            features = {}

        rows.append(features)

    return pd.DataFrame(rows)


def build_event_dataframe(
    events: list[dict[str, Any]],
) -> pd.DataFrame:
    """Extract event identifiers and timestamps."""

    return pd.DataFrame({
        "event_id": [
            event.get("event_id") for event in events
        ],
        "timestamp": [
            event.get("timestamp") for event in events
        ],
    })


# ============================================================
# STATUS AGGREGATION
# ============================================================

def determine_overall_status(
    checks: dict[str, dict[str, Any]],
) -> str:
    """Aggregate individual check statuses."""

    statuses = [
        result["status"]
        for result in checks.values()
    ]

    if "failed" in statuses:
        return "failed"

    if "warning" in statuses:
        return "warning"

    return "passed"


# ============================================================
# MAIN MONITORING PIPELINE
# ============================================================

def run_data_quality_checks() -> dict[str, Any]:
    """Run all data-quality checks and save the monitoring report."""

    # Load configuration and reference baseline.
    config = load_monitoring_config()
    baseline = load_json(BASELINE_PATH)

    try:
        feature_columns = baseline["dataset"]["feature_columns"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Baseline must define dataset.feature_columns."
        ) from exc

    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or not all(
            isinstance(column, str) and column.strip()
            for column in feature_columns
        )
    ):
        raise ValueError(
            "Baseline feature_columns must be a non-empty list of names."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "Baseline feature_columns contains duplicate names."
        )

    # Load telemetry.
    events = load_events()

    # 1. Validate event-level structure before feature extraction.
    structure_report = validate_telemetry_structure(events)

    # Extract feature and event-level data.
    feature_df = build_feature_dataframe(events)
    event_df = build_event_dataframe(events)

    quality_config = config["data_quality"]

    max_missing_rate = quality_config["max_missing_rate"]
    max_event_age = quality_config["max_event_age"]
    duplicate_policy = quality_config["duplicate_event_id_policy"]

    # 2. Schema validation.
    schema_report = validate_schema(
        feature_df,
        feature_columns,
    )

    # 3. Numeric and non-finite feature validation.
    numeric_report = validate_numeric_features(
        feature_df,
        feature_columns,
    )

    # 4. Domain-specific feature validation.
    domain_report = validate_domain_rules(
        feature_df,
        feature_columns,
    )

    # 5. Missing-value validation.
    missing_report = check_missing_values(
        feature_df,
        max_missing_rate_by_column={
            feature: max_missing_rate
            for feature in feature_columns
        },
    )

    # 6. Duplicate event-ID detection.
    duplicate_report = check_duplicates(
        event_df,
        subset=["event_id"],
    )

    if (
        duplicate_report["status"] == "warning"
        and duplicate_policy == "failed"
    ):
        duplicate_report["status"] = "failed"

    duplicate_report["configured_policy"] = duplicate_policy

    # 7. Telemetry freshness.
    freshness_report = check_freshness(
        event_df,
        timestamp_column="timestamp",
        max_age=max_event_age,
    )

    # Aggregate every check.
    checks = {
        "telemetry_structure": structure_report,
        "schema": schema_report,
        "numeric_values": numeric_report,
        "domain_rules": domain_report,
        "missing_values": missing_report,
        "duplicates": duplicate_report,
        "freshness": freshness_report,
    }

    overall_status = determine_overall_status(checks)

    # Build monitoring report.
    report = {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config.get(
            "model",
            {
                "model_name": "xgboost_credit_risk",
                "model_version": "1.0",
            },
        ),
        "telemetry": {
            "source": str(TELEMETRY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "event_count": len(events),
        },
        "configuration": {
            "config_source": (
                "data/artifacts/monitoring/monitoring_config.json"
            ),
            "max_missing_rate": max_missing_rate,
            "max_event_age": max_event_age,
            "duplicate_event_id_policy": duplicate_policy,
        },
        "overall_status": overall_status,
        "checks": checks,
    }

    # Save report.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    return report


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """Execute monitoring and print a concise summary."""

    report = run_data_quality_checks()

    print("=" * 60)
    print("INTELLICORE AI - DATA QUALITY MONITOR")
    print("=" * 60)

    print(f"Model:            {report['model']['model_name']}")
    print(f"Model version:    {report['model']['model_version']}")
    print(f"Events evaluated: {report['telemetry']['event_count']}")
    print(f"Overall status:   {report['overall_status']}")

    print("\nIndividual checks:")

    for name, result in report["checks"].items():
        print(f"  {name}: {result['status']}")

        issues = result.get("issues", [])
        if issues:
            print(f"    Issues: {', '.join(issues)}")

    print(f"\nReport saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
