
"""Feature-level data drift detection for IntelliCore AI."""

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MONITORING_DIR = PROJECT_ROOT / "data" / "artifacts" / "monitoring"
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
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"

PSI_LIKE_THRESHOLD = 0.10


# ============================================================
# CONFIGURATION
# ============================================================

def load_monitoring_config() -> dict[str, Any]:
    """Load and validate monitoring configuration."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Monitoring configuration not found: {CONFIG_PATH}"
        )

    try:
        with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in monitoring configuration: {CONFIG_PATH}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("Monitoring configuration must be a JSON object.")

    for section in ("data_drift", "prediction_drift"):
        section_config = config.get(section)

        if not isinstance(section_config, dict):
            raise ValueError(
                f"Configuration must define the {section} object."
            )

        alpha = section_config.get("significance_level")

        if isinstance(alpha, bool):
            raise ValueError(
                f"{section}.significance_level must be numeric "
                "and strictly between 0 and 1."
            )

        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{section}.significance_level must be numeric."
            ) from exc

        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError(
                f"{section}.significance_level must be "
                "strictly between 0 and 1."
            )

        section_config["significance_level"] = alpha

    correction = config["data_drift"].get(
        "multiple_testing_correction", "none"
    )

    if not isinstance(correction, str):
        raise ValueError(
            "data_drift.multiple_testing_correction must be a string."
        )

    correction = correction.lower()

    if correction not in {"none", "bonferroni"}:
        raise ValueError(
            "multiple_testing_correction must be 'none' or 'bonferroni'."
        )

    config["data_drift"]["multiple_testing_correction"] = correction

    return config


def validate_significance_level(value):
    """Validate that a statistical significance level is between 0 and 1."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value < 1
    ):
        raise ValueError(
            "significance_level must be a number strictly between 0 and 1."
        )

    return float(value)



def get_significance_level() -> float:
    config = load_monitoring_config()
    significance_level = config["data_drift"]["significance_level"]

    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, (int, float))
        or not 0 < significance_level < 1
    ):
        raise ValueError(
            "data_drift.significance_level must be a number "
            "strictly between 0 and 1."
        )

    return float(significance_level)


# ============================================================
# JSON AND DATA LOADING
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and validate its root structure."""

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


def load_baseline() -> dict[str, Any]:
    """Load and validate the reference baseline."""

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
            "Baseline feature_columns must be a non-empty "
            "list of feature names."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Baseline contains duplicate feature names.")

    return baseline


def load_telemetry() -> list[dict[str, Any]]:
    """Load prediction events from the JSONL telemetry file."""

    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {TELEMETRY_PATH}"
        )

    events = []

    with TELEMETRY_PATH.open("r", encoding="utf-8-sig") as file:
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


def telemetry_to_dataframe(
    events: list[dict[str, Any]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Extract expected input features from prediction events."""

    rows = []

    for event in events:
        input_data = event.get("input")
        if not isinstance(input_data, dict):
            input_data = {}

        features = input_data.get("features")
        if not isinstance(features, dict):
            features = {}

        rows.append({
            feature: features.get(feature)
            for feature in feature_columns
        })

    return pd.DataFrame(rows, columns=feature_columns)


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def prepare_numeric_series(values: pd.Series) -> pd.Series:
    """Convert values to numeric and remove missing/non-finite values."""

    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    return numeric.dropna()


# ============================================================
# KS DRIFT TEST
# ============================================================

def calculate_ks_drift(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Compare two distributions using the configured KS threshold."""

    alpha = get_significance_level()
    reference = prepare_numeric_series(reference)
    current = prepare_numeric_series(current)

    if len(reference) == 0 or len(current) == 0:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "status": "insufficient_data",
            "reference_sample_count": int(len(reference)),
            "current_sample_count": int(len(current)),
            "significance_level": alpha,
            "effective_significance_level": alpha,
        }

    test_result = ks_2samp(reference.to_numpy(), current.to_numpy())
    statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)
    drift_detected = p_value < alpha

    return {
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": bool(drift_detected),
        "status": (
            "drift_detected"
            if drift_detected
            else "no_significant_drift"
        ),
        "reference_sample_count": int(len(reference)),
        "current_sample_count": int(len(current)),
        "significance_level": alpha,
        "effective_significance_level": alpha,
    }


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary_statistics(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Calculate distribution statistics and missing-rate changes."""

    reference_numeric = pd.to_numeric(reference, errors="coerce")
    current_numeric = pd.to_numeric(current, errors="coerce")

    reference_missing = float(reference_numeric.isna().mean())
    current_missing = float(current_numeric.isna().mean())

    reference_clean = prepare_numeric_series(reference_numeric)
    current_clean = prepare_numeric_series(current_numeric)

    reference_mean = reference_clean.mean()
    current_mean = current_clean.mean()
    reference_std = reference_clean.std()
    current_std = current_clean.std()

    if (
        pd.notna(reference_mean)
        and reference_mean != 0
        and pd.notna(current_mean)
    ):
        mean_change = (
            (current_mean - reference_mean) / abs(reference_mean)
        ) * 100
    else:
        mean_change = None

    return {
        "reference_mean": (
            float(reference_mean) if pd.notna(reference_mean) else None
        ),
        "current_mean": (
            float(current_mean) if pd.notna(current_mean) else None
        ),
        "mean_change_percentage": (
            float(mean_change) if mean_change is not None else None
        ),
        "reference_std": (
            float(reference_std) if pd.notna(reference_std) else None
        ),
        "current_std": (
            float(current_std) if pd.notna(current_std) else None
        ),
        "reference_missing_rate": reference_missing,
        "current_missing_rate": current_missing,
        "missing_rate_change": current_missing - reference_missing,
    }


# ============================================================
# FEATURE DRIFT ANALYSIS
# ============================================================

def analyze_feature_drift(
    baseline: dict[str, Any],
    current_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compare model features and apply multiple-testing correction."""

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference training data not found: {REFERENCE_DATA_PATH}"
        )

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    feature_columns = baseline["dataset"]["feature_columns"]

    missing_reference = [
        feature for feature in feature_columns
        if feature not in reference_df.columns
    ]
    missing_current = [
        feature for feature in feature_columns
        if feature not in current_df.columns
    ]

    if missing_reference:
        raise ValueError(
            f"Reference dataset is missing required features: "
            f"{missing_reference}"
        )

    if missing_current:
        raise ValueError(
            f"Current telemetry is missing required features: "
            f"{missing_current}"
        )

    config = load_monitoring_config()["data_drift"]
    alpha = config["significance_level"]
    correction = config["multiple_testing_correction"]

    # Run the tests first, then determine the number of valid tests.
    results = {}

    for feature in feature_columns:
        reference = prepare_numeric_series(reference_df[feature])
        current = prepare_numeric_series(current_df[feature])

        if len(reference) == 0 or len(current) == 0:
            test_result = {
                "ks_statistic": None,
                "p_value": None,
                "reference_sample_count": int(len(reference)),
                "current_sample_count": int(len(current)),
                "status": "insufficient_data",
                "drift_detected": False,
            }
        else:
            test = ks_2samp(reference.to_numpy(), current.to_numpy())
            test_result = {
                "ks_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "reference_sample_count": int(len(reference)),
                "current_sample_count": int(len(current)),
                "status": "pending",
                "drift_detected": False,
            }

        results[feature] = {
            **test_result,
            **calculate_summary_statistics(
                reference_df[feature],
                current_df[feature],
            ),
        }

    tested_count = sum(
        result["p_value"] is not None
        for result in results.values()
    )

    effective_alpha = (
        alpha / tested_count
        if correction == "bonferroni" and tested_count > 0
        else alpha
    )

    for result in results.values():
        p_value = result["p_value"]

        if p_value is None:
            result["status"] = "insufficient_data"
            result["drift_detected"] = False
        else:
            detected = p_value < effective_alpha
            result["drift_detected"] = bool(detected)
            result["status"] = (
                "drift_detected"
                if detected
                else "no_significant_drift"
            )

        result["significance_level"] = alpha
        result["effective_significance_level"] = effective_alpha
        result["multiple_testing_correction"] = correction
        result["tests_evaluated"] = tested_count

    return results


# ============================================================
# OVERALL DRIFT SUMMARY
# ============================================================

def create_overall_summary(
    feature_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create an overall summary without hiding insufficient data."""

    total_features = len(feature_results)

    drifted = [
        feature for feature, result in feature_results.items()
        if result["drift_detected"]
    ]
    insufficient = [
        feature for feature, result in feature_results.items()
        if result["status"] == "insufficient_data"
    ]

    drift_rate = len(drifted) / total_features if total_features else 0.0

    if insufficient:
        status = "insufficient_data"
    elif drifted:
        status = "drift_detected"
    else:
        status = "no_significant_drift"

    return {
        "total_features": total_features,
        "drifted_features": len(drifted),
        "drifted_feature_names": drifted,
        "insufficient_data_features": insufficient,
        "drift_rate": float(drift_rate),
        "overall_status": status,
    }


# ============================================================
# CREATE DRIFT REPORT
# ============================================================

def create_drift_report() -> dict[str, Any]:
    """Run feature drift analysis and save the resulting report."""

    print("=" * 60)
    print("ML DATA DRIFT DETECTION")
    print("=" * 60)

    config = load_monitoring_config()
    drift_config = config["data_drift"]
    alpha = drift_config["significance_level"]
    correction = drift_config["multiple_testing_correction"]

    baseline = load_baseline()
    events = load_telemetry()
    feature_columns = baseline["dataset"]["feature_columns"]

    current_df = telemetry_to_dataframe(events, feature_columns)

    feature_results = analyze_feature_drift(baseline, current_df)
    summary = create_overall_summary(feature_results)

    tested_count = sum(
        result["p_value"] is not None
        for result in feature_results.values()
    )

    effective_alpha = (
        alpha / tested_count
        if correction == "bonferroni" and tested_count > 0
        else alpha
    )

    report = {
        "monitoring_version": "1.0",
        "reference": {
            "dataset": "data/processed/train.csv",
            "rows": int(baseline["dataset"]["rows"]),
        },
        "current_window": {
            "source": "data/telemetry/prediction_events.jsonl",
            "rows": len(events),
        },
        "method": {
            "test": "Kolmogorov-Smirnov two-sample test",
            "significance_level": alpha,
            "multiple_testing_correction": correction,
            "tests_evaluated": tested_count,
            "effective_significance_level": effective_alpha,
        },
        "overall_summary": summary,
        "feature_results": feature_results,
    }

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    with DRIFT_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, allow_nan=False)

    print("\n" + "=" * 60)
    print("DRIFT DETECTION RESULTS")
    print("=" * 60)

    print(f"Overall significance level: {alpha}")
    print(f"Correction: {correction}")
    print(f"Features tested: {tested_count}")
    print(f"Effective per-feature threshold: {effective_alpha}")

    for feature, result in feature_results.items():
        print(f"\n{feature}")
        print(f"  KS statistic: {result['ks_statistic']}")
        print(f"  p-value: {result['p_value']}")
        print(f"  Status: {result['status']}")
        print(f"  Reference mean: {result['reference_mean']}")
        print(f"  Current mean: {result['current_mean']}")

    print("\n" + "-" * 60)
    print(f"Total features: {summary['total_features']}")
    print(f"Drifted features: {summary['drifted_features']}")
    print(f"Drift rate: {summary['drift_rate']:.2%}")
    print(f"Overall status: {summary['overall_status']}")
    print(f"Drift report saved to: {DRIFT_REPORT_PATH}")

    return report


if __name__ == "__main__":
    create_drift_report()
