
"""
Performance Degradation Detector
---------------------------------

Compares current model performance against an established
baseline and detects significant performance degradation.

Higher is better:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - PR-AUC

Lower is better:
    - Log Loss
    - Brier Score

A metric is considered degraded when its relative change
reaches or exceeds the configured degradation threshold.

Configuration:
    data/artifacts/monitoring/monitoring_config.json

Baseline:
    data/artifacts/monitoring/performance_report.json

Controlled validation:
    data/artifacts/monitoring/performance_degradation_test.json

Output:
    data/artifacts/monitoring/performance_degradation_report.json
"""

import json
import math
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "monitoring"
)

CONFIG_PATH = MONITORING_DIR / "monitoring_config.json"

BASELINE_REPORT_PATH = (
    MONITORING_DIR / "performance_report.json"
)

DEGRADED_TEST_PATH = (
    MONITORING_DIR / "performance_degradation_test.json"
)

OUTPUT_DIR = MONITORING_DIR

OUTPUT_PATH = (
    OUTPUT_DIR / "performance_degradation_report.json"
)


# ============================================================
# METRIC DEFINITIONS
# ============================================================

# Higher values indicate better performance.
HIGHER_IS_BETTER = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]

# Lower values indicate better performance.
LOWER_IS_BETTER = [
    "log_loss",
    "brier_score",
]

MONITORED_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER


# ============================================================
# CONFIGURATION
# ============================================================

def load_json(path):
    """Load a JSON object from a file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in: {path}"
        )

    return data


def load_degradation_threshold():
    """Load and validate the degradation threshold."""

    config = load_json(CONFIG_PATH)

    performance_config = config.get("performance", {})

    if not isinstance(performance_config, dict):
        raise ValueError(
            "'performance' configuration must be a JSON object."
        )

    threshold = performance_config.get(
        "relative_degradation_threshold"
    )

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError(
            "performance.relative_degradation_threshold "
            "must be a finite number greater than 0 and less than 1."
        )

    return float(threshold)


# The configuration file is the single source of truth.
RELATIVE_DEGRADATION_THRESHOLD = load_degradation_threshold()


# ============================================================
# METRIC VALIDATION
# ============================================================

def validate_metric_value(value, metric_name):
    """Convert a metric value to a valid finite float."""

    if isinstance(value, bool):
        raise ValueError(
            f"Metric '{metric_name}' must be numeric, not boolean."
        )

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metric '{metric_name}' must be numeric."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"Metric '{metric_name}' must be finite."
        )

    bounded_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",
    }

    if metric_name in bounded_metrics and not 0 <= value <= 1:
        raise ValueError(
            f"Metric '{metric_name}' must be between 0 and 1."
        )

    if metric_name == "log_loss" and value < 0:
        raise ValueError(
            "Metric 'log_loss' cannot be negative."
        )

    return value


# ============================================================
# CALCULATE RELATIVE CHANGE
# ============================================================

def calculate_relative_change(baseline_value, current_value):
    """
    Calculate relative change.

    Formula:
        (current - baseline) / abs(baseline)

    Returns None when the baseline is zero because relative
    change cannot be calculated in that case.
    """

    if isinstance(baseline_value, bool) or isinstance(
        current_value, bool
    ):
        raise ValueError(
            "Metric values must be numeric, not boolean."
        )

    try:
        baseline_value = float(baseline_value)
        current_value = float(current_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Metric values must be numeric."
        ) from exc

    if (
        not math.isfinite(baseline_value)
        or not math.isfinite(current_value)
    ):
        raise ValueError(
            "Metric values must be finite."
        )

    if baseline_value == 0:
        return None

    return (
        current_value - baseline_value
    ) / abs(baseline_value)


# ============================================================
# DETECT SINGLE METRIC DEGRADATION
# ============================================================

def evaluate_metric(
    metric_name,
    baseline_value,
    current_value,
):
    """
    Evaluate degradation for one performance metric.

    Higher-is-better metrics degrade when the relative decrease
    reaches the negative threshold.

    Lower-is-better metrics degrade when the relative increase
    reaches the positive threshold.

    math.isclose handles floating-point values that are
    mathematically equal to the threshold.
    """

    if metric_name not in MONITORED_METRICS:
        raise ValueError(
            f"Unsupported performance metric: {metric_name}"
        )

    baseline_value = validate_metric_value(
        baseline_value,
        metric_name,
    )

    current_value = validate_metric_value(
        current_value,
        metric_name,
    )

    relative_change = calculate_relative_change(
        baseline_value,
        current_value,
    )

    expected_direction = (
        "higher"
        if metric_name in HIGHER_IS_BETTER
        else "lower"
    )

    if relative_change is None:
        return {
            "baseline": baseline_value,
            "current": current_value,
            "relative_change": None,
            "relative_change_percent": None,
            "expected_good_direction": expected_direction,
            "degradation_direction": "not_calculable",
            "degradation_threshold_percent": (
                RELATIVE_DEGRADATION_THRESHOLD * 100
            ),
            "degradation_detected": False,
            "reason": (
                "Relative change cannot be calculated because "
                "the baseline value is zero."
            ),
        }

    if metric_name in HIGHER_IS_BETTER:
        threshold_boundary = -RELATIVE_DEGRADATION_THRESHOLD

        degraded = (
            math.isclose(
                relative_change,
                threshold_boundary,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or relative_change < threshold_boundary
        )

        direction = "decrease"

    else:
        threshold_boundary = RELATIVE_DEGRADATION_THRESHOLD

        degraded = (
            math.isclose(
                relative_change,
                threshold_boundary,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or relative_change > threshold_boundary
        )

        direction = "increase"

    return {
        "baseline": baseline_value,
        "current": current_value,
        "relative_change": relative_change,
        "relative_change_percent": (
            relative_change * 100
        ),
        "expected_good_direction": expected_direction,
        "degradation_direction": direction,
        "degradation_threshold_percent": (
            RELATIVE_DEGRADATION_THRESHOLD * 100
        ),
        "degradation_detected": bool(degraded),
        "reason": (
            "Relative change reached the degradation threshold."
            if degraded
            else "Relative change did not reach the degradation threshold."
        ),
    }


# ============================================================
# DETECT PERFORMANCE DEGRADATION
# ============================================================

def detect_degradation(
    baseline_metrics,
    current_metrics,
):
    """Compare available baseline and current performance metrics."""

    if not isinstance(baseline_metrics, dict):
        raise ValueError(
            "Baseline metrics must be a JSON object."
        )

    if not isinstance(current_metrics, dict):
        raise ValueError(
            "Current metrics must be a JSON object."
        )

    results = {}
    degraded_metrics = []

    for metric_name in MONITORED_METRICS:
        if (
            metric_name not in baseline_metrics
            or metric_name not in current_metrics
        ):
            continue

        result = evaluate_metric(
            metric_name,
            baseline_metrics[metric_name],
            current_metrics[metric_name],
        )

        results[metric_name] = result

        if result["degradation_detected"]:
            degraded_metrics.append(metric_name)

    return results, degraded_metrics


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    metric_results,
    degraded_metrics,
    skipped_metrics=None,
):
    """Print a readable performance degradation summary."""

    if skipped_metrics is None:
        skipped_metrics = []

    print("\n" + "=" * 70)
    print("PERFORMANCE DEGRADATION ANALYSIS")
    print("=" * 70)

    print(
        f"Configured degradation threshold: "
        f"{RELATIVE_DEGRADATION_THRESHOLD:.0%}"
    )

    print(f"Metrics evaluated: {len(metric_results)}")
    print(f"Metrics skipped: {len(skipped_metrics)}")

    print("\nMetric comparison:")
    print("-" * 70)

    for metric_name, result in metric_results.items():
        baseline = result["baseline"]
        current = result["current"]
        change = result["relative_change_percent"]

        change_text = (
            "N/A (zero baseline)"
            if change is None
            else f"{change:+.2f}%"
        )

        status = (
            "DEGRADED"
            if result["degradation_detected"]
            else "OK"
        )

        print(
            f"{metric_name:<15} "
            f"baseline={baseline:.6f}  "
            f"current={current:.6f}  "
            f"change={change_text:<22} "
            f"{status}"
        )

    if skipped_metrics:
        print("\nSkipped metrics:")

        for metric_name in skipped_metrics:
            print(
                f"  - {metric_name} "
                "(missing baseline or current value)"
            )

    print("\n" + "-" * 70)

    if degraded_metrics:
        print("STATUS: PERFORMANCE DEGRADATION DETECTED")
        print("Degraded metrics:")

        for metric_name in degraded_metrics:
            print(f"  - {metric_name}")
    else:
        print("STATUS: NO DEGRADATION DETECTED")

    print("=" * 70)


# ============================================================
# MAIN DETECTOR
# ============================================================

def main():
    """Run the controlled performance degradation detector."""

    print("=" * 70)
    print("INTELLICORE AI - PERFORMANCE DEGRADATION DETECTOR")
    print("=" * 70)

    print(
        f"\nConfigured relative degradation threshold: "
        f"{RELATIVE_DEGRADATION_THRESHOLD:.0%}"
    )

    # Load baseline metrics.
    print("\nLoading baseline performance...")

    baseline_report = load_json(BASELINE_REPORT_PATH)
    baseline_metrics = baseline_report.get("metrics")

    if not isinstance(baseline_metrics, dict):
        raise ValueError(
            "Baseline performance report must contain a "
            "'metrics' object."
        )

    print(f"Baseline source: {BASELINE_REPORT_PATH}")

    # Load the controlled degraded metrics.
    # This is a validation experiment, not live production data.
    print("\nLoading current performance...")

    current_report = load_json(DEGRADED_TEST_PATH)
    current_metrics = current_report.get("degraded_metrics")

    if not isinstance(current_metrics, dict):
        raise ValueError(
            "Controlled degradation report must contain a "
            "'degraded_metrics' object."
        )

    print(f"Current source: {DEGRADED_TEST_PATH}")

    # Compare metrics.
    metric_results, degraded_metrics = detect_degradation(
        baseline_metrics,
        current_metrics,
    )

    skipped_metrics = [
        metric
        for metric in MONITORED_METRICS
        if metric not in metric_results
    ]

    print_results(
        metric_results,
        degraded_metrics,
        skipped_metrics,
    )

    # Construct report.
    report = {
        "monitoring_type": "performance_degradation",
        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },
        "method": {
            "type": "relative_metric_change",
            "threshold": RELATIVE_DEGRADATION_THRESHOLD,
            "threshold_percent": (
                RELATIVE_DEGRADATION_THRESHOLD * 100
            ),
        },
        "baseline_source": str(
            BASELINE_REPORT_PATH.relative_to(PROJECT_ROOT)
        ),
        "current_source": str(
            DEGRADED_TEST_PATH.relative_to(PROJECT_ROOT)
        ),
        "evaluation_context": "controlled_degradation_test",
        "metrics": metric_results,
        "degraded_metrics": degraded_metrics,
        "skipped_metrics": skipped_metrics,
        "summary": {
            "total_metrics_evaluated": len(metric_results),
            "degraded_metric_count": len(degraded_metrics),
            "skipped_metric_count": len(skipped_metrics),
            "performance_degradation_detected": bool(
                degraded_metrics
            ),
            "overall_status": (
                "performance_degradation_detected"
                if degraded_metrics
                else "no_significant_degradation"
            ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    print(f"\nSaved report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
