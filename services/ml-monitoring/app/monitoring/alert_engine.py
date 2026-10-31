
"""
Unified Alert Engine for IntelliCore AI ML Monitoring.

Combines:
- Data drift report
- Prediction drift report
- Performance degradation report
- Data quality report

Into a single structured alert.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "monitoring"
)

DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"

PREDICTION_DRIFT_REPORT_PATH = (
    MONITORING_DIR / "prediction_drift_report.json"
)

PERFORMANCE_REPORT_PATH = (
    MONITORING_DIR / "performance_degradation_report.json"
)

DATA_QUALITY_REPORT_PATH = (
    MONITORING_DIR / "data_quality_report.json"
)

OUTPUT_PATH = MONITORING_DIR / "unified_alert.json"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON report and return its contents."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(
            f"Expected a JSON object in report: {path}"
        )

    return report


def determine_severity(
    data_drift_detected: bool,
    prediction_drift_detected: bool,
    performance_degradation_detected: bool,
    data_quality_failed: bool = False,
) -> str:
    """
    Determine alert severity from detected monitoring conditions.

    Initial project rules, not a production risk policy.

    Rules:
    - Performance degradation: HIGH.
    - Performance degradation combined with drift: HIGH.
    - Two or more detected conditions: HIGH.
    - One detected condition: MEDIUM.
    - No detected conditions: INFO.
    """

    detected_count = sum([
        data_drift_detected,
        prediction_drift_detected,
        performance_degradation_detected,
        data_quality_failed,
    ])

    if performance_degradation_detected:
        return "HIGH"

    if detected_count >= 2:
        return "HIGH"

    if detected_count == 1:
        return "MEDIUM"

    return "INFO"


# -------------------------------------------------------------------
# Main alert generation
# -------------------------------------------------------------------

def generate_unified_alert() -> dict[str, Any]:
    """Read monitoring reports and generate a unified alert."""

    # Load all four monitoring reports.
    drift_report = load_json(DRIFT_REPORT_PATH)

    prediction_drift_report = load_json(
        PREDICTION_DRIFT_REPORT_PATH
    )

    performance_report = load_json(
        PERFORMANCE_REPORT_PATH
    )

    data_quality_report = load_json(
        DATA_QUALITY_REPORT_PATH
    )

    # ---------------------------------------------------------------
    # Extract detection statuses
    # ---------------------------------------------------------------

    drift_summary = drift_report.get(
        "overall_summary", {}
    )

    prediction_comparison = prediction_drift_report.get(
        "comparison", {}
    )

    performance_summary = performance_report.get(
        "summary", {}
    )

    data_quality_checks = data_quality_report.get(
        "checks", {}
    )

    data_drift_detected = (
        drift_summary.get("overall_status")
        == "drift_detected"
    )

    prediction_drift_detected = bool(
        prediction_comparison.get(
            "drift_detected", False
        )
    )

    performance_degradation_report_detected = bool(
        performance_summary.get(
            "performance_degradation_detected", False
        )
    )

    performance_evaluation_context = performance_report.get(
        "evaluation_context", "operational"
    )
    performance_degradation_controlled_test = (
        performance_evaluation_context == "controlled_degradation_test"
    )

    # Preserve detector results, but do not classify an intentional
    # degradation test as an operational performance incident.
    performance_degradation_detected = (
        performance_degradation_report_detected
        and not performance_degradation_controlled_test
    )

    data_quality_failed = (
        data_quality_report.get("overall_status")
        == "failed"
    )

    data_quality_warning = (
        data_quality_report.get("overall_status")
        == "warning"
    )

    # ---------------------------------------------------------------
    # Extract useful details
    # ---------------------------------------------------------------

    degraded_metrics = performance_report.get(
        "degraded_metrics", []
    )

    failed_checks = [
        name
        for name, result in data_quality_checks.items()
        if result.get("status") == "failed"
    ]

    warning_checks = [
        name
        for name, result in data_quality_checks.items()
        if result.get("status") == "warning"
    ]

    # ---------------------------------------------------------------
    # Build individual monitoring alerts
    # ---------------------------------------------------------------

    alerts = [
        {
            "type": "data_drift",
            "status": (
                "detected"
                if data_drift_detected
                else "not_detected"
            ),
            "drifted_feature_count": drift_summary.get(
                "drifted_features", 0
            ),
            "total_feature_count": drift_summary.get(
                "total_features", 0
            ),
            "drift_rate": drift_summary.get(
                "drift_rate", 0.0
            ),
            "drifted_features": drift_summary.get(
                "drifted_feature_names", []
            ),
        },
        {
            "type": "prediction_drift",
            "status": (
                "detected"
                if prediction_drift_detected
                else "not_detected"
            ),
            "ks_statistic": prediction_comparison.get(
                "ks_statistic"
            ),
            "p_value": prediction_comparison.get(
                "p_value"
            ),
            "positive_prediction_rate_difference": (
                prediction_comparison.get(
                    "positive_prediction_rate_difference"
                )
            ),
        },
        {
            "type": "performance_degradation",
            "status": (
                "detected"
                if performance_degradation_detected
                else "test_detected"
                if performance_degradation_controlled_test
                and performance_degradation_report_detected
                else "not_detected"
            ),
            "evaluation_context": performance_evaluation_context,
            "detected_in_report": performance_degradation_report_detected,
            "degraded_metric_count": performance_summary.get(
                "degraded_metric_count", 0
            ),
            "total_metrics_evaluated": performance_summary.get(
                "total_metrics_evaluated", 0
            ),
            "degraded_metrics": degraded_metrics,
        },
        {
            "type": "data_quality",
            "status": (
                "failed"
                if data_quality_failed
                else "warning"
                if data_quality_warning
                else "passed"
            ),
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
            "checks": data_quality_checks,
        },
    ]

    # ---------------------------------------------------------------
    # Determine overall status and severity
    # ---------------------------------------------------------------

    any_alert_detected = any([
        data_drift_detected,
        prediction_drift_detected,
        performance_degradation_detected,
        data_quality_failed,
    ])

    severity = determine_severity(
        data_drift_detected=data_drift_detected,
        prediction_drift_detected=prediction_drift_detected,
        performance_degradation_detected=(
            performance_degradation_detected
        ),
        data_quality_failed=data_quality_failed,
    )

    overall_status = (
        "alert"
        if any_alert_detected
        else "no_alert"
    )

    # ---------------------------------------------------------------
    # Build final unified alert
    # ---------------------------------------------------------------

    generated_at = datetime.now(timezone.utc)

    unified_alert = {
        "alert_id": (
            "ml-monitoring-"
            + generated_at.strftime("%Y%m%d%H%M%S")
        ),
        "generated_at": generated_at.isoformat(),
        "model": {
            "model_name": prediction_drift_report
            .get("model", {})
            .get("model_name", "unknown"),
            "model_version": prediction_drift_report
            .get("model", {})
            .get("model_version", "unknown"),
        },
        "overall_status": overall_status,
        "severity": severity,
        "alerts": alerts,
        "summary": {
            "data_drift_detected": data_drift_detected,
            "prediction_drift_detected": (
                prediction_drift_detected
            ),
            "performance_degradation_detected": (
                performance_degradation_detected
            ),
            "performance_degradation_report_detected": (
                performance_degradation_report_detected
            ),
            "performance_degradation_controlled_test": (
                performance_degradation_controlled_test
            ),
            "performance_degradation_evaluation_context": (
                performance_evaluation_context
            ),
            "data_quality_failed": data_quality_failed,
            "data_quality_warning": data_quality_warning,
            "total_alert_conditions": sum([
                data_drift_detected,
                prediction_drift_detected,
                performance_degradation_detected,
                data_quality_failed,
            ]),
        },
    }

    return unified_alert


# -------------------------------------------------------------------
# Save result
# -------------------------------------------------------------------

def main() -> None:
    """Generate, save, and print the unified alert."""

    alert = generate_unified_alert()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            alert,
            file,
            indent=4,
        )

    print("=" * 60)
    print("UNIFIED ALERT ENGINE")
    print("=" * 60)

    summary = alert["summary"]

    print(
        f"Data Drift:              "
        f"{summary['data_drift_detected']}"
    )

    print(
        f"Prediction Drift:        "
        f"{summary['prediction_drift_detected']}"
    )

    print(
        f"Performance Degradation: "
        f"{summary['performance_degradation_detected']}"
    )

    print(
        f"Data Quality Failed:     "
        f"{summary['data_quality_failed']}"
    )

    print(
        f"Data Quality Warning:    "
        f"{summary['data_quality_warning']}"
    )

    print(
        f"Overall Status:          "
        f"{alert['overall_status']}"
    )

    print(
        f"Severity:                "
        f"{alert['severity']}"
    )

    print(
        f"Alert Conditions:        "
        f"{summary['total_alert_conditions']}"
    )

    print("\nIndividual alert statuses:")

    for item in alert["alerts"]:
        print(
            f"  {item['type']}: {item['status']}"
        )

        if item["type"] == "data_quality":
            print(
                f"    Failed checks: "
                f"{item['failed_checks']}"
            )

    print()
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
