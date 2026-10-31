
import pytest

from app.monitoring.performance_degradation_detector import (
    evaluate_metric,
    calculate_relative_change,
    detect_degradation,
)


def test_recall_decrease_at_20_percent_is_degradation():
    result = evaluate_metric("recall", 0.50, 0.40)
    assert result["degradation_detected"] is True


def test_recall_decrease_below_20_percent_is_not_degradation():
    result = evaluate_metric("recall", 0.50, 0.405)
    assert result["degradation_detected"] is False


def test_log_loss_increase_at_20_percent_is_degradation():
    result = evaluate_metric("log_loss", 0.50, 0.60)
    assert result["degradation_detected"] is True


def test_log_loss_decrease_is_not_degradation():
    result = evaluate_metric("log_loss", 0.50, 0.40)
    assert result["degradation_detected"] is False


def test_zero_baseline_has_no_relative_change():
    assert calculate_relative_change(0, 0.10) is None


@pytest.mark.parametrize(
    "baseline,current",
    [
        (float("nan"), 0.5),
        (0.5, float("inf")),
        (float("-inf"), 0.5),
    ],
)
def test_nonfinite_metric_values_are_rejected(baseline, current):
    with pytest.raises(ValueError):
        calculate_relative_change(baseline, current)


def test_missing_metrics_are_skipped():
    results, degraded = detect_degradation(
        {"recall": 0.50, "accuracy": 0.90},
        {"recall": 0.30},
    )

    assert "recall" in results
    assert "accuracy" not in results
    assert degraded == ["recall"]
