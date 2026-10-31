
import pytest

from app.fairness.fairness_metrics import calculate_fairness_metrics


def test_calculates_group_fairness_metrics():
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 0, 0]
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]

    result = calculate_fairness_metrics(y_true, y_pred, groups)

    assert result["metric_type"] == "group_fairness"
    assert result["group_count"] == 2

    assert result["groups"]["A"]["sample_count"] == 4
    assert result["groups"]["A"]["selection_rate"] == 0.5
    assert result["groups"]["A"]["true_positive_rate"] == 0.5
    assert result["groups"]["A"]["false_positive_rate"] == 0.5

    assert result["groups"]["B"]["sample_count"] == 4
    assert result["groups"]["B"]["selection_rate"] == 0.25
    assert result["groups"]["B"]["true_positive_rate"] == 0.5
    assert result["groups"]["B"]["false_positive_rate"] == 0.0

    assert (
        result["disparities"]["demographic_parity_difference"] == 0.25
    )
    assert (
        result["disparities"]["equal_opportunity_difference"] == 0.0
    )
    assert (
        result["disparities"]["false_positive_rate_difference"] == 0.5
    )


def test_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        calculate_fairness_metrics(
            y_true=[0, 1],
            y_pred=[0],
            sensitive_features=["A", "B"],
        )


def test_rejects_empty_inputs():
    with pytest.raises(ValueError, match="At least one"):
        calculate_fairness_metrics([], [], [])


def test_rejects_missing_group_values():
    with pytest.raises(ValueError, match="missing values"):
        calculate_fairness_metrics(
            y_true=[0, 1],
            y_pred=[0, 1],
            sensitive_features=["A", None],
        )


def test_rejects_non_binary_predictions():
    with pytest.raises(ValueError, match="binary labels"):
        calculate_fairness_metrics(
            y_true=[0, 1],
            y_pred=[0, 2],
            sensitive_features=["A", "B"],
        )


def test_handles_group_without_positive_actuals():
    result = calculate_fairness_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 0],
        sensitive_features=["A", "A", "B", "B"],
    )

    assert result["groups"]["A"]["true_positive_rate"] is None
    assert result["groups"]["B"]["true_positive_rate"] == 0.5
    assert (
        result["disparities"]["equal_opportunity_difference"] is None
    )
