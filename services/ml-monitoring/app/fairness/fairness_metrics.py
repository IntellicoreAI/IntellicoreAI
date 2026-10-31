
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_fairness_metrics(
    y_true: Any,
    y_pred: Any,
    sensitive_features: Any,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Calculate binary classification fairness metrics by group.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Binary model predictions.
        sensitive_features: Explicit group label for each observation.
        positive_label: Label treated as the positive outcome.

    Returns:
        Group-level selection rate, TPR, FPR, and max-min disparities.

    Notes:
        This function calculates descriptive group comparisons. It does
        not determine whether a model is legally or ethically fair.
    """
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    groups = np.asarray(sensitive_features)

    if (
        true_values.ndim != 1
        or predicted_values.ndim != 1
        or groups.ndim != 1
    ):
        raise ValueError("Inputs must be one-dimensional.")

    if not (
        len(true_values) == len(predicted_values) == len(groups)
    ):
        raise ValueError("All inputs must have the same length.")

    if len(true_values) == 0:
        raise ValueError("At least one observation is required.")

    if pd.isna(true_values).any():
        raise ValueError("y_true must not contain missing values.")

    if pd.isna(predicted_values).any():
        raise ValueError("y_pred must not contain missing values.")

    if pd.isna(groups).any():
        raise ValueError("sensitive_features must not contain missing values.")

    allowed_labels = {0, 1}
    if not set(np.unique(true_values)).issubset(allowed_labels):
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    if not set(np.unique(predicted_values)).issubset(allowed_labels):
        raise ValueError("y_pred must contain only binary labels 0 and 1.")

    if positive_label not in allowed_labels:
        raise ValueError("positive_label must be 0 or 1.")

    true_positive = true_values == positive_label
    predicted_positive = predicted_values == positive_label

    frame = pd.DataFrame({
        "group": groups,
        "actual_positive": true_positive,
        "predicted_positive": predicted_positive,
    })

    group_results: dict[str, Any] = {}

    for group, subset in frame.groupby("group", sort=True, dropna=False):
        actual = subset["actual_positive"].to_numpy(dtype=bool)
        predicted = subset["predicted_positive"].to_numpy(dtype=bool)

        actual_positive_count = int(actual.sum())
        actual_negative_count = int((~actual).sum())

        selection_rate = float(predicted.mean())

        true_positive_rate = (
            float((predicted & actual).sum() / actual_positive_count)
            if actual_positive_count > 0
            else None
        )

        false_positive_rate = (
            float((predicted & ~actual).sum() / actual_negative_count)
            if actual_negative_count > 0
            else None
        )

        group_results[str(group)] = {
            "sample_count": int(len(subset)),
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "actual_positive_count": actual_positive_count,
            "actual_negative_count": actual_negative_count,
        }

    def disparity(metric_name: str) -> float | None:
        values = [
            result[metric_name]
            for result in group_results.values()
            if result[metric_name] is not None
        ]

        if len(values) < 2:
            return None

        return float(max(values) - min(values))

    return {
        "metric_type": "group_fairness",
        "positive_label": positive_label,
        "group_count": len(group_results),
        "groups": group_results,
        "disparities": {
            "demographic_parity_difference": disparity("selection_rate"),
            "equal_opportunity_difference": disparity("true_positive_rate"),
            "false_positive_rate_difference": disparity("false_positive_rate"),
        },
        "interpretation_note": (
            "Differences are descriptive, not proof of unfairness or "
            "fairness. Review group definitions, sample sizes, uncertainty, "
            "and the consequences of the positive label before interpreting."
        ),
    }
