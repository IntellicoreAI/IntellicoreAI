
"""Missing-value statistics and configurable threshold checks."""

from typing import Mapping, Any

import pandas as pd


def check_missing_values(
    df: pd.DataFrame,
    *,
    max_missing_rate_by_column: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    Calculate missing-value counts and rates per column.

    Thresholds are optional fractions between 0 and 1.
    A column without a configured threshold is monitored
    but does not fail solely because it contains missing values.
    """

    thresholds = dict(max_missing_rate_by_column or {})

    for column, threshold in thresholds.items():
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                f"Threshold for '{column}' must be between 0 and 1."
            )

    columns = {}
    violations = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        missing_rate = (
            float(df[column].isna().mean())
            if len(df) > 0
            else 0.0
        )

        threshold = thresholds.get(column)

        exceeded = (
            threshold is not None
            and missing_rate > threshold
        )

        columns[column] = {
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "max_allowed_missing_rate": threshold,
            "threshold_exceeded": bool(exceeded),
        }

        if exceeded:
            violations.append(column)

    unknown_columns = sorted(
        set(thresholds) - set(df.columns)
    )

    violations.extend(unknown_columns)

    return {
        "check": "missing_values",
        "status": "failed" if violations else "passed",
        "row_count": int(len(df)),
        "columns": columns,
        "threshold_violations": sorted(set(violations)),
        "unknown_threshold_columns": unknown_columns,
    }
