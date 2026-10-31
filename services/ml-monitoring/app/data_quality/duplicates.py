
"""Duplicate-row detection for ML monitoring."""

from typing import Iterable, Any

import pandas as pd


def check_duplicates(
    df: pd.DataFrame,
    *,
    subset: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Detect duplicate rows without modifying the input DataFrame.

    Args:
        df: Input monitoring window.
        subset: Columns used to identify duplicates.
            If None, all columns are compared.

    Returns:
        Structured duplicate-detection report.
    """

    subset_columns = (
        list(subset) if subset is not None else None
    )

    missing_columns = (
        [
            column
            for column in subset_columns
            if column not in df.columns
        ]
        if subset_columns is not None
        else []
    )

    if missing_columns:
        return {
            "check": "duplicate_rows",
            "status": "failed",
            "row_count": int(len(df)),
            "subset": subset_columns,
            "missing_subset_columns": missing_columns,
            "duplicate_count": None,
            "duplicate_rate": None,
            "issues": ["duplicate_check_subset_columns_missing"],
        }

    duplicate_mask = df.duplicated(
        subset=subset_columns,
        keep="first",
    )

    duplicate_count = int(duplicate_mask.sum())

    duplicate_rate = (
        float(duplicate_count / len(df))
        if len(df) > 0
        else 0.0
    )

    return {
        "check": "duplicate_rows",
        "status": (
            "warning" if duplicate_count > 0 else "passed"
        ),
        "row_count": int(len(df)),
        "subset": subset_columns,
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
        "issues": (
            ["duplicate_rows_detected"]
            if duplicate_count
            else []
        ),
        "note": "Detection only; no rows are removed.",
    }
