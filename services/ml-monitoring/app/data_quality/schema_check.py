
"""Schema, numeric-value, and domain-rule validation for ML monitoring."""

from typing import Iterable, Any

import numpy as np
import pandas as pd


def validate_schema(
    df: pd.DataFrame,
    expected_columns: Iterable[str],
    *,
    allow_extra_columns: bool = False,
) -> dict[str, Any]:
    """Validate required columns and duplicate column names."""

    expected = list(expected_columns)
    actual = list(df.columns)

    missing = [col for col in expected if col not in actual]
    unexpected = [col for col in actual if col not in expected]
    duplicate_names = df.columns[
        df.columns.duplicated()
    ].tolist()

    issues = []

    if missing:
        issues.append("missing_required_columns")

    if unexpected and not allow_extra_columns:
        issues.append("unexpected_columns")

    if duplicate_names:
        issues.append("duplicate_column_names")

    passed = (
        not missing
        and (allow_extra_columns or not unexpected)
        and not duplicate_names
    )

    return {
        "check": "schema_validation",
        "status": "passed" if passed else "failed",
        "row_count": int(len(df)),
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "duplicate_column_names": duplicate_names,
        "issues": issues,
    }


def validate_numeric_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> dict[str, Any]:
    """Detect non-numeric and infinite feature values."""

    results = {}
    missing_columns = []
    issues = []

    for column in feature_columns:
        if column not in df.columns:
            missing_columns.append(column)
            continue

        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")

        non_numeric = series.notna() & numeric.isna()

        values = numeric.to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        non_finite = pd.Series(
            np.isinf(values),
            index=series.index,
        ) & numeric.notna()

        results[column] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "non_numeric_count": int(non_numeric.sum()),
            "non_finite_count": int(non_finite.sum()),
        }

        if non_numeric.any():
            issues.append(f"non_numeric_values:{column}")

        if non_finite.any():
            issues.append(f"non_finite_values:{column}")

    if missing_columns:
        issues.append("missing_required_columns")

    return {
        "check": "numeric_validation",
        "status": "passed" if not issues else "failed",
        "feature_results": results,
        "missing_columns": missing_columns,
        "issues": issues,
    }



def validate_domain_rules(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> dict[str, Any]:
    """
    Validate domain constraints for credit-risk features.

    Missing values are handled by the separate missing-value check.
    Non-numeric and non-finite values are handled by numeric validation.
    No arbitrary upper bounds are applied.
    """

    non_negative_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
    ]

    non_negative_integer_features = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]

    strictly_positive_features = ["age"]

    available_features = set(feature_columns)

    rules = {
        **{
            column: "non_negative"
            for column in non_negative_features
            if column in available_features
        },
        **{
            column: "non_negative_integer"
            for column in non_negative_integer_features
            if column in available_features
        },
        **{
            column: "strictly_positive"
            for column in strictly_positive_features
            if column in available_features
        },
    }

    feature_results = {}
    missing_columns = []
    issues = []

    for column, rule in rules.items():
        if column not in df.columns:
            missing_columns.append(column)
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")

        values = numeric.to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        valid_numeric = numeric.notna() & np.isfinite(values)

        if rule == "non_negative":
            invalid = valid_numeric & (numeric < 0)
        elif rule == "strictly_positive":
            invalid = valid_numeric & (numeric <= 0)
        else:
            invalid = valid_numeric & (
                (numeric < 0) | (numeric % 1 != 0)
            )

        invalid_count = int(invalid.sum())

        feature_results[column] = {
            "rule": rule,
            "invalid_count": invalid_count,
        }

        if invalid_count:
            issues.append(f"domain_rule_violation:{column}")

    if missing_columns:
        issues.append("missing_required_columns")

    return {
        "check": "domain_validation",
        "status": "passed" if not issues else "failed",
        "feature_results": feature_results,
        "missing_columns": missing_columns,
        "issues": issues,
    }
