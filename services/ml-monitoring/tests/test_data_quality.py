
import pandas as pd
import pytest
from datetime import datetime, timezone

from app.data_quality.schema_check import (
    validate_schema,
    validate_numeric_features,
    validate_domain_rules,
)
from app.data_quality.missing_values import check_missing_values
from app.data_quality.duplicates import check_duplicates
from app.data_quality.freshness import check_freshness


# --------------------------------------------------
# Schema validation tests
# --------------------------------------------------

def test_schema_passes_with_expected_columns():
    df = pd.DataFrame({
        "age": [25, 30],
        "income": [5000, 6000],
    })

    result = validate_schema(df, ["age", "income"])

    assert result["status"] == "passed"
    assert result["missing_columns"] == []
    assert result["unexpected_columns"] == []


def test_schema_fails_when_required_column_is_missing():
    df = pd.DataFrame({"age": [25, 30]})

    result = validate_schema(df, ["age", "income"])

    assert result["status"] == "failed"
    assert result["missing_columns"] == ["income"]


def test_schema_can_allow_extra_columns():
    df = pd.DataFrame({
        "age": [25],
        "income": [5000],
        "extra": ["value"],
    })

    result = validate_schema(
        df,
        ["age", "income"],
        allow_extra_columns=True,
    )

    assert result["status"] == "passed"
    assert result["unexpected_columns"] == ["extra"]


# --------------------------------------------------
# Numeric validation tests
# --------------------------------------------------

def test_numeric_validation_detects_invalid_values():
    df = pd.DataFrame({
        "age": [25, "invalid", 40],
        "income": [5000, float("inf"), 7000],
    })

    result = validate_numeric_features(df, ["age", "income"])

    assert result["status"] == "failed"
    assert result["feature_results"]["age"]["non_numeric_count"] == 1
    assert result["feature_results"]["income"]["non_finite_count"] == 1


def test_numeric_validation_detects_missing_columns():
    df = pd.DataFrame({"age": [25, 30]})

    result = validate_numeric_features(df, ["age", "income"])

    assert result["status"] == "failed"
    assert result["missing_columns"] == ["income"]


def test_numeric_validation_allows_missing_values_for_separate_check():
    df = pd.DataFrame({
        "age": [25, None, 40],
    })

    result = validate_numeric_features(df, ["age"])

    assert result["status"] == "passed"
    assert result["feature_results"]["age"]["missing_count"] == 1


# --------------------------------------------------
# Domain validation tests
# --------------------------------------------------

def test_domain_validation_passes_for_valid_values():
    df = pd.DataFrame({
        "age": [25, 40],
        "MonthlyIncome": [30000, 0],
        "DebtRatio": [0.5, 1000],
        "RevolvingUtilizationOfUnsecuredLines": [0.8, 500],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 2],
        "NumberOfOpenCreditLinesAndLoans": [5, 10],
        "NumberOfTimes90DaysLate": [0, 1],
        "NumberRealEstateLoansOrLines": [1, 3],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 1],
        "NumberOfDependents": [0, 2],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "passed"
    assert result["issues"] == []


def test_domain_validation_rejects_fractional_credit_counts():
    df = pd.DataFrame({
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 1.5],
    })

    result = validate_domain_rules(
        df,
        ["NumberOfTime30-59DaysPastDueNotWorse"],
    )

    assert result["status"] == "failed"
    assert (
        result["feature_results"][
            "NumberOfTime30-59DaysPastDueNotWorse"
        ]["invalid_count"]
        == 1
    )


def test_domain_validation_rejects_negative_integer_counts():
    df = pd.DataFrame({
        "NumberOfDependents": [0, -1],
    })

    result = validate_domain_rules(
        df,
        ["NumberOfDependents"],
    )

    assert result["status"] == "failed"
    assert (
        result["feature_results"]["NumberOfDependents"][
            "invalid_count"
        ]
        == 1
    )


def test_domain_validation_accepts_non_negative_integer_counts():
    df = pd.DataFrame({
        "NumberOfDependents": [0, 1, 3],
    })

    result = validate_domain_rules(
        df,
        ["NumberOfDependents"],
    )

    assert result["status"] == "passed"
    assert (
        result["feature_results"]["NumberOfDependents"][
            "invalid_count"
        ]
        == 0
    )


def test_domain_validation_detects_negative_income():
    df = pd.DataFrame({
        "MonthlyIncome": [-100],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "failed"
    assert (
        "domain_rule_violation:MonthlyIncome"
        in result["issues"]
    )


def test_domain_validation_detects_zero_age():
    df = pd.DataFrame({
        "age": [0],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "failed"
    assert "domain_rule_violation:age" in result["issues"]


def test_domain_validation_detects_negative_credit_counts():
    df = pd.DataFrame({
        "NumberOfOpenCreditLinesAndLoans": [-1],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "failed"
    assert (
        "domain_rule_violation:NumberOfOpenCreditLinesAndLoans"
        in result["issues"]
    )


def test_domain_validation_detects_negative_debt_ratio():
    df = pd.DataFrame({
        "DebtRatio": [-0.1],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "failed"
    assert "domain_rule_violation:DebtRatio" in result["issues"]


def test_domain_validation_ignores_missing_values():
    df = pd.DataFrame({
        "age": [25, None],
        "MonthlyIncome": [30000, None],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "passed"


def test_domain_validation_allows_legitimate_high_values():
    df = pd.DataFrame({
        "age": [25],
        "MonthlyIncome": [3008750],
        "DebtRatio": [329664],
        "RevolvingUtilizationOfUnsecuredLines": [50708],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "passed"


# --------------------------------------------------
# Missing-value tests
# --------------------------------------------------

def test_missing_values_pass_without_threshold_violations():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [5000, 6000, 7000],
    })

    result = check_missing_values(
        df,
        max_missing_rate_by_column={"age": 0.1},
    )

    assert result["status"] == "passed"
    assert result["columns"]["age"]["missing_count"] == 0


def test_missing_values_fail_when_threshold_is_exceeded():
    df = pd.DataFrame({
        "age": [25, None, 40],
    })

    result = check_missing_values(
        df,
        max_missing_rate_by_column={"age": 0.2},
    )

    assert result["status"] == "failed"
    assert result["columns"]["age"]["missing_count"] == 1
    assert "age" in result["threshold_violations"]
def test_strict_policy_passes_when_no_values_are_missing():
    df = pd.DataFrame({
        "age": [25, 30],
        "MonthlyIncome": [5000, 6000],
    })

    result = check_missing_values(
        df,
        max_missing_rate_by_column={
            "age": 0.0,
            "MonthlyIncome": 0.0,
        },
    )

    assert result["status"] == "passed"
    assert result["threshold_violations"] == []


def test_strict_policy_fails_when_one_value_is_missing():
    df = pd.DataFrame({
        "age": [25, None],
        "MonthlyIncome": [5000, 6000],
    })

    result = check_missing_values(
        df,
        max_missing_rate_by_column={
            "age": 0.0,
            "MonthlyIncome": 0.0,
        },
    )

    assert result["status"] == "failed"
    assert result["columns"]["age"]["missing_count"] == 1
    assert result["columns"]["age"]["missing_rate"] == 0.5
    assert "age" in result["threshold_violations"]


def test_strict_policy_fails_when_multiple_features_have_missing_values():
    df = pd.DataFrame({
        "age": [25, None],
        "MonthlyIncome": [None, 6000],
    })

    result = check_missing_values(
        df,
        max_missing_rate_by_column={
            "age": 0.0,
            "MonthlyIncome": 0.0,
        },
    )

    assert result["status"] == "failed"
    assert result["columns"]["age"]["missing_count"] == 1
    assert result["columns"]["MonthlyIncome"]["missing_count"] == 1
    assert set(result["threshold_violations"]) == {
        "age",
        "MonthlyIncome",
    }

def test_missing_values_reject_invalid_threshold():
    df = pd.DataFrame({"age": [25, 30]})

    with pytest.raises(ValueError):
        check_missing_values(
            df,
            max_missing_rate_by_column={"age": 1.5},
        )


# --------------------------------------------------
# Duplicate-row tests
# --------------------------------------------------

def test_duplicates_are_detected_by_subset():
    df = pd.DataFrame({
        "event_id": [1, 2, 2],
        "value": [5, 6, 99],
    })

    result = check_duplicates(df, subset=["event_id"])

    assert result["status"] == "warning"
    assert result["duplicate_count"] == 1
    assert result["duplicate_rate"] == pytest.approx(1 / 3)


def test_duplicates_fail_when_subset_column_is_missing():
    df = pd.DataFrame({"age": [20]})

    result = check_duplicates(df, subset=["event_id"])

    assert result["status"] == "failed"
    assert result["missing_subset_columns"] == ["event_id"]


def test_duplicate_check_does_not_modify_dataframe():
    df = pd.DataFrame({
        "event_id": [1, 1],
        "value": [5, 5],
    })
    original = df.copy(deep=True)

    check_duplicates(df)

    pd.testing.assert_frame_equal(df, original)


# --------------------------------------------------
# Freshness tests
# --------------------------------------------------

REFERENCE_TIME = datetime(
    2026, 10, 13, 11, 0, tzinfo=timezone.utc
)


def test_fresh_data_passes():
    df = pd.DataFrame({
        "event_time": [
            "2026-10-13T10:55:00Z",
            "2026-10-13T10:58:00Z",
        ]
    })

    result = check_freshness(
        df,
        timestamp_column="event_time",
        max_age="10min",
        reference_time=REFERENCE_TIME,
    )

    assert result["status"] == "passed"
    assert result["age_seconds"] == 120


def test_stale_data_fails():
    df = pd.DataFrame({
        "event_time": ["2026-10-13T10:30:00Z"]
    })

    result = check_freshness(
        df,
        timestamp_column="event_time",
        max_age="10min",
        reference_time=REFERENCE_TIME,
    )

    assert result["status"] == "failed"
    assert "data_stale" in result["issues"]


def test_freshness_fails_when_timestamp_column_is_missing():
    df = pd.DataFrame({"age": [25]})

    result = check_freshness(
        df,
        timestamp_column="event_time",
        max_age="10min",
        reference_time=REFERENCE_TIME,
    )

    assert result["status"] == "failed"
    assert "timestamp_column_missing" in result["issues"]


def test_freshness_fails_when_no_valid_timestamps_exist():
    df = pd.DataFrame({
        "event_time": ["not-a-timestamp", None]
    })

    result = check_freshness(
        df,
        timestamp_column="event_time",
        max_age="10min",
        reference_time=REFERENCE_TIME,
    )

    assert result["status"] == "failed"
    assert result["valid_timestamp_count"] == 0
    assert "no_valid_timestamps" in result["issues"]
