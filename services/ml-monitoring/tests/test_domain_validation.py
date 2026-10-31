
import pandas as pd

from app.data_quality.schema_check import validate_domain_rules


def test_valid_credit_risk_values_pass():
    df = pd.DataFrame({
        "age": [25, 40],
        "MonthlyIncome": [30000, 0],
        "DebtRatio": [0.5, 329664],
        "RevolvingUtilizationOfUnsecuredLines": [0.8, 50708],
    })

    result = validate_domain_rules(
        df,
        df.columns,
    )

    assert result["status"] == "passed"


def test_negative_values_fail():
    df = pd.DataFrame({
        "age": [25, 40],
        "MonthlyIncome": [30000, -100],
        "DebtRatio": [0.5, 0.2],
        "RevolvingUtilizationOfUnsecuredLines": [0.8, 0.4],
    })

    result = validate_domain_rules(
        df,
        df.columns,
    )

    assert result["status"] == "failed"
    assert "domain_rule_violation:MonthlyIncome" in result["issues"]


def test_zero_age_fails():
    df = pd.DataFrame({
        "age": [0],
    })

    result = validate_domain_rules(df, df.columns)

    assert result["status"] == "failed"
    assert "domain_rule_violation:age" in result["issues"]
