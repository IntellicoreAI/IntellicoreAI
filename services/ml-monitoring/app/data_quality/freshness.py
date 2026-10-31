
"""Freshness checks for incoming ML monitoring data."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def check_freshness(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    max_age: str | pd.Timedelta,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Check whether the latest valid timestamp is sufficiently recent.

    Args:
        df: Incoming monitoring records.
        timestamp_column: Column containing event timestamps.
        max_age: Maximum permitted age, e.g. "15min" or "2h".
        reference_time: Time against which freshness is evaluated.
            Defaults to the current UTC time.

    Returns:
        A structured freshness report. This function does not
        modify the input DataFrame.
    """

    if timestamp_column not in df.columns:
        return {
            "check": "freshness",
            "status": "failed",
            "row_count": int(len(df)),
            "timestamp_column": timestamp_column,
            "issues": ["timestamp_column_missing"],
        }

    try:
        allowed_age = pd.Timedelta(max_age)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_age must be a valid duration, such as '15min' or '2h'."
        ) from exc

    if pd.isna(allowed_age) or allowed_age < pd.Timedelta(0):
        raise ValueError("max_age must be a non-negative duration.")

    now = pd.Timestamp(
        reference_time
        if reference_time is not None
        else datetime.now(timezone.utc)
    )

    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True,
    )

    missing_count = int(df[timestamp_column].isna().sum())
    invalid_count = int(
        (df[timestamp_column].notna() & timestamps.isna()).sum()
    )

    valid_timestamps = timestamps.dropna()

    if valid_timestamps.empty:
        issues = ["no_valid_timestamps"]

        if missing_count:
            issues.append("timestamps_missing")
        if invalid_count:
            issues.append("timestamps_unparseable")

        return {
            "check": "freshness",
            "status": "failed",
            "row_count": int(len(df)),
            "timestamp_column": timestamp_column,
            "valid_timestamp_count": 0,
            "missing_timestamp_count": missing_count,
            "invalid_timestamp_count": invalid_count,
            "latest_timestamp": None,
            "age_seconds": None,
            "max_age_seconds": allowed_age.total_seconds(),
            "issues": issues,
        }

    latest_timestamp = valid_timestamps.max()
    age = now - latest_timestamp

    issues = []

    if age > allowed_age:
        issues.append("data_stale")

    if latest_timestamp > now:
        issues.append("future_timestamp_detected")

    if missing_count:
        issues.append("timestamps_missing")

    if invalid_count:
        issues.append("timestamps_unparseable")

    return {
        "check": "freshness",
        "status": "failed" if issues else "passed",
        "row_count": int(len(df)),
        "timestamp_column": timestamp_column,
        "valid_timestamp_count": int(len(valid_timestamps)),
        "missing_timestamp_count": missing_count,
        "invalid_timestamp_count": invalid_count,
        "latest_timestamp": latest_timestamp.isoformat(),
        "age_seconds": age.total_seconds(),
        "max_age_seconds": allowed_age.total_seconds(),
        "issues": issues,
    }
