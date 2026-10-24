
import numpy as np
import pandas as pd
import pytest

from app.anomaly.isolation_forest import IsolationForestAnomalyDetector


FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


@pytest.fixture
def reference_data():
    """Create reproducible numeric data for testing."""
    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, 200),
        "age": rng.integers(21, 75, 200),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.3, 200),
        "DebtRatio": rng.uniform(0, 1, 200),
        "MonthlyIncome": rng.lognormal(8.3, 0.5, 200),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(1, 20, 200),
        "NumberOfTimes90DaysLate": rng.poisson(0.1, 200),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, 200),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.2, 200),
        "NumberOfDependents": rng.integers(0, 5, 200),
    })

    return data[FEATURES]


@pytest.fixture
def fitted_detector(reference_data):
    """Return a detector fitted on reference data."""
    detector = IsolationForestAnomalyDetector(
        feature_names=FEATURES,
        contamination=0.05,
        random_state=42,
    )

    return detector.fit(reference_data)


def test_detector_rejects_empty_feature_list():
    with pytest.raises(ValueError, match="must not be empty"):
        IsolationForestAnomalyDetector(feature_names=[])


def test_detector_rejects_duplicate_feature_names():
    with pytest.raises(ValueError, match="duplicates"):
        IsolationForestAnomalyDetector(
            feature_names=["age", "age"]
        )


def test_detector_rejects_invalid_contamination():
    with pytest.raises(ValueError, match="contamination"):
        IsolationForestAnomalyDetector(
            feature_names=FEATURES,
            contamination=0,
        )


def test_detector_requires_fit(reference_data):
    detector = IsolationForestAnomalyDetector(
        feature_names=FEATURES
    )

    with pytest.raises(RuntimeError, match="fitted"):
        detector.detect(reference_data.head(1))


def test_detect_returns_expected_report(fitted_detector, reference_data):
    ids = ["event-001", "event-002", "event-003"]

    report = fitted_detector.detect(
        reference_data.head(3),
        record_ids=ids,
    )

    assert report["detector"] == "isolation_forest"
    assert report["contamination"] == 0.05
    assert report["total_records"] == 3
    assert 0 <= report["anomaly_count"] <= 3
    assert 0.0 <= report["anomaly_rate"] <= 1.0
    assert len(report["results"]) == 3

    assert [
        item["record_id"] for item in report["results"]
    ] == ids

    for item in report["results"]:
        assert isinstance(item["anomaly_score"], float)
        assert np.isfinite(item["anomaly_score"])
        assert isinstance(item["is_anomaly"], bool)


def test_detect_flags_extreme_injected_record(
    fitted_detector, reference_data
):
    """An intentionally extreme input should be flagged as anomalous."""
    normal = reference_data.head(20).copy()

    extreme = normal.iloc[[0]].copy()
    extreme.loc[:, "age"] = 150
    extreme.loc[:, "MonthlyIncome"] = 1_000_000_000
    extreme.loc[:, "DebtRatio"] = 100_000

    batch = pd.concat(
        [normal, extreme],
        ignore_index=True,
    )

    report = fitted_detector.detect(
        batch,
        record_ids=[
            f"event-{i}" for i in range(len(batch))
        ],
    )

    injected = report["results"][-1]

    assert injected["record_id"] == "event-20"
    assert injected["is_anomaly"] is True


def test_detect_rejects_missing_feature(
    fitted_detector, reference_data
):
    invalid = reference_data.head(2).drop(
        columns=["age"]
    )

    with pytest.raises(
        ValueError, match="Missing required features"
    ):
        fitted_detector.detect(invalid)


def test_detect_rejects_unexpected_feature(
    fitted_detector, reference_data
):
    invalid = reference_data.head(2).copy()
    invalid["unexpected_feature"] = 1

    with pytest.raises(
        ValueError, match="Unexpected features"
    ):
        fitted_detector.detect(invalid)


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_detect_rejects_non_finite_values(
    fitted_detector, reference_data, bad_value
):
    invalid = reference_data.head(2).copy()

    # Pandas requires a floating-point column to represent infinity.
    invalid["age"] = invalid["age"].astype(float)
    invalid.loc[invalid.index[0], "age"] = bad_value

    with pytest.raises(
        ValueError, match="NaN or infinity"
    ):
        fitted_detector.detect(invalid)


def test_detect_rejects_record_id_count_mismatch(
    fitted_detector, reference_data
):
    with pytest.raises(
        ValueError, match="record_ids length"
    ):
        fitted_detector.detect(
            reference_data.head(2),
            record_ids=["only-one"],
        )


def test_detect_rejects_empty_batch(fitted_detector):
    empty = pd.DataFrame(columns=FEATURES)

    with pytest.raises(
        ValueError, match="At least one record"
    ):
        fitted_detector.detect(empty)
