
import pytest
import pandas as pd

from app.monitoring import drift_detector


def test_load_monitoring_config_reads_significance_level():
    config = drift_detector.load_monitoring_config()

    assert config["data_drift"]["significance_level"] == 0.05


def test_calculate_ks_drift_detects_distribution_shift():
    reference = pd.Series([0.0] * 100)
    current = pd.Series([10.0] * 100)

    result = drift_detector.calculate_ks_drift(
        reference,
        current,
    )

    assert result["drift_detected"] is True
    assert result["status"] == "drift_detected"
    assert result["p_value"] < 0.05


def test_calculate_ks_drift_detects_no_significant_shift():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    current = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = drift_detector.calculate_ks_drift(
        reference,
        current,
    )

    assert result["drift_detected"] is False
    assert result["status"] == "no_significant_drift"


def test_calculate_ks_drift_handles_empty_data():
    reference = pd.Series([], dtype=float)
    current = pd.Series([1.0, 2.0, 3.0])

    result = drift_detector.calculate_ks_drift(
        reference,
        current,
    )

    assert result["status"] == "insufficient_data"
    assert result["ks_statistic"] is None
    assert result["p_value"] is None


def test_invalid_significance_level_is_rejected(monkeypatch):
    monkeypatch.setattr(
        drift_detector,
        "load_monitoring_config",
        lambda: {"data_drift": {"significance_level": 1.0}},
    )

    with pytest.raises(ValueError):
        drift_detector.get_significance_level()
