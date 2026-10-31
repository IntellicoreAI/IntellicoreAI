
import json

import numpy as np
import pytest

from app.monitoring import prediction_drift


def write_config(path, significance_level=0.05, method="ks_2samp"):
    """Create a minimal prediction-drift configuration."""
    config = {
        "prediction_drift": {
            "method": method,
            "significance_level": significance_level,
        }
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_load_prediction_drift_config_reads_significance_level(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "monitoring_config.json"
    write_config(config_path, significance_level=0.01)

    monkeypatch.setattr(
        prediction_drift, "CONFIG_PATH", config_path
    )

    config = prediction_drift.load_prediction_drift_config()

    assert config["method"] == "ks_2samp"
    assert config["significance_level"] == 0.01


@pytest.mark.parametrize(
    "alpha",
    [0, 1, -0.1, 1.1, True, "0.05", None],
)
def test_invalid_significance_level_is_rejected(
    tmp_path, monkeypatch, alpha
):
    config_path = tmp_path / "monitoring_config.json"
    write_config(config_path, significance_level=alpha)

    monkeypatch.setattr(
        prediction_drift, "CONFIG_PATH", config_path
    )

    with pytest.raises(ValueError):
        prediction_drift.load_prediction_drift_config()


def test_unsupported_method_is_rejected(tmp_path, monkeypatch):
    config_path = tmp_path / "monitoring_config.json"
    write_config(config_path, method="unsupported_test")

    monkeypatch.setattr(
        prediction_drift, "CONFIG_PATH", config_path
    )

    with pytest.raises(ValueError, match="method"):
        prediction_drift.load_prediction_drift_config()


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.1, float("nan")],
        [0.1, float("inf")],
        [-0.1, 0.5],
        [0.5, 1.1],
    ],
)
def test_invalid_probabilities_are_rejected(values):
    with pytest.raises(ValueError):
        prediction_drift.validate_probabilities(
            values, "Test"
        )


def test_valid_probabilities_are_accepted():
    values = prediction_drift.validate_probabilities(
        [0.1, 0.5, 0.9], "Test"
    )

    assert np.array_equal(values, [0.1, 0.5, 0.9])


def test_prediction_drift_detects_distribution_shift(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "monitoring_config.json"
    report_path = tmp_path / "prediction_drift_report.json"

    write_config(config_path, significance_level=0.05)

    monkeypatch.setattr(
        prediction_drift, "CONFIG_PATH", config_path
    )
    monkeypatch.setattr(
        prediction_drift,
        "PREDICTION_DRIFT_REPORT_PATH",
        report_path,
    )

    reference = np.linspace(0.01, 0.20, 500)
    current = np.linspace(0.70, 0.95, 500)
    current_classes = (current >= 0.50).astype(int)

    report = prediction_drift.detect_prediction_drift(
        reference,
        current,
        current_classes,
    )

    assert report["comparison"]["drift_detected"] is True
    assert report["overall_status"] == "drift_detected"
    assert report["method"]["alpha"] == 0.05
    assert report["comparison"]["p_value"] < 0.05
    assert report_path.exists()


def test_prediction_drift_does_not_flag_identical_distributions(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "monitoring_config.json"
    report_path = tmp_path / "prediction_drift_report.json"

    write_config(config_path, significance_level=0.05)

    monkeypatch.setattr(
        prediction_drift, "CONFIG_PATH", config_path
    )
    monkeypatch.setattr(
        prediction_drift,
        "PREDICTION_DRIFT_REPORT_PATH",
        report_path,
    )

    reference = np.linspace(0.01, 0.40, 500)
    current = reference.copy()
    current_classes = (current >= 0.50).astype(int)

    report = prediction_drift.detect_prediction_drift(
        reference,
        current,
        current_classes,
    )

    assert report["comparison"]["drift_detected"] is False
    assert report["overall_status"] == "no_significant_drift"
    assert report["comparison"]["p_value"] >= 0.05


def test_mismatched_class_count_is_rejected(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "monitoring_config.json"
    write_config(config_path)

    monkeypatch.setattr(
        prediction_drift, "CONFIG_PATH", config_path
    )

    with pytest.raises(ValueError, match="same length"):
        prediction_drift.detect_prediction_drift(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.2]),
            np.array([0, 0, 0]),
        )
