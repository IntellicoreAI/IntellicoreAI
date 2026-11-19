
import uuid
from datetime import datetime

import pytest

from app.adapters.ml_adapter import create_ml_event


@pytest.fixture
def sample_model():
    return {
        "model_name": "xgboost_credit_risk",
        "model_version": "1.0",
    }


@pytest.fixture
def sample_report():
    return {
        "overall_status": "drift_detected",
        "drifted_features": ["age", "income"],
    }


def test_create_monitoring_event(sample_model, sample_report):
    event = create_ml_event(
        event_type="monitoring_result",
        signal_type="data_drift",
        report=sample_report,
        model=sample_model,
        project_id="intellicore-credit-risk",
        environment="development",
    )

    assert event["schema_version"] == "1.0"
    assert event["event_type"] == "monitoring_result"
    assert event["project_id"] == "intellicore-credit-risk"
    assert event["environment"] == "development"
    assert event["data"]["signal_type"] == "data_drift"
    assert event["data"]["report"] == sample_report
    assert "alert" not in event["data"]

    uuid.UUID(event["event_id"])
    datetime.fromisoformat(event["occurred_at"])


def test_create_alert_event(sample_model, sample_report):
    event = create_ml_event(
        event_type="alert",
        signal_type="unified_alert",
        report=sample_report,
        model=sample_model,
        project_id="intellicore-credit-risk",
        environment="development",
    )

    assert event["event_type"] == "alert"
    assert event["data"]["signal_type"] == "unified_alert"
    assert event["data"]["alert"] == sample_report
    assert "report" not in event["data"]


def test_invalid_event_type_fails(sample_model, sample_report):
    with pytest.raises(ValueError):
        create_ml_event(
            event_type="prediction_invalid",
            signal_type="data_drift",
            report=sample_report,
            model=sample_model,
            project_id="intellicore-credit-risk",
            environment="development",
        )


def test_invalid_environment_fails(sample_model, sample_report):
    with pytest.raises(ValueError):
        create_ml_event(
            event_type="monitoring_result",
            signal_type="data_drift",
            report=sample_report,
            model=sample_model,
            project_id="intellicore-credit-risk",
            environment="testing",
        )


def test_empty_project_id_fails(sample_model, sample_report):
    with pytest.raises(ValueError):
        create_ml_event(
            event_type="monitoring_result",
            signal_type="data_drift",
            report=sample_report,
            model=sample_model,
            project_id="",
            environment="development",
        )


def test_invalid_model_fails(sample_report):
    with pytest.raises(ValueError):
        create_ml_event(
            event_type="monitoring_result",
            signal_type="data_drift",
            report=sample_report,
            model={"model_name": "xgboost_credit_risk"},
            project_id="intellicore-credit-risk",
            environment="development",
        )
