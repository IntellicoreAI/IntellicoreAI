from datetime import datetime, timezone

from app.schemas.prediction import PredictionEvent


def test_valid_prediction_event():
    event = PredictionEvent(
        project_id="proj_123",
        model_id="fraud_model",
        model_version="v1.0",
        environment="production",
        timestamp=datetime.now(timezone.utc),
        features={
            "age": 42,
            "income": 75000,
            "credit_score": 712,
        },
        prediction=0,
        confidence=0.91,
        actual_label=0,
        latency_ms=132,
    )

    assert event.model_id == "fraud_model"
    assert event.confidence == 0.91