from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import sys

# Add the ML monitoring app directory to Python path
APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(APP_ROOT))

from prediction.predictor_02 import XGBoostPredictor


class TelemetryGenerator:
    """
    Generates structured telemetry events for ML predictions.

    The generated telemetry contains:
    - event metadata
    - model information
    - input features
    - prediction results
    - local SHAP explanation
    """

    def __init__(self):
        print("Initializing Telemetry Generator...")

        self.predictor = XGBoostPredictor()

        print("Telemetry Generator initialized successfully.")

    def generate_event(self, features):
        """
        Generate one telemetry event for a model prediction.
        """

        # Generate prediction + local SHAP explanation
        prediction_result = self.predictor.predict(features)

        # Create unique event ID
        event_id = str(uuid.uuid4())

        # Generate UTC timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        telemetry_event = {
            "event_id": event_id,
            "timestamp": timestamp,

            "model": {
                "model_name": "xgboost_credit_risk",
                "model_version": "1.0"
            },

            "input": {
                "features": features
            },

            "prediction": {
                "prediction_probability": prediction_result[
                    "prediction_probability"
                ],
                "predicted_class": prediction_result[
                    "predicted_class"
                ]
            },

            "explanation": {
                "method": "SHAP",
                "feature_contributions": prediction_result[
                    "explanation"
                ]
            }
        }

        return telemetry_event


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INTELLICORE AI - TELEMETRY GENERATOR TEST")
    print("=" * 60)

    generator = TelemetryGenerator()

    # Example prediction input
    example_features = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.4,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    }

    telemetry = generator.generate_event(example_features)

    print("\nTelemetry Event:")
    print(json.dumps(telemetry, indent=4))

    print("\n" + "=" * 60)
    print("TELEMETRY GENERATION TEST COMPLETED")
    print("=" * 60)
