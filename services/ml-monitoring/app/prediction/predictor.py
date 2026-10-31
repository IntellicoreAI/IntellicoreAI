from pathlib import Path
import json

import pandas as pd
import xgboost as xgb


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = PROJECT_ROOT / "data" / "artifacts" / "xgboost" / "model.json"

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "xgboost"
    / "feature_names.json"
)


class XGBoostPredictor:
    """
    Prediction wrapper for the IntelliCore AI XGBoost model.

    Responsibilities:
    - Load the trained XGBoost model
    - Load the expected feature names
    - Validate incoming feature data
    - Generate prediction probability
    - Generate predicted class
    """

    def __init__(self):
        """
        Initialize the predictor by loading the model
        and feature configuration.
        """

        print("Initializing XGBoost Predictor...")

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at: {MODEL_PATH}"
            )

        self.model = xgb.XGBClassifier()

        self.model.load_model(str(MODEL_PATH))

        print("XGBoost model loaded successfully.")

        # ----------------------------------------------------
        # Load feature names
        # ----------------------------------------------------

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Feature names file not found at: {FEATURES_PATH}"
            )

        with open(FEATURES_PATH, "r") as f:
            self.feature_names = json.load(f)

        print(
            f"Loaded {len(self.feature_names)} expected features."
        )

        # ----------------------------------------------------
        # Display features
        # ----------------------------------------------------

        print("\nExpected features:")

        for feature in self.feature_names:
            print(f" - {feature}")

        print("\nPredictor initialization completed.")

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    def _validate_input(self, features):
        """
        Validate the input feature dictionary.

        Parameters
        ----------
        features : dict
            Dictionary containing feature names and values.

        Raises
        ------
        TypeError
            If input is not a dictionary.

        ValueError
            If required features are missing or unexpected
            features are supplied.
        """

        if not isinstance(features, dict):
            raise TypeError(
                "Input features must be provided as a dictionary."
            )

        provided_features = set(features.keys())
        expected_features = set(self.feature_names)

        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_features = expected_features - provided_features

        if missing_features:
            raise ValueError(
                "Missing required features: "
                + ", ".join(sorted(missing_features))
            )

        # ----------------------------------------------------
        # Check unexpected features
        # ----------------------------------------------------

        unexpected_features = provided_features - expected_features

        if unexpected_features:
            raise ValueError(
                "Unexpected features provided: "
                + ", ".join(sorted(unexpected_features))
            )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, features):
        """
        Generate a prediction for one observation.

        Parameters
        ----------
        features : dict
            Feature dictionary containing all model features.

        Returns
        -------
        dict
            Prediction result containing:

            prediction_probability
            predicted_class
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        self._validate_input(features)

        # ----------------------------------------------------
        # Convert dictionary to DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [[features[feature] for feature in self.feature_names]],
            columns=self.feature_names
        )

        # ----------------------------------------------------
        # Generate probability
        # ----------------------------------------------------

        probability = self.model.predict_proba(input_df)[0][1]

        # ----------------------------------------------------
        # Generate predicted class
        # ----------------------------------------------------

        predicted_class = int(probability >= 0.5)

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        result = {
            "prediction_probability": float(probability),
            "predicted_class": predicted_class
        }

        return result


# ============================================================
# TEST THE PREDICTOR
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INTELLICORE AI - XGBOOST PREDICTOR TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create predictor
    # --------------------------------------------------------

    predictor = XGBoostPredictor()

    # --------------------------------------------------------
    # Example input
    #
    # These values represent ONE processed observation.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    result = predictor.predict(example_features)

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nPrediction Result:")
    print(
        f"Prediction probability: "
        f"{result['prediction_probability']:.4f}"
    )

    print(
        f"Predicted class: "
        f"{result['predicted_class']}"
    )

    print("\n" + "=" * 60)
    print("PREDICTOR TEST COMPLETED")
    print("=" * 60)