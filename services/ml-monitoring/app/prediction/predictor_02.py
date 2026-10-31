from pathlib import Path
import json

import pandas as pd
import shap
import xgboost as xgb


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "xgboost"
    / "model.json"
)

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
    - Load expected feature names
    - Validate input features
    - Generate prediction probability
    - Generate predicted class
    - Generate local SHAP explanations
    """

    def __init__(self):
        """
        Initialize the predictor.
        """

        print("Initializing XGBoost Predictor...")

        # ----------------------------------------------------
        # Load XGBoost model
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

        # ----------------------------------------------------
        # Create SHAP explainer
        # ----------------------------------------------------

        print("\nCreating SHAP TreeExplainer...")

        self.explainer = shap.TreeExplainer(self.model)

        print("SHAP TreeExplainer created successfully.")

        print("\nPredictor initialization completed.")

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    def _validate_input(self, features):
        """
        Validate incoming feature dictionary.
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
    # GENERATE LOCAL SHAP EXPLANATION
    # ========================================================

    def _generate_local_shap(self, input_df):
        """
        Generate SHAP explanation for one prediction.

        Returns
        -------
        list
            List of feature-level SHAP explanations.
        """

        # ----------------------------------------------------
        # Calculate SHAP values
        # ----------------------------------------------------

        shap_output = self.explainer(input_df)

        shap_values = shap_output.values

        # ----------------------------------------------------
        # Handle possible multi-class / 3D SHAP output
        # ----------------------------------------------------

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        # We have one observation, so take the first row
        shap_values = shap_values[0]

        # ----------------------------------------------------
        # Create explanation records
        # ----------------------------------------------------

        explanation = []

        for feature, shap_value in zip(
            self.feature_names,
            shap_values
        ):

            shap_value = float(shap_value)

            # Positive contribution → prediction moves upward
            if shap_value > 0:
                direction = "up"

            elif shap_value < 0:
                direction = "down"

            else:
                direction = "neutral"

            explanation.append(
                {
                    "feature": feature,
                    "shap_value": shap_value,
                    "direction": direction
                }
            )

        # ----------------------------------------------------
        # Sort by absolute contribution
        # ----------------------------------------------------

        explanation.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        return explanation

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, features):
        """
        Generate prediction and local SHAP explanation.

        Parameters
        ----------
        features : dict
            Dictionary containing model features.

        Returns
        -------
        dict
            Prediction result with local SHAP explanation.
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
        # Generate prediction probability
        # ----------------------------------------------------

        probability = self.model.predict_proba(input_df)[0][1]

        # ----------------------------------------------------
        # Generate predicted class
        # ----------------------------------------------------

        predicted_class = int(probability >= 0.5)

        # ----------------------------------------------------
        # Generate local SHAP explanation
        # ----------------------------------------------------

        explanation = self._generate_local_shap(input_df)

        # ----------------------------------------------------
        # Create result
        # ----------------------------------------------------

        result = {
            "prediction_probability": float(probability),
            "predicted_class": predicted_class,
            "explanation": explanation
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
    # Display prediction
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

    # --------------------------------------------------------
    # Display local SHAP explanation
    # --------------------------------------------------------

    print("\nLocal SHAP Explanation:")

    for item in result["explanation"]:

        print(
            f"{item['feature']:<50} "
            f"{item['shap_value']:>10.6f} "
            f"({item['direction']})"
        )

    print("\n" + "=" * 60)
    print("PREDICTOR + LOCAL SHAP TEST COMPLETED")
    print("=" * 60)