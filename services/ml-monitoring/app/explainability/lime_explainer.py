
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "artifacts" / "xgboost" / "model.json"
FEATURE_NAMES_PATH = (
    PROJECT_ROOT / "data" / "artifacts" / "xgboost" / "feature_names.json"
)
LIME_ARTIFACT_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "xgboost" / "lime"
)


def validate_instance(
    instance: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """Validate one record and return it as a one-row DataFrame."""
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary.")

    if not feature_names:
        raise ValueError("feature_names must not be empty.")

    missing = set(feature_names) - set(instance)
    unexpected = set(instance) - set(feature_names)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if unexpected:
        raise ValueError(f"Unexpected features: {sorted(unexpected)}")

    values = []

    for feature in feature_names:
        value = instance[feature]

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Feature '{feature}' must be numeric, not boolean.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{feature}' must be numeric.") from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"Feature '{feature}' must be finite.")

        values.append(numeric_value)

    return pd.DataFrame([values], columns=feature_names)


def explain_instance(
    model: XGBClassifier,
    training_data: pd.DataFrame,
    instance: dict[str, Any],
    feature_names: list[str],
    num_features: int = 10,
    num_samples: int = 5000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Generate a local LIME explanation for one model prediction."""
    if num_features < 1:
        raise ValueError("num_features must be at least 1.")

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    if training_data.empty:
        raise ValueError("training_data must not be empty.")

    if list(training_data.columns) != feature_names:
        raise ValueError("training_data columns must match feature_names exactly.")

    training_values = training_data.to_numpy(dtype=float)

    if not np.isfinite(training_values).all():
        raise ValueError("training_data must contain only finite numeric values.")

    row = validate_instance(instance, feature_names)
    instance_values = row.to_numpy(dtype=float)[0]

    explainer = LimeTabularExplainer(
        training_data=training_values,
        feature_names=feature_names,
        class_names=["No serious delinquency", "Serious delinquency"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    explanation = explainer.explain_instance(
        data_row=instance_values,
        predict_fn=model.predict_proba,
        labels=(1,),
        num_features=min(num_features, len(feature_names)),
        num_samples=num_samples,
    )

    probabilities = model.predict_proba(row)[0]
    local_contributions = explanation.as_list(label=1)

    return {
        "method": "LIME",
        "prediction": int(np.argmax(probabilities)),
        "class_probabilities": {
            str(index): float(probability)
            for index, probability in enumerate(probabilities)
        },
        "explained_class": 1,
        "local_explanation": [
            {
                "condition": str(condition),
                "contribution": float(contribution),
            }
            for condition, contribution in local_contributions
        ],
        "local_prediction": float(explanation.local_pred[0]),
        "score": float(explanation.score),
        "interpretation_note": (
            "LIME approximates the model locally around this instance. "
            "Contributions describe the local surrogate, not causal effects."
        ),
    }


def save_explanation(
    result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a LIME result as JSON without modifying the source model."""
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def load_model() -> XGBClassifier:
    """Load the trained XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def load_feature_names() -> list[str]:
    """Load the model's saved input feature names."""
    import json

    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(f"Feature names not found: {FEATURE_NAMES_PATH}")

    names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))

    if not isinstance(names, list) or not names:
        raise ValueError("Saved feature names must be a non-empty list.")

    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Every feature name must be a non-empty string.")

    if len(names) != len(set(names)):
        raise ValueError("Saved feature names must be unique.")

    return names


def main() -> None:
    """Generate one example LIME explanation from the processed test data."""
    import json

    model = load_model()
    feature_names = load_feature_names()

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    test_data = pd.read_csv(TEST_PATH)

    missing = set(feature_names) - set(test_data.columns)
    if missing:
        raise ValueError(f"Test data is missing features: {sorted(missing)}")

    training_data = test_data[feature_names].copy()
    instance = training_data.iloc[0].to_dict()

    result = explain_instance(
        model=model,
        training_data=training_data,
        instance=instance,
        feature_names=feature_names,
    )

    output_path = LIME_ARTIFACT_DIR / "sample_explanation.json"
    save_explanation(result, output_path)

    print(json.dumps(result, indent=2))
    print(f"\nSaved LIME explanation to: {output_path}")


if __name__ == "__main__":
    main()
