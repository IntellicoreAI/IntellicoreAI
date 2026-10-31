"""
IntelliCore AI - SHAP Explainability

Purpose:
    Generate global explainability artifacts for the final XGBoost model.

This script:
    1. Loads the trained XGBoost model.
    2. Loads the processed test dataset.
    3. Loads the saved feature names.
    4. Computes SHAP values on a representative sample.
    5. Calculates global feature importance.
    6. Generates SHAP bar and beeswarm plots.
    7. Saves artifacts for IntelliCore.

Important:
    Global SHAP explains the overall behavior of the model.
    Local SHAP explanations for individual predictions will be
    implemented later as part of prediction telemetry.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# File location:
# services/ml-monitoring/app/explainability/shap_explainer.py
#
# parents[4] points to the IntelliCoreAI project root.

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "xgboost"
    / "model.json"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "xgboost"
    / "feature_names.json"
)

SHAP_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "xgboost"
    / "shap"
)

FEATURE_IMPORTANCE_PATH = (
    SHAP_ARTIFACT_DIR
    / "feature_importance.json"
)

SHAP_VALUES_PATH = (
    SHAP_ARTIFACT_DIR
    / "sample_shap_values.csv"
)

SUMMARY_BAR_PATH = (
    SHAP_ARTIFACT_DIR
    / "summary_bar.png"
)

SUMMARY_BEESWARM_PATH = (
    SHAP_ARTIFACT_DIR
    / "summary_beeswarm.png"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Number of test records used for SHAP analysis.
#
# We don't need to calculate SHAP values for every test record
# during this development/research stage.
SHAP_SAMPLE_SIZE = 5000

RANDOM_STATE = 42


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained XGBoost model from model.json.
    """

    print("\nLoading XGBoost model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = XGBClassifier()

    model.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# ============================================================
# 4. LOAD FEATURE NAMES
# ============================================================

def load_feature_names():
    """
    Load the feature names saved during final model training.
    """

    print("\nLoading feature names...")

    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature names file not found:\n"
            f"{FEATURE_NAMES_PATH}"
        )

    with open(FEATURE_NAMES_PATH, "r") as file:
        feature_names = json.load(file)

    print(f"Number of features: {len(feature_names)}")

    for feature in feature_names:
        print(f" - {feature}")

    return feature_names


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

def load_test_data(feature_names):
    """
    Load the processed test dataset and select only the
    model's input features.
    """

    print("\nLoading processed test data...")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_PATH}"
        )

    test_df = pd.read_csv(TEST_PATH)

    print(f"Full test dataset shape: {test_df.shape}")

    # Select only model input features.
    X_test = test_df[feature_names].copy()

    print(f"Feature matrix shape: {X_test.shape}")

    return X_test


# ============================================================
# 6. CREATE SHAP SAMPLE
# ============================================================

def create_sample(X_test):
    """
    Create a reproducible sample of the test dataset.

    This keeps SHAP computation manageable while still giving
    us a representative global explanation.
    """

    if len(X_test) <= SHAP_SAMPLE_SIZE:
        X_sample = X_test.copy()
    else:
        X_sample = X_test.sample(
            n=SHAP_SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    print(
        f"\nSHAP sample size: {len(X_sample)}"
    )

    return X_sample


# ============================================================
# 7. CALCULATE SHAP VALUES
# ============================================================

def calculate_shap_values(model, X_sample):
    """
    Calculate SHAP values using TreeExplainer.

    TreeExplainer is designed for tree-based models such as
    XGBoost.
    """

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    shap_output = explainer(X_sample)

    # SHAP 0.51 returns an Explanation object.
    #
    # For binary classification, the values normally have
    # shape:
    #
    #     samples × features
    #
    shap_values = shap_output.values

    # Some SHAP/model combinations can return an additional
    # output dimension. Handle that safely.
    if shap_values.ndim == 3:
        print(
            "SHAP output has 3 dimensions. "
            "Selecting the positive-class explanation."
        )

        shap_values = shap_values[:, :, 1]

    print(
        f"SHAP values shape: {shap_values.shape}"
    )

    return shap_values


# ============================================================
# 8. GLOBAL FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    shap_values,
    feature_names
):
    """
    Calculate mean absolute SHAP value for each feature.

    Larger values mean the feature has a larger average impact
    on the model's predictions.
    """

    print("\nCalculating global feature importance...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    importance_df = importance_df.sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    print("\nGlobal Feature Importance:")
    print(importance_df.to_string(index=False))

    return importance_df


# ============================================================
# 9. SAVE FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(importance_df):
    """
    Save ranked feature importance as JSON.
    """

    importance_dict = dict(
        zip(
            importance_df["feature"],
            importance_df["mean_abs_shap"]
        )
    )

    importance_dict = {
        feature: float(value)
        for feature, value in importance_dict.items()
    }

    with open(
        FEATURE_IMPORTANCE_PATH,
        "w"
    ) as file:
        json.dump(
            importance_dict,
            file,
            indent=4
        )

    print(
        f"\nSaved feature importance:\n"
        f"{FEATURE_IMPORTANCE_PATH}"
    )


# ============================================================
# 10. SAVE RAW SAMPLE SHAP VALUES
# ============================================================

def save_shap_values(
    shap_values,
    X_sample
):
    """
    Save SHAP values for the sampled test records.

    Each row represents one prediction instance.
    Each column represents a feature's SHAP contribution.
    """

    shap_df = pd.DataFrame(
        shap_values,
        columns=X_sample.columns,
        index=X_sample.index
    )

    shap_df.insert(
        0,
        "row_index",
        X_sample.index
    )

    shap_df.to_csv(
        SHAP_VALUES_PATH,
        index=False
    )

    print(
        f"Saved sample SHAP values:\n"
        f"{SHAP_VALUES_PATH}"
    )


# ============================================================
# 11. CREATE BAR PLOT
# ============================================================

def create_bar_plot(
    shap_values,
    X_sample
):
    """
    Generate a global SHAP bar plot.

    This answers:

        Which features have the largest average impact?
    """

    print("\nGenerating SHAP bar plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        SUMMARY_BAR_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{SUMMARY_BAR_PATH}"
    )


# ============================================================
# 12. CREATE BEESWARM PLOT
# ============================================================

def create_beeswarm_plot(
    shap_values,
    X_sample
):
    """
    Generate a SHAP beeswarm plot.

    This shows:

        - feature importance
        - direction of influence
        - distribution of SHAP values
    """

    print("\nGenerating SHAP beeswarm plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        SUMMARY_BEESWARM_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{SUMMARY_BEESWARM_PATH}"
    )


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("INTELLICORE AI - SHAP EXPLAINABILITY")
    print("=" * 60)

    # Create output directory.
    SHAP_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load model.
    model = load_model()

    # Load feature names.
    feature_names = load_feature_names()

    # Load test data.
    X_test = load_test_data(feature_names)

    # Create reproducible SHAP sample.
    X_sample = create_sample(X_test)

    # Calculate SHAP values.
    shap_values = calculate_shap_values(
        model,
        X_sample
    )

    # Calculate global importance.
    importance_df = calculate_feature_importance(
        shap_values,
        feature_names
    )

    # Save artifacts.
    save_feature_importance(
        importance_df
    )

    save_shap_values(
        shap_values,
        X_sample
    )

    create_bar_plot(
        shap_values,
        X_sample
    )

    create_beeswarm_plot(
        shap_values,
        X_sample
    )

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETED")
    print("=" * 60)

    print("\nArtifacts generated:")

    print(
        f"1. {FEATURE_IMPORTANCE_PATH}"
    )

    print(
        f"2. {SHAP_VALUES_PATH}"
    )

    print(
        f"3. {SUMMARY_BAR_PATH}"
    )

    print(
        f"4. {SUMMARY_BEESWARM_PATH}"
    )

    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()