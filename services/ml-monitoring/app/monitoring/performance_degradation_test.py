"""
Controlled Performance Degradation Test
-----------------------------------------

Creates a controlled experiment to validate whether
IntelliCore AI can detect degradation in model performance.

The original test dataset and trained model are NOT modified.

A selected percentage of predictions are deliberately
moved toward the incorrect outcome.

Output:
    data/artifacts/monitoring/performance_degradation_test.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "monitoring"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "performance_degradation_test.json"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Percentage of predictions to deliberately degrade.
DEGRADATION_RATE = 0.30

# Used to make the experiment reproducible.
RANDOM_STATE = 42

# Classification threshold.
THRESHOLD = 0.50


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    """Load a JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """Load the processed test dataset."""

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    df = pd.read_csv(TEST_PATH)

    target_column = "SeriousDlqin2yrs"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    y = df[target_column].astype(int)

    X = df.drop(
        columns=[target_column]
    )

    feature_names = load_json(
        FEATURE_NAMES_PATH
    )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Ensure the same feature order used during training.
    X = X[feature_names]

    print(f"Test rows: {len(X):,}")
    print(f"Features: {len(feature_names)}")

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    return X, y


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained XGBoost model."""

    print("\n" + "=" * 70)
    print("LOADING XGBOOST MODEL")
    print("=" * 70)

    model = XGBClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    print(
        f"Model loaded from:\n{MODEL_PATH}"
    )

    return model


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate classification and probability metrics.
    """

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
            )
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }

    return metrics


# ============================================================
# CREATE CONTROLLED DEGRADATION
# ============================================================

def create_degraded_predictions(
    y_true,
    original_probabilities,
):
    """
    Deliberately make a controlled portion of predictions
    less aligned with the actual outcome.

    For actual class 0:
        probability is moved to 0.70

    For actual class 1:
        probability is moved to 0.20
    """

    print("\n" + "=" * 70)
    print("CREATING CONTROLLED PERFORMANCE DEGRADATION")
    print("=" * 70)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    degraded_probabilities = (
        original_probabilities.copy()
    )

    total_predictions = len(
        degraded_probabilities
    )

    number_to_degrade = int(
        total_predictions
        * DEGRADATION_RATE
    )

    selected_indices = rng.choice(
        total_predictions,
        size=number_to_degrade,
        replace=False,
    )

    # --------------------------------------------------------
    # Intentionally move selected predictions toward
    # the incorrect class.
    # --------------------------------------------------------

    for index in selected_indices:

        actual_class = int(
            y_true.iloc[index]
        )

        if actual_class == 0:

            # Actual = negative.
            # Make model more confident that it is positive.
            degraded_probabilities[index] = 0.70

        else:

            # Actual = positive.
            # Make model more confident that it is negative.
            degraded_probabilities[index] = 0.20

    print(
        f"Total predictions: "
        f"{total_predictions:,}"
    )

    print(
        f"Predictions intentionally degraded: "
        f"{number_to_degrade:,}"
    )

    print(
        f"Degradation rate: "
        f"{DEGRADATION_RATE:.0%}"
    )

    return degraded_probabilities


# ============================================================
# CALCULATE METRIC CHANGES
# ============================================================

def calculate_metric_changes(
    baseline_metrics,
    degraded_metrics,
):
    """
    Calculate the change between degraded and baseline
    performance.
    """

    changes = {}

    for metric in baseline_metrics:

        changes[metric] = (
            degraded_metrics[metric]
            - baseline_metrics[metric]
        )

    return changes


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    baseline_metrics,
    degraded_metrics,
    metric_changes,
):
    """Print baseline versus degraded performance."""

    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<15}"
        f"{'Baseline':>15}"
        f"{'Degraded':>15}"
        f"{'Change':>15}"
    )

    print("-" * 60)

    for metric in baseline_metrics:

        print(
            f"{metric:<15}"
            f"{baseline_metrics[metric]:>15.6f}"
            f"{degraded_metrics[metric]:>15.6f}"
            f"{metric_changes[metric]:>15.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "INTELLICORE AI - "
        "CONTROLLED PERFORMANCE DEGRADATION TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Generate original model predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING BASELINE PREDICTIONS")
    print("=" * 70)

    original_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    print(
        f"Generated "
        f"{len(original_probabilities):,} "
        f"prediction probabilities."
    )

    # --------------------------------------------------------
    # Calculate baseline performance
    # --------------------------------------------------------

    baseline_metrics = calculate_metrics(
        y_test,
        original_probabilities,
    )

    # --------------------------------------------------------
    # Create deliberately degraded predictions
    # --------------------------------------------------------

    degraded_probabilities = (
        create_degraded_predictions(
            y_test,
            original_probabilities,
        )
    )

    # --------------------------------------------------------
    # Calculate degraded performance
    # --------------------------------------------------------

    degraded_metrics = calculate_metrics(
        y_test,
        degraded_probabilities,
    )

    # --------------------------------------------------------
    # Calculate changes
    # --------------------------------------------------------

    metric_changes = calculate_metric_changes(
        baseline_metrics,
        degraded_metrics,
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print_comparison(
        baseline_metrics,
        degraded_metrics,
        metric_changes,
    )

    # --------------------------------------------------------
    # Build experiment report
    # --------------------------------------------------------

    report = {
        "test_type": (
            "controlled_performance_degradation"
        ),

        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },

        "dataset": {
            "path": str(
                TEST_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "rows": int(len(y_test)),
        },

        "experiment": {
            "degradation_rate": (
                DEGRADATION_RATE
            ),
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,

            "description": (
                "A controlled subset of model predictions "
                "was intentionally moved toward the "
                "incorrect outcome to validate the "
                "performance degradation monitoring "
                "workflow."
            ),
        },

        "baseline_metrics": baseline_metrics,

        "degraded_metrics": degraded_metrics,

        "metric_changes": metric_changes,
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nSaved degradation test report to:"
        f"\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print(
        "CONTROLLED PERFORMANCE "
        "DEGRADATION TEST COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()