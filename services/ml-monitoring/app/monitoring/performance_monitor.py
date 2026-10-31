"""
Model Performance Monitoring
----------------------------

Evaluates the trained XGBoost credit-risk model against the
fixed processed test dataset.

Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - PR-AUC
    - Log Loss
    - Brier Score

Also generates:
    - Confusion matrix
    - Performance report

Output:
    data/artifacts/monitoring/performance_report.json
    data/artifacts/monitoring/performance_confusion_matrix.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
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
    confusion_matrix,
    ConfusionMatrixDisplay,
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

REPORT_PATH = (
    OUTPUT_DIR
    / "performance_report.json"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "performance_confusion_matrix.png"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    """Load a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load the processed test dataset and separate
    features from the ground-truth target.
    """

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    df = pd.read_csv(TEST_PATH)

    target_column = "SeriousDlqin2yrs"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in test dataset."
        )

    y = df[target_column].astype(int)

    X = df.drop(columns=[target_column])

    feature_names = load_json(FEATURE_NAMES_PATH)

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Ensure exact model feature order
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
    """
    Load the trained XGBoost model.
    """

    print("\n" + "=" * 70)
    print("LOADING XGBOOST MODEL")
    print("=" * 70)

    model = XGBClassifier()

    model.load_model(str(MODEL_PATH))

    print(f"Model loaded from:")
    print(MODEL_PATH)

    return model


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

def calculate_metrics(y_true, y_probability, threshold=0.50):
    """
    Calculate classification and probability-based metrics.
    """

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_probability,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                y_probability,
            )
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),
    }

    return metrics, y_pred


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, y_pred):
    """
    Generate and save the confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[0, 1],
    )

    display.plot()

    plt.title(
        "XGBoost Credit Risk - Confusion Matrix"
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved confusion matrix to:"
        f"\n{CONFUSION_MATRIX_PATH}"
    )

    return matrix


# ============================================================
# BUILD PERFORMANCE REPORT
# ============================================================

def build_report(
    metrics,
    confusion_matrix_values,
    y_true,
):
    """
    Build a structured JSON performance report.
    """

    tn, fp, fn, tp = (
        confusion_matrix_values.ravel()
    )

    report = {
        "monitoring_type": "model_performance",

        "model": {
            "model_name": "xgboost_credit_risk",
            "model_version": "1.0",
        },

        "dataset": {
            "path": str(
                TEST_PATH.relative_to(PROJECT_ROOT)
            ),
            "rows": int(len(y_true)),
            "positive_samples": int(
                (y_true == 1).sum()
            ),
            "negative_samples": int(
                (y_true == 0).sum()
            ),
        },

        "metrics": metrics,

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "interpretation": {
            "accuracy": (
                "Overall fraction of correctly "
                "classified observations."
            ),
            "precision": (
                "Fraction of positive predictions "
                "that were actually positive."
            ),
            "recall": (
                "Fraction of actual positive cases "
                "correctly identified."
            ),
            "f1_score": (
                "Harmonic mean of precision and recall."
            ),
            "roc_auc": (
                "Ability of the model to rank positive "
                "cases above negative cases."
            ),
            "pr_auc": (
                "Precision-recall based measure that "
                "is informative under class imbalance."
            ),
            "log_loss": (
                "Measures the quality of predicted "
                "probabilities."
            ),
            "brier_score": (
                "Measures squared error of predicted "
                "probabilities."
            ),
        },
    }

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("INTELLICORE AI - MODEL PERFORMANCE MONITORING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING MODEL PREDICTIONS")
    print("=" * 70)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    print(
        f"Generated {len(y_probability):,} "
        "prediction probabilities."
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics, y_pred = calculate_metrics(
        y_test,
        y_probability,
        threshold=0.50,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    for name, value in metrics.items():

        if name == "threshold":
            print(
                f"{name:<15}: {value:.2f}"
            )
        else:
            print(
                f"{name:<15}: {value:.6f}"
            )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = save_confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = build_report(
        metrics,
        matrix,
        y_test,
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nSaved performance report to:"
        f"\n{REPORT_PATH}"
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE MONITORING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()