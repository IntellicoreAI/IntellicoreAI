"""
Final XGBoost training and evaluation for IntelliCore AI.

This script:
1. Loads leakage-free train/test datasets
2. Separates features and target
3. Trains the final XGBoost classifier
4. Generates prediction probabilities
5. Evaluates the model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
   - PR-AUC
   - Log Loss
   - Brier Score
6. Displays a confusion matrix
7. Performs threshold analysis
8. Saves evaluation metrics
9. Saves the trained XGBoost model
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "xgboost"
)

MODEL_PATH = ARTIFACT_DIR / "model.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES_PATH = ARTIFACT_DIR / "feature_names.json"

CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold_analysis.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "SeriousDlqin2yrs"

RANDOM_STATE = 42

CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("INTELLICORE AI - FINAL XGBOOST TRAINING")
    print("=" * 70)

    print("\n[1] Loading processed datasets...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found:\n{TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape:  {test_df.shape}")

    return train_df, test_df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(train_df, test_df):

    print("\n[2] Preparing features and target...")

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in testing data."
        )

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")

    print(f"\nFeatures ({len(X_train.columns)}):")

    for feature in X_train.columns:
        print(f"  - {feature}")

    return X_train, X_test, y_train, y_test


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_model(X_train, y_train):

    print("\n[3] Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print("XGBoost training completed.")

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, X_test):

    print("\n[4] Generating predictions...")

    # Probability of class 1
    y_probability = model.predict_proba(X_test)[:, 1]

    # Classification using default threshold
    y_prediction = (
        y_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    print(
        f"Classification threshold: "
        f"{CLASSIFICATION_THRESHOLD}"
    )

    return y_probability, y_prediction


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    y_test,
    y_probability,
    y_prediction
):

    print("\n[5] Evaluating model...")

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    logloss = log_loss(
        y_test,
        y_probability
    )

    brier = brier_score_loss(
        y_test,
        y_probability
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(logloss),
        "brier_score": float(brier),
    }

    print("\n" + "-" * 50)
    print("FINAL XGBOOST METRICS")
    print("-" * 50)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Log Loss:  {logloss:.4f}")
    print(f"Brier:     {brier:.4f}")

    print("-" * 50)

    return metrics


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_test,
    y_prediction
):

    print("\n[6] Creating confusion matrix...")

    # Create artifact directory BEFORE saving the image
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cm = confusion_matrix(
        y_test,
        y_prediction
    )

    print("\nConfusion Matrix:")

    print(
        f"True Negatives : {cm[0, 0]}"
    )

    print(
        f"False Positives: {cm[0, 1]}"
    )

    print(
        f"False Negatives: {cm[1, 0]}"
    )

    print(
        f"True Positives : {cm[1, 1]}"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Class 0", "Class 1"]
    )

    display.plot()

    plt.title("XGBoost Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Saved confusion matrix to:\n"
        f"{CONFUSION_MATRIX_PATH}"
    )

    return cm


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    y_test,
    y_probability
):

    print("\n[7] Performing threshold analysis...")

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    for threshold in thresholds:

        predictions = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2
                ),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
        )

    threshold_df = pd.DataFrame(
        results
    )

    print("\nThreshold Analysis:")

    print(
        threshold_df.to_string(
            index=False
        )
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False
    )

    print(
        f"\nSaved threshold analysis to:\n"
        f"{THRESHOLD_PATH}"
    )

    return threshold_df


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, feature_names):

    print("\n[8] Saving model artifacts...")

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save XGBoost model
    model.save_model(
        MODEL_PATH
    )

    # Save feature names
    with open(
        FEATURES_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            list(feature_names),
            file,
            indent=4
        )

    print(
        f"Model saved to:\n{MODEL_PATH}"
    )

    print(
        f"Feature names saved to:\n{FEATURES_PATH}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    print("\n[9] Saving evaluation metrics...")

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"Metrics saved to:\n{METRICS_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Load data
    train_df, test_df = load_data()

    # Prepare features
    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df
    )

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    # Generate predictions
    y_probability, y_prediction = generate_predictions(
        model,
        X_test
    )

    # Evaluate
    metrics = evaluate_model(
        y_test,
        y_probability,
        y_prediction
    )

    # Confusion matrix
    create_confusion_matrix(
        y_test,
        y_prediction
    )

    # Threshold analysis
    threshold_analysis(
        y_test,
        y_probability
    )

    # Save model
    save_model(
        model,
        X_train.columns
    )

    # Save metrics
    save_metrics(
        metrics
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL XGBOOST PIPELINE COMPLETE")
    print("=" * 70)

    print("\nArtifacts created:")

    print(f"  Model:             {MODEL_PATH}")
    print(f"  Metrics:           {METRICS_PATH}")
    print(f"  Feature names:     {FEATURES_PATH}")
    print(f"  Confusion matrix:  {CONFUSION_MATRIX_PATH}")
    print(f"  Threshold analysis:{THRESHOLD_PATH}")

    print("\nNext stage:")
    print("  SHAP explainability")
    print("  ↓")
    print("  Prediction telemetry")
    print("  ↓")
    print("  ML monitoring")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()