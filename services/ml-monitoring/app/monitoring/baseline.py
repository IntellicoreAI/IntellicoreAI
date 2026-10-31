from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "monitoring"
BASELINE_PATH = ARTIFACTS_DIR / "baseline.json"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "SeriousDlqin2yrs"


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():
    """
    Load the processed training dataset.

    This dataset is used as the reference population
    for ML monitoring.
    """

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_DATA_PATH}"
        )

    df = pd.read_csv(TRAIN_DATA_PATH)

    print(f"Training dataset loaded: {df.shape}")

    return df


# ============================================================
# FEATURE STATISTICS
# ============================================================

def calculate_feature_statistics(df):
    """
    Calculate baseline statistics for every model feature.
    """

    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    statistics = {}

    for feature in feature_columns:

        series = pd.to_numeric(df[feature], errors="coerce")

        statistics[feature] = {
            "dtype": str(df[feature].dtype),

            "count": int(series.count()),

            "missing_count": int(series.isna().sum()),

            "missing_rate": float(series.isna().mean()),

            "mean": float(series.mean()),

            "median": float(series.median()),

            "std": float(series.std()),

            "min": float(series.min()),

            "max": float(series.max()),

            "quantiles": {
                "q01": float(series.quantile(0.01)),
                "q05": float(series.quantile(0.05)),
                "q25": float(series.quantile(0.25)),
                "q50": float(series.quantile(0.50)),
                "q75": float(series.quantile(0.75)),
                "q95": float(series.quantile(0.95)),
                "q99": float(series.quantile(0.99)),
            },
        }

    return statistics


# ============================================================
# TARGET STATISTICS
# ============================================================

def calculate_target_statistics(df):
    """
    Calculate baseline statistics for the target variable.
    """

    if TARGET_COLUMN not in df.columns:
        return None

    target = df[TARGET_COLUMN]

    class_counts = target.value_counts().sort_index()

    total = len(target)

    class_distribution = {}

    for class_value, count in class_counts.items():

        class_distribution[str(int(class_value))] = {
            "count": int(count),
            "rate": float(count / total),
        }

    return {
        "column": TARGET_COLUMN,
        "total_count": int(total),
        "class_distribution": class_distribution,
    }


# ============================================================
# CORRELATION BASELINE
# ============================================================

def calculate_correlations(df):
    """
    Calculate feature-to-feature correlations.

    These can later help diagnose changes in relationships
    between variables.
    """

    feature_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    correlation_matrix = feature_df.corr(numeric_only=True)

    correlations = {}

    for column in correlation_matrix.columns:

        correlations[column] = {
            other_column: float(
                correlation_matrix.loc[column, other_column]
            )
            for other_column in correlation_matrix.columns
        }

    return correlations


# ============================================================
# CREATE BASELINE
# ============================================================

def create_baseline():

    print("=" * 60)
    print("ML MONITORING BASELINE")
    print("=" * 60)

    # Load reference dataset
    df = load_training_data()

    print(f"Reference rows: {len(df)}")
    print(f"Reference columns: {len(df.columns)}")

    # Identify features
    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    print(f"Model features: {len(feature_columns)}")

    # Feature statistics
    print("\nCalculating feature statistics...")

    feature_statistics = calculate_feature_statistics(df)

    # Target statistics
    print("Calculating target statistics...")

    target_statistics = calculate_target_statistics(df)

    # Correlations
    print("Calculating feature correlations...")

    correlations = calculate_correlations(df)

    # Create baseline object
    baseline = {
        "baseline_version": "1.0",

        "dataset": {
            "source": "data/processed/train.csv",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target_column": TARGET_COLUMN,
            "feature_columns": feature_columns,
        },

        "feature_statistics": feature_statistics,

        "target_statistics": target_statistics,

        "feature_correlations": correlations,
    }

    # Create directory
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save baseline
    with open(
        BASELINE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            baseline,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("BASELINE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Saved to:")
    print(BASELINE_PATH)

    print("\nFeatures included:")

    for feature in feature_columns:
        print(f"  - {feature}")

    print("\nTarget distribution:")

    if target_statistics:

        for class_value, stats in target_statistics[
            "class_distribution"
        ].items():

            print(
                f"  Class {class_value}: "
                f"{stats['count']} "
                f"({stats['rate']:.2%})"
            )

    print("\nBaseline is ready for drift monitoring.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    create_baseline()