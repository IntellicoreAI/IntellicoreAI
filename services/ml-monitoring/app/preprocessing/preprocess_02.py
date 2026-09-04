"""
Leakage-free preprocessing for the Give Me Some Credit dataset.

Pipeline:
1. Load raw training data
2. Remove unnecessary index column
3. Remove duplicate rows
4. Separate features and target
5. Split into training and testing sets
6. Fit median imputation ONLY on training data
7. Apply the fitted imputer to both train and test data
8. Validate the processed data
9. Save processed train/test datasets and preprocessing metadata
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "GiveMeSomeCredit_2011"
    / "cs-training.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
METADATA_PATH = PROCESSED_DIR / "preprocessing_metadata.json"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "SeriousDlqin2yrs"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_data():

    print("=" * 70)
    print("INTELLICORE AI - DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------------

    print("\n[1] Loading raw dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Original shape: {df.shape}")


    # --------------------------------------------------------
    # 2. Remove unnecessary index column
    # --------------------------------------------------------

    print("\n[2] Removing unnecessary columns...")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Removed: Unnamed: 0")
    else:
        print("Unnamed: 0 not found")


    # --------------------------------------------------------
    # 3. Remove duplicate rows
    # --------------------------------------------------------

    print("\n[3] Checking duplicate rows...")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    print(f"Shape after duplicate removal: {df.shape}")


    # --------------------------------------------------------
    # 4. Validate target column
    # --------------------------------------------------------

    print("\n[4] Validating target column...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    print(f"Target column: {TARGET_COLUMN}")


    # --------------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------------

    print("\n[5] Separating features and target...")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Number of features: {X.shape[1]}")
    print(f"Features: {list(X.columns)}")


    # --------------------------------------------------------
    # 6. Check target values
    # --------------------------------------------------------

    print("\n[6] Target distribution...")

    target_counts = y.value_counts().sort_index()
    target_percentages = (
        y.value_counts(normalize=True).sort_index() * 100
    )

    for value in target_counts.index:
        print(
            f"Class {value}: "
            f"{target_counts[value]} "
            f"({target_percentages[value]:.2f}%)"
        )


    # --------------------------------------------------------
    # 7. Train/Test split
    # --------------------------------------------------------

    print("\n[7] Performing train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")


    # --------------------------------------------------------
    # 8. Check missing values BEFORE imputation
    # --------------------------------------------------------

    print("\n[8] Missing values before imputation...")

    missing_train = X_train.isnull().sum()
    missing_test = X_test.isnull().sum()

    train_missing = missing_train[missing_train > 0]
    test_missing = missing_test[missing_test > 0]

    if len(train_missing) > 0:
        print("\nTraining data:")
        print(train_missing)
    else:
        print("No missing values in training data.")

    if len(test_missing) > 0:
        print("\nTesting data:")
        print(test_missing)
    else:
        print("No missing values in testing data.")


    # --------------------------------------------------------
    # 9. Fit imputer ONLY on training data
    # --------------------------------------------------------

    print("\n[9] Fitting median imputer on TRAINING data only...")

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    print("Training imputer fitted.")


    # --------------------------------------------------------
    # 10. Apply SAME imputer to test data
    # --------------------------------------------------------

    print("\n[10] Applying training imputer to TEST data...")

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print("Test data transformed using training statistics.")


    # --------------------------------------------------------
    # 11. Display imputation values
    # --------------------------------------------------------

    print("\n[11] Imputation values learned from training data:")

    imputation_values = {}

    for feature, value in zip(
        X_train.columns,
        imputer.statistics_
    ):
        imputation_values[feature] = float(value)
        print(f"{feature}: {value}")


    # --------------------------------------------------------
    # 12. Validate missing values
    # --------------------------------------------------------

    print("\n[12] Validating missing values...")

    train_nan_count = X_train_imputed.isnull().sum().sum()
    test_nan_count = X_test_imputed.isnull().sum().sum()

    print(f"Training NaN count: {train_nan_count}")
    print(f"Testing NaN count:  {test_nan_count}")


    # --------------------------------------------------------
    # 13. Validate infinite values
    # --------------------------------------------------------

    print("\n[13] Checking infinite values...")

    train_infinite_count = np.isinf(
        X_train_imputed.to_numpy()
    ).sum()

    test_infinite_count = np.isinf(
        X_test_imputed.to_numpy()
    ).sum()

    print(f"Training infinite values: {train_infinite_count}")
    print(f"Testing infinite values:  {test_infinite_count}")


    # --------------------------------------------------------
    # 14. Reconstruct final datasets
    # --------------------------------------------------------

    print("\n[14] Creating final processed datasets...")

    train_processed = X_train_imputed.copy()
    train_processed[TARGET_COLUMN] = y_train.values

    test_processed = X_test_imputed.copy()
    test_processed[TARGET_COLUMN] = y_test.values


    # --------------------------------------------------------
    # 15. Create processed directory
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 16. Save processed datasets
    # --------------------------------------------------------

    print("\n[15] Saving processed datasets...")

    train_processed.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_processed.to_csv(
        TEST_PATH,
        index=False
    )

    print(f"Training data saved to:\n{TRAIN_PATH}")
    print(f"Testing data saved to:\n{TEST_PATH}")


    # --------------------------------------------------------
    # 17. Save preprocessing metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": "Give Me Some Credit",
        "source_file": str(RAW_DATA_PATH),

        "target_column": TARGET_COLUMN,

        "original_rows": int(len(pd.read_csv(RAW_DATA_PATH))),
        "rows_after_duplicate_removal": int(len(df)),
        "duplicates_removed": int(duplicate_count),

        "number_of_features": int(X.shape[1]),
        "features": list(X.columns),

        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified_split": True,

        "imputation": {
            "method": "median",
            "fit_on": "training_data_only",
            "values": imputation_values
        },

        "train_shape": list(X_train_imputed.shape),
        "test_shape": list(X_test_imputed.shape),

        "train_class_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().items()
        },

        "test_class_distribution": {
            str(k): int(v)
            for k, v in y_test.value_counts().items()
        },

        "validation": {
            "train_nan_count": int(train_nan_count),
            "test_nan_count": int(test_nan_count),
            "train_infinite_count": int(train_infinite_count),
            "test_infinite_count": int(test_infinite_count)
        }
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print(f"Metadata saved to:\n{METADATA_PATH}")


    # --------------------------------------------------------
    # 18. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"Final training shape: {train_processed.shape}")
    print(f"Final testing shape:  {test_processed.shape}")

    print("\nValidation:")
    print(f"Training NaN values: {train_nan_count}")
    print(f"Testing NaN values:  {test_nan_count}")
    print(f"Training infinite values: {train_infinite_count}")
    print(f"Testing infinite values:  {test_infinite_count}")

    print("\nSaved files:")
    print(f"1. {TRAIN_PATH}")
    print(f"2. {TEST_PATH}")
    print(f"3. {METADATA_PATH}")

    print("\nNo test-set statistics were used during imputation.")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess_data()