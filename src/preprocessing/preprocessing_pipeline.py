"""

Purpose:
Prepare the clean, feature-engineered base dataset for machine-learning
models using a reusable and leakage-safe preprocessing pipeline.

Main responsibilities:
1. Validate the expected target column.
2. Replace textual "unknown" values with missing values.
3. Correct numeric columns that were loaded as object/string.
4. Separate predictors and target.
5. Remove identifier columns that should not be model predictors.
6. Impute missing numerical and categorical values.
7. Standardize numerical features.
8. One-hot encode categorical features.
9. Provide reusable fit/transform functions.
10. Prevent preprocessing leakage by fitting transformations on
    training data only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

TARGET_COLUMN = "pm25_risk_category_next_year"

# These columns identify a monitoring site rather than describe
# its air-quality behaviour. Keep them in the original dataframe
# for grouping/splitting if needed, but do not use them as model
# predictors.
IDENTIFIER_COLUMNS = [
    "site_id",
    "state_code",
    "county_code",
    "site_num",
]

# In the current ML-ready CSV these columns contain some textual
# "unknown" values, so pandas may read them as object columns even
# though they represent numerical measurements.
FORCE_NUMERIC_COLUMNS = [
    "ten_percentile",
    "p99_p10_range",
]

# Missing-value markers that may appear in CSV data.
MISSING_MARKERS = [
    "unknown",
    "Unknown",
    "UNKNOWN",
    "",
    "NA",
    "N/A",
    "null",
    "NULL",
    "None",
]

# Columns that must never be used as predictors if they appear
# in future versions of the dataset.
#
# The current target is generated from next-year PM2.5 information.
# Any direct next-year measurement would leak future information.
KNOWN_LEAKAGE_COLUMNS = [
    "pm25_next_year",
    "next_year_pm25",
    "next_year_arithmetic_mean",
    "pm25_mean_next_year",
]


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_ml_ready_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the ML-ready base dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to ml_ready_base.csv.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path, low_memory=False)

    print(f"Dataset loaded successfully: {file_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

def validate_input_dataframe(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Validate basic requirements before preprocessing.

    This function does not modify the dataframe.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Required target column '{target_column}' was not found."
        )

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()

    if duplicated_columns:
        raise ValueError(
            f"Duplicate column names found: {duplicated_columns}"
        )

    print("Basic dataframe validation passed.")


# ============================================================
# 4. CLEAN SPECIAL MISSING-VALUE MARKERS
# ============================================================

def replace_unknown_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace textual missing-value markers such as 'unknown'
    with np.nan.

    The operation is performed on a copy so the original
    dataframe is not modified.
    """

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.replace(MISSING_MARKERS, np.nan)

    return cleaned_df


# ============================================================
# 5. CORRECT DATA TYPES
# ============================================================

def convert_expected_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: Iterable[str] = FORCE_NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """
    Convert columns that should be numeric but may have been loaded
    as object/string because they contained 'unknown' values.

    Invalid values are converted to NaN and later handled by the
    training-only numerical imputer.
    """

    converted_df = df.copy()

    for column in numeric_columns:
        if column in converted_df.columns:
            converted_df[column] = pd.to_numeric(
                converted_df[column],
                errors="coerce",
            )

    return converted_df


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows if any exist.

    Member 2's EDA found no exact duplicates in the current
    dataset, but this keeps the preprocessing code robust.
    """

    cleaned_df = df.copy()

    duplicate_count = cleaned_df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count:,} duplicate rows.")
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicate rows found.")

    return cleaned_df


# ============================================================
# 7. CHECK FOR OBVIOUS FUTURE-TARGET LEAKAGE
# ============================================================

def check_known_leakage_columns(
    df: pd.DataFrame,
    leakage_columns: Iterable[str] = KNOWN_LEAKAGE_COLUMNS,
) -> None:
    """
    Check whether obvious future-information columns are present.

    Raises an error if a known direct future PM2.5 measurement
    appears among the predictors.
    """

    detected = [
        column
        for column in leakage_columns
        if column in df.columns
    ]

    if detected:
        raise ValueError(
            "Potential target leakage columns detected: "
            f"{detected}. These columns must not be used as predictors."
        )

    print("No known direct future-target leakage columns detected.")


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    identifier_columns: Iterable[str] = IDENTIFIER_COLUMNS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate model predictors (X) and target (y).

    Identifier columns are removed from X because they mainly
    identify individual monitoring sites and can encourage
    memorisation rather than generalisable learning.

    The original dataframe can still be retained separately for
    chronological/group-based splitting by Member 4.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # Remove rows with missing target values.
    # We should not impute labels.
    missing_target = df[target_column].isna().sum()

    if missing_target > 0:
        print(
            f"Dropping {missing_target:,} rows with missing target values."
        )
        df = df.loc[df[target_column].notna()].copy()

    y = df[target_column].copy()

    columns_to_drop = [target_column]

    columns_to_drop.extend(
        column
        for column in identifier_columns
        if column in df.columns
    )

    X = df.drop(columns=columns_to_drop).copy()

    # Additional protection: target must never remain in X.
    if target_column in X.columns:
        raise RuntimeError(
            "Target column unexpectedly remained in feature matrix."
        )

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(
        "Excluded identifier columns:",
        [
            column
            for column in identifier_columns
            if column in df.columns
        ],
    )

    return X, y


# ============================================================
# 9. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

def identify_feature_types(
    X: pd.DataFrame,
) -> Tuple[list[str], list[str]]:
    """
    Automatically identify numerical and categorical predictors.
    """

    numeric_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    recognised = set(numeric_features + categorical_features)

    unsupported = [
        column
        for column in X.columns
        if column not in recognised
    ]

    if unsupported:
        raise TypeError(
            "Unsupported feature dtypes found for columns: "
            f"{unsupported}"
        )

    print(
        f"Numerical features: {len(numeric_features)}"
    )
    print(
        f"Categorical features: {len(categorical_features)}"
    )

    return numeric_features, categorical_features


# ============================================================
# 10. NUMERICAL PREPROCESSING PIPELINE
# ============================================================

def build_numeric_pipeline() -> Pipeline:
    """
    Build preprocessing steps for numerical predictors.

    Median imputation:
        Robust to skewed distributions and outliers.

    StandardScaler:
        Places numerical variables on comparable scales and is
        useful for scale-sensitive algorithms such as Logistic
        Regression, SVM and KNN.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return numeric_pipeline


# ============================================================
# 11. CATEGORICAL PREPROCESSING PIPELINE
# ============================================================

def build_categorical_pipeline() -> Pipeline:
    """
    Build preprocessing steps for categorical predictors.

    Most-frequent imputation:
        Replaces missing categories using values learned only
        from the training data.

    OneHotEncoder:
        Converts categories to numeric model features.

    handle_unknown='ignore':
        Prevents failure when validation/test data contains a
        category not observed during training.
    """

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return categorical_pipeline


# ============================================================
# 12. BUILD COMPLETE COLUMN TRANSFORMER
# ============================================================

def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Construct the complete preprocessing transformer.

    IMPORTANT:
    X_train should contain TRAINING DATA ONLY.

    This function determines feature groups from X_train and
    returns an unfitted ColumnTransformer.
    """

    numeric_features, categorical_features = identify_feature_types(
        X_train
    )

    numeric_pipeline = build_numeric_pipeline()
    categorical_pipeline = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


# ============================================================
# 13. FIT PREPROCESSOR - TRAINING DATA ONLY
# ============================================================

def fit_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build and fit the preprocessing pipeline using TRAINING DATA ONLY.

    Fitting the imputer, scaler or encoder on validation/test data
    would introduce information leakage.
    """

    preprocessor = build_preprocessor(X_train)

    preprocessor.fit(X_train)

    print("Preprocessing pipeline fitted successfully.")
    print(
        "IMPORTANT: preprocessing statistics were learned "
        "from training data only."
    )

    return preprocessor


# ============================================================
# 14. TRANSFORM DATA
# ============================================================

def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
):
    """
    Transform a feature dataframe using an already-fitted
    preprocessing pipeline.

    Use this function for training, validation and test sets after
    the preprocessor has been fitted on X_train.
    """

    transformed = preprocessor.transform(X)

    return transformed


# ============================================================
# 15. FEATURE NAMES AFTER PREPROCESSING
# ============================================================

def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """
    Return output feature names after scaling and one-hot encoding.
    """

    return preprocessor.get_feature_names_out().tolist()


# ============================================================
# 16. DATA QUALITY SUMMARY
# ============================================================

def preprocessing_summary(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Print a short summary useful for testing and viva evidence.
    """

    print("\n" + "=" * 60)
    print("PREPROCESSING INPUT SUMMARY")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")

    print(
        f"Duplicate rows: {df.duplicated().sum():,}"
    )

    total_missing = df.isna().sum().sum()

    print(
        f"Standard missing values before marker replacement: "
        f"{total_missing:,}"
    )

    object_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    unknown_count = 0

    for column in object_columns:
        unknown_count += (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("unknown")
            .sum()
        )

    print(
        f"Textual 'unknown' values: {unknown_count:,}"
    )

    if target_column in df.columns:
        print("\nTarget distribution:")
        print(df[target_column].value_counts(dropna=False))

    print("=" * 60)


# ============================================================
# 17. COMPLETE PRE-SPLIT PREPARATION FUNCTION
# ============================================================

def prepare_base_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply only deterministic cleaning that is safe before the
    train/validation/test split.

    This function DOES NOT calculate medians, means, scaling
    statistics or category vocabularies.

    Therefore it does not learn information from future/test rows.
    """

    validate_input_dataframe(df)

    prepared_df = replace_unknown_values(df)

    prepared_df = convert_expected_numeric_columns(prepared_df)

    prepared_df = remove_duplicate_rows(prepared_df)

    check_known_leakage_columns(prepared_df)

    return prepared_df


# ============================================================
# 18. EXAMPLE / STANDALONE TEST
# ============================================================

def main() -> None:
    """
    Demonstrate the preprocessing preparation using the project's
    current ml_ready_base.csv.

    This does NOT perform the final train/validation/test split,
    because dataset splitting belongs to Member 4.

    It validates that Member 3's preprocessing components can be
    constructed successfully.
    """

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "interim"
        / "ml_ready_base.csv"
    )

    print("\n" + "=" * 60)
    print("MEMBER 3 - PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load data
    df = load_ml_ready_data(data_path)

    # Show current data-quality summary
    preprocessing_summary(df)

    # Apply deterministic pre-split preparation
    prepared_df = prepare_base_dataframe(df)

    # Separate X and y for pipeline inspection.
    # Member 4 will later perform the official chronological split.
    X, y = prepare_features_and_target(prepared_df)

    numeric_features, categorical_features = identify_feature_types(X)

    print("\nNumerical columns:")
    for column in numeric_features:
        print(f"  - {column}")

    print("\nCategorical columns:")
    for column in categorical_features:
        print(f"  - {column}")

    # Build but DO NOT fit on the full dataset.
    #
    # Fitting here would learn preprocessing statistics from future
    # observations before Member 4 creates the chronological split.
    preprocessor = build_preprocessor(X)

    print("\nPipeline created successfully:")
    print(preprocessor)

    print("\nTarget classes:")
    for target_class in sorted(y.unique()):
        print(f"  - {target_class}")

    print("\n" + "=" * 60)
    print("MEMBER 3 PREPROCESSING SETUP COMPLETED")
    print("=" * 60)

    print(
        "\nThe pipeline has NOT been fitted on the complete dataset."
    )
    print(
        "After Member 4 creates the chronological split, use:"
    )

    print(
        """
preprocessor = fit_preprocessor(X_train)

X_train_processed = transform_features(
    preprocessor,
    X_train
)

X_val_processed = transform_features(
    preprocessor,
    X_val
)

X_test_processed = transform_features(
    preprocessor,
    X_test
)
"""
    )


if __name__ == "__main__":
    main()