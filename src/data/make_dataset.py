from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ml_ready_base.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "pm25_risk_category_next_year"

# Chronological split boundaries
TRAIN_END_YEAR = 2012
VALIDATION_START_YEAR = 2013
VALIDATION_END_YEAR = 2014
TEST_START_YEAR = 2015


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_dataset():
    """Load the ML-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset was not found: {INPUT_FILE}"
        )

    identifier_types = {
        "site_id": "string",
        "state_code": "string",
        "county_code": "string",
        "site_num": "string",
    }

    df = pd.read_csv(
        INPUT_FILE,
        dtype=identifier_types,
        low_memory=False,
    )

    print(
        f"Dataset loaded successfully: "
        f"{df.shape[0]} rows, {df.shape[1]} columns"
    )

    return df


# --------------------------------------------------
# Merge rare target class
# --------------------------------------------------

def merge_rare_target_class(df):
    """
    Merge Hazardous into Very Unhealthy.

    The Hazardous class contains only four records.
    This decision was approved by the group leader.
    """

    df = df.copy()

    hazardous_before = (
        df[TARGET_COLUMN]
        .eq("Hazardous")
        .sum()
    )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(
        {"Hazardous": "Very Unhealthy"}
    )

    hazardous_after = (
        df[TARGET_COLUMN]
        .eq("Hazardous")
        .sum()
    )

    print("\nRare-class merging")
    print("-" * 50)
    print(
        f"Hazardous records before merging: "
        f"{hazardous_before}"
    )
    print(
        f"Hazardous records after merging: "
        f"{hazardous_after}"
    )

    return df


# --------------------------------------------------
# Create chronological splits
# --------------------------------------------------

def create_time_based_splits(df):
    """Create chronological train, validation and test sets."""

    required_columns = {"year", TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    train_data = df[
        df["year"] <= TRAIN_END_YEAR
    ].copy()

    validation_data = df[
        (df["year"] >= VALIDATION_START_YEAR)
        & (df["year"] <= VALIDATION_END_YEAR)
    ].copy()

    test_data = df[
        df["year"] >= TEST_START_YEAR
    ].copy()

    if train_data.empty:
        raise ValueError("Training dataset is empty.")

    if validation_data.empty:
        raise ValueError("Validation dataset is empty.")

    if test_data.empty:
        raise ValueError("Testing dataset is empty.")

    return train_data, validation_data, test_data


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

def separate_features_and_target(dataset):
    """Separate predictor features (X) and target labels (y)."""

    X = dataset.drop(columns=[TARGET_COLUMN]).copy()
    y = dataset[[TARGET_COLUMN]].copy()

    return X, y


# --------------------------------------------------
# Verify splits
# --------------------------------------------------

def verify_splits(
    original_data,
    train_data,
    validation_data,
    test_data,
):
    """Check that splitting did not lose or duplicate rows."""

    total_split_rows = (
        len(train_data)
        + len(validation_data)
        + len(test_data)
    )

    if total_split_rows != len(original_data):
        raise ValueError(
            "The total number of split rows does not match "
            "the original dataset."
        )

    if train_data["year"].max() >= validation_data["year"].min():
        raise ValueError(
            "Training and validation years overlap."
        )

    if validation_data["year"].max() >= test_data["year"].min():
        raise ValueError(
            "Validation and test years overlap."
        )

    print("\nSplit verification passed.")
    print(f"Original rows: {len(original_data)}")
    print(f"Total split rows: {total_split_rows}")


# --------------------------------------------------
# Print summary
# --------------------------------------------------

def print_split_summary(name, dataset):
    """Display size, years and target distribution."""

    print(f"\n{name.upper()} SET")
    print("=" * 50)
    print(f"Number of rows: {len(dataset)}")
    print(
        f"Year range: "
        f"{dataset['year'].min()} - "
        f"{dataset['year'].max()}"
    )

    class_counts = dataset[TARGET_COLUMN].value_counts()
    class_percentages = (
        dataset[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    summary = pd.DataFrame(
        {
            "count": class_counts,
            "percentage": class_percentages,
        }
    )

    print("\nTarget distribution:")
    print(summary)


# --------------------------------------------------
# Save split datasets
# --------------------------------------------------

def save_datasets(
    train_data,
    validation_data,
    test_data,
):
    """Save full splits and separate X/y files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = separate_features_and_target(
        train_data
    )

    X_validation, y_validation = separate_features_and_target(
        validation_data
    )

    X_test, y_test = separate_features_and_target(
        test_data
    )

    output_files = {
        "train_data.csv": train_data,
        "validation_data.csv": validation_data,
        "test_data.csv": test_data,
        "X_train.csv": X_train,
        "X_validation.csv": X_validation,
        "X_test.csv": X_test,
        "y_train.csv": y_train,
        "y_validation.csv": y_validation,
        "y_test.csv": y_test,
    }

    print("\nSaving datasets")
    print("-" * 50)

    for filename, dataset in output_files.items():
        output_path = OUTPUT_DIR / filename
        dataset.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():
    """Run the complete dataset-splitting process."""

    df = load_dataset()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column was not found: {TARGET_COLUMN}"
        )

    # Merge the approved rare class
    df = merge_rare_target_class(df)

    print("\nFinal five target classes:")
    print(df[TARGET_COLUMN].value_counts())

    # Create chronological splits
    train_data, validation_data, test_data = (
        create_time_based_splits(df)
    )

    # Confirm no rows were lost and no years overlap
    verify_splits(
        df,
        train_data,
        validation_data,
        test_data,
    )

    # Display split details
    print_split_summary("Training", train_data)
    print_split_summary("Validation", validation_data)
    print_split_summary("Testing", test_data)

    # Save all output files
    save_datasets(
        train_data,
        validation_data,
        test_data,
    )

    print("\nDataset splitting completed successfully.")


if __name__ == "__main__":
    main()