from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "ml_ready_base.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_DIR / "data_quality_summary.txt"

TARGET_COLUMN = "pm25_risk_category_next_year"


def evaluate_data_quality():
    """Evaluate the quality of the ML-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset was not found at: {INPUT_FILE}"
        )

    # Read identifier columns as strings to preserve leading zeros
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

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows, total_columns = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_site_years = int(
        df.duplicated(subset=["site_id", "year"]).sum()
    )

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(
        ascending=False
    )

    numeric_df = df.select_dtypes(include=np.number)
    infinite_values = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    target_percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    report_lines = [
        "AIR QUALITY DATA QUALITY REPORT",
        "=" * 50,
        f"Input file: {INPUT_FILE}",
        f"Total rows: {total_rows}",
        f"Total columns: {total_columns}",
        f"Unique monitoring sites: {df['site_id'].nunique()}",
        f"Year range: {df['year'].min()} - {df['year'].max()}",
        "",
        "DUPLICATE CHECKS",
        "-" * 50,
        f"Fully duplicated rows: {duplicate_rows}",
        f"Duplicated site-year records: {duplicate_site_years}",
        "",
        "MISSING-VALUE CHECKS",
        "-" * 50,
        f"Columns containing missing values: {len(missing_counts)}",
    ]

    if missing_counts.empty:
        report_lines.append("No missing values were found.")
    else:
        for column, count in missing_counts.items():
            percentage = (count / total_rows) * 100
            report_lines.append(
                f"{column}: {count} ({percentage:.2f}%)"
            )

    report_lines.extend(
        [
            "",
            "INVALID-VALUE CHECKS",
            "-" * 50,
            f"Infinite numeric values: {infinite_values}",
            f"Missing target values: "
            f"{df[TARGET_COLUMN].isna().sum()}",
            "",
            "TARGET CLASS DISTRIBUTION",
            "-" * 50,
        ]
    )

    for category, count in target_counts.items():
        percentage = target_percentages.loc[category]
        report_lines.append(
            f"{category}: {count} ({percentage:.2f}%)"
        )

    report_lines.extend(
        [
            "",
            "DATA TYPES",
            "-" * 50,
        ]
    )

    for column, dtype in df.dtypes.items():
        report_lines.append(f"{column}: {dtype}")

    report_text = "\n".join(report_lines)

    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print(report_text)
    print("\nData-quality report saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    evaluate_data_quality()