"""Create a leakage-safe next-year PM2.5 health-risk target."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PM2.5 24-hour concentration breakpoints (µg/m³)
# used for the project's AQI-style engineered risk target.
#
# The target is based on the annual summary's maximum
# reported 24-hour PM2.5 concentration (first_max_value).
#
# This is an engineered annual risk indicator and should
# not be described as an official annual AQI value.
# ---------------------------------------------------------

PM25_AQI_BINS = [
    -float("inf"),
    9.0,
    35.4,
    55.4,
    125.4,
    225.4,
    float("inf"),
]

PM25_AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]


def add_next_year_target(
    df: pd.DataFrame,
    *,
    value_col: str = "first_max_value",
    site_col: str = "site_id",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Create a leakage-safe next-year PM2.5 risk target.

    For a row representing site S in year t, the target is
    derived only from the observed PM2.5 risk statistic for
    the same site in exactly year t+1.

    Rows where year t+1 is unavailable are left without a
    target and later excluded from supervised modelling.

    The future concentration is used ONLY to construct the
    target and must not be included as a predictor.
    """

    required = {
        site_col,
        year_col,
        value_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing target-engineering columns: "
            f"{sorted(missing)}"
        )

    x = df.copy()

    # -----------------------------------------------------
    # Convert target-related fields to numeric
    # -----------------------------------------------------
    x[year_col] = pd.to_numeric(
        x[year_col],
        errors="coerce",
    ).astype("Int64")

    x[value_col] = pd.to_numeric(
        x[value_col],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Construct future-year lookup
    # -----------------------------------------------------
    # Example:
    #
    # Original:
    # site A | 2010 | concentration = 20
    # site A | 2011 | concentration = 40
    #
    # For the 2010 feature row, the target should use the
    # observed concentration from 2011.
    # -----------------------------------------------------

    future = x[
        [
            site_col,
            year_col,
            value_col,
        ]
    ].copy()

    # Shift future year backward for matching.
    future[year_col] = future[year_col] - 1

    future = future.rename(
        columns={
            value_col:
            "next_year_pm25_target_concentration"
        }
    )

    # -----------------------------------------------------
    # Match year t with exactly year t+1
    # -----------------------------------------------------
    x = x.merge(
        future,
        on=[
            site_col,
            year_col,
        ],
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Convert future PM2.5 concentration into AQI-style
    # health-risk category.
    # -----------------------------------------------------
    x["pm25_risk_category_next_year"] = pd.cut(
        x["next_year_pm25_target_concentration"],
        bins=PM25_AQI_BINS,
        labels=PM25_AQI_LABELS,
        right=True,
        include_lowest=True,
    )

    return x


def create_modeling_base(
    input_path: str | Path = (
        "data/interim/pm25_aggregated.csv"
    ),
    output_path: str | Path = (
        "data/interim/pm25_with_target.csv"
    ),
) -> pd.DataFrame:
    """
    Build the supervised-learning dataset containing
    historical site-year features and next-year risk target.
    """

    # -----------------------------------------------------
    # Load cleaned site-year dataset
    # -----------------------------------------------------
    df = pd.read_csv(
        input_path,
        dtype={
            "state_code": "string",
            "county_code": "string",
            "site_num": "string",
            "site_id": "string",
        },
    )

    # -----------------------------------------------------
    # Create next-year target
    # -----------------------------------------------------
    out = add_next_year_target(df)

    # -----------------------------------------------------
    # Keep only rows where an actual next-year observation
    # exists.
    # -----------------------------------------------------
    out = out[
        out["pm25_risk_category_next_year"].notna()
    ].copy()

    # -----------------------------------------------------
    # Sort for reproducibility
    # -----------------------------------------------------
    out = (
        out.sort_values(
            [
                "year",
                "site_id",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Save modelling base
    # -----------------------------------------------------
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    return out


if __name__ == "__main__":

    out = create_modeling_base()

    print("Target distribution:")
    print(
        out[
            "pm25_risk_category_next_year"
        ]
        .value_counts(dropna=False)
    )

    print()

    print(
        "Rows with next-year targets:",
        len(out),
    )

    print()

    print(
        "Prediction years:",
        int(out["year"].min()),
        "-",
        int(out["year"].max()),
    )

    print()

    print(
        "Target concentration range:",
        out[
            "next_year_pm25_target_concentration"
        ].min(),
        "-",
        out[
            "next_year_pm25_target_concentration"
        ].max(),
    )