"""Leakage-safe feature engineering for next-year PM2.5 risk prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Numeric variables available in the current prediction year (t).
BASE_NUMERIC = [
    "pm25_mean",
    "arithmetic_standard_dev",
    "first_max_value",
    "ninety_nine_percentile",
    "ninety_eight_percentile",
    "ninety_five_percentile",
    "ninety_percentile",
    "seventy_five_percentile",
    "fifty_percentile",
    "ten_percentile",
    "observation_count",
    "observation_percent",
    "valid_day_count",
]


TARGET_COLUMN = "pm25_risk_category_next_year"

# This column was used only to construct the target.
# It must NEVER be supplied to the ML model.
FUTURE_TARGET_VALUE = "next_year_pm25_target_concentration"


def add_history_features(
    df: pd.DataFrame,
    *,
    site_col: str = "site_id",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Create leakage-safe historical PM2.5 features.

    All generated features use information from the current
    prediction year t or earlier. No information from year
    t+1 is used as a predictor.
    """

    x = df.copy()

    # -----------------------------------------------------
    # 1. Basic validation
    # -----------------------------------------------------
    required = {
        site_col,
        year_col,
        "pm25_mean",
    }

    missing = required - set(x.columns)

    if missing:
        raise ValueError(
            "Missing feature-engineering columns: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # 2. Convert numeric columns
    # -----------------------------------------------------
    x[year_col] = pd.to_numeric(
        x[year_col],
        errors="coerce",
    )

    available_numeric = [
        col for col in BASE_NUMERIC
        if col in x.columns
    ]

    for col in available_numeric:
        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    # Sort chronologically within each site.
    x = (
        x.sort_values(
            [site_col, year_col]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # 3. Previous-year PM2.5 concentration
    # -----------------------------------------------------
    # Use lag-1 only if the previous available observation
    # is exactly the previous calendar year.
    grouped = x.groupby(
        site_col,
        sort=False,
    )

    previous_year = grouped[year_col].shift(1)
    previous_pm25 = grouped["pm25_mean"].shift(1)

    exact_previous_year = (
        x[year_col] - previous_year
    ).eq(1)

    x["pm25_lag1"] = previous_pm25.where(
        exact_previous_year
    )

    # -----------------------------------------------------
    # 4. One-year concentration trend
    # -----------------------------------------------------
    x["pm25_change_1yr"] = (
        x["pm25_mean"]
        - x["pm25_lag1"]
    )

    x["pm25_pct_change_1yr"] = (
        x["pm25_change_1yr"]
        / x["pm25_lag1"].replace(0, pd.NA)
    )

    # -----------------------------------------------------
    # 5. Exact two-year lag
    # -----------------------------------------------------
    previous_year_2 = grouped[year_col].shift(2)
    previous_pm25_2 = grouped["pm25_mean"].shift(2)

    exact_two_year_history = (
        x[year_col] - previous_year_2
    ).eq(2)

    x["pm25_lag2"] = previous_pm25_2.where(
        exact_two_year_history
    )

    # -----------------------------------------------------
    # 6. Historical rolling features
    # -----------------------------------------------------
    # 2-year mean = year t and t-1.
    # It is available only where an exact t-1 observation
    # exists.
    x["pm25_rolling_2yr_mean"] = (
        (
            x["pm25_mean"]
            + x["pm25_lag1"]
        )
        / 2
    )

    # 3-year statistics require observations from
    # t, t-1 and t-2.
    three_year_values = x[
        [
            "pm25_mean",
            "pm25_lag1",
            "pm25_lag2",
        ]
    ]

    x["pm25_rolling_3yr_mean"] = (
        three_year_values.mean(
            axis=1,
            skipna=False,
        )
    )

    x["pm25_rolling_3yr_std"] = (
        three_year_values.std(
            axis=1,
            skipna=False,
            ddof=1,
        )
    )

    # -----------------------------------------------------
    # 7. Distribution / pollution-spread features
    # -----------------------------------------------------
    if {
        "ninety_five_percentile",
        "fifty_percentile",
    } <= set(x.columns):

        x["p95_minus_median"] = (
            x["ninety_five_percentile"]
            - x["fifty_percentile"]
        )

    if {
        "ninety_nine_percentile",
        "ten_percentile",
    } <= set(x.columns):

        x["p99_p10_range"] = (
            x["ninety_nine_percentile"]
            - x["ten_percentile"]
        )

    # -----------------------------------------------------
    # 8. Peak relative to typical PM2.5 level
    # -----------------------------------------------------
    if {
        "first_max_value",
        "pm25_mean",
    } <= set(x.columns):

        x["max_to_mean_ratio"] = (
            x["first_max_value"]
            / x["pm25_mean"].replace(0, pd.NA)
        )

    # -----------------------------------------------------
    # 9. Observation-density feature
    # -----------------------------------------------------
    if {
        "valid_day_count",
        "observation_count",
    } <= set(x.columns):

        x["observations_per_valid_day"] = (
            x["observation_count"]
            / x["valid_day_count"].replace(
                0,
                pd.NA,
            )
        )

    # -----------------------------------------------------
    # 10. Monitoring-history indicator
    # -----------------------------------------------------
    x["has_previous_year"] = (
        x["pm25_lag1"]
        .notna()
        .astype(int)
    )

    x["has_three_year_history"] = (
        x["pm25_lag2"]
        .notna()
        .astype(int)
    )

    return x


def build_feature_dataset(
    input_path: str | Path = (
        "data/interim/pm25_with_target.csv"
    ),
    output_path: str | Path = (
        "data/interim/ml_ready_base.csv"
    ),
) -> pd.DataFrame:
    """
    Build the final leakage-safe feature-engineered base
    dataset for downstream preprocessing and modelling.
    """

    # -----------------------------------------------------
    # Load target-engineered dataset
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

    # Ensure target exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found."
        )

    # -----------------------------------------------------
    # Generate historical features
    # -----------------------------------------------------
    out = add_history_features(df)

    # -----------------------------------------------------
    # CRITICAL LEAKAGE PREVENTION
    # -----------------------------------------------------
    # next_year_pm25_target_concentration contains the
    # actual future concentration used to generate the
    # supervised label. It must not be a model feature.
    out = out.drop(
        columns=[
            FUTURE_TARGET_VALUE,
            "next_year_pm25_mean",
            "target_risk",
        ],
        errors="ignore",
    )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------
    if FUTURE_TARGET_VALUE in out.columns:
        raise RuntimeError(
            "Target leakage detected: future target "
            "concentration remains in feature dataset."
        )

    if TARGET_COLUMN not in out.columns:
        raise RuntimeError(
            "Target column was accidentally removed."
        )

    # Sort for reproducibility.
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
    # Save ML-ready base dataset
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

    out = build_feature_dataset()

    print(
        "ML-ready base shape:",
        out.shape,
    )

    print()

    print("Target distribution:")
    print(
        out[
            TARGET_COLUMN
        ].value_counts()
    )

    print()

    print("Engineered features:")

    engineered_features = [
        "pm25_lag1",
        "pm25_lag2",
        "pm25_change_1yr",
        "pm25_pct_change_1yr",
        "pm25_rolling_2yr_mean",
        "pm25_rolling_3yr_mean",
        "pm25_rolling_3yr_std",
        "p95_minus_median",
        "p99_p10_range",
        "max_to_mean_ratio",
        "observations_per_valid_day",
        "has_previous_year",
        "has_three_year_history",
    ]

    for feature in engineered_features:
        if feature in out.columns:
            print("-", feature)

    print()

    print(
        "Future target concentration present:",
        FUTURE_TARGET_VALUE in out.columns,
    )

    print()

    print(
        "Final target column:",
        TARGET_COLUMN,
    )