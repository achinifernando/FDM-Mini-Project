"""Extract and clean PM2.5 mass-concentration records from EPA annual summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import iter_air_quality
from src.data.validate_data import validate_schema


# EPA AQS parameter 88101 = PM2.5 mass concentration under local conditions.
PM25_PARAMETER_CODE = 88101

EXPECTED_PM25_UNIT = "Micrograms/cubic meter (LC)"


KEEP_COLUMNS = [
    "state_code",
    "county_code",
    "site_num",
    "parameter_code",
    "poc",
    "latitude",
    "longitude",
    "datum",
    "parameter_name",
    "sample_duration",
    "pollutant_standard",
    "method_name",
    "year",
    "units_of_measure",
    "observation_count",
    "observation_percent",
    "completeness_indicator",
    "valid_day_count",
    "required_day_count",
    "exceptional_data_count",
    "null_data_count",
    "num_obs_below_mdl",
    "arithmetic_mean",
    "arithmetic_standard_dev",
    "first_max_value",
    "ninety_nine_percentile",
    "ninety_eight_percentile",
    "ninety_five_percentile",
    "ninety_percentile",
    "seventy_five_percentile",
    "fifty_percentile",
    "ten_percentile",
    "local_site_name",
    "state_name",
    "county_name",
    "city_name",
    "cbsa_name",
]


# Priority used when the same monitor/year/sample-duration combination
# appears under multiple pollutant-standard representations.
STANDARD_PRIORITY = {
    "PM25 24-hour 2012": 1,
    "PM25 24-hour 2006": 2,
    "PM25 Annual 2012": 3,
    "PM25 Annual 2006": 4,
}


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize EPA state/county/site identifiers."""

    df = df.copy()

    widths = {
        "state_code": 2,
        "county_code": 3,
        "site_num": 4,
    }

    for col, width in widths.items():
        df[col] = (
            df[col]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(width)
        )

    return df


def _deduplicate_pollutant_standards(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve multiple pollutant-standard representations.

    The same site/year/POC/sample-duration combination may appear under
    multiple EPA pollutant standards. Prefer the configured standard
    priority while retaining separate POCs and sample durations.

    This function can be applied both inside individual chunks and again
    after all chunks are concatenated. The second/global application is
    important because competing records may fall in different CSV chunks.
    """

    if df.empty:
        return df.copy()

    x = df.copy()

    x["_standard_priority"] = (
        x["pollutant_standard"]
        .astype("string")
        .str.strip()
        .map(STANDARD_PRIORITY)
        .fillna(99)
    )

    dedup_keys = [
        "site_id",
        "year",
        "poc",
        "sample_duration",
    ]

    x = (
        x.sort_values("_standard_priority")
        .drop_duplicates(
            subset=dedup_keys,
            keep="first",
        )
        .drop(columns="_standard_priority")
        .reset_index(drop=True)
    )

    return x


def clean_pm25_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one raw-data chunk and retain relevant 24-hour PM2.5 records."""

    validate_schema(df)

    # ---------------------------------------------------------
    # 1. Keep only EPA PM2.5 mass concentration parameter 88101
    # ---------------------------------------------------------
    x = df.loc[
        pd.to_numeric(
            df["parameter_code"],
            errors="coerce",
        ).eq(PM25_PARAMETER_CODE)
    ].copy()

    # Prevent accidental inclusion of PM2.5 chemical-speciation
    # parameters such as 88152.
    if x.empty:
        return x.reindex(
            columns=[
                c for c in KEEP_COLUMNS
                if c in df.columns
            ]
        )

    # ---------------------------------------------------------
    # 2. Keep only columns required by this project
    # ---------------------------------------------------------
    x = x[
        [
            c for c in KEEP_COLUMNS
            if c in x.columns
        ]
    ].copy()

    # ---------------------------------------------------------
    # 3. Normalize monitoring-site identifiers
    # ---------------------------------------------------------
    x = _normalize_ids(x)

    # ---------------------------------------------------------
    # 4. Convert required columns to numeric
    # ---------------------------------------------------------
    numeric_cols = [
        "latitude",
        "longitude",
        "year",
        "observation_count",
        "observation_percent",
        "valid_day_count",
        "required_day_count",
        "exceptional_data_count",
        "null_data_count",
        "num_obs_below_mdl",
        "arithmetic_mean",
        "arithmetic_standard_dev",
        "first_max_value",
        "ninety_nine_percentile",
        "ninety_eight_percentile",
        "ninety_five_percentile",
        "ninety_percentile",
        "seventy_five_percentile",
        "fifty_percentile",
        "ten_percentile",
    ]

    for col in numeric_cols:
        if col in x.columns:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

    # ---------------------------------------------------------
    # 5. Remove invalid records
    # ---------------------------------------------------------
    x = x[
        x["latitude"].between(-90, 90)
        & x["longitude"].between(-180, 180)
        & x["observation_percent"].between(0, 100)
        & x["observation_count"].ge(0)
        & x["valid_day_count"].ge(0)
    ].copy()

    # PM2.5 mass concentration cannot be negative.
    x.loc[
        x["arithmetic_mean"] < 0,
        "arithmetic_mean",
    ] = pd.NA

    # Negative maximum/percentile concentrations are also invalid.
    concentration_cols = [
        "first_max_value",
        "ninety_nine_percentile",
        "ninety_eight_percentile",
        "ninety_five_percentile",
        "ninety_percentile",
        "seventy_five_percentile",
        "fifty_percentile",
        "ten_percentile",
    ]

    for col in concentration_cols:
        if col in x.columns:
            x.loc[x[col] < 0, col] = pd.NA

    # ---------------------------------------------------------
    # 6. Keep PM2.5 mass concentration units
    # ---------------------------------------------------------
    if "units_of_measure" in x.columns:
        unit_mask = (
            x["units_of_measure"]
            .astype("string")
            .str.contains(
                r"Micrograms/cubic meter",
                case=False,
                na=False,
            )
        )

        x = x[unit_mask].copy()

    # ---------------------------------------------------------
    # 7. Keep only 24-hour PM2.5 summaries
    # ---------------------------------------------------------
    # The project target uses a 24-hour PM2.5 concentration
    # statistic mapped to AQI-style health-risk categories.
    # Therefore 1-hour summaries are excluded.
    duration = (
        x["sample_duration"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    x = x[
        duration.isin(
            [
                "24 HOUR",
                "24-HR BLK AVG",
            ]
        )
    ].copy()

    # Normalize retained duration text so deduplication is stable.
    x["sample_duration"] = (
        x["sample_duration"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # ---------------------------------------------------------
    # 8. Create unique monitoring-site ID
    # ---------------------------------------------------------
    x["site_id"] = (
        x["state_code"]
        + "-"
        + x["county_code"]
        + "-"
        + x["site_num"]
    )

    # ---------------------------------------------------------
    # 9. Resolve pollutant-standard duplicates within this chunk
    # ---------------------------------------------------------
    # Separate POCs are retained because different POCs can
    # represent different monitors at the same physical site.
    x = _deduplicate_pollutant_standards(x)

    # Remove any remaining exact duplicate rows.
    x = (
        x.drop_duplicates()
        .reset_index(drop=True)
    )

    return x


def aggregate_site_year(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse cleaned PM2.5 monitor records into one record per site-year.

    Historical PM2.5 statistics are retained for downstream feature
    engineering.
    """

    if df.empty:
        return df.copy()

    keys = [
        "site_id",
        "state_code",
        "county_code",
        "site_num",
        "year",
    ]

    # ---------------------------------------------------------
    # Observation-weighted PM2.5 mean
    # ---------------------------------------------------------
    weights = pd.to_numeric(
        df["observation_count"],
        errors="coerce",
    ).fillna(0)

    values = pd.to_numeric(
        df["arithmetic_mean"],
        errors="coerce",
    )

    tmp = df.copy()

    tmp["_weighted_sum"] = (
        values * weights
    )

    tmp["_weight"] = (
        weights.where(
            values.notna(),
            0,
        )
    )

    # ---------------------------------------------------------
    # Aggregate to one site-year
    # ---------------------------------------------------------
    agg = tmp.groupby(
        keys,
        as_index=False,
    ).agg(
        latitude=(
            "latitude",
            "median",
        ),
        longitude=(
            "longitude",
            "median",
        ),

        state_name=(
            "state_name",
            "first",
        ),
        county_name=(
            "county_name",
            "first",
        ),
        city_name=(
            "city_name",
            "first",
        ),
        cbsa_name=(
            "cbsa_name",
            "first",
        ),

        observation_count=(
            "observation_count",
            "max",
        ),
        observation_percent=(
            "observation_percent",
            "max",
        ),
        valid_day_count=(
            "valid_day_count",
            "max",
        ),

        arithmetic_standard_dev=(
            "arithmetic_standard_dev",
            "mean",
        ),

        # High-concentration statistics.
        first_max_value=(
            "first_max_value",
            "max",
        ),
        ninety_nine_percentile=(
            "ninety_nine_percentile",
            "max",
        ),
        ninety_eight_percentile=(
            "ninety_eight_percentile",
            "max",
        ),
        ninety_five_percentile=(
            "ninety_five_percentile",
            "max",
        ),
        ninety_percentile=(
            "ninety_percentile",
            "max",
        ),
        seventy_five_percentile=(
            "seventy_five_percentile",
            "max",
        ),

        fifty_percentile=(
            "fifty_percentile",
            "mean",
        ),
        ten_percentile=(
            "ten_percentile",
            "mean",
        ),

        weighted_sum=(
            "_weighted_sum",
            "sum",
        ),
        weight=(
            "_weight",
            "sum",
        ),
    )

    # ---------------------------------------------------------
    # Calculate observation-weighted PM2.5 mean
    # ---------------------------------------------------------
    agg["pm25_mean"] = (
        agg["weighted_sum"]
        / agg["weight"].replace(
            0,
            pd.NA,
        )
    )

    agg = agg.drop(
        columns=[
            "weighted_sum",
            "weight",
        ]
    )

    # ---------------------------------------------------------
    # Sort chronologically within monitoring site
    # ---------------------------------------------------------
    agg = (
        agg.sort_values(
            [
                "site_id",
                "year",
            ]
        )
        .reset_index(drop=True)
    )

    return agg


def build_interim_files(
    raw_path: str | Path = "data/raw/air_quality.csv",
    extracted_path: str | Path = "data/interim/pm25_extracted.csv",
    aggregated_path: str | Path = "data/interim/pm25_aggregated.csv",
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cleaned PM2.5 interim datasets from the raw EPA file."""

    parts: list[pd.DataFrame] = []

    # ---------------------------------------------------------
    # 1. Process large source CSV in chunks
    # ---------------------------------------------------------
    for chunk in iter_air_quality(
        raw_path,
        chunksize=chunksize,
    ):
        cleaned = clean_pm25_chunk(
            chunk
        )

        if not cleaned.empty:
            parts.append(
                cleaned
            )

    if not parts:
        raise ValueError(
            "No valid EPA PM2.5 parameter_code=88101 "
            "24-hour records were found. "
            "Check parameter_code, sample_duration, and units."
        )

    # ---------------------------------------------------------
    # 2. Combine cleaned chunks
    # ---------------------------------------------------------
    extracted = pd.concat(
        parts,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 3. IMPORTANT: global pollutant-standard deduplication
    # ---------------------------------------------------------
    # Competing standard rows may have been located in different
    # source CSV chunks. Therefore the priority rule must be
    # applied again after concatenation.
    extracted = _deduplicate_pollutant_standards(
        extracted
    )

    # Remove any remaining exact duplicates globally.
    extracted = (
        extracted
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 4. Aggregate to one row per site-year
    # ---------------------------------------------------------
    aggregated = aggregate_site_year(
        extracted
    )

    # Safety check: aggregation must produce unique site-years.
    duplicate_site_years = int(
        aggregated.duplicated(
            subset=[
                "site_id",
                "year",
            ]
        ).sum()
    )

    if duplicate_site_years != 0:
        raise ValueError(
            "Site-year aggregation failed: "
            f"{duplicate_site_years} duplicate site-year rows remain."
        )

    # ---------------------------------------------------------
    # 5. Save interim datasets
    # ---------------------------------------------------------
    outputs = [
        (
            extracted_path,
            extracted,
        ),
        (
            aggregated_path,
            aggregated,
        ),
    ]

    for path, frame in outputs:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_csv(
            path,
            index=False,
        )

    return extracted, aggregated


if __name__ == "__main__":
    extracted, aggregated = build_interim_files()

    print(
        "PM2.5 extracted:",
        extracted.shape,
    )

    print(
        "Site-year aggregated:",
        aggregated.shape,
    )

    print()
    print("Year range:")
    print(
        int(aggregated["year"].min()),
        "-",
        int(aggregated["year"].max()),
    )

    print()
    print("Unique sites:")
    print(
        aggregated["site_id"].nunique()
    )

    print()
    print("Duplicate site-year rows:")
    print(
        aggregated.duplicated(
            subset=[
                "site_id",
                "year",
            ]
        ).sum()
    )

    print()
    print("Sample durations retained:")
    print(
        extracted[
            "sample_duration"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print("Pollutant standards retained:")
    print(
        extracted[
            "pollutant_standard"
        ].value_counts(
            dropna=False
        )
    )