"""Schema and data-quality validation for the EPA annual-summary dataset."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

REQUIRED_COLUMNS = {
    "state_code", "county_code", "site_num", "parameter_code", "poc",
    "latitude", "longitude", "parameter_name", "sample_duration",
    "pollutant_standard", "year", "units_of_measure", "observation_count",
    "observation_percent", "valid_day_count", "arithmetic_mean",
    "arithmetic_standard_dev", "first_max_value",
    "ninety_nine_percentile", "ninety_eight_percentile",
    "ninety_five_percentile", "ninety_percentile", "seventy_five_percentile",
    "fifty_percentile", "ten_percentile", "state_name", "county_name",
    "city_name"
}

def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required EPA columns: {missing}")

def quality_report(df: pd.DataFrame) -> dict:
    validate_schema(df)

    numeric_checks = {}
    for col in ["latitude", "longitude", "year", "observation_count",
                "observation_percent", "valid_day_count", "arithmetic_mean"]:
        s = pd.to_numeric(df[col], errors="coerce")
        numeric_checks[col] = {
            "non_numeric_or_missing": int(s.isna().sum()),
            "min": None if s.dropna().empty else float(s.min()),
            "max": None if s.dropna().empty else float(s.max()),
        }

    invalid = {
        "latitude_out_of_range": int((pd.to_numeric(df["latitude"], errors="coerce").abs() > 90).sum()),
        "longitude_out_of_range": int((pd.to_numeric(df["longitude"], errors="coerce").abs() > 180).sum()),
        "observation_percent_out_of_range": int(
            ((pd.to_numeric(df["observation_percent"], errors="coerce") < 0) |
             (pd.to_numeric(df["observation_percent"], errors="coerce") > 100)).sum()
        ),
        "negative_observation_count": int((pd.to_numeric(df["observation_count"], errors="coerce") < 0).sum()),
        "negative_valid_day_count": int((pd.to_numeric(df["valid_day_count"], errors="coerce") < 0).sum()),
    }

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "numeric_checks": numeric_checks,
        "invalid_value_checks": invalid,
    }

def save_quality_report(df: pd.DataFrame, output_path: str | Path) -> dict:
    report = quality_report(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    from src.data.load_data import load_air_quality
    data = load_air_quality()
    report = save_quality_report(data, "reports/data_quality_report.json")
    print(json.dumps(report["invalid_value_checks"], indent=2))
