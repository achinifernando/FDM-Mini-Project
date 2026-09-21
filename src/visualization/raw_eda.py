"""Exploratory analysis of the raw EPA annual air-quality summary CSV."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "epa_air_quality_annual_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "raw_eda"
CHUNK_SIZE = 100_000
SAMPLE_SIZE = 100_000

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "year",
    "observation_count",
    "observation_percent",
    "valid_day_count",
    "arithmetic_mean",
    "arithmetic_standard_dev",
    "first_max_value",
    "ninety_five_percentile",
    "fifty_percentile",
]

CATEGORY_COLUMNS = [
    "parameter_name",
    "units_of_measure",
    "sample_duration",
    "state_name",
]

USECOLS = sorted(set(NUMERIC_COLUMNS + CATEGORY_COLUMNS + ["state_code"]))


def _update_counter(counter: Counter, values: pd.Series) -> None:
    counter.update(values.dropna().astype(str).str.strip())


def _read_chunks() -> tuple[dict, pd.DataFrame]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw EPA dataset not found: {RAW_PATH}")

    row_count = 0
    missing = Counter()
    category_counts = {column: Counter() for column in CATEGORY_COLUMNS}
    numeric_values = {column: [] for column in NUMERIC_COLUMNS}
    sample_parts = []

    for chunk in pd.read_csv(RAW_PATH, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False):
        row_count += len(chunk)
        missing.update(chunk.isna().sum().to_dict())

        for column in CATEGORY_COLUMNS:
            _update_counter(category_counts[column], chunk[column])
        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(chunk[column], errors="coerce")
            numeric_values[column].append(values)

        if sum(len(part) for part in sample_parts) < SAMPLE_SIZE:
            sample_parts.append(chunk.head(SAMPLE_SIZE - sum(len(part) for part in sample_parts)))

    numeric_frame = pd.DataFrame({
        column: pd.concat(values, ignore_index=True) for column, values in numeric_values.items()
    })
    sample = pd.concat(sample_parts, ignore_index=True).head(SAMPLE_SIZE)
    metadata = {
        "rows": row_count,
        "columns": len(pd.read_csv(RAW_PATH, nrows=0).columns),
        "missing": missing,
        "category_counts": category_counts,
        "numeric_frame": numeric_frame,
    }
    return metadata, sample


def _save_figures(metadata: dict, sample: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    yearly = metadata["numeric_frame"].groupby("year", dropna=True).size()
    yearly.plot(figsize=(10, 5), marker="o", color="#176b87")
    plt.title("Raw records by year")
    plt.xlabel("Year")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "records_by_year.png", dpi=160)
    plt.close()

    plot_data = sample["arithmetic_mean"].pipe(pd.to_numeric, errors="coerce").dropna()
    plot_data = plot_data[plot_data >= 0]
    sns.histplot(plot_data, bins=50, color="#e07a5f")
    plt.title("Distribution of arithmetic mean concentrations (sample)")
    plt.xlabel("Arithmetic mean")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arithmetic_mean_distribution.png", dpi=160)
    plt.close()

    top_states = metadata["category_counts"]["state_name"].most_common(15)
    state_frame = pd.DataFrame(top_states, columns=["state_name", "records"]).sort_values("records")
    sns.barplot(data=state_frame, x="records", y="state_name", color="#3a86ff")
    plt.title("Top 15 states by raw record count")
    plt.xlabel("Number of records")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_states.png", dpi=160)
    plt.close()


def _write_report(metadata: dict) -> None:
    numeric = metadata["numeric_frame"]
    missing = pd.DataFrame({
        "column": list(metadata["missing"]),
        "missing_values": list(metadata["missing"].values()),
    })
    missing["missing_percent"] = missing["missing_values"] / metadata["rows"] * 100
    missing = missing.sort_values("missing_values", ascending=False)
    missing.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

    numeric.describe().T.to_csv(OUTPUT_DIR / "numeric_summary.csv")
    for column, counts in metadata["category_counts"].items():
        pd.DataFrame(counts.most_common(20), columns=[column, "records"]).to_csv(
            OUTPUT_DIR / f"top_{column}.csv", index=False
        )

    report = [
        "# Raw EPA Air-Quality EDA",
        "",
        f"- Source: `{RAW_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Rows: **{metadata['rows']:,}**",
        f"- Columns: **{metadata['columns']}**",
        f"- Numeric summaries use all raw chunks; the concentration histogram uses the first {SAMPLE_SIZE:,} rows.",
        "",
        "## Key observations",
        "",
        f"- The dataset spans years {int(numeric['year'].min())} to {int(numeric['year'].max())}.",
        f"- The most frequent parameter is **{metadata['category_counts']['parameter_name'].most_common(1)[0][0]}**.",
        f"- The most represented state is **{metadata['category_counts']['state_name'].most_common(1)[0][0]}**.",
        f"- `arithmetic_mean` has {int(numeric['arithmetic_mean'].isna().sum()):,} non-numeric or missing values after coercion.",
        "",
        "## Data-quality signals",
        "",
        "The complete missing-value table is in `missing_values.csv`; numeric statistics are in `numeric_summary.csv`.",
        "Raw records contain multiple pollutants, units, durations, and monitoring years. PM2.5-specific filtering should therefore remain in the cleaning stage before modeling.",
        "",
        "## Figures",
        "",
        "![Records by year](records_by_year.png)",
        "![Arithmetic mean distribution](arithmetic_mean_distribution.png)",
        "![Top states](top_states.png)",
    ]
    (OUTPUT_DIR / "raw_eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata, sample = _read_chunks()
    _save_figures(metadata, sample)
    _write_report(metadata)
    print(f"EDA complete: {metadata['rows']:,} rows analysed")
    print(f"Report: {OUTPUT_DIR / 'raw_eda_report.md'}")


if __name__ == "__main__":
    main()