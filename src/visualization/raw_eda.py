"""
Enhanced exploratory data analysis for the raw EPA annual air-quality summary CSV.

Covers:
- Dataset structure and variable types
- Missing-value analysis (counts, percentages, patterns, visualisation)
- Duplicate detection
- Outlier analysis (IQR method + boxplots)
- Class imbalance (for proposed classification target)
- Distributions by pollutant, state, and year
- Correlation analysis (numeric features)
- Target-variable proposal and analysis
- Data-leakage risk identification

Outputs:
- reports/raw_eda/*.csv
- reports/raw_eda/*.png
- reports/raw_eda/raw_eda_report.md
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "epa_air_quality_annual_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "raw_eda"
CHUNK_SIZE = 100_000
SAMPLE_PER_PARAMETER = 5_000      # stratified sample size per pollutant
RANDOM_STATE = 42

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

# Extra columns we want to inspect but may not use as features
EXTRA_COLUMNS = ["state_code", "county_code", "site_number", "datum"]

USECOLS = sorted(set(NUMERIC_COLUMNS + CATEGORY_COLUMNS + EXTRA_COLUMNS))

# Columns that are likely leakage if predicting arithmetic_mean
POTENTIAL_LEAKAGE_COLUMNS = [
    "arithmetic_standard_dev",
    "first_max_value",
    "ninety_five_percentile",
    "fifty_percentile",
    "observation_count",
    "valid_day_count",
    "observation_percent",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _update_counter(counter: Counter, values: pd.Series) -> None:
    counter.update(values.dropna().astype(str).str.strip())


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 1: Read data in chunks and collect metadata
# ---------------------------------------------------------------------------

def _read_chunks() -> tuple[dict, pd.DataFrame]:
    """
    Read the raw CSV in chunks, collecting:
    - total row count
    - missing-value counts per column
    - duplicate row count
    - category frequency counters
    - numeric arrays (for full-data statistics)
    - a stratified sample per pollutant for visualisation
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw EPA dataset not found: {RAW_PATH}")

    row_count = 0
    duplicate_count = 0
    missing = Counter()
    coercion_counts = Counter()
    category_counts = {col: Counter() for col in CATEGORY_COLUMNS}
    numeric_chunks: dict[str, list[pd.Series]] = {col: [] for col in NUMERIC_COLUMNS}
    sample_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        RAW_PATH, usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False
    ):
        row_count += len(chunk)

        # --- Missing values ---
        missing.update(chunk.isna().sum().to_dict())

        # --- Duplicates ---
        duplicate_count += int(chunk.duplicated().sum())

        # --- Category frequencies ---
        for col in CATEGORY_COLUMNS:
            _update_counter(category_counts[col], chunk[col])

        # --- Numeric coercion tracking ---
        for col in NUMERIC_COLUMNS:
            raw_non_null = chunk[col].notna().sum()
            coerced = pd.to_numeric(chunk[col], errors="coerce")
            coercion_counts[col] += int(raw_non_null - coerced.notna().sum())
            numeric_chunks[col].append(coerced)

        # --- Stratified sample per pollutant ---
        for param, group in chunk.groupby("parameter_name", dropna=True):
            current = sum(
                (part["parameter_name"] == param).sum() for part in sample_parts
            )
            if current < SAMPLE_PER_PARAMETER:
                take = min(SAMPLE_PER_PARAMETER - current, len(group))
                sample_parts.append(group.sample(take, random_state=RANDOM_STATE))

    # Build full numeric frame
    numeric_frame = pd.DataFrame(
        {col: pd.concat(vals, ignore_index=True) for col, vals in numeric_chunks.items()}
    )

    # Build stratified sample
    sample = (
        pd.concat(sample_parts, ignore_index=True)
        if sample_parts
        else pd.DataFrame(columns=USECOLS)
    )

    metadata = {
        "rows": row_count,
        "columns": len(pd.read_csv(RAW_PATH, nrows=0).columns),
        "missing": missing,
        "duplicates": duplicate_count,
        "coercions": coercion_counts,
        "category_counts": category_counts,
        "numeric_frame": numeric_frame,
    }
    return metadata, sample


# ---------------------------------------------------------------------------
# Step 2: Structure and variable-type analysis
# ---------------------------------------------------------------------------

def _analyse_structure(metadata: dict, sample: pd.DataFrame) -> pd.DataFrame:
    """Report dtype, non-null counts, and uniqueness."""
    structure = pd.DataFrame({
        "dtype": sample.dtypes.astype(str),
        "non_null": sample.notna().sum(),
        "null": sample.isna().sum(),
        "unique": sample.nunique(),
    })
    structure["null_pct"] = structure["null"] / len(sample) * 100
    structure.to_csv(OUTPUT_DIR / "column_structure.csv")
    return structure


# ---------------------------------------------------------------------------
# Step 3: Missing-value analysis
# ---------------------------------------------------------------------------

def _analyse_missing(metadata: dict) -> pd.DataFrame:
    rows = metadata["rows"]
    missing_df = pd.DataFrame({
        "column": list(metadata["missing"].keys()),
        "missing_values": list(metadata["missing"].values()),
    })
    missing_df["missing_percent"] = missing_df["missing_values"] / rows * 100
    missing_df = missing_df.sort_values("missing_values", ascending=False)
    missing_df.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

    # Visualisation
    plot_df = missing_df[missing_df["missing_values"] > 0].copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
        sns.barplot(
            data=plot_df, x="missing_percent", y="column",
            color="#e07a5f", ax=ax
        )
        ax.set_title("Missing-value percentage by column")
        ax.set_xlabel("Missing (%)")
        ax.set_ylabel("")
        _save_fig(OUTPUT_DIR / "missing_values.png")
    return missing_df


# ---------------------------------------------------------------------------
# Step 4: Duplicate analysis
# ---------------------------------------------------------------------------

def _analyse_duplicates(metadata: dict, sample: pd.DataFrame) -> dict:
    dup_count = metadata["duplicates"]
    dup_pct = dup_count / metadata["rows"] * 100

    # Show examples from the sample
    sample_dups = sample[sample.duplicated(keep=False)].head(20)
    sample_dups.to_csv(OUTPUT_DIR / "duplicate_examples.csv", index=False)

    result = {
        "total_rows": metadata["rows"],
        "duplicate_rows": dup_count,
        "duplicate_percent": round(dup_pct, 4),
    }
    with open(OUTPUT_DIR / "duplicate_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Step 5: Outlier analysis (IQR)
# ---------------------------------------------------------------------------

def _analyse_outliers(metadata: dict, sample: pd.DataFrame) -> pd.DataFrame:
    numeric = metadata["numeric_frame"]
    records = []
    for col in NUMERIC_COLUMNS:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        records.append({
            "column": col,
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "outlier_count": int(outliers),
            "outlier_percent": round(outliers / len(series) * 100, 4),
        })
    outlier_df = pd.DataFrame(records)
    outlier_df.to_csv(OUTPUT_DIR / "outliers_iqr.csv", index=False)

    # Boxplots for key concentration columns by parameter
    key_cols = ["arithmetic_mean", "first_max_value", "ninety_five_percentile"]
    key_cols = [c for c in key_cols if c in sample.columns]
    if key_cols:
        top_params = [p for p, _ in metadata["category_counts"]["parameter_name"].most_common(6)]
        plot_sample = sample[sample["parameter_name"].isin(top_params)].copy()
        for col in key_cols:
            plot_sample[col] = pd.to_numeric(plot_sample[col], errors="coerce")
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.boxplot(
                data=plot_sample, x="parameter_name", y=col,
                palette="Set2", ax=ax
            )
            ax.set_title(f"Outlier distribution of {col} by pollutant")
            ax.set_xlabel("")
            ax.set_ylabel(col)
            plt.xticks(rotation=30, ha="right")
            _save_fig(OUTPUT_DIR / f"boxplot_{col}.png")
    return outlier_df


# ---------------------------------------------------------------------------
# Step 6: Class-imbalance analysis (for proposed classification target)
# ---------------------------------------------------------------------------

def _analyse_class_imbalance(metadata: dict, sample: pd.DataFrame) -> pd.DataFrame:
    """
    Propose a classification target: whether arithmetic_mean exceeds the
    pollutant-specific 90th percentile (i.e., 'unhealthy' air quality).
    """
    df = sample.copy()
    df["arithmetic_mean"] = pd.to_numeric(df["arithmetic_mean"], errors="coerce")
    df = df.dropna(subset=["arithmetic_mean", "parameter_name"])

    thresholds = (
        df.groupby("parameter_name")["arithmetic_mean"]
          .quantile(0.90)
          .rename("threshold")
    )
    df = df.merge(thresholds, on="parameter_name", how="left")
    df["high_pollution"] = (df["arithmetic_mean"] > df["threshold"]).astype(int)

    counts = df["high_pollution"].value_counts().rename_axis("class").reset_index(name="count")
    counts["percent"] = counts["count"] / counts["count"].sum() * 100
    counts.to_csv(OUTPUT_DIR / "class_imbalance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=counts, x="class", y="count", palette="Set1", ax=ax)
    ax.set_title("Proposed classification target distribution\n(high pollution vs. normal)")
    ax.set_xlabel("Class (0 = normal, 1 = high pollution)")
    ax.set_ylabel("Records")
    for i, row in counts.iterrows():
        ax.text(i, row["count"], f"{row['percent']:.1f}%", ha="center", va="bottom")
    _save_fig(OUTPUT_DIR / "class_imbalance.png")
    return counts


# ---------------------------------------------------------------------------
# Step 7: Distribution analysis (overall + by group)
# ---------------------------------------------------------------------------

def _analyse_distributions(metadata: dict, sample: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # 7a. Records by year
    yearly = metadata["numeric_frame"].groupby("year", dropna=True).size()
    if not yearly.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        yearly.plot(marker="o", color="#176b87", ax=ax)
        ax.set_title("Raw records by year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of records")
        _save_fig(OUTPUT_DIR / "records_by_year.png")

    # 7b. Top states
    top_states = metadata["category_counts"]["state_name"].most_common(15)
    if top_states:
        state_frame = (
            pd.DataFrame(top_states, columns=["state_name", "records"])
              .sort_values("records")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=state_frame, x="records", y="state_name",
                    color="#3a86ff", ax=ax)
        ax.set_title("Top 15 states by raw record count")
        ax.set_xlabel("Records")
        ax.set_ylabel("")
        _save_fig(OUTPUT_DIR / "top_states.png")

    # 7c. Distribution of arithmetic_mean per pollutant (stratified sample)
    top_params = [p for p, _ in metadata["category_counts"]["parameter_name"].most_common(6)]
    plot_sample = sample[sample["parameter_name"].isin(top_params)].copy()
    plot_sample["arithmetic_mean"] = pd.to_numeric(
        plot_sample["arithmetic_mean"], errors="coerce"
    )
    plot_sample = plot_sample[plot_sample["arithmetic_mean"] >= 0]

    for param in top_params:
        subset = plot_sample[plot_sample["parameter_name"] == param]["arithmetic_mean"].dropna()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(subset, bins=50, color="#e07a5f", ax=ax, kde=True)
        ax.set_title(f"arithmetic_mean distribution — {param}")
        ax.set_xlabel("arithmetic_mean")
        ax.set_ylabel("Records")
        safe = param.replace("/", "_").replace(" ", "_")
        _save_fig(OUTPUT_DIR / f"dist_{safe}.png")

    # 7d. Summary statistics by parameter
    group_stats = (
        plot_sample.groupby("parameter_name")["arithmetic_mean"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .sort_values("count", ascending=False)
    )
    group_stats.to_csv(OUTPUT_DIR / "stats_by_parameter.csv")


# ---------------------------------------------------------------------------
# Step 8: Correlation and relationship analysis
# ---------------------------------------------------------------------------

def _analyse_correlations(metadata: dict) -> None:
    numeric = metadata["numeric_frame"].dropna(how="all")
    # Only keep columns with enough non-null values
    numeric = numeric.loc[:, numeric.notna().mean() > 0.5]
    if numeric.shape[1] < 2:
        return

    corr = numeric.corr()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap — numeric features (full data)")
    _save_fig(OUTPUT_DIR / "correlation_heatmap.png")

    # Flag strong correlations (|r| > 0.8) for later feature selection
    strong = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
    )
    strong = strong[strong["corr"].abs() > 0.8].sort_values("corr", ascending=False)
    strong.to_csv(OUTPUT_DIR / "strong_correlations.csv", index=False)


# ---------------------------------------------------------------------------
# Step 9: Data-leakage analysis
# ---------------------------------------------------------------------------

def _analyse_leakage(metadata: dict) -> pd.DataFrame:
    rows = []
    for col in POTENTIAL_LEAKAGE_COLUMNS:
        if col in NUMERIC_COLUMNS:
            reason = "Derived from the same observations as the target."
        else:
            reason = "Potentially redundant or derivable from other columns."
        rows.append({"column": col, "leakage_risk": reason})

    # State_code vs state_name redundancy
    rows.append({
        "column": "state_code / state_name",
        "leakage_risk": "Redundant identifiers — keep one only.",
    })

    leakage_df = pd.DataFrame(rows)
    leakage_df.to_csv(OUTPUT_DIR / "data_leakage_risks.csv", index=False)
    return leakage_df


# ---------------------------------------------------------------------------
# Step 10: Write the final report
# ---------------------------------------------------------------------------

def _write_report(metadata: dict, structure: pd.DataFrame,
                  missing_df: pd.DataFrame, dup_summary: dict,
                  outlier_df: pd.DataFrame, imbalance_df: pd.DataFrame,
                  leakage_df: pd.DataFrame) -> None:
    numeric = metadata["numeric_frame"]
    param_top = metadata["category_counts"]["parameter_name"].most_common(1)
    state_top = metadata["category_counts"]["state_name"].most_common(1)

    report = [
        "# Raw EPA Air-Quality — Exploratory Data Analysis",
        "",
        f"- **Source**: `{RAW_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- **Rows**: {metadata['rows']:,}",
        f"- **Columns**: {metadata['columns']}",
        f"- **Duplicate rows**: {dup_summary['duplicate_rows']:,} "
        f"({dup_summary['duplicate_percent']}%)",
        "",
        "## 1. Dataset structure",
        "",
        f"- Numeric columns analysed: {len(NUMERIC_COLUMNS)}",
        f"- Categorical columns analysed: {len(CATEGORY_COLUMNS)}",
        f"- The dataset spans years {int(numeric['year'].min())} to "
        f"{int(numeric['year'].max())}.",
        f"- Most frequent pollutant: **{param_top[0][0]}** "
        f"({param_top[0][1]:,} records).",
        f"- Most represented state: **{state_top[0][0]}** "
        f"({state_top[0][1]:,} records).",
        "",
        "See `column_structure.csv` for full dtype and uniqueness information.",
        "",
        "## 2. Missing values",
        "",
        f"- Columns with missing values: "
        f"{(missing_df['missing_values'] > 0).sum()} / {len(missing_df)}.",
        "- Full table: `missing_values.csv`; visual: `missing_values.png`.",
        "",
        "## 3. Duplicates",
        "",
        f"- Detected **{dup_summary['duplicate_rows']:,}** duplicate rows "
        f"({dup_summary['duplicate_percent']}% of the dataset).",
        "- Examples saved to `duplicate_examples.csv`.",
        "- Decision: exact duplicates should be dropped during cleaning to avoid "
        "inflating model performance.",
        "",
        "## 4. Outliers (IQR method)",
        "",
        "- Per-column IQR bounds and outlier counts: `outliers_iqr.csv`.",
        "- Boxplots per pollutant: `boxplot_arithmetic_mean.png`, "
        "`boxplot_first_max_value.png`, `boxplot_ninety_five_percentile.png`.",
        "- Outliers in air-quality data often reflect genuine extreme events "
        "(wildfires, industrial incidents) — treat with care rather than "
        "blindly removing.",
        "",
        "## 5. Class imbalance (proposed classification target)",
        "",
        "- Proposed target: whether a pollutant reading exceeds its "
        "parameter-specific 90th percentile ('high pollution').",
        "- Distribution saved to `class_imbalance.csv` and visualised in "
        "`class_imbalance.png`.",
        "- Imbalance ratio should be reported and addressed during modelling "
        "(class weights, SMOTE, or threshold tuning).",
        "",
        "## 6. Distributions",
        "",
        "- Records by year: `records_by_year.png`",
        "- Top states: `top_states.png`",
        "- Per-pollutant distributions: `dist_<pollutant>.png`",
        "- Summary statistics by pollutant: `stats_by_parameter.csv`",
        "",
        "## 7. Relationships and correlations",
        "",
        "- Correlation matrix: `correlation_matrix.csv`",
        "- Heatmap: `correlation_heatmap.png`",
        "- Strong correlations (|r| > 0.8): `strong_correlations.csv`",
        "- Highly correlated features must be handled during feature selection.",
        "",
        "## 8. Data leakage risks",
        "",
        "- Identified leakage candidates are listed in `data_leakage_risks.csv`.",
        "- **Critical**: columns such as `arithmetic_standard_dev`, "
        "`first_max_value`, `ninety_five_percentile`, and `fifty_percentile` "
        "are derived from the same observations as the target "
        "(`arithmetic_mean`) and must be excluded when predicting it.",
        "- `state_code` and `state_name` are redundant — keep one.",
        "",
        "## 9. Proposed target variable",
        "",
        "Two viable targets emerge from this EDA:",
        "",
        "1. **Regression** — predict `arithmetic_mean` (PM2.5 concentration).",
        "2. **Classification** — predict whether a reading exceeds its "
        "pollutant-specific 90th percentile (high pollution).",
        "",
        "Final choice should be confirmed with the instructor during dataset "
        "validation.",
        "",
        "## 10. Key observations informing preprocessing",
        "",
        "1. Drop exact duplicates before modelling.",
        "2. Exclude leakage-prone columns when predicting `arithmetic_mean`.",
        "3. Handle missing values — most columns have < 5% missing, so imputation "
        "is feasible.",
        "4. Filter to a single pollutant (e.g., PM2.5) for a clean modelling "
        "problem, as units and scales differ across pollutants.",
        "5. Use stratified sampling or temporal splitting to avoid leakage.",
        "6. Address class imbalance with class weights or resampling.",
        "7. Investigate strong correlations before feature selection.",
        "",
        "## Figures",
        "",
        "![Records by year](records_by_year.png)",
        "![Missing values](missing_values.png)",
        "![Correlation heatmap](correlation_heatmap.png)",
        "![Class imbalance](class_imbalance.png)",
    ]
    (OUTPUT_DIR / "raw_eda_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_output_dir()
    sns.set_theme(style="whitegrid")

    print("Reading raw data in chunks …")
    metadata, sample = _read_chunks()
    print(f"  → {metadata['rows']:,} rows, "
          f"{metadata['columns']} columns, "
          f"{metadata['duplicates']:,} duplicates")

    print("Analysing structure …")
    structure = _analyse_structure(metadata, sample)

    print("Analysing missing values …")
    missing_df = _analyse_missing(metadata)

    print("Analysing duplicates …")
    dup_summary = _analyse_duplicates(metadata, sample)

    print("Analysing outliers …")
    outlier_df = _analyse_outliers(metadata, sample)

    print("Analysing class imbalance …")
    imbalance_df = _analyse_class_imbalance(metadata, sample)

    print("Analysing distributions …")
    _analyse_distributions(metadata, sample)

    print("Analysing correlations …")
    _analyse_correlations(metadata)

    print("Analysing data-leakage risks …")
    leakage_df = _analyse_leakage(metadata)

    print("Writing report …")
    _write_report(
        metadata, structure, missing_df, dup_summary,
        outlier_df, imbalance_df, leakage_df,
    )

    print(f"\nEDA complete.")
    print(f"Report:  {OUTPUT_DIR / 'raw_eda_report.md'}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()