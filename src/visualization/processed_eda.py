"""Exploratory analysis of the model-ready PM2.5 feature dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ml_ready_base.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "processed_eda"
TARGET = "pm25_risk_category_next_year"

TARGET_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

EXCLUDED_COLUMNS = {
    "site_id",
    "state_code",
    "county_code",
    "site_num",
    "state_name",
    "county_name",
    "city_name",
    "cbsa_name",
    TARGET,
}


def _load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    return pd.read_csv(
        DATA_PATH,
        dtype={"site_id": "string", "state_code": "string"},
        low_memory=False,
    )


def _save_figures(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    sns.set_theme(style="whitegrid")
    numeric_frame = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    target_counts = df[TARGET].value_counts().reindex(TARGET_ORDER, fill_value=0)

    target_frame = target_counts.rename_axis("risk_category").reset_index(name="records")
    target_frame["percent"] = target_frame["records"] / len(df) * 100
    target_frame.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)

    sns.barplot(data=target_frame, x="records", y="risk_category", color="#176b87")
    plt.title("Next-year PM2.5 risk-category distribution")
    plt.xlabel("Records")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
    plt.close()

    sns.boxplot(
        data=df,
        x=TARGET,
        y="pm25_mean",
        order=TARGET_ORDER,
        color="#e07a5f",
        showfliers=False,
    )
    plt.title("Current-year PM2.5 mean by next-year risk category")
    plt.xlabel("Next-year risk category")
    plt.ylabel("Current-year PM2.5 mean")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pm25_mean_by_target.png", dpi=160)
    plt.close()

    history = pd.DataFrame({
        "history_status": ["Previous year available", "Three-year history available"],
        "records": [df["has_previous_year"].sum(), df["has_three_year_history"].sum()],
    })
    sns.barplot(data=history, x="history_status", y="records", color="#3a86ff")
    plt.title("Availability of historical monitoring features")
    plt.xlabel("")
    plt.ylabel("Records")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "history_availability.png", dpi=160)
    plt.close()

    correlation_columns = [column for column in numeric_columns if numeric_frame[column].notna().sum() > 1]
    correlation = numeric_frame[correlation_columns].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, linewidths=0.2)
    plt.title("Correlation among numeric model-ready features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=160)
    plt.close()


def _write_report(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    data_quality = pd.DataFrame({
        "column": df.columns,
        "null_values": df.isna().sum().values,
        "unknown_values": [(df[column] == "unknown").sum() for column in df.columns],
    })
    data_quality["null_percent"] = data_quality["null_values"] / len(df) * 100
    data_quality["unknown_percent"] = data_quality["unknown_values"] / len(df) * 100
    data_quality = data_quality.sort_values(["null_values", "unknown_values"], ascending=False)
    data_quality.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)
    numeric_frame = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_frame.describe().T.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    target_counts = df[TARGET].value_counts().reindex(TARGET_ORDER, fill_value=0)
    target_percent = target_counts / len(df) * 100
    rarest = target_counts[target_counts > 0].idxmin()
    unknown_counts = data_quality.set_index("column")["unknown_values"]
    unknown_total = int(unknown_counts.sum())
    unknown_columns = unknown_counts[unknown_counts > 0]
    report = [
        "# Processed PM2.5 Dataset EDA",
        "",
        f"- Source: `{DATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{df.shape[1]}**",
        f"- Monitoring sites: **{df['site_id'].nunique():,}**",
        f"- Prediction years: **{int(df['year'].min())}–{int(df['year'].max())}**",
        "",
        "## Key observations",
        "",
        f"- The largest target class is **{target_counts.idxmax()}** with {target_counts.max():,} records ({target_percent.max():.1f}%).",
        f"- The rarest non-empty target class is **{rarest}** with {target_counts[rarest]:,} records ({target_percent[rarest]:.2f}%).",
        f"- `pm25_mean` has a median of **{df['pm25_mean'].median():.2f}** and a maximum of **{df['pm25_mean'].max():.2f}**.",
        f"- Previous-year history is available for {df['has_previous_year'].mean() * 100:.1f}% of rows; three-year history is available for {df['has_three_year_history'].mean() * 100:.1f}%.",
        "",
        "## Data-quality and modeling signals",
        "",
        "Unavailable lag and rolling values are encoded as zero, while the history indicator columns preserve whether prior observations existed.",
        f"There are **{unknown_total}** literal `unknown` values and {len(unknown_columns)} affected columns: {', '.join(unknown_columns.index)}.",
        "Numeric `unknown` values must be converted and imputed within the training split before modeling.",
        "The target is complete and should not be included as a predictor. Identifier and location text columns are retained for analysis but require encoding or exclusion before model fitting.",
        "Complete missingness and numeric summaries are available in `missing_values.csv` and `numeric_summary.csv`.",
        "",
        "## Figures",
        "",
        "![Target distribution](target_distribution.png)",
        "![PM2.5 mean by target](pm25_mean_by_target.png)",
        "![History availability](history_availability.png)",
        "![Feature correlation](feature_correlation.png)",
    ]
    (OUTPUT_DIR / "processed_eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_data()
    numeric_columns = [
        column for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and pd.to_numeric(df[column], errors="coerce").notna().sum() > 0
    ]
    _save_figures(df, numeric_columns)
    _write_report(df, numeric_columns)
    print(f"Processed EDA complete: {len(df):,} rows analysed")
    print(f"Report: {OUTPUT_DIR / 'processed_eda_report.md'}")


if __name__ == "__main__":
    main()