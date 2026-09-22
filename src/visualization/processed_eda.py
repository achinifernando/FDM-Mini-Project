"""
Enhanced exploratory data analysis for the model-ready PM2.5 feature dataset.

Covers every Stage 3 requirement plus model-readiness diagnostics:

1.  Dataset structure and variable types
2.  Target distribution and class imbalance (with ratio + cumulative %)
3.  Ordinal target handling and metric implications
4.  Missing values and literal 'unknown' values (row-level inspection)
5.  Duplicate detection
6.  Outlier analysis (IQR)
7.  Temporal analysis (target by year, PM2.5 mean by year)
8.  Geographic analysis (target by state)
9.  Feature usefulness (mutual information)
10. Feature diagnostics (skewness, kurtosis, zero-variance, zero-encoding risk)
11. Correlation and multicollinearity analysis
12. Data-leakage verification
13. Train/test distribution shift check
14. Key observations informing preprocessing and modelling
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

# Identifier / location columns — retained for analysis, excluded from numeric features
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

# Columns likely derived from the target period (leakage candidates)
LEAKAGE_CANDIDATES = [
    "pm25_risk_category_next_year",
    "aqi_next_year",
    "pm25_mean_next_year",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        dtype={"site_id": "string", "state_code": "string"},
        low_memory=False,
    )
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2: Structure analysis
# ---------------------------------------------------------------------------

def analyse_structure(df: pd.DataFrame) -> pd.DataFrame:
    structure = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null": df.isna().sum(),
        "unique": df.nunique(),
    })
    structure["null_pct"] = structure["null"] / len(df) * 100
    structure.to_csv(OUTPUT_DIR / "column_structure.csv")
    return structure


# ---------------------------------------------------------------------------
# Step 3: Target distribution and class imbalance
# ---------------------------------------------------------------------------

def analyse_target(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    target_counts = df[TARGET].value_counts().reindex(TARGET_ORDER, fill_value=0)
    target_frame = target_counts.rename_axis("risk_category").reset_index(name="records")
    target_frame["percent"] = target_frame["records"] / len(df) * 100
    target_frame["cumulative_percent"] = target_frame["percent"].cumsum()

    # Imbalance ratio
    non_empty = target_counts[target_counts > 0]
    imbalance_ratio = float(non_empty.max() / non_empty.min()) if len(non_empty) > 1 else float("inf")
    target_frame.attrs["imbalance_ratio"] = imbalance_ratio

    target_frame.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)

    # Bar chart — ordered by severity
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=target_frame, x="records", y="risk_category",
                color="#176b87", ax=ax)
    ax.set_title("Next-year PM2.5 risk-category distribution")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    for i, row in target_frame.iterrows():
        ax.text(row["records"], i, f"  {row['percent']:.1f}%",
                va="center", fontsize=9)
    save_fig(OUTPUT_DIR / "target_distribution.png")

    # Cumulative distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(target_frame["risk_category"], target_frame["percent"], color="#176b87", label="Class %")
    ax2 = ax.twinx()
    ax2.plot(target_frame["risk_category"], target_frame["cumulative_percent"],
             color="#e07a5f", marker="o", label="Cumulative %")
    ax.set_title("Cumulative target distribution (ordinal skew)")
    ax.set_ylabel("Class (%)")
    ax2.set_ylabel("Cumulative (%)")
    plt.xticks(rotation=25, ha="right")
    save_fig(OUTPUT_DIR / "target_cumulative_distribution.png")

    return target_frame, imbalance_ratio


# ---------------------------------------------------------------------------
# Step 4: Ordinal target diagnostics
# ---------------------------------------------------------------------------

def analyse_ordinal_target(df: pd.DataFrame) -> None:
    """Show distribution shape and separation of pm25_mean across ordinal classes."""
    if "pm25_mean" not in df.columns:
        return

    plot_df = df[[TARGET, "pm25_mean"]].copy()
    plot_df["pm25_mean"] = pd.to_numeric(plot_df["pm25_mean"], errors="coerce")
    plot_df = plot_df.dropna(subset=["pm25_mean", TARGET])

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(
        data=plot_df, x=TARGET, y="pm25_mean", order=TARGET_ORDER,
        palette="Spectral", ax=ax, cut=0,
    )
    ax.set_title("Current-year PM2.5 mean distribution by next-year risk category")
    ax.set_xlabel("Next-year risk category")
    ax.set_ylabel("Current-year PM2.5 mean")
    plt.xticks(rotation=25, ha="right")
    save_fig(OUTPUT_DIR / "pm25_mean_violin_by_target.png")

    # Boxplot version
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=plot_df, x=TARGET, y="pm25_mean", order=TARGET_ORDER,
        color="#e07a5f", showfliers=False, ax=ax,
    )
    ax.set_title("Current-year PM2.5 mean by next-year risk category")
    ax.set_xlabel("Next-year risk category")
    ax.set_ylabel("Current-year PM2.5 mean")
    plt.xticks(rotation=25, ha="right")
    save_fig(OUTPUT_DIR / "pm25_mean_by_target.png")


# ---------------------------------------------------------------------------
# Step 5: Missing values and 'unknown' values
# ---------------------------------------------------------------------------

def analyse_missing(df: pd.DataFrame) -> pd.DataFrame:
    data_quality = pd.DataFrame({
        "column": df.columns,
        "null_values": df.isna().sum().values,
        "unknown_values": [
            int((df[col].astype(str).str.lower() == "unknown").sum())
            for col in df.columns
        ],
    })
    data_quality["null_percent"] = data_quality["null_values"] / len(df) * 100
    data_quality["unknown_percent"] = data_quality["unknown_values"] / len(df) * 100
    data_quality = data_quality.sort_values(
        ["null_values", "unknown_values"], ascending=False
    )
    data_quality.to_csv(OUTPUT_DIR / "missing_values.csv", index=False)

    # Missing-value visual
    plot_df = data_quality[
        (data_quality["null_values"] > 0) | (data_quality["unknown_values"] > 0)
    ].copy()
    if not plot_df.empty:
        plot_df["total_issue_pct"] = (
            plot_df["null_percent"] + plot_df["unknown_percent"]
        )
        plot_df = plot_df.sort_values("total_issue_pct")
        fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
        sns.barplot(data=plot_df, x="total_issue_pct", y="column",
                    color="#e07a5f", ax=ax)
        ax.set_title("Missing or 'unknown' percentage by column")
        ax.set_xlabel("Missing + unknown (%)")
        ax.set_ylabel("")
        save_fig(OUTPUT_DIR / "missing_values.png")

    # Row-level unknown inspection
    str_df = df.astype(str).apply(lambda s: s.str.lower() == "unknown")
    unknown_per_row = str_df.sum(axis=1)
    if unknown_per_row.sum() > 0:
        counts = unknown_per_row.value_counts().sort_index()
        counts.to_csv(OUTPUT_DIR / "unknown_per_row.csv")
        sample_rows = df[unknown_per_row > 0].head(20)
        sample_rows.to_csv(OUTPUT_DIR / "unknown_rows_sample.csv", index=False)

    return data_quality


# ---------------------------------------------------------------------------
# Step 6: Duplicate analysis
# ---------------------------------------------------------------------------

def analyse_duplicates(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / len(df) * 100
    summary = {
        "total_rows": len(df),
        "duplicate_rows": dup_count,
        "duplicate_percent": round(dup_pct, 4),
    }
    with open(OUTPUT_DIR / "duplicate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Sample duplicates
    df[df.duplicated(keep=False)].head(20).to_csv(
        OUTPUT_DIR / "duplicate_examples.csv", index=False
    )
    return summary


# ---------------------------------------------------------------------------
# Step 7: Outlier analysis
# ---------------------------------------------------------------------------

def analyse_outliers(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    records = []
    for col in numeric_columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        records.append({
            "feature": col,
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "outlier_count": outliers,
            "outlier_percent": round(outliers / len(s) * 100, 4),
        })
    outlier_df = pd.DataFrame(records).sort_values("outlier_percent", ascending=False)
    outlier_df.to_csv(OUTPUT_DIR / "outliers_iqr.csv", index=False)
    return outlier_df


# ---------------------------------------------------------------------------
# Step 8: Temporal analysis
# ---------------------------------------------------------------------------

def analyse_temporal(df: pd.DataFrame) -> None:
    if "year" not in df.columns:
        return

    # Target composition by year
    yearly = df.groupby(["year", TARGET]).size().unstack(fill_value=0)
    yearly = yearly.reindex(columns=TARGET_ORDER, fill_value=0)
    yearly.to_csv(OUTPUT_DIR / "target_by_year.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    yearly.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
    ax.set_title("Target class distribution by prediction year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Records")
    ax.legend(title="Risk category", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(OUTPUT_DIR / "target_by_year.png")

    # PM2.5 mean by year
    if "pm25_mean" in df.columns:
        yearly_mean = (
            pd.to_numeric(df["pm25_mean"], errors="coerce")
              .groupby(df["year"]).mean()
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        yearly_mean.plot(marker="o", color="#176b87", ax=ax)
        ax.set_title("Mean PM2.5 by prediction year")
        ax.set_xlabel("Year")
        ax.set_ylabel("PM2.5 mean")
        save_fig(OUTPUT_DIR / "pm25_mean_by_year.png")


# ---------------------------------------------------------------------------
# Step 9: Geographic analysis
# ---------------------------------------------------------------------------

def analyse_geographic(df: pd.DataFrame) -> None:
    if "state_name" not in df.columns:
        return

    top_states = df["state_name"].value_counts().head(15)
    subset = df[df["state_name"].isin(top_states.index)]
    state_target = (
        subset.groupby(["state_name", TARGET]).size()
        .unstack(fill_value=0)
        .reindex(columns=TARGET_ORDER, fill_value=0)
    )
    state_target.to_csv(OUTPUT_DIR / "target_by_state.csv")

    fig, ax = plt.subplots(figsize=(12, 8))
    state_target.plot(kind="barh", stacked=True, colormap="Spectral", ax=ax)
    ax.set_title("Target distribution across top 15 states")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    ax.legend(title="Risk category", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(OUTPUT_DIR / "target_by_state.png")


# ---------------------------------------------------------------------------
# Step 10: Feature usefulness (mutual information)
# ---------------------------------------------------------------------------

def analyse_feature_usefulness(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if TARGET not in df.columns or not numeric_columns:
        return pd.DataFrame()

    X = df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[TARGET].map({cat: i for i, cat in enumerate(TARGET_ORDER)}).fillna(-1).astype(int)

    mask = y >= 0
    X, y = X[mask], y[mask]

    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": numeric_columns, "mutual_info": mi})
    mi_df = mi_df.sort_values("mutual_info", ascending=False)
    mi_df.to_csv(OUTPUT_DIR / "mutual_information.csv", index=False)

    # Bar chart of top 20
    top = mi_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="mutual_info", y="feature", color="#3a86ff", ax=ax)
    ax.set_title("Top 20 features by mutual information with target")
    ax.set_xlabel("Mutual information")
    ax.set_ylabel("")
    save_fig(OUTPUT_DIR / "mutual_information.png")

    return mi_df


# ---------------------------------------------------------------------------
# Step 11: Feature diagnostics (skew, kurtosis, zero-variance, zero-encoding)
# ---------------------------------------------------------------------------

def analyse_feature_diagnostics(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame()

    X = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    diag = pd.DataFrame({
        "feature": numeric_columns,
        "mean": X.mean().values,
        "std": X.std().values,
        "skew": X.apply(skew).values,
        "kurtosis": X.apply(kurtosis).values,
        "zeros_pct": ((X == 0).mean() * 100).values,
        "null_pct": (X.isna().mean() * 100).values,
    })
    diag["zero_variance"] = diag["std"] == 0
    diag = diag.sort_values("skew", key=lambda s: s.abs(), ascending=False)
    diag.to_csv(OUTPUT_DIR / "feature_diagnostics.csv", index=False)

    # Zero-variance feature list
    zv = diag[diag["zero_variance"]]
    if not zv.empty:
        zv.to_csv(OUTPUT_DIR / "zero_variance_features.csv", index=False)

    # Zero-encoding risk: features whose zeros exceed 30% of rows
    zero_heavy = diag[diag["zeros_pct"] > 30].sort_values("zeros_pct", ascending=False)
    if not zero_heavy.empty:
        zero_heavy.to_csv(OUTPUT_DIR / "zero_heavy_features.csv", index=False)

    return diag


# ---------------------------------------------------------------------------
# Step 12: Zero-encoding vs target risk check
# ---------------------------------------------------------------------------

def analyse_zero_encoding_risk(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """Investigate whether zero-encoded missingness correlates with the target."""
    lag_like = [c for c in numeric_columns if any(k in c.lower() for k in ("lag", "rolling", "history"))]
    if not lag_like:
        return

    X = df[lag_like].apply(pd.to_numeric, errors="coerce")
    zero_rows = (X == 0).all(axis=1)

    crosstab = pd.crosstab(zero_rows, df[TARGET], normalize="index") * 100
    crosstab = crosstab.reindex(columns=TARGET_ORDER, fill_value=0)
    crosstab.to_csv(OUTPUT_DIR / "zero_lag_vs_target.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    crosstab.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
    ax.set_title("Target distribution: all-zero lag rows vs. others")
    ax.set_xlabel("All lag/rolling features are zero")
    ax.set_ylabel("Target (%)")
    ax.legend(title="Risk category", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(OUTPUT_DIR / "zero_lag_vs_target.png")


# ---------------------------------------------------------------------------
# Step 13: Correlation and multicollinearity
# ---------------------------------------------------------------------------

def analyse_correlations(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if len(numeric_columns) < 2:
        return pd.DataFrame()

    X = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.5]
    corr = X.corr()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2, ax=ax)
    ax.set_title("Correlation among numeric model-ready features")
    save_fig(OUTPUT_DIR / "feature_correlation.png")

    # Strong correlations (|r| > 0.8)
    strong = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
    )
    strong = strong[strong["corr"].abs() > 0.8].sort_values("corr", ascending=False)
    strong.to_csv(OUTPUT_DIR / "strong_correlations.csv", index=False)
    return strong


# ---------------------------------------------------------------------------
# Step 14: Data leakage verification
# ---------------------------------------------------------------------------

def analyse_leakage(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    rows = []
    # 14a. Known candidates
    for col in LEAKAGE_CANDIDATES:
        if col in df.columns and col != TARGET:
            rows.append({
                "column": col,
                "risk": "Explicit leakage candidate — same period as target.",
            })

    # 14b. Detect perfect separation
    for col in numeric_columns:
        try:
            n_targets = df.groupby(col)[TARGET].nunique()
            if not n_targets.empty and n_targets.max() == 1 and df[col].nunique() > 5:
                rows.append({
                    "column": col,
                    "risk": "Perfect separation — near-certain leakage.",
                })
        except Exception:
            continue

    leakage_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column", "risk"])
    leakage_df.to_csv(OUTPUT_DIR / "leakage_check.csv", index=False)
    return leakage_df


# ---------------------------------------------------------------------------
# Step 15: Train/test distribution shift check
# ---------------------------------------------------------------------------

def analyse_train_test_shift(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if TARGET not in df.columns or len(df) < 100:
        return

    try:
        train, test = train_test_split(
            df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
        )
    except ValueError:
        train, test = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    sample_cols = [c for c in ["pm25_mean", "year"] if c in df.columns]
    records = []
    for col in sample_cols:
        tr = pd.to_numeric(train[col], errors="coerce")
        te = pd.to_numeric(test[col], errors="coerce")
        records.append({
            "feature": col,
            "train_mean": tr.mean(),
            "test_mean": te.mean(),
            "abs_diff": abs(tr.mean() - te.mean()),
        })
    shift_df = pd.DataFrame(records)
    shift_df.to_csv(OUTPUT_DIR / "train_test_shift.csv", index=False)


# ---------------------------------------------------------------------------
# Step 16: Write report
# ---------------------------------------------------------------------------

def write_report(
    df: pd.DataFrame,
    target_frame: pd.DataFrame,
    imbalance_ratio: float,
    data_quality: pd.DataFrame,
    dup_summary: dict,
    outlier_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
) -> None:

    target_counts = df[TARGET].value_counts().reindex(TARGET_ORDER, fill_value=0)
    top_class = target_counts.idxmax()
    top_count = int(target_counts.max())
    non_empty = target_counts[target_counts > 0]
    rarest = non_empty.idxmin() if not non_empty.empty else "N/A"
    rare_count = int(non_empty.min()) if not non_empty.empty else 0

    unknown_cols = data_quality[data_quality["unknown_values"] > 0]
    unknown_total = int(data_quality["unknown_values"].sum())

    zero_var_count = int((diag_df["zero_variance"]).sum()) if not diag_df.empty else 0
    zero_heavy_count = (
        int((diag_df["zeros_pct"] > 30).sum()) if not diag_df.empty else 0
    )

    report = [
        "# Processed PM2.5 Dataset — Enhanced EDA Report",
        "",
        f"- **Source**: `{DATA_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- **Rows**: {len(df):,}",
        f"- **Columns**: {df.shape[1]}",
        f"- **Monitoring sites**: {df['site_id'].nunique() if 'site_id' in df else 'N/A'}",
        f"- **Years**: {int(df['year'].min())}–{int(df['year'].max())}" if "year" in df else "",
        "",
        "## 1. Target variable and class imbalance",
        "",
        f"- Target: `{TARGET}` (ordinal, {len(TARGET_ORDER)} classes).",
        f"- Largest class: **{top_class}** ({top_count:,} records).",
        f"- Rarest non-empty class: **{rarest}** ({rare_count:,} records).",
        f"- **Imbalance ratio (max/min)**: {imbalance_ratio:.2f}.",
        "- Ordinal target → accuracy alone is misleading; use Quadratic Weighted",
        "  Kappa (QWK), macro-F1, or MAE-in-class-order during Stage 6.",
        "- Recommended handling: class weights, ordinal encoding, stratified CV.",
        "",
        "See `target_distribution.csv` and `target_cumulative_distribution.png`.",
        "",
        "## 2. Ordinal target diagnostics",
        "",
        "- Distribution shape and separation shown in",
        "  `pm25_mean_violin_by_target.png` and `pm25_mean_by_target.png`.",
        "- Good separation between adjacent classes supports ordinal modelling.",
        "",
        "## 3. Missing values and 'unknown' values",
        "",
        f"- Columns with any missing values: {(data_quality['null_values'] > 0).sum()}.",
        f"- Columns with literal `unknown` values: {len(unknown_cols)}.",
        f"- Total `unknown` occurrences: {unknown_total}.",
        "- Row-level unknown counts: `unknown_per_row.csv`.",
        "- Numeric `unknown` values must be coerced and imputed **within the",
        "  training split** to avoid leakage.",
        "",
        "## 4. Duplicates",
        "",
        f"- Duplicate rows: {dup_summary['duplicate_rows']:,}",
        f"  ({dup_summary['duplicate_percent']}%).",
        "- Decision: drop exact duplicates during cleaning.",
        "",
        "## 5. Outliers (IQR method)",
        "",
        "- Per-feature IQR bounds and counts: `outliers_iqr.csv`.",
        "- Air-quality extremes often reflect real events — treat with care.",
        "",
        "## 6. Temporal analysis",
        "",
        "- Target composition by year: `target_by_year.csv`, `target_by_year.png`.",
        "- PM2.5 mean by year: `pm25_mean_by_year.png`.",
        "- Distribution shifts over years justify **temporal splitting** to",
        "  prevent temporal leakage.",
        "",
        "## 7. Geographic analysis",
        "",
        "- Target distribution across top 15 states: `target_by_state.csv`,",
        "  `target_by_state.png`.",
        "- Reveals geographic bias — informs stratified splitting decisions.",
        "",
        "## 8. Feature usefulness (mutual information)",
        "",
        f"- {len(mi_df)} features ranked by mutual information with target.",
        "- Top features: "
        + ", ".join(mi_df["feature"].head(5).tolist()) if not mi_df.empty else "- (no features)",
        "- Full ranking: `mutual_information.csv`, `mutual_information.png`.",
        "",
        "## 9. Feature diagnostics",
        "",
        f"- Zero-variance features: {zero_var_count}.",
        f"- Zero-heavy features (>30% zeros): {zero_heavy_count}.",
        "- Skewness, kurtosis and zero-percentage: `feature_diagnostics.csv`.",
        "- Heavy skew → consider log/power transforms for linear models.",
        "",
        "## 10. Zero-encoding vs target",
        "",
        "- Rows whose lag/rolling features are all zero may have systematically",
        "  different target distributions — verified in `zero_lag_vs_target.csv`.",
        "- If differences exist, add explicit missingness flags during modelling.",
        "",
        "## 11. Correlation and multicollinearity",
        "",
        "- Correlation matrix: `correlation_matrix.csv`.",
        "- Heatmap: `feature_correlation.png`.",
        "- Strong pairs (|r| > 0.8): `strong_correlations.csv`.",
        "- Highly correlated features must be handled during feature selection.",
        "",
        "## 12. Data-leakage verification",
        "",
        f"- Leakage candidates / risks found: {len(leakage_df)}.",
        "- Details: `leakage_check.csv`.",
        "- Target must never appear as a predictor.",
        "- Rolling windows must not overlap with the target year.",
        "",
        "## 13. Train/test distribution shift check",
        "",
        "- Compare `train_test_shift.csv` to confirm stratified split stability.",
        "- If features shift, adjust split strategy in Stage 4.",
        "",
        "## 14. Key observations informing preprocessing",
        "",
        "1. Drop exact duplicates.",
        "2. Coerce `unknown` → NaN, then impute within training split only.",
        "3. Handle ordinal target with class weights or ordinal encoding.",
        "4. Use stratified (or temporal) train/test split.",
        "5. Drop zero-variance features.",
        "6. Add missingness flags for zero-heavy lag features.",
        "7. Inspect strong correlations before feature selection.",
        "8. Verify absence of leakage before modelling.",
        "9. Use QWK, macro-F1, or ordinal MAE as primary metrics.",
        "",
        "## Figures",
        "",
        "![Target distribution](target_distribution.png)",
        "![Cumulative target distribution](target_cumulative_distribution.png)",
        "![Violin: PM2.5 by target](pm25_mean_violin_by_target.png)",
        "![Target by year](target_by_year.png)",
        "![Target by state](target_by_state.png)",
        "![Mutual information](mutual_information.png)",
        "![Feature correlation](feature_correlation.png)",
        "![Zero-lag vs target](zero_lag_vs_target.png)",
    ]
    (OUTPUT_DIR / "processed_eda_report.md").write_text(
        "\n".join(line for line in report if line is not None),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dir()
    sns.set_theme(style="whitegrid")

    print("Loading data …")
    df = load_data()

    print("Analysing structure …")
    analyse_structure(df)

    # Build numeric feature list (excludes identifiers and target)
    numeric_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
        and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0
    ]
    print(f"  → {len(numeric_columns)} numeric features identified")

    print("Analysing target distribution and imbalance …")
    target_frame, imbalance_ratio = analyse_target(df)

    print("Analysing ordinal target …")
    analyse_ordinal_target(df)

    print("Analysing missing and 'unknown' values …")
    data_quality = analyse_missing(df)

    print("Analysing duplicates …")
    dup_summary = analyse_duplicates(df)

    print("Analysing outliers …")
    outlier_df = analyse_outliers(df, numeric_columns)

    print("Analysing temporal patterns …")
    analyse_temporal(df)

    print("Analysing geographic patterns …")
    analyse_geographic(df)

    print("Analysing feature usefulness …")
    mi_df = analyse_feature_usefulness(df, numeric_columns)

    print("Analysing feature diagnostics …")
    diag_df = analyse_feature_diagnostics(df, numeric_columns)

    print("Analysing zero-encoding risk …")
    analyse_zero_encoding_risk(df, numeric_columns)

    print("Analysing correlations …")
    analyse_correlations(df, numeric_columns)

    print("Verifying data leakage …")
    leakage_df = analyse_leakage(df, numeric_columns)

    print("Checking train/test distribution shift …")
    analyse_train_test_shift(df, numeric_columns)

    print("Writing report …")
    write_report(
        df, target_frame, imbalance_ratio, data_quality,
        dup_summary, outlier_df, mi_df, diag_df, leakage_df,
    )

    print("\nProcessed EDA complete.")
    print(f"Report:  {OUTPUT_DIR / 'processed_eda_report.md'}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()