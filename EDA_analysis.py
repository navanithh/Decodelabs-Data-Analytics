"""
Exploratory Data Analysis — Project 2 (DecodeLabs Data Analytics Internship)
=============================================================================
Dataset : E-commerce order records (1,200 rows x 14 columns)
Goal    : Understand patterns, trends and distributions; identify outliers;
          summarise key observations.

Pipeline
--------
    load_data()            -> read + prepare types
    descriptive_stats()    -> mean / median / count / five-number summary
    categorical_profile()  -> frequency distributions
    trend_analysis()       -> monthly & yearly aggregation, trend strength
    seasonality_check()    -> guards against the partial-year artifact
    detect_outliers()      -> IQR and Z-score, side by side
    correlation_analysis() -> Pearson matrix
    build_figures()        -> all charts to ./figures
    main()                 -> runs everything, prints a report

Usage
-----
    python eda_analysis.py [input_file] [output_dir]

    Accepts .xlsx or .csv. Defaults:
        input_file = Dataset_for_Data_Analytics_projeect_2.xlsx
        output_dir = .
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chisquare

NUMERIC_COLS = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
CATEGORICAL_COLS = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource", "CouponCode"]

ACCENT, WARN, GREY, DARK = "#2e7d8f", "#e07a3f", "#9aa5a8", "#1a3c40"


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """Read the orders file and normalise dtypes.

    Blank CouponCode means 'no coupon applied', not missing data, so it is
    filled with the explicit label 'None' rather than dropped.
    """
    path = Path(path)
    df = pd.read_excel(path) if path.suffix in {".xlsx", ".xlsm"} else pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["CouponCode"] = df["CouponCode"].fillna("None")
    for col in ["UnitPrice", "TotalPrice"]:
        df[col] = pd.to_numeric(df[col]).round(2)
    for col in ["Quantity", "ItemsInCart"]:
        df[col] = pd.to_numeric(df[col]).astype(int)

    # Derived time columns used throughout
    df["Year"] = df["Date"].dt.year
    df["MonthNum"] = df["Date"].dt.month
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df


# --------------------------------------------------------------------------
# 2. Descriptive statistics
# --------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Five-number summary plus mean and skew for every numeric column."""
    stats = df[NUMERIC_COLS].describe().T
    stats["median"] = df[NUMERIC_COLS].median()
    stats["skew"] = df[NUMERIC_COLS].skew()
    return stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max", "skew"]].round(2)


def categorical_profile(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Frequency and percentage share for each categorical column."""
    out = {}
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts()
        out[col] = pd.DataFrame({"count": counts, "pct": (counts / len(df) * 100).round(1)})
    return out


# --------------------------------------------------------------------------
# 3. Trends
# --------------------------------------------------------------------------
def trend_analysis(df: pd.DataFrame) -> dict:
    """Aggregate by month and year, and measure how strong the trend is.

    r^2 is reported because a visible slope means little on its own — it is
    the share of variance explained that says whether the trend is real.
    """
    monthly = df.groupby("Month").agg(
        Orders=("OrderID", "count"), Revenue=("TotalPrice", "sum"), AOV=("TotalPrice", "mean")
    )
    yearly = df.groupby("Year").agg(
        Orders=("OrderID", "count"), Revenue=("TotalPrice", "sum"), AOV=("TotalPrice", "mean")
    )

    x = np.arange(len(monthly))
    slope, intercept = np.polyfit(x, monthly["Revenue"], 1)
    r = float(np.corrcoef(x, monthly["Revenue"])[0, 1])

    return {
        "monthly": monthly.round(2),
        "yearly": yearly.round(2),
        "slope_per_month": round(slope, 2),
        "pearson_r": round(r, 3),
        "r_squared": round(r ** 2, 3),
    }


def seasonality_check(df: pd.DataFrame) -> dict:
    """Test for monthly seasonality, correcting for unequal year coverage.

    The dataset ends mid-2025, so Jan-Jun is observed in three years while
    Jul-Dec is observed in two. Raw monthly totals therefore overstate H1.
    Normalising by the number of observed years removes the artifact; the
    chi-square test then runs on complete years only.
    """
    pivot = df.pivot_table(index="MonthNum", columns="Year", values="OrderID", aggfunc="count", fill_value=0)
    years_observed = (pivot > 0).sum(axis=1)
    normalised = (pivot.sum(axis=1) / years_observed).round(1)

    complete_years = [y for y in df["Year"].unique() if (df["Year"] == y).sum() and years_observed.min() > 0]
    full = df[df["Year"].isin(sorted(df["Year"].unique())[:-1])]  # drop the partial final year
    observed = full.groupby("MonthNum")["OrderID"].count().values
    chi2, p_value = chisquare(observed)

    return {
        "raw_by_month": pivot.sum(axis=1),
        "years_observed": years_observed,
        "normalised": normalised,
        "h1_per_year": round(normalised.loc[1:6].mean(), 1),
        "h2_per_year": round(normalised.loc[7:12].mean(), 1),
        "chi2": round(chi2, 2),
        "p_value": round(p_value, 3),
        "seasonal": bool(p_value < 0.05),
    }


# --------------------------------------------------------------------------
# 4. Outliers
# --------------------------------------------------------------------------
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag outliers with both IQR (1.5x) and Z-score (|z|>3).

    The two methods are reported together because they disagree on bounded
    business data: IQR is robust and stricter, Z-score is pulled outward by
    the very values it is meant to catch.
    """
    rows = []
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_n = int(((df[col] < lower) | (df[col] > upper)).sum())

        z = (df[col] - df[col].mean()) / df[col].std()
        z_n = int((z.abs() > 3).sum())

        rows.append(
            {
                "Column": col,
                "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
                "LowerFence": round(lower, 2), "UpperFence": round(upper, 2),
                "IQR_Outliers": iqr_n, "IQR_Pct": f"{iqr_n / len(df) * 100:.2f}%",
                "Z_Outliers": z_n, "MaxAbsZ": round(z.abs().max(), 2),
            }
        )
    return pd.DataFrame(rows)


def validate_outliers(df: pd.DataFrame) -> dict:
    """Decide whether flagged outliers are noise (errors) or signal (real).

    An outlier is only an error if it breaks a rule. Here the rule is the
    line-total identity, so any row satisfying it is a genuine large order.
    """
    q1, q3 = df["TotalPrice"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    flagged = df[df["TotalPrice"] > upper]

    arithmetic_errors = int((abs(df["Quantity"] * df["UnitPrice"] - df["TotalPrice"]) > 0.02).sum())
    theoretical_max = df["Quantity"].max() * df["UnitPrice"].max()

    return {
        "upper_fence": round(upper, 2),
        "n_flagged": len(flagged),
        "arithmetic_errors": arithmetic_errors,
        "theoretical_max": round(theoretical_max, 2),
        "observed_max": round(df["TotalPrice"].max(), 2),
        "verdict": "SIGNAL — legitimate high-value orders" if arithmetic_errors == 0 else "NOISE — data errors present",
        "flagged_orders": flagged.nlargest(10, "TotalPrice")[
            ["OrderID", "Date", "Product", "Quantity", "UnitPrice", "TotalPrice", "OrderStatus"]
        ],
    }


# --------------------------------------------------------------------------
# 5. Correlation
# --------------------------------------------------------------------------
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC_COLS].corr().round(3)


# --------------------------------------------------------------------------
# 6. Figures
# --------------------------------------------------------------------------
def build_figures(df: pd.DataFrame, outdir: Path) -> None:
    """Write all six EDA charts as PNGs."""
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.titleweight": "bold"})

    # 1 — distribution shape
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["TotalPrice"], bins=40, color=ACCENT, edgecolor="white")
    ax[0].axvline(df["TotalPrice"].mean(), color=WARN, ls="--", lw=2, label=f"Mean {df['TotalPrice'].mean():.0f}")
    ax[0].axvline(df["TotalPrice"].median(), color=DARK, lw=2, label=f"Median {df['TotalPrice'].median():.0f}")
    ax[0].set_title(f"Order Value is Right-Skewed (skew = {df['TotalPrice'].skew():.2f})")
    ax[0].set_xlabel("Total Price"); ax[0].set_ylabel("Orders"); ax[0].legend()
    ax[1].hist(df["UnitPrice"], bins=40, color=GREY, edgecolor="white")
    ax[1].axvline(df["UnitPrice"].mean(), color=WARN, ls="--", lw=2, label=f"Mean {df['UnitPrice'].mean():.0f}")
    ax[1].axvline(df["UnitPrice"].median(), color=DARK, lw=2, label=f"Median {df['UnitPrice'].median():.0f}")
    ax[1].set_title(f"Unit Price is Symmetric (skew = {df['UnitPrice'].skew():.2f})")
    ax[1].set_xlabel("Unit Price"); ax[1].legend()
    plt.tight_layout(); plt.savefig(outdir / "fig1_distributions.png", bbox_inches="tight"); plt.close()

    # 2 — trend
    monthly = df.groupby("Month").agg(Revenue=("TotalPrice", "sum"))
    x = np.arange(len(monthly))
    slope, intercept = np.polyfit(x, monthly["Revenue"], 1)
    r2 = np.corrcoef(x, monthly["Revenue"])[0, 1] ** 2
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly.index, monthly["Revenue"], marker="o", ms=4, color=ACCENT, lw=1.5, label="Monthly revenue")
    ax.plot(monthly.index, slope * x + intercept, ls="--", color=WARN, lw=2, label=f"Trend (r² = {r2:.2f}, weak)")
    ax.set_title("Monthly Revenue Shows No Reliable Trend"); ax.set_ylabel("Revenue"); ax.legend()
    plt.tight_layout(); plt.savefig(outdir / "fig2_trend.png", bbox_inches="tight"); plt.close()

    # 3 — the seasonality artifact
    pivot = df.pivot_table(index="MonthNum", columns="Year", values="OrderID", aggfunc="count", fill_value=0)
    normalised = pivot.sum(axis=1) / (pivot > 0).sum(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(pivot.index, pivot.sum(axis=1), color=[WARN if i <= 6 else GREY for i in pivot.index])
    ax[0].set_title("RAW totals: looks like H1 seasonality")
    ax[0].set_xlabel("Calendar month"); ax[0].set_ylabel("Total orders")
    ax[1].bar(normalised.index, normalised.values, color=[WARN if i <= 6 else GREY for i in normalised.index])
    ax[1].axhline(normalised.mean(), color=DARK, ls="--", lw=2, label=f"{normalised.mean():.1f} orders/mo")
    ax[1].set_title("NORMALIZED per observed year: flat")
    ax[1].set_xlabel("Calendar month"); ax[1].set_ylabel("Orders per year"); ax[1].legend()
    plt.tight_layout(); plt.savefig(outdir / "fig3_seasonality_artifact.png", bbox_inches="tight"); plt.close()

    # 4 — outliers
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bp = ax[0].boxplot([df["TotalPrice"]], vert=False, widths=0.6, patch_artist=True,
                       flierprops=dict(marker="o", ms=5, mfc=WARN, mec=WARN))
    bp["boxes"][0].set_facecolor("#d7e8ec"); bp["medians"][0].set_color(DARK)
    ax[0].set_title("TotalPrice: high-value outliers (IQR)"); ax[0].set_yticks([]); ax[0].set_xlabel("Total Price")
    order = df.groupby("Product")["TotalPrice"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="TotalPrice", y="Product", order=order, ax=ax[1], color="#d7e8ec", fliersize=3)
    ax[1].set_title("Order Value by Product: near-identical spread")
    ax[1].set_xlabel("Total Price"); ax[1].set_ylabel("")
    plt.tight_layout(); plt.savefig(outdir / "fig4_outliers.png", bbox_inches="tight"); plt.close()

    # 5 — correlation
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[NUMERIC_COLS].corr(), annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation: UnitPrice is the Strongest Revenue Driver")
    plt.tight_layout(); plt.savefig(outdir / "fig5_correlation.png", bbox_inches="tight"); plt.close()

    # 6 — categorical uniformity
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax_, col in zip(axes.flat, ["Product", "PaymentMethod", "OrderStatus", "ReferralSource"]):
        counts = df[col].value_counts()
        ax_.bar(range(len(counts)), counts.values, color=ACCENT)
        ax_.axhline(len(df) / len(counts), color=WARN, ls="--", lw=1.5)
        ax_.set_xticks(range(len(counts)))
        ax_.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
        ax_.set_title(f"{col} (dashed = perfectly uniform)", fontsize=10)
    plt.tight_layout(); plt.savefig(outdir / "fig6_categorical.png", bbox_inches="tight"); plt.close()


# --------------------------------------------------------------------------
# 7. Orchestration
# --------------------------------------------------------------------------
def main(input_file: str, output_dir: str) -> None:
    outdir = Path(output_dir)
    df = load_data(input_file)

    print("=" * 74)
    print("EXPLORATORY DATA ANALYSIS — E-COMMERCE ORDERS")
    print("=" * 74)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("\n[1] DESCRIPTIVE STATISTICS")
    print(descriptive_stats(df).to_string())

    print("\n[2] CATEGORICAL DISTRIBUTIONS")
    for col, table in categorical_profile(df).items():
        print(f"\n-- {col} --")
        print(table.to_string())

    print("\n[3] TREND ANALYSIS")
    trend = trend_analysis(df)
    print(trend["yearly"].to_string())
    print(f"\nMonthly revenue slope: {trend['slope_per_month']:+,.2f}/month")
    print(f"Pearson r vs time: {trend['pearson_r']}   r² = {trend['r_squared']}")
    print("Interpretation: r² below 0.10 means time explains <10% of variance — no reliable trend.")

    print("\n[4] SEASONALITY CHECK (partial-year corrected)")
    seas = seasonality_check(df)
    print(f"H1 orders/year: {seas['h1_per_year']}   H2 orders/year: {seas['h2_per_year']}")
    print(f"Chi-square (complete years): chi2 = {seas['chi2']}, p = {seas['p_value']}")
    print(f"Seasonal pattern detected: {seas['seasonal']}")

    print("\n[5] OUTLIER DETECTION")
    print(detect_outliers(df).to_string(index=False))
    validation = validate_outliers(df)
    print(f"\nFlagged by IQR above {validation['upper_fence']}: {validation['n_flagged']} orders")
    print(f"Rows failing TotalPrice = Quantity x UnitPrice: {validation['arithmetic_errors']}")
    print(f"Theoretical max order: {validation['theoretical_max']}   Observed max: {validation['observed_max']}")
    print(f"VERDICT: {validation['verdict']}")

    print("\n[6] CORRELATION MATRIX")
    print(correlation_analysis(df).to_string())

    build_figures(df, outdir / "figures")
    print(f"\nFigures written to {outdir / 'figures'}")
    print("=" * 74)


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "Dataset_for_Data_Analytics_projeect_2.xlsx"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    main(infile, outdir)
