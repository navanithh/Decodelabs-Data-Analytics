# Project 2 — Exploratory Data Analysis: E-Commerce Orders

DecodeLabs Data Analytics Internship (Batch 2026)

Exploratory analysis of 1,200 e-commerce order records spanning January 2023 to June 2025, covering descriptive statistics, trend and seasonality testing, outlier detection, and correlation analysis.

---

## Headline findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | Order value is right-skewed — use the median | Mean 1,053.97 vs median 823.62 (skew 0.89) |
| 2 | The "H1 sales spike" is a reporting artifact | 40.0 orders/month in both halves once normalised; chi-square p = 0.744 |
| 3 | Revenue has no reliable trend | r² = 0.088 — time explains under 9% of variance |
| 4 | The 8 flagged outliers are signal, not noise | All pass `TotalPrice = Quantity × UnitPrice`; all are max-quantity premium orders |
| 5 | Unit price drives revenue more than quantity | r = 0.717 vs 0.615; price/quantity uncorrelated at r = 0.015 |
| 6 | Every categorical variable is near-uniform | Product share spans only 13.0%–15.1% — indicates synthetic data |

**The most consequential finding is #2.** Raw monthly totals show 741 orders in January–June against 480 in July–December, which reads as strong first-half seasonality. The dataset ends on 30 June 2025, so the first half of the year is observed across three years and the second half across only two. Correcting for that gives exactly 40.0 orders per month in both halves. Acting on the uncorrected figures would have meant shifting budget toward a half-year with no real advantage.

---

## Repository contents

```
.
├── eda_analysis.py              # Modular analysis pipeline
├── EDA_Report_Project2.pdf      # Full report with figures and recommendations
├── orders_prepared.csv          # Prepared dataset used for analysis
├── figures/                     # Generated charts
│   ├── fig1_distributions.png           # Skew: TotalPrice vs UnitPrice
│   ├── fig2_trend.png                   # Monthly revenue with fitted trend
│   ├── fig3_seasonality_artifact.png    # Raw vs corrected seasonality
│   ├── fig4_outliers.png                # Boxplots and per-product spread
│   ├── fig5_correlation.png             # Pearson correlation heatmap
│   └── fig6_categorical.png             # Category frequencies vs uniform
└── README.md
```

---

## Running the analysis

Requires Python 3.9+.

```bash
pip install pandas numpy scipy matplotlib seaborn openpyxl

python eda_analysis.py <input_file> <output_dir>
```

Both arguments are optional. Accepts `.xlsx` or `.csv`:

```bash
# Defaults to the project dataset, writes figures to ./figures
python eda_analysis.py

# Explicit
python eda_analysis.py Dataset_for_Data_Analytics_projeect_2.xlsx ./output
```

The script prints the full statistical report to stdout and writes all six figures to `<output_dir>/figures/`.

---

## Method

**Pipeline** — `eda_analysis.py` is organised as independent functions so each stage can be run or tested alone:

| Function | Purpose |
|----------|---------|
| `load_data()` | Read file, normalise dtypes, derive time columns |
| `descriptive_stats()` | Five-number summary, mean, median, skew |
| `categorical_profile()` | Frequency distributions for all categorical fields |
| `trend_analysis()` | Monthly/yearly aggregation and trend strength (r²) |
| `seasonality_check()` | Chi-square test, corrected for unequal year coverage |
| `detect_outliers()` | IQR and Z-score side by side |
| `validate_outliers()` | Classifies flagged values as noise vs signal |
| `correlation_analysis()` | Pearson correlation matrix |
| `build_figures()` | Writes all charts |

**Handling missing values** — 309 blank `CouponCode` entries mean "no coupon applied", not missing data. They were labelled explicitly rather than imputed or dropped. No rows were deleted at any stage.

**Outlier detection** — Both methods were run so they could be compared. They disagree: IQR flags 8 orders in `TotalPrice`, Z-score flags none (max |z| = 2.93, just under the threshold, because the standard deviation is inflated by the very values being tested). On bounded business data the robust IQR method is more trustworthy. Every flagged value was then validated against the line-total identity before being judged.

**Trend strength** — Judged by r², not by the visual slope. A fitted line always has a slope; r² says whether it means anything.

---

## Limitations

- **2025 is a partial year** (January–June only). All year-on-year comparisons are asymmetric and are flagged wherever they appear.
- **`TotalPrice` is derived** from `Quantity × UnitPrice`, so their mutual correlations are partly structural rather than behavioural. The informative result is the near-zero price/quantity correlation.
- **The data is very likely synthetic.** Uniformity across all five categorical fields — including a 41% combined cancel-and-return rate spread evenly across statuses — is not characteristic of real transaction data. This dataset demonstrates method; it cannot support segmentation, forecasting, or customer analysis.

---

## Tools

Python · pandas · NumPy · SciPy · Matplotlib · seaborn · ReportLab
