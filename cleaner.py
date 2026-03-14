"""
cleaner.py
==========
Post-load data cleaning, validation, and quality reporting.

After loading raw data with data_loader.py, pass the DataFrame through
clean_dataset() to:
  1. Cap extreme income outliers
  2. Validate distributions match known benchmarks
  3. Generate a data quality report
  4. Save a publication-ready cleaned dataset

Usage:
    from src.data_loader import load_data
    from src.cleaner import clean_dataset, validate_dataset, quality_report

    df_raw = load_data(source="acs")
    df = clean_dataset(df_raw)
    validate_dataset(df)
    quality_report(df, save_path="results/tables/data_quality.csv")
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLEANING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataset(
    df: pd.DataFrame,
    income_floor: float = 1_000,
    income_cap_pct: float = 99.5,
    drop_missing_race: bool = False,
) -> pd.DataFrame:
    """
    Apply post-load cleaning steps.

    Parameters
    ----------
    df : Raw DataFrame from load_data()
    income_floor : Drop workers earning less than this annually (likely part-year).
    income_cap_pct : Winsorize income at this percentile (handles extreme outliers).
    drop_missing_race : If True, drop rows where race_ethnicity is 'Unknown'.

    Returns
    -------
    Cleaned pd.DataFrame
    """
    df = df.copy()
    n_start = len(df)
    log.info(f"Cleaning dataset: {n_start:,} rows")

    # ── Step 1: Income floor ─────────────────────────────────────────────────
    df = df[df["income"] >= income_floor].copy()
    log.info(f"  [Floor ${income_floor:,}] Dropped {n_start - len(df):,} rows → {len(df):,} remain")

    # ── Step 2: Winsorize income at top percentile ───────────────────────────
    cap = df["income"].quantile(income_cap_pct / 100)
    n_capped = (df["income"] > cap).sum()
    df["income"] = df["income"].clip(upper=cap)
    df["log_income"] = np.log1p(df["income"])
    log.info(f"  [Cap @p{income_cap_pct}] Capped {n_capped:,} values at ${cap:,.0f}")

    # ── Step 3: Hourly wage (if hours available) ─────────────────────────────
    if "hours_per_week" in df.columns and "weeks_worked" in df.columns:
        annual_hours = df["hours_per_week"] * df["weeks_worked"]
        df["hourly_wage"] = np.where(annual_hours > 0, df["income"] / annual_hours, np.nan)
        # Sanity: drop implausibly low hourly wages (below federal minimum wage / 2)
        df = df[(df["hourly_wage"].isna()) | (df["hourly_wage"] >= 3.6)].copy()
        log.info(f"  Added hourly_wage; removed implausible hourly wages → {len(df):,} rows")

    # ── Step 4: Drop unknown race (optional) ────────────────────────────────
    if drop_missing_race:
        before = len(df)
        df = df[df["race_ethnicity"] != "Unknown"].copy()
        log.info(f"  Dropped {before - len(df):,} unknown race rows")

    # ── Step 5: Encode education as integer (for models) ────────────────────
    educ_int_map = {"HS_or_less": 0, "Some_College": 1, "Bachelors": 2, "Graduate": 3}
    df["education_int"] = df["education"].map(educ_int_map)

    # ── Step 6: Dummy variables for key categoricals ─────────────────────────
    df["is_female"] = (df["sex"] == "Female").astype(int)
    df["is_college_plus"] = (df["education"] >= "Bachelors").astype(int)

    # ── Step 7: Interaction term ─────────────────────────────────────────────
    df["educ_x_age"] = df["education_int"].astype(float) * df["age"].astype(float)

    log.info(f"Cleaning complete. Final shape: {df.shape}")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Run sanity checks against known Census benchmarks.
    Logs warnings for anything suspicious.

    Returns True if all checks pass, False otherwise.
    """
    log.info("Running dataset validation...")
    passed = True

    checks = []

    # ── Check 1: Sample size ─────────────────────────────────────────────────
    n = len(df)
    if n < 10_000:
        log.warning(f"  ⚠ Small sample: {n:,} rows (expect 50k+ for ACS, 100k+ for CPS)")
        checks.append(("Sample size", "WARN", f"{n:,}"))
        passed = False
    else:
        log.info(f"  ✓ Sample size: {n:,}")
        checks.append(("Sample size", "OK", f"{n:,}"))

    # ── Check 2: Income distribution ─────────────────────────────────────────
    med = df["income"].median()
    if not (30_000 < med < 80_000):
        log.warning(f"  ⚠ Median income {med:,.0f} seems off (expect $40k–70k for US workers)")
        checks.append(("Median income", "WARN", f"${med:,.0f}"))
        passed = False
    else:
        log.info(f"  ✓ Median income: ${med:,.0f}")
        checks.append(("Median income", "OK", f"${med:,.0f}"))

    # ── Check 3: Education distribution ──────────────────────────────────────
    educ_pct = df["education"].value_counts(normalize=True) * 100
    # US adults 25–64 roughly: HS~27%, Some college~28%, Bach~24%, Grad~14%
    expected = {"HS_or_less": (20, 45), "Some_College": (15, 40),
                "Bachelors": (15, 35), "Graduate": (8, 25)}
    for cat, (lo, hi) in expected.items():
        pct = educ_pct.get(cat, 0)
        if not (lo <= pct <= hi):
            log.warning(f"  ⚠ Education '{cat}': {pct:.1f}% (expected {lo}–{hi}%)")
            checks.append((f"Educ: {cat}", "WARN", f"{pct:.1f}%"))
        else:
            log.info(f"  ✓ Education '{cat}': {pct:.1f}%")
            checks.append((f"Educ: {cat}", "OK", f"{pct:.1f}%"))

    # ── Check 4: Sex ratio ───────────────────────────────────────────────────
    female_pct = (df["sex"] == "Female").mean() * 100
    if not (40 <= female_pct <= 60):
        log.warning(f"  ⚠ Female %: {female_pct:.1f}% (expect ~46–54%)")
        checks.append(("Female %", "WARN", f"{female_pct:.1f}%"))
    else:
        log.info(f"  ✓ Female %: {female_pct:.1f}%")
        checks.append(("Female %", "OK", f"{female_pct:.1f}%"))

    # ── Check 5: Expected earnings premium direction ──────────────────────────
    medians = df.groupby("education", observed=True)["income"].median()
    if len(medians) >= 2:
        is_monotone = all(
            medians.iloc[i] <= medians.iloc[i + 1]
            for i in range(len(medians) - 1)
        )
        if not is_monotone:
            log.warning(f"  ⚠ Education-income relationship not monotone: {medians.to_dict()}")
            checks.append(("Educ-income monotone", "WARN", str(medians.to_dict())))
        else:
            log.info(f"  ✓ Education-income relationship is monotone")
            checks.append(("Educ-income monotone", "OK", "Yes"))

    status = "PASSED" if passed else "FAILED (see warnings)"
    log.info(f"Validation {status}")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def quality_report(
    df: pd.DataFrame,
    save_path: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """
    Generate a detailed data quality / descriptive statistics report.

    Returns a DataFrame summarizing:
      - Missing rates
      - Key income statistics by group
      - Sample counts by education, sex, race

    Optionally saves to CSV.
    """
    report_sections = {}

    # ── Missing rates ─────────────────────────────────────────────────────────
    missing = (df.isnull().sum() / len(df) * 100).rename("missing_pct").reset_index()
    missing.columns = ["variable", "missing_pct"]
    report_sections["missing"] = missing

    # ── Income stats by education ─────────────────────────────────────────────
    income_by_educ = df.groupby("education", observed=True)["income"].agg([
        ("n", "count"),
        ("median", "median"),
        ("mean", "mean"),
        ("p25", lambda x: x.quantile(0.25)),
        ("p75", lambda x: x.quantile(0.75)),
        ("std", "std"),
    ]).round(0)
    print("\n── Income by Education ──────────────────────")
    print(income_by_educ.to_string())

    # ── Income stats by sex ───────────────────────────────────────────────────
    income_by_sex = df.groupby("sex", observed=True)["income"].agg([
        ("n", "count"), ("median", "median"), ("mean", "mean")
    ]).round(0)
    print("\n── Income by Sex ────────────────────────────")
    print(income_by_sex.to_string())

    # ── Income by race/ethnicity ──────────────────────────────────────────────
    income_by_race = df.groupby("race_ethnicity", observed=True)["income"].agg([
        ("n", "count"), ("median", "median"), ("mean", "mean")
    ]).round(0).sort_values("median", ascending=False)
    print("\n── Income by Race/Ethnicity ─────────────────")
    print(income_by_race.to_string())

    # ── Gender gap by education ───────────────────────────────────────────────
    pivot = df.pivot_table(
        values="income", index="education", columns="sex",
        aggfunc="median", observed=True
    )
    if "Male" in pivot.columns and "Female" in pivot.columns:
        pivot["gender_gap_pct"] = ((pivot["Male"] - pivot["Female"]) / pivot["Male"] * 100).round(1)
        print("\n── Gender Pay Gap by Education (Median) ─────")
        print(pivot.to_string())

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "metric": ["n_total", "median_income", "mean_income",
                   "pct_female", "pct_college_plus", "n_education_groups"],
        "value": [
            len(df),
            round(df["income"].median(), 0),
            round(df["income"].mean(), 0),
            round((df["sex"] == "Female").mean() * 100, 1),
            round((df["education"] >= "Bachelors").mean() * 100, 1),
            df["education"].nunique(),
        ]
    })
    print("\n── Summary ──────────────────────────────────")
    print(summary.to_string(index=False))

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        income_by_educ.to_csv(save_path.parent / "income_by_education.csv")
        income_by_sex.to_csv(save_path.parent / "income_by_sex.csv")
        income_by_race.to_csv(save_path.parent / "income_by_race.csv")
        summary.to_csv(save_path, index=False)
        log.info(f"Quality report saved to {save_path.parent}")

    return summary
