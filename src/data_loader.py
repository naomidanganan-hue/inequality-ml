"""
data_loader.py
==============
Loads and standardizes income microdata from two sources:
  - IPUMS CPS  (data/raw/cps_raw.csv or cps_raw.csv.gz)
  - ACS via folktables (data/raw/acs_raw.csv)

Both are normalized to a common schema defined in DATA_ACQUISITION.md.

Usage:
    from src.data_loader import load_cps, load_acs, load_data
    df = load_data(source="cps")   # or "acs"
"""

import logging
import pathlib
from typing import Literal, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CPS LOADER
# ══════════════════════════════════════════════════════════════════════════════

# IPUMS CPS education codes → readable labels
# Source: https://cps.ipums.org/cps-action/variables/EDUC#codes_section
_CPS_EDUC_MAP = {
    # No schooling / preschool
    **{k: "HS_or_less" for k in [0, 1, 2, 10, 11, 12, 13, 14, 20, 21, 22, 30, 31,
                                   32, 40, 50, 60, 70, 71, 72, 73]},
    # HS diploma or GED
    **{k: "HS_or_less" for k in [80, 90, 91, 92]},
    # Some college, no degree / Associate
    **{k: "Some_College" for k in [100, 110, 111, 120, 121, 122, 123]},
    # Bachelor's
    124: "Bachelors",
    # Graduate / Professional / Doctorate
    **{k: "Graduate" for k in [125, 126]},
}

_CPS_RACE_MAP = {
    100: "White",
    200: "Black",
    300: "AIAN",       # American Indian / Alaska Native
    650: "Asian",
    651: "Asian",
    652: "Asian",
    700: "Other",
    801: "White",      # White-Black
    802: "Other",
    900: "Other",
    999: "Unknown",
}

_CPS_REGION_MAP = {
    11: "Northeast", 12: "Northeast",
    21: "Midwest",   22: "Midwest",
    31: "South",     32: "South",   33: "South",
    41: "West",      42: "West",
}


def load_cps(
    path: Optional[pathlib.Path] = None,
    years: Optional[list[int]] = None,
    full_time_only: bool = False,
) -> pd.DataFrame:
    """
    Load and clean IPUMS CPS extract.

    Parameters
    ----------
    path : Path to the raw CPS CSV (or .csv.gz). Defaults to data/raw/cps_raw.csv.gz
    years : List of survey years to keep. None = keep all.
    full_time_only : If True, keep only full-time, full-year workers.

    Returns
    -------
    pd.DataFrame with standardized schema.
    """
    if path is None:
        # Try both compressed and uncompressed
        for candidate in ["cps_raw.csv.gz", "cps_raw.csv"]:
            p = RAW_DIR / candidate
            if p.exists():
                path = p
                break
        else:
            raise FileNotFoundError(
                f"CPS raw data not found in {RAW_DIR}. "
                "See DATA_ACQUISITION.md for download instructions."
            )

    log.info(f"Loading CPS data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    # ── Normalize column names ──────────────────────────────────────────────
    df.columns = df.columns.str.upper().str.strip()

    _require_columns(df, ["AGE", "SEX", "RACE", "EDUC", "INCWAGE", "WTFINL"], source="CPS")

    # ── Year filter ─────────────────────────────────────────────────────────
    if years is not None and "YEAR" in df.columns:
        df = df[df["YEAR"].isin(years)].copy()
        log.info(f"  After year filter {years}: {len(df):,} rows")

    # ── Age filter: prime working age ───────────────────────────────────────
    df = df[(df["AGE"] >= 25) & (df["AGE"] <= 64)].copy()
    log.info(f"  After age filter (25–64): {len(df):,} rows")

    # ── Income: INCWAGE ─────────────────────────────────────────────────────
    # IPUMS sentinel: 999999.9 = N/A; 999998 = missing
    df["income"] = pd.to_numeric(df["INCWAGE"], errors="coerce")
    df.loc[df["income"] >= 999998, "income"] = np.nan
    # Drop zero/negative (not employed for wages)
    df = df[df["income"] > 0].copy()
    log.info(f"  After positive income filter: {len(df):,} rows")

    # ── Full-time filter ────────────────────────────────────────────────────
    if full_time_only:
        if "UHRSWORK" in df.columns and "WKSWORK2" in df.columns:
            # WKSWORK2 categories: 1=1–13wks, 2=14–26, 3=27–39, 4=40–47, 5=48–49, 6=50–52
            df["hours_per_week"] = pd.to_numeric(df["UHRSWORK"], errors="coerce")
            df["weeks_worked"] = df["WKSWORK2"].map({1: 7, 2: 20, 3: 33, 4: 43.5, 5: 48.5, 6: 51})
            df["full_time_full_year"] = (df["hours_per_week"] >= 35) & (df["weeks_worked"] >= 40)
            df = df[df["full_time_full_year"]].copy()
            log.info(f"  After full-time filter: {len(df):,} rows")
        else:
            log.warning("  UHRSWORK/WKSWORK2 not found; skipping full-time filter")

    # ── Hours / weeks (even without filtering) ──────────────────────────────
    if "UHRSWORK" in df.columns:
        df["hours_per_week"] = pd.to_numeric(df["UHRSWORK"], errors="coerce")
        df.loc[df["hours_per_week"] == 999, "hours_per_week"] = np.nan
    else:
        df["hours_per_week"] = np.nan

    if "WKSWORK2" in df.columns:
        df["weeks_worked"] = df["WKSWORK2"].map({1: 7, 2: 20, 3: 33, 4: 43.5, 5: 48.5, 6: 51})
    else:
        df["weeks_worked"] = np.nan

    df["full_time_full_year"] = (
        (df["hours_per_week"] >= 35) & (df["weeks_worked"] >= 40)
    ).fillna(False)

    # ── Education ───────────────────────────────────────────────────────────
    df["education"] = df["EDUC"].map(_CPS_EDUC_MAP)
    df = df[df["education"].notna()].copy()

    # ── Sex ─────────────────────────────────────────────────────────────────
    df["sex"] = df["SEX"].map({1: "Male", 2: "Female"})

    # ── Race / ethnicity ────────────────────────────────────────────────────
    df["race_ethnicity"] = df["RACE"].map(_CPS_RACE_MAP).fillna("Other")
    # Hispanic override (HISPAN: 0=not Hispanic, 1–499=Hispanic)
    if "HISPAN" in df.columns:
        df.loc[
            (df["HISPAN"] > 0) & (df["HISPAN"] < 900),
            "race_ethnicity"
        ] = "Hispanic"

    # ── Region ──────────────────────────────────────────────────────────────
    if "REGION" in df.columns:
        df["region"] = df["REGION"].map(_CPS_REGION_MAP).fillna("Unknown")
    else:
        df["region"] = "Unknown"

    # ── Metro ───────────────────────────────────────────────────────────────
    if "METRO" in df.columns:
        df["metro"] = df["METRO"].map({
            0: "Not identified",
            1: "Non-metro",
            2: "Metro",
            3: "Metro",
            4: "Metro",
        }).fillna("Unknown")
    else:
        df["metro"] = "Unknown"

    # ── Weight & year ───────────────────────────────────────────────────────
    df["weight"] = pd.to_numeric(df["WTFINL"], errors="coerce").fillna(1.0)
    df["year"] = df["YEAR"].astype(int) if "YEAR" in df.columns else np.nan
    df["source"] = "CPS"

    return _finalize(df)


# ══════════════════════════════════════════════════════════════════════════════
# ACS / folktables LOADER
# ══════════════════════════════════════════════════════════════════════════════

# ACS education codes (SCHL variable)
_ACS_EDUC_MAP = {
    **{k: "HS_or_less" for k in range(1, 17)},   # No HS diploma
    **{k: "HS_or_less" for k in [17, 18, 19]},   # HS diploma / GED
    **{k: "Some_College" for k in [20, 21]},      # Some college / Associate's
    22: "Bachelors",
    23: "Graduate",   # Master's
    24: "Graduate",   # Professional degree
    25: "Graduate",   # Doctorate
}

_ACS_SEX_MAP = {1: "Male", 2: "Female"}

_ACS_RAC1P_MAP = {
    1: "White", 2: "Black", 3: "AIAN", 4: "AIAN",
    5: "AIAN", 6: "Asian", 7: "NHPI", 8: "Other", 9: "Other",
}


def load_acs(
    path: Optional[pathlib.Path] = None,
    full_time_only: bool = False,
) -> pd.DataFrame:
    """
    Load and clean ACS microdata (downloaded via folktables or manually).

    Parameters
    ----------
    path : Path to acs_raw.csv. Defaults to data/raw/acs_raw.csv
    full_time_only : Keep only full-time, full-year workers.

    Returns
    -------
    pd.DataFrame with standardized schema.
    """
    if path is None:
        path = RAW_DIR / "acs_raw.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"ACS raw data not found at {path}. "
                "See DATA_ACQUISITION.md for download instructions."
            )

    log.info(f"Loading ACS data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw shape: {df.shape}")

    df.columns = df.columns.str.upper().str.strip()

    # ACS income: WAGP (wage and salary income)
    # PINCP = total personal income (alternative)
    income_col = "WAGP" if "WAGP" in df.columns else "PINCP"
    _require_columns(df, ["AGEP", "SEX", "SCHL", income_col], source="ACS")

    # ── Age filter ──────────────────────────────────────────────────────────
    df["age"] = pd.to_numeric(df["AGEP"], errors="coerce")
    df = df[(df["age"] >= 25) & (df["age"] <= 64)].copy()

    # ── Income ──────────────────────────────────────────────────────────────
    df["income"] = pd.to_numeric(df[income_col], errors="coerce")
    df = df[df["income"] > 0].copy()
    log.info(f"  After age + income filter: {len(df):,} rows")

    # ── Full-time filter ────────────────────────────────────────────────────
    # ACS: WKHP = usual hours worked, WKW = weeks worked last year
    if full_time_only and "WKHP" in df.columns and "WKW" in df.columns:
        df["hours_per_week"] = pd.to_numeric(df["WKHP"], errors="coerce")
        # WKW: 1=50–52wks, 2=48–49, 3=40–47, 4=27–39, 5=14–26, 6=1–13
        df["weeks_worked"] = df["WKW"].map({1: 51, 2: 48.5, 3: 43.5, 4: 33, 5: 20, 6: 7})
        df["full_time_full_year"] = (df["hours_per_week"] >= 35) & (df["weeks_worked"] >= 40)
        df = df[df["full_time_full_year"]].copy()
        log.info(f"  After full-time filter: {len(df):,} rows")
    else:
        df["hours_per_week"] = pd.to_numeric(df.get("WKHP", pd.Series(dtype=float)), errors="coerce")
        df["weeks_worked"] = df.get("WKW", pd.Series(dtype=float)).map(
            {1: 51, 2: 48.5, 3: 43.5, 4: 33, 5: 20, 6: 7}
        ) if "WKW" in df.columns else np.nan
        df["full_time_full_year"] = (
            (df["hours_per_week"] >= 35) & (df["weeks_worked"] >= 40)
        ).fillna(False)

    # ── Education ───────────────────────────────────────────────────────────
    df["education"] = pd.to_numeric(df["SCHL"], errors="coerce").map(_ACS_EDUC_MAP)

    # ── Sex ─────────────────────────────────────────────────────────────────
    df["sex"] = pd.to_numeric(df["SEX"], errors="coerce").map(_ACS_SEX_MAP)

    # ── Race / ethnicity ────────────────────────────────────────────────────
    if "RAC1P" in df.columns:
        df["race_ethnicity"] = pd.to_numeric(df["RAC1P"], errors="coerce").map(_ACS_RAC1P_MAP).fillna("Other")
    else:
        df["race_ethnicity"] = "Unknown"
    if "HISP" in df.columns:
        df.loc[pd.to_numeric(df["HISP"], errors="coerce") > 1, "race_ethnicity"] = "Hispanic"

    # ── Region / metro ──────────────────────────────────────────────────────
    df["region"] = "Unknown"
    df["metro"] = "Unknown"

    # ── Weight ──────────────────────────────────────────────────────────────
    weight_col = "PWGTP" if "PWGTP" in df.columns else None
    df["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0) if weight_col else 1.0
    df["year"] = pd.to_numeric(df["YEAR"], errors="coerce") if "YEAR" in df.columns else 2022
    df["source"] = "ACS"

    return _finalize(df)


# ══════════════════════════════════════════════════════════════════════════════
# COMMON FINALIZATION
# ══════════════════════════════════════════════════════════════════════════════

_EDUCATION_ORDER = pd.CategoricalDtype(
    categories=["HS_or_less", "Some_College", "Bachelors", "Graduate"],
    ordered=True,
)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final transformations common to all sources."""

    if "age" not in df.columns:
        age_col = "AGE" if "AGE" in df.columns else "AGEP"
        df["age"] = pd.to_numeric(df[age_col], errors="coerce")

    # ── Log income ──────────────────────────────────────────────────────────
    df["log_income"] = np.log1p(df["income"])

    # ── Ordered education category ───────────────────────────────────────────
    df["education"] = df["education"].astype(_EDUCATION_ORDER)

    # ── Age bins ─────────────────────────────────────────────────────────────
    df["age_group"] = pd.cut(
        df["age"],
        bins=[24, 34, 44, 54, 64],
        labels=["25–34", "35–44", "45–54", "55–64"],
    )

    # ── Keep only the standardized columns ──────────────────────────────────
    keep = [
        "age", "age_group", "sex", "race_ethnicity",
        "education", "income", "log_income",
        "hours_per_week", "weeks_worked", "full_time_full_year",
        "region", "metro", "weight", "year", "source",
    ]
    # Keep only columns that actually exist
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # ── Drop rows missing critical fields ────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["age", "income", "education", "sex"])
    after = len(df)
    log.info(f"  Dropped {before - after:,} rows with missing critical fields")
    log.info(f"  Final shape: {df.shape}")

    return df.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns[:20])}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_data(
    source: Literal["cps", "acs"] = "acs",
    full_time_only: bool = False,
    years: Optional[list[int]] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load, clean, and optionally cache microdata.

    Parameters
    ----------
    source : 'cps' or 'acs'
    full_time_only : Restrict to full-time, full-year workers.
    years : CPS only — filter to specific survey years.
    cache : If True, save processed parquet to data/processed/ and reload on next call.

    Returns
    -------
    pd.DataFrame with standardized schema.

    Example
    -------
    >>> from src.data_loader import load_data
    >>> df = load_data(source="acs", full_time_only=True)
    >>> print(df.shape)
    """
    suffix = f"{'_ft' if full_time_only else ''}"
    cache_path = PROCESSED_DIR / f"{source}_clean{suffix}.parquet"

    if cache and cache_path.exists():
        log.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    if source == "cps":
        df = load_cps(full_time_only=full_time_only, years=years)
    elif source == "acs":
        df = load_acs(full_time_only=full_time_only)
    else:
        raise ValueError(f"Unknown source: {source!r}. Choose 'cps' or 'acs'.")

    if cache:
        df.to_parquet(cache_path, index=False)
        log.info(f"Cached to {cache_path}")

    return df
