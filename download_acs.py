#!/usr/bin/env python3
"""
download_acs.py
===============
One-command ACS data download using the `folktables` library.

Usage:
    pip install folktables pyarrow
    python download_acs.py

Output: data/raw/acs_raw.csv
"""

import pathlib
import sys

try:
    from folktables import ACSDataSource
except ImportError:
    print("ERROR: folktables not installed. Run: pip install folktables")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RAW_DIR / "acs_raw.csv"

# 5 large, geographically diverse states — ~80-100k rows, manageable RAM
# Covers all 4 Census regions: West, Northeast, South, Midwest
STATES = [
    "CA",   # West
    "NY",   # Northeast
    "TX",   # South
    "IL",   # Midwest
    "GA",   # South (racially diverse — important for fairness analysis)
]

print(f"Downloading 2022 ACS 1-Year PUMS for {len(STATES)} states: {STATES}")
print("This may take 1–2 minutes...")

data_source = ACSDataSource(
    survey_year="2022",
    horizon="1-Year",
    survey="person",
)

acs_data = data_source.get_data(states=STATES, download=True)

print(f"\nRaw ACS data: {len(acs_data):,} rows x {acs_data.shape[1]} columns")

acs_data.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
print(f"\nColumn sample: {list(acs_data.columns[:12])}")
print("\nDone! Next step: run notebooks/02_data_loading_and_cleaning.ipynb")
