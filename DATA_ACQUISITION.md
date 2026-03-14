# Data Acquisition Guide

This project uses **two data sources** — pick one (or both):

- **Option A: IPUMS CPS** — Current Population Survey microdata. Best for income + education + demographics across time. Used in thousands of peer-reviewed papers.
- **Option B: ACS via `folktables`** — American Community Survey, easier to download programmatically. Used in major ML fairness papers (e.g., Ding et al. 2021).

---

## Option A: IPUMS CPS (Recommended for publication)

### Why
- Citable in academic papers ("IPUMS CPS, Sarah Flood et al., 2023")
- Covers 1962–present, enabling time-series analysis
- Includes detailed occupation, race, gender, region variables

### Steps

1. **Create a free account** at https://cps.ipums.org/cps/
2. Click **"Get Data"** → **"Select Samples"**
3. Select years: check **2018, 2019, 2021, 2022, 2023** (skip 2020 — COVID distortions)
4. Click **"Select Variables"** and add these:

   | Variable | Description |
   |----------|-------------|
   | `AGE` | Age |
   | `SEX` | Sex |
   | `RACE` | Race |
   | `HISPAN` | Hispanic origin |
   | `EDUC` | Educational attainment |
   | `WKSWORK2` | Weeks worked last year |
   | `UHRSWORK` | Usual hours worked per week |
   | `INCWAGE` | Wage and salary income |
   | `INCTOT` | Total personal income |
   | `CLASSWKR` | Class of worker |
   | `OCC` | Occupation |
   | `IND` | Industry |
   | `STATEFIP` | State FIPS code |
   | `METRO` | Metropolitan area status |
   | `WTFINL` | Final person weight (CRITICAL — use for weighted statistics) |

5. Click **"View Cart"** → **"Create Data Extract"**
6. Select format: **CSV** (not fixed-width)
7. Submit extract → you'll get an email when it's ready (usually < 5 min)
8. Download the `.csv.gz` file and place it at: `data/raw/cps_raw.csv.gz`

### Citation (add to your paper)
```
Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren,
Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler,
and Michael Westberry. IPUMS CPS: Version 11.0 [dataset].
Minneapolis, MN: IPUMS, 2023. https://doi.org/10.18128/D030.V11.0
```

---

## Option B: ACS via `folktables` (Faster, great for ML fairness)

### Why
- Downloadable in one Python command (no account needed)
- Used in the landmark Ding et al. 2021 NeurIPS paper on ML benchmarks
- Perfect for framing this as an ML fairness project (strong angle for MIT/Stanford)

### Install
```bash
pip install folktables
```

### Download
```python
# Run this once to download data to data/raw/
from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year='2022', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA", "NY", "TX", "IL", "PA",
                                         "OH", "GA", "NC", "MI", "WA"],
                                download=True)
features, labels, group = ACSIncome.df_to_numpy(acs_data)

# Save raw
acs_data.to_csv('data/raw/acs_raw.csv', index=False)
print(f"Downloaded {len(acs_data):,} rows")
```

### Citation
```
Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021).
Retiring Adult: New Datasets for Fair Machine Learning.
Advances in Neural Information Processing Systems, 34.
```

---

## Data Dictionary (after cleaning)

Our pipeline standardizes both sources into a common schema:

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Age in years |
| `sex` | category | 'Male', 'Female' |
| `race_ethnicity` | category | Collapsed race/ethnicity group |
| `education` | category | Ordered: HS_or_less, Some_College, Bachelors, Graduate |
| `log_income` | float | log(annual wage income + 1) |
| `income` | float | Annual wage/salary income (USD) |
| `hours_per_week` | float | Usual hours worked per week |
| `weeks_worked` | float | Weeks worked last year |
| `full_time_full_year` | bool | Works 35+ hrs/wk, 40+ weeks/yr |
| `region` | category | US Census region |
| `metro` | category | Metro/non-metro |
| `weight` | float | Survey sampling weight |
| `year` | int | Survey year |
| `source` | str | 'CPS' or 'ACS' |
