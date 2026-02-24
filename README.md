# Inequality-ML

This project explores **income inequality** by analyzing how **income differs across education levels** (and age) using a small sample dataset.

## What’s inside
- **Data**: `data/raw/income_sample.csv` (columns: `age`, `education`, `income`)
- **Notebooks**:
  - `notebooks/01_python_basics.ipynb` — Python/pandas practice
  - `notebooks/02_data_analysis.ipynb` — main analysis (load → group → summarize → plot)
- **Outputs** (saved in `docs/`):
  - `avg_income_by_education.png` — bar chart of average income by education
  - `income_by_education_summary.csv` — summary table (count/mean/min/max)
  - `report.md` — short written report

## Method (high level)
1. Load the CSV into a pandas DataFrame
2. Group by `education` and compute summary statistics
3. Visualize average income by education using a bar chart
4. Export results (chart + summary table) and document findings in a short report

## Key takeaway (from this sample)
Average income increases with education level in the dataset (Graduate > College > HS).

## How to run
Open `notebooks/02_data_analysis.ipynb` in VS Code and run cells from top to bottom.

### Requirements
- Python 3.x
- pandas
- matplotlib

Install:
```bash
pip install pandas matplotlib
