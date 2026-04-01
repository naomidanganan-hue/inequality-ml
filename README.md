# Income Inequality & ML Fairness Analysis

A machine learning fairness analysis of wage inequality across demographic 
groups in the United States, using 2022 American Community Survey (ACS) data.

## Paper
📄 [Read on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6417458)

## Acknowledgments
The author thanks Professor Yaser S. Abu-Mostafa of the California 
Institute of Technology for his encouragement and support.

## Key Findings
- Education is the single strongest predictor of income (feature importance = 0.241)
- A graduate degree earns 2.18× the median income of a high school diploma
- 100% of the gender wage gap is unexplained by observable characteristics
- 102.5% of the White–Black wage gap is unexplained by observable characteristics
- Gradient Boosting achieves R²=0.213, outperforming OLS baseline

## Methods
- 4 ML models: OLS, Ridge, Random Forest, Gradient Boosting
- Fairness audit: demographic parity, residual bias, equalized R² by group
- Oaxaca-Blinder decomposition of gender and racial wage gaps
- Dataset: 85,000 records from 2022 ACS PUMS (5 states)

## Reproduce
pip install -r requirements.txt
python download_acs.py
python run_analysis.py

## Author
Naomi Danganan — Independent Researcher
