# 🧬 Biotech Clinical Trial — Data Analysis Portfolio

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Notebooks: 3](https://img.shields.io/badge/Notebooks-3-orange?logo=jupyter)](notebooks/)

> **End-to-end data analysis of a 600-patient oncology clinical trial**, demonstrating professional analyst workflow: data generation → missing-data handling → EDA → statistical inference → predictive modelling → SHAP explainability.

---

## 📋 Project Overview

This project simulates the analytical pipeline a data analyst would run on a real-world Phase II oncology trial comparing two targeted therapies (DrugA_TKI, DrugB_mAb) against a placebo across four cancer types.

**Key skills demonstrated:**
| Skill | Where |
|-------|--------|
| Missing-data classification (MCAR / MAR / MNAR) | Notebook 1 |
| Outlier detection & winsorisation | Notebook 1 |
| KNN imputation vs median imputation | Notebook 1 |
| EDA — distributions, correlations | Notebook 2 |
| Statistical inference (χ², Mann-Whitney U, Logistic OR) | Notebook 2 |
| Empirical survival curves (PFS/OS) | Notebook 2 |
| XGBoost classifier with cross-validation | Notebook 3 |
| ROC-AUC, Precision-Recall, Confusion Matrix | Notebook 3 |
| SHAP explainability (beeswarm, waterfall, dependence) | Notebook 3 |
| Modular, reusable Python source code | `src/` |

---

## 📁 Project Structure

```
biotech-data-analysis/
├── data/
│   ├── raw/                  # Synthetic dataset (generated at runtime)
│   └── processed/            # Cleaned, imputed dataset
├── notebooks/
│   ├── 01_data_ingestion_missing_data.ipynb   # Data audit & imputation
│   ├── 02_EDA.ipynb                           # EDA & statistical tests
│   └── 03_predictive_modelling.ipynb          # XGBoost + SHAP
├── reports/
│   └── figures/              # All exported plots (PNG)
├── src/
│   ├── data_generator.py     # Synthetic data generator
│   ├── preprocessing.py      # Audit, validate, impute, feature engineering
│   └── visualisation.py      # Reusable plot helpers
├── requirements.txt
└── README.md
```

---

## 🗄️ Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Unique patient identifier |
| `drug` | cat | DrugA_TKI / DrugB_mAb / Placebo |
| `cancer_type` | cat | NSCLC, Breast, CRC, Melanoma |
| `stage` | cat | I–IV |
| `ecog_status` | int | 0–3 (performance status) |
| `age`, `bmi` | float | Demographics |
| `tumor_size_mm` | float | Baseline tumour diameter |
| `gene_expr_EGFR` | float | log₂ EGFR expression **(MNAR)** |
| `gene_expr_PD_L1` | float | log₂ PD-L1 expression **(MNAR)** |
| `wbc_10e9_L`, `hemoglobin_g_dL`, `creatinine_mg_dL`, `alt_U_L` | float | Lab values **(MCAR)** |
| `ldh_U_L` | float | Lactate dehydrogenase **(MAR – stage-dependent)** |
| `prior_chemo`, `prior_radiation` | int | Prior treatment flags |
| `treatment_response` | int | 0 = stable/PD, 1 = partial/complete response |
| `pfs_months` | float | Progression-free survival |
| `os_months` | float | Overall survival |
| `adverse_event_grade` | int | CTCAE grade 0–4 |

**Injected missingness patterns:**
- **MCAR (~7%)** — lab values randomly missing
- **MAR** — LDH missing more in early stages (25% vs 8%)
- **MNAR** — Gene expression missing more when value is low (35% vs 5%)

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/biotech-data-analysis.git
cd biotech-data-analysis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate raw data & run preprocessing pipeline
python src/data_generator.py
python src/preprocessing.py

# 5. Open notebooks
jupyter notebook notebooks/
```

Run notebooks in order: **01 → 02 → 03**

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Overall response rate | ~47% |
| DrugA_TKI response rate | ~62% |
| Placebo response rate | ~18% |
| XGBoost CV AUC | ~0.86 |
| Top SHAP predictor | `gene_expr_EGFR` |

**Statistical findings:**
- Response rate differs significantly across treatment arms (χ² p < 0.001)
- EGFR expression is significantly higher in responders (Mann-Whitney p < 0.05)
- Stage IV patients have 50% lower median PFS vs Stage I

---

## 🛠️ Technical Stack

- **Python 3.11** — pandas, numpy, scipy, statsmodels
- **Visualisation** — matplotlib, seaborn, missingno
- **Machine Learning** — scikit-learn, XGBoost
- **Explainability** — SHAP
- **Notebooks** — Jupyter

---

## 📄 License

MIT License — feel free to use and adapt for your own portfolio.

---

*Built as a data analyst portfolio project demonstrating end-to-end biotech data analysis skills.*
