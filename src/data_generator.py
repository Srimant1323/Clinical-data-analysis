"""
data_generator.py
-----------------
Generates a realistic synthetic clinical trial dataset for
drug efficacy in oncology — mimics real-world data with:
  - Realistic distributions (age, biomarkers, dosage)
  - Controlled missing-value patterns (MCAR, MAR, MNAR)
  - Outliers and data-entry noise

Run this once to create data/raw/clinical_trial_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── Constants ────────────────────────────────────────────────────────────────
N_PATIENTS   = 600
DRUGS        = ["DrugA_TKI", "DrugB_mAb", "Placebo"]
CANCER_TYPES = ["NSCLC", "Breast", "CRC", "Melanoma"]
STAGES       = ["I", "II", "III", "IV"]
ECOG         = [0, 1, 2, 3]          # performance status

# ── Helper generators ─────────────────────────────────────────────────────────

def _age(n):
    """Skewed-right age distribution typical of oncology trials (40–80)."""
    ages = RNG.normal(loc=62, scale=10, size=n).clip(30, 85)
    return np.round(ages).astype(int)

def _bmi(n):
    return np.round(RNG.normal(28, 4, n).clip(17, 45), 1)

def _tumor_size_mm(n):
    """Log-normal tumour size."""
    return np.round(np.exp(RNG.normal(3.5, 0.6, n)).clip(5, 150), 1)

def _biomarker(name, n, loc, scale, low, high):
    vals = RNG.normal(loc, scale, n).clip(low, high)
    return np.round(vals, 2)

def _drug_response(drug_arr, stage_arr, gene_expr, bmi):
    """
    Simulate a treatment response (0=stable/PD, 1=partial/complete response).
    All inputs are NumPy arrays of length n.
    """
    base_map  = {"DrugA_TKI": 0.62, "DrugB_mAb": 0.55, "Placebo": 0.18}
    stage_map = {"I": 0.10, "II": 0.05, "III": -0.05, "IV": -0.15}

    # Vectorised lookup
    p_base  = np.array([base_map[d]  for d in drug_arr])
    p_stage = np.array([stage_map[s] for s in stage_arr])
    p = p_base + p_stage

    # High EGFR boosts TKI response
    tki_mask = drug_arr == "DrugA_TKI"
    p[tki_mask] += 0.12 * (gene_expr[tki_mask] > 8.5).astype(float)

    # Obesity reduces mAb efficacy
    mab_mask = drug_arr == "DrugB_mAb"
    p[mab_mask] -= 0.08 * (bmi[mab_mask] > 30).astype(float)

    p = np.clip(p, 0.05, 0.95)
    return RNG.binomial(1, p).astype(int)

# ── Main generator ────────────────────────────────────────────────────────────

def generate_dataset(n: int = N_PATIENTS, output_path: str | None = None) -> pd.DataFrame:
    """Generate the full synthetic clinical-trial DataFrame."""

    drug        = RNG.choice(DRUGS,        size=n, p=[0.38, 0.38, 0.24])
    cancer_type = RNG.choice(CANCER_TYPES, size=n, p=[0.30, 0.30, 0.25, 0.15])
    stage       = RNG.choice(STAGES,       size=n, p=[0.10, 0.25, 0.35, 0.30])
    ecog        = RNG.choice(ECOG,         size=n, p=[0.30, 0.40, 0.20, 0.10])

    age         = _age(n)
    bmi         = _bmi(n)
    tumor_size  = _tumor_size_mm(n)

    # Gene-expression proxy (log2 normalised, range 4–12)
    gene_expr_EGFR  = _biomarker("EGFR",  n, 7.8, 1.5, 4, 12)
    gene_expr_PD_L1 = _biomarker("PD-L1", n, 6.5, 1.8, 4, 12)

    # Lab values
    wbc         = _biomarker("WBC",  n, 7.2, 2.1, 2.0, 18.0)
    hemoglobin  = _biomarker("Hgb",  n, 12.5, 1.8, 7.0, 17.5)
    creatinine  = _biomarker("Cr",   n, 0.95, 0.22, 0.4, 2.5)
    alt         = np.round(RNG.lognormal(3.4, 0.5, n).clip(10, 250), 1)   # skewed
    ldh         = np.round(RNG.lognormal(5.3, 0.4, n).clip(100, 800), 1)

    # Prior treatments
    prior_chemo = RNG.binomial(1, 0.65, n)
    prior_radio = RNG.binomial(1, 0.45, n)

    # Primary outcomes
    response    = _drug_response(drug, stage, gene_expr_EGFR, bmi)
    pfs_months  = np.round(
        RNG.exponential(scale=np.where(response == 1, 14, 6), size=n).clip(0.5, 36), 1
    )
    os_months   = np.round(pfs_months + RNG.exponential(scale=8, size=n), 1).clip(1, 60)
    adverse_grade = RNG.choice([0, 1, 2, 3, 4], size=n, p=[0.20, 0.35, 0.25, 0.15, 0.05])

    df = pd.DataFrame({
        "patient_id"       : [f"PT-{i:04d}" for i in range(1, n + 1)],
        "drug"             : drug,
        "cancer_type"      : cancer_type,
        "stage"            : stage,
        "ecog_status"      : ecog,
        "age"              : age,
        "bmi"              : bmi,
        "tumor_size_mm"    : tumor_size,
        "gene_expr_EGFR"   : gene_expr_EGFR,
        "gene_expr_PD_L1"  : gene_expr_PD_L1,
        "wbc_10e9_L"       : wbc,
        "hemoglobin_g_dL"  : hemoglobin,
        "creatinine_mg_dL" : creatinine,
        "alt_U_L"          : alt,
        "ldh_U_L"          : ldh,
        "prior_chemo"      : prior_chemo,
        "prior_radiation"  : prior_radio,
        "treatment_response": response,
        "pfs_months"       : pfs_months,
        "os_months"        : os_months,
        "adverse_event_grade": adverse_grade,
    })

    # ── Inject realistic missing values ──────────────────────────────────────
    # MCAR  – lab values randomly missing (~5-8 %)
    for col in ["wbc_10e9_L", "hemoglobin_g_dL", "creatinine_mg_dL", "alt_U_L"]:
        mask = RNG.random(n) < 0.07
        df.loc[mask, col] = np.nan

    # MAR   – LDH missing more often in early stages (less clinical suspicion)
    early_stage_mask = df["stage"].isin(["I", "II"])
    ldh_missing_prob = np.where(early_stage_mask, 0.25, 0.08)
    df.loc[RNG.random(n) < ldh_missing_prob, "ldh_U_L"] = np.nan

    # MNAR  – gene expression missing more when value is low (not ordered if not suspected)
    egfr_low = df["gene_expr_EGFR"] < 6.0
    egfr_miss_prob = np.where(egfr_low, 0.35, 0.05)
    df.loc[RNG.random(n) < egfr_miss_prob, "gene_expr_EGFR"] = np.nan

    pdl1_low = df["gene_expr_PD_L1"] < 5.5
    pdl1_miss_prob = np.where(pdl1_low, 0.30, 0.06)
    df.loc[RNG.random(n) < pdl1_miss_prob, "gene_expr_PD_L1"] = np.nan

    # Demographic missingness (~3 %)
    df.loc[RNG.random(n) < 0.03, "bmi"]          = np.nan
    df.loc[RNG.random(n) < 0.04, "tumor_size_mm"] = np.nan

    # ── Introduce outliers (data entry / instrument error) ────────────────────
    outlier_idx = RNG.choice(n, size=6, replace=False)
    df.loc[outlier_idx[:2], "alt_U_L"]          = RNG.uniform(600, 1200, 2)   # extreme ALT
    df.loc[outlier_idx[2:4], "tumor_size_mm"]   = RNG.uniform(180, 250, 2)   # implausible size
    df.loc[outlier_idx[4:], "wbc_10e9_L"]       = RNG.uniform(40, 80, 2)     # leukocytosis artefact

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[OK] Dataset saved -> {output_path}  ({n} rows x {df.shape[1]} cols)")
        print(f"   Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df


if __name__ == "__main__":
    out = Path(__file__).parents[1] / "data" / "raw" / "clinical_trial_data.csv"
    generate_dataset(output_path=str(out))
