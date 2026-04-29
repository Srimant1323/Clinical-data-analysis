"""
preprocessing.py
----------------
Reusable data-cleaning and imputation utilities used across notebooks.
Demonstrates professional analyst workflow:
  - Audit → validate → clean → impute → engineer → export
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


# ── 1. Audit ─────────────────────────────────────────────────────────────────

def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a structured audit table of every column."""
    rows = []
    for col in df.columns:
        series = df[col]
        n_miss   = series.isna().sum()
        pct_miss = round(100 * n_miss / len(df), 2)
        dtype    = str(series.dtype)
        n_unique = series.nunique(dropna=True)
        q25, q50, q75 = ("", "", "")
        if pd.api.types.is_numeric_dtype(series):
            q25, q50, q75 = series.quantile([0.25, 0.50, 0.75]).round(3).tolist()

        rows.append({
            "column"        : col,
            "dtype"         : dtype,
            "n_missing"     : n_miss,
            "pct_missing"   : pct_miss,
            "n_unique"      : n_unique,
            "Q25"           : q25,
            "median"        : q50,
            "Q75"           : q75,
        })
    return pd.DataFrame(rows)


# ── 2. Validate ───────────────────────────────────────────────────────────────

VALID_RANGES = {
    "age"              : (18, 100),
    "bmi"              : (10, 60),
    "tumor_size_mm"    : (1, 160),
    "wbc_10e9_L"       : (0.5, 30),
    "hemoglobin_g_dL"  : (4, 20),
    "creatinine_mg_dL" : (0.2, 15),
    "alt_U_L"          : (5, 500),
    "ldh_U_L"          : (80, 900),
    "gene_expr_EGFR"   : (1, 15),
    "gene_expr_PD_L1"  : (1, 15),
    "pfs_months"       : (0, 72),
    "os_months"        : (0, 72),
}

def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a boolean mask DataFrame where True = likely outlier.
    Uses IQR-fence AND domain-knowledge hard limits.
    """
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        s = df[col]
        # IQR fence
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_lo, iqr_hi = q1 - 3 * iqr, q3 + 3 * iqr
        # Domain limit
        out_mask = s.notna() & ((s < max(lo, iqr_lo)) | (s > min(hi, iqr_hi)))
        flags[col] = out_mask

    return flags


def cap_outliers(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Winsorise values flagged as outliers to the domain hard limits."""
    df = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        df.loc[flags[col], col] = df.loc[flags[col], col].clip(lo, hi)
    return df


# ── 3. Impute ─────────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "age", "bmi", "tumor_size_mm",
    "gene_expr_EGFR", "gene_expr_PD_L1",
    "wbc_10e9_L", "hemoglobin_g_dL",
    "creatinine_mg_dL", "alt_U_L", "ldh_U_L",
]

CATEGORICAL_COLS = ["drug", "cancer_type", "stage", "ecog_status"]


def impute_missing(df: pd.DataFrame,
                   numeric_strategy: str = "knn",
                   k: int = 5) -> pd.DataFrame:
    """
    Impute missing values:
      - Numeric  : KNN (default) or median imputation.
      - Categorical : mode imputation.

    Returns a new DataFrame; never mutates the original.
    """
    df = df.copy()

    # ── Numeric ──────────────────────────────────────────────────────────────
    num_subset = [c for c in NUMERIC_COLS if c in df.columns]
    if numeric_strategy == "knn":
        imputer = KNNImputer(n_neighbors=k, weights="distance")
    else:
        imputer = SimpleImputer(strategy="median")

    df[num_subset] = imputer.fit_transform(df[num_subset])
    df[num_subset] = df[num_subset].round(2)

    # ── Categorical ───────────────────────────────────────────────────────────
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode().iloc[0]
            df[col].fillna(mode_val, inplace=True)

    return df


# ── 4. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clinically-meaningful features from base columns.
    """
    df = df.copy()

    # Neutrophil-to-Lymphocyte Ratio proxy (WBC-based approximation)
    df["nlr_proxy"] = (df["wbc_10e9_L"] * 0.65 /
                       (df["wbc_10e9_L"] * 0.25 + 1e-6)).round(2)

    # EGFR / PD-L1 ratio — used in combined biomarker analysis
    df["egfr_pdl1_ratio"] = (df["gene_expr_EGFR"] /
                              (df["gene_expr_PD_L1"] + 1e-6)).round(3)

    # Tumour burden index (size × LDH, both log-scaled)
    df["tumor_burden_idx"] = (
        np.log1p(df["tumor_size_mm"]) * np.log1p(df["ldh_U_L"])
    ).round(3)

    # Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 50, 65, 100],
        labels=["<50", "50–65", ">65"],
    )

    # Binary: high biomarker expression
    df["high_EGFR"]  = (df["gene_expr_EGFR"]  > 8.0).astype(int)
    df["high_PD_L1"] = (df["gene_expr_PD_L1"] > 7.5).astype(int)

    # Stage severity (ordinal encoding)
    stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    df["stage_num"] = df["stage"].map(stage_map)

    return df


# ── 5. Full Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline: load → audit → validate → impute → engineer → save.
    """
    print("=" * 60)
    print(" BIOTECH DATA ANALYSIS — PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    df = pd.read_csv(raw_path)
    print(f"\n[1/5] Loaded raw data  ->  {df.shape[0]} rows x {df.shape[1]} cols")

    # Audit
    audit = audit_dataframe(df)
    missing_report = audit[audit["n_missing"] > 0][
        ["column", "n_missing", "pct_missing"]
    ]
    print(f"\n[2/5] Missing-value audit:\n{missing_report.to_string(index=False)}")

    # Validate & cap outliers
    flags   = flag_outliers(df)
    n_flags = flags.sum().sum()
    df      = cap_outliers(df, flags)
    print(f"\n[3/5] Outlier detection  ->  {n_flags} values winsorised")

    # Impute
    df = impute_missing(df, numeric_strategy="knn", k=5)
    still_missing = df.isnull().sum().sum()
    print(f"\n[4/5] Imputation complete  ->  {still_missing} missing values remain")

    # Feature engineering
    df = engineer_features(df)
    print(f"\n[5/5] Features engineered  ->  {df.shape[1]} total columns")

    # Save
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\n[OK] Processed data saved -> {processed_path}\n{'=' * 60}\n")

    return df


if __name__ == "__main__":
    base = Path(__file__).parents[1]
    run_pipeline(
        raw_path=str(base / "data" / "raw"       / "clinical_trial_data.csv"),
        processed_path=str(base / "data" / "processed" / "clinical_trial_clean.csv"),
    )
