"""
visualisation.py
----------------
Reusable, publication-quality plot helpers.
All functions return matplotlib Figure objects so notebooks can save them.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Colour palette (biotech-inspired) ────────────────────────────────────────
PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
DRUG_PAL  = {"DrugA_TKI": "#4C72B0", "DrugB_mAb": "#55A868", "Placebo": "#C44E52"}
RESP_PAL  = {0: "#C44E52", 1: "#55A868"}

plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#F8F9FA",
    "axes.grid"        : True,
    "grid.color"       : "white",
    "grid.linewidth"   : 1.2,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})


# ── 1. Missing-value heatmap ──────────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame, title: str = "Missing Value Map") -> plt.Figure:
    """Heatmap where each cell shows whether a value is present or absent."""
    fig, ax = plt.subplots(figsize=(14, 5))
    miss_matrix = df.isnull().astype(int)
    sns.heatmap(
        miss_matrix.T,
        ax=ax,
        cbar=False,
        cmap=["#E8F5E9", "#E53935"],
        yticklabels=miss_matrix.columns,
        xticklabels=False,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Patients →")
    ax.set_ylabel("")
    legend_patches = [
        mpatches.Patch(color="#E8F5E9", label="Present"),
        mpatches.Patch(color="#E53935", label="Missing"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_missing_bar(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of % missing per column (only columns with missings)."""
    pct = df.isnull().mean().mul(100).sort_values(ascending=False)
    pct = pct[pct > 0]

    fig, ax = plt.subplots(figsize=(8, max(4, len(pct) * 0.45)))
    colors  = ["#E53935" if v > 20 else "#FB8C00" if v > 10 else "#43A047" for v in pct]
    bars    = ax.barh(pct.index, pct.values, color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, pct.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9)

    ax.set_xlabel("% Missing")
    ax.set_title("Missing Data — Column Summary", fontweight="bold")
    ax.set_xlim(0, pct.max() + 8)
    fig.tight_layout()
    return fig


# ── 2. Distribution plots ─────────────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame,
                                cols: list[str],
                                ncols: int = 3) -> plt.Figure:
    """Grid of histograms with KDE and normal-fit overlay."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=30, density=True,
                color="#4C72B0", alpha=0.65, edgecolor="white")
        data.plot.kde(ax=ax, color="#1A237E", lw=2)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        _, p = stats.shapiro(data.sample(min(len(data), 5000), random_state=42))
        ax.text(0.97, 0.95, f"Shapiro p={p:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="grey")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── 3. Correlation ────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame,
                              cols: list[str],
                              title: str = "Pearson Correlation Matrix") -> plt.Figure:
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="white",
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


# ── 4. Survival / KM-style PFS ────────────────────────────────────────────────

def plot_pfs_by_drug(df: pd.DataFrame) -> plt.Figure:
    """Empirical CDF of progression-free survival per drug arm."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for drug, color in DRUG_PAL.items():
        sub = df[df["drug"] == drug]["pfs_months"].dropna().sort_values()
        ecdf_y = np.arange(1, len(sub) + 1) / len(sub)
        survival = 1 - ecdf_y
        ax.step(sub, survival, where="post", label=drug, color=color, lw=2.2)

    ax.set_xlabel("Progression-Free Survival (months)")
    ax.set_ylabel("Proportion Progression-Free")
    ax.set_title("Empirical PFS Curves by Treatment Arm", fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


# ── 5. Response rate ──────────────────────────────────────────────────────────

def plot_response_rate(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of treatment response rate by drug and stage."""
    pivot = (
        df.groupby(["drug", "stage"])["treatment_response"]
        .mean()
        .mul(100)
        .reset_index()
    )
    pivot.columns = ["drug", "stage", "response_rate"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=pivot, x="stage", y="response_rate", hue="drug",
        palette=DRUG_PAL, ax=ax, edgecolor="white",
        order=["I", "II", "III", "IV"],
    )
    ax.set_title("Treatment Response Rate by Cancer Stage & Drug", fontweight="bold")
    ax.set_xlabel("Cancer Stage")
    ax.set_ylabel("Response Rate (%)")
    ax.legend(title="Drug", frameon=True)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


# ── 6. Biomarker scatter ──────────────────────────────────────────────────────

def plot_biomarker_scatter(df: pd.DataFrame) -> plt.Figure:
    """EGFR vs PD-L1 coloured by treatment response."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for resp, label, color in [(0, "No Response", "#C44E52"), (1, "Response", "#55A868")]:
        sub = df[df["treatment_response"] == resp]
        ax.scatter(
            sub["gene_expr_EGFR"], sub["gene_expr_PD_L1"],
            c=color, alpha=0.55, s=28, label=label, edgecolors="white", lw=0.3,
        )
    ax.set_xlabel("EGFR Expression (log₂)")
    ax.set_ylabel("PD-L1 Expression (log₂)")
    ax.set_title("Biomarker Expression vs Treatment Response", fontweight="bold")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


# ── 7. Box plot – lab values by response ─────────────────────────────────────

def plot_lab_boxplots(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    nrows = int(np.ceil(len(cols) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, cols):
        sns.boxplot(
            data=df, x="treatment_response", y=col, ax=ax,
            palette={0: "#C44E52", 1: "#55A868"},
            width=0.55, linewidth=1.2,
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Response (0=No, 1=Yes)")
        ax.set_ylabel("")
        # Significance annotation
        g0 = df.loc[df["treatment_response"] == 0, col].dropna()
        g1 = df.loc[df["treatment_response"] == 1, col].dropna()
        if len(g0) > 5 and len(g1) > 5:
            _, p = stats.mannwhitneyu(g0, g1)
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(0.5, 0.97, f"MWU p={p:.3f} {sig}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color="grey")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Lab Values by Treatment Response", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig
