"""
Quantify expression-replication consistency between two wet-lab runs of the
same plate (NMR_RUN_2_P3 and its REDO).

For continuous fields (`yield_per_Leq`, `tot_yield`):
  - Pearson + Spearman correlation
  - linear regression (slope, intercept, R^2)
  - Bland-Altman style stats: mean / SD / median |diff|, mean |%diff|

For boolean SEC-quality flags (`correct_Vel_95CI`, `correct_Vel_99CI`):
  - per-cell agreement rate
  - confusion matrix (both True / True->False / False->True / both False)
  - Cohen's kappa

Inputs are matched well-by-well on `Design ID` (falls back to `Name` / `Source Well`
if `Design ID` collisions appear). Outputs:
  - expression_replication_summary.csv  : one-row-per-metric summary
  - expression_replication_paired.csv   : per-design paired values for all fields
  - expression_replication_scatter.png  : 2x2 scatter (yield + tot_yield raw / log)
"""

# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from config import EXPRESSION_RUNS_DIR, OUTPUTS_DIR

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

RUN_A_LABEL = "250910_NMR_RUN_2_P3"
RUN_B_LABEL = "250913_NMR_RUN_2_P3_REDO"

RUN_A_PATH = EXPRESSION_RUNS_DIR / f"{RUN_A_LABEL}_outputs" / "processed_results.h5"
RUN_B_PATH = EXPRESSION_RUNS_DIR / f"{RUN_B_LABEL}_outputs" / "processed_results.h5"

CONTINUOUS_FIELDS = ["yield_per_Leq", "tot_yield"]
BOOLEAN_FIELDS = ["correct_Vel_95CI", "correct_Vel_99CI"]

OUT_DIR = OUTPUTS_DIR / "expression_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_hdf(path, key="df")
    # `Design ID` is the construct identifier; same Design ID across files
    # = same designed sequence assayed in both runs.
    if "Design ID" not in df.columns:
        raise KeyError(f"'Design ID' not in {path}: {list(df.columns)[:10]}")
    return df

df_a = load_run(RUN_A_PATH)
df_b = load_run(RUN_B_PATH)

print(f"{RUN_A_LABEL}: {len(df_a)} wells")
print(f"{RUN_B_LABEL}: {len(df_b)} wells")

# -----------------------------------------------------------------------------
# Pair on Design ID (the construct), not well position
# -----------------------------------------------------------------------------

def collisions(df, key):
    counts = df[key].value_counts()
    return counts[counts > 1]

for label, df in [(RUN_A_LABEL, df_a), (RUN_B_LABEL, df_b)]:
    dups = collisions(df, "Design ID")
    if len(dups):
        print(f"WARNING: duplicated Design IDs in {label}: {dups.head().to_dict()}")

keep = ["Design ID", "Source Well", "Name"] + CONTINUOUS_FIELDS + BOOLEAN_FIELDS
paired = df_a[keep].merge(
    df_b[keep],
    on="Design ID",
    how="inner",
    suffixes=(f"__{RUN_A_LABEL}", f"__{RUN_B_LABEL}"),
)

print(f"Paired designs (inner join on Design ID): {len(paired)}")
unmatched_a = set(df_a["Design ID"]) - set(df_b["Design ID"])
unmatched_b = set(df_b["Design ID"]) - set(df_a["Design ID"])
if unmatched_a or unmatched_b:
    print(f"Unmatched in {RUN_A_LABEL}: {len(unmatched_a)}; in {RUN_B_LABEL}: {len(unmatched_b)}")

paired.to_csv(OUT_DIR / "expression_replication_paired.csv", index=False)

# -----------------------------------------------------------------------------
# Continuous-field consistency
# -----------------------------------------------------------------------------

def continuous_stats(field: str, paired: pd.DataFrame) -> dict:
    a = paired[f"{field}__{RUN_A_LABEL}"].to_numpy(dtype=float)
    b = paired[f"{field}__{RUN_B_LABEL}"].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return {"field": field, "n_paired": len(a)}

    pearson_r, pearson_p = stats.pearsonr(a, b)
    spearman_r, spearman_p = stats.spearmanr(a, b)
    slope, intercept, r, p, se = stats.linregress(a, b)

    diff = b - a
    mean_ab = (a + b) / 2.0
    # symmetric percent diff, robust to mean ~ 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = np.where(np.abs(mean_ab) > 0, np.abs(diff) / np.abs(mean_ab), np.nan)

    return {
        "field": field,
        "n_paired": int(len(a)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "ols_slope": slope,
        "ols_intercept": intercept,
        "ols_R2": r ** 2,
        "mean_diff_B_minus_A": float(np.mean(diff)),
        "sd_diff": float(np.std(diff, ddof=1)),
        "median_abs_diff": float(np.median(np.abs(diff))),
        "mean_abs_pct_diff": float(np.nanmean(pct_diff)),
        "median_abs_pct_diff": float(np.nanmedian(pct_diff)),
        "mean_A": float(np.mean(a)),
        "mean_B": float(np.mean(b)),
    }

continuous_rows = [continuous_stats(f, paired) for f in CONTINUOUS_FIELDS]

# -----------------------------------------------------------------------------
# Boolean-field agreement (SEC "correct Vel" calls)
# -----------------------------------------------------------------------------

def boolean_stats(field: str, paired: pd.DataFrame) -> dict:
    a = paired[f"{field}__{RUN_A_LABEL}"].astype(bool).to_numpy()
    b = paired[f"{field}__{RUN_B_LABEL}"].astype(bool).to_numpy()

    both_true = int(np.sum(a & b))
    both_false = int(np.sum(~a & ~b))
    a_only = int(np.sum(a & ~b))      # True in A, False in B
    b_only = int(np.sum(~a & b))      # False in A, True in B

    agreement = (both_true + both_false) / len(a) if len(a) else np.nan
    kappa = cohen_kappa_score(a, b) if len(a) and a.var() + b.var() > 0 else np.nan

    return {
        "field": field,
        "n_paired": int(len(a)),
        "agreement": agreement,
        "cohens_kappa": float(kappa) if kappa is not np.nan else np.nan,
        "both_true": both_true,
        "both_false": both_false,
        f"true_in_{RUN_A_LABEL}_only": a_only,
        f"true_in_{RUN_B_LABEL}_only": b_only,
        f"frac_true_{RUN_A_LABEL}": float(np.mean(a)),
        f"frac_true_{RUN_B_LABEL}": float(np.mean(b)),
    }

boolean_rows = [boolean_stats(f, paired) for f in BOOLEAN_FIELDS]

# -----------------------------------------------------------------------------
# Save summary
# -----------------------------------------------------------------------------

summary_continuous = pd.DataFrame(continuous_rows)
summary_boolean = pd.DataFrame(boolean_rows)
summary_continuous.to_csv(OUT_DIR / "expression_replication_summary_continuous.csv", index=False)
summary_boolean.to_csv(OUT_DIR / "expression_replication_summary_boolean.csv", index=False)

print("\n=== Continuous-field replication ===")
print(summary_continuous.to_string(index=False))
print("\n=== Boolean-field replication ===")
print(summary_boolean.to_string(index=False))

# -----------------------------------------------------------------------------
# Scatter plots
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, (field, log) in zip(
    axes.ravel(),
    [("yield_per_Leq", False), ("yield_per_Leq", True),
     ("tot_yield", False), ("tot_yield", True)],
):
    a = paired[f"{field}__{RUN_A_LABEL}"].to_numpy(dtype=float)
    b = paired[f"{field}__{RUN_B_LABEL}"].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    ax.scatter(a, b, s=14, alpha=0.7)
    lim_lo = float(np.nanmin([a.min(), b.min()])) if len(a) else 0
    lim_hi = float(np.nanmax([a.max(), b.max()])) if len(a) else 1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    r = stats.pearsonr(a, b)[0] if len(a) >= 3 else np.nan
    ax.set_title(f"{field}{' (log)' if log else ''}  Pearson r = {r:.3f}  n={len(a)}")
    ax.set_xlabel(f"{field} — {RUN_A_LABEL}")
    ax.set_ylabel(f"{field} — {RUN_B_LABEL}")

fig.tight_layout()
fig.savefig(OUT_DIR / "expression_replication_scatter.png", dpi=200)
print(f"\nWrote: {OUT_DIR / 'expression_replication_scatter.png'}")
print(f"Wrote: {OUT_DIR / 'expression_replication_paired.csv'}")
print(f"Wrote: {OUT_DIR / 'expression_replication_summary_continuous.csv'}")
print(f"Wrote: {OUT_DIR / 'expression_replication_summary_boolean.csv'}")
