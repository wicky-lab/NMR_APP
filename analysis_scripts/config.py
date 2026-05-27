"""
Single source of truth for data locations used by every script in
`analysis_scripts/`. Resolve once, import everywhere.

The data root is taken from the `NMR_PAPER_DATA` environment variable; if
unset, it falls back to `~/Desktop/nmr_revelations_paper/data`. To re-run
the pipeline on a different machine, set the env var instead of editing
any script:

    export NMR_PAPER_DATA=/path/to/nmr_revelations_paper/data
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get(
        "NMR_PAPER_DATA",
        str(Path.home() / "Desktop" / "nmr_revelations_paper" / "data"),
    )
)

# --- drylab ----------------------------------------------------------------

DRYLAB_DIR = DATA_ROOT / "drylab"

DESIGN_METRICS_DIR = DRYLAB_DIR / "design_metrics" / "boltz_metrics"
BOLTZ_METRICS_CSV = DESIGN_METRICS_DIR / "boltz_metrics_designs.csv"
DSSP_CSV = DESIGN_METRICS_DIR / "dssp.csv"
# Per-design wide table for the whole design library (250,648 rows × 23 cols)
FINAL_METRICS_PARQUET = DESIGN_METRICS_DIR / "FINAL_METRICS.parquet"

ENSEMBLES_DIR = DRYLAB_DIR / "model_ensembles"
METRICS_PATH = DRYLAB_DIR / "ensemble_metrics"
COMBINED_DYNAMICS_CSV = METRICS_PATH / "combined_dynamics_metrics.csv"

# Joined wet+dry metrics table — maps NMR `source` (e.g. p3_A1) -> design member
ALL_METRICS_AND_EXP_RESULTS_CSV = DATA_ROOT / "all_metrics_and_exp_results.csv"

# DSSP secondary-structure dump + sequence clustering inputs
DSSP_DIR = DRYLAB_DIR / "dssp"
DSSP_ALL_DESIGNS_DIR = DSSP_DIR / "all_designs"      # per-design .parquet files
DSSP_PDB_DIR = DSSP_DIR / "pdb"                      # natural-protein reference set
MMSEQS_CLUSTER_TSV = DSSP_DIR / "mmseqs_result_cluster.tsv"

# --- wetlab ----------------------------------------------------------------

WETLAB_DIR = DATA_ROOT / "wetlab"

HSQC_DIR = WETLAB_DIR / "nmr" / "hsqc"
HSQC_SUMMARY_PARQUET = HSQC_DIR / "summary.parquet"
HSQC_PEAKS_PARQUET = HSQC_DIR / "peak_data.parquet"

RELAXATION_DIR = WETLAB_DIR / "nmr" / "relaxation_data"

EXPRESSION_DIR = WETLAB_DIR / "expression"
EXPRESSION_METRICS_CSV = EXPRESSION_DIR / "expression_metrics.csv"
EXPRESSION_RUNS_DIR = EXPRESSION_DIR / "tables" / "wetlab_data"
# Aggregate expression HDF5 (sec_data figure)
FULL_AGGREGATE_H5 = EXPRESSION_DIR / "full_aggregate_data.h5"

# --- script outputs --------------------------------------------------------

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
