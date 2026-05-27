# Analysis Scripts

Code that turns raw structure-prediction ensembles and wet-lab NMR / SEC
outputs into the publication figures and supporting tables. Two stages:

1. **Pipeline**: flatten raw predictions → compute per-residue pLDDT and
   RMSF → aggregate to a per-design wide table (`combined_dynamics_metrics.csv`).
2. **Figures**: merge the per-design + per-residue metrics with wet-lab
   observables (HSQC peak lists, R1 / R2 / HetNOE relaxation, SEC traces,
   expression yields) and render figure panels + summary CSVs.

All paths are resolved by `config.py` from the `NMR_PAPER_DATA` environment
variable. No script has hard-coded user-specific paths — point that env var
at the unpacked data root (see `deposit/MANIFEST_zenodo.csv`) and everything works:

```bash
export NMR_PAPER_DATA=/path/to/data
```

When unset, `config.py` falls back to `~/Desktop/nmr_revelations_paper/data`.

---

## Layout

```
analysis_scripts/
├── config.py                          # single source of truth for data paths
├── flatten_models.py                  # raw predictions → uniform <design>/model_{0..N}.cif
├── compute_metrics.py                 # per-residue pLDDT / RMSF CLI
├── compute_all_metrics.sh             # driver over every ensemble source
├── aggregate_ensemble_metrics.py      # combined_dynamics_metrics.csv
├── expression_replication.py          # SEC / yield consistency across replicate plates
├── figures/                           # figure-generation scripts
│   ├── make_hsqc_correlation_figure.py
│   ├── make_relaxation_figures.py
│   ├── make_dendo.py
│   ├── clustering_utility.py
│   ├── dssp_filter.py                 # writes outputs/dssp/pdb_clustered.csv
│   ├── dssp_full.py                   # consumes the above
│   └── sec_data.py
└── outputs/                           # generated CSVs / figures (gitignored)
    ├── expression_replication/
    ├── hsqc/
    ├── relaxation/
    ├── dssp/
    └── figures/                       # publication-ready panels
        ├── hsqc_dendrogram/
        ├── clustering_utility/
        ├── dssp_full/
        └── sec_data/
```

---

## Pipeline overview

```
raw predictions (AF3 seed folders, Boltz1/2 prediction folders)
        │
        ▼  flatten_models.py
flattened ensembles: <ensemble_dir>/<design_name>/model_{0..N}.cif (+ JSONs / pLDDT npz)
        │
        ▼  compute_all_metrics.sh  →  compute_metrics.py {bfactor,plddt,rmsf}
per-source per-residue CSVs:  ensemble_metrics/<source>_plddt.csv,  <source>_rmsf.csv
        │
        ▼  aggregate_ensemble_metrics.py
combined_dynamics_metrics.csv  (one row per design × source, wide format)
        │
        ▼  figures/{make_hsqc_correlation_figure, make_relaxation_figures, make_dendo, ...}
publication figures + summary CSVs (R² tables, chamfer distance matrix, etc.)
```

For the shipped deposit, every output of the pipeline through
`combined_dynamics_metrics.csv` is already present, so reviewers who just
want the figures can skip steps 1–3.

---

## Pipeline scripts (re-build metrics from scratch)

### `flatten_models.py`
Reorganizes raw prediction outputs into a uniform `<design_name>/model_{i}.cif`
layout that downstream tools expect.

- **AF3 input**: `batch_X/<design_name>/seed-S_sample-0/model.cif` (+ `summary_confidences.json`)
- **Boltz2 input**: `batch_X/boltz_results_batch_X/predictions/<design_name>/<design_name>_model_N.cif` (+ `confidence_*.json`, `plddt_*.npz`)
- **Output**: `<output_dir>/<design_name>/model_{0..N}.cif` plus `model_confidences.csv` of pTM values.
- `--boltz` for Boltz2 inputs (default is AF3); `--copy` to copy instead of move; `--single-batch` for a single batch dir.

### `compute_metrics.py`
Unified CLI for per-residue metrics from a flattened ensemble. Three subcommands:

- `bfactor <folder>`: reads `B_iso_or_equiv` from CIF (AF3 / Boltz2 store pLDDT here) → per-residue pLDDT CSV.
- `plddt <folder>`: reads pLDDT arrays from Boltz `plddt_*.npz` files → per-residue pLDDT CSV.
- `rmsf <folder>`: superimposes all models in each subfolder on the first via peptide backbone and writes mean/min/max RMSF per residue.

All write a CSV with at minimum `folder_name, residue_id, residue_name, …`.

### `compute_all_metrics.sh`
Driver that runs `compute_metrics.py` over every subdirectory of
`$NMR_PAPER_DATA/drylab/model_ensembles/`. Folder-name heuristic picks the
right pLDDT extractor (`af3*` → bfactor, `boltz*` → npz) and computes RMSF
for everything. Outputs to `$NMR_PAPER_DATA/drylab/ensemble_metrics/<folder>_{plddt,rmsf}.csv`.

### `aggregate_ensemble_metrics.py`
Notebook-style script (`# %%` cells) that:
1. Auto-discovers every `*_plddt.csv` / `*_rmsf.csv` in `ensemble_metrics/`,
2. Tags each by source (e.g. `af3_recycle_0`, `boltz2_recycle_3`),
3. Aggregates to one row per design × source (mean/std/min/max), then pivots wide,
4. Writes `combined_dynamics_metrics.csv`.

---

## Main figure scripts

### `figures/make_hsqc_correlation_figure.py`
HSQC correlation pipeline. Loads HSQC summary + peak lists, expression
metrics, DSSP, and the per-design dynamics aggregate.
Derives per-design NMR observables (linewidths, dispersion, peak count,
intensity CV, etc.), merges with computational metrics, applies the
spectral-quality + intensity-outlier filters, and renders:

- `outputs/hsqc/combined_metrics_before_filter.parquet`
- `outputs/hsqc/nmr_comp_correlations/fig_hsqc_*.svg` (6 publication panels)
- `outputs/hsqc/nmr_comp_correlations/<observable>_correlations.html` (interactive scatter per observable)

### `figures/make_relaxation_figures.py`
Per-residue R₁ / R₂ / HetNOE correlated against every pLDDT / RMSF
source via linear regression. Reads the per-source per-residue CSVs from
`ensemble_metrics/` (so all six source × recycle columns appear in the heatmap).

Outputs:
- `outputs/relaxation/relaxation_full_merged_data.csv`
- `outputs/relaxation/relaxation_r2_results_per_folder.csv`
- `outputs/relaxation/relaxation_r2_results_pooled.csv`
- `outputs/relaxation/figures/fig_relaxation_*.svg` (7 panels)

`MSG_TAG_OFFSET = 2` accounts for the cleaved Met when followed by the
flexible linker in *E. coli* expression — change if you switch construct.

### `figures/make_dendo.py`
HSQC dendrogram. Uses the same loader as `make_hsqc_correlation_figure.py`,
then:

1. Computes pairwise **Chamfer distance** between 2D peak lists
   (`(¹H, ¹⁵N)` clouds) — preferred to Hausdorff because it averages
   min-distances instead of being dominated by a single outlier peak.
2. Hierarchical clustering (scipy linkage).
3. Renders the dendrogram with a miniature HSQC scatter at each leaf,
   plus a Plotly HTML version.
4. Writes `outputs/figures/hsqc_dendrogram/chamfer_distance_matrix.csv`.

### `expression_replication.py`
Quantifies expression / SEC reproducibility between the duplicate
`NMR_RUN_2_P3` plate runs (`250910` original vs `250913_REDO`):

- For continuous fields (`yield_per_Leq`, `tot_yield`): Pearson + Spearman,
  OLS R², median |%diff|.
- For boolean SEC quality flags (`correct_Vel_95CI`, `correct_Vel_99CI`):
  agreement, Cohen's κ.

Outputs four files under `outputs/expression_replication/`: paired-by-design
CSV, two summary CSVs, and a 2×2 scatter PNG. Numbers cited in the paper:
Pearson r = 0.85 for yield (n=96); κ = 0.94 for `correct_Vel_95CI`,
κ = 0.89 for `correct_Vel_99CI`.

---

## Auxiliary figure scripts

### `figures/clustering_utility.py`
Sampling-strategy ternary plot — compares H/E/C composition distributions of
(a) the actually-sampled subset, (b) what RMSD-based selection would have
picked, (c) what pTM-based selection would have picked, against the full
designable library. Writes `outputs/figures/clustering_utility/ternary_plot_density.pdf`.

### `figures/dssp_filter.py`
Walks every per-design DSSP parquet under
`$NMR_PAPER_DATA/drylab/dssp/pdb/`, restricts to MMseqs cluster
representatives (from `mmseqs_result_cluster.tsv`), and writes
`outputs/dssp/pdb_clustered.csv`. **Must run before `dssp_full.py`.**

### `figures/dssp_full.py`
Combined DSSP ternary plot — overlays the natural-protein KDE (from
`pdb_clustered.csv`) with sampled Proteina and RFdiffusion scatter points.
Writes `outputs/figures/dssp_full/dssp_comparison.svg`.

### `figures/sec_data.py`
For each design `category`, clusters the SEC chromatograms and renders
overlay + offset-stack plots. Writes
`outputs/figures/sec_data/all_sec_traces_<category>.png`.

---

## Suggested execution order

```bash
export NMR_PAPER_DATA=/path/to/data

# ── pipeline (skip if combined_dynamics_metrics.csv is already in $NMR_PAPER_DATA) ──
python flatten_models.py <af3_batch_root>    "$NMR_PAPER_DATA"/drylab/model_ensembles/af3_recycle_0
python flatten_models.py <boltz2_batch_root> "$NMR_PAPER_DATA"/drylab/model_ensembles/boltz2_recycle_0 --boltz
./compute_all_metrics.sh
python aggregate_ensemble_metrics.py

# ── main figures ──
python figures/make_hsqc_correlation_figure.py
python figures/make_relaxation_figures.py
python figures/make_dendo.py
python expression_replication.py

# ── auxiliary figures ──
python figures/dssp_filter.py        # produces outputs/dssp/pdb_clustered.csv
python figures/dssp_full.py          # consumes the above
python figures/clustering_utility.py
python figures/sec_data.py
```

All outputs land under `analysis_scripts/outputs/` (gitignored).

---

## Environment

```bash
conda env create -f ../environment.yml
conda activate nmr-paper
```

`python-ternary` is installed via `pip` (it's not on conda-forge) and is
required by the two DSSP figure scripts.

---

## Reproducing the figures (for reviewers)

Assumes the data deposit is unpacked at `$NMR_PAPER_DATA/` with the layout
described in `deposit/MANIFEST_zenodo.csv`.

### 1. Environment

```bash
conda env create -f environment.yml
conda activate nmr-paper
```

### 2. Point at the data

```bash
export NMR_PAPER_DATA=/path/to/unpacked/deposit/data
```

No script edits required.

### 3. (Skip if `ensemble_metrics/` is included in the deposit) Recompute per-residue metrics

```bash
cd analysis_scripts
./compute_all_metrics.sh
python aggregate_ensemble_metrics.py
```

Expected runtime: ~5–15 min per source on a laptop (RMSF dominates).
Inputs: `$NMR_PAPER_DATA/drylab/model_ensembles/<source>/<design>/model_{0..N}.cif`
(+ `plddt_*.npz` for Boltz sources).
Outputs: `$NMR_PAPER_DATA/drylab/ensemble_metrics/<source>_{plddt,rmsf}.csv` and
`combined_dynamics_metrics.csv`.

### 4. Regenerate figures

```bash
python figures/make_hsqc_correlation_figure.py
python figures/make_relaxation_figures.py
python figures/make_dendo.py
python expression_replication.py

python figures/dssp_filter.py
python figures/dssp_full.py
python figures/clustering_utility.py
python figures/sec_data.py
```

Outputs land in `analysis_scripts/outputs/`.

### 5. Expected outputs

| Script | Artefact | Compare to shipped file |
|--------|----------|-------------------------|
| `make_hsqc_correlation_figure.py` | `outputs/hsqc/combined_metrics_before_filter.parquet` + 6 SVG panels + 18 HTML scatter | values, not pixels — small float differences are OK |
| `make_relaxation_figures.py` | `outputs/relaxation/relaxation_r2_results_pooled.csv` + 7 SVG panels | top hit: Boltz-1 pLDDT (r0) vs R₁_mean, R² ≈ 0.12; matches the paper figure |
| `make_dendo.py` | `outputs/figures/hsqc_dendrogram/chamfer_distance_matrix.csv` + dendrogram PDF/SVG | distances should agree to ≥ 5 sig-figs |
| `expression_replication.py` | `outputs/expression_replication/*.csv` + scatter PNG | Pearson r = 0.85; κ = 0.94 / 0.89 |
| `clustering_utility.py` | `outputs/figures/clustering_utility/ternary_plot_density.pdf` | 384 designs in each of three sampling strategies |
| `dssp_filter.py` + `dssp_full.py` | `outputs/dssp/pdb_clustered.csv` (3659 rows) + `outputs/figures/dssp_full/dssp_comparison.svg` | 3650 PDB + 192 Proteina + 192 RFdiffusion |
| `sec_data.py` | 3 SEC overview PNGs | one per category |

If a numerical table diverges by more than rounding, the most likely cause
is a missing `*_plddt.csv` / `*_rmsf.csv` source in
`ensemble_metrics/` (the relaxation script silently falls back to four
pre-aggregated columns, which caps R² around 0.03 and blanks the heatmap
annotations).

---

## Notes on construct / sequence handling

- HSQC `Sequence` strings are stripped of trailing `*` and newline
  characters during merge — keep the parquet sources as-is; the cleanup
  happens in code.
- Relaxation residue indices are shifted by `MSG_TAG_OFFSET = 2` to
  account for the cleaved N-terminal Met before the flexible linker.
  Document this in any per-residue tables exported to the supplement.
- The HSQC `Quality` flag is set manually during peak picking from the raw
  spectrum, before any correlation with dry-lab metrics is computed. The
  filter `Quality ∈ {Medium, High}` is applied at line 207 of
  `make_hsqc_correlation_figure.py`. The intensity-outlier filter at line
  210 drops 2 of the 238 quality-passing designs whose
  `intensity_max ≥ 1×10⁸` (anomalously bright peaks, indicative of
  processing artefacts).

---

## Data inputs catalogue

See `../deposit/MANIFEST_zenodo.csv` for the sha256 of `data.tar.zst`
(the single archive uploaded to Zenodo).
