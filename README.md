# nmr_app_utilities

Code and metadata accompanying the *NMR revelations* paper. The repository contains the analysis pipeline,
design-time launch configurations, and a `data/` directory with both local artefacts and
pointers to larger externally-deposited data.

## Repository layout

```
nmr_app_utilities/
├── environment.yml            # conda env pinning Python deps
├── analysis_scripts/          # dry/wet-lab analysis + figure scripts
│   ├── README.md              # detailed pipeline description
│   ├── config.py              # central data-path resolver (uses NMR_PAPER_DATA env var)
│   ├── flatten_models.py
│   ├── compute_metrics.py
│   ├── compute_all_metrics.sh
│   ├── aggregate_ensemble_metrics.py
│   ├── expression_replication.py
│   ├── figures/              
│   │   ├── make_hsqc_correlation_figure.py
│   │   ├── make_relaxation_figures.py
│   │   ├── make_dendo.py
│   │   ├── clustering_utility.py
│   │   ├── dssp_filter.py
│   │   ├── dssp_full.py
│   │   └── sec_data.py
│   └── outputs/               # generated CSVs / figures (gitignored)
├── design_scripts/            # launch configs / version notes for generators
│   ├── rfdiffusion.md
│   ├── proteina.md
│   └── foldseek.sh
└── data/
    ├── MANIFEST.md            # catalogue of all data items + retrieval pointers
    ├── all_metrics_and_exp_results.csv   # combined dry- + wet-lab table (one row per design)
    ├── model_ensembles.tar.zst           # flattened prediction ensembles
    ├── designable/            # top-ranked CIF structures (ptm > 0.70, RMSD_ca < 2.0)
    ├── non_designable/        # remaining CIF structures
    ├── drylab/                # dry-lab artefacts
    │   ├── boltz2rank_scores/ # Boltz2Rank confidence JSONs + pLDDT npz files
    │   ├── design_metrics/    # boltz_metrics_designs.csv, dssp.csv
    │   ├── dssp/
    │   ├── ensemble_metrics/  # combined_dynamics_metrics.csv + per-source pLDDT/RMSF CSVs
    │   └── raw_designs/       # metrics.parquet, designable.tar.zst, non_designable.tar.zst
    └── wetlab/                # wet-lab artefacts (NMR + SEC/expression)
```

## Quick start (reviewers)

1. **Get the data.** Bulk inputs live outside the repo. See `data/MANIFEST.md`
   for the Zenodo DOI, BMRB ID and the wet-lab raw-data location. Unpack
   the Zenodo deposit into a local directory — everything under `deposit/data/`
   maps directly to the layout `analysis_scripts/config.py` expects.

2. **Tell the code where the data lives.**

   ```bash
   export NMR_PAPER_DATA=/path/to/deposit/data
   ```

   When unset, `config.py` falls back to `~/Desktop/nmr_revelations_paper/data`.
   No other path edits are needed — every script reads from `config.py`.

3. **Create the environment.**

   ```bash
   conda env create -f environment.yml
   conda activate nmr-paper
   ```

4. **Re-run the pipeline.** See `analysis_scripts/README.md` for the full
   sequence. The short version:

   ```bash
   cd analysis_scripts

   # pipeline (only needed if rebuilding combined_dynamics_metrics.csv from raw ensembles)
   ./compute_all_metrics.sh                       # pLDDT + RMSF per design × source
   python aggregate_ensemble_metrics.py           # combined_dynamics_metrics.csv

   # main figures
   python figures/make_hsqc_correlation_figure.py
   python figures/make_relaxation_figures.py
   python figures/make_dendo.py
   python expression_replication.py               # SEC/yield consistency between replicate runs

   # auxiliary figures
   python figures/dssp_filter.py                  # produces pdb_clustered.csv (input to dssp_full)
   python figures/dssp_full.py
   python figures/clustering_utility.py
   python figures/sec_data.py
   ```

   Outputs land in `analysis_scripts/outputs/`.

## Deposit workflow (maintainers)

The Zenodo deposit is assembled from `data/` and checked into `deposit/`.
