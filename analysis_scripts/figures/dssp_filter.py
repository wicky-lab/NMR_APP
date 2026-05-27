"""
DSSP filtering step — read every per-design DSSP parquet under
$NMR_PAPER_DATA/drylab/dssp/pdb/, restrict to MMseqs cluster representatives,
and write the filtered table to outputs/dssp/pdb_clustered.csv. Consumed
downstream by dssp_full.py.

Inputs:
  - $NMR_PAPER_DATA/drylab/dssp/pdb/*.parquet
  - $NMR_PAPER_DATA/drylab/dssp/mmseqs_result_cluster.tsv

Outputs:
  - analysis_scripts/outputs/dssp/pdb_clustered.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DSSP_PDB_DIR, MMSEQS_CLUSTER_TSV, OUTPUTS_DIR  # noqa: E402

OUT_DIR = OUTPUTS_DIR / "dssp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    all_parqs = [pd.read_parquet(p) for p in DSSP_PDB_DIR.glob("*.parquet")]
    if not all_parqs:
        raise FileNotFoundError(f"no parquet files under {DSSP_PDB_DIR}")
    pdb = pd.concat(all_parqs, ignore_index=True)
    pdb["pdb_id"] = pdb["file_name"].str.replace(".cif", "", regex=False).str.upper()

    cluster_tsv = pd.read_csv(MMSEQS_CLUSTER_TSV, sep="\t", header=None)
    rep_ids = cluster_tsv[0].str.split("_").str[0].unique()

    pdb = pdb[pdb["pdb_id"].isin(rep_ids)]

    out = OUT_DIR / "pdb_clustered.csv"
    pdb.to_csv(out, index=False)
    print(f"wrote {out}  ({len(pdb)} rows)")


if __name__ == "__main__":
    main()
