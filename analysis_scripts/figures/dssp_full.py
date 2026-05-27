"""
Combined DSSP ternary plot — overlay the helix/sheet/loop distributions of
(a) the natural PDB reference set (KDE + contours), (b) sampled Proteina
designs, and (c) sampled RFDiffusion designs.

Inputs:
  - $NMR_PAPER_DATA/drylab/design_metrics/boltz_metrics/boltz_metrics_designs.csv
  - analysis_scripts/outputs/dssp/pdb_clustered.csv   (produced by dssp_filter.py)
  - $NMR_PAPER_DATA/drylab/dssp/all_designs/*.parquet

Outputs:
  - analysis_scripts/outputs/figures/dssp_full/dssp_comparison.svg
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BOLTZ_METRICS_CSV, DSSP_ALL_DESIGNS_DIR, OUTPUTS_DIR  # noqa: E402

PDB_CLUSTERED_CSV = OUTPUTS_DIR / "dssp" / "pdb_clustered.csv"  # from dssp_filter.py
OUT_DIR = OUTPUTS_DIR / "figures" / "dssp_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_dssp_results(all_results):
    """Return per-row [sheet, helix, loop] fractions (note swapped order)."""
    proportions = []
    for sse in all_results:
        if not isinstance(sse, str):
            continue
        h = sum(1 for s in sse if s in "HGI")
        e = sum(1 for s in sse if s in "EB")
        l = sum(1 for s in sse if s in "TSC- ")
        total = h + e + l
        if total > 0:
            proportions.append([e / total, h / total, l / total])
    return np.array(proportions)


def ternary_to_cartesian(points, scale):
    """[sheet, helix, loop] → (x, y) in matplotlib coords."""
    out = []
    for p in points:
        s, h, l = p[0] / scale, p[1] / scale, p[2] / scale
        x = h + s / 2
        y = s * np.sqrt(3) / 2
        out.append([x * scale, y * scale])
    return np.array(out)


def main() -> None:
    if not PDB_CLUSTERED_CSV.exists():
        raise FileNotFoundError(
            f"{PDB_CLUSTERED_CSV} not found. Run "
            "`python -m figures.dssp_filter` first."
        )
    metrics_wetlab = pd.read_csv(BOLTZ_METRICS_CSV)
    DESIGN_IDS = metrics_wetlab["design_name"].tolist()

    pdb = pd.read_csv(PDB_CLUSTERED_CSV)
    pdb = pdb[pdb["dssp_string"].notna()]

    all_parqs = [pd.read_parquet(p) for p in DSSP_ALL_DESIGNS_DIR.glob("*.parquet")]
    if not all_parqs:
        raise FileNotFoundError(f"no parquet files under {DSSP_ALL_DESIGNS_DIR}")
    all_designs = pd.concat(all_parqs, ignore_index=True)
    all_designs["model_name"] = all_designs["file_name"].str.replace(".cif", "", regex=False)
    all_designs["sampled"]    = all_designs["model_name"].isin(DESIGN_IDS)
    all_designs["proteina"]   = all_designs["model_name"].str.contains("n_100")

    proportions_pdb = process_dssp_results(pdb["dssp_string"].values)
    sampled_proteina    = all_designs[(all_designs["sampled"]) & (all_designs["proteina"])]
    sampled_rfdiffusion = all_designs[(all_designs["sampled"]) & (~all_designs["proteina"])]
    proportions_sampled_proteina    = process_dssp_results(sampled_proteina["dssp_string"].values)
    proportions_sampled_rfdiffusion = process_dssp_results(sampled_rfdiffusion["dssp_string"].values)

    print(f"PDB:                  {len(proportions_pdb)} structures (KDE)")
    print(f"Proteina (sampled):   {len(proportions_sampled_proteina)}")
    print(f"RFDiffusion (sampled):{len(proportions_sampled_rfdiffusion)}")

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    scale = 100
    _, tax = ternary.figure(ax=ax, scale=scale)

    points_pdb = proportions_pdb * scale
    cart_pdb = ternary_to_cartesian(points_pdb, scale)

    # KDE grid
    grid_steps = 200
    grid_cart = []
    for i in range(grid_steps + 1):
        for j in range(grid_steps + 1):
            if i + j <= grid_steps:
                sheet = i * scale / grid_steps
                helix = j * scale / grid_steps
                s, h = sheet / scale, helix / scale
                x = h + s / 2
                y = s * np.sqrt(3) / 2
                grid_cart.append([x * scale, y * scale])
    grid_cart = np.array(grid_cart)

    kde_pdb = gaussian_kde(cart_pdb.T, bw_method=0.15)
    density_pdb = kde_pdb(grid_cart.T)
    density_pdb_norm = (density_pdb - density_pdb.min()) / (density_pdb.max() - density_pdb.min())

    triang = Triangulation(grid_cart[:, 0], grid_cart[:, 1])
    color_pdb         = "#2196F3"
    color_proteina    = "#E91E63"
    color_rfdiffusion = "#4CAF50"

    ax.tricontourf(triang, density_pdb_norm, levels=[0.3, 0.5, 0.7, 0.9, 1.0],
                   colors=[color_pdb], alpha=0.12)
    ax.tricontour(triang, density_pdb_norm, levels=np.linspace(0.2, 0.9, 4),
                  colors=color_pdb, linewidths=2.5, alpha=0.9, linestyles="solid")

    if len(proportions_sampled_proteina) > 0:
        cart = ternary_to_cartesian(proportions_sampled_proteina * scale, scale)
        ax.scatter(cart[:, 0], cart[:, 1], c=color_proteina, s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=10)
    if len(proportions_sampled_rfdiffusion) > 0:
        cart = ternary_to_cartesian(proportions_sampled_rfdiffusion * scale, scale)
        ax.scatter(cart[:, 0], cart[:, 1], c=color_rfdiffusion, s=60, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=10)

    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=10, color="#BDBDBD", alpha=0.4, linewidth=0.5)
    tax.left_axis_label("← Coil",  fontsize=22, offset=0.14, weight="bold")
    tax.right_axis_label("← Sheet", fontsize=22, offset=0.14, weight="bold")
    tax.bottom_axis_label("Helix →", fontsize=22, offset=0.02, weight="bold")
    tax.ticks(axis="lbr", linewidth=1, multiple=10, offset=0.025,
              tick_formats="%.0f%%", fontsize=12)
    tax.set_title("Secondary Structure Composition Comparison",
                  fontsize=26, weight="bold", pad=20)

    legend_elements = [
        Line2D([0], [0], marker="o", color=color_pdb, markerfacecolor=color_pdb,
               markersize=10, label="PDB (small proteins)", linestyle="solid", linewidth=3),
        Line2D([0], [0], marker="o", color=color_proteina, markerfacecolor=color_proteina,
               markersize=10, label="Proteina", linestyle="solid", linewidth=3),
        Line2D([0], [0], marker="o", color=color_rfdiffusion, markerfacecolor=color_rfdiffusion,
               markersize=10, label="RFDiffusion", linestyle="solid", linewidth=3),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12,
              framealpha=0.95, edgecolor="gray")

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    plt.tight_layout()
    out = OUT_DIR / "dssp_comparison.svg"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
