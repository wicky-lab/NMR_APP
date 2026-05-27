"""
Sampling-strategy ternary plot — compare the H/E/C composition distribution
of (a) the actually-sampled subset, (b) what RMSD-based selection would have
picked, (c) what pTM-based selection would have picked, against the full
designable library.

Inputs:
  - $NMR_PAPER_DATA/drylab/design_metrics/boltz_metrics/FINAL_METRICS.parquet
  - $NMR_PAPER_DATA/all_metrics_and_exp_results.csv
  - $NMR_PAPER_DATA/drylab/dssp/all_designs/*.parquet

Outputs:
  - analysis_scripts/outputs/figures/clustering_utility/ternary_plot_density.pdf
"""

import colorsys
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ALL_METRICS_AND_EXP_RESULTS_CSV,
    DSSP_ALL_DESIGNS_DIR,
    FINAL_METRICS_PARQUET,
    OUTPUTS_DIR,
)

OUT_DIR = OUTPUTS_DIR / "figures" / "clustering_utility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def parse_dssp_composition(dssp_string):
    if pd.isna(dssp_string) or len(dssp_string) == 0:
        return np.nan, np.nan, np.nan
    total = len(dssp_string)
    n_helix = sum(1 for c in dssp_string if c in "HGI")
    n_sheet = sum(1 for c in dssp_string if c in "EB")
    n_coil  = total - n_helix - n_sheet
    return n_helix / total, n_sheet / total, n_coil / total


def ternary_coords(h, e, c):
    total = h + e + c + 1e-10
    x = (h + 0.5 * e) / total
    y = (np.sqrt(3) / 2) * e / total
    return x, y


def make_density_cmap(hex_color, name="custom"):
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r_l, g_l, b_l = colorsys.hls_to_rgb(h, min(l + 0.22, 0.78), s)
    r_d, g_d, b_d = colorsys.hls_to_rgb(h, max(l - 0.35, 0.08), max(s, 0.7))
    return LinearSegmentedColormap.from_list(
        name, [(r_l, g_l, b_l), (r, g, b), (r_d, g_d, b_d)], N=256,
    )


def main() -> None:
    all_metrics = pd.read_parquet(FINAL_METRICS_PARQUET, engine="fastparquet")
    full_merged = pd.read_csv(ALL_METRICS_AND_EXP_RESULTS_CSV)

    all_metrics["model_name"] = all_metrics["file_name"].str.replace(".cif", "", regex=False)
    all_metrics["sampled"]    = all_metrics["model_name"].isin(full_merged["design_name"].values)

    all_parqs = [pd.read_parquet(p) for p in DSSP_ALL_DESIGNS_DIR.glob("*.parquet")]
    if not all_parqs:
        raise FileNotFoundError(f"no parquet files under {DSSP_ALL_DESIGNS_DIR}")
    all_designs_dssp = pd.concat(all_parqs, ignore_index=True)
    all_designs_dssp["model_name"] = all_designs_dssp["file_name"].str.replace(".cif", "", regex=False)
    all_metrics = all_metrics.merge(all_designs_dssp, how="left", on="model_name")

    ss = all_metrics["dssp_string"].apply(parse_dssp_composition)
    all_metrics["frac_helix"] = ss.apply(lambda x: x[0])
    all_metrics["frac_sheet"] = ss.apply(lambda x: x[1])
    all_metrics["frac_coil"]  = ss.apply(lambda x: x[2])

    n_samples = 384
    actual_sampled = all_metrics[all_metrics["sampled"]].copy()
    rmsd_best_per_ref = all_metrics.loc[
        all_metrics.groupby("reference_path")["RMSD_ca"].idxmin()]
    rmsd_sampled = rmsd_best_per_ref.nsmallest(n_samples, "RMSD_ca").copy()
    ptm_best_per_ref = all_metrics.loc[
        all_metrics.groupby("reference_path")["ptm"].idxmax()]
    ptm_sampled = ptm_best_per_ref.nlargest(n_samples, "ptm").copy()

    print(f"actual sampled: {len(actual_sampled)}")
    print(f"RMSD-based:     {len(rmsd_sampled)}")
    print(f"pTM-based:      {len(ptm_sampled)}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=300)
    fig.subplots_adjust(wspace=0.05)

    datasets = [
        (actual_sampled, "Foldseek Sampling\n(Actual)", "#2ecc71"),
        (rmsd_sampled,   "RMSD-Based\nSampling",        "#e74c3c"),
        (ptm_sampled,    "pTM-Based\nSampling",         "#3498db"),
    ]

    all_x, all_y = ternary_coords(
        all_metrics["frac_helix"].values,
        all_metrics["frac_sheet"].values,
        all_metrics["frac_coil"].values,
    )
    np.random.seed(42)
    downsample_mask = np.random.random(len(all_x)) < 0.7
    all_x_ds = all_x[downsample_mask]
    all_y_ds = all_y[downsample_mask]

    SQ3_2 = np.sqrt(3) / 2
    arrow_t0, arrow_t1 = 0.30, 0.70
    arrow_props = dict(arrowstyle="-|>", lw=1.4, color="black", mutation_scale=8)
    off = 0.06

    for ax, (df, title, color) in zip(axes, datasets):
        ax.scatter(all_x_ds, all_y_ds, c="lightgray", s=5, alpha=0.3,
                   zorder=1, rasterized=True, clip_on=False)

        x, y = ternary_coords(
            df["frac_helix"].values, df["frac_sheet"].values, df["frac_coil"].values,
        )
        if len(x) > 2:
            try:
                kde = gaussian_kde(np.vstack([x, y]), bw_method=0.15)
                density = kde(np.vstack([x, y]))
            except np.linalg.LinAlgError:
                density = np.ones(len(x))
        else:
            density = np.ones(len(x))

        cmap = make_density_cmap(color)
        order = np.argsort(density)
        sc = ax.scatter(
            x[order], y[order], c=density[order], cmap=cmap,
            s=35, alpha=0.85, edgecolors="#b0b0b0", linewidths=0.4,
            zorder=3, rasterized=True, clip_on=False,
        )

        triangle = plt.Polygon([[0, 0], [1, 0], [0.5, SQ3_2]],
                               fill=False, edgecolor="black", linewidth=1.5, zorder=2)
        ax.add_patch(triangle)

        ax.annotate("", xy=(arrow_t1, -off), xytext=(arrow_t0, -off), arrowprops=arrow_props)
        ax.text(0.5, -off - 0.06, "Helix", ha="center", va="top", fontsize=13, fontweight="bold")

        x0_l, y0_l = 0.5 * (1 - arrow_t0), SQ3_2 * (1 - arrow_t0)
        x1_l, y1_l = 0.5 * (1 - arrow_t1), SQ3_2 * (1 - arrow_t1)
        nx_l, ny_l = -SQ3_2 * off, 0.5 * off
        ax.annotate("", xy=(x1_l + nx_l, y1_l + ny_l), xytext=(x0_l + nx_l, y0_l + ny_l),
                    arrowprops=arrow_props)
        ax.text(0.5 * (x0_l + x1_l) + nx_l * 2.8, 0.5 * (y0_l + y1_l) + ny_l * 2.8,
                "Coil", ha="center", va="center", fontsize=13, fontweight="bold", rotation=60)

        x0_r, y0_r = 1 * (1 - arrow_t0) + 0.5 * arrow_t0, SQ3_2 * arrow_t0
        x1_r, y1_r = 1 * (1 - arrow_t1) + 0.5 * arrow_t1, SQ3_2 * arrow_t1
        nx_r, ny_r = SQ3_2 * off, 0.5 * off
        ax.annotate("", xy=(x1_r + nx_r, y1_r + ny_r), xytext=(x0_r + nx_r, y0_r + ny_r),
                    arrowprops=arrow_props)
        ax.text(0.5 * (x0_r + x1_r) + nx_r * 2.8, 0.5 * (y0_r + y1_r) + ny_r * 2.8,
                "Sheet", ha="center", va="center", fontsize=13, fontweight="bold", rotation=-60)

        cax = inset_axes(ax, width="10%", height="44%", loc="center",
                         bbox_to_anchor=(0.83, 0.50, 0.2, 0.3),
                         bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_ticks([])
        cbar.outline.set_visible(False)
        cbar.set_label("Density", fontsize=11, labelpad=3)

        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.22, SQ3_2 + 0.08)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    out = OUT_DIR / "ternary_plot_density.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
