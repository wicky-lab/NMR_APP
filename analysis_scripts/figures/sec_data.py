"""
SEC-trace overview figure — for each design `category`, cluster the SEC
chromatograms and plot them as overlay + offset stack.

Inputs:
  - $NMR_PAPER_DATA/wetlab/expression/full_aggregate_data.h5
  - $NMR_PAPER_DATA/all_metrics_and_exp_results.csv

Outputs:
  - analysis_scripts/outputs/figures/sec_data/all_sec_traces_<category>.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALL_METRICS_AND_EXP_RESULTS_CSV, FULL_AGGREGATE_H5, OUTPUTS_DIR  # noqa: E402

OUT_DIR = OUTPUTS_DIR / "figures" / "sec_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_sec_traces_overview(df: pd.DataFrame, output_dir: Path,
                             save_plots: bool = True, plot_dpi: int = 300) -> None:
    for cat, sel in df.groupby("category"):
        sel = sel.reset_index(drop=True)

        vol_light = np.vstack(sel["vol_light"].to_numpy())
        Abs_light = np.vstack(sel["Abs_light"].to_numpy())
        clustered_idx = hierarchy.dendrogram(
            hierarchy.linkage(Abs_light, method="average", optimal_ordering=True),
            no_plot=True,
        )["leaves"]

        delta = np.max([np.max(x) for x in sel.Abs.to_numpy()]) / 20
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        for i, r in sel.iterrows():
            ax[0].plot(r.vol, r.Abs, color="C0", alpha=0.1)
            ax[1].fill_between(
                x=vol_light[clustered_idx[i]],
                y1=Abs_light[clustered_idx[i]] + i * delta,
                y2=i * delta,
                color="C0", alpha=0.1, zorder=i,
            )

        ax[0].set(xlabel="Retention vol. / mL", ylabel="A280 / mAU")
        ax[1].set(xlabel="Retention vol. / mL", yticks=[])
        ax[1].spines["left"].set_visible(False)
        ax[0].set(title=f"{cat} ($N = {len(sel)}$)")
        plt.tight_layout()

        if save_plots:
            out = output_dir / f"all_sec_traces_{cat}.png"
            plt.savefig(out, dpi=plot_dpi)
            print(f"wrote {out}")
        plt.close()


def main() -> None:
    expression_df = pd.read_hdf(FULL_AGGREGATE_H5, key="full_aggregate_data")
    nmr_df = pd.read_csv(ALL_METRICS_AND_EXP_RESULTS_CSV)
    df = expression_df.merge(nmr_df, left_on="Design ID", right_on="design_name")
    plot_sec_traces_overview(df=df, output_dir=OUT_DIR)


if __name__ == "__main__":
    main()
