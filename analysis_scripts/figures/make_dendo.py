"""
HSQC Dendrogram with Chamfer Distance
======================================
Computes pairwise Chamfer distances between HSQC spectra (2D peak lists),
performs hierarchical clustering, and renders a dendrogram with miniature
HSQC scatter plots at each leaf.

Chamfer distance is preferred over Hausdorff here because:
- Hausdorff = max(min-distances) -> one outlier peak dominates
- Chamfer  = mean(min-distances) -> robust average spectral similarity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial import cKDTree
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import ast
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PUBLICATION SETTINGS
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================================
# DATA LOADING (from make_hsqc_correlation_figure.py)
# ============================================================================

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from config import (
    HSQC_SUMMARY_PARQUET,
    HSQC_PEAKS_PARQUET,
    BOLTZ_METRICS_CSV,
    EXPRESSION_METRICS_CSV,
    DSSP_CSV,
    COMBINED_DYNAMICS_CSV,
    OUTPUTS_DIR,
)


def convert_str_to_array(s):
    """Convert string representation of array to numpy array."""
    if isinstance(s, np.ndarray):
        return s
    if pd.isna(s) or len(s) == 0:
        return np.array([])
    return np.array(ast.literal_eval(s))


def load_hsqc_data():
    """
    Load and preprocess HSQC data following the same pipeline as make_hsqc_correlation_figure.py
    
    Returns:
        DataFrame with columns: design_name, 1_h, 15_n, intensity, Quality, etc.
    """
    print("Loading HSQC data...")
    
    # Load raw data
    nmr_summary = pd.read_parquet(HSQC_SUMMARY_PARQUET, engine='fastparquet')
    nmr_peaks = pd.read_parquet(HSQC_PEAKS_PARQUET, engine='fastparquet')
    design_metrics = pd.read_csv(BOLTZ_METRICS_CSV)
    expression_metrics = pd.read_csv(EXPRESSION_METRICS_CSV)
    dssp_csv = pd.read_csv(DSSP_CSV)
    ensemble_metrics = pd.read_csv(COMBINED_DYNAMICS_CSV)
    
    # Preprocess expression metrics
    del expression_metrics["Sequence"]
    expression_metrics.rename(columns={"exp_aa_seq_from_LM0627": "Sequence"}, inplace=True)
    expression_metrics["Sequence"] = expression_metrics["Sequence"].str.replace("*", "")
    
    # Merge expression_metrics with design_metrics
    expression_metrics = design_metrics.merge(expression_metrics, left_index=True, right_index=True, how="left")
    
    # Merge nmr_summary with expression_metrics
    df = nmr_summary.merge(expression_metrics, on="Sequence", how="left")
    if "design_name_y" in df.columns:
        del df["design_name_y"]
    if "design_name_x" in df.columns:
        df.rename(columns={"design_name_x": "design_name"}, inplace=True)
    
    # Clean peak data sequences
    nmr_peaks["Sequence"] = nmr_peaks["Sequence"].str.replace("\n", "")
    nmr_peaks["Sequence"] = nmr_peaks["Sequence"].str.replace("*", "")
    
    # Merge with peak data
    df = df.merge(nmr_peaks, on="Sequence", how="left")
    
    # Merge DSSP secondary structure data (following make_hsqc_correlation_figure.py)
    dssp_csv["design_name"] = dssp_csv["file_name"].str.split(".cif").str[0]
    df = df.merge(dssp_csv, on="design_name", how="left")
    df = df[df["Quality"].isin(["Medium", "High"])]
    # Compute loop metrics from dssp_string
    import re
    def avg_loop_length(dssp_string, loop_chars="-TSC "):
        if pd.isna(dssp_string):
            return np.nan
        pattern = f"[{re.escape(loop_chars)}]+"
        loops = re.findall(pattern, dssp_string)
        if not loops:
            return 0.0
        return sum(len(loop) for loop in loops) / len(loops)

    def max_loop_length(dssp_string, loop_chars="-TSC "):
        if pd.isna(dssp_string):
            return np.nan
        pattern = f"[{re.escape(loop_chars)}]+"
        loops = re.findall(pattern, dssp_string)
        if not loops:
            return 0.0
        return max(len(loop) for loop in loops)

    df["avg_loop_length"] = df["dssp_string"].apply(avg_loop_length)
    df["max_loop_length"] = df["dssp_string"].apply(max_loop_length)
    
    # Merge ensemble dynamics metrics
    df = df.merge(ensemble_metrics, left_on="design_name", right_on="folder_name", how="left")
    
    # Convert string arrays to numpy arrays
    df["intensity"] = df["Peaklist Intensities"].apply(convert_str_to_array)
    df["volumes"] = df["Peaklist Volumes"].apply(convert_str_to_array)
    df["1_h"] = df["Peaklist 1H-Shifts"].apply(convert_str_to_array)
    df["15_n"] = df["Peaklist 15N-Shifts"].apply(convert_str_to_array)
    df["1_h_linewidths"] = df["Peaklist 1H-Linewidths"].apply(convert_str_to_array)
    df["15_n_linewidths"] = df["Peaklist 15N-Linewidths"].apply(convert_str_to_array)
    
    # Compute NMR metrics
    def safe_mean(arr):
        return np.mean(arr) if isinstance(arr, np.ndarray) and len(arr) > 0 else np.nan
    def safe_std(arr):
        return np.std(arr) if isinstance(arr, np.ndarray) and len(arr) > 0 else np.nan
    def safe_cv(arr):
        if isinstance(arr, np.ndarray) and len(arr) > 0:
            return np.std(arr) / (np.mean(arr) + 1e-10)
        return np.nan
    
    df['LW_1H_mean'] = df['1_h_linewidths'].apply(safe_mean)
    df['LW_15N_mean'] = df['15_n_linewidths'].apply(safe_mean)
    df['LW_1H_std'] = df['1_h_linewidths'].apply(safe_std)
    df['LW_15N_std'] = df['15_n_linewidths'].apply(safe_std)
    df['1H_shift_std'] = df['1_h'].apply(safe_std)
    df['15N_shift_std'] = df['15_n'].apply(safe_std)
    df['mean_intensity'] = df['intensity'].apply(safe_mean)
    df['cv_intensity'] = df['intensity'].apply(safe_cv)
    df['peak_count'] = df['1_h'].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
    
    return df


# ============================================================================
# CHAMFER DISTANCE
# ============================================================================

def chamfer_distance(peaks_a: np.ndarray, peaks_b: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two 2D peak sets.
    
    For each peak in A, find nearest neighbor in B (and vice versa),
    then average all those minimum distances.
    
    Args:
        peaks_a: (N, 2) array of (1H, 15N) shifts
        peaks_b: (M, 2) array of (1H, 15N) shifts
    
    Returns:
        Symmetric Chamfer distance (mean of both directions)
    """
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return np.nan
    
    tree_b = cKDTree(peaks_b)
    tree_a = cKDTree(peaks_a)

    # A -> B: for each point in A, distance to nearest in B
    dists_a2b, _ = tree_b.query(peaks_a, k=1)
    # B -> A: for each point in B, distance to nearest in A
    dists_b2a, _ = tree_a.query(peaks_b, k=1)

    chamfer = 0.5 * (np.mean(dists_a2b) + np.mean(dists_b2a))
    return chamfer


def compute_distance_matrix(
    h_shifts: list,
    n_shifts: list,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute all-by-all Chamfer distance matrix.
    
    Args:
        h_shifts: list of 1H shift arrays
        n_shifts: list of 15N shift arrays  
        normalize: if True, z-score normalize each dimension before distance calc
                   (critical because 1H ~6-10 ppm vs 15N ~100-130 ppm)
    
    Returns:
        (n, n) symmetric distance matrix
    """
    n = len(h_shifts)
    
    # Collect all shifts for normalization
    if normalize:
        all_h = np.concatenate([h for h in h_shifts if len(h) > 0])
        all_n = np.concatenate([ns for ns in n_shifts if len(ns) > 0])
        h_mean, h_std = np.mean(all_h), np.std(all_h)
        n_mean, n_std = np.mean(all_n), np.std(all_n)
    
    # Build normalized 2D peak arrays
    peak_arrays = []
    for i in range(n):
        h, ns = h_shifts[i], n_shifts[i]
        if len(h) == 0 or len(ns) == 0:
            peak_arrays.append(np.empty((0, 2)))
            continue
        if normalize:
            h_norm = (h - h_mean) / (h_std + 1e-10)
            n_norm = (ns - n_mean) / (n_std + 1e-10)
        else:
            h_norm, n_norm = h, ns
        peak_arrays.append(np.column_stack([h_norm, n_norm]))
    
    # Compute pairwise distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = chamfer_distance(peak_arrays[i], peak_arrays[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    return dist_matrix 


# ============================================================================
# MINI HSQC RENDERING
# ============================================================================

def render_mini_hsqc(
    h_shifts: np.ndarray,
    n_shifts: np.ndarray,
    intensities: np.ndarray = None,
    size_px: int = 120,
    h_range: tuple = None,
    n_range: tuple = None,
    bg_color: str = '#FAFAFA',
    peak_color: str = '#2166AC',
    highlight_color: str = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Render a miniature HSQC scatter plot as an image array.
    Uses consistent axis ranges across all spectra for fair comparison.
    """
    fig_size = size_px / 100
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    if len(h_shifts) > 0 and len(n_shifts) > 0:
        if intensities is not None and len(intensities) == len(h_shifts):
            sizes = np.clip(intensities / (np.median(intensities) + 1e-10) * 3, 0.5, 8)
        else:
            sizes = np.full(len(h_shifts), 2.0)
        
        color = highlight_color if highlight_color else peak_color
        ax.scatter(h_shifts, n_shifts, s=sizes, c=color, alpha=alpha, 
                   edgecolors='none', rasterized=True)
    
    if h_range:
        ax.set_xlim(h_range)
    if n_range:
        ax.set_ylim(n_range)
    
    # NMR convention: invert axes
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#AAAAAA')
        spine.set_visible(True)
    
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf).copy()
    plt.close(fig)
    
    return img


# ============================================================================
# DISTOGRAM: CLUSTERED DISTANCE MATRIX WITH DENDROGRAMS
# ============================================================================

# Define a pleasing color palette for highlighted samples
HIGHLIGHT_COLORS = [
    '#E63946',  # Coral red
    '#2A9D8F',  # Teal
    '#E9C46A',  # Golden yellow
    '#264653',  # Dark slate
    '#F4A261',  # Sandy orange
    '#9B5DE5',  # Purple
    '#00BBF9',  # Sky blue
    '#00F5D4',  # Mint
]


def plot_dendrogram(
    dist_matrix: np.ndarray,
    names: list,
    h_shifts: list = None,
    n_shifts: list = None,
    intensities: list = None,
    leaf_values: np.ndarray = None,
    leaf_value_label: str = 'Helix Count',
    leaf_color_bars: list = None,
    output_path: str = None,
    figsize: tuple = (14, 8),
    linkage_method: str = 'ward',
    orientation: str = 'top',
    cmap_name: str = 'viridis',
    linewidth: float = 1.5,
    highlight_n_leaves: int = 0,
    highlight_indices: list = None,
    highlight_size: float = 120,
    random_seed: int = 42,
    hsqc_output_dir: str = None,
) -> tuple:
    """
    Render a publication-quality dendrogram tree from the Chamfer distance matrix,
    with branches colored by the mean value of a per-leaf metric (e.g. helix count)
    and optional colored annotation bars at the leaves.

    Args:
        dist_matrix:        (n, n) symmetric distance matrix
        names:              sample labels
        leaf_values:        (n,) array of per-sample values to color branches by
        leaf_value_label:   label for the branch colorbar
        leaf_color_bars:    list of dicts for annotation strips at the leaves.
                            Each dict: {'values': array(n,), 'label': str,
                                        'cmap': str, 'categorical': bool (opt)}
                            Categorical bars use discrete colors; continuous use
                            a gradient. Each bar becomes a thin row below the tree.
        output_path:        save path (.pdf and .svg written)
        figsize:            figure size
        linkage_method:     hierarchical clustering method
        orientation:        'top', 'bottom', 'left', or 'right'
        cmap_name:          matplotlib colormap name for branches
        linewidth:          branch line width
        highlight_n_leaves: number of random leaves to highlight (0 = none)
        highlight_indices:  explicit sample indices to highlight (overrides n)
        highlight_color:    marker / label color
        highlight_marker:   matplotlib marker style
        highlight_size:     marker size
        random_seed:        seed for reproducible random selection

    Returns:
        (matplotlib Figure, list of highlighted design_name strings)
    """
    from matplotlib.cm import ScalarMappable
    import matplotlib.gridspec as gridspec

    n = len(names)
    n_bars = len(leaf_color_bars) if leaf_color_bars else 0

    # -- Hierarchical clustering --
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)

    # -- Compute mean leaf_value for every node (leaves + internal) --
    if leaf_values is not None:
        leaf_values = np.asarray(leaf_values, dtype=float)
        node_mean = np.zeros(2 * n - 1)
        node_mean[:n] = leaf_values

        node_leaves_list = [None] * (2 * n - 1)
        for i in range(n):
            node_leaves_list[i] = [i]
        for i, row in enumerate(Z):
            left, right = int(row[0]), int(row[1])
            members = node_leaves_list[left] + node_leaves_list[right]
            node_leaves_list[n + i] = members
            node_mean[n + i] = np.nanmean(leaf_values[members])

        vmin = np.nanmin(leaf_values)
        vmax = np.nanmax(leaf_values)
        branch_cmap = plt.get_cmap(cmap_name)
        branch_norm = Normalize(vmin=vmin, vmax=vmax)

        def link_color_func(k):
            rgba = branch_cmap(branch_norm(node_mean[k]))
            return '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
    else:
        link_color_func = lambda k: '#333333'

    # -- Create figure with gridspec: dendrogram + color bar rows --
    if n_bars > 0:
        bar_height_ratio = 0.03  # each bar is thin
        height_ratios = [1.0] + [bar_height_ratio] * n_bars
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = gridspec.GridSpec(
            nrows=1 + n_bars, ncols=1,
            height_ratios=height_ratios,
            hspace=0.05,
        )
        ax = fig.add_subplot(gs[0])
        bar_axes = [fig.add_subplot(gs[i + 1]) for i in range(n_bars)]
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        bar_axes = []

    # -- Draw dendrogram --
    dn = dendrogram(
        Z,
        ax=ax,
        labels=names,
        orientation=orientation,
        no_labels=True,
        color_threshold=0,
        above_threshold_color='#333333',
        link_color_func=link_color_func,
    )

    # Thicken the branches
    for coll in ax.collections:
        coll.set_linewidth(linewidth)
    for line in ax.lines:
        line.set_linewidth(linewidth)

    # -- Leaf display order and x-positions --
    display_order = dn['leaves']
    leaf_x_positions = np.arange(len(display_order)) * 10 + 5

    # -- Draw leaf color bars --
    bar_colorbars = []  # (ScalarMappable, label) pairs for continuous bars
    bar_legends = []    # (patches, label) for categorical bars
    for bar_idx, bar_spec in enumerate(leaf_color_bars or []):
        bar_ax = bar_axes[bar_idx]
        bar_vals = np.asarray(bar_spec['values'], dtype=float)
        bar_label = bar_spec.get('label', '')
        bar_cmap_name = bar_spec.get('cmap', 'viridis')
        is_categorical = bar_spec.get('categorical', False)

        # Reorder values to match dendrogram leaf order
        ordered_vals = bar_vals[display_order]

        if is_categorical:
            # Use explicit category_colors if provided, else fall back to cmap
            cat_colors = bar_spec.get('category_colors', None)
            cat_names = bar_spec.get('category_names', None)

            if cat_colors:
                # Explicit color dict: {value: hex_color}
                from matplotlib.colors import to_rgba
                colors_2d = np.zeros((1, len(ordered_vals), 4))
                for j, v in enumerate(ordered_vals):
                    if np.isnan(v):
                        colors_2d[0, j] = [0.85, 0.85, 0.85, 1.0]
                    else:
                        colors_2d[0, j] = to_rgba(cat_colors.get(int(v), '#CCCCCC'))
            else:
                unique_vals = np.unique(ordered_vals[~np.isnan(ordered_vals)])
                n_unique = len(unique_vals)
                cat_cmap = plt.get_cmap(bar_cmap_name, n_unique)
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                colors_2d = np.zeros((1, len(ordered_vals), 4))
                for j, v in enumerate(ordered_vals):
                    if np.isnan(v):
                        colors_2d[0, j] = [0.85, 0.85, 0.85, 1.0]
                    else:
                        colors_2d[0, j] = cat_cmap(val_to_idx[v])

            bar_ax.imshow(
                colors_2d, aspect='auto',
                extent=[leaf_x_positions[0] - 5, leaf_x_positions[-1] + 5, 0, 1],
                interpolation='nearest',
            )

            # Build legend patches
            if cat_colors and cat_names:
                patches = [
                    mpatches.Patch(color=cat_colors[k], label=cat_names[k])
                    for k in sorted(cat_names.keys())
                ]
                bar_legends.append((patches, bar_label))
        else:
            # Continuous
            bar_cmap = plt.get_cmap(bar_cmap_name)
            bmin = np.nanmin(bar_vals)
            bmax = np.nanmax(bar_vals)
            bar_norm = Normalize(vmin=bmin, vmax=bmax)

            colors_2d = np.zeros((1, len(ordered_vals), 4))
            for j, v in enumerate(ordered_vals):
                if np.isnan(v):
                    colors_2d[0, j] = [0.85, 0.85, 0.85, 1.0]
                else:
                    colors_2d[0, j] = bar_cmap(bar_norm(v))

            bar_ax.imshow(
                colors_2d, aspect='auto',
                extent=[leaf_x_positions[0] - 5, leaf_x_positions[-1] + 5, 0, 1],
                interpolation='nearest',
            )
            # Store for colorbar
            sm = ScalarMappable(cmap=bar_cmap, norm=bar_norm)
            sm.set_array([])
            bar_colorbars.append((sm, bar_label, bar_ax))

        # Style the bar axis
        bar_ax.set_xlim(ax.get_xlim())
        bar_ax.set_yticks([0.5])
        bar_ax.set_yticklabels([bar_label.replace(' ', '\n')], fontsize=12)
        bar_ax.tick_params(left=False, bottom=False, labelbottom=False)
        for spine in bar_ax.spines.values():
            spine.set_visible(False)

    # -- Highlight selected leaf nodes --
    highlighted_names = []
    highlighted_colors = []
    if highlight_indices is not None:
        sel_orig_indices = list(highlight_indices)
    elif highlight_n_leaves > 0:
        rng = np.random.default_rng(random_seed)
        n_leaves = len(display_order)
        k = min(highlight_n_leaves, n_leaves)
        base_positions = np.linspace(0, n_leaves - 1, k + 2)[1:-1]
        jitter = rng.integers(-max(1, n_leaves // (k * 4)),
                              max(2, n_leaves // (k * 4)),
                              size=k)
        sel_display_positions = np.clip(
            (base_positions + jitter).astype(int), 0, n_leaves - 1
        )
        sel_orig_indices = [display_order[p] for p in sel_display_positions]
    else:
        sel_orig_indices = []

    highlighted_colors = []  # Store colors for return
    if sel_orig_indices:
        orig_to_display = {orig: pos for pos, orig in enumerate(display_order)}

        for label_num, orig_idx in enumerate(sel_orig_indices, start=1):
            disp_pos = orig_to_display.get(orig_idx)
            if disp_pos is None:
                continue
            x = leaf_x_positions[disp_pos]
            sample_name = names[orig_idx]
            highlighted_names.append(sample_name)
            
            # Get color from palette (cycle if needed)
            color = HIGHLIGHT_COLORS[(label_num - 1) % len(HIGHLIGHT_COLORS)]
            highlighted_colors.append(color)

            # Draw filled circle marker
            ax.scatter(
                x, 0, s=highlight_size,
                marker='o', color=color,
                zorder=10, edgecolors='white', linewidths=1.5,
                clip_on=False,
            )
            # Label with letter inside circle
            ax.annotate(
                chr(64 + label_num),
                xy=(x, 0),
                xytext=(0, -20),
                textcoords='offset points',
                ha='center', va='top',
                fontsize=14, fontweight='bold', color=color,
                clip_on=False,
            )

        print(f"\nHighlighted leaves for pullout:")
        for i, name in enumerate(highlighted_names, start=1):
            color = highlighted_colors[i-1]
            extra = ''
            if leaf_values is not None:
                idx = names.index(name)
                extra = f'  (helix_count={leaf_values[idx]:.0f})'
            print(f"  {chr(64+i)}: {name} [{color}]{extra}")
        
        # ── Save individual HSQC plots for each highlighted sample ──
        if hsqc_output_dir and h_shifts is not None and n_shifts is not None:
            os.makedirs(hsqc_output_dir, exist_ok=True)
            
            # Fixed axis ranges (inverted for NMR convention: high to low)
            h_range = (12, 6)    # 1H: 6-12 ppm
            n_range = (140, 90)  # 15N: 90-140 ppm
            
            for i, orig_idx in enumerate(sel_orig_indices):
                sample_name = names[orig_idx]
                color = highlighted_colors[i]
                letter = chr(65 + i)
                
                h = h_shifts[orig_idx]
                ns = n_shifts[orig_idx]
                ints = intensities[orig_idx] if intensities else None
                
                # Create HSQC figure
                fig_hsqc, ax_hsqc = plt.subplots(figsize=(6, 5), facecolor='white')
                
                if ints is not None and len(ints) == len(h):
                    sizes = np.clip(ints / (np.median(ints) + 1e-10) * 60, 80, 160)
                else:
                    sizes = np.full(len(h), 80)
                
                ax_hsqc.scatter(h, ns, s=sizes, c=color, alpha=0.7,
                               edgecolors='white', linewidths=0.5)
                
                ax_hsqc.set_xlim(h_range)
                ax_hsqc.set_ylim(n_range)
                ax_hsqc.set_xlabel(r'$^{1}$H', fontsize=32, labelpad=10)
                ax_hsqc.set_ylabel(r'$^{15}$N', fontsize=32)
                ax_hsqc.set_xticks([])
                ax_hsqc.set_yticks([])
                for spine in ax_hsqc.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.0)

                plt.tight_layout()

                # Save
                hsqc_path = os.path.join(hsqc_output_dir, f'hsqc_{letter}_{sample_name}.pdf')
                fig_hsqc.savefig(hsqc_path, dpi=300, bbox_inches='tight', facecolor='white')
                fig_hsqc.savefig(hsqc_path.replace('.pdf', '.svg'), bbox_inches='tight', facecolor='white')
                plt.close(fig_hsqc)
                print(f"  Saved HSQC: {hsqc_path}")

    # -- Axis labels --
    dist_label = f'Chamfer Distance\nWard Linkage'
    if orientation in ('top', 'bottom'):
        ax.set_ylabel(dist_label, fontsize=14)
    else:
        ax.set_xlabel(dist_label, fontsize=14)

    ax.set_title(
        f'HSQC Spectral Dendrogram  (n={n})',
        fontsize=16, pad=4,
    )

    # Clean up dendrogram spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])  # Hide y-axis ticks
    if orientation == 'top':
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False)

    # -- Branch colorbar --
    if leaf_values is not None:
        sm = ScalarMappable(cmap=branch_cmap, norm=branch_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=30)
        cb.set_label(f'{leaf_value_label} (branches)', fontsize=12)
        cb.ax.tick_params(labelsize=10)

    # -- Leaf bar colorbars (small, right-aligned to each bar) --
    for sm, label, bar_ax in bar_colorbars:
        cb = fig.colorbar(sm, ax=bar_ax, shrink=0.8, pad=0.02, aspect=8,
                          orientation='vertical')
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=8)

    # -- Categorical legends --
    for patches, label in bar_legends:
        ax.legend(
            handles=patches,
            loc='upper right',
            bbox_to_anchor=(0.88, 0.95),
            fontsize=12,
            title=label,
            title_fontsize=13,
            frameon=True,
            edgecolor='#CCCCCC',
        )

    # -- Footer --
    fig.text(
        0.5, 0.005,
        f'{n} samples  |  Chamfer distance (z-scored 1H & 15N)  |  {linkage_method} linkage',
        ha='center', fontsize=8, color='#888888', style='italic',
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        svg_path = output_path.replace('.pdf', '.svg')
        fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
        print(f"Saved dendrogram: {output_path}")

    return fig, highlighted_names, highlighted_colors


# ============================================================================
# COMPANION: SIMPLE DISTANCE MATRIX HEATMAP (legacy)
# ============================================================================

def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    names: list,
    Z: np.ndarray = None,
    output_path: str = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot clustered distance matrix heatmap (seaborn, no dendrograms)."""
    if Z is not None:
        order = leaves_list(Z)
    else:
        order = np.arange(len(names))
    
    ordered_names = [names[i] for i in order]
    ordered_dist = dist_matrix[np.ix_(order, order)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#FFFFFF', '#FEE5D9', '#FCBBA1', '#FC9272',
              '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']
    cmap = LinearSegmentedColormap.from_list('dist', colors)
    
    sns.heatmap(
        ordered_dist,
        xticklabels=ordered_names,
        yticklabels=ordered_names,
        cmap=cmap, square=True,
        linewidths=0.3, linecolor='#EEEEEE',
        ax=ax,
        cbar_kws={'label': 'Chamfer Distance', 'shrink': 0.7},
    )
    
    ax.set_title('HSQC Spectral Distance Matrix (Chamfer)', fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


# ============================================================================
# MAIN FIGURE: t-SNE + REPRESENTATIVE HSQCs
# ============================================================================

def plot_hsqc_tsne(
    names: list,
    h_shifts: list,
    n_shifts: list,
    intensities: list = None,
    quality_labels: list = None,
    dist_matrix: np.ndarray = None,
    figsize: tuple = (10, 8),
    mini_hsqc_size: int = 100,
    n_hsqc_examples: int = 4,
    perplexity: int = 30,
    title: str = 'HSQC Spectral Clustering (t-SNE)',
    output_path: str = None,
    show_only_high: bool = True,
    show_pullouts: bool = False,
) -> tuple:
    """
    Create t-SNE visualization of HSQC spectra with representative examples.
    Clean, publication-ready figure for large datasets.
    """
    from sklearn.manifold import TSNE
    from scipy.cluster.hierarchy import fcluster
    from matplotlib import cm
    
    n_samples = len(names)
    print(f"Processing {n_samples} samples...")
    
    # -- Compute distance matrix if needed --
    if dist_matrix is None:
        print("Computing Chamfer distance matrix...")
        dist_matrix = compute_distance_matrix(h_shifts, n_shifts, normalize=True)
    
    # Handle NaN distances
    dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix) * 1.5)
    
    # -- t-SNE embedding --
    print("Computing t-SNE embedding...")
    tsne = TSNE(
        n_components=2,
        metric='precomputed',
        perplexity=min(perplexity, n_samples - 1),
        random_state=42,
        init='random',
    )
    embedding = tsne.fit_transform(dist_matrix)
    
    # -- Filter for High quality only if requested --
    if show_only_high and quality_labels is not None:
        high_mask = np.array([q == 'High' or q == "Medium" for q in quality_labels])
        plot_embedding = embedding[high_mask]
        plot_indices = np.where(high_mask)[0]
        n_high = len(plot_indices)
        print(f"Showing {n_high} High quality samples out of {n_samples}")
    else:
        plot_embedding = embedding
        plot_indices = np.arange(n_samples)
        high_mask = np.ones(n_samples, dtype=bool)
    
    # -- Create figure --
    fig, ax_tsne = plt.subplots(figsize=figsize, facecolor='white')
    
    # -- Plot t-SNE (only High quality points) --
    scatter = ax_tsne.scatter(
        plot_embedding[:, 0], plot_embedding[:, 1],
        c='#2166AC', s=40, alpha=0.7,
        edgecolors='white', linewidths=0.5,
        rasterized=True
    )
    
    ax_tsne.set_xlabel('t-SNE 1', fontsize=15)
    ax_tsne.set_ylabel('t-SNE 2', fontsize=15)
    quality_text = ' (High Quality Only)' if show_only_high else ''
    ax_tsne.set_title(f'{title}{quality_text}', fontsize=17, pad=4)
    ax_tsne.spines['top'].set_visible(False)
    ax_tsne.spines['right'].set_visible(False)
    ax_tsne.tick_params(labelsize=10)

    # Footer
    n_shown = len(plot_indices)
    fig.text(
        0.5, 0.02,
        f'{n_shown} samples  |  Chamfer distance  |  perplexity={perplexity}',
        ha='center', fontsize=9, color='#888888', style='italic'
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(output_path.replace('.pdf', '.svg'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig, dist_matrix, embedding


# ============================================================================
# INTERACTIVE PLOTLY HTML VISUALIZATIONS
# ============================================================================

def create_plotly_tsne_html(
    df: pd.DataFrame,
    embedding: np.ndarray,
    color_col: str,
    output_path: str,
    title: str = None,
    show_only_high: bool = True,
    colorscale: str = 'Viridis',
) -> None:
    """
    Create an interactive Plotly t-SNE scatter plot colored by a feature.
    """
    plot_df = df.copy()
    plot_df['tsne_1'] = embedding[:, 0]
    plot_df['tsne_2'] = embedding[:, 1]

    if show_only_high and 'Quality' in plot_df.columns:
        plot_df = plot_df[plot_df['Quality'] == 'High'].copy()

    if color_col not in plot_df.columns:
        print(f"  Warning: {color_col} not found in dataframe, skipping...")
        return

    valid_values = plot_df[color_col].dropna()
    if len(valid_values) == 0:
        print(f"  Warning: {color_col} has no valid values, skipping...")
        return
    
    is_categorical = plot_df[color_col].dtype == 'object' or plot_df[color_col].nunique() < 10
    
    hover_cols = ['design_name']
    for col in ['Quality', 'helix_count', 'sheet_count', 'peak_count', 'LW_1H_mean']:
        if col in plot_df.columns and col != color_col:
            hover_cols.append(col)
    
    plot_title = title if title else f't-SNE colored by {color_col}'
    
    if is_categorical:
        fig = px.scatter(
            plot_df, x='tsne_1', y='tsne_2',
            color=color_col, hover_data=hover_cols, title=plot_title,
        )
    else:
        fig = px.scatter(
            plot_df, x='tsne_1', y='tsne_2',
            color=color_col, color_continuous_scale=colorscale,
            hover_data=hover_cols, title=plot_title,
        )
    
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')))
    fig.update_layout(
        font=dict(family='Arial', size=12),
        title=dict(font=dict(size=16)),
        xaxis_title='t-SNE 1', yaxis_title='t-SNE 2',
        width=900, height=700, template='plotly_white',
    )
    
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def generate_all_plotly_html(
    df: pd.DataFrame,
    embedding: np.ndarray,
    output_dir: str,
    show_only_high: bool = True,
) -> None:
    """Generate multiple interactive Plotly HTML files with different feature colorings."""
    os.makedirs(output_dir, exist_ok=True)
    
    features = {
        'helix_count': ('Viridis', 'Number of Helices'),
        'sheet_count': ('Plasma', 'Number of Sheets'),
        'loop_count': ('Cividis', 'Number of Loops'),
        'avg_loop_length': ('Turbo', 'Average Loop Length'),
        'max_loop_length': ('Inferno', 'Maximum Loop Length'),
        'coil_fraction': ('RdYlBu', 'Coil Fraction'),
        'plddt': ('RdYlGn', 'pLDDT Score'),
        'ptm': ('RdYlGn', 'pTM Score'),
        'mean_plddt_boltz1_recycle_3': ('RdYlGn', 'Boltz1 pLDDT (recycle 3)'),
        'mean_plddt_af3_recycle_3': ('RdYlGn', 'AF3 pLDDT (recycle 3)'),
        'mean_rmsf_boltz1_recycle_3': ('YlOrRd', 'Boltz1 RMSF (recycle 3)'),
        'mean_rmsf_af3_recycle_3': ('YlOrRd', 'AF3 RMSF (recycle 3)'),
        'comp_RMSD_ca': ('Oranges', 'Computational RMSD (Cα)'),
        'comp_RMSD_ca_var': ('Reds', 'RMSD Variance'),
        'LW_1H_mean': ('Magma', 'Mean 1H Linewidth'),
        'LW_15N_mean': ('Magma', 'Mean 15N Linewidth'),
        'LW_1H_std': ('Plasma', '1H Linewidth Std Dev'),
        '1H_shift_std': ('Viridis', '1H Chemical Shift Dispersion'),
        '15N_shift_std': ('Viridis', '15N Chemical Shift Dispersion'),
        'cv_intensity': ('Turbo', 'Intensity Coefficient of Variation'),
        'mean_intensity': ('YlGnBu', 'Mean Peak Intensity'),
        'peak_count': ('Blues', 'Number of Peaks'),
        'Quality': ('Set1', 'HSQC Quality Category'),
    }
    
    print(f"\nGenerating {len(features)} Plotly HTML visualizations...")
    
    for feature, (colorscale, title) in features.items():
        if feature in df.columns:
            filename = f'tsne_{feature}.html'
            create_plotly_tsne_html(
                df=df, embedding=embedding, color_col=feature,
                output_path=os.path.join(output_dir, filename),
                title=title, show_only_high=show_only_high, colorscale=colorscale,
            )
    
    print(f"\nAll HTML files saved to: {output_dir}")


# ============================================================================
# MAIN EXECUTION WITH REAL DATA
# ============================================================================

def run_dendrogram(output_dir='hsqc_dendrogram_output'):
    """
    Run the dendrogram pipeline with real HSQC data:
      1. Load and filter HSQC peak data
      2. Compute Chamfer distance matrix
      3. Render dendrogram tree (Ward linkage)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load real data
    df = load_hsqc_data()
    
    # Filter for samples with valid HSQC data
    df_plot = df.drop_duplicates(subset=['design_name']).copy()
    df_plot = df_plot[df_plot['1_h'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
    df_plot = df_plot[df_plot['15_n'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
    df_plot = df_plot.reset_index(drop=True)
    
    print(f"Processing {len(df_plot)} samples with valid HSQC data")
    
    # Extract data
    names = df_plot['design_name'].tolist()
    h_shifts = df_plot['1_h'].tolist()
    n_shifts = df_plot['15_n'].tolist()
    
    # ── Classify secondary structure: mostly α / mixed / mostly β ──
    # Standard criteria (SCOP/CATH-like):
    #   helix_frac ≥ 0.40 and strand_frac < 0.10  →  Mostly α
    #   strand_frac ≥ 0.40 and helix_frac < 0.10  →  Mostly β
    #   otherwise                                   →  Mixed α/β
    def classify_ss(dssp_string):
        if pd.isna(dssp_string) or len(dssp_string) == 0:
            return np.nan
        total = len(dssp_string)
        helix_frac = sum(c in 'HGI' for c in dssp_string) / total   # H=α, G=3₁₀, I=π
        strand_frac = sum(c in 'EB' for c in dssp_string) / total    # E=strand, B=bridge
        if helix_frac >= 0.40 and strand_frac < 0.10:
            return 0  # Mostly α
        elif strand_frac >= 0.40 and helix_frac < 0.10:
            return 2  # Mostly β
        else:
            return 1  # Mixed α/β

    if 'dssp_string' in df_plot.columns:
        ss_class = df_plot['dssp_string'].apply(classify_ss).values
    else:
        ss_class = None

    # Build leaf color bar — single categorical strip with appealing colors
    leaf_color_bars = []
    if ss_class is not None:
        leaf_color_bars.append({
            'values': ss_class,
            'label': 'Fold Class',
            'categorical': True,
            'category_names': {0: 'Mostly α', 1: 'Mixed α/β', 2: 'Mostly β'},
            'category_colors': {0: "#D4D4D4", 1: "#949494", 2: "#000000"},  # Coral, Teal, Sky blue
        })

    # Compute distance matrix
    print("Computing Chamfer distance matrix...")
    dist_matrix = compute_distance_matrix(h_shifts, n_shifts, normalize=True)
    dist_matrix = np.nan_to_num(dist_matrix, nan=np.nanmax(dist_matrix) * 1.5)

    # ── DENDROGRAMS — one per linkage strategy ──
    linkage_methods = ['ward']
    fig_dendros = {}
    # Get intensities for HSQC plots
    intensities = df_plot['intensity'].tolist() if 'intensity' in df_plot.columns else None

    for method in linkage_methods:
        print(f"\n--- Linkage: {method} ---")
        fig_dendro, highlighted, colors = plot_dendrogram(
            dist_matrix=dist_matrix,
            names=names,
            h_shifts=h_shifts,
            n_shifts=n_shifts,
            intensities=intensities,
            leaf_values=None,
            leaf_color_bars=leaf_color_bars,
            output_path=f'{output_dir}/hsqc_dendrogram_{method}.pdf',
            figsize=(16, 10),
            linkage_method=method,
            orientation='top',
            highlight_n_leaves=4,
            hsqc_output_dir=f'{output_dir}/hsqc_pullouts',
        )
        fig_dendros[method] = fig_dendro
    
    # ── Save distance matrix ──
    dist_df = pd.DataFrame(dist_matrix, index=names, columns=names)
    dist_df.to_csv(f'{output_dir}/chamfer_distance_matrix.csv')
    print(f"Saved distance matrix CSV")
    
    return fig_dendros, dist_matrix, df_plot


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HSQC Dendrogram - Real Data")
    print("=" * 60)
    
    figs, dm, df = run_dendrogram(
        output_dir=str(OUTPUTS_DIR / "figures" / "hsqc_dendrogram")
    )
    
    plt.show()
    print("\nDone!")