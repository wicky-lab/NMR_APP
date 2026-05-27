"""
Script to correlate NMR relaxation observables with structural model confidence metrics.
Analyzes relaxation data (R1, R2, HetNOE) against pLDDT and RMSF.
Uses R² from linear regression to quantify explained variance.

Publication-ready visualization style matching HSQC analysis.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from typing import Dict, List, Optional, Tuple

# Configuration — paths come from config.py (env var NMR_PAPER_DATA)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from config import METRICS_PATH, RELAXATION_DIR, ALL_METRICS_AND_EXP_RESULTS_CSV, OUTPUTS_DIR

RELAX_OUT = OUTPUTS_DIR / "relaxation"
RELAX_OUT.mkdir(parents=True, exist_ok=True)
(RELAX_OUT / "figures").mkdir(exist_ok=True)

# MSG tag offset
# M is cleaved when followed by flexible linker in e coli. 
MSG_TAG_OFFSET = 2

# ============================================================================
# PUBLICATION SETTINGS
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================================
# NAME MAPPINGS - NMR RELAXATION METRICS
# ============================================================================

NMR_METRIC_NAME_MAP = {
    # Relaxation rates
    'R1_mean': '$R_1$ (s$^{-1}$)',
    'R2_mean': '$R_2$ (s$^{-1}$)',
    'R1_uncertainty': '$R_1$ Uncertainty',
    'R2_uncertainty': '$R_2$ Uncertainty',
    
    # Heteronuclear NOE
    'HetNOE_mean': 'HetNOE',
    'HetNOE_uncertainty': 'HetNOE Uncertainty',
    
    # Derived metrics
    'R2_R1_ratio': '$R_2/R_1$ Ratio',
}

# ============================================================================
# NAME MAPPINGS - COMPUTATIONAL METRICS
# ============================================================================

COMP_METRIC_NAME_MAP = {
    # pLDDT metrics - individual sources
    'plddt_af3_recycle_0': 'AF3 pLDDT (r0, ensemble mean)',
    'plddt_af3_recycle_3': 'AF3 pLDDT (r3, ensemble mean)',
    'plddt_boltz1_recycle_0': 'Boltz-1 pLDDT (r0, ensemble mean)',
    'plddt_boltz1_recycle_3': 'Boltz-1 pLDDT (r3, ensemble mean)',
    'plddt_boltz2_recycle_0': 'Boltz-2 pLDDT (r0, ensemble mean)',
    'plddt_boltz2_recycle_3': 'Boltz-2 pLDDT (r3, ensemble mean)',
    
    # RMSF metrics - individual sources
    'rmsf_af3_recycle_0': 'AF3 RMSF (r0)',
    'rmsf_af3_recycle_3': 'AF3 RMSF (r3)',
    'rmsf_boltz1_recycle_0': 'Boltz-1 RMSF (r0)',
    'rmsf_boltz1_recycle_3': 'Boltz-1 RMSF (r3)',
    'rmsf_boltz2_recycle_0': 'Boltz-2 RMSF (r0)',
    'rmsf_boltz2_recycle_3': 'Boltz-2 RMSF (r3)',

    # Ensemble metrics
    'mean_plddt_all': 'Mean pLDDT (all)',
    'max_plddt_all': 'Max pLDDT (all)',
    'min_plddt_all': 'Min pLDDT (all)',
    'mean_rmsf_all': 'Mean RMSF (all)',
    'max_rmsf_all': 'Max RMSF (all)',
}

# ============================================================================
# METRIC CATEGORIZATIONS
# ============================================================================

# NMR relaxation metric categories
NMR_CATEGORIES = {
    # Fast dynamics (ps-ns timescale)
    'R1_mean': 'fast_dynamics',
    'R1_uncertainty': 'fast_dynamics',
    'HetNOE_mean': 'fast_dynamics',
    'HetNOE_uncertainty': 'fast_dynamics',
    
    # Slow dynamics / exchange (µs-ms timescale)
    'R2_mean': 'slow_dynamics',
    'R2_uncertainty': 'slow_dynamics',
    'R2_R1_ratio': 'slow_dynamics',
}

# Computational metric categories
COMP_CATEGORIES = {
    # Confidence metrics
    'plddt_af3_recycle_0': 'confidence',
    'plddt_af3_recycle_3': 'confidence',
    'plddt_boltz1_recycle_0': 'confidence',
    'plddt_boltz1_recycle_3': 'confidence',
    'plddt_boltz2_recycle_0': 'confidence',
    'plddt_boltz2_recycle_3': 'confidence',
    'mean_plddt_all': 'confidence',
    'max_plddt_all': 'confidence',
    'min_plddt_all': 'confidence',
    'consensus_rigidity_z': 'confidence',
    'consensus_rigidity_rank': 'confidence',
    'plddt_rmsf_agreement': 'confidence',
    
    # Flexibility metrics
    'rmsf_af3_recycle_0': 'flexibility',
    'rmsf_af3_recycle_3': 'flexibility',
    'rmsf_boltz1_recycle_0': 'flexibility',
    'rmsf_boltz1_recycle_3': 'flexibility',
    'rmsf_boltz2_recycle_0': 'flexibility',
    'rmsf_boltz2_recycle_3': 'flexibility',
    'mean_rmsf_all': 'flexibility',
    'max_rmsf_all': 'flexibility',
}

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# NMR category colors
NMR_CATEGORY_COLORS = {
    'fast_dynamics': '#E64B35',      # Coral red - ps-ns motions
    'slow_dynamics': '#4DBBD5',      # Teal - µs-ms exchange
}

# Computational category colors
COMP_CATEGORY_COLORS = {
    'confidence': '#3C5488',    # Navy blue
    'flexibility': '#F39B7F',   # Salmon
}

# ============================================================================
# COLORMAPS
# ============================================================================

def create_correlation_cmap():
    """Blue-white-red diverging colormap for correlations."""
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', 
              '#FFFFFF',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    return LinearSegmentedColormap.from_list('correlation', colors)

def create_r2_cmap():
    """Sequential colormap for R² values."""
    colors = ['#FFFFFF', '#FEE5D9', '#FCBBA1', '#FC9272', 
              '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']
    return LinearSegmentedColormap.from_list('rsquared', colors)

CORRELATION_CMAP = create_correlation_cmap()
R2_CMAP = create_r2_cmap()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_nmr_display_name(code_name: str) -> str:
    """Get publication-ready name for NMR metric."""
    return NMR_METRIC_NAME_MAP.get(code_name, code_name)

def get_comp_display_name(code_name: str) -> str:
    """Get publication-ready name for computational metric."""
    return COMP_METRIC_NAME_MAP.get(code_name, code_name)

def get_nmr_color(metric: str) -> str:
    """Get color based on NMR metric category."""
    category = NMR_CATEGORIES.get(metric, 'fast_dynamics')
    return NMR_CATEGORY_COLORS[category]

def get_comp_color(metric: str) -> str:
    """Get color based on computational metric category."""
    category = COMP_CATEGORIES.get(metric, 'confidence')
    return COMP_CATEGORY_COLORS[category]

def get_nmr_category(metric: str) -> str:
    """Get category for NMR metric."""
    return NMR_CATEGORIES.get(metric, 'fast_dynamics')

def get_comp_category(metric: str) -> str:
    """Get category for computational metric."""
    return COMP_CATEGORIES.get(metric, 'confidence')


def extract_source_from_filename(filename: str) -> str:
    """Extract source name from filename like 'af3_recycle_0_plddt.csv' -> 'af3_recycle_0'"""
    name = re.sub(r'_(plddt|rmsf)\.csv$', '', filename)
    return name


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_relaxation_data(data_dir=str(RELAXATION_DIR)):
    """Load and parse all relaxation CSV files."""
    all_data = []
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            split_str = f.replace("_results.csv", "").split("_")
            well = split_str[0] + "_" + split_str[1]
            df_relax = pd.read_csv(f"{data_dir}/{f}")
            df_relax["source"] = well
            print(df_relax["source"])
            if "Residue" in df_relax.columns:
                df_relax.rename(columns={"Residue": "Residue ID"}, inplace=True)
            all_data.append(df_relax)
        elif f.endswith(".xlsx"):
            split_str = f.split("_")
            well = split_str[0] + "_" + split_str[1]
            df_relax = pd.read_excel(f"{data_dir}/{f}", engine="openpyxl")
            df_relax["source"] = well
            all_data.append(df_relax)

    return pd.concat(all_data, ignore_index=True)


def parse_uncertainty_columns(df):
    """Parse relaxation data columns into standardized _mean and _uncertainty columns."""
    df = df.copy()

    df.rename(columns={'R₁ [Hz]': 'R1', 'R₂ [Hz]': 'R2'}, inplace=True)
    
    if 'hetNOE' in df.columns and 'HetNOE' not in df.columns:
        df.rename(columns={'hetNOE': 'HetNOE', 'hetNOE_err': 'HetNOE_err'}, inplace=True)

    for col in ['HetNOE', 'R1', 'R2']:
        if col in df.columns:
            if df[col].dtype == 'object':
                split_data = df[col].str.split('+/-', regex=False, expand=True)
                df[f'{col}_mean'] = split_data[0].str.strip().astype(float)
                df[f'{col}_uncertainty'] = split_data[1].str.strip().astype(float)
            else:
                df[f'{col}_mean'] = df[col]
                if f'{col}_err' in df.columns:
                    df[f'{col}_uncertainty'] = df[f'{col}_err']

    return df


def create_complete_residue_grid(df, max_residue=102):
    """Create a complete grid of all residues for each source."""
    df = df[df["Residue ID"] <= max_residue].copy()
    df['is_present'] = True

    sources = df['source'].unique()
    all_residues = pd.DataFrame([
        {'Residue ID': residue_id, 'source': source}
        for source in sources
        for residue_id in range(1, max_residue + 1)
    ])

    result = all_residues.merge(df, on=['Residue ID', 'source'], how='left')
    result['is_present'] = result['is_present'].fillna(False)

    return result.sort_values(['source', 'Residue ID']).reset_index(drop=True)


def load_confidence_metrics(metrics_dir: Path = METRICS_PATH):
    """Auto-load pLDDT scores from all *_plddt.csv files."""
    plddt_data = {}
    
    for csv_file in sorted(metrics_dir.glob("*_plddt.csv")):
        source = extract_source_from_filename(csv_file.name)
        print(f"Loading pLDDT: {csv_file.name} -> {source}")
        
        df = pd.read_csv(csv_file)
        df["folder_name"] = df["folder_name"].str.replace("seq", "SEQ")
        if "af3" in str(csv_file):
            df["plddt"] /= 100
        df_summary = df.groupby(["folder_name", "residue_id"])["plddt"].mean().reset_index()
        df_summary.rename(columns={"plddt": f"plddt_{source}"}, inplace=True)
        
        plddt_data[source] = df_summary
    
    return plddt_data


def load_rmsf_metrics(metrics_dir: Path = METRICS_PATH):
    """Auto-load RMSF scores from all *_rmsf.csv files."""
    rmsf_data = {}
    
    for csv_file in sorted(metrics_dir.glob("*_rmsf.csv")):
        source = extract_source_from_filename(csv_file.name)
        print(f"Loading RMSF: {csv_file.name} -> {source}")
        
        df = pd.read_csv(csv_file)
        df["folder_name"] = df["folder_name"].str.replace("seq", "SEQ")
        df_summary = df.groupby(["folder_name", "residue_id"])["mean_rmsf"].mean().reset_index()
        print(df_summary)
        df_summary.rename(columns={"mean_rmsf": f"rmsf_{source}"}, inplace=True)
        
        rmsf_data[source] = df_summary
    
    return rmsf_data



def merge_all_metrics(nmr_df):
    """Merge all confidence metrics with NMR data."""
    metric_df = pd.read_csv(ALL_METRICS_AND_EXP_RESULTS_CSV)
    metric_df["source"] = metric_df["Database ID"].str.split("-").str[-1]
    print(metric_df["source"].unique())

    plddt_data = load_confidence_metrics()
    rmsf_data = load_rmsf_metrics()

    df = nmr_df.merge(metric_df[["source", "member"]], on="source", how="left")
    df["folder"] = df["member"]

    for source, plddt_df in plddt_data.items():
        plddt_df = plddt_df.copy()
        plddt_df["residue_id"] = plddt_df["residue_id"] + MSG_TAG_OFFSET
        df = df.merge(plddt_df, left_on=["folder", "Residue ID"],
                      right_on=["folder_name", "residue_id"], how="left", suffixes=('', f'_{source}'))
        df = df.drop(columns=[c for c in df.columns if c.startswith('folder_name') or c.startswith('residue_id_')], errors='ignore')
    
    for source, rmsf_df in rmsf_data.items():
        rmsf_df = rmsf_df.copy()
        rmsf_df["residue_id"] = rmsf_df["residue_id"] + MSG_TAG_OFFSET
        df = df.merge(rmsf_df, left_on=["folder", "Residue ID"],
                      right_on=["folder_name", "residue_id"], how="left", suffixes=('', f'_{source}'))
        df = df.drop(columns=[c for c in df.columns if c.startswith('folder_name') or c.startswith('residue_id_')], errors='ignore')

    plddt_cols = [col for col in df.columns if col.startswith('plddt_')]
    rmsf_cols = [col for col in df.columns if col.startswith('rmsf_')]

    if plddt_cols:
        df["mean_plddt_all"] = df[plddt_cols].mean(axis=1)
        df["max_plddt_all"] = df[plddt_cols].max(axis=1)
        df["min_plddt_all"] = df[plddt_cols].min(axis=1)
    
    if rmsf_cols:
        df["mean_rmsf_all"] = df[rmsf_cols].mean(axis=1)
        df["max_rmsf_all"] = df[rmsf_cols].max(axis=1)

    return df

# ============================================================================
# CORRELATION ANALYSIS - R² FROM LINEAR REGRESSION
# ============================================================================
def compute_r2_with_nmr(df, min_samples=10):
    """
    Compute R² from linear regression between model metrics and NMR observables.
    CRITICAL: Only use residues where is_present=True (have actual NMR data).
    """
    df_present = df[df["is_present"] == True].copy()

    plddt_cols = [col for col in df.columns if col.startswith('plddt_') 
                  and not col.endswith('_z') and not col.endswith('_rank')]
    rmsf_cols = [col for col in df.columns if col.startswith('rmsf_')
                 and not col.endswith('_z') and not col.endswith('_rank')]

    metrics = (
        plddt_cols + rmsf_cols +
        ["mean_plddt_all", "max_plddt_all", "min_plddt_all",
         "mean_rmsf_all", "max_rmsf_all",
         "plddt_rmsf_agreement"]
    )

    metrics = [m for m in metrics if m in df.columns]

    observables = ["R1_mean", "R2_mean", "HetNOE_mean"]

    # Per-folder R²
    folder_results = {}
    for folder in df_present["folder"].unique():
        df_folder = df_present[df_present["folder"] == folder]
        for metric in metrics:
            for observable in observables:
                # Skip self-correlation
                if metric == observable:
                    continue
                valid_data = df_folder.dropna(subset=[metric, observable])
                if len(valid_data) >= min_samples:
                    X = valid_data[[metric]].values
                    y = valid_data[observable].values
                    
                    reg = LinearRegression()
                    reg.fit(X, y)
                    r2 = reg.score(X, y)
                    
                    # Calculate p-value for the regression
                    n = len(valid_data)
                    if r2 < 1.0 and n > 2:
                        f_stat = (r2 / 1) / ((1 - r2) / (n - 2))
                        p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
                    else:
                        p_value = 0.0

                    # Calculate correlation for sign
                    corr = np.corrcoef(valid_data[metric], valid_data[observable])[0, 1]
                    
                    folder_results[(folder, metric, observable)] = {
                        'r2': r2, 'p': p_value, 'n': len(valid_data),
                        'corr_sign': np.sign(corr), 'slope': reg.coef_[0]
                    }

    # Pooled R² (across all folders)
    pooled_results = {}
    for metric in metrics:
        for observable in observables:
            # Skip self-correlation
            if metric == observable:
                continue
            valid_data = df_present.dropna(subset=[metric, observable])
            if len(valid_data) >= min_samples:
                X = valid_data[[metric]].values
                y = valid_data[observable].values
                
                reg = LinearRegression()
                reg.fit(X, y)
                r2 = reg.score(X, y)
                
                n = len(valid_data)
                if r2 < 1.0 and n > 2:
                    f_stat = (r2 / 1) / ((1 - r2) / (n - 2))
                    p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
                else:
                    p_value = 0.0
                
                corr = np.corrcoef(valid_data[metric], valid_data[observable])[0, 1]
                
                pooled_results[(metric, observable)] = {
                    'r2': r2, 'p': p_value, 'n': len(valid_data),
                    'corr_sign': np.sign(corr), 'slope': reg.coef_[0]
                }

    return folder_results, pooled_results


def create_r2_summary(folder_results, pooled_results):
    """Create DataFrames summarizing R² results."""
    folder_df = pd.DataFrame([
        {
            "folder": key[0],
            "metric": key[1],
            "observable": key[2],
            "r2": value['r2'],
            "p_value": value['p'],
            "n_samples": value['n'],
            "corr_sign": value['corr_sign'],
            "slope": value['slope']
        }
        for key, value in folder_results.items()
    ])

    pooled_df = pd.DataFrame([
        {
            "metric": key[0],
            "observable": key[1],
            "r2": value['r2'],
            "p_value": value['p'],
            "n_samples": value['n'],
            "corr_sign": value['corr_sign'],
            "slope": value['slope']
        }
        for key, value in pooled_results.items()
    ])

    return folder_df, pooled_df


# ============================================================================
# PUBLICATION PLOTTING FUNCTIONS
# ============================================================================

def plot_r2_heatmap(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    cluster: bool = True,
    weak_threshold: float = 0.05,
) -> plt.Figure:
    """
    Create publication-ready R² heatmap with clustering and weak correlation masking.
    
    Args:
        r2_df: DataFrame with columns ['observable', 'metric', 'r2']
        figsize: Figure size
        title: Optional title
        output_path: Path to save figure
        cluster: Whether to use hierarchical clustering
        weak_threshold: R² threshold below which to hide annotation text
    
    Returns:
        matplotlib Figure
    """
    df = r2_df.copy()
    
    # Map to display names
    df['nmr_display'] = df['observable'].map(get_nmr_display_name)
    df['comp_display'] = df['metric'].map(get_comp_display_name)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='nmr_display',
        columns='comp_display',
        values='r2'
    )

    # Create mask for weak correlations
    mask_weak = pivot < weak_threshold

    if cluster:
        # Use seaborn's clustermap for hierarchical clustering
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist

        # Calculate linkage for rows and columns
        # Use 1-R² as distance metric (so similar patterns cluster together)
        pivot_filled = pivot.fillna(0)

        # Row clustering (NMR metrics)
        row_distances = pdist(pivot_filled.values, metric='euclidean')
        row_linkage = hierarchy.linkage(row_distances, method='average')
        row_order = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']

        # Column clustering (computational metrics)
        col_distances = pdist(pivot_filled.T.values, metric='euclidean')
        col_linkage = hierarchy.linkage(col_distances, method='average')
        col_order = hierarchy.dendrogram(col_linkage, no_plot=True)['leaves']

        # Reorder pivot table
        pivot = pivot.iloc[row_order, col_order]
        mask_weak = mask_weak.iloc[row_order, col_order]

    # Create figure with more space
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1, left=0.15, right=0.88, top=0.92, bottom=0.25)
    ax = fig.add_subplot(gs[0])

    # Base colormap
    colors_strong = ['#FFFFFF', '#FEE5D9', '#FCBBA1', '#FC9272', 
                     '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']
    cmap_strong = LinearSegmentedColormap.from_list('rsquared', colors_strong)
    
    # Create custom annotation array - empty string for weak correlations
    annot_array = pivot.copy().astype(object)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.isna(val) or mask_weak.iloc[i, j]:
                annot_array.iloc[i, j] = ''
            else:
                annot_array.iloc[i, j] = f'{val:.2f}'

    # Plot heatmap
    heatmap = sns.heatmap(
        pivot,
        annot=annot_array,
        fmt='',
        cmap=cmap_strong,
        vmin=0,
        vmax=0.5,
        linewidths=0.5,
        linecolor='#CCCCCC',
        ax=ax,
        cbar_kws={
            'label': 'R²', 
            'shrink': 0.6,
            'aspect': 20,
            'pad': 0.02
        },
        annot_kws={'size': 7, 'weight': 'normal'},
        square=True,
    )
    
    # Color row labels by NMR category with better styling
    for i, label in enumerate(ax.get_yticklabels()):
        text = label.get_text()
        orig = [k for k, v in NMR_METRIC_NAME_MAP.items() if v == text]
        if orig:
            color = get_nmr_color(orig[0])
            label.set_color(color)
            label.set_weight('bold')
            label.set_fontsize(12)
    
    # Color column labels by comp category with better styling
    for i, label in enumerate(ax.get_xticklabels()):
        text = label.get_text()
        orig = [k for k, v in COMP_METRIC_NAME_MAP.items() if v == text]
        if orig:
            color = get_comp_color(orig[0])
            label.set_color(color)
            label.set_weight('bold')
            label.set_fontsize(12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Title with better styling
    title_text = title or 'R² Heatmap: NMR Relaxation vs Computational Metrics'
    ax.set_title(title_text, fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Computational Metric Type', fontsize=11, weight='bold', labelpad=10)
    ax.set_ylabel('NMR Metric Type', fontsize=11, weight='bold', labelpad=10)
    
    # Add category legends with improved layout
    nmr_patches = [
        mpatches.Patch(color=c, label=k.replace('_', ' ').title(), edgecolor='white', linewidth=0.5) 
        for k, c in NMR_CATEGORY_COLORS.items()
    ]
    comp_patches = [
        mpatches.Patch(color=c, label=k.replace('_', ' ').title(), edgecolor='white', linewidth=0.5)
        for k, c in COMP_CATEGORY_COLORS.items()
    ]
    
    # NMR legend (left side, below plot)
    leg1 = fig.legend(
        handles=nmr_patches, 
        loc='lower left',
        bbox_to_anchor=(0.15, -0.05),
        ncol=2, 
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9, 
        title='NMR Metric Type',
        title_fontsize=10,
        edgecolor='#CCCCCC',
        facecolor='white'
    )
    
    # Comp legend (right side, below plot)
    leg2 = fig.legend(
        handles=comp_patches, 
        loc='lower right',
        bbox_to_anchor=(0.55, -0.05),
        ncol=2, 
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9, 
        title='Computational Metric Type',
        title_fontsize=10,
        edgecolor='#CCCCCC',
        facecolor='white'
    )
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


def plot_top_correlations(
    r2_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create bar chart of top correlations by R²."""
    df = r2_df.nlargest(top_n, 'r2').copy()
    df['nmr_display'] = df['observable'].map(get_nmr_display_name)
    df['comp_display'] = df['metric'].map(get_comp_display_name)
    df['pair'] = df['comp_display'] + ' → ' + df['nmr_display']
    df = df.sort_values('r2', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [get_nmr_color(m) for m in df['observable']]
    
    bars = ax.barh(range(len(df)), df['r2'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['pair'], fontsize=9)
    
    for bar, r2, p in zip(bars, df['r2'], df['p_value']):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax.text(r2 + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{r2:.3f}{sig}', va='center', fontsize=8, color='#333333')
    
    ax.set_xlabel('R²', fontsize=11)
    ax.set_title(f'Top {top_n} Computational → NMR Relaxation Correlations',
                 fontsize=12, weight='bold')
    
    nmr_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title()) 
                   for k, c in NMR_CATEGORY_COLORS.items()]
    ax.legend(handles=nmr_patches, loc='lower right', frameon=False, fontsize=8,
              title='NMR Timescale')
    
    ax.set_xlim(0, df['r2'].max() * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


def plot_r2_barplot_ranked(
    r2_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[float, float] = (16, 7),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create vertical bar plot of top R² correlations, ordered by R² value.
    
    Each bar represents a metric-observable pair with hyphen-separated labels.
    Labels are colored by both computational metric type and NMR observable type.
    
    Args:
        r2_df: DataFrame with columns ['observable', 'metric', 'r2', 'p_value']
        top_n: Number of top correlations to show (default 20)
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Get top N correlations
    df = r2_df.nlargest(top_n, 'r2').copy()
    df = df.sort_values('r2', ascending=False)  # Sort descending for left-to-right
    
    # Create hyphen-separated pair labels
    df['comp_short'] = df['metric'].apply(lambda x: x.replace('_', '-'))
    df['nmr_short'] = df['observable'].apply(lambda x: x.replace('_', '-'))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color for a cleaner look
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(df))
    
    # Create gradient colors based on R² value for visual appeal
    # But also incorporate category color as edge
    bar_colors = [get_nmr_color(m) for m in df['observable']]
    edge_colors = [get_comp_color(m) for m in df['metric']]
    
    # Create bars with rounded appearance effect via alpha gradient
    bars = ax.bar(x, df['r2'], color=bar_colors, edgecolor=edge_colors, 
                  linewidth=2.5, width=0.75, zorder=3)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
    ax.set_axisbelow(True)
    
    # Add R² values on top of bars with significance
    for i, (bar, r2, p) in enumerate(zip(bars, df['r2'], df['p_value'])):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                f'{r2:.3f}{sig}', ha='center', va='bottom', fontsize=8, 
                color='#333333', weight='medium')
    
    # Create two-line x-axis labels with different colors
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Clear default labels
    
    # Add colored labels below x-axis
    for i, (idx, row) in enumerate(df.iterrows()):
        comp_color = get_comp_color(row['metric'])
        nmr_color = get_nmr_color(row['observable'])
        
        # Computational metric (top line)
        ax.text(i, -0.02, row['comp_short'], ha='center', va='top',
                fontsize=7, color=comp_color, weight='bold', rotation=45,
                transform=ax.get_xaxis_transform())
        # NMR metric (bottom line)  
        ax.text(i, -0.08, row['nmr_short'], ha='center', va='top',
                fontsize=7, color=nmr_color, weight='bold', rotation=45,
                transform=ax.get_xaxis_transform())
    
    # Labels and title
    ax.set_ylabel('R²', fontsize=13, weight='bold')
    ax.set_title(f'Top {top_n} Computational → NMR Relaxation Correlations',
                 fontsize=16, weight='bold', pad=20)
    
    # Create combined legend with separator
    nmr_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title(), 
                                   edgecolor='white', linewidth=0.5) 
                   for k, c in NMR_CATEGORY_COLORS.items()]
    comp_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title(),
                                    edgecolor='white', linewidth=0.5) 
                   for k, c in COMP_CATEGORY_COLORS.items()]
    
    # Position legends nicely
    leg1 = ax.legend(handles=nmr_patches, loc='upper right', frameon=True,
                     fancybox=True, fontsize=9, title='NMR (bar fill)',
                     title_fontsize=10, framealpha=0.95)
    leg1.get_frame().set_edgecolor('#CCCCCC')
    ax.add_artist(leg1)
    
    leg2 = ax.legend(handles=comp_patches, loc='upper right', frameon=True,
                     fancybox=True, fontsize=9, title='Computational (bar edge)',
                     title_fontsize=10, bbox_to_anchor=(0.78, 1.0), framealpha=0.95)
    leg2.get_frame().set_edgecolor('#CCCCCC')
    
    # Adjust axis limits
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(0, df['r2'].max() * 1.18)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Add significance note
    fig.text(0.5, -0.02, '*** p<0.001   ** p<0.01   * p<0.05', 
             ha='center', fontsize=9, style='italic', color='#888888')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


def plot_best_predictors(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create summary showing best computational predictor for each NMR metric."""
    best = r2_df.loc[r2_df.groupby('observable')['r2'].idxmax()].copy()
    best['nmr_display'] = best['observable'].map(get_nmr_display_name)
    best['comp_display'] = best['metric'].map(get_comp_display_name)
    
    best['nmr_cat'] = best['observable'].map(get_nmr_category)
    cat_order = {'fast_dynamics': 0, 'slow_dynamics': 1}
    best['cat_order'] = best['nmr_cat'].map(cat_order)
    best = best.sort_values(['cat_order', 'r2'], ascending=[True, False])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(best))
    colors = [get_nmr_color(m) for m in best['observable']]
    
    bars = ax.bar(x, best['r2'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(best['nmr_display'], rotation=45, ha='right', fontsize=9)
    
    for i, (bar, comp) in enumerate(zip(bars, best['comp_display'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                comp, ha='center', va='bottom', fontsize=7, rotation=90, color='#555555')
    
    ax.set_ylabel('Best R²', fontsize=11)
    ax.set_title('Best Computational Predictor for Each NMR Relaxation Metric',
                 fontsize=12, weight='bold')
    
    nmr_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title()) 
                   for k, c in NMR_CATEGORY_COLORS.items()]
    ax.legend(handles=nmr_patches, loc='upper right', frameon=False, fontsize=8)
    
    ax.set_ylim(0, best['r2'].max() * 1.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


def plot_category_summary(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 5),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Create heatmap summarizing R² by NMR and computational categories.
    
    Axes: Computational categories on X, NMR categories on Y.
    """
    df = r2_df.copy()
    df['nmr_cat'] = df['observable'].map(get_nmr_category)
    df['comp_cat'] = df['metric'].map(get_comp_category)
    
    cat_summary = df.groupby(['nmr_cat', 'comp_cat'])['r2'].agg(['mean', 'max', 'count']).reset_index()
    
    # NMR categories on rows (Y), computational categories on columns (X)
    pivot_mean = cat_summary.pivot(index='nmr_cat', columns='comp_cat', values='mean')
    pivot_max = cat_summary.pivot(index='nmr_cat', columns='comp_cat', values='max')
    
    nmr_order = ['fast_dynamics', 'slow_dynamics']
    comp_order = ['confidence', 'flexibility']
    
    pivot_mean = pivot_mean.reindex([n for n in nmr_order if n in pivot_mean.index])
    pivot_max = pivot_max.reindex([n for n in nmr_order if n in pivot_max.index])
    
    pivot_mean = pivot_mean[[c for c in comp_order if c in pivot_mean.columns]]
    pivot_max = pivot_max[[c for c in comp_order if c in pivot_max.columns]]
    
    pivot_mean.index = [n.replace('_', ' ').title() for n in pivot_mean.index]
    pivot_mean.columns = [c.replace('_', ' ').title() for c in pivot_mean.columns]
    pivot_max.index = pivot_mean.index
    pivot_max.columns = pivot_mean.columns
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap=R2_CMAP,
                vmin=0, vmax=0.15, linewidths=1, linecolor='white',
                ax=axes[0], cbar_kws={'label': 'Mean R²', 'shrink': 0.7})
    axes[0].set_title('A  Mean R² by Category', fontsize=11, weight='bold', loc='left')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    sns.heatmap(pivot_max, annot=True, fmt='.3f', cmap=R2_CMAP,
                vmin=0, vmax=0.3, linewidths=1, linecolor='white',
                ax=axes[1], cbar_kws={'label': 'Max R²', 'shrink': 0.7})
    axes[1].set_title('B  Max R² by Category', fontsize=11, weight='bold', loc='left')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


def plot_method_comparison(
    r2_df: pd.DataFrame,
    nmr_metrics_subset: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare prediction methods (AF3, Boltz1, Boltz2) for key NMR metrics."""
    df = r2_df.copy()
    
    if nmr_metrics_subset is None:
        nmr_metrics_subset = ['R1_mean', 'R2_mean', 'HetNOE_mean']
    
    df = df[df['observable'].isin(nmr_metrics_subset)]
    
    def extract_method(m):
        if 'af3' in m.lower():
            return 'AF3'
        elif 'boltz1' in m.lower():
            return 'Boltz-1'
        elif 'boltz2' in m.lower():
            return 'Boltz-2'
        else:
            return 'Other'
    
    df['method'] = df['metric'].apply(extract_method)
    df = df[df['method'] != 'Other']
    
    best_per_method = df.groupby(['observable', 'method'])['r2'].max().reset_index()
    best_per_method['nmr_display'] = best_per_method['observable'].map(get_nmr_display_name)
    
    pivot = best_per_method.pivot(index='nmr_display', columns='method', values='r2')
    method_cols = [m for m in ['AF3', 'Boltz-1', 'Boltz-2'] if m in pivot.columns]
    pivot = pivot[method_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(pivot))
    width = 0.25
    
    method_colors = {'AF3': '#3C5488', 'Boltz-1': '#E64B35', 'Boltz-2': '#00A087'}
    
    for i, method in enumerate(method_cols):
        bars = ax.bar(x + i*width, pivot[method], width, 
                     label=method, color=method_colors[method],
                     edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Best R²', fontsize=11)
    ax.set_title('Method Comparison: Predicting NMR Relaxation',
                 fontsize=12, weight='bold')
    ax.legend(frameon=False, fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig

def plot_r2_heatmap_by_observable_type(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (16, 10),
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    p_threshold: float = 0.05,
    r2_threshold: float = 0.0,
) -> plt.Figure:
    """Create publication-ready R² heatmap split by observable type.
    
    Creates 2 panels:
    1. Relaxation Rates (R1, R2)
    2. Dynamics (HetNOE)
    
    Gray out non-significant correlations for clarity.
    """
    df = r2_df.copy()
    
    # Add display names
    df['nmr_display'] = df['observable'].map(get_nmr_display_name)
    df['comp_display'] = df['metric'].map(get_comp_display_name)
    
    # Define observable groups
    observable_groups = {
        'Relaxation Rates': ['R1_mean', 'R2_mean'],
        'Dynamics': ['HetNOE_mean'],
    }
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes = np.atleast_1d(axes)
    
    # Track which computational metrics appear in significant results
    significant_metrics = set()
    for obs_list in observable_groups.values():
        obs_df = df[df['observable'].isin(obs_list)]
        sig_df = obs_df[(obs_df['p_value'] < p_threshold) & (obs_df['r2'] > r2_threshold)]
        significant_metrics.update(sig_df['comp_display'].unique())
    
    # Sort computational metrics by category
    comp_order = []
    for cat in ['confidence', 'flexibility']:
        metrics = [m for m, c in COMP_CATEGORIES.items() if c == cat]
        display_names = [get_comp_display_name(m) for m in metrics 
                        if get_comp_display_name(m) in significant_metrics]
        comp_order.extend(display_names)
    
    # Add any remaining metrics
    all_comp = df['comp_display'].unique()
    comp_order.extend([c for c in all_comp if c not in comp_order])
    
    # Plot each group
    for idx, (group_name, obs_list) in enumerate(observable_groups.items()):
        ax = axes[idx]
        
        # Filter data for this group
        group_df = df[df['observable'].isin(obs_list)].copy()
        
        # Create pivot table
        pivot = group_df.pivot_table(
            index='nmr_display',
            columns='comp_display',
            values='r2'
        )
        
        # Also get p-values
        pivot_p = group_df.pivot_table(
            index='nmr_display',
            columns='comp_display',
            values='p_value'
        )
        
        # Reorder columns
        available_cols = [c for c in comp_order if c in pivot.columns]
        pivot = pivot[available_cols]
        pivot_p = pivot_p[available_cols]
        
        # Create custom annotations with significance masking
        annot_array = np.empty_like(pivot.values, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                r2_val = pivot.iloc[i, j]
                p_val = pivot_p.iloc[i, j]
                
                if pd.isna(r2_val):
                    annot_array[i, j] = ''
                elif p_val < 0.001:
                    annot_array[i, j] = f'{r2_val:.3f}***'
                elif p_val < 0.01:
                    annot_array[i, j] = f'{r2_val:.3f}**'
                elif p_val < 0.05:
                    annot_array[i, j] = f'{r2_val:.3f}*'
                else:
                    # Non-significant - show but grayed
                    annot_array[i, j] = f'{r2_val:.3f}'
        
        # Create mask for non-significant values
        mask_nonsig = (pivot_p >= p_threshold) | (pivot <= r2_threshold)
        
        # Plot base heatmap
        sns.heatmap(
            pivot,
            annot=annot_array,
            fmt='',
            cmap=R2_CMAP,
            vmin=0,
            vmax=0.5,
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            cbar=idx == 0,  # Only show colorbar on first plot
            cbar_kws={'label': 'R²', 'shrink': 0.8} if idx == 0 else None,
            annot_kws={'size': 7}
        )
        
        # Gray out non-significant cells by overlaying semi-transparent white
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                if mask_nonsig.iloc[i, j]:
                    rect = plt.Rectangle((j, i), 1, 1, 
                                        facecolor='white', 
                                        alpha=0.6, 
                                        edgecolor='white',
                                        linewidth=0.5)
                    ax.add_patch(rect)
                    # Make text lighter
                    ax.texts[i * pivot.shape[1] + j].set_color('#AAAAAA')
                    ax.texts[i * pivot.shape[1] + j].set_fontsize(6)
        
        # Color row labels by NMR category
        for label in ax.get_yticklabels():
            text = label.get_text()
            orig = [k for k, v in NMR_METRIC_NAME_MAP.items() if v == text]
            if orig:
                color = get_nmr_color(orig[0])
                label.set_color(color)
                label.set_weight('medium')
        
        # Color column labels by comp category
        for label in ax.get_xticklabels():
            text = label.get_text()
            orig = [k for k, v in COMP_METRIC_NAME_MAP.items() if v == text]
            if orig:
                color = get_comp_color(orig[0])
                label.set_color(color)
                label.set_weight('medium')
                # Gray out non-significant metrics
                col_idx = list(pivot.columns).index(text)
                if mask_nonsig.iloc[:, col_idx].all():
                    label.set_color('#CCCCCC')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # Panel title
        panel_label = chr(65 + idx)  # A, B, C, D
        ax.set_title(f'{panel_label}  {group_name}', 
                    fontsize=11, weight='bold', loc='left', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Add legend at bottom
    fig.text(0.5, 0.02, 
            f'*** p<0.001  ** p<0.01  * p<0.05  |  Grayed cells: p≥{p_threshold} or R²≤{r2_threshold}',
            ha='center', fontsize=9, style='italic')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    else:
        fig.suptitle('R²: Computational Metrics vs NMR Relaxation by Observable Type',
                    fontsize=14, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


# Add this to generate_all_relaxation_figures function
def generate_all_relaxation_figures(
    r2_df: pd.DataFrame,
    output_dir: str = '.',
    prefix: str = 'fig_relaxation'
) -> Dict[str, plt.Figure]:
    """Generate all publication figures for relaxation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # NEW: Multi-panel heatmap by observable type
    figures['heatmap_by_type'] = plot_r2_heatmap_by_observable_type(
        r2_df,
        output_path=f'{output_dir}/{prefix}_r2_by_type.svg'
    )
    
    # Original full heatmap (kept for comparison)
    figures['heatmap'] = plot_r2_heatmap(
        r2_df,
        output_path=f'{output_dir}/{prefix}_r2_heatmap_full.svg'
    )
    
    figures['top_correlations'] = plot_top_correlations(
        r2_df,
        output_path=f'{output_dir}/{prefix}_top_correlations.svg'
    )
    
    # Ranked bar plot with colored labels
    figures['r2_barplot_ranked'] = plot_r2_barplot_ranked(
        r2_df,
        top_n=20,
        output_path=f'{output_dir}/{prefix}_r2_barplot_ranked.svg'
    )
    
    figures['best_predictors'] = plot_best_predictors(
        r2_df,
        output_path=f'{output_dir}/{prefix}_best_predictors.svg'
    )
    
    figures['category_summary'] = plot_category_summary(
        r2_df,
        output_path=f'{output_dir}/{prefix}_category_summary.svg'
    )
    
    figures['method_comparison'] = plot_method_comparison(
        r2_df,
        output_path=f'{output_dir}/{prefix}_method_comparison.svg'
    )
    
    print(f"\nGenerated {len(figures)} publication figures in {output_dir}/")
    
    return figures

# ============================================================================
# AUROC AND CLASSIFICATION (kept from original)
# ============================================================================

def compute_auroc_for_presence(df):
    """Compute AUROC for predicting residue presence in NMR data."""
    auroc_results = {}

    plddt_cols = [col for col in df.columns if col.startswith('plddt_') 
                  and not col.endswith('_z') and not col.endswith('_rank')]
    rmsf_cols = [col for col in df.columns if col.startswith('rmsf_')
                 and not col.endswith('_z') and not col.endswith('_rank')]

    positive_metrics = plddt_cols + [
        "mean_plddt_all", "max_plddt_all", "min_plddt_all",
        "consensus_rigidity_z", "consensus_rigidity_rank",
        "plddt_rmsf_agreement"
    ]
    negative_metrics = rmsf_cols + [
        "mean_rmsf_all", "max_rmsf_all"
    ]
    
    positive_metrics = [m for m in positive_metrics if m in df.columns]
    negative_metrics = [m for m in negative_metrics if m in df.columns]

    for metric in positive_metrics + negative_metrics:
        valid_data = df.dropna(subset=[metric])
        if valid_data["is_present"].nunique() > 1 and len(valid_data) > 0:
            if metric in negative_metrics:
                auroc = roc_auc_score(valid_data["is_present"], -valid_data[metric])
            else:
                auroc = roc_auc_score(valid_data["is_present"], valid_data[metric])
            auroc_results[metric] = auroc
        else:
            auroc_results[metric] = None

    return auroc_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load and process NMR data
    print("Loading relaxation data...")
    nmr_data = load_relaxation_data()
    nmr_data = parse_uncertainty_columns(nmr_data)
    nmr_data = create_complete_residue_grid(nmr_data, max_residue=100)

    # Merge with model metrics
    print("Merging with model confidence metrics...")
    print(f"  (Applying MSG tag offset: computational residue + {MSG_TAG_OFFSET} = NMR residue)")
    full_df = merge_all_metrics(nmr_data)


    # Compute AUROC for residue presence prediction
    print("\nComputing AUROC for residue presence prediction...")
    auroc_results = compute_auroc_for_presence(full_df)
    print("\nAUROC Results:")
    for metric, auroc in auroc_results.items():
        if auroc is not None:
            print(f"  {metric}: {auroc:.3f}")

    # Compute R² with NMR observables
    print("\nComputing R² with NMR observables...")
    folder_results, pooled_results = compute_r2_with_nmr(full_df)
    folder_df, pooled_df = create_r2_summary(folder_results, pooled_results)

    # Display pooled results
    print("\nPooled R² Results (R² > 0.05):")
    significant = pooled_df[pooled_df["r2"] > 0.05].sort_values("r2", ascending=False)
    print(significant.to_string(index=False))

    # Highlight consensus metrics performance
    print("\n" + "="*70)
    print("CONSENSUS METRICS PERFORMANCE")
    print("="*70)
    consensus_metrics = ["consensus_rigidity_z", "consensus_rigidity_rank",
                        "plddt_rmsf_agreement"]
    consensus_results = pooled_df[pooled_df["metric"].isin(consensus_metrics)]
    if len(consensus_results) > 0:
        print("\nTop consensus metric R² values:")
        top_consensus = consensus_results.sort_values("r2", ascending=False).head(10)
        print(top_consensus[["metric", "observable", "r2", "p_value"]].to_string(index=False))
    else:
        print("No consensus metrics found in results.")

    # Generate publication figures
    print("\n" + "="*70)
    print("GENERATING PUBLICATION FIGURES")
    print("="*70)
    
    output_dir = str(RELAX_OUT / "figures")
    figures = generate_all_relaxation_figures(pooled_df, output_dir=output_dir)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    folder_df.to_csv(RELAX_OUT / "relaxation_r2_results_per_folder.csv", index=False)
    pooled_df.to_csv(RELAX_OUT / "relaxation_r2_results_pooled.csv", index=False)
    full_df.to_csv(RELAX_OUT / "relaxation_full_merged_data.csv", index=False)

    print(f"\nDone! Results saved under: {RELAX_OUT}")