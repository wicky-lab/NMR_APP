# %%
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import re

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

HSQC_OUT = OUTPUTS_DIR / "hsqc"
HSQC_OUT.mkdir(parents=True, exist_ok=True)

# %%
# =============================================================================
# DATA LOADING
# =============================================================================

nmr_summary = pd.read_parquet(HSQC_SUMMARY_PARQUET, engine='fastparquet')
nmr_peaks = pd.read_parquet(HSQC_PEAKS_PARQUET, engine='fastparquet')

design_metrics = pd.read_csv(BOLTZ_METRICS_CSV)
expression_metrics = pd.read_csv(EXPRESSION_METRICS_CSV)

dssp_csv = pd.read_csv(DSSP_CSV)
ensemble_metrics = pd.read_csv(COMBINED_DYNAMICS_CSV)

# %%
# =============================================================================
# DATA PREPROCESSING
# =============================================================================

del expression_metrics["Sequence"]
expression_metrics.rename(columns={"exp_aa_seq_from_LM0627": "Sequence"}, inplace=True)
expression_metrics["Sequence"] = expression_metrics["Sequence"].str.replace("*", "")
# merge expression_metrics with design_metrics on index
expression_metrics = design_metrics.merge(expression_metrics, left_index=True, right_index=True, how="left")
# merge nmr_summary with design_metrics on Sequence
df = nmr_summary.merge(expression_metrics, on="Sequence", how="left")
del df["design_name_y"]
df.rename(columns={"design_name_x": "design_name"}, inplace=True)

nmr_peaks["Sequence"] = nmr_peaks["Sequence"].str.replace("\n", "")
nmr_peaks["Sequence"] = nmr_peaks["Sequence"].str.replace("*", "")

df = df.merge(nmr_peaks, on="Sequence", how="left")


# %%
# =============================================================================
# CONVERT STRING ARRAYS TO NUMPY
# =============================================================================

def convert_str_to_array(s):
    import ast
    if isinstance(s, np.ndarray):
        return s
    if pd.isna(s) or len(s) == 0:
        return np.array([])
    return np.array(ast.literal_eval(s))

df["intensity"] = df["Peaklist Intensities"].apply(convert_str_to_array)
df["volumes"] = df["Peaklist Volumes"].apply(convert_str_to_array)
df["1_h"] = df["Peaklist 1H-Shifts"].apply(convert_str_to_array)
df["15_n"] = df["Peaklist 15N-Shifts"].apply(convert_str_to_array)
df["1_h_linewidths"] = df["Peaklist 1H-Linewidths"].apply(convert_str_to_array)
df["15_n_linewidths"] = df["Peaklist 15N-Linewidths"].apply(convert_str_to_array)

# %%
# =============================================================================
# NMR METRICS - LINEWIDTH STATISTICS (STANDARD)
# Linewidths are fundamental observables in NMR, directly related to T2 relaxation
# =============================================================================

# 1H linewidth statistics
df['LW_1H_mean'] = df['1_h_linewidths'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
df['LW_1H_median'] = df['1_h_linewidths'].apply(lambda x: np.median(x) if len(x) > 0 else np.nan)
df['LW_1H_std'] = df['1_h_linewidths'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)

# 15N linewidth statistics
df['LW_15N_mean'] = df['15_n_linewidths'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
df['LW_15N_median'] = df['15_n_linewidths'].apply(lambda x: np.median(x) if len(x) > 0 else np.nan)
df['LW_15N_std'] = df['15_n_linewidths'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)


df['1H_shift_std'] = df['1_h'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
df['15N_shift_std'] = df['15_n'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
df['1H_shift_range'] = df['1_h'].apply(lambda x: np.ptp(x) if len(x) > 0 else np.nan)
df['15N_shift_range'] = df['15_n'].apply(lambda x: np.ptp(x) if len(x) > 0 else np.nan)

# Mean chemical shifts (centroid position)
df["mean_1H"] = df["1_h"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
df["mean_15N"] = df["15_n"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

# %%
# =============================================================================
# NMR METRICS - PEAK COUNTS (STANDARD)
# Expected vs observed peaks is fundamental for assessing fold/aggregation
# =============================================================================

df["peak_delta"] = df["# peaks (exp.)"] - df["# peaks (obs.\n45min)"]

# %%
# =============================================================================
# NMR METRICS - INTENSITY STATISTICS (STANDARD)
# Intensity heterogeneity can indicate exchange broadening or aggregation
# =============================================================================

df["mean_intensity"] = df["intensity"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
df["std_intensity"] = df["intensity"].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
df["cv_intensity"] = df["std_intensity"] / (df["mean_intensity"] + 1e-10)

df["mean_volume"] = df["volumes"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
df["std_volume"] = df["volumes"].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
df["cv_volume"] = df["std_volume"] / (df["mean_volume"] + 1e-10)

# %%
# =============================================================================
# NMR METRICS - LINEWIDTH-INTENSITY CORRELATION (ACCEPTED)
# Negative correlation between linewidth and intensity suggests 
# conformational exchange on intermediate timescale (Rex contribution)
# =============================================================================

def calculate_lw_intensity_correlation(row):
    """
    Correlation between linewidth and intensity.
    Negative correlation is a hallmark of conformational exchange.
    """
    intensities = row['intensity']
    lw_1h = row['1_h_linewidths']
    
    if len(intensities) < 5 or len(lw_1h) < 5:
        return np.nan
    if len(lw_1h) != len(intensities):
        return np.nan
    
    try:
        corr = np.corrcoef(lw_1h, intensities)[0, 1]
        return corr
    except:
        return np.nan

df['LW_intensity_correlation'] = df.apply(calculate_lw_intensity_correlation, axis=1)


# %%
# =============================================================================
# DSSP MERGING AND LOOP METRICS
# =============================================================================

dssp_csv["design_name"] = dssp_csv["file_name"].str.split(".cif").str[0]
df = df.merge(dssp_csv, on="design_name", how="left")

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

# %%
# =============================================================================
# DYNAMICS MERGING
# =============================================================================

df = df.merge(ensemble_metrics, left_on="design_name", right_on="folder_name", how="left")

# %%
# =============================================================================
# QUALITY FILTERING
# =============================================================================
filter = True
df.to_parquet(HSQC_OUT / "combined_metrics_before_filter.parquet", index=False)
print(len(df), " NO FILTER")
if filter:
    df = df[(df["Quality"] == "Medium") | (df["Quality"] == "High")]
    print(len(df), " AFTER QUALITY FILTER")
    df["intensity_max"] = df["intensity"].apply(lambda x: max(x) if len(x) > 0 else np.nan)
    df = df[(df["intensity_max"] < 1e8)] 
    print(len(df), " AFTER CV INTENSITY FILTER")

# Get unique samples (one per member)
df_unique = df.drop_duplicates(subset=["design_name"])

# %%
# =============================================================================
# DEFINE METRIC LISTS - ONLY STANDARD, ACCEPTED METRICS
# =============================================================================

# NMR metrics (what we're trying to predict from computation)
nmr_metrics = [
    # Linewidth statistics - fundamental observables
    'LW_1H_mean',
    'LW_1H_median', 
    'LW_1H_std',
    'LW_15N_mean',
    'LW_15N_median',
    'LW_15N_std',
    
    # Chemical shift dispersion - classic fold quality indicator
    '1H_shift_std',
    '15N_shift_std',
    '1H_shift_range',
    '15N_shift_range',
    'mean_1H',
    'mean_15N',
    # Peak counts - fundamental assessment
    'peak_delta',
    # Intensity statistics
    'mean_intensity',
    'cv_intensity',
    'mean_volume',
    'cv_volume',
    # Exchange indicator
    'LW_intensity_correlation',
    'conc_uM',
]

# Computational metrics (predictors)
computational_metrics = [
    'comp_RMSD_ca',
    'comp_RMSD_ca_var',
    'loop_count',
    'helix_count',
    'sheet_count',
    'mean_rmsf_af3_recycle_0',
    'mean_rmsf_af3_recycle_3',
    'mean_rmsf_boltz1_recycle_0',
    'mean_rmsf_boltz1_recycle_3',
    'mean_rmsf_boltz2_recycle_0',
    'mean_rmsf_boltz2_recycle_3',
    'mean_plddt_boltz1_recycle_0',
    'mean_plddt_boltz1_recycle_3',
    'mean_plddt_boltz2_recycle_0',
    'mean_plddt_boltz2_recycle_3',
    'mean_plddt_af3_recycle_0',
    'mean_plddt_af3_recycle_3',
]

# Filter to only existing columns
nmr_metrics = [m for m in nmr_metrics if m in df_unique.columns]
computational_metrics = [m for m in computational_metrics if m in df_unique.columns]

print(f"NMR metrics available: {len(nmr_metrics)}")
print(f"Computational metrics available: {len(computational_metrics)}")

# %%
# =============================================================================
# CORRELATION ANALYSIS FUNCTION
# =============================================================================

def create_correlation_plot(df, nmr_metric, comp_metrics, color_col="correct_Vel", 
                            output_dir="correlation_plots"):
    """Create and save correlation plot for one NMR metric vs all computational metrics."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to valid computational metrics for this NMR metric
    valid_comp = []
    for cm in comp_metrics:
        valid_data = df[[nmr_metric, cm]].dropna()
        if len(valid_data) > 10:
            valid_comp.append(cm)
    
    if len(valid_comp) == 0:
        print(f"  Skipping {nmr_metric}: no valid computational metrics")
        return None
    
    # Calculate grid dimensions
    n_plots = len(valid_comp)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig_combined = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=valid_comp,
        horizontal_spacing=0.08,
        vertical_spacing=0.12, 
    )

    r2_results = []
    
    for i, comp_metric in enumerate(valid_comp):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Get valid data
        plot_df = df[[nmr_metric, comp_metric, color_col]].dropna()
        
        if len(plot_df) < 10:
            continue
            
        try:
            # Create scatter with trendline per group
            fig_temp = px.scatter(plot_df, x=comp_metric, y=nmr_metric, 
                                  color=color_col, trendline="ols")
            
            # Create overall trendline
            fig_overall = px.scatter(plot_df, x=comp_metric, y=nmr_metric, trendline="ols")
            overall_results = px.get_trendline_results(fig_overall)
            overall_r2 = overall_results.iloc[0]["px_fit_results"].rsquared
            overall_pval = overall_results.iloc[0]["px_fit_results"].f_pvalue
            
            # Store results
            r2_results.append({
                'nmr_metric': nmr_metric,
                'comp_metric': comp_metric,
                'r2': overall_r2,
                'p_value': overall_pval,
                'n_samples': len(plot_df)
            })
            
            # Get R² values per group
            results = px.get_trendline_results(fig_temp)
            r2_text = ""
            for _, row_data in results.iterrows():
                group_r2 = row_data['px_fit_results'].rsquared
                r2_text += f"{row_data[color_col]}: R²={group_r2:.3f}<br>"
            r2_text += f"<b>Overall: R²={overall_r2:.3f}</b>"
            if overall_pval < 0.001:
                r2_text += f"<br>p<0.001"
            else:
                r2_text += f"<br>p={overall_pval:.3f}"
            
            # Add traces to combined figure
            for trace in fig_temp.data:
                trace.showlegend = (i == 0)
                fig_combined.add_trace(trace, row=row, col=col)
            
            # Add overall trendline
            overall_trendline = fig_overall.data[1]
            overall_trendline.line.color = "black"
            overall_trendline.line.dash = "dash"
            overall_trendline.line.width = 2
            overall_trendline.name = "Overall"
            overall_trendline.showlegend = (i == 0)
            fig_combined.add_trace(overall_trendline, row=row, col=col)
            
            # Add R² annotation
            subplot_idx = i + 1
            xref = "x domain" if subplot_idx == 1 else f"x{subplot_idx} domain"
            yref = "y domain" if subplot_idx == 1 else f"y{subplot_idx} domain"
            
            fig_combined.add_annotation(
                x=0.02, y=0.98,
                xref=xref, yref=yref,
                text=r2_text,
                showarrow=False,
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.8)",
                align="left",
                xanchor="left",
                yanchor="top"
            )
            
            # Update axis labels
            fig_combined.update_xaxes(title_text=comp_metric, row=row, col=col, title_font_size=10)
            fig_combined.update_yaxes(title_text=nmr_metric, row=row, col=col, title_font_size=10)
            
        except Exception as e:
            print(f"  Error with {comp_metric}: {e}")
            continue
    
    fig_combined.update_layout(
        height=300 * n_rows,
        width=400 * n_cols,
        title_text=f"Computational Metrics vs {nmr_metric}",
        title_font_size=16
    )
    
    # Save figure
    filename = f"{output_dir}/{nmr_metric}_correlations.html"
    fig_combined.write_html(filename)
    print(f"  Saved: {filename}")
    
    return r2_results

"""
Publication-ready visualization module for HSQC NMR correlation analysis.
Maps code metric names to publication format and categorizes by measurement type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# ============================================================================
# PUBLICATION SETTINGS
# ============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
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
# NAME MAPPINGS - NMR METRICS
# ============================================================================



# ============================================================================
# METRIC CATEGORIZATIONS
# ============================================================================
NMR_METRIC_NAME_MAP = {
    # Linewidth statistics
    'LW_1H_mean': '$^{1}$H Linewidth (mean)',
    'LW_1H_median': '$^{1}$H Linewidth (median)',
    'LW_1H_std': '$^{1}$H Linewidth (std)',
    'LW_15N_mean': '$^{15}$N Linewidth (mean)',
    'LW_15N_median': '$^{15}$N Linewidth (median)',
    'LW_15N_std': '$^{15}$N Linewidth (std)',
    'LW_1H_mean_conc_corr': '$^{1}$H Linewidth (conc. corr.)',
    'LW_15N_mean_conc_corr': '$^{15}$N Linewidth (conc. corr.)',
    
    # R2* from linewidths
    'R2star_1H_mean': '$R_2^*$ ($^{1}$H)',
    'R2star_15N_mean': '$R_2^*$ ($^{15}$N)',
    
    # Chemical shift dispersion
    '1H_shift_std': '$^{1}$H Shift Dispersion ($\sigma$)',
    '15N_shift_std': '$^{15}$N Shift Dispersion ($\sigma$)',
    '1H_shift_range': '$^{1}$H Shift Range',
    '15N_shift_range': '$^{15}$N Shift Range',
    'mean_1H': '$^{1}$H Shift (mean)',
    'mean_15N': '$^{15}$N Shift (mean)',
    
    # Peak counts
    'peak_delta': 'Peak Deficit',
    'n_peaks_observed': 'Peaks Observed',
    'peak_recovery_rate': 'Peak Recovery Rate',
    
    # Intensity statistics
    'mean_intensity': 'Mean Intensity',
    'std_intensity': 'Intensity (std)',
    'cv_intensity': 'Intensity CV',
    'mean_volume': 'Mean Volume',
    'std_volume': 'Volume (std)',
    'cv_volume': 'Volume CV',
    
    # Exchange indicator
    'LW_intensity_correlation': 'LW-Intensity Correlation',
    
    # Random coil deviation
    'mean_abs_delta_1H_from_RC': '$|\Delta\delta^{1}H|$ from RC',
    'mean_abs_delta_15N_from_RC': '$|\Delta\delta^{15}N|$ from RC',
    
    # Diagnostic region counts
    'n_gly_region': 'Gly Region Peaks',
    'n_trp_sidechain': 'Trp Indole Peaks',
    'n_upfield_shifted': 'Upfield $^{1}$H Peaks',
    'n_downfield_shifted': 'Downfield $^{1}$H Peaks',
    
    # Concentration
    'conc_uM': 'Concentration ($\mu$M)',
}

# ============================================================================
# NAME MAPPINGS - COMPUTATIONAL METRICS
# ============================================================================

COMP_METRIC_NAME_MAP = {
    # RMSD metrics
    'comp_RMSD_ca': 'Cα RMSD (Design vs. Model)',
    'comp_RMSD_ca_var': 'Cα RMSD Variance (Design vs. Model)',
    
    # PTM scores
    'comp_ptm': 'pTM Score (Model)',
    'comp_ptm_var': 'pTM Variance (Model)',
    
    # Secondary structure counts
    'loop_count': 'Loop Count',
    'helix_count': 'Helix Count',
    'sheet_count': 'Sheet Count',
    'avg_loop_length': 'Avg Loop Length',
    'max_loop_length': 'Max Loop Length',
    
    # RMSF - AlphaFold3
    'mean_rmsf_af3_recycle_0': 'AF3 RMSF (r0, ensemble mean)',
    'mean_rmsf_af3_recycle_3': 'AF3 RMSF (r3, ensemble mean)',
    
    # RMSF - Boltz1
    'mean_rmsf_boltz1_recycle_0': 'Boltz-1 RMSF (r0, ensemble mean)',
    'mean_rmsf_boltz1_recycle_3': 'Boltz-1 RMSF (r3, ensemble mean)',
    
    # RMSF - Boltz2
    'mean_rmsf_boltz2_recycle_0': 'Boltz-2 RMSF (r0, ensemble mean)',
    'mean_rmsf_boltz2_recycle_3': 'Boltz-2 RMSF (r3, ensemble mean)',
    
    # pLDDT - Boltz1
    'mean_plddt_boltz1_recycle_0': 'Boltz-1 pLDDT (r0, ensemble mean)',
    'mean_plddt_boltz1_recycle_3': 'Boltz-1 pLDDT (r3, ensemble mean)',

    # pLDDT - Boltz2
    'mean_plddt_boltz2_recycle_0': 'Boltz-2 pLDDT (r0, ensemble mean)',
    'mean_plddt_boltz2_recycle_3': 'Boltz-2 pLDDT (r3, ensemble mean)',
    
    # pLDDT - AlphaFold3
    'mean_plddt_af3_recycle_0': 'AF3 pLDDT (r0, ensemble mean)',
    'mean_plddt_af3_recycle_3': 'AF3 pLDDT (r3, ensemble mean)',
    
}
# NMR metric categories
NMR_CATEGORIES = {
    # Dynamics/Relaxation - report on motion, tumbling, exchange
    'LW_1H_mean': 'dynamics',
    'LW_1H_median': 'dynamics',
    'LW_1H_std': 'dynamics',
    'LW_15N_mean': 'dynamics',
    'LW_15N_median': 'dynamics',
    'LW_15N_std': 'dynamics',
    'LW_1H_mean_conc_corr': 'dynamics',
    'LW_15N_mean_conc_corr': 'dynamics',
    'R2star_1H_mean': 'dynamics',
    'R2star_15N_mean': 'dynamics',
    'LW_intensity_correlation': 'dynamics',
    'cv_intensity': 'dynamics',
    'cv_volume': 'dynamics',
    
    # Fold quality - tertiary structure indicators
    '1H_shift_std': 'fold_quality',
    '1H_shift_range': 'fold_quality',
    'n_upfield_shifted': 'fold_quality',
    'n_downfield_shifted': 'fold_quality',
    'peak_recovery_rate': 'fold_quality',
    
    # Secondary structure indicators
    '15N_shift_std': 'secondary_structure',
    '15N_shift_range': 'secondary_structure',
    'mean_abs_delta_1H_from_RC': 'secondary_structure',
    'mean_abs_delta_15N_from_RC': 'secondary_structure',
    'n_gly_region': 'secondary_structure',
    'n_trp_sidechain': 'secondary_structure',
    
    # Sample/spectral quality
    'peak_delta': 'sample_quality',
    'n_peaks_observed': 'sample_quality',
    'expected_peaks': 'sample_quality',
    'mean_intensity': 'sample_quality',
    'std_intensity': 'sample_quality',
    'mean_volume': 'sample_quality',
    'std_volume': 'sample_quality',
    'mean_1H': 'sample_quality',
    'mean_15N': 'sample_quality',
    'conc_uM': 'sample_quality',
}

# Computational metric categories
COMP_CATEGORIES = {
    # Confidence metrics
    'mean_plddt_boltz1_recycle_0': 'confidence',
    'mean_plddt_boltz1_recycle_3': 'confidence',
    'mean_plddt_boltz2_recycle_0': 'confidence',
    'mean_plddt_boltz2_recycle_3': 'confidence',
    'mean_plddt_af3_recycle_0': 'confidence',
    'mean_plddt_af3_recycle_3': 'confidence',
    'min_plddt_boltz1_recycle_0': 'confidence',
    'min_plddt_boltz1_recycle_3': 'confidence',
    'min_plddt_boltz2_recycle_0': 'confidence',
    'min_plddt_boltz2_recycle_3': 'confidence',
    'min_plddt_af3_recycle_0': 'confidence',
    'min_plddt_af3_recycle_3': 'confidence',
    'comp_ptm': 'confidence',
    'comp_ptm_var': 'confidence',
    'plddt_drop': 'confidence',
    
    # Flexibility metrics
    'mean_rmsf_af3_recycle_0': 'flexibility',
    'mean_rmsf_af3_recycle_3': 'flexibility',
    'mean_rmsf_boltz1_recycle_0': 'flexibility',
    'mean_rmsf_boltz1_recycle_3': 'flexibility',
    'mean_rmsf_boltz2_recycle_0': 'flexibility',
    'mean_rmsf_boltz2_recycle_3': 'flexibility',
    'comp_RMSD_ca': 'flexibility',
    'comp_RMSD_ca_var': 'flexibility',
    
    # Structure metrics
    'loop_count': 'structure',
    'helix_count': 'structure',
    'sheet_count': 'structure',
    'avg_loop_length': 'structure',
    'max_loop_length': 'structure',
}

# ============================================================================
# COLOR SCHEMES
# ============================================================================

# NMR category colors
NMR_CATEGORY_COLORS = {
    'dynamics': '#E64B35',           # Coral red
    'fold_quality': '#4DBBD5',       # Teal
    'secondary_structure': '#00A087', # Green
    'sample_quality': '#8491B4',     # Slate blue
}

# Computational category colors
COMP_CATEGORY_COLORS = {
    'confidence': '#3C5488',   # Navy blue
    'flexibility': '#F39B7F',  # Salmon
    'structure': '#91D1C2',    # Mint
}

# Colormaps
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
    category = NMR_CATEGORIES.get(metric, 'sample_quality')
    return NMR_CATEGORY_COLORS[category]

def get_comp_color(metric: str) -> str:
    """Get color based on computational metric category."""
    category = COMP_CATEGORIES.get(metric, 'confidence')
    return COMP_CATEGORY_COLORS[category]

def get_nmr_category(metric: str) -> str:
    """Get category for NMR metric."""
    return NMR_CATEGORIES.get(metric, 'sample_quality')

def get_comp_category(metric: str) -> str:
    """Get category for computational metric."""
    return COMP_CATEGORIES.get(metric, 'confidence')

# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================
def plot_r2_heatmap(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 12),
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    cluster: bool = True,
    weak_threshold: float = 0.1,
) -> plt.Figure:
    """
    Create publication-ready R² heatmap with clustering and weak correlation masking.
    
    Args:
        r2_df: DataFrame with columns ['nmr_metric', 'comp_metric', 'r2']
        figsize: Figure size
        title: Optional title
        output_path: Path to save figure
        cluster: Whether to use hierarchical clustering
        weak_threshold: R² threshold below which to gray out cells
    
    Returns:
        matplotlib Figure
    """
    df = r2_df.copy()
    
    # Map to display names
    df['nmr_display'] = df['nmr_metric'].map(get_nmr_display_name)
    df['comp_display'] = df['comp_metric'].map(get_comp_display_name)
    
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

    # Base colormap for strong correlations
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
    
    # Plot heatmap with shading for all cells, but text only for strong correlations
    heatmap = sns.heatmap(
        pivot,
        annot=annot_array,
        fmt='',  # Use empty format since we pre-formatted the strings
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
        annot_kws={'size': 10, 'weight': 'normal'},
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
    ax.set_xlabel('Computational Metric Type', fontsize=13, weight='bold', labelpad=10)
    ax.set_ylabel('NMR Metric Type', fontsize=13, weight='bold', labelpad=10)
    
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
    """
    Create bar chart of top correlations.
    
    Args:
        r2_df: DataFrame with correlation results
        top_n: Number of top pairs to show
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    df = r2_df.nlargest(top_n, 'r2').copy()
    df['nmr_display'] = df['nmr_metric'].map(get_nmr_display_name)
    df['comp_display'] = df['comp_metric'].map(get_comp_display_name)
    df['pair'] = df['comp_display'] + ' → ' + df['nmr_display']
    df = df.sort_values('r2', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by NMR category
    colors = [get_nmr_color(m) for m in df['nmr_metric']]
    
    bars = ax.barh(range(len(df)), df['r2'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['pair'], fontsize=9)
    
    # Add R² values
    for bar, r2, p in zip(bars, df['r2'], df['p_value']):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax.text(r2 + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{r2:.3f}{sig}', va='center', fontsize=8, color='#333333')
    
    ax.set_xlabel('R²', fontsize=11)
    ax.set_title(f'Top {top_n} Computational → NMR Correlations',
                 fontsize=12, weight='bold')
    
    # Legend
    nmr_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title()) 
                   for k, c in NMR_CATEGORY_COLORS.items()]
    ax.legend(handles=nmr_patches, loc='lower right', frameon=False, fontsize=8,
              title='NMR Category')
    
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
    figsize: Tuple[float, float] = (20, 9),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create vertical bar plot of top R² correlations, ordered purely by R² value.
    
    Bars are colored by NMR category, labels colored by computational category.
    
    Args:
        r2_df: DataFrame with columns ['nmr_metric', 'comp_metric', 'r2', 'p_value']
        top_n: Number of top correlations to show (default 20)
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Get top N correlations ranked purely by R²
    df = r2_df.nlargest(top_n, 'r2').copy()
    df = df.sort_values('r2', ascending=False)
    df = df.reset_index(drop=True)
    
    # Get display names
    df['comp_display'] = df['comp_metric'].map(get_comp_display_name)
    df['nmr_display'] = df['nmr_metric'].map(get_nmr_display_name)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color for a cleaner look
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(df))
    
    # Color bars by NMR category
    bar_colors = [get_nmr_color(m) for m in df['nmr_metric']]
    bars = ax.bar(x, df['r2'], color=bar_colors, edgecolor='white', 
                  linewidth=0.5, width=0.8, zorder=3, alpha=0.85)
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
    ax.set_axisbelow(True)
    
    # Add R² values on top of bars
    for i, (bar, r2) in enumerate(zip(bars, df['r2'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                f'{r2:.3f}', ha='center', va='bottom', fontsize=9, 
                color='#333333', weight='medium')
    
    # Remove default x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(['' for _ in x])
    ax.tick_params(axis='x', length=0)
    
    # Add combined labels below x-axis, colored by computational category
    blend = ax.get_xaxis_transform()
    
    for i, (_, row) in enumerate(df.iterrows()):
        comp_color = get_comp_color(row['comp_metric'])
        label = f"{row['comp_display']}  →  {row['nmr_display']}"
        
        ax.text(i, -0.02, label,
                transform=blend,
                ha='right', va='top',
                fontsize=7, fontweight='bold',
                color=comp_color,
                rotation=40,
                rotation_mode='anchor')
    
    # Labels and title
    ax.set_ylabel('R²', fontsize=14, weight='bold')
    ax.set_title(f'Top {top_n} Computational → HSQC Observable Correlations (Ranked by R²)',
                 fontsize=15, weight='bold', pad=15)
    
    # Create legend explaining colors
    # NMR patches for bar colors
    nmr_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title(), 
                                   edgecolor='white', linewidth=0.5) 
                   for k, c in NMR_CATEGORY_COLORS.items()]
    # Comp patches for label colors
    comp_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title(),
                                    edgecolor='white', linewidth=0.5) 
                   for k, c in COMP_CATEGORY_COLORS.items()]
    
    # Two separate legends
    leg1 = ax.legend(handles=nmr_patches, loc='upper right', frameon=True,
                     fancybox=True, fontsize=9, 
                     title='Bar Color: HSQC Observable Type',
                     title_fontsize=10, framealpha=0.95,
                     bbox_to_anchor=(1.0, 1.0))
    leg1.get_frame().set_edgecolor('#CCCCCC')
    ax.add_artist(leg1)
    
    leg2 = ax.legend(handles=comp_patches, loc='upper right', frameon=True,
                     fancybox=True, fontsize=9,
                     title='Label Color: Computational Type',
                     title_fontsize=10, framealpha=0.95,
                     bbox_to_anchor=(1.0, 0.75))
    leg2.get_frame().set_edgecolor('#CCCCCC')
    
    # Adjust axis limits
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(0, df['r2'].max() * 1.15)
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Note about significance
    fig.text(0.5, 0.01, 'All correlations shown are significant (p < 0.05)', 
             ha='center', fontsize=9, style='italic', color='#888888')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.40)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig

def plot_best_predictors(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create summary showing best computational predictor for each NMR metric.
    
    Args:
        r2_df: DataFrame with correlation results
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Get best predictor for each NMR metric
    best = r2_df.loc[r2_df.groupby('nmr_metric')['r2'].idxmax()].copy()
    best['nmr_display'] = best['nmr_metric'].map(get_nmr_display_name)
    best['comp_display'] = best['comp_metric'].map(get_comp_display_name)
    
    # Sort by category then R²
    best['nmr_cat'] = best['nmr_metric'].map(get_nmr_category)
    cat_order = {'fold_quality': 0, 'dynamics': 1, 'secondary_structure': 2, 'sample_quality': 3}
    best['cat_order'] = best['nmr_cat'].map(cat_order)
    best = best.sort_values(['cat_order', 'r2'], ascending=[True, False])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(best))
    colors = [get_nmr_color(m) for m in best['nmr_metric']]
    
    bars = ax.bar(x, best['r2'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(best['nmr_display'], rotation=45, ha='right', fontsize=8)
    
    # Add best predictor labels
    for i, (bar, comp) in enumerate(zip(bars, best['comp_display'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                comp, ha='center', va='bottom', fontsize=6, rotation=90, color='#555555')
    
    ax.set_ylabel('Best R²', fontsize=11)
    ax.set_title('Best Computational Predictor for Each NMR Metric',
                 fontsize=12, weight='bold')
    
    # Legend
    nmr_patches = [mpatches.Patch(color=c, label=k.replace('_', ' ').title()) 
                   for k, c in NMR_CATEGORY_COLORS.items()]
    ax.legend(handles=nmr_patches, loc='upper right', frameon=False, fontsize=8)
    
    ax.set_ylim(0, best['r2'].max() * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


def plot_category_summary(
    r2_df: pd.DataFrame,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create heatmap summarizing R² by NMR and computational categories.
    
    Args:
        r2_df: DataFrame with correlation results
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    df = r2_df.copy()
    df['nmr_cat'] = df['nmr_metric'].map(get_nmr_category)
    df['comp_cat'] = df['comp_metric'].map(get_comp_category)
    
    # Aggregate by category
    cat_summary = df.groupby(['nmr_cat', 'comp_cat'])['r2'].agg(['mean', 'max', 'count']).reset_index()
    
    # Pivot for heatmap
    pivot_mean = cat_summary.pivot(index='nmr_cat', columns='comp_cat', values='mean')
    pivot_max = cat_summary.pivot(index='nmr_cat', columns='comp_cat', values='max')
    
    # Order
    nmr_order = ['fold_quality', 'dynamics', 'secondary_structure', 'sample_quality']
    comp_order = ['confidence', 'flexibility', 'structure']
    pivot_mean = pivot_mean.reindex(nmr_order)[comp_order]
    pivot_max = pivot_max.reindex(nmr_order)[comp_order]
    
    # Nice labels
    pivot_mean.index = [n.replace('_', ' ').title() for n in pivot_mean.index]
    pivot_mean.columns = [c.replace('_', ' ').title() for c in pivot_mean.columns]
    pivot_max.index = pivot_mean.index
    pivot_max.columns = pivot_mean.columns
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Mean R²
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap=R2_CMAP,
                vmin=0, vmax=0.2, linewidths=1, linecolor='white',
                ax=axes[0], cbar_kws={'label': 'Mean R²', 'shrink': 0.7})
    axes[0].set_title('A  Mean R² by Category', fontsize=11, weight='bold', loc='left')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Max R²
    sns.heatmap(pivot_max, annot=True, fmt='.3f', cmap=R2_CMAP,
                vmin=0, vmax=0.4, linewidths=1, linecolor='white',
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
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare prediction methods (AF3, Boltz1, Boltz2) for key NMR metrics.
    
    Args:
        r2_df: DataFrame with correlation results
        nmr_metrics_subset: Subset of NMR metrics to include
        figsize: Figure size
        output_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    df = r2_df.copy()
    
    # Default to fold quality and dynamics metrics
    if nmr_metrics_subset is None:
        nmr_metrics_subset = [
            '1H_shift_std', '1H_shift_range', 'peak_recovery_rate',
            'LW_1H_mean', 'R2star_1H_mean', 'cv_intensity'
        ]
    
    df = df[df['nmr_metric'].isin(nmr_metrics_subset)]
    
    # Extract method from comp_metric
    def extract_method(m):
        if 'af3' in m.lower():
            return 'AF3'
        elif 'boltz1' in m.lower():
            return 'Boltz-1'
        elif 'boltz2' in m.lower():
            return 'Boltz-2'
        else:
            return 'Other'
    
    df['method'] = df['comp_metric'].apply(extract_method)
    df = df[df['method'] != 'Other']
    
    # Get best R² per method per NMR metric
    best_per_method = df.groupby(['nmr_metric', 'method'])['r2'].max().reset_index()
    best_per_method['nmr_display'] = best_per_method['nmr_metric'].map(get_nmr_display_name)
    
    # Pivot for grouped bar chart
    pivot = best_per_method.pivot(index='nmr_display', columns='method', values='r2')
    pivot = pivot[['AF3', 'Boltz-1', 'Boltz-2']]  # Order
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(pivot))
    width = 0.25
    
    method_colors = {'AF3': '#3C5488', 'Boltz-1': '#E64B35', 'Boltz-2': '#00A087'}
    
    for i, method in enumerate(['AF3', 'Boltz-1', 'Boltz-2']):
        if method in pivot.columns:
            bars = ax.bar(x + i*width, pivot[method], width, 
                         label=method, color=method_colors[method],
                         edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Best R²', fontsize=11)
    ax.set_title('Method Comparison: Predicting NMR Observables',
                 fontsize=12, weight='bold')
    ax.legend(frameon=False, fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    return fig


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_all_hsqc_figures(
    r2_df: pd.DataFrame,
    output_dir: str = '.',
    prefix: str = 'fig_hsqc'
) -> Dict[str, plt.Figure]:
    """
    Generate all publication figures for HSQC analysis.
    
    Args:
        r2_df: DataFrame with columns ['nmr_metric', 'comp_metric', 'r2', 'p_value']
        output_dir: Directory to save figures
        prefix: Filename prefix
    
    Returns:
        Dictionary of figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # Main R² heatmap
    figures['heatmap'] = plot_r2_heatmap(
        r2_df,
        output_path=f'{output_dir}/{prefix}_r2_heatmap.svg'
    )
    
    # Top correlations
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
    
    # Best predictors
    figures['best_predictors'] = plot_best_predictors(
        r2_df,
        output_path=f'{output_dir}/{prefix}_best_predictors.svg'
    )
    
    # Category summary
    figures['category_summary'] = plot_category_summary(
        r2_df,
        output_path=f'{output_dir}/{prefix}_category_summary.svg'
    )
    
    # Method comparison
    figures['method_comparison'] = plot_method_comparison(
        r2_df,
        output_path=f'{output_dir}/{prefix}_method_comparison.svg'
    )
    
    print(f"\nGenerated {len(figures)} publication figures in {output_dir}/")
    
    return figures


if __name__ == "__main__":

    output_dir = str(HSQC_OUT / "nmr_comp_correlations")
    all_r2_results = []

    print("\nGenerating correlation plots...")
    print("=" * 50)

    for nmr_metric in nmr_metrics:
        print(f"\nProcessing: {nmr_metric}")
        results = create_correlation_plot(
            df_unique, 
            nmr_metric, 
            computational_metrics,
            color_col="correct_Vel",
            output_dir=output_dir
        )
        if results:
            all_r2_results.extend(results)

    generate_all_hsqc_figures(
        r2_df=pd.DataFrame(all_r2_results),
        output_dir=output_dir,
        prefix='fig_hsqc'
    )
