# %%
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import entropy
from pathlib import Path
import re

# %%
# Configuration — data root is set in config.py (via NMR_PAPER_DATA env var)
from config import METRICS_PATH

# %%
# Helper functions
def compute_cluster_entropy(cluster_ids):
    """
    Compute entropy of cluster assignment.
    
    Args:
        cluster_ids: Series or array of cluster IDs
    
    Returns:
        Entropy value (higher = more diverse/uniform distribution)
    """
    value_counts = pd.Series(cluster_ids).value_counts()
    probabilities = value_counts / len(cluster_ids)
    return entropy(probabilities, base=2)


def compute_folder_cluster_stats(df):
    """
    Compute cluster statistics for each folder.
    
    Args:
        df: DataFrame with columns [folder_name, model_id, cluster_id]
    
    Returns:
        DataFrame with columns [folder_name, num_clusters, cluster_entropy]
    """
    return df.groupby('folder_name').agg(
        num_clusters=('cluster_id', 'nunique'),
        cluster_entropy=('cluster_id', compute_cluster_entropy)
    ).reset_index()


def load_and_tag(filepath, source_name, folder_col='folder_name'):
    """Load CSV and add source tag."""
    df = pd.read_csv(filepath)
    df['source'] = source_name
    # Normalize folder column name if needed
    if folder_col != 'folder_name' and folder_col in df.columns:
        df = df.rename(columns={folder_col: 'folder_name'})
    # Normalize seq -> SEQ in folder column if present
    if 'folder_name' in df.columns:
        df['folder_name'] = df['folder_name'].str.replace('seq', 'SEQ', regex=False)
    return df


def extract_source_from_filename(filename: str) -> str:
    """
    Extract source name from filename like 'af3_recycle_0_plddt.csv' -> 'af3_recycle_0'
    """
    # Remove _plddt.csv or _rmsf.csv suffix
    name = re.sub(r'_(plddt|rmsf)\.csv$', '', filename)
    return name


def auto_load_metrics(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Automatically load all pLDDT and RMSF CSV files from the metrics directory.
    
    Args:
        metrics_dir: Path to directory containing metric CSV files
        
    Returns:
        Tuple of (plddt_df, rmsf_df) DataFrames with all sources combined
    """
    plddt_dfs = []
    rmsf_dfs = []
    
    for csv_file in sorted(metrics_dir.glob("*.csv")):
        source = extract_source_from_filename(csv_file.name)
        
        if csv_file.name.endswith('_plddt.csv'):
            print(f"Loading pLDDT: {csv_file.name} -> source={source}")
            df = load_and_tag(csv_file, source)
            plddt_dfs.append(df)
        elif csv_file.name.endswith('_rmsf.csv'):
            print(f"Loading RMSF: {csv_file.name} -> source={source}")
            df = load_and_tag(csv_file, source)
            rmsf_dfs.append(df)
    
    plddt_all = pd.concat(plddt_dfs, ignore_index=True) if plddt_dfs else pd.DataFrame()
    rmsf_all = pd.concat(rmsf_dfs, ignore_index=True) if rmsf_dfs else pd.DataFrame()
    
    return plddt_all, rmsf_all


# %%
# Auto-load all metrics from the metrics directory
print(f"Loading metrics from: {METRICS_PATH}")
plddt_raw, rmsf_raw = auto_load_metrics(METRICS_PATH)

print(f"\npLDDT sources: {plddt_raw['source'].unique() if not plddt_raw.empty else 'None'}")
print(f"RMSF sources: {rmsf_raw['source'].unique() if not rmsf_raw.empty else 'None'}")

# %%
# Aggregate RMSF per folder (mean RMSF across residues)
def aggregate_rmsf(df):
    if df.empty:
        return pd.DataFrame()
    return df.groupby(['folder_name', 'source']).agg(
        mean_rmsf=('mean_rmsf', 'mean'),
        std_rmsf=('mean_rmsf', 'std'),
        max_rmsf=('mean_rmsf', 'max')
    ).reset_index()

# %%
# Aggregate pLDDT per folder (mean pLDDT across residues/models)
def aggregate_plddt(df):
    if df.empty:
        return pd.DataFrame()
    return df.groupby(['folder_name', 'source']).agg(
        mean_plddt=('plddt', 'mean'),
        std_plddt=('plddt', 'std'),
        min_plddt=('plddt', 'min'),
        max_plddt=('plddt', 'max')
    ).reset_index()

rmsf_agg = aggregate_rmsf(rmsf_raw)
plddt_agg = aggregate_plddt(plddt_raw)


# Merge all metrics into one large DataFrame
metrics_df = rmsf_agg.merge(
    plddt_agg, on=['folder_name', 'source'], how='outer'
)

# Reorder columns for clarity
column_order = [
    'folder_name', 'source',
    'mean_rmsf', 'std_rmsf', 'max_rmsf',
    'mean_plddt', 'std_plddt', 'min_plddt', 'max_plddt',
]
metrics_df = metrics_df[[c for c in column_order if c in metrics_df.columns]]

print(f"\nCombined metrics DataFrame shape: {metrics_df.shape}")
print(f"Sources: {metrics_df['source'].unique()}")
print(f"Folders: {metrics_df['folder_name'].nunique()}")
metrics_df.head(10)

# %%
# Optional: Pivot to wide format (one row per folder, columns per source)
if not metrics_df.empty:
    metrics_wide = metrics_df.pivot(index='folder_name', columns='source')
    metrics_wide.columns = ['_'.join(col).strip() for col in metrics_wide.columns.values]
    metrics_wide = metrics_wide.reset_index()

    print(f"Wide format shape: {metrics_wide.shape}")
    metrics_wide.head()

# %%
metrics_wide.to_csv(f"{METRICS_PATH}/combined_dynamics_metrics.csv", index=False)


