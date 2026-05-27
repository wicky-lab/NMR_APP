#!/usr/bin/env python3
"""
Unified script to compute metrics from structural ensemble predictions.

Supports:
- B-factors/pLDDT from CIF files (AF3, Boltz2)
- pLDDT from NPZ files (Boltz1, Boltz2)
- RMSF computation from structure ensembles

Usage:
    # Parse B-factors from CIF files (AF3/Boltz2)
    python compute_metrics.py bfactor path/to/folder --output bfactors.csv

    # Parse pLDDT from NPZ files (Boltz1/Boltz2)
    python compute_metrics.py plddt path/to/folder --output plddt.csv

    # Compute RMSF from ensemble
    python compute_metrics.py rmsf path/to/folder --output rmsf.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import biotite.structure.io.pdbx as pdbx
from biotite.structure import get_residues, filter_peptide_backbone, superimpose
from biotite.structure.io import load_structure


def extract_model_id(filename: str) -> int:
    """Extract model ID from filename like 'model_0.cif' or 'plddt_xxx_model_0.npz'."""
    match = re.search(r'model_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def save_to_csv(data: list, output_csv: str, fieldnames: list = None):
    """Save list of dicts to CSV file."""
    if not data:
        print("No data to save")
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nSaved {len(data)} rows to {output_csv}")


# =============================================================================
# B-factor Parsing (from CIF files)
# =============================================================================

def load_bfactors_from_cif(cif_path: str, use_ca_only: bool = False) -> tuple:
    """
    Load B-factors from a CIF file.
    
    Args:
        cif_path: Path to the CIF file
        use_ca_only: If True, use only CA atoms; if False, average all atoms per residue
        
    Returns:
        Tuple of (residue_ids, residue_names, chain_ids, b_factors) arrays
    """

    cif_file = pdbx.CIFFile.read(cif_path)
    structure = pdbx.get_structure(cif_file, model=1)

    # Get B-factors from the CIF file's atom_site category
    block = list(cif_file.values())[0]
    atom_site = block["atom_site"]
    b_factors = atom_site["B_iso_or_equiv"].as_array().astype(float)

    # Get auth_seq_id (author residue IDs) directly from CIF
    residue_ids = atom_site["auth_seq_id"].as_array().astype(int)
    residue_names = np.array(structure.res_name)
    chain_ids = np.array(structure.chain_id)
    atom_names = np.array(structure.atom_name)

    if use_ca_only:
        # Get unique residues and use CA B-factors
        unique_residues = get_residues(structure)
        res_ids_out = []
        res_names_out = []
        chain_ids_out = []
        bfactors_out = []

        for start_idx, stop_idx in zip(unique_residues[0], 
                                        np.append(unique_residues[0][1:], len(structure))):
            res_mask = slice(start_idx, stop_idx)
            res_atom_names = atom_names[res_mask]
            res_bfactors = b_factors[res_mask]
            
            # Try to get CA B-factor, otherwise average all atoms
            ca_mask = res_atom_names == 'CA'
            if np.any(ca_mask):
                bfactor = res_bfactors[ca_mask][0]
            else:
                bfactor = np.mean(res_bfactors)

            res_ids_out.append(residue_ids[start_idx])
            res_names_out.append(residue_names[start_idx])
            chain_ids_out.append(chain_ids[start_idx])
            bfactors_out.append(bfactor)

        return (np.array(res_ids_out), np.array(res_names_out), 
                np.array(chain_ids_out), np.array(bfactors_out))
    else:
        # Average B-factors over all atoms per residue
        unique_res_ids = np.unique(residue_ids)
        
        res_ids_out = []
        res_names_out = []
        chain_ids_out = []
        bfactors_out = []
        
        for res_id in unique_res_ids:
            mask = residue_ids == res_id
            bfactor = np.mean(b_factors[mask])
            
            res_ids_out.append(res_id)
            res_names_out.append(residue_names[mask][0])
            chain_ids_out.append(chain_ids[mask][0])
            bfactors_out.append(bfactor)
        
        return (np.array(res_ids_out), np.array(res_names_out), 
                np.array(chain_ids_out), np.array(bfactors_out))


def process_bfactors(folder_path: str, output_csv: str = None, use_ca_only: bool = False) -> list:
    """
    Process all CIF files in a folder and extract B-factors.
    
    Args:
        folder_path: Path to folder containing CIF files (can have subfolders)
        output_csv: Optional path to output CSV file
        use_ca_only: If True, use CA atoms only; if False, average all atoms
        
    Returns:
        List of dictionaries with B-factor data
    """
    folder = Path(folder_path)
    cif_files = sorted(folder.rglob('**/*.cif'))
    
    if not cif_files:
        print(f"No CIF files found in {folder_path}")
        return []
    
    print(f"Found {len(cif_files)} CIF files")
    
    all_data = []
    
    for cif_file in cif_files:
        print(f"Processing {cif_file.name}...")
        folder_name = cif_file.parent.name
        model_id = extract_model_id(cif_file.name)
        
        try:
            res_ids, res_names, chain_ids, bfactors = load_bfactors_from_cif(
                str(cif_file), use_ca_only=use_ca_only
            )
            
            for chain, res_id, res_name, bf in zip(chain_ids, res_ids, res_names, bfactors):
                all_data.append({
                    'folder_name': folder_name,
                    'model_id': model_id,
                    'chain_id': chain,
                    'residue_id': int(res_id),
                    'residue_name': res_name,
                    'plddt': float(bf)  # B-factor column contains pLDDT in AF3/Boltz
                })
        except Exception as e:
            print(f"  Error processing {cif_file.name}: {e}")
            continue
    
    if output_csv and all_data:
        fieldnames = ['folder_name', 'model_id', 'chain_id', 'residue_id', 'residue_name', 'plddt']
        save_to_csv(all_data, output_csv, fieldnames)
    
    return all_data


# =============================================================================
# pLDDT Parsing (from NPZ files - Boltz1/Boltz2)
# =============================================================================

def load_plddt_from_npz(npz_path: str) -> np.ndarray:
    """Load pLDDT scores from an npz file."""
    data = np.load(npz_path)
    return data['plddt']


def process_plddt_npz(folder_path: str, output_csv: str = None) -> list:
    """
    Process all pLDDT npz files in a folder.
    
    Args:
        folder_path: Path to folder containing npz files
        output_csv: Optional path to output CSV file
        
    Returns:
        List of dictionaries with pLDDT data
    """
    folder = Path(folder_path)
    npz_files = sorted(folder.rglob('**/plddt_*.npz'))
    
    if not npz_files:
        print(f"No pLDDT npz files found in {folder_path}")
        return []
    
    print(f"Found {len(npz_files)} pLDDT npz files")
    
    all_data = []
    
    for npz_file in npz_files:
        try:
            folder_name = npz_file.parent.name
            model_id = extract_model_id(npz_file.name)
            plddt_scores = load_plddt_from_npz(str(npz_file))
            
            for residue_id, plddt in enumerate(plddt_scores, start=1):
                all_data.append({
                    'folder_name': folder_name,
                    'model_id': model_id,
                    'residue_id': residue_id,
                    'plddt': float(plddt)
                })
        except Exception as e:
            print(f"  Error processing {npz_file.name}: {e}")
            continue
    
    if output_csv and all_data:
        fieldnames = ['folder_name', 'model_id', 'residue_id', 'plddt']
        save_to_csv(all_data, output_csv, fieldnames)
    
    return all_data


# =============================================================================
# RMSF Computation
# =============================================================================

def compute_rmsf(folder_path: str, output_csv: str = None) -> list:
    """
    Compute RMSF for all model ensembles in a folder.
    
    Expects folder structure:
        folder_path/model_folder/model_0.cif, model_1.cif, ...
    
    Args:
        folder_path: Path to folder containing model subfolders
        output_csv: Optional path to output CSV file
        
    Returns:
        List of dictionaries with RMSF data per residue
    """

    base_path = Path(folder_path)
    
    # Group CIF files by folder
    folder_files = defaultdict(list)
    for cif_file in base_path.rglob("*.cif"):
        folder_files[cif_file.parent].append(cif_file)
    
    if not folder_files:
        print(f"No CIF files found in {folder_path}")
        return []
    
    print(f"Found {len(folder_files)} folders with CIF files")
    
    all_results = []
    
    for folder, cif_files in sorted(folder_files.items()):
        print(f"\nProcessing folder: {folder.name}")
        print(f"  CIF files: {len(cif_files)}")
        
        # Load all structures
        structures = []
        
        for cif_file in sorted(cif_files):
            try:
                struct = load_structure(cif_file)
                structures.append(struct)
            except Exception as e:
                print(f"  Error loading {cif_file}: {e}")
        
        if len(structures) < 2:
            print(f"  Skipping - need at least 2 structures for RMSF")
            continue
        
        print(f"  Loaded {len(structures)} structures, {len(structures[0])} atoms each")
        
        # Use first structure as reference
        reference = structures[0]
        
        # Align all structures to reference using backbone atoms
        aligned_structures = [reference]
        for struct in structures[1:]:
            try:
                ref_backbone = reference[filter_peptide_backbone(reference)]
                mob_backbone = struct[filter_peptide_backbone(struct)]
                
                _, transformation = superimpose(ref_backbone, mob_backbone)
                aligned_struct = transformation.apply(struct)
                aligned_structures.append(aligned_struct)
            except Exception as e:
                print(f"  Warning: Could not align structure: {e}")
                aligned_structures.append(struct)
        
        # Stack all coordinates
        all_coords = np.array([s.coord for s in aligned_structures])
        
        # Compute RMSF for each atom
        mean_coords = np.mean(all_coords, axis=0)
        deviations = all_coords - mean_coords
        atom_rmsf = np.sqrt(np.mean(np.sum(deviations**2, axis=-1), axis=0))
        
        # Pool to residue level
        res_ids = reference.res_id
        res_names = reference.res_name
        chain_ids = reference.chain_id
        
        # Group atom RMSF by residue (preserving order)
        residue_rmsf = defaultdict(list)
        residue_order = []
        seen = set()
        for i in range(len(atom_rmsf)):
            key = (chain_ids[i], res_ids[i], res_names[i])
            residue_rmsf[key].append(atom_rmsf[i])
            if key not in seen:
                residue_order.append(key)
                seen.add(key)
        
        # Compute pooled RMSF per residue
        for res_idx, (chain, res_id, res_name) in enumerate(residue_order):
            values = residue_rmsf[(chain, res_id, res_name)]
            mean_rmsf = np.mean(values)
            max_rmsf = np.max(values)
            min_rmsf = np.min(values)
            
            result = {
                "folder_name": folder.name,
                "chain_id": chain,
                "residue_id": int(res_id),
                "residue_name": res_name,
                "mean_rmsf": float(mean_rmsf),
                "max_rmsf": float(max_rmsf),
                "min_rmsf": float(min_rmsf),
            }
            all_results.append(result)
    
    if output_csv and all_results:
        fieldnames = ["folder_name", "chain_id", "residue_id", "residue_name", 
                      "mean_rmsf", "max_rmsf", "min_rmsf"]
        save_to_csv(all_results, output_csv, fieldnames)
    
    return all_results


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary_stats(data: list, value_key: str = 'plddt') -> dict:
    """Compute summary statistics from data."""
    if not data:
        return {}
    
    values = [d[value_key] for d in data if d.get(value_key) is not None]
    if not values:
        return {}
    
    return {
        'total_entries': len(data),
        'min': min(values),
        'max': max(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'unique_folders': len(set(d['folder_name'] for d in data)),
        'unique_models': len(set((d['folder_name'], d.get('model_id', 0)) for d in data))
    }


def print_stats(stats: dict, metric_name: str = "pLDDT"):
    """Print summary statistics."""
    if not stats:
        print("No statistics to display")
        return
    
    print(f"\nSummary Statistics ({metric_name}):")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Unique folders: {stats['unique_folders']}")
    print(f"  Unique models: {stats['unique_models']}")
    print(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute metrics from structural ensemble predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Parse pLDDT from CIF files (AF3)
    python compute_metrics.py bfactor path/to/folder -o bfactors.csv

    # Parse pLDDT from NPZ files (Boltz1, Boltz2)
    python compute_metrics.py plddt path/to/folder -o plddt.csv

    # Compute RMSF from structure ensemble
    python compute_metrics.py rmsf path/to/folder -o rmsf.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # B-factor subcommand
    bfactor_parser = subparsers.add_parser(
        'bfactor', 
        help='Extract B-factors (pLDDT) from CIF files'
    )
    bfactor_parser.add_argument('input', help='Path to folder containing CIF files')
    bfactor_parser.add_argument('--output', '-o', help='Output CSV file path')
    bfactor_parser.add_argument(
        '--ca-only', 
        action='store_true',
        help='Use CA atoms only instead of averaging all atoms'
    )
    bfactor_parser.add_argument('--stats', action='store_true', help='Print summary statistics')
    
    # pLDDT subcommand (NPZ files)
    plddt_parser = subparsers.add_parser(
        'plddt', 
        help='Extract pLDDT from NPZ files (Boltz1/Boltz2)'
    )
    plddt_parser.add_argument('input', help='Path to folder containing NPZ files')
    plddt_parser.add_argument('--output', '-o', help='Output CSV file path')
    plddt_parser.add_argument('--stats', action='store_true', help='Print summary statistics')
    
    # RMSF subcommand
    rmsf_parser = subparsers.add_parser(
        'rmsf', 
        help='Compute RMSF from structure ensembles'
    )
    rmsf_parser.add_argument('input', help='Path to folder containing model subfolders')
    rmsf_parser.add_argument('--output', '-o', help='Output CSV file path')
    rmsf_parser.add_argument('--stats', action='store_true', help='Print summary statistics')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1
    
    if args.command == 'bfactor':
        data = process_bfactors(str(input_path), args.output, use_ca_only=args.ca_only)
        if args.stats and data:
            stats = compute_summary_stats(data, 'plddt')
            print_stats(stats, "pLDDT (from B-factor)")
    
    elif args.command == 'plddt':
        data = process_plddt_npz(str(input_path), args.output)
        if args.stats and data:
            stats = compute_summary_stats(data, 'plddt')
            print_stats(stats, "pLDDT")
    
    elif args.command == 'rmsf':
        data = compute_rmsf(str(input_path), args.output)
        if args.stats and data:
            stats = compute_summary_stats(data, 'mean_rmsf')
            print_stats(stats, "RMSF")
    
    return 0


if __name__ == '__main__':
    exit(main())
