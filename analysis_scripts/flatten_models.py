#!/usr/bin/env python3
"""
Flattens model.cif files from seed folders into a simplified structure.

AF3 Input structure:
    batch_X/uncoditional_Y_Z_seq_N_model_M/seed-S_sample-0/model.cif

Boltz2 Input structure:
    batch_X/boltz_results_batch_X/predictions/model_folder/model_folder_model_N.cif

Output structure:
    output_folder/uncoditional_Y_Z_seq_N_model_M/model_0.cif
    output_folder/uncoditional_Y_Z_seq_N_model_M/model_1.cif
    ...
"""

import os
import shutil
import re
import argparse
import json
import csv
from pathlib import Path

def get_seed_number(seed_folder_name: str) -> int:
    """Extract the seed number from folder name like 'seed-1_sample-0'."""
    match = re.match(r'seed-(\d+)_sample-\d+', seed_folder_name)
    if match:
        return int(match.group(1))
    return -1


def get_boltz_model_number(filename: str) -> int:
    """Extract the model number from Boltz2 filename like 'model_folder_model_5.cif'."""
    match = re.search(r'_model_(\d+)\.cif$', filename)
    if match:
        return int(match.group(1))
    return -1

def flatten_models(input_dir: str, output_dir: str, copy: bool = False):
    """
    Flatten model.cif files from seed folders into output directory (AF3 format).
    
    Args:
        input_dir: Base directory containing batch folders
        output_dir: Output directory for flattened structure
        copy: If True, copy files; if False, move files (default)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all batch folders
    batch_folders = sorted([d for d in input_path.iterdir() 
                           if d.is_dir() and d.name.startswith('batch_')])
    
    if not batch_folders:
        print(f"No batch folders found in {input_dir}")
        return
    
    total_files = 0
    csv_rows = []  # Collect data for CSV
    
    for batch_folder in batch_folders:
        print(f"\nProcessing {batch_folder.name}...")
        
        # Find all model folders (uncoditional_*_model_*)
        model_folders = [d for d in batch_folder.iterdir() 
                        if d.is_dir() and d.name.startswith('uncoditional_')]
        
        # also find folders that start with 'n_100'
        model_folders += [d for d in batch_folder.iterdir() 
                          if d.is_dir() and d.name.startswith('n_100')]
        for model_folder in model_folders:
            model_name = model_folder.name
            
            # Find all seed folders
            seed_folders = [d for d in model_folder.iterdir() 
                          if d.is_dir() and d.name.startswith('seed-')]
            
            if not seed_folders:
                continue
            
            # Sort by seed number
            seed_folders.sort(key=lambda x: get_seed_number(x.name))
            
            # Create output directory for this model
            model_output_dir = output_path / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each seed folder
            for idx, seed_folder in enumerate(seed_folders):
                model_cif = seed_folder / 'model.cif'
                summary_json = seed_folder / 'summary_confidences.json'
                
                if model_cif.exists():
                    output_file = model_output_dir / f'model_{idx}.cif'
                    
                    if copy:
                        shutil.copy2(model_cif, output_file)
                    else: 
                        shutil.move(str(model_cif), str(output_file))
                    
                    total_files += 1
                    
                    # Copy/move all other files in the seed folder
                    for other_file in seed_folder.iterdir():
                        if other_file.is_file() and other_file.name != 'model.cif':
                            # Rename files to include model index prefix
                            output_other = model_output_dir / f'model_{idx}_{other_file.name}'
                            if copy:
                                shutil.copy2(other_file, output_other)
                            else:
                                shutil.move(str(other_file), str(output_other))
                    
                    # Extract ptm from summary_confidences.json
                    ptm = None
                    if summary_json.exists():
                        try:
                            with open(summary_json, 'r') as f:
                                confidences = json.load(f)
                                ptm = confidences.get('ptm')
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"  Warning: Could not read {summary_json}: {e}")
                    
                    csv_rows.append({
                        'folder_name': model_name,
                        'model_index': idx,
                        'ptm': ptm
                    })
                else:
                    print(f"  Warning: {model_cif} not found")
            
            print(f"  {model_name}: {len(seed_folders)} models -> {model_output_dir}")
    
    # Write CSV file
    csv_path = output_path / 'model_confidences.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folder_name', 'model_index', 'ptm'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to: {csv_path}")
    
    print(f"Total files processed: {total_files}")


def flatten_models_boltz(input_dir: str, output_dir: str, copy: bool = False):
    """
    Flatten model.cif files from Boltz2 predictions folder into output directory.
    
    Boltz2 structure:
        batch_X/boltz_results_batch_X/predictions/model_folder/model_folder_model_N.cif
    
    Args:
        input_dir: Base directory containing batch folders
        output_dir: Output directory for flattened structure
        copy: If True, copy files; if False, move files (default)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all batch folders
    batch_folders = sorted([d for d in input_path.iterdir() 
                           if d.is_dir() and d.name.startswith('batch_')])
    
    if not batch_folders:
        print(f"No batch folders found in {input_dir}")
        return
    
    total_files = 0
    csv_rows = []  # Collect data for CSV
    
    for batch_folder in batch_folders:
        print(f"\nProcessing {batch_folder.name}...")
        
        # Find boltz_results folder (e.g., boltz_results_batch_0)
        boltz_results_folders = [d for d in batch_folder.iterdir() 
                                 if d.is_dir() and d.name.startswith('boltz_results_')]
        
        if not boltz_results_folders:
            print(f"  No boltz_results folder found in {batch_folder.name}")
            continue
        
        for boltz_results in boltz_results_folders:
            predictions_dir = boltz_results / 'predictions'
            
            if not predictions_dir.exists():
                print(f"  No predictions folder found in {boltz_results.name}")
                continue
            
            # Find all model folders in predictions
            model_folders = [d for d in predictions_dir.iterdir() 
                            if d.is_dir() and (d.name.startswith('uncoditional_') or d.name.startswith('n_100'))]
            
            for model_folder in model_folders:
                model_name = model_folder.name
                
                # Find all CIF files (named like model_folder_model_N.cif)
                cif_files = [f for f in model_folder.iterdir() 
                            if f.is_file() and f.suffix == '.cif' and f.name.startswith(model_name)]
                
                if not cif_files:
                    continue
                
                # Sort by model number
                cif_files.sort(key=lambda x: get_boltz_model_number(x.name))
                
                # Create output directory for this model
                model_output_dir = output_path / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each CIF file
                for cif_file in cif_files:
                    model_num = get_boltz_model_number(cif_file.name)
                    if model_num == -1:
                        continue
                    
                    output_file = model_output_dir / f'model_{model_num}.cif'
                    
                    if copy:
                        shutil.copy2(cif_file, output_file)
                    else:
                        shutil.move(str(cif_file), str(output_file))
                    
                    total_files += 1
                    
                    # Extract ptm from confidence JSON file
                    ptm = None
                    confidence_json = model_folder / f'confidence_{model_name}_model_{model_num}.json'
                    if confidence_json.exists():
                        try:
                            with open(confidence_json, 'r') as f:
                                confidences = json.load(f)
                                ptm = confidences.get('ptm')
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"  Warning: Could not read {confidence_json}: {e}")
                    
                    csv_rows.append({
                        'folder_name': model_name,
                        'model_index': model_num,
                        'ptm': ptm
                    })
                
                # Copy/move all other files in the model folder that aren't CIF files
                for other_file in model_folder.iterdir():
                    if other_file.is_file() and other_file.suffix != '.cif':
                        output_other = model_output_dir / other_file.name
                        if copy:
                            shutil.copy2(other_file, output_other)
                        else:
                            shutil.move(str(other_file), str(output_other))
                
                print(f"  {model_name}: {len(cif_files)} models -> {model_output_dir}")
    
    # Write CSV file
    csv_path = output_path / 'model_confidences.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folder_name', 'model_index', 'ptm'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to: {csv_path}")
    
    print(f"Total files processed: {total_files}")


def flatten_single_batch(batch_dir: str, output_dir: str, copy: bool = False):
    """
    Flatten model.cif files from a single batch folder.
    
    Args:
        batch_dir: Path to a specific batch folder
        output_dir: Output directory for flattened structure
        copy: If True, copy files; if False, move files (default)
    """
    batch_path = Path(batch_dir)
    output_path = Path(output_dir)
    
    if not batch_path.exists():
        print(f"Batch folder not found: {batch_dir}")
        return
    
    total_files = 0
    csv_rows = []  # Collect data for CSV
    
    # Find all model folders (uncoditional_*_model_*)
    model_folders = [d for d in batch_path.iterdir() 
                    if d.is_dir() and d.name.startswith('uncoditional_')]
    
    for model_folder in model_folders:
        model_name = model_folder.name
        
        # Find all seed folders
        seed_folders = [d for d in model_folder.iterdir() 
                      if d.is_dir() and d.name.startswith('seed-')]
        
        if not seed_folders:
            continue
        
        # Sort by seed number
        seed_folders.sort(key=lambda x: get_seed_number(x.name))
        
        # Create output directory for this model
        model_output_dir = output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each seed folder
        for idx, seed_folder in enumerate(seed_folders):
            model_cif = seed_folder / 'model.cif'
            summary_json = seed_folder / 'summary_confidences.json'
            
            if model_cif.exists():
                output_file = model_output_dir / f'model_{idx}.cif'
                
                if copy:
                    shutil.copy2(model_cif, output_file)
                else:
                    shutil.move(str(model_cif), str(output_file))
                
                total_files += 1
                
                # Copy/move all other files in the seed folder
                for other_file in seed_folder.iterdir():
                    if other_file.is_file() and other_file.name != 'model.cif':
                        # Rename files to include model index prefix
                        output_other = model_output_dir / f'model_{idx}_{other_file.name}'
                        if copy:
                            shutil.copy2(other_file, output_other)
                        else:
                            shutil.move(str(other_file), str(output_other))
                
                # Extract ptm from summary_confidences.json
                ptm = None
                if summary_json.exists():
                    try:
                        with open(summary_json, 'r') as f:
                            confidences = json.load(f)
                            ptm = confidences.get('ptm')
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"  Warning: Could not read {summary_json}: {e}")
                
                csv_rows.append({
                    'folder_name': model_name,
                    'model_index': idx,
                    'ptm': ptm
                })
            else:
                print(f"  Warning: {model_cif} not found")
        
        print(f"  {model_name}: {len(seed_folders)} models -> {model_output_dir}")
    
    # Write CSV file
    csv_path = output_path / 'model_confidences.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folder_name', 'model_index', 'ptm'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to: {csv_path}")
    
    print(f"Total files processed: {total_files}")


def main():
    parser = argparse.ArgumentParser(
        description='Flatten model.cif files from seed folders into a simplified structure.'
    )
    parser.add_argument(
        'input_dir',
        help='Input directory containing batch folders (or a single batch folder with --single-batch)'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for flattened models'
    )
    parser.add_argument(
        '--single-batch',
        action='store_true',
        help='Process a single batch folder instead of all batches'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    parser.add_argument(
        '--boltz',
        action='store_true',
        help='Use Boltz2 format instead of AF3 format'
    )
    
    args = parser.parse_args()
    
    copy = args.copy
    
    if args.boltz:
        # Boltz2 format
        flatten_models_boltz(args.input_dir, args.output_dir, copy=copy)
    elif args.single_batch:
        flatten_single_batch(args.input_dir, args.output_dir, copy=copy)
    else:
        flatten_models(args.input_dir, args.output_dir, copy=copy)


if __name__ == '__main__':
    main()
