#!/bin/bash
#
# Compute all metrics for model ensembles. Data root is taken from
# $NMR_PAPER_DATA (set via `export NMR_PAPER_DATA=/path/to/data`);
# defaults to ~/Desktop/nmr_revelations_paper/data when unset.
#
# Uses:
#   - bfactor for folders with 'af3' in the name (pLDDT stored in B-factor column)
#   - plddt for folders with 'boltz' in the name (pLDDT stored in NPZ files)
#   - rmsf for all folders
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${NMR_PAPER_DATA:-$HOME/Desktop/nmr_revelations_paper/data}"
ENSEMBLES_DIR="$DATA_ROOT/drylab/model_ensembles"
OUTPUT_DIR="$DATA_ROOT/drylab/ensemble_metrics"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Computing metrics for all model ensembles"
echo "Ensembles dir: $ENSEMBLES_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Process each folder in the ensembles directory
for folder in "$ENSEMBLES_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        echo ""
        echo "----------------------------------------------"
        echo "Processing: $folder_name"
        echo "----------------------------------------------"
        
        # Determine which pLDDT extraction method to use based on folder name
        if [[ "$folder_name" == *"af3"* ]]; then
            echo "  -> Using bfactor (AF3 format)"
            python "$SCRIPT_DIR/compute_metrics.py" bfactor "$folder" \
                -o "$OUTPUT_DIR/${folder_name}_plddt.csv" --stats
        elif [[ "$folder_name" == *"boltz"* ]]; then
            echo "  -> Using plddt (Boltz format)"
            python "$SCRIPT_DIR/compute_metrics.py" plddt "$folder" \
                -o "$OUTPUT_DIR/${folder_name}_plddt.csv" --stats
        else
            echo "  -> Unknown format, skipping pLDDT extraction"
        fi
        
        # Compute RMSF for all folders
        echo "  -> Computing RMSF"
        python "$SCRIPT_DIR/compute_metrics.py" rmsf "$folder" \
            -o "$OUTPUT_DIR/${folder_name}_rmsf.csv" --stats
    fi
done

echo ""
echo "=============================================="
echo "Done! All metrics saved to: $OUTPUT_DIR"
echo "=============================================="
ls -la "$OUTPUT_DIR"
