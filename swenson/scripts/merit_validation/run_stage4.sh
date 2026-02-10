#!/bin/bash
#SBATCH --job-name=swenson-stage4
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#uBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/merit_validation/stage4/stage4_%j.log

# Stage 4: Compare to Swenson's Published Data
# Compares our results to the published global hillslope dataset

set -e

echo "=========================================="
echo "Stage 4: Compare to Published Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Load conda environment
module load conda
conda activate ctsm

# Script directory (absolute path)
SCRIPT_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/scripts/merit_validation"
cd "$SCRIPT_DIR"

# Run the stage 4 script
echo ""
echo "Running stage4_comparison.py..."
echo ""

python "$SCRIPT_DIR/stage4_comparison.py"

echo ""
echo "=========================================="
echo "Stage 4 Complete"
echo "Finished: $(date)"
echo "=========================================="
