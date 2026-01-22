#!/bin/bash
#SBATCH --job-name=swenson-stage2
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/stage2_%j.log

# Stage 2: Spatial Scale Analysis using FFT
# Determines characteristic length scale from DEM Laplacian

set -e

echo "=========================================="
echo "Stage 2: Spatial Scale Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Load conda environment
module load conda
conda activate ctsm

# Add pysheds fork to path (for potential future use)
export PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"
export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH}"

# Script directory (absolute path)
SCRIPT_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/scripts"
cd "$SCRIPT_DIR"

# Run the stage 2 script
echo ""
echo "Running stage2_spatial_scale.py..."
echo ""

python "$SCRIPT_DIR/stage2_spatial_scale.py"

echo ""
echo "=========================================="
echo "Stage 2 Complete"
echo "Finished: $(date)"
echo "=========================================="
