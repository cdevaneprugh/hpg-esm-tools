#!/bin/bash
#SBATCH --job-name=stage5_units
#SBATCH --output=../output/stage5/stage5_%j.out
#SBATCH --error=../output/stage5/stage5_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --qos=gerber
#SBATCH --account=gerber

# Stage 5: Unit Conversion Analysis
# Quick analysis of existing Stage 3/4 outputs
# Runtime: ~1 minute

set -e

echo "Starting Stage 5: Unit Conversion Analysis"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Create output directory
mkdir -p ../output/stage5

# Load conda environment
module load conda
conda activate ctsm

# Run Stage 5
python stage5_unit_fix.py

echo ""
echo "Stage 5 complete"
echo "Time: $(date)"
