#!/bin/bash
#SBATCH --job-name=stage6_width
#SBATCH --output=../output/stage6/stage6_%j.out
#SBATCH --error=../output/stage6/stage6_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --qos=gerber
#SBATCH --account=gerber

# Stage 6: Width Bug Diagnosis and Fix
# Analyzes the width calculation bug and creates corrected parameters
# Runtime: ~1-2 minutes

set -e

echo "Starting Stage 6: Width Bug Fix"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Create output directory
mkdir -p ../output/stage6

# Load conda environment
module load conda
conda activate ctsm

# Run Stage 6 (full analysis and fix)
python stage6_width_fix.py

echo ""
echo "Stage 6 complete"
echo "Time: $(date)"
