#!/bin/bash
#SBATCH --job-name=swenson-stage9
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage9/stage9_%j.log

# Stage 9: Accumulation Threshold Sensitivity Analysis
# Tests different accumulation thresholds to determine if area correlation
# can be improved beyond current 0.82.

set -e

echo "=========================================="
echo "Stage 9: Threshold Sensitivity Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Load conda environment
module load conda
conda activate ctsm

# Add pysheds fork to path
export PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"
export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH}"

# Verify pysheds is importable
echo ""
echo "Verifying pysheds import..."
python -c "from pysheds.pgrid import Grid; print('pysheds.pgrid import: OK')"

# Script directory (absolute path)
SCRIPT_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/scripts/merit_validation"
cd "$SCRIPT_DIR"

# Run the stage 9 script
echo ""
echo "Running stage9_threshold_sensitivity.py..."
echo "Testing thresholds: 20, 34, 50, 100, 200 cells"
echo ""

python "$SCRIPT_DIR/stage9_threshold_sensitivity.py"

echo ""
echo "=========================================="
echo "Stage 9 Complete"
echo "Finished: $(date)"
echo "=========================================="
