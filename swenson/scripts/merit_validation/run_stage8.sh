#!/bin/bash
#SBATCH --job-name=swenson-stage8
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/stage8_%j.log

# Stage 8: Gradient Calculation Comparison
# Validates hypothesis that gradient differences cause E/S classification issue

set -e

echo "=========================================="
echo "Stage 8: Gradient Calculation Comparison"
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
SCRIPT_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/scripts"
cd "$SCRIPT_DIR"

# Ensure output directory exists
mkdir -p /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/stage8

# Run the stage 8 script
echo ""
echo "Running stage8_gradient_comparison.py..."
echo ""

python "$SCRIPT_DIR/stage8_gradient_comparison.py"

echo ""
echo "=========================================="
echo "Stage 8 Complete"
echo "Finished: $(date)"
echo "=========================================="
