#!/bin/bash
#SBATCH --job-name=swenson-stage1
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage1/stage1_%j.log

# Stage 1: pgrid Validation on MERIT DEM
# Validates pysheds fork processing on full MERIT tile

set -e

echo "=========================================="
echo "Stage 1: pgrid Validation"
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

# Run the stage 1 script
echo ""
echo "Running stage1_pgrid_validation.py..."
echo ""

python "$SCRIPT_DIR/stage1_pgrid_validation.py"

echo ""
echo "=========================================="
echo "Stage 1 Complete"
echo "Finished: $(date)"
echo "=========================================="
