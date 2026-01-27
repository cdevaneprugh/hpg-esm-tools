#!/bin/bash
#SBATCH --job-name=swenson-stage3
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/stage3_%j.log

# Stage 3: Hillslope Parameter Computation
# Computes the 6 geomorphic parameters for each hillslope element

set -e

echo "=========================================="
echo "Stage 3: Hillslope Parameter Computation"
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

# Run the stage 3 script
echo ""
echo "Running stage3_hillslope_params.py..."
echo ""

python "$SCRIPT_DIR/stage3_hillslope_params.py"

echo ""
echo "=========================================="
echo "Stage 3 Complete"
echo "Finished: $(date)"
echo "=========================================="
