#!/bin/bash
#SBATCH --job-name=osbs_production
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/production_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/production_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Production run: R4C5-R12C14 (90 tiles, 9x10 km)
# Requires: data/mosaics/OSBS_production.tif
# MOSAIC_PATH in run_pipeline.py must point to the production mosaic (default).

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export OUTPUT_DESCRIPTOR="production"

echo "=== OSBS Hillslope Pipeline — Production (R4C5-R12C14, 90 tiles) ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/run_pipeline.py

echo ""
echo "Completed: $(date)"
