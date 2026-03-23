#!/bin/bash
#SBATCH --job-name=osbs_smoke
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/smoke_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Smoke test: single tile R6C10 (lake, swamp, upland)
# Before running: change MOSAIC_PATH in run_pipeline.py to point at the R6C10 tile:
#   MOSAIC_PATH = DATA_DIR / "neon" / "dtm" / "NEON_D03_OSBS_DP3_404000_3286000_DTM.tif"

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export OUTPUT_DESCRIPTOR="smoke_r6c10"

echo "=== OSBS Hillslope Pipeline — Smoke Test (R6C10, single tile) ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/run_pipeline.py

echo ""
echo "Completed: $(date)"
