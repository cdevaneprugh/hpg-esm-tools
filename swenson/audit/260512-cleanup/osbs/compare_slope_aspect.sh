#!/bin/bash
#SBATCH --job-name=slope_aspect_cmp
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/slope_aspect_cmp_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/slope_aspect_cmp_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Compare pipeline slope/aspect (pgrid Horn 1981) vs NEON DP3.30025.001
# 90 tiles, ~1M pixels each, pgrid slope_aspect only (no flow routing)

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export TILE_RANGES="${TILE_RANGES:-R4C5-R12C14}"

echo "=== Slope/Aspect Comparison: pgrid vs NEON ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Tiles: ${TILE_RANGES}"
echo ""

python scripts/osbs/compare_slope_aspect.py

echo ""
echo "Completed: $(date)"
