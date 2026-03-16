#!/bin/bash
#SBATCH --job-name=osbs_t1_fb
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier1_flooded_fb_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier1_flooded_fb_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Tier 1 — flooded DEM fallback test (R6C10 single tile)
# Compare results to 2026-02-26_tier1_r6c10 baseline

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export TILE_RANGES="R6C10"
export OUTPUT_DESCRIPTOR="tier1_r6c10_flooded_fb"

echo "=== OSBS Hillslope Pipeline — Tier 1 Flooded Fallback (R6C10) ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/run_pipeline.py

echo ""
echo "Completed: $(date)"
