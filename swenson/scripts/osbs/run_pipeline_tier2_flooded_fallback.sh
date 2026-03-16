#!/bin/bash
#SBATCH --job-name=osbs_t2_fb
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier2_flooded_fb_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier2_flooded_fb_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Tier 2 — flooded DEM fallback test (5x5 block)
# Compare results to 2026-02-26_tier2_5x5 baseline

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export TILE_RANGES="R6C7-R10C11"
export OUTPUT_DESCRIPTOR="tier2_5x5_flooded_fb"

echo "=== OSBS Hillslope Pipeline — Tier 2 Flooded Fallback (5x5) ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/run_pipeline.py

echo ""
echo "Completed: $(date)"
