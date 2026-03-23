#!/bin/bash
#SBATCH --job-name=osbs_tier3
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier3_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier3_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Tier 3: Full contiguous region (R4-R12, C5-C14)

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export TILE_RANGES="R4C5-R12C14"
export OUTPUT_DESCRIPTOR="tier3_contiguous_1x8"

echo "=== OSBS Hillslope Pipeline — Tier 3 (Full contiguous, 1x8 log-spaced) ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/run_pipeline.py

echo ""
echo "Completed: $(date)"
