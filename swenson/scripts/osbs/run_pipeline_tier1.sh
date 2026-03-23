#!/bin/bash
#SBATCH --job-name=osbs_tier1
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier1_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/tier1_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Tier 1: Single tile R6C10 (smoke test)

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"
export TILE_RANGES="R6C10"
export OUTPUT_DESCRIPTOR="tier1_r6c10_1x8"

echo "=== OSBS Hillslope Pipeline — Tier 1 (R6C10, 1x8 log-spaced) ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/run_pipeline.py

echo ""
echo "Completed: $(date)"
