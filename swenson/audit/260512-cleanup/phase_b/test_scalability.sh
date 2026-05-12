#!/bin/bash
#SBATCH --job-name=phase-b-scale
#SBATCH --output=output/osbs/phase_b/scalability_%j.log
#SBATCH --partition=hpg-default
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

# Phase B — Scalability Test
#
# Tests whether pysheds resolve_flats() can complete on the full
# contiguous interior region (90M pixels at 1m) at various memory
# allocations.
#
# Override memory at submit time:
#   sbatch --mem=64gb  scripts/phase_b/test_scalability.sh
#   sbatch --mem=128gb scripts/phase_b/test_scalability.sh
#   sbatch --mem=256gb --qos=gerber-b scripts/phase_b/test_scalability.sh

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"

echo "=== Phase B Scalability Test ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Node: $(hostname)"
echo "pysheds fork: $(cd "$PYSHEDS_FORK" && git branch --show-current) @ $(cd "$PYSHEDS_FORK" && git rev-parse --short HEAD)"
echo ""

python scripts/phase_b/test_scalability.py
