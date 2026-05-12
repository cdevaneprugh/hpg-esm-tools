#!/bin/bash
#SBATCH --job-name=phase-b-rescomp
#SBATCH --output=output/osbs/phase_b/resolution_comparison/rescomp_%j.log
#SBATCH --partition=hpg-default
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

# Phase B — Resolution Comparison
#
# Compares hillslope parameters at 1m, 2m, and 4m on the 5x5 tile block
# (R6-R10, C7-C11). 64GB is sufficient — Phase C ran 1m in 67s at 64GB.

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"

echo "=== Phase B Resolution Comparison ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Node: $(hostname)"
echo "pysheds fork: $(cd "$PYSHEDS_FORK" && git branch --show-current) @ $(cd "$PYSHEDS_FORK" && git rev-parse --short HEAD)"
echo ""

python scripts/phase_b/test_resolution_comparison.py
