#!/bin/bash
#SBATCH --job-name=5x5-utm-smoke
#SBATCH --output=logs/5x5_utm_smoke_%j.log
#SBATCH --partition=hpg-default
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

# 5x5 UTM Smoke Test — Phase A Validation at Scale
#
# Validates pysheds UTM CRS code paths on 25 NEON tiles (R6-R10, C7-C11,
# 25M pixels at 1m). Includes Lc sanity checks (Swenson Section 2.4).
#
# Memory: 64GB for 25M pixels. If resolve_flats OOMs, retry at 128GB.
#
# Usage:
#   cd $SWENSON
#   sbatch scripts/smoke_tests/run_5x5_utm.sh

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"
mkdir -p logs

# --- Environment setup ---
module load conda 2>/dev/null
conda activate ctsm

# Use the UTM-aware pysheds fork (feature/utm-crs-support branch)
export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"

echo "=== 5x5 UTM Smoke Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Verify pysheds fork branch
FORK_BRANCH=$(cd "$PYSHEDS_FORK" && git branch --show-current)
echo "pysheds fork branch: $FORK_BRANCH"
if [[ "$FORK_BRANCH" != "feature/utm-crs-support" ]]; then
    echo "WARNING: Expected branch 'feature/utm-crs-support', got '$FORK_BRANCH'"
fi
echo ""

# Verify pysheds imports from fork
python -c "
import pysheds.pgrid as pg
import os
fork_path = os.path.dirname(os.path.dirname(pg.__file__))
print(f'pysheds loaded from: {fork_path}')
assert 'pysheds_fork' in fork_path, f'pysheds not loading from fork: {fork_path}'
print('OK: pysheds loading from fork')
"
echo ""

# --- Run smoke test ---
python scripts/smoke_tests/run_5x5_utm.py

echo ""
echo "=== Job completed: $(date) ==="
