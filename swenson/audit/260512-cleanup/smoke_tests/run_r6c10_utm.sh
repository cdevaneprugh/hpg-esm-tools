#!/bin/bash
#SBATCH --job-name=r6c10-utm
#SBATCH --output=logs/r6c10_utm_%j.log
#SBATCH --partition=hpg-default
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

# R6C10 UTM Smoke Test — Phase A Validation
#
# Validates the UTM CRS code path in the pysheds fork on the standard
# R6C10 smoke test tile (1000x1000, 1m, EPSG:32617).
#
# Memory: 8GB is generous for 1M pixels. Bump to 32/64/128GB if OOM
# before considering subsampling.
#
# Usage:
#   cd $SWENSON
#   sbatch scripts/smoke_tests/run_r6c10_utm.sh

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

echo "=== R6C10 UTM Smoke Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
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
python scripts/smoke_tests/run_r6c10_utm.py

echo ""
echo "=== Job completed: $(date) ==="
