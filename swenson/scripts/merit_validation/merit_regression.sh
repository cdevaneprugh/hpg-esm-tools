#!/bin/bash
#SBATCH --job-name=merit-regression
#SBATCH --output=scripts/merit_validation/output/merit_regression_%j.log
#SBATCH --partition=hpg-default
#SBATCH --mem=48gb
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"

module load conda 2>/dev/null
conda activate ctsm

export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"

echo "=== MERIT Geographic Regression Test ==="
echo "Date: $(date)"
echo "pysheds fork branch: $(cd "$PYSHEDS_FORK" && git branch --show-current)"
echo "pysheds fork commit: $(cd "$PYSHEDS_FORK" && git rev-parse --short HEAD)"
echo ""

python scripts/merit_validation/merit_regression.py
