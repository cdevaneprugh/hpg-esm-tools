#!/bin/bash
#SBATCH --job-name=hand_bin_cmp
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/hand_bin_cmp_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/hand_bin_cmp_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Compare 5 HAND binning strategies on the production domain.
# Runs Steps 1-4 once (~14 min), then tests all strategies in seconds.

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

echo "=== HAND Binning Strategy Comparison ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/compare_hand_binning.py

echo ""
echo "Completed: $(date)"
