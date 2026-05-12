#!/bin/bash
#SBATCH --job-name=lc_water_cmp
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/lc_water_cmp_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/lc_water_cmp_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Compare Lc with and without NWI lake masking (3 methods, sequential).
# Production domain: 9000x10000 at 1m = 90M pixels.

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"

cd "$SWENSON"
mkdir -p logs

module load conda 2>/dev/null
conda activate ctsm

echo "=== Lc Water Masking Comparison ==="
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

python scripts/osbs/compare_lc_water_masking.py

echo ""
echo "Completed: $(date)"
