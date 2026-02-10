#!/bin/bash
#SBATCH --job-name=phase_c_lc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --qos=gerber-b
#SBATCH --output=logs/phase_c_lc_%j.log

# Phase C: Characteristic Length Scale Analysis
# Runs full-resolution FFT on OSBS interior mosaic with parameter sensitivity sweep.
# Expected runtime: < 5 minutes. 32GB is generous (peak ~15GB for intermediates).

set -euo pipefail

SWENSON_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"

cd "$SWENSON_DIR"

# Activate conda environment
module load conda 2>/dev/null
conda activate ctsm

# Add pysheds fork to path (not needed for this script, but consistent with pipeline)
export PYSHEDS_FORK="${PYSHEDS_FORK:-/blue/gerber/cdevaneprugh/pysheds_fork}"

echo "Starting Phase C Lc analysis at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

python scripts/phase_c_lc_analysis.py --plot-dir output/osbs/phase_c

echo ""
echo "Finished at $(date)"
