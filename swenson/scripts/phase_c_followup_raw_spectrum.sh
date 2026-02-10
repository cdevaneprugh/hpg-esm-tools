#!/bin/bash
#SBATCH --job-name=phase_c_raw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:15:00
#SBATCH --qos=gerber-b
#SBATCH --output=logs/phase_c_raw_%j.log

# Phase C Follow-up: Raw DEM Spectrum Test
# Compares Laplacian vs raw elevation vs k²-corrected spectra.
# Faster than the baseline job (no sensitivity sweep, fewer gradient computations).

set -euo pipefail

SWENSON_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"

cd "$SWENSON_DIR"

# Activate conda environment
module load conda 2>/dev/null
conda activate ctsm

echo "Starting Phase C follow-up (raw spectrum) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

python scripts/phase_c_followup_raw_spectrum.py --plot-dir output/osbs/phase_c

echo ""
echo "Finished at $(date)"
