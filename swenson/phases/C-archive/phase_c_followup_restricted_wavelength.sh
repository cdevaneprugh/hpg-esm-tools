#!/bin/bash
#SBATCH --job-name=phase_c_rwl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:15:00
#SBATCH --qos=gerber-b
#SBATCH --output=logs/phase_c_rwl_%j.log

# Phase C Follow-up: Restricted Wavelength Sweep
# Parts A (single-tile FFT), B (contiguous mosaic FFT), C (restricted wavelength sweep).
# Expected runtime: < 5 minutes. 32GB is generous (single tile is tiny, mosaic block is 9x10 km).

set -euo pipefail

SWENSON_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"

cd "$SWENSON_DIR"

# Activate conda environment
module load conda 2>/dev/null
conda activate ctsm

echo "Starting Phase C follow-up (restricted wavelength) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

python scripts/phase_c_followup_restricted_wavelength.py --plot-dir output/osbs/phase_c

echo ""
echo "Finished at $(date)"
