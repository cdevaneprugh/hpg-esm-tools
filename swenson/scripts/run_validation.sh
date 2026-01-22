#!/bin/bash
#SBATCH --job-name=hillslope-validate
#SBATCH --output=validation_%j.out
#SBATCH --error=validation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --qos=gerber

# Swenson hillslope validation job
# Runs our HillslopeGrid on MERIT sample and compares to published data

module load conda
conda activate ctsm

# Add pysheds fork to path
export PYTHONPATH="/blue/gerber/cdevaneprugh/pysheds_fork:$PYTHONPATH"

cd /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data

echo "Starting validation at $(date)"
echo "Running on $(hostname)"
echo ""

python ../scripts/validate_against_swenson.py

echo ""
echo "Completed at $(date)"
