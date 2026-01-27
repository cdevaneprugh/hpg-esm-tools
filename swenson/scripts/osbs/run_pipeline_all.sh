#!/bin/bash
#SBATCH --job-name=osbs_hillslope_all
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/hillslope_all_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/hillslope_all_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

echo "============================================================"
echo "OSBS Hillslope Pipeline - ALL tiles"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================"

module load conda
conda activate ctsm

export PYSHEDS_FORK=/blue/gerber/cdevaneprugh/pysheds_fork
export TILE_SELECTION_MODE=all
export OUTPUT_DESCRIPTOR=full

cd /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson || exit 1
mkdir -p logs

echo "Starting pipeline (all tiles)..."
python scripts/osbs/run_pipeline.py

echo ""
echo "============================================================"
echo "Job completed: $(date)"
echo "============================================================"
