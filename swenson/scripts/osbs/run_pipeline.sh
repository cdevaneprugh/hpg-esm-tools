#!/bin/bash
#SBATCH --job-name=osbs_hillslope
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/hillslope_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/hillslope_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Print job info
echo "============================================================"
echo "OSBS Hillslope Pipeline"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

# Load modules
module load conda

# Activate conda environment
conda activate ctsm

# Set pysheds fork path
export PYSHEDS_FORK=/blue/gerber/cdevaneprugh/pysheds_fork

# Tile selection mode: "all" for full mosaic, "interior" for interior tiles only
export TILE_SELECTION_MODE=interior

# Output descriptor (used in output directory name and NetCDF filename)
export OUTPUT_DESCRIPTOR=interior

# Change to script directory
cd /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the pipeline
echo "Starting pipeline..."
python scripts/osbs/run_pipeline.py

# Print completion info
echo ""
echo "============================================================"
echo "Job completed: $(date)"
echo "============================================================"
