#!/bin/bash
#SBATCH --job-name=neon_forcing
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/neon_forcing_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/neon_forcing_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Step 2 of the Option B NEON->DATM pipeline (Phase I): run the forked, OFFLINE
# flow.api.clm.R against the pre-staged shared archive. NO token -- it does not
# download (doDnld<-FALSE). Reads the archive, gap-fills, writes monthly
# OSBS_atm_YYYY-MM.nc under DIROUT/OSBS/atm/ (mirrors the v4 layout).
#
# Config via the script's METHPARAFLOW env path. Scope the window per run:
#   DATEBGN=2018-06-01 DATEEND=2018-06-30 sbatch scripts/neon_forcing/run_forcing.sh
# Full record: DATEBGN=2016-08-01 DATEEND=2025-06-30 LOWMEM=TRUE sbatch ... (LOWMEM
# chunks the multi-year EC stack; bump --mem/--time for the full run).
#
# See phases/I-neon-forcing.md sec 10.

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
NCAR_NEON="/blue/gerber/cdevaneprugh/ncar-neon"   # fork clone (uf-osbs branch); not in bashrc

cd "$SWENSON"
mkdir -p logs

# Config read by flow.api.clm.R only when METHPARAFLOW is set (dates are full YYYY-MM-DD).
export METHPARAFLOW=1
export SITE="OSBS"
export DATEBGN="${DATEBGN:-2018-06-01}"            # smoke default; override per run
export DATEEND="${DATEEND:-2018-06-30}"
export DIROUT="$SWENSON/data/datm/neon_OSBS/custom"
export LOWMEM="${LOWMEM:-FALSE}"                   # TRUE chunks stackEddy for the full 10-yr EC

# Keep transient files off the shared node /tmp.
export TMPDIR="/blue/gerber/cdevaneprugh/.tmp/neon_forcing"
mkdir -p "$TMPDIR"

module load conda 2>/dev/null
conda activate neon-forcing

echo "=== NEON forcing generation (Step 2, offline) ==="
echo "Date:    $(date)"
echo "Fork:    $NCAR_NEON ($(git -C "$NCAR_NEON" branch --show-current) @ $(git -C "$NCAR_NEON" rev-parse --short HEAD))"
echo "Window:  $DATEBGN .. $DATEEND | LOWMEM=$LOWMEM"
echo "DIROUT:  $DIROUT/OSBS/atm"
echo "TMPDIR:  $TMPDIR"
echo ""

Rscript "$NCAR_NEON/TowerTools_ForcingData/flow.api.clm.R"
