#!/bin/bash
#SBATCH --job-name=neon_download
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/neon_download_%j.log
#SBATCH --error=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/logs/neon_download_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --partition=hpg-default
#SBATCH --account=gerber
#SBATCH --qos=gerber-b

# Step 1 of the Option B NEON->DATM pipeline (Phase I): authenticated raw download
# of the 10 OSBS products into the shared archive. Runs on a compute node -- a NEON
# API token lifts the /data/ 403 from HPG (phases/I-neon-forcing.md sec 12).
# zipsByProduct skips files already present, so re-runs resume and a scoped test
# pull is a strict subset of the eventual full pull (no re-download).
#
# Scope the window for a test:   NEON_START=2018-05 NEON_END=2018-06 sbatch scripts/neon_forcing/run_download.sh
# Full pull uses script defaults (2016-08 .. 2025-06). For the long full pull, use
# the non-burst qos + more time:  sbatch --qos=gerber --time=24:00:00 scripts/neon_forcing/run_download.sh
#
# See phases/I-neon-forcing.md sec 10 / sec 12.

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
ARCHIVE="/blue/gerber/earth_models/neon/raw/OSBS"

cd "$SWENSON"
mkdir -p logs
mkdir -p "$ARCHIVE"

# NEON API token -- lifts the /data/ 403 from HPG (phases/I sec 12). Private, mode 600, NOT in repo.
if [[ ! -r "$HOME/.neon_token" ]]; then
  echo "ERROR: ~/.neon_token not readable; the /data/ download will 403 from HPG." >&2
  exit 1
fi
export NEON_TOKEN
NEON_TOKEN="$(cat "$HOME/.neon_token")"

# download_raw.R default already points here; assert it for clarity.
export NEON_DIRDNLD="$ARCHIVE"

# Keep transient files off the shared node /tmp.
export TMPDIR="/blue/gerber/cdevaneprugh/.tmp/neon_forcing"
mkdir -p "$TMPDIR"

module load conda 2>/dev/null
conda activate neon-forcing

echo "=== NEON raw download (Step 1) ==="
echo "Date:    $(date)"
echo "Archive: $ARCHIVE"
echo "Window:  ${NEON_START:-2016-08 (default)} .. ${NEON_END:-2025-06 (default)}"
echo "TMPDIR:  $TMPDIR"
echo ""

Rscript "$SWENSON/scripts/neon_forcing/download_raw.R"
