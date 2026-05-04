#!/usr/bin/env bash
#SBATCH --job-name=hillslope_plots
#SBATCH --output=/blue/gerber/cdevaneprugh/hpg-esm-tools/scripts/hillslope.analysis/logs/hillslope_plots_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --qos=gerber
#SBATCH --account=gerber

# Generate hillslope analysis plots for all 4 OSBS branch cases.
#
# Workflow per case:
#   1. Concatenate h0/h1 archive files → tmpdir
#   2. Bin to annual averages
#   3. Run all plotting scripts
#   4. Clean up intermediate files
#
# Usage:
#   sbatch generate_case_plots.sh

set -euo pipefail

SCRIPT_DIR=/blue/gerber/cdevaneprugh/hpg-esm-tools/scripts/hillslope.analysis
PLOT_BASE="$SCRIPT_DIR/plots"
ARCH=/blue/gerber/cdevaneprugh/earth_model_output/cime_output_root/archive
TMPDIR_BASE=/blue/gerber/cdevaneprugh/.tmp/hillslope_plots_$$
BIN_SCRIPT="$SCRIPT_DIR/bin_temporal.sh"

# Load environment
module load conda 2>/dev/null || true
conda activate ctsm

mkdir -p "$TMPDIR_BASE" logs

# ============================================================================
# Case definitions: name, hist_dir, h0_glob, h1_glob
# ============================================================================
declare -A CASE_HIST CASE_H0GLOB CASE_H1GLOB

CASE_HIST[spillheight]="$ARCH/osbs2.branch.spillheight/lnd/hist"
CASE_H0GLOB[spillheight]="*.h0.*.nc"
CASE_H1GLOB[spillheight]="*.h1.*.nc"

CASE_HIST[v4]="$ARCH/osbs2.branch.v4/lnd/hist"
CASE_H0GLOB[v4]="*.h0a.*.nc"
CASE_H1GLOB[v4]="*.h1a.*.nc"

CASE_HIST[v1]="$ARCH/osbs4.branch.v1/lnd/hist"
CASE_H0GLOB[v1]="*.h0a.*.nc"
CASE_H1GLOB[v1]="*.h1a.*.nc"

CASE_HIST[v2]="$ARCH/osbs4.branch.v2/lnd/hist"
CASE_H0GLOB[v2]="*.h0a.*.nc"
CASE_H1GLOB[v2]="*.h1a.*.nc"

# Full case names for plot directory labels
declare -A CASE_LABEL
CASE_LABEL[spillheight]="osbs2.branch.spillheight"
CASE_LABEL[v4]="osbs2.branch.v4"
CASE_LABEL[v1]="osbs4.branch.v1"
CASE_LABEL[v2]="osbs4.branch.v2"

# ============================================================================
# Process each case
# ============================================================================
for tag in spillheight v4 v1 v2; do
    label="${CASE_LABEL[$tag]}"
    hist_dir="${CASE_HIST[$tag]}"
    h0_glob="${CASE_H0GLOB[$tag]}"
    h1_glob="${CASE_H1GLOB[$tag]}"
    plot_dir="$PLOT_BASE/$label"
    tmp="$TMPDIR_BASE/$tag"

    echo ""
    echo "================================================================"
    echo "Processing: $label"
    echo "================================================================"

    mkdir -p "$plot_dir" "$tmp"

    # --- Step 1: Concatenate ---
    echo "[$(date +%H:%M:%S)] Concatenating h0..."
    # shellcheck disable=SC2086
    ncrcat -O "$hist_dir"/$h0_glob "$tmp/h0_combined.nc"

    echo "[$(date +%H:%M:%S)] Concatenating h1..."
    # shellcheck disable=SC2086
    ncrcat -O "$hist_dir"/$h1_glob "$tmp/h1_combined.nc"

    # --- Step 2: Bin to annual ---
    echo "[$(date +%H:%M:%S)] Binning h0 to annual..."
    "$BIN_SCRIPT" "$tmp/h0_combined.nc" "$tmp/h0_1yr.nc" --years=1

    echo "[$(date +%H:%M:%S)] Binning h1 to annual..."
    "$BIN_SCRIPT" "$tmp/h1_combined.nc" "$tmp/h1_1yr.nc" --years=1

    # --- Step 3: Generate plots ---
    echo "[$(date +%H:%M:%S)] Generating plots..."
    cd "$SCRIPT_DIR"

    # Full timeseries (h0, annual bins): GPP and TOTECOSYSC
    for var in GPP TOTECOSYSC; do
        python3 plot_timeseries_full.py "$tmp/h0_1yr.nc" "$plot_dir/${var}_full.png" "$var"
    done

    # Last 20 years by HAND zone (h1, annual bins): GPP and TOTECOSYSC
    for var in GPP TOTECOSYSC; do
        python3 plot_timeseries_last20.py "$tmp/h1_1yr.nc" "$plot_dir/${var}_last20.png" "$var" --years=20
    done

    # ZWT hillslope profile (h1, annual bins)
    python3 plot_zwt_hillslope_profile.py "$tmp/h1_1yr.nc" "$plot_dir/ZWT_profile.png"

    # Cross-section (h1, annual bins)
    python3 plot_hillslope_cross_section.py "$tmp/h1_1yr.nc" "$plot_dir/cross_section.png"

    # TAI heatmaps (h1, annual bins): ZWT and GPP
    python3 plot_tai_heatmap.py "$tmp/h1_1yr.nc" "$plot_dir/ZWT_heatmap.png" ZWT
    python3 plot_tai_heatmap.py "$tmp/h1_1yr.nc" "$plot_dir/GPP_heatmap.png" GPP

    # Carbon-water coupling (h1, annual bins)
    python3 plot_carbon_water_coupling.py "$tmp/h1_1yr.nc" "$plot_dir/carbon_water_coupling.png"

    echo "[$(date +%H:%M:%S)] Done: $label → $plot_dir/"

    # --- Step 4: Cleanup ---
    rm -rf "$tmp"
done

# Final cleanup
rm -rf "$TMPDIR_BASE"

echo ""
echo "================================================================"
echo "All cases complete. Plots in: $PLOT_BASE/"
echo "================================================================"
ls -R "$PLOT_BASE/"
