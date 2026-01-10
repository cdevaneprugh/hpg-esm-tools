#!/bin/bash
# =============================================================================
# bin_temporal.sh - Temporal binning for CTSM history files
# =============================================================================
#
# PURPOSE:
#   Create N-year binned averages from concatenated CTSM monthly history files.
#   Reduces seasonal variability for long-term trend analysis.
#
# USAGE:
#   ./bin_temporal.sh <input_file> <output_file> [--years=N]
#   ./bin_temporal.sh -h | --help
#
# ARGUMENTS:
#   input_file      Input NetCDF file (concatenated monthly history file)
#   output_file     Output NetCDF file (binned time series)
#   --years=N       Number of years per bin (default: 1)
#
# EXAMPLES:
#   # Annual averages (default)
#   ./bin_temporal.sh combined_h0.nc combined_h0_1yr.nc
#
#   # 20-year averages
#   ./bin_temporal.sh combined_h0.nc combined_h0_20yr.nc --years=20
#
#   # 5-year averages
#   ./bin_temporal.sh combined_h1.nc combined_h1_5yr.nc --years=5
#
# OUTPUT:
#   Creates single NetCDF file with reduced time dimension.
#   Each output time step is the average of N years of monthly data.
#   Incomplete final bin (< N years) is automatically discarded.
#
# DEPENDENCIES:
#   - NCO tools: ncdump, ncra, ncrcat
#   - Assumes monthly input data (12 time steps per year)
#
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Function: show_usage
# Purpose: Display help message and usage examples
# -----------------------------------------------------------------------------
show_usage() {
    cat <<'EOF'
bin_temporal.sh - Temporal binning for CTSM history files

USAGE:
    ./bin_temporal.sh <input_file> <output_file> [--years=N]
    ./bin_temporal.sh -h | --help

ARGUMENTS:
    input_file      Input NetCDF file (concatenated monthly history)
    output_file     Output NetCDF file (binned time series)
    --years=N       Years per bin (default: 1 for annual averages)

EXAMPLES:
    ./bin_temporal.sh combined_h0.nc combined_h0_1yr.nc
    ./bin_temporal.sh combined_h0.nc combined_h0_20yr.nc --years=20

EOF
}

# -----------------------------------------------------------------------------
# Parse command line arguments
# -----------------------------------------------------------------------------

# Check for help flag
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

# Require at least 2 arguments
if [[ $# -lt 2 ]]; then
    echo "ERROR: Missing required arguments" >&2
    echo "" >&2
    show_usage >&2
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
YEARS_PER_BIN=1  # Default: annual binning

# Parse optional --years=N argument
for arg in "${@:3}"; do
    case "$arg" in
        --years=*)
            YEARS_PER_BIN="${arg#*=}"
            # Validate it's a positive integer
            if ! [[ "$YEARS_PER_BIN" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: --years must be a positive integer, got: $YEARS_PER_BIN" >&2
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unknown argument: $arg" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate input file
# -----------------------------------------------------------------------------
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Extract time dimension info from input file
# -----------------------------------------------------------------------------
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Binning: $YEARS_PER_BIN year(s) per bin"
echo ""

# Get total time steps from input file using ncdump header
TOTAL_STEPS=$(ncdump -h "$INPUT_FILE" | grep "time = UNLIMITED" | sed -n 's/.*(\([0-9]*\) currently).*/\1/p')

if [[ -z "$TOTAL_STEPS" ]] || [[ "$TOTAL_STEPS" -eq 0 ]]; then
    echo "ERROR: Could not determine time dimension size from input file" >&2
    exit 1
fi

echo "Input time steps: $TOTAL_STEPS ($(( TOTAL_STEPS / 12 )) years of monthly data)"

# -----------------------------------------------------------------------------
# Calculate binning parameters
# -----------------------------------------------------------------------------
STEPS_PER_BIN=$((YEARS_PER_BIN * 12))  # 12 months per year
NUM_BINS=$((TOTAL_STEPS / STEPS_PER_BIN))  # Number of complete bins
REMAINDER=$((TOTAL_STEPS % STEPS_PER_BIN))

if [[ $NUM_BINS -eq 0 ]]; then
    echo "ERROR: Not enough data for ${YEARS_PER_BIN}-year binning" >&2
    echo "       Need at least $STEPS_PER_BIN time steps, have $TOTAL_STEPS" >&2
    exit 1
fi

echo "Output time steps: $NUM_BINS (${YEARS_PER_BIN}-year bins)"

if [[ $REMAINDER -gt 0 ]]; then
    echo "Note: Discarding incomplete final bin ($REMAINDER months)"
fi
echo ""

# -----------------------------------------------------------------------------
# Create temporary directory for intermediate files
# -----------------------------------------------------------------------------
TEMP_DIR=$(mktemp -d -t bin_temporal.XXXXXX)

# Cleanup on exit (success or failure)
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Process each bin
# -----------------------------------------------------------------------------
echo "Processing $NUM_BINS bins..."

for ((bin=0; bin<NUM_BINS; bin++)); do
    # Calculate time indices for this bin
    start_idx=$((bin * STEPS_PER_BIN))
    end_idx=$((start_idx + STEPS_PER_BIN - 1))

    # Output file for this bin
    bin_file="$TEMP_DIR/bin_$(printf "%04d" $bin).nc"

    # Progress indicator
    printf "  Bin %d/%d (time steps %d-%d)\r" $((bin + 1)) $NUM_BINS $start_idx $end_idx

    # Average this time range using ncra
    # Suppress INFO messages but show errors
    ncra -O -d time,$start_idx,$end_idx "$INPUT_FILE" "$bin_file" 2>&1 | grep -v "ncra: INFO" || true
done

echo ""  # Newline after progress
echo "Concatenating bins..."

# -----------------------------------------------------------------------------
# Concatenate all bins into final output
# -----------------------------------------------------------------------------
ncrcat -O "$TEMP_DIR"/bin_*.nc "$OUTPUT_FILE" 2>&1 | grep -v "ncrcat: INFO" || true

# Report output file size
OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo ""
echo "Done! Output: $OUTPUT_FILE ($OUTPUT_SIZE)"
