#!/bin/bash
# Script: bin_20yr.sh
# Purpose: Create N-year binned averages from concatenated CTSM history files
# Usage: ./bin_20yr.sh <input_file> <output_file> [years_per_bin]

set -euo pipefail

show_usage() {
    cat <<EOF
Usage: $0 <input_file> <output_file> [years_per_bin]

Create N-year binned averages from concatenated CTSM history files.
Works directly on concatenated NetCDF files (e.g., combined_h0.nc, combined_h1.nc).

Arguments:
  input_file      Input NetCDF file (concatenated history file)
  output_file     Output NetCDF file (binned time series)
  years_per_bin   Number of years to average per bin (default: 20)

EOF
}

# Parse arguments
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
YEARS_PER_BIN="${3:-20}"

# Get total time steps from input file
TOTAL_STEPS=$(ncdump -h "$INPUT_FILE" | grep "time = UNLIMITED" | sed -n 's/.*(\([0-9]*\) currently).*/\1/p')

# Calculate binning parameters
STEPS_PER_BIN=$((YEARS_PER_BIN * 12))  # Assuming monthly data
NUM_BINS=$((TOTAL_STEPS / STEPS_PER_BIN))  # Number of complete bins

# Create temporary directory
TEMP_DIR=$(mktemp -d -t bin_20yr.XXXXXX)
trap "rm -rf $TEMP_DIR" EXIT

# Process each bin
for ((bin=0; bin<NUM_BINS; bin++)); do
    start_idx=$((bin * STEPS_PER_BIN))
    end_idx=$((start_idx + STEPS_PER_BIN - 1))

    bin_start_year=$((bin * YEARS_PER_BIN + 1))
    bin_end_year=$(((bin + 1) * YEARS_PER_BIN))

    output_file="$TEMP_DIR/bin_$(printf "%04d" $bin).nc"


    # Average this time range
    # Suppress ncra info messages but show errors
    ncra -O -d time,$start_idx,$end_idx "$INPUT_FILE" "$output_file" 2>&1 | grep -v "ncra: INFO" || true
done

# Concatenate all bins
ncrcat -O "$TEMP_DIR"/bin_*.nc "$OUTPUT_FILE" 2>&1 | grep -v "ncrcat: INFO" || true
