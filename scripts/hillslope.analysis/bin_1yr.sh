#!/bin/bash
# Script: bin_1yr.sh
# Purpose: Create annual (1-year) binned averages from concatenated CTSM history files
# Usage: ./bin_1yr.sh <input_file> <output_file>

set -euo pipefail

show_usage() {
    cat <<EOF
Usage: $0 <input_file> <output_file>

Create annual (1-year) binned averages from concatenated CTSM history files.
Works directly on concatenated NetCDF files (e.g., combined_h0.nc, combined_h1.nc).

Arguments:
  input_file      Input NetCDF file (concatenated history file)
  output_file     Output NetCDF file (binned time series)

Output:
  Creates single NetCDF file with reduced time dimension
  Incomplete final year is automatically discarded

EOF
}

# Parse arguments
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
YEARS_PER_BIN=1  # Fixed at 1 year for annual binning

# Get total time steps from input file
TOTAL_STEPS=$(ncdump -h "$INPUT_FILE" | grep "time = UNLIMITED" | sed -n 's/.*(\([0-9]*\) currently).*/\1/p')

# Calculate binning parameters
STEPS_PER_BIN=$((YEARS_PER_BIN * 12))  # 12 months per year
NUM_BINS=$((TOTAL_STEPS / STEPS_PER_BIN))  # Number of complete bins

# Create temporary directory
TEMP_DIR=$(mktemp -d -t bin_1yr.XXXXXX)
trap "rm -rf $TEMP_DIR" EXIT

# Process each bin
for ((bin=0; bin<NUM_BINS; bin++)); do
    start_idx=$((bin * STEPS_PER_BIN))
    end_idx=$((start_idx + STEPS_PER_BIN - 1))

    year=$((bin + 1))

    output_file="$TEMP_DIR/bin_$(printf "%04d" $bin).nc"

    # Average this time range
    # Suppress ncra info messages but show errors
    ncra -O -d time,$start_idx,$end_idx "$INPUT_FILE" "$output_file" 2>&1 | grep -v "ncra: INFO" || true
done

# Concatenate all bins
ncrcat -O "$TEMP_DIR"/bin_*.nc "$OUTPUT_FILE" 2>&1 | grep -v "ncrcat: INFO" || true
