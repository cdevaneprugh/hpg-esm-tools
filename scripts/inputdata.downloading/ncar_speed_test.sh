#!/bin/bash

# Test download speed from NCAR SVN server
echo "Testing download speed from NCAR SVN server..."
echo "========================================"

# URL to test - pick a reasonably sized file from your to_download.txt
TEST_URL="${1:-https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/datm7/atm_forcing.datm7.GSWP3.0.5d.v1.c170516/TPHWL/clmforc.GSWP3.c2011.0.5x0.5.TPQWL.1901-01.nc}"
OUTPUT_DIR="speed_test_$$"

mkdir -p "$OUTPUT_DIR"

# Single file download test
echo "Downloading test file..."
START_TIME=$(date +%s)

# Use wget with clean progress bar
wget --progress=bar:force "$TEST_URL" -O "$OUTPUT_DIR/test_file"

END_TIME=$(date +%s)

# Extract file size
FILE_SIZE=$(stat -f%z "$OUTPUT_DIR/test_file" 2>/dev/null || stat -c%s "$OUTPUT_DIR/test_file" 2>/dev/null)

# Calculate speed
DURATION=$((END_TIME - START_TIME))
if [ $DURATION -gt 0 ]; then
    SPEED_BPS=$((FILE_SIZE / DURATION))
    SPEED_MBPS=$(awk "BEGIN {printf \"%.2f\", $SPEED_BPS / 1048576}")
    echo "Download completed!"
    echo "  Time taken: ${DURATION} seconds"
    echo "  File size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes")"
    echo "  Average speed: ${SPEED_MBPS} MB/s"
else
    echo "Download too fast to measure accurately"
fi

# Cleanup
rm -rf "$OUTPUT_DIR"

echo -e "\n========================================"
echo "ESTIMATED DOWNLOAD TIMES"
echo "========================================"

# Estimate based on the speed test
estimate_time() {
    local total_size=$1
    local speed_mbps=$2
    
    # Convert TB to MB
    local size_mb=$((total_size * 1024 * 1024))
    
    # Calculate time in seconds
    local time_seconds=$(awk "BEGIN {printf \"%.0f\", $size_mb / $speed_mbps}")
    
    # Convert to hours and days
    local hours=$(awk "BEGIN {printf \"%.1f\", $time_seconds / 3600}")
    local days=$(awk "BEGIN {printf \"%.1f\", $hours / 24}")
    
    echo "$hours hours ($days days)"
}

if [ -n "$SPEED_MBPS" ] && [ "$DURATION" -gt 0 ]; then
    echo "At measured speed (${SPEED_MBPS} MB/s):"
    echo "  1 TB: $(estimate_time 1 $SPEED_MBPS)"
    echo "  2 TB: $(estimate_time 2 $SPEED_MBPS)"
    echo "  5 TB: $(estimate_time 5 $SPEED_MBPS)"
    
    echo -e "\nWith 50% safety buffer:"
    BUFFER_SPEED=$(awk "BEGIN {printf \"%.2f\", $SPEED_MBPS * 0.67}")
    echo "  1 TB: $(estimate_time 1 $BUFFER_SPEED)"
    echo "  2 TB: $(estimate_time 2 $BUFFER_SPEED)"
    echo "  5 TB: $(estimate_time 5 $BUFFER_SPEED)"
    
    echo -e "\n========================================"
    echo "SCHEDULER RECOMMENDATIONS"
    echo "========================================"
    echo "- These estimates assume sustained transfer rate"
    echo "- Add extra time for job startup/cleanup"
    echo "- Consider breaking very large downloads into multiple jobs"
    echo "- Network speeds often vary by time of day"
fi
