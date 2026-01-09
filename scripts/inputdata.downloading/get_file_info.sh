#!/bin/bash

# Default values
REMOTE_URL="https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/datm7/"
LOCAL_PATH="inputdata"
OUTPUT_DIR="."

# Parse command line options
while getopts "o:l:r:h" opt; do
    case $opt in
        o) OUTPUT_DIR="$OPTARG" ;;
        l) LOCAL_PATH="$OPTARG" ;;
        r) REMOTE_URL="$OPTARG" ;;
        h) echo "Usage: $0 [-o output_directory] [-l local_path] [-r remote_url]"
           echo "  -o: Specify output directory for file lists (default: current directory)"
           echo "  -l: Local path to check for existing files (default: inputdata)"
           echo "  -r: Remote SVN URL (default: $REMOTE_URL)"
           exit 0 ;;
        *) echo "Invalid option. Use -h for help."
           exit 1 ;;
    esac
done

# Function to convert bytes to human readable format
human_readable() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt 1048576 ]; then
        echo "$(awk "BEGIN {printf \"%.1f\", $bytes/1024}")KB"
    elif [ $bytes -lt 1073741824 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1048576}")MB"
    elif [ $bytes -lt 1099511627776 ]; then
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1073741824}")GB"
    else
        echo "$(awk "BEGIN {printf \"%.2f\", $bytes/1099511627776}")TB"
    fi
}

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Fetching file list from SVN server..."
echo "Remote URL: $REMOTE_URL"
echo "Local path: $LOCAL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo

# Initialize output files
> "$OUTPUT_DIR/all_files.txt"
> "$OUTPUT_DIR/to_download.txt"

echo "Getting file list with sizes..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/parse_svn_xml.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    echo "Please ensure parse_svn_xml.py is in the same directory as this script."
    exit 1
fi

# Use svn list with XML output and parse it with the Python script
svn list -R --xml "$REMOTE_URL" | python3 "$PYTHON_SCRIPT" "$REMOTE_URL" "$LOCAL_PATH" "$OUTPUT_DIR" > "$OUTPUT_DIR/stats.txt"

# Read the stats
if [ -f "$OUTPUT_DIR/stats.txt" ] && [ -s "$OUTPUT_DIR/stats.txt" ]; then
    IFS='|' read -r total_count total_size existing_count existing_size missing_count missing_size < "$OUTPUT_DIR/stats.txt"
    rm "$OUTPUT_DIR/stats.txt"
else
    echo "Error processing SVN output. Trying fallback method..."
    
    # Fallback: just get files without sizes
    svn list -R "$REMOTE_URL" | grep -v '/$' | while IFS= read -r filename; do
        echo "${REMOTE_URL}${filename}|0" >> "$OUTPUT_DIR/all_files.txt"
        
        # Extract repo subpath for correct local file checking
        repo_subpath=$(echo "$REMOTE_URL" | sed 's|.*/trunk/inputdata/||')
        local_file="$LOCAL_PATH/${repo_subpath}${filename}"
        
        if [ ! -f "$local_file" ]; then
            echo "${REMOTE_URL}${filename}" >> "$OUTPUT_DIR/to_download.txt"
        fi
    done
    
    total_count=$(wc -l < "$OUTPUT_DIR/all_files.txt")
    missing_count=$(wc -l < "$OUTPUT_DIR/to_download.txt")
    existing_count=$((total_count - missing_count))
    total_size=0
    existing_size=0
    missing_size=0
fi

# Print summary
echo
echo "===== SUMMARY ====="
echo "Total files:            $total_count ($(human_readable $total_size))"
echo "Already downloaded:     $existing_count ($(human_readable $existing_size))"
echo "Still need to download: $missing_count ($(human_readable $missing_size))"
echo
echo "File lists generated:"
echo "  - $OUTPUT_DIR/all_files.txt (all files with sizes)"
echo "  - $OUTPUT_DIR/to_download.txt (files to download)"
echo
