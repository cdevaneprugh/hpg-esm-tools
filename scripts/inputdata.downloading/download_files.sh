#!/bin/bash

# Generate timestamp for log directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Default values
TO_DOWNLOAD="to_download.txt"
LOCAL_PATH="inputdata"
LOG_DIR_BASE="logs"  # Base directory for logs
LOG_DIR="${LOG_DIR_BASE}/${TIMESTAMP}"  # Actual log directory with timestamp
LOG_FILE="${LOG_DIR}/download_log.txt"
FAILED_FILE="${LOG_DIR}/failed_downloads.txt"

# Parse command line options
while getopts "i:l:d:o:f:h" opt; do
    case $opt in
        i) TO_DOWNLOAD="$OPTARG" ;;
        l) LOCAL_PATH="$OPTARG" ;;
        d) # If user specifies log directory base, update paths
           LOG_DIR_BASE="$OPTARG"
           LOG_DIR="${LOG_DIR_BASE}/${TIMESTAMP}"
           LOG_FILE="${LOG_DIR}/download_log.txt"
           FAILED_FILE="${LOG_DIR}/failed_downloads.txt"
           ;;
        o) LOG_FILE="$OPTARG" ;;  # Allow explicit override of log file
        f) FAILED_FILE="$OPTARG" ;;  # Allow explicit override of failed file
	h) echo "Usage: $0 [-i input_file] [-l local_path] [-d log_dir_base] [-o log_file] [-f failed_file]"
           echo "  -i: Input file with URLs to download (default: to_download.txt)"
           echo "  -l: Local path for downloads (default: inputdata)"
           echo "  -d: Base directory for logs (default: logs)"
           echo "      A timestamped subdirectory will be created under this"
           echo "  -o: Override log file path (default: logs/TIMESTAMP/download_log.txt)"
           echo "  -f: Override failed file path (default: logs/TIMESTAMP/failed_downloads.txt)"
           echo ""
           echo "Examples:"
           echo "  # Default - creates logs/20240315_143022/"
           echo "  $0"
           echo ""
           echo "  # Custom base directory - creates /scratch/downloads/logs/20240315_143022/"
           echo "  $0 -d /scratch/downloads/logs"
           echo ""
           echo "  # Override everything - use exact paths"
           echo "  $0 -o /tmp/my_download.log -f /tmp/my_failed.txt"
           echo ""
           echo "Example log directory: ${LOG_DIR_BASE}/${TIMESTAMP}/"
           exit 0 ;;
        *) echo "Invalid option. Use -h for help."
           exit 1 ;;
    esac
done

# Remove trailing slashes from LOCAL_PATH to ensure consistent path handling
LOCAL_PATH="${LOCAL_PATH%/}"

# Check if input file exists
if [ ! -f "$TO_DOWNLOAD" ]; then
    echo "Error: Input file '$TO_DOWNLOAD' not found!"
    exit 1
fi

# Create log directory if it doesn't exist
# This handles both the timestamped default and any custom paths
LOG_FILE_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_FILE_DIR" ]; then
    mkdir -p "$LOG_FILE_DIR"
fi

FAILED_FILE_DIR=$(dirname "$FAILED_FILE")
if [ ! -d "$FAILED_FILE_DIR" ]; then
    mkdir -p "$FAILED_FILE_DIR"
fi

# Initialize log files
echo "========================================" >> "$LOG_FILE"
echo "Download started at: $(date)" >> "$LOG_FILE"
echo "Log directory: $LOG_FILE_DIR" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
> "$FAILED_FILE"

# Count total files for progress
TOTAL_FILES=$(wc -l < "$TO_DOWNLOAD")
CURRENT_FILE=0
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "Starting download of $TOTAL_FILES files..."
echo "Logging to: $LOG_FILE"
echo "Failed downloads will be saved to: $FAILED_FILE"
echo

# Function to extract relative path from URL
get_relative_path() {
    local url="$1"
    # Remove everything up to and including '/trunk/inputdata/'
    # This preserves the full directory structure (lnd/clm2/surfdata_esmf/...)
    echo "$url" | sed 's|.*/trunk/inputdata/||'
}

# Process each URL
while IFS= read -r url; do
    # Skip empty lines
    [ -z "$url" ] && continue
    
    CURRENT_FILE=$((CURRENT_FILE + 1))
    
    # Extract relative path from URL
    # Example: https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/surfdata_esmf/NEON/file.nc
    # Becomes: lnd/clm2/surfdata_esmf/NEON/file.nc
    rel_path=$(get_relative_path "$url")
    local_file="$LOCAL_PATH/$rel_path"
    
    # Create directory structure if it doesn't exist
    local_dir=$(dirname "$local_file")
    if [ ! -d "$local_dir" ]; then
        mkdir -p "$local_dir"
        # Force permissions to rwxrwsr-x (2775) - overrides umask
        chmod 2775 "$local_dir"
        # Also set permissions on any parent directories that were created
        temp_dir="$local_dir"
        while [ "$temp_dir" != "$LOCAL_PATH" ] && [ "$temp_dir" != "." ] && [ "$temp_dir" != "/" ]; do
            chmod 2775 "$temp_dir" 2>/dev/null
            temp_dir=$(dirname "$temp_dir")
        done
    fi
    
    # Display progress
    echo "[$CURRENT_FILE/$TOTAL_FILES] Downloading: $rel_path"
    
    # Timestamp for log
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Download with wget
    # -c: continue/resume partial downloads
    # -t 3: retry 3 times
    # --timeout=30: 30 second timeout for initial connection
    # --read-timeout=300: 5 minute timeout for reading data (adjust for large files)
    # -q: quiet mode (no wget output)
    # --show-progress: still show progress bar
    wget -c \
         -t 3 \
         --timeout=30 \
         --read-timeout=300 \
         -q --show-progress \
         "$url" \
         -O "$local_file" 2>/dev/null
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        # Verify file exists and has size > 0
        if [ -f "$local_file" ] && [ -s "$local_file" ]; then
            echo "$timestamp | $rel_path | SUCCESS" >> "$LOG_FILE"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "  SUCCESS"
        else
            echo "$timestamp | $rel_path | FAIL (zero size)" >> "$LOG_FILE"
            echo "$url" >> "$FAILED_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "  FAIL (zero size file)"
        fi
    else
        echo "$timestamp | $rel_path | FAIL (wget error $?)" >> "$LOG_FILE"
        echo "$url" >> "$FAILED_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  FAIL (download error)"
    fi
    
    # Optional: Add small delay between downloads to be nice to the server
    # sleep 0.5
    
done < "$TO_DOWNLOAD"

# Final summary
echo
echo "========================================" 
echo "Download Summary"
echo "========================================" 
echo "Total files:      $TOTAL_FILES"
echo "Successful:       $SUCCESS_COUNT"
echo "Failed:           $FAIL_COUNT"
echo "========================================" 

# Add summary to log
echo "========================================" >> "$LOG_FILE"
echo "Download completed at: $(date)" >> "$LOG_FILE"
echo "Total: $TOTAL_FILES | Success: $SUCCESS_COUNT | Failed: $FAIL_COUNT" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

if [ $FAIL_COUNT -gt 0 ]; then
    echo
    echo "Failed downloads saved to: $FAILED_FILE"
    echo "To retry failed downloads, run:"
    echo "  $0 -i $FAILED_FILE"
fi

# Exit with error if any downloads failed
[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
