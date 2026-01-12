#!/bin/bash

# CTSM Custom Modifications Deployment Script
# Deploys custom files to a CTSM installation with optional logging

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Script directory and paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKING_DIR="$(pwd)"
LOGS_DIR="${WORKING_DIR}/logs"
DEPLOY_SOURCE_DIR="${SCRIPT_DIR}/files.to.deploy"

# Colors for terminal output (but not for log file)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
CTSMROOT=""
ENABLE_LOG=false
FORCE_COPY=false
LOG_FILE=""
FILES_ADDED=0
FILES_MODIFIED=0
FILES_SKIPPED=0
FILES_PROCESSED=()

# Function to print colored output to terminal only
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_skip() {
    echo -e "${BLUE}[SKIP]${NC} $1"
}

# Function to write to log file if logging is enabled
log_write() {
    if [ "$ENABLE_LOG" = true ] && [ -n "$LOG_FILE" ]; then
        echo "$1" >> "$LOG_FILE"
    fi
}

# Function to compare two files
files_are_identical() {
    local file1="$1"
    local file2="$2"
    
    # If destination doesn't exist, files are not identical
    if [ ! -f "$file2" ]; then
        return 1
    fi
    
    # Use diff to check if files are identical
    if diff -q "$file1" "$file2" > /dev/null 2>&1; then
        return 0  # Files are identical
    else
        return 1  # Files differ
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [CTSM_ROOT_PATH] [OPTIONS]"
    echo ""
    echo "Deploy custom CTSM modifications to a CTSM installation."
    echo ""
    echo "Arguments:"
    echo "  CTSM_ROOT_PATH    Path to the CTSM root directory (default: current directory)"
    echo ""
    echo "Options:"
    echo "  --log             Enable logging with git-style diffs"
    echo "  --logs-dir DIR    Specify logs directory (default: ./logs)"
    echo "  --force           Force copy all files, even if unchanged"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Directory Structure:"
    echo "  This script expects the following structure:"
    echo "    $(dirname $0)/"
    echo "    ├── deploy.sh (this script)"
    echo "    └── files.to.deploy/ (your CTSM modifications)"
    echo "  And creates:"
    echo "    \$PWD/logs/ (if logging enabled)"
    echo ""
    echo "Behavior:"
    echo "  - By default, only copies files that have changed"
    echo "  - Use --force to copy all files regardless of changes"
    echo "  - Files are compared using diff before copying"
    echo "  - Logs directory defaults to current working directory"
    echo "  - CTSM root defaults to current working directory"
    echo ""
    echo "Examples:"
    echo "  $0                            # Deploy to current dir, only changed files"
    echo "  $0 /path/to/ctsm              # Deploy to specific path"
    echo "  $0 --log                      # Deploy to current dir with logging"
    echo "  $0 /path/to/ctsm --log        # Deploy with logging"
    echo "  $0 --logs-dir /tmp/logs --log # Use custom logs directory"
    echo "  $0 /path/to/ctsm --force      # Force copy all files"
    echo ""
    echo "For more information, see the README or documentation."
    exit 0
}

# Check for help flag first
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
fi

# Parse command line arguments
# First check if first argument is a path or an option
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    CTSMROOT="$1"
    shift
else
    # Default to current directory if no path provided
    CTSMROOT="$WORKING_DIR"
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log)
            ENABLE_LOG=true
            shift
            ;;
        --logs-dir)
            if [ -n "$2" ] && [[ "$2" != --* ]]; then
                LOGS_DIR="$2"
                shift 2
            else
                echo "Error: --logs-dir requires a directory path"
                exit 1
            fi
            ;;
        --force)
            FORCE_COPY=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            usage
            ;;
    esac
done

# Validate CTSM root directory
if [ ! -d "$CTSMROOT" ]; then
    print_error "CTSM root directory does not exist: $CTSMROOT"
    exit 1
fi

# Validate deploy source directory
if [ ! -d "$DEPLOY_SOURCE_DIR" ]; then
    print_error "Deploy source directory does not exist: $DEPLOY_SOURCE_DIR"
    print_error "Please ensure 'files.to.deploy/' directory exists with your custom files"
    exit 1
fi

# Check if there are any files to deploy
if [ -z "$(find "$DEPLOY_SOURCE_DIR" -type f 2>/dev/null)" ]; then
    print_warn "No files found in $DEPLOY_SOURCE_DIR"
    exit 0
fi

# Convert to absolute path
CTSMROOT="$(cd "$CTSMROOT" && pwd)"

print_info "CTSM Root: $CTSMROOT"
print_info "Deploy Source: $DEPLOY_SOURCE_DIR"
print_info "Logs Directory: $LOGS_DIR"
if [ "$FORCE_COPY" = true ]; then
    print_info "Force mode: All files will be copied"
else
    print_info "Smart mode: Only changed files will be copied"
fi

# Setup logging if enabled
if [ "$ENABLE_LOG" = true ]; then
    # Create logs directory if it doesn't exist
    mkdir -p "$LOGS_DIR"
    
    # Generate timestamp for log file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOGS_DIR}/deploy_${TIMESTAMP}.log"
    
    print_info "Logging enabled: $LOG_FILE"
    
    # Write log header
    {
        echo "========================================"
        echo "CTSM Custom Modifications Deployment Log"
        echo "========================================"
        echo "Timestamp: $(date)"
        echo "Script Location: $SCRIPT_DIR"
        echo "Deploy Source: $DEPLOY_SOURCE_DIR"
        echo "CTSM Root: $CTSMROOT"
        echo "Force Copy: $FORCE_COPY"
        echo ""
        
        # Get git information from CTSM repository
        echo "CTSM Repository Information:"
        echo "----------------------------"
        if [ -d "$CTSMROOT/.git" ]; then
            cd "$CTSMROOT"
            echo "Branch/Tag: $(git describe --tags --exact-match 2>/dev/null || git rev-parse --abbrev-ref HEAD)"
            echo "Commit: $(git rev-parse HEAD)"
            echo "Status: $(git status --short | wc -l) uncommitted changes"
            cd - > /dev/null
        else
            echo "Not a git repository"
        fi
        echo ""
        echo "Deployment Details:"
        echo "----------------------------"
    } > "$LOG_FILE"
fi

# Function to deploy a file
deploy_file() {
    local src_file="$1"
    local dest_file="$2"
    local rel_path="$3"
    
    # Check if files are identical (unless force mode is enabled)
    if [ "$FORCE_COPY" = false ] && files_are_identical "$src_file" "$dest_file"; then
        print_skip "Unchanged: $rel_path"
        FILES_SKIPPED=$((FILES_SKIPPED + 1))
        
        # Log skipped file if logging is enabled
        if [ "$ENABLE_LOG" = true ]; then
            echo "--- Skipped (unchanged): $rel_path" >> "$LOG_FILE"
        fi
        return
    fi
    
    # Check if destination file exists (for determining if it's new or modified)
    if [ -f "$dest_file" ]; then
        print_info "Modifying: $rel_path"
        FILES_MODIFIED=$((FILES_MODIFIED + 1))
        
        # Generate diff if logging is enabled
        if [ "$ENABLE_LOG" = true ]; then
            {
                echo ""
                echo "--- Modified: $rel_path"
                echo "diff --git a/$rel_path b/$rel_path"
                diff -u "$dest_file" "$src_file" || true
            } >> "$LOG_FILE"
        fi
    else
        print_info "Adding: $rel_path"
        FILES_ADDED=$((FILES_ADDED + 1))
        
        # Log new file if logging is enabled
        if [ "$ENABLE_LOG" = true ]; then
            {
                echo ""
                echo "--- Added: $rel_path"
            } >> "$LOG_FILE"
        fi
    fi
    
    # Create destination directory if it doesn't exist
    dest_dir=$(dirname "$dest_file")
    if [ ! -d "$dest_dir" ]; then
        mkdir -p "$dest_dir"
        print_info "Created directory: ${dest_dir#$CTSMROOT/}"
    fi
    
    # Copy the file
    cp "$src_file" "$dest_file"
    FILES_PROCESSED+=("$rel_path")
}

# Main deployment logic
print_info "Starting deployment..."

# Counter for total files to process (for progress indication)
TOTAL_FILES=$(find "$DEPLOY_SOURCE_DIR" -type f | wc -l)
CURRENT_FILE=0

# Recursively find and deploy all files from files.to.deploy directory
# Using process substitution to avoid subshell issues
while IFS= read -r -d '' file; do
    # Get relative path from files.to.deploy directory
    rel_path="${file#$DEPLOY_SOURCE_DIR/}"
    
    src_file="$file"
    dest_file="$CTSMROOT/$rel_path"
    
    CURRENT_FILE=$((CURRENT_FILE + 1))
    echo -ne "\rProgress: $CURRENT_FILE/$TOTAL_FILES files processed"
    
    deploy_file "$src_file" "$dest_file" "$rel_path"
done < <(find "$DEPLOY_SOURCE_DIR" -type f -print0)
echo ""  # New line after progress indicator

# Write summary to log
if [ "$ENABLE_LOG" = true ]; then
    {
        echo ""
        echo "========================================"
        echo "Deployment Summary:"
        echo "========================================"
        echo "Files Added: $FILES_ADDED"
        echo "Files Modified: $FILES_MODIFIED"
        echo "Files Skipped (unchanged): $FILES_SKIPPED"
        echo "Total Files Processed: $((FILES_ADDED + FILES_MODIFIED))"
        echo "Total Files Examined: $TOTAL_FILES"
        echo ""
        if [ ${#FILES_PROCESSED[@]} -gt 0 ]; then
            echo "Files Changed:"
            for file in "${FILES_PROCESSED[@]}"; do
                echo "  - $file"
            done
        fi
        echo ""
        echo "Deployment completed at: $(date)"
    } >> "$LOG_FILE"
fi

# Print summary
echo ""
print_info "========================================"
print_info "Deployment Complete!"
print_info "========================================"
echo "Files Added:     $FILES_ADDED"
echo "Files Modified:  $FILES_MODIFIED"
echo "Files Skipped:   $FILES_SKIPPED"
echo "-------------------"
echo "Total Changed:   $((FILES_ADDED + FILES_MODIFIED))"
echo "Total Examined:  $TOTAL_FILES"

# Special message if nothing was changed
if [ $((FILES_ADDED + FILES_MODIFIED)) -eq 0 ]; then
    echo ""
    print_info "No changes were needed. All files are up to date!"
fi

if [ "$ENABLE_LOG" = true ]; then
    echo ""
    print_info "Log file created: $LOG_FILE"
fi

echo ""
if [ $((FILES_ADDED + FILES_MODIFIED)) -gt 0 ]; then
    print_info "Deployment successful! $((FILES_ADDED + FILES_MODIFIED)) file(s) updated."
else
    print_info "No deployment needed - all files already up to date."
fi
