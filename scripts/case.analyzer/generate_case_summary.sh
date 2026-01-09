#!/bin/bash
# Script: generate_case_summary.sh
# Purpose: Extract case XML configuration and catalog output files
# Usage: ./generate_case_summary.sh <CASEDIR> <OUTPUT_FILE|-> <XML_VARS_CSV>

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "ERROR: Usage: $0 <CASEDIR> <OUTPUT_FILE|-> <XML_VARS_CSV>" >&2
    exit 1
fi

CASEDIR="$1"
OUTPUT_FILE="$2"
XML_VARS_CSV="$3"

# Convert CSV to array
IFS=',' read -ra XML_VARS <<< "$XML_VARS_CSV"

# Global associative arrays
declare -A xml_values
declare -A hist_stream_counts
declare -A hist_stream_first
declare -A hist_stream_last
hist_dir=""
restart_count=0

extract_xml_variables() {
    cd "$CASEDIR"

    # Batch query all XML variables
    local batch_output
    batch_output=$(./xmlquery "${XML_VARS[@]}" 2>/dev/null)

    # Parse output
    while IFS= read -r line; do
        [[ -z "$line" ]] || [[ "$line" =~ ^Results\ in\ group ]] && continue

        if [[ "$line" =~ ^[[:space:]]*([A-Z_]+):[[:space:]]*(.*) ]]; then
            xml_values["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
        fi
    done <<< "$batch_output"

    # Set N/A for missing variables
    for var in "${XML_VARS[@]}"; do
        [[ -z "${xml_values[$var]:-}" ]] && xml_values["$var"]="N/A"
    done

    cd - >/dev/null
}

catalog_history_files() {
    hist_dir="${xml_values[DOUT_S_ROOT]}/lnd/hist"

    for stream in h0 h1 h2 h3 h4 h5; do
        # Count files first to decide on strategy
        local count=$(find "$hist_dir" -maxdepth 1 -name "*.clm2.$stream.*.nc" 2>/dev/null | wc -l)

        if [ "$count" -gt 0 ]; then
            hist_stream_counts["$stream"]=$count

            # For large file counts, just show count without first/last (avoid memory issues)
            if [ "$count" -le 1000 ]; then
                local first=$(find "$hist_dir" -maxdepth 1 -name "*.clm2.$stream.*.nc" -printf "%f\n" 2>/dev/null | sort | head -n 1)
                local last=$(find "$hist_dir" -maxdepth 1 -name "*.clm2.$stream.*.nc" -printf "%f\n" 2>/dev/null | sort | tail -n 1)
                hist_stream_first["$stream"]="$first"
                hist_stream_last["$stream"]="$last"
            else
                hist_stream_first["$stream"]="(too many files)"
                hist_stream_last["$stream"]="(too many files)"
            fi
        fi
    done
}

catalog_restart_files() {
    local run_dir="${xml_values[RUNDIR]}"
    restart_count=$(ls "$run_dir"/*.r.*.nc 2>/dev/null | wc -l)
}

print_summary() {
    cat <<EOF
=================================
CASE ANALYSIS SUMMARY
=================================
Case: ${xml_values[CASE]}
Analysis Date: $(date "+%Y-%m-%d %H:%M:%S")

CASE CONFIGURATION
------------------
EOF

    for var in "${XML_VARS[@]}"; do
        echo "$var: ${xml_values[$var]}"
    done

    # Only print file catalog if it was generated
    if [ -n "${xml_values[DOUT_S_ROOT]:-}" ] && [ "${xml_values[DOUT_S_ROOT]}" != "N/A" ]; then
        cat <<EOF

OUTPUT FILES
------------
CLM History Files: $hist_dir
EOF

        if [ -n "${!hist_stream_counts[*]}" ]; then
            for stream in h0 h1 h2 h3 h4 h5; do
                if [ -n "${hist_stream_counts[$stream]:-}" ]; then
                    echo "  Stream $stream: ${hist_stream_counts[$stream]} files"
                    echo "    First: ${hist_stream_first[$stream]}"
                    echo "    Last: ${hist_stream_last[$stream]}"
                fi
            done
        else
            echo "  No history files found"
        fi
    fi

    if [ -n "${xml_values[RUNDIR]:-}" ] && [ "${xml_values[RUNDIR]}" != "N/A" ]; then
        cat <<EOF

Restart Files: ${xml_values[RUNDIR]}
  Count: $restart_count files

EOF
    fi
}

# Main execution
extract_xml_variables

# Catalog files only if relevant variables were queried
if [ -n "${xml_values[DOUT_S_ROOT]:-}" ] && [ "${xml_values[DOUT_S_ROOT]}" != "N/A" ]; then
    catalog_history_files
fi

if [ -n "${xml_values[RUNDIR]:-}" ] && [ "${xml_values[RUNDIR]}" != "N/A" ]; then
    catalog_restart_files
fi

if [ "$OUTPUT_FILE" = "-" ]; then
    print_summary
else
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    print_summary > "$OUTPUT_FILE"
    echo "Summary generated: $OUTPUT_FILE" >&2
fi
