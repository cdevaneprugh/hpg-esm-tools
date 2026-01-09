#!/usr/bin/env python3
"""
Parse SVN XML output and compare with local files.

This script reads SVN XML listing from stdin and compares it with local files
to determine which files need to be downloaded. It handles the complex path
mapping between SVN repository structure and local directory structure.

Usage: svn list -R --xml URL | python3 parse_svn_xml.py URL LOCAL_PATH OUTPUT_DIR

Arguments:
    URL: The SVN repository URL (e.g., https://svn.../trunk/inputdata/lnd/clm2/...)
    LOCAL_PATH: Base local directory where files should exist (e.g., inputdata)
    OUTPUT_DIR: Directory where output files will be written

Output files:
    all_files.txt: List of all files in format "URL|size_in_bytes"
    to_download.txt: List of URLs for files that need to be downloaded

Exit codes:
    0: Success
    1: Error (wrong arguments or parsing failure)
"""

import sys
import xml.etree.ElementTree as ET
import os

def main():
    # Validate command line arguments
    if len(sys.argv) != 4:
        print("Error: Expected 3 arguments: URL LOCAL_PATH OUTPUT_DIR", file=sys.stderr)
        sys.exit(1)
    
    # Extract arguments
    url = sys.argv[1]              # SVN repository URL
    local_path = sys.argv[2]       # Local base directory
    output_dir = sys.argv[3]       # Where to write output files
    
    try:
        # Parse XML from stdin (piped from svn list --xml)
        tree = ET.parse(sys.stdin)
        root = tree.getroot()
        
        # Initialize data structures
        all_files = []      # List of all files with their URLs and sizes
        to_download = []    # List of file URLs that need downloading
        
        # Initialize counters for statistics
        total_count = 0         # Total number of files in repository
        total_size = 0          # Total size of all files in bytes
        existing_count = 0      # Number of files that exist locally with correct size
        existing_size = 0       # Total size of existing files
        missing_count = 0       # Number of files that need downloading
        missing_size = 0        # Total size of files to download
        
        # Extract the repository subpath to properly construct local paths
        # Example: If URL is https://svn.../trunk/inputdata/lnd/clm2/surfdata/
        # We need to extract 'lnd/clm2/surfdata/' to build correct local paths
        url_parts = url.split('/trunk/inputdata/')
        if len(url_parts) > 1:
            # Everything after /trunk/inputdata/ is our subpath
            repo_subpath = url_parts[1]  # e.g., 'lnd/clm2/surfdata_esmf/NEON/'
        else:
            # URL doesn't contain /trunk/inputdata/, no subpath needed
            repo_subpath = ''
        
        # Process each file entry in the XML
        # XML structure: <entry kind="file"><name>...</name><size>...</size></entry>
        for entry in root.findall('.//entry[@kind="file"]'):
            # Extract file name from XML
            # This is relative to the repository URL (e.g., "16PFT/file.nc")
            name = entry.find('name').text
            
            # Extract file size (might be None for some repositories)
            size_elem = entry.find('size')
            size = int(size_elem.text) if size_elem is not None else 0
            
            # Build full URL for this file
            file_url = url + name
            
            # Store file info for all_files.txt
            all_files.append(f'{file_url}|{size}')
            
            # Update total statistics
            total_count += 1
            total_size += size
            
            # Construct the correct local file path
            # Example: local_path='inputdata', repo_subpath='lnd/clm2/', name='file.nc'
            # Result: 'inputdata/lnd/clm2/file.nc'
            local_file = os.path.join(local_path, repo_subpath, name)
            
            # Check if file exists locally and has correct size
            if os.path.exists(local_file):
                if size > 0:
                    # We have size information, so verify the local file size
                    local_size = os.path.getsize(local_file)
                    if local_size == size:
                        # File exists with correct size, no download needed
                        existing_count += 1
                        existing_size += size
                    else:
                        # File exists but size mismatch, need to re-download
                        to_download.append(file_url)
                        missing_count += 1
                        missing_size += size
                else:
                    # No size info available, assume file is OK if it exists
                    existing_count += 1
            else:
                # File doesn't exist locally, need to download
                to_download.append(file_url)
                missing_count += 1
                missing_size += size
            
            # Progress indicator for large repositories
            if total_count % 100 == 0:
                print(f'Processed {total_count} files...', file=sys.stderr)
        
        # Write all_files.txt - complete list of all files in repository
        # Format: URL|size_in_bytes (one per line)
        with open(f'{output_dir}/all_files.txt', 'w') as f:
            for line in all_files:
                f.write(line + '\n')
        
        # Write to_download.txt - just URLs of files that need downloading
        # Format: URL (one per line)
        with open(f'{output_dir}/to_download.txt', 'w') as f:
            for url in to_download:
                f.write(url + '\n')
        
        # Output statistics to stdout for the bash script to parse
        # Format: total_count|total_size|existing_count|existing_size|missing_count|missing_size
        # The bash script will read this with IFS='|' to extract all values
        print(f'{total_count}|{total_size}|{existing_count}|{existing_size}|{missing_count}|{missing_size}')
        
    except ET.ParseError as e:
        # Handle XML parsing errors (malformed XML, etc.)
        print(f'Error parsing XML: {e}', file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        # Handle file I/O errors (permissions, disk full, etc.)
        print(f'Error writing output files: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        print(f'Unexpected error: {e}', file=sys.stderr)
        sys.exit(1)

# Entry point - only run main() if script is executed directly
if __name__ == "__main__":
    main()
