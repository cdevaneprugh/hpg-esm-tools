#!/usr/bin/env python3
"""
Plot all 4 hillslope elevation and width profiles overlaid
Shows geometric similarity across aspects
Uses h1 concat stream file
"""

import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_overlay_profiles(nc_file, output_file):
    """
    Plot elevation and width profiles with all 4 hillslopes overlaid

    Parameters:
    -----------
    nc_file : str
        Path to h1 NetCDF file
    output_file : str
        Output PNG filename
    """

    # Open NetCDF file
    ds = xr.open_dataset(nc_file)

    # Extract data (only hillslope columns 0-15, exclude stream)
    hillslope_idx = ds['hillslope_index'].values[0:16]
    elevation = ds['hillslope_elev'].values[0:16]
    width = ds['hillslope_width'].values[0:16]
    distance = ds['hillslope_distance'].values[0:16]

    # Aspect names, colors, and markers
    aspects = {
        1: {'name': 'North', 'color': '#3498db'},
        2: {'name': 'East', 'color': '#e74c3c'},
        3: {'name': 'South', 'color': '#f39c12'},
        4: {'name': 'West', 'color': '#2ecc71'}
    }

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot each hillslope
    for hillslope_num in range(1, 5):
        # Filter for this hillslope
        mask = hillslope_idx == hillslope_num
        elev = elevation[mask]
        w = width[mask]
        dist = distance[mask]

        # Sort by distance (ridge to outlet)
        sort_idx = np.argsort(dist)[::-1]  # Descending (ridge first)
        elev = elev[sort_idx]
        w = w[sort_idx]
        dist = dist[sort_idx]

        # Get styling for this hillslope
        style = aspects[hillslope_num]

        # Plot elevation on left subplot
        ax1.plot(dist, elev,
                marker='o',
                #color=style['color'],
                linewidth=2,
                markersize=8,
                label=style['name'],
                alpha=0.85)

        # Plot width on right subplot
        ax2.plot(dist, w,
                marker='o',
                #color=style['color'],
                linewidth=2,
                markersize=8,
                label=style['name'],
                alpha=0.85)

    # Configure left subplot (elevation)
    ax1.set_xlabel('Distance from Stream (m)', fontsize=12)
    ax1.set_ylabel('Elevation above Stream (m)', fontsize=12)
    ax1.set_title('Elevation Profiles', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.set_ylim(0, elevation.max() * 1.15)

    # Configure right subplot (width)
    ax2.set_xlabel('Distance from Stream (m)', fontsize=12)
    ax2.set_ylabel('Hillslope Width (m)', fontsize=12)
    ax2.set_title('Width Profiles', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, width.max() * 1.15)

    # Overall title
    fig.suptitle('OSBS Hillslope Geometry: All Aspects Overlaid',
                 fontsize=14, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    ds.close()

if __name__ == '__main__':
    # Default parameters
    nc_file = './data/combined_h1.nc'
    output_file = './plots/elevation_width_overlay.png'

    # Check command line arguments
    if len(sys.argv) >= 2:
        nc_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    print(f"Input file: {nc_file}")
    print(f"Output file: {output_file}")

    # Generate plot
    plot_overlay_profiles(nc_file, output_file)
