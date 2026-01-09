#!/usr/bin/env python3
"""
Plot column areas as bar chart
Uses h1 concat stream file
"""

import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_column_areas(nc_file, output_file):
    """
    Bar chart of hillslope_area by column

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
    areas = ds['hillslope_area'].values[0:16]

    # Column indices are just 0-15 (already sorted)
    column_idx = np.arange(16)

    # Aspect names
    aspect_names = {1: 'North', 2: 'East', 3: 'South', 4: 'West'}

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get matplotlib's default color cycle and assign by hillslope
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']

    # Map hillslope index to color (1->color[0], 2->color[1], etc.)
    bar_colors = [default_colors[h - 1] for h in hillslope_idx]

    # Plot bars
    x_pos = np.arange(len(areas))
    bars = ax.bar(x_pos, areas, color=bar_colors, edgecolor='black', linewidth=1)

    # Add percentage labels on bars
    total_area = areas.sum() # sum all areas
    for i, (bar, area) in enumerate(zip(bars, areas)):
        height = bar.get_height()
        pct = (area / total_area) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{pct:.2f}%',
                ha='center', va='bottom', fontsize=8)

    # X-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Col {i}' for i in range(len(areas))], rotation=0, fontsize=9)

    # Add vertical lines to separate hillslopes
    for i in [3.5, 7.5, 11.5]:
        ax.axvline(i, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add hillslope labels at top
    ax.text(1.5, ax.get_ylim()[1] * 0.95, 'North', ha='center', fontsize=11,
            fontweight='bold')#, color=colors[1])
    ax.text(5.5, ax.get_ylim()[1] * 0.95, 'East', ha='center', fontsize=11,
            fontweight='bold')#, color=colors[2])
    ax.text(9.5, ax.get_ylim()[1] * 0.95, 'South', ha='center', fontsize=11,
            fontweight='bold')#, color=colors[3])
    ax.text(13.5, ax.get_ylim()[1] * 0.95, 'West', ha='center', fontsize=11,
            fontweight='bold')#, color=colors[4])

    # Labels
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Column Area (m²)', fontsize=12)
    ax.set_title('OSBS Hillslope Column Areas',
                 fontsize=13, fontweight='bold')

    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    ds.close()

if __name__ == '__main__':
    # Default parameters
    nc_file = './data/combined_h1.nc'
    output_file = './plots/column_areas.png'

    # Check command line arguments
    if len(sys.argv) >= 2:
        nc_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    print(f"Input file: {nc_file}")
    print(f"Output file: {output_file}")

    # Generate plot
    plot_column_areas(nc_file, output_file)
