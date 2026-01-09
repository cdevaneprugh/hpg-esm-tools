#!/usr/bin/env python3
"""
Plot last 20 years time series from annual binned h1 data

Creates 2-panel plot:
- Top: Binned by direction (North/East/South/West)
- Bottom: Binned by elevation (Outlet/Lower/Upper/Ridge)

Uses weighted averaging based on column area fractions (cols1d_wtgcell).
Hillslope binning excludes stream column (16).
Gridcell average includes all 17 columns (including stream).

Usage:
    python3 plot_timeseries_last20.py <input_file> <output_file> <variable>

Example:
    python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/GPP_last20.png GPP
"""

import sys
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries_last20(input_file, output_file, variable, n_years=20):
    """Plot last N years of column data binned by direction and elevation"""

    # Load dataset
    ds = xr.open_dataset(input_file, decode_times=False)

    # Check if variable exists
    if variable not in ds:
        print(f"ERROR: Variable '{variable}' not found in {input_file}")
        print(f"Available variables: {list(ds.data_vars)}")
        ds.close()
        sys.exit(1)

    # Extract variable (should have time and column dimensions)
    var_data = ds[variable].values

    # Check dimensions
    if var_data.ndim != 2:
        print(f"ERROR: Expected 2D variable (time, column), got shape {var_data.shape}")
        ds.close()
        sys.exit(1)

    # Get metadata
    units = ds[variable].attrs.get('units', '')

    # Extract last N years
    total_years = var_data.shape[0]
    start_idx = max(0, total_years - n_years)
    var_data = var_data[start_idx:total_years, :]

    # Get column weights
    weights_all = ds['cols1d_wtgcell'].values[0:17]  # All columns including stream
    weights = weights_all[0:16]  # Hillslope columns only (exclude stream)

    # Calculate gridcell average (all 17 columns including stream)
    gridcell_avg = (var_data[:, 0:17] * weights_all).sum(axis=1)

    # Define column groups
    direction_cols = {
        'North': [0, 1, 2, 3],
        'East':  [4, 5, 6, 7],
        'South': [8, 9, 10, 11],
        'West':  [12, 13, 14, 15]
    }

    elevation_cols = {
        'Outlet': [0, 4, 8, 12],
        'Lower':  [1, 5, 9, 13],
        'Upper':  [2, 6, 10, 14],
        'Ridge':  [3, 7, 11, 15]
    }

    # Compute weighted averages for each direction
    direction_data = {}
    for name, cols in direction_cols.items():
        group_weights = weights[cols]
        renorm_weights = group_weights / group_weights.sum()
        direction_data[name] = (var_data[:, cols] * renorm_weights).sum(axis=1)

    # Compute weighted averages for each elevation
    elevation_data = {}
    for name, cols in elevation_cols.items():
        group_weights = weights[cols]
        renorm_weights = group_weights / group_weights.sum()
        elevation_data[name] = (var_data[:, cols] * renorm_weights).sum(axis=1)

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    time = np.arange(len(var_data))
    ylabel = f"{variable} ({units})" if units else variable

    # Top panel: Direction
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot gridcell average first (so colored lines are on top)
    ax1.plot(time, gridcell_avg, linewidth=2, color='black',
            linestyle='--', alpha=0.6, label='Gridcell Avg', zorder=1)

    for i, (name, data) in enumerate(direction_data.items()):
        ax1.plot(time, data, linewidth=2, color=colors[i],
                alpha=0.85, label=name, zorder=2)

    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(f"{variable} (binned annually) - Last {n_years} Years\nBy Direction",
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)

    # Bottom panel: Elevation
    # Plot gridcell average first (so colored lines are on top)
    ax2.plot(time, gridcell_avg, linewidth=2, color='black',
            linestyle='--', alpha=0.6, label='Gridcell Avg', zorder=1)

    for i, (name, data) in enumerate(elevation_data.items()):
        ax2.plot(time, data, linewidth=2, color=colors[i],
                alpha=0.85, label=name, zorder=2)

    ax2.set_xlabel(f'Year (Last {n_years})', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title('By Elevation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)

    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    ds.close()

if __name__ == '__main__':
    # Check arguments
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    variable = sys.argv[3]

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Variable: {variable}")
    print()

    plot_timeseries_last20(input_file, output_file, variable)
