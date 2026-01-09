#!/usr/bin/env python3
"""
Plot full time series from 20-year binned h0 data (gridcell average)

Usage:
    python3 plot_timeseries_full.py <input_file> <output_file> <variable>

Example:
    python3 plot_timeseries_full.py data/combined_h0_20yr.nc plots/GPP_full.png GPP
"""

import sys
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries_full(input_file, output_file, variable):
    """Plot full time series for gridcell-averaged variable"""

    # Load dataset
    ds = xr.open_dataset(input_file, decode_times=False)

    # Check if variable exists
    if variable not in ds:
        print(f"ERROR: Variable '{variable}' not found in {input_file}")
        print(f"Available variables: {list(ds.data_vars)}")
        ds.close()
        sys.exit(1)

    # Extract variable
    var_data = ds[variable]

    # Squeeze any non-time dimensions (should already be gridcell-averaged)
    for dim in var_data.dims:
        if dim != 'time' and var_data.sizes[dim] == 1:
            var_data = var_data.squeeze(dim)

    # Extract metadata
    units = var_data.attrs.get('units', '')

    # Get simulation years from mcdate
    years = ds['mcdate'].values // 10000
    values = var_data.values

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(years, values, linewidth=2.5, color='#1f77b4')

    # Labels and formatting
    ylabel = f"{variable} ({units})" if units else variable
    ax.set_xlabel('Simulation Year', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{variable} (20 year bins)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set x-axis ticks every 100 years starting at 0
    year_min = 0
    year_max = int(np.ceil(years[-1] / 100) * 100)
    ax.set_xticks(np.arange(year_min, year_max + 1, 100))
    ax.set_xlim(0, year_max)

    # Save
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)

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

    plot_timeseries_full(input_file, output_file, variable)
