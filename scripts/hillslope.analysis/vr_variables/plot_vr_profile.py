#!/usr/bin/env python3
"""
Plot vertically resolved variable depth profiles for North hillslope

Shows 4 "core sample" profiles - one at each hillslope position (Outlet, Lower,
Upper, Ridge). X-axis is variable value, Y-axis is depth. Uses most recent time step.

"""

import sys
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# North hillslope columns
NORTH_COLS = [0, 1, 2, 3]
POSITION_NAMES = ['Outlet', 'Lower', 'Upper', 'Ridge']


def plot_vr_profile(input_file, variable):

    # Load dataset
    ds = xr.open_dataset(input_file, decode_times=False)

    # Check if variable exists
    if variable not in ds:
        print(f"ERROR: Variable '{variable}' not found in {input_file}")
        print(f"Available variables: {list(ds.data_vars)}")
        ds.close()
        sys.exit(1)

    var_data = ds[variable]

    # Identify vertical dimension
    vert_dims = {'levsoi', 'levgrnd', 'levdcmp'}
    var_dims = set(var_data.dims)
    vert_dim = var_dims & vert_dims

    if not vert_dim:
        print(f"ERROR: Variable '{variable}' has no vertical dimension")
        print(f"Dimensions: {var_data.dims}")
        print(f"Expected one of: {vert_dims}")
        ds.close()
        sys.exit(1)

    vert_dim = vert_dim.pop()

    # Get coordinates
    depths = ds[vert_dim].values
    distances = ds['hillslope_distance'].values[NORTH_COLS]
    mcdate = ds['mcdate'].values

    # Get most recent time step
    last_time_idx = -1
    year = mcdate[last_time_idx] // 10000

    # Extract data for North hillslope at last time step
    # Shape: (levels, columns)
    data = var_data.isel(time=last_time_idx, column=NORTH_COLS).values

    # Get metadata
    units = var_data.attrs.get('units', '')
    long_name = var_data.attrs.get('long_name', variable)

    ds.close()

    # Sort by distance (outlet to ridge)
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    data = data[:, sort_idx]
    sorted_positions = [POSITION_NAMES[i] for i in sort_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 10))

    # Plot each position as a vertical profile
    for i, (name, dist) in enumerate(zip(sorted_positions, distances)):
        values = data[:, i]
        label = f"{name} ({dist:.0f}m)"
        ax.plot(values, depths, 'o-', linewidth=2,
                markersize=5, label=label)

    # Invert y-axis (depth increases downward, surface at top)
    ax.invert_yaxis()

    # Labels
    xlabel = f"{variable} ({units})" if units else variable
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'{long_name}\nNorth Hillslope - Year {year} (20-yr avg)',
                 fontsize=13, fontweight='bold')

    # Log scale x-axis
    ax.set_xscale('log')

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, title='Position (dist from stream)')

    # Get script directory for output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f'{variable}_north_profile.png')

    # Save
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    variable = sys.argv[2]


    plot_vr_profile(input_file, variable)
