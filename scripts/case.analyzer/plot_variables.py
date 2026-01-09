#!/usr/bin/env python3
"""
Script: plot_variables.py
Purpose: Generate time series plots from concatenated CTSM NetCDF files
Usage: ./plot_variables.py <CONCAT_FILE> <OUTPUT_DIR> <VAR1> [VAR2 ...]
"""

import sys
from pathlib import Path
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (faster)
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Main execution function."""
    if len(sys.argv) < 4:
        print("Usage: plot_variables.py <CONCAT_FILE> <OUTPUT_DIR> <VAR1> [VAR2 ...]", file=sys.stderr)
        sys.exit(2)

    concat_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    variables = sys.argv[3:]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset once
    ds = xr.open_dataset(concat_file, decode_times=False)

    # Plot each variable
    for var in variables:
        var_data = ds[var]

        # Squeeze all non-time dimensions (take first index for spatial/vertical)
        for dim in var_data.dims:
            if dim != 'time' and var_data.sizes[dim] > 0:
                var_data = var_data.isel({dim: 0})

        # Extract 1D arrays
        time_index = np.arange(len(var_data))
        values = var_data.values
        units = var_data.attrs.get('units', '')

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_index, values, linewidth=2, color='#1f77b4')
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel(f"{var} ({units})" if units else var, fontsize=12)
        ax.set_title(var, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Save and close
        output_png = output_dir / f"{var}.png"
        fig.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Plotted {var}")

    ds.close()


if __name__ == '__main__':
    main()
