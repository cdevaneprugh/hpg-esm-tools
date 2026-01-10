#!/usr/bin/env python3
"""
plot_timeseries_full.py - Full simulation time series plot

PURPOSE:
    Plot complete simulation time series from binned h0 (gridcell-averaged) data.
    Designed for long-term trend visualization with 20-year binned data.

USAGE:
    python3 plot_timeseries_full.py <input_file> <output_file> <variable>
    python3 plot_timeseries_full.py -h

ARGUMENTS:
    input_file      Binned h0 NetCDF file (e.g., combined_h0_20yr.nc)
    output_file     Output PNG filename
    variable        Variable name to plot (e.g., GPP, TOTECOSYSC, ZWT)

EXAMPLE:
    python3 plot_timeseries_full.py data/combined_h0_20yr.nc plots/GPP_full.png GPP

NOTES:
    - Input should be binned gridcell data (h0 stream)
    - X-axis shows simulation years extracted from mcdate variable
    - Units are read from variable attributes
    - Works best with 20-year binned data for multi-century simulations
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import argparse
import xarray as xr
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless systems
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Main plotting function
# =============================================================================


def plot_timeseries_full(input_file: str, output_file: str, variable: str) -> None:
    """
    Plot full time series for gridcell-averaged variable.

    Parameters
    ----------
    input_file : str
        Path to binned h0 NetCDF file
    output_file : str
        Path for output PNG file
    variable : str
        Name of variable to plot

    Returns
    -------
    None
        Saves plot to output_file
    """

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)

    # Validate variable exists
    if variable not in ds:
        print(f"ERROR: Variable '{variable}' not found in {input_file}")
        print(f"Available variables: {list(ds.data_vars)}")
        ds.close()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Extract variable data
    # -------------------------------------------------------------------------
    var_data = ds[variable]

    # Squeeze any singleton dimensions (e.g., single gridcell)
    for dim in var_data.dims:
        if dim != "time" and var_data.sizes[dim] == 1:
            var_data = var_data.squeeze(dim)

    # Get metadata
    units = var_data.attrs.get("units", "")

    # Extract years from mcdate (YYYYMMDD format)
    years = ds["mcdate"].values // 10000
    values = var_data.values

    ds.close()

    # -------------------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(years, values, linewidth=2.5, color="#1f77b4")

    # -------------------------------------------------------------------------
    # Axis labels and formatting
    # -------------------------------------------------------------------------
    ylabel = f"{variable} ({units})" if units else variable
    ax.set_xlabel("Simulation Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{variable} Time Series (20-year bins)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Set x-axis ticks at 100-year intervals
    year_min = 0
    year_max = int(np.ceil(years[-1] / 100) * 100)
    ax.set_xticks(np.arange(year_min, year_max + 1, 100))
    ax.set_xlim(0, year_max)

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot full simulation time series from binned data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h0_20yr.nc plots/GPP_full.png GPP
  %(prog)s data/combined_h0_20yr.nc plots/TOTECOSYSC.png TOTECOSYSC
        """,
    )
    parser.add_argument("input_file", help="Binned h0 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument("variable", help="Variable name to plot")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Variable: {args.variable}")
    print()

    plot_timeseries_full(args.input_file, args.output_file, args.variable)


if __name__ == "__main__":
    main()
