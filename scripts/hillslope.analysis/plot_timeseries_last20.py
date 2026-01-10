#!/usr/bin/env python3
"""
plot_timeseries_last20.py - Recent period time series by hillslope groups

PURPOSE:
    Plot last N years of column-level data, binned by direction (aspect) and
    elevation position. Shows how different hillslope components contribute
    to the gridcell average.

USAGE:
    python3 plot_timeseries_last20.py <input_file> <output_file> <variable> [--years=N]
    python3 plot_timeseries_last20.py -h

ARGUMENTS:
    input_file      Annual binned h1 NetCDF file (e.g., combined_h1_1yr.nc)
    output_file     Output PNG filename
    variable        Variable name to plot (e.g., GPP, ZWT, TOTECOSYSC)
    --years=N       Number of recent years to plot (default: 20)

EXAMPLE:
    python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/GPP_last20.png GPP
    python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/ZWT.png ZWT --years=50

OUTPUT:
    2-panel figure:
    - Top: Variable binned by direction (North/East/South/West)
    - Bottom: Variable binned by elevation (Outlet/Lower/Upper/Ridge)

NOTES:
    - Uses weighted averaging based on column area fractions (cols1d_wtgcell)
    - Hillslope binning uses columns 0-15 (excludes stream column 16)
    - Gridcell average includes all 17 columns (including stream)
    - Weights are renormalized within each group for proper averaging
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
# Constants
# =============================================================================

# Column groupings for hillslope analysis
# Each hillslope has 4 columns: Outlet (0), Lower (1), Upper (2), Ridge (3)
DIRECTION_COLS = {
    "North": [0, 1, 2, 3],
    "East": [4, 5, 6, 7],
    "South": [8, 9, 10, 11],
    "West": [12, 13, 14, 15],
}

ELEVATION_COLS = {
    "Outlet": [0, 4, 8, 12],
    "Lower": [1, 5, 9, 13],
    "Upper": [2, 6, 10, 14],
    "Ridge": [3, 7, 11, 15],
}

# =============================================================================
# Main plotting function
# =============================================================================


def plot_timeseries_last20(
    input_file: str, output_file: str, variable: str, n_years: int = 20
) -> None:
    """
    Plot last N years of column data binned by direction and elevation.

    Parameters
    ----------
    input_file : str
        Path to annual binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    variable : str
        Name of variable to plot
    n_years : int
        Number of recent years to plot (default: 20)

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
    var_data = ds[variable].values

    # Validate dimensions (should be time x column)
    if var_data.ndim != 2:
        print(f"ERROR: Expected 2D variable (time, column), got shape {var_data.shape}")
        ds.close()
        sys.exit(1)

    # Get metadata
    units = ds[variable].attrs.get("units", "")

    # -------------------------------------------------------------------------
    # Extract last N years
    # -------------------------------------------------------------------------
    total_years = var_data.shape[0]
    start_idx = max(0, total_years - n_years)
    var_data = var_data[start_idx:total_years, :]
    actual_years = var_data.shape[0]

    # -------------------------------------------------------------------------
    # Get column weights for area-weighted averaging
    # -------------------------------------------------------------------------
    # All 17 columns (including stream) for gridcell average
    weights_all = ds["cols1d_wtgcell"].values[0:17]
    # Hillslope columns only (exclude stream) for grouped averages
    weights = weights_all[0:16]

    ds.close()

    # -------------------------------------------------------------------------
    # Calculate gridcell average (all 17 columns including stream)
    # -------------------------------------------------------------------------
    gridcell_avg = (var_data[:, 0:17] * weights_all).sum(axis=1)

    # -------------------------------------------------------------------------
    # Calculate weighted averages for each direction group
    # -------------------------------------------------------------------------
    direction_data = {}
    for name, cols in DIRECTION_COLS.items():
        group_weights = weights[cols]
        # Renormalize weights so they sum to 1 within the group
        renorm_weights = group_weights / group_weights.sum()
        direction_data[name] = (var_data[:, cols] * renorm_weights).sum(axis=1)

    # -------------------------------------------------------------------------
    # Calculate weighted averages for each elevation group
    # -------------------------------------------------------------------------
    elevation_data = {}
    for name, cols in ELEVATION_COLS.items():
        group_weights = weights[cols]
        renorm_weights = group_weights / group_weights.sum()
        elevation_data[name] = (var_data[:, cols] * renorm_weights).sum(axis=1)

    # -------------------------------------------------------------------------
    # Create 2-panel figure
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    time = np.arange(actual_years)
    ylabel = f"{variable} ({units})" if units else variable
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # -------------------------------------------------------------------------
    # Top panel: By direction (aspect)
    # -------------------------------------------------------------------------
    # Plot gridcell average first (underneath colored lines)
    ax1.plot(
        time,
        gridcell_avg,
        linewidth=2,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Gridcell Avg",
        zorder=1,
    )

    # Plot each direction
    for i, (name, data) in enumerate(direction_data.items()):
        ax1.plot(
            time, data, linewidth=2, color=colors[i], alpha=0.85, label=name, zorder=2
        )

    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(
        f"{variable} (Annual) - Last {actual_years} Years\nBy Direction",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="best", fontsize=10)

    # -------------------------------------------------------------------------
    # Bottom panel: By elevation position
    # -------------------------------------------------------------------------
    ax2.plot(
        time,
        gridcell_avg,
        linewidth=2,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Gridcell Avg",
        zorder=1,
    )

    for i, (name, data) in enumerate(elevation_data.items()):
        ax2.plot(
            time, data, linewidth=2, color=colors[i], alpha=0.85, label=name, zorder=2
        )

    ax2.set_xlabel(f"Year (Last {actual_years})", fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title("By Elevation Position", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="best", fontsize=10)

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot recent time series binned by hillslope groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_1yr.nc plots/GPP_last20.png GPP
  %(prog)s data/combined_h1_1yr.nc plots/ZWT_last50.png ZWT --years=50
        """,
    )
    parser.add_argument("input_file", help="Annual binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument("variable", help="Variable name to plot")
    parser.add_argument(
        "--years",
        type=int,
        default=20,
        help="Number of recent years to plot (default: 20)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Variable: {args.variable}")
    print(f"Years:    {args.years}")
    print()

    plot_timeseries_last20(args.input_file, args.output_file, args.variable, args.years)


if __name__ == "__main__":
    main()
