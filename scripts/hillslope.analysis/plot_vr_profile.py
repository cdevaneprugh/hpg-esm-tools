#!/usr/bin/env python3
"""
plot_vr_profile.py - Vertically resolved variable depth profiles

PURPOSE:
    Plot depth profiles of vertically resolved variables (soil temperature,
    moisture, carbon, etc.) as "core samples" at each hillslope position.
    X-axis shows variable values, Y-axis shows depth below surface.

USAGE:
    python3 plot_vr_profile.py <input_file> <output_file> <variable> [--hillslope=NAME]
    python3 plot_vr_profile.py -h

ARGUMENTS:
    input_file      20-year binned h1 NetCDF file (e.g., combined_h1_20yr.nc)
    output_file     Output PNG filename
    variable        Vertically resolved variable name (e.g., TSOI, H2OSOI_LIQ)
    --hillslope=N   Hillslope aspect: North, East, South, West (default: North)

EXAMPLE:
    python3 plot_vr_profile.py data/combined_h1_20yr.nc plots/TSOI_profile.png TSOI
    python3 plot_vr_profile.py data/combined_h1_20yr.nc plots/SOILC_south.png SOILLIQ --hillslope=South

OUTPUT:
    Single panel with 4 vertical profiles (one per hillslope position).
    Uses most recent time step from input file.

NOTES:
    - Variable must have a vertical dimension (levsoi, levgrnd, or levdcmp)
    - X-axis uses log scale for better visualization of soil profiles
    - Depth increases downward (inverted Y-axis)
    - Each hillslope has 4 columns: Outlet, Lower, Upper, Ridge
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

# Hillslope index mapping and column ranges
HILLSLOPE_INFO = {
    "North": {"index": 1, "cols": [0, 1, 2, 3]},
    "East": {"index": 2, "cols": [4, 5, 6, 7]},
    "South": {"index": 3, "cols": [8, 9, 10, 11]},
    "West": {"index": 4, "cols": [12, 13, 14, 15]},
}

# Column position names (ordered by distance from stream)
POSITION_NAMES = ["Outlet", "Lower", "Upper", "Ridge"]

# Recognized vertical dimensions in CLM output
VERTICAL_DIMS = {"levsoi", "levgrnd", "levdcmp"}

# =============================================================================
# Main plotting function
# =============================================================================


def plot_vr_profile(
    input_file: str, output_file: str, variable: str, hillslope: str = "North"
) -> None:
    """
    Plot vertical depth profile for a variable across hillslope positions.

    Parameters
    ----------
    input_file : str
        Path to 20-year binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    variable : str
        Name of vertically resolved variable to plot
    hillslope : str
        Hillslope aspect to plot ('North', 'East', 'South', 'West')

    Returns
    -------
    None
        Saves plot to output_file
    """

    # -------------------------------------------------------------------------
    # Validate hillslope argument
    # -------------------------------------------------------------------------
    if hillslope not in HILLSLOPE_INFO:
        print(f"ERROR: Invalid hillslope '{hillslope}'")
        print(f"Valid options: {list(HILLSLOPE_INFO.keys())}")
        sys.exit(1)

    h_info = HILLSLOPE_INFO[hillslope]
    h_cols = h_info["cols"]

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

    var_data = ds[variable]

    # -------------------------------------------------------------------------
    # Identify vertical dimension
    # -------------------------------------------------------------------------
    var_dims = set(var_data.dims)
    vert_dim = var_dims & VERTICAL_DIMS

    if not vert_dim:
        print(f"ERROR: Variable '{variable}' has no vertical dimension")
        print(f"Dimensions: {var_data.dims}")
        print(f"Expected one of: {VERTICAL_DIMS}")
        ds.close()
        sys.exit(1)

    vert_dim = vert_dim.pop()

    # -------------------------------------------------------------------------
    # Extract coordinates and metadata
    # -------------------------------------------------------------------------
    depths = ds[vert_dim].values
    distances = ds["hillslope_distance"].values[h_cols]
    mcdate = ds["mcdate"].values

    # Get most recent time step
    last_time_idx = -1
    year = mcdate[last_time_idx] // 10000

    # Extract data for selected hillslope at last time step
    # Shape: (levels, columns)
    data = var_data.isel(time=last_time_idx, column=h_cols).values

    # Get variable metadata
    units = var_data.attrs.get("units", "")
    long_name = var_data.attrs.get("long_name", variable)

    ds.close()

    # -------------------------------------------------------------------------
    # Sort columns by distance from stream
    # -------------------------------------------------------------------------
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    data = data[:, sort_idx]
    sorted_positions = [POSITION_NAMES[i] for i in sort_idx]

    # -------------------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))

    # Plot each position as a vertical profile
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, dist) in enumerate(zip(sorted_positions, distances)):
        values = data[:, i]
        label = f"{name} ({dist:.0f}m)"
        ax.plot(
            values,
            depths,
            "o-",
            linewidth=2,
            markersize=5,
            color=colors[i],
            label=label,
        )

    # Invert y-axis (depth increases downward, surface at top)
    ax.invert_yaxis()

    # -------------------------------------------------------------------------
    # Axis labels and formatting
    # -------------------------------------------------------------------------
    xlabel = f"{variable} ({units})" if units else variable
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(
        f"{long_name}\n{hillslope} Hillslope - Year {year} (20-yr avg)",
        fontsize=13,
        fontweight="bold",
    )

    # Log scale x-axis for better visualization
    ax.set_xscale("log")

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, title="Position (dist from stream)")

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
        description="Plot vertical depth profiles of soil variables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_20yr.nc plots/TSOI_profile.png TSOI
  %(prog)s data/combined_h1_20yr.nc plots/H2OSOI_south.png H2OSOI_LIQ --hillslope=South
        """,
    )
    parser.add_argument("input_file", help="20-year binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument("variable", help="Vertically resolved variable name")
    parser.add_argument(
        "--hillslope",
        choices=["North", "East", "South", "West"],
        default="North",
        help="Hillslope aspect to plot (default: North)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:     {args.input_file}")
    print(f"Output:    {args.output_file}")
    print(f"Variable:  {args.variable}")
    print(f"Hillslope: {args.hillslope}")
    print()

    plot_vr_profile(args.input_file, args.output_file, args.variable, args.hillslope)


if __name__ == "__main__":
    main()
