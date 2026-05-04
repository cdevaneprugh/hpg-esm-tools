#!/usr/bin/env python3
"""
plot_vr_profile.py - Vertically resolved variable depth profiles

PURPOSE:
    Plot depth profiles of vertically resolved variables (soil temperature,
    moisture, carbon, etc.) as "core samples" at representative hillslope
    positions. X-axis shows variable values, Y-axis shows depth below surface.

    Supports any single-aspect hillslope layout (1xN columns).

USAGE:
    python3 plot_vr_profile.py <input_file> <output_file> <variable>
    python3 plot_vr_profile.py -h

ARGUMENTS:
    input_file      Binned h1 NetCDF file (e.g., combined_h1_20yr.nc)
    output_file     Output PNG filename
    variable        Vertically resolved variable name (e.g., TSOI, H2OSOI_LIQ)

EXAMPLE:
    python3 plot_vr_profile.py data/combined_h1_20yr.nc plots/TSOI_profile.png TSOI

OUTPUT:
    Single panel with vertical profiles at ~5 representative columns
    spanning the hillslope from stream to ridge.

NOTES:
    - Variable must have a vertical dimension (levsoi, levgrnd, or levdcmp)
    - Depth increases downward (inverted Y-axis)
    - Current h1a output has no VR variables; this script is ready for future
      cases that add TSOI/H2OSOI_LIQ/etc. to hist_fincl2
"""

# =============================================================================
# Imports
# =============================================================================
import sys
import argparse
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Constants
# =============================================================================

VERTICAL_DIMS = {"levsoi", "levgrnd", "levdcmp"}


# =============================================================================
# Helpers
# =============================================================================


def detect_hillslope_columns(ds):
    """Return indices of hillslope columns, sorted by elevation."""
    h_idx = ds["hillslope_index"].values
    mask = (h_idx > 0) & (h_idx < 9000)
    cols = np.where(mask)[0]
    elevs = ds["hillslope_elev"].values[cols]
    order = np.argsort(elevs)
    return cols[order]


def select_representative_columns(n_cols, max_show=5):
    """Pick ~max_show evenly spaced indices spanning 0..n_cols-1."""
    if n_cols <= max_show:
        return np.arange(n_cols)
    indices = np.linspace(0, n_cols - 1, max_show, dtype=int)
    return np.unique(indices)


# =============================================================================
# Main plotting function
# =============================================================================


def plot_vr_profile(input_file: str, output_file: str, variable: str) -> None:
    """
    Plot vertical depth profile for a variable across hillslope positions.

    Parameters
    ----------
    input_file : str
        Path to binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    variable : str
        Name of vertically resolved variable to plot
    """

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)

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
    # Detect columns and select representatives
    # -------------------------------------------------------------------------
    h_cols = detect_hillslope_columns(ds)
    n_cols = len(h_cols)
    rep = select_representative_columns(n_cols)

    elevations = ds["hillslope_elev"].values[h_cols]
    distances = ds["hillslope_distance"].values[h_cols]
    depths = ds[vert_dim].values
    mcdate = ds["mcdate"].values

    # Get most recent time step
    year = mcdate[-1] // 10000
    data = var_data.isel(time=-1, column=h_cols[rep]).values

    units = var_data.attrs.get("units", "")
    long_name = var_data.attrs.get("long_name", variable)

    ds.close()

    # -------------------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, idx in enumerate(rep):
        elev = elevations[idx]
        dist = distances[idx]
        values = data[:, i]
        label = f"HAND {elev:.1f}m ({dist:.0f}m)"
        ax.plot(
            values,
            depths,
            "o-",
            linewidth=2,
            markersize=5,
            color=colors[i % len(colors)],
            label=label,
        )

    ax.invert_yaxis()

    xlabel = f"{variable} ({units})" if units else variable
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(
        f"{long_name}\n{n_cols}-column hillslope - Year {year}",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, title="Position (HAND elev, dist)")

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
  %(prog)s data/combined_h1_20yr.nc plots/H2OSOI.png H2OSOI_LIQ
        """,
    )
    parser.add_argument("input_file", help="Binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument("variable", help="Vertically resolved variable name")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Variable: {args.variable}")
    print()

    plot_vr_profile(args.input_file, args.output_file, args.variable)


if __name__ == "__main__":
    main()
