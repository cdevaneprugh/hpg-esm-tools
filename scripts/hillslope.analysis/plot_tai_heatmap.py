#!/usr/bin/env python3
"""
plot_tai_heatmap.py - Space-time heatmap of TAI boundary evolution

PURPOSE:
    Visualize how a variable (default: ZWT) evolves across the hillslope
    over time. Y-axis shows columns ordered by HAND elevation, X-axis
    shows simulation time. Color encodes the variable value. The TAI
    boundary appears as a color gradient transition.

USAGE:
    python3 plot_tai_heatmap.py <input_file> <output_file> [variable]
    python3 plot_tai_heatmap.py -h

ARGUMENTS:
    input_file      Annual binned h1 NetCDF file (e.g., combined_h1_1yr.nc)
    output_file     Output PNG filename
    variable        Variable to plot (default: ZWT)

EXAMPLE:
    python3 plot_tai_heatmap.py data/combined_h1_1yr.nc plots/tai_heatmap.png
    python3 plot_tai_heatmap.py data/combined_h1_1yr.nc plots/gpp_heatmap.png GPP

OUTPUT:
    Heatmap with HAND elevation on Y-axis and time on X-axis. For ZWT,
    uses a diverging colormap (blue=saturated, brown=deep water table).
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


# =============================================================================
# Main plotting function
# =============================================================================


def plot_tai_heatmap(input_file: str, output_file: str, variable: str = "ZWT") -> None:
    """
    Create space-time heatmap of variable across hillslope columns.

    Parameters
    ----------
    input_file : str
        Path to annually binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    variable : str
        Variable to plot (default: ZWT)
    """

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)

    if variable not in ds:
        print(f"ERROR: Variable '{variable}' not found in {input_file}")
        print(f"Available: {[v for v in ds.data_vars if 'column' in ds[v].dims]}")
        ds.close()
        sys.exit(1)

    h_cols = detect_hillslope_columns(ds)
    n_cols = len(h_cols)

    var_data = ds[variable].values[:, h_cols]  # (time, n_hillslope_cols)
    elevations = ds["hillslope_elev"].values[h_cols]
    mcdate = ds["mcdate"].values
    units = ds[variable].attrs.get("units", "")

    ds.close()

    years = mcdate // 10000

    # -------------------------------------------------------------------------
    # Choose colormap
    # -------------------------------------------------------------------------
    if variable == "ZWT":
        # Diverging: blue (saturated, low ZWT) to brown (deep water table)
        cmap = "RdYlBu_r"
        vmin, vmax = 0, min(np.percentile(var_data, 95), 15)
        cbar_label = f"ZWT ({units})" if units else "ZWT (m)"
    elif variable in ("H2OSFC", "FH2OSFC"):
        cmap = "Blues"
        vmin, vmax = 0, np.percentile(var_data, 95)
        cbar_label = f"{variable} ({units})" if units else variable
    elif variable == "GPP":
        cmap = "Greens"
        vmin, vmax = np.percentile(var_data, 5), np.percentile(var_data, 95)
        cbar_label = f"GPP ({units})" if units else "GPP"
    else:
        cmap = "viridis"
        vmin, vmax = np.percentile(var_data, 5), np.percentile(var_data, 95)
        cbar_label = f"{variable} ({units})" if units else variable

    # -------------------------------------------------------------------------
    # Build grid edges for pcolormesh
    # -------------------------------------------------------------------------
    # Time edges: midpoints between years, plus outer bounds
    if len(years) > 1:
        dt = np.diff(years)
        t_edges = np.concatenate(
            [
                [years[0] - dt[0] / 2],
                years[:-1] + dt / 2,
                [years[-1] + dt[-1] / 2],
            ]
        )
    else:
        t_edges = np.array([years[0] - 0.5, years[0] + 0.5])

    # Column edges: midpoints between elevations, plus outer bounds
    if n_cols > 1:
        de = np.diff(elevations)
        c_edges = np.concatenate(
            [
                [elevations[0] - de[0] / 2],
                elevations[:-1] + de / 2,
                [elevations[-1] + de[-1] / 2],
            ]
        )
    else:
        c_edges = np.array([elevations[0] - 0.5, elevations[0] + 0.5])

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))

    im = ax.pcolormesh(
        t_edges,
        c_edges,
        var_data.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)

    # TAI zone boundary lines
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(5.0, color="white", linestyle="--", linewidth=1, alpha=0.7)

    # Zone labels on right margin
    ax.text(
        t_edges[-1] + (t_edges[-1] - t_edges[0]) * 0.01,
        0.25,
        "TAI",
        fontsize=8,
        color="gray",
        va="center",
    )
    if elevations[-1] > 5.0:
        ax.text(
            t_edges[-1] + (t_edges[-1] - t_edges[0]) * 0.01,
            2.75,
            "Trans.",
            fontsize=8,
            color="gray",
            va="center",
        )

    ax.set_xlabel("Simulation Year", fontsize=12)
    ax.set_ylabel("HAND Elevation (m)", fontsize=12)
    ax.set_title(
        f"{variable} Across Hillslope ({n_cols} columns)\nYears {years[0]}-{years[-1]}",
        fontsize=13,
        fontweight="bold",
    )

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
        description="Space-time heatmap of hillslope variable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_1yr.nc plots/zwt_heatmap.png
  %(prog)s data/combined_h1_1yr.nc plots/gpp_heatmap.png GPP
  %(prog)s data/combined_h1_1yr.nc plots/h2osfc_heatmap.png H2OSFC
        """,
    )
    parser.add_argument("input_file", help="Annual binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument(
        "variable", nargs="?", default="ZWT", help="Variable to plot (default: ZWT)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Variable: {args.variable}")
    print()

    plot_tai_heatmap(args.input_file, args.output_file, args.variable)


if __name__ == "__main__":
    main()
