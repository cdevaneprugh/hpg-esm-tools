#!/usr/bin/env python3
"""
plot_zwt_hillslope_profile.py - Water table depth vs hillslope elevation

PURPOSE:
    Plot water table position along a hillslope profile, comparing early
    simulation period to recent period. Shows how water table equilibrates
    relative to surface topography over time.

USAGE:
    python3 plot_zwt_hillslope_profile.py <input_file> <output_file> <hillslope>
    python3 plot_zwt_hillslope_profile.py -h

ARGUMENTS:
    input_file      20-year binned h1 NetCDF file (e.g., combined_h1_20yr.nc)
    output_file     Output PNG filename
    hillslope       Hillslope aspect to plot: North, East, South, or West

EXAMPLE:
    python3 plot_zwt_hillslope_profile.py data/combined_h1_20yr.nc plots/zwt_profile.png North

OUTPUT:
    2-panel figure:
    - Top: Early simulation period (first 20-year bin)
    - Bottom: Recent period (last 20-year bin)
    Each panel shows surface elevation, water table position, and ZWT values.

NOTES:
    - ZWT = depth of water table below surface (m)
    - Water table elevation = hillslope_elev - ZWT
    - Requires 20-year binned data for meaningful period comparison
    - mcdate represents the center year of each 20-year bin
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

# Hillslope index mapping (matches CLM hillslope indexing)
HILLSLOPE_MAP = {
    "North": 1,
    "East": 2,
    "South": 3,
    "West": 4,
}

# Column position names (ordered by distance from stream)
POSITION_NAMES = ["Outlet", "Lower", "Upper", "Ridge"]

# =============================================================================
# Main plotting function
# =============================================================================


def plot_zwt_profile(
    input_file: str, output_file: str, hillslope: str = "North"
) -> None:
    """
    Plot ZWT vs hillslope profile for two time periods (early and recent).

    Parameters
    ----------
    input_file : str
        Path to 20-year binned h1 NetCDF file
    output_file : str
        Path for output PNG file
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
    if hillslope not in HILLSLOPE_MAP:
        print(f"ERROR: Invalid hillslope '{hillslope}'")
        print(f"Valid options: {list(HILLSLOPE_MAP.keys())}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)

    # Validate required variables exist
    required_vars = [
        "ZWT",
        "hillslope_elev",
        "hillslope_distance",
        "mcdate",
        "hillslope_index",
    ]
    for var in required_vars:
        if var not in ds:
            print(f"ERROR: Required variable '{var}' not found in {input_file}")
            ds.close()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Extract data
    # -------------------------------------------------------------------------
    zwt = ds["ZWT"].values
    elev = ds["hillslope_elev"].values
    dist = ds["hillslope_distance"].values
    mcdate = ds["mcdate"].values
    hillslope_index = ds["hillslope_index"].values

    ds.close()

    # -------------------------------------------------------------------------
    # Get columns for selected hillslope
    # -------------------------------------------------------------------------
    h_idx = HILLSLOPE_MAP[hillslope]
    h_cols = np.where(hillslope_index == h_idx)[0]

    # Sort by distance from stream (outlet to ridge)
    sort_order = np.argsort(dist[h_cols])
    h_cols = h_cols[sort_order]

    h_elev = elev[h_cols]
    h_dist = dist[h_cols]

    # -------------------------------------------------------------------------
    # Select time periods: first and last 20-year bin
    # -------------------------------------------------------------------------
    n_times = zwt.shape[0]
    time_indices = [0, n_times - 1]

    # Get years from mcdate (YYYYMMDD format)
    years = mcdate // 10000

    # Calculate year ranges for each bin
    # mcdate represents the average/center of the bin due to ncra averaging
    year_ranges = []
    for t_idx in time_indices:
        bin_center_year = years[t_idx]
        bin_start_year = bin_center_year - 9  # 20 year bin centered at mcdate
        bin_end_year = bin_center_year + 10
        year_ranges.append((bin_start_year, bin_end_year))

    time_labels = [
        f"Years {year_ranges[0][0]}-{year_ranges[0][1]} (Early, 20-yr avg)",
        f"Years {year_ranges[1][0]}-{year_ranges[1][1]} (Recent, 20-yr avg)",
    ]

    # -------------------------------------------------------------------------
    # Create 2-panel figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, t_idx, label in zip(axes, time_indices, time_labels):
        # Calculate water table elevation
        wt_elev = h_elev - zwt[t_idx, h_cols]

        # Plot surface elevation
        ax.plot(
            h_dist,
            h_elev,
            "o-",
            linewidth=2.5,
            markersize=10,
            color="brown",
            label="Hillslope Surface",
            zorder=3,
        )

        # Plot water table elevation
        ax.plot(
            h_dist,
            wt_elev,
            "o-",
            linewidth=2.5,
            markersize=10,
            color="blue",
            label="Water Table",
            zorder=2,
        )

        # Fill unsaturated zone (between surface and water table)
        ax.fill_between(
            h_dist, wt_elev, h_elev, alpha=0.3, color="tan", label="Unsaturated Zone"
        )

        # Fill saturated zone (below water table)
        ax.fill_between(
            h_dist, -10, wt_elev, alpha=0.3, color="lightblue", label="Saturated Zone"
        )

        # Stream level reference
        ax.axhline(
            0,
            color="darkblue",
            linestyle="--",
            linewidth=2,
            label="Stream Level",
            zorder=1,
        )

        # Add position labels and ZWT values
        for i, (d, e, wt) in enumerate(zip(h_dist, h_elev, wt_elev)):
            # Position name above surface
            ax.text(
                d,
                e + 0.5,
                POSITION_NAMES[i],
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

            # ZWT value in unsaturated zone
            zwt_val = zwt[t_idx, h_cols[i]]
            ax.text(
                d,
                (e + wt) / 2,
                f"ZWT={zwt_val:.1f}m",
                ha="center",
                fontsize=8,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        # Formatting
        ax.set_ylabel("Elevation above Stream (m)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(-10, 9)

    # X-axis label on bottom panel only
    axes[-1].set_xlabel("Distance from Stream (m)", fontsize=11)

    # -------------------------------------------------------------------------
    # Overall title
    # -------------------------------------------------------------------------
    first_year = years[0]
    last_year = years[-1]
    total_years = last_year - first_year

    fig.suptitle(
        f"Water Table Depth (ZWT) vs Hillslope Profile - {hillslope} Hillslope\n"
        f"Simulation Years {first_year}-{last_year} ({total_years} years total)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")

    # Print summary info
    print()
    print("Time periods plotted:")
    for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
        print(f"  {i + 1}. {label}")
        print(
            f"     ZWT: Outlet={zwt[t_idx, h_cols[0]]:.1f}m, "
            f"Ridge={zwt[t_idx, h_cols[3]]:.1f}m"
        )


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot water table depth vs hillslope elevation profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_20yr.nc plots/zwt_north.png North
  %(prog)s data/combined_h1_20yr.nc plots/zwt_south.png South
        """,
    )
    parser.add_argument("input_file", help="20-year binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument(
        "hillslope",
        choices=["North", "East", "South", "West"],
        help="Hillslope aspect to plot",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:     {args.input_file}")
    print(f"Output:    {args.output_file}")
    print(f"Hillslope: {args.hillslope}")
    print()

    plot_zwt_profile(args.input_file, args.output_file, args.hillslope)


if __name__ == "__main__":
    main()
