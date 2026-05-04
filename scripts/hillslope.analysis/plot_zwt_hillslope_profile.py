#!/usr/bin/env python3
"""
plot_zwt_hillslope_profile.py - Water table depth vs hillslope elevation

PURPOSE:
    Plot water table position along a hillslope profile, comparing early
    simulation period to recent period. Shows how water table equilibrates
    relative to surface topography over time.

    Supports any single-aspect hillslope layout (1xN columns).

USAGE:
    python3 plot_zwt_hillslope_profile.py <input_file> <output_file>
    python3 plot_zwt_hillslope_profile.py -h

ARGUMENTS:
    input_file      Binned h1 NetCDF file (e.g., combined_h1_20yr.nc)
    output_file     Output PNG filename

EXAMPLE:
    python3 plot_zwt_hillslope_profile.py data/combined_h1_20yr.nc plots/zwt_profile.png

OUTPUT:
    2-panel figure:
    - Top: Early simulation period (first time bin)
    - Bottom: Recent period (last time bin)
    Each panel shows surface elevation, water table position, and fills
    for saturated/unsaturated zones.

NOTES:
    - ZWT = depth of water table below surface (m)
    - Water table elevation = hillslope_elev - ZWT
    - Auto-detects hillslope columns (hillslope_index > 0)
    - Works with any number of HAND bins (8, 16, etc.)
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


def plot_zwt_profile(input_file: str, output_file: str) -> None:
    """
    Plot ZWT vs hillslope profile for two time periods (early and recent).

    Parameters
    ----------
    input_file : str
        Path to binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    """

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)

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
    # Detect columns and extract data
    # -------------------------------------------------------------------------
    h_cols = detect_hillslope_columns(ds)
    n_cols = len(h_cols)

    zwt = ds["ZWT"].values
    h_elev = ds["hillslope_elev"].values[h_cols]
    h_dist = ds["hillslope_distance"].values[h_cols]
    mcdate = ds["mcdate"].values

    ds.close()

    # -------------------------------------------------------------------------
    # Select time periods: first and last bin
    # -------------------------------------------------------------------------
    n_times = zwt.shape[0]
    time_indices = [0, n_times - 1]

    years = mcdate // 10000

    time_labels = [
        f"Year {years[0]} (Early)",
        f"Year {years[-1]} (Recent)",
    ]

    # -------------------------------------------------------------------------
    # Create 2-panel figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax, t_idx, label in zip(axes, time_indices, time_labels):
        wt_elev = h_elev - zwt[t_idx, h_cols]

        # Surface and water table lines
        ax.plot(
            h_dist,
            h_elev,
            "-",
            linewidth=2.5,
            color="brown",
            label="Hillslope Surface",
            zorder=3,
        )
        ax.plot(
            h_dist,
            wt_elev,
            "-",
            linewidth=2.5,
            color="blue",
            label="Water Table",
            zorder=2,
        )

        # Add markers for small N
        if n_cols <= 10:
            ax.plot(h_dist, h_elev, "o", markersize=8, color="brown", zorder=3)
            ax.plot(h_dist, wt_elev, "o", markersize=8, color="blue", zorder=2)

        # Fills
        y_floor = min(np.min(wt_elev), np.min(h_elev)) - 2
        ax.fill_between(
            h_dist, wt_elev, h_elev, alpha=0.3, color="tan", label="Unsaturated Zone"
        )
        ax.fill_between(
            h_dist,
            y_floor,
            wt_elev,
            alpha=0.3,
            color="lightblue",
            label="Saturated Zone",
        )

        # Stream level reference
        ax.axhline(
            0,
            color="darkblue",
            linestyle="--",
            linewidth=1.5,
            label="Stream Level",
            zorder=1,
        )

        # Annotations: label a few representative columns
        if n_cols <= 10:
            # Label every column
            for i in range(n_cols):
                zwt_val = zwt[t_idx, h_cols[i]]
                ax.text(
                    h_dist[i],
                    h_elev[i] + 0.4,
                    f"{h_elev[i]:.1f}m",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )
                ax.text(
                    h_dist[i],
                    (h_elev[i] + wt_elev[i]) / 2,
                    f"ZWT={zwt_val:.1f}m",
                    ha="center",
                    fontsize=7,
                    style="italic",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )
        else:
            # Label ~5 representative columns
            indices = np.linspace(0, n_cols - 1, 5, dtype=int)
            for i in indices:
                zwt_val = zwt[t_idx, h_cols[i]]
                ax.text(
                    h_dist[i],
                    h_elev[i] + 0.4,
                    f"{h_elev[i]:.1f}m",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )
                ax.text(
                    h_dist[i],
                    (h_elev[i] + wt_elev[i]) / 2,
                    f"ZWT={zwt_val:.1f}m",
                    ha="center",
                    fontsize=7,
                    style="italic",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

        # Formatting
        ax.set_ylabel("Elevation above Stream (m)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(y_floor, np.max(h_elev) * 1.15)

    axes[-1].set_xlabel("Distance from Stream (m)", fontsize=11)

    # -------------------------------------------------------------------------
    # Overall title
    # -------------------------------------------------------------------------
    fig.suptitle(
        f"Water Table vs Hillslope Profile ({n_cols} columns)\n"
        f"Simulation Years {years[0]}-{years[-1]}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"\n  Columns: {n_cols}")
    for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
        print(
            f"  {label}: ZWT stream={zwt[t_idx, h_cols[0]]:.2f}m, "
            f"ridge={zwt[t_idx, h_cols[-1]]:.2f}m"
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
  %(prog)s data/combined_h1_20yr.nc plots/zwt_profile.png
        """,
    )
    parser.add_argument("input_file", help="Binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print()

    plot_zwt_profile(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
