#!/usr/bin/env python3
"""
plot_elevation_width_overlay.py - Hillslope geometry profiles

PURPOSE:
    Plot elevation and width profiles for all 4 hillslope aspects overlaid.
    Shows geometric similarity (or differences) between North, East, South,
    and West facing hillslopes.

USAGE:
    python3 plot_elevation_width_overlay.py <input_file> <output_file>
    python3 plot_elevation_width_overlay.py -h

ARGUMENTS:
    input_file      h1 stream NetCDF file with hillslope geometry data
    output_file     Output PNG filename

EXAMPLE:
    python3 plot_elevation_width_overlay.py data/combined_h1.nc plots/geometry.png

NOTES:
    - Uses hillslope columns 0-15 (excludes stream column 16)
    - Each hillslope has 4 positions: Outlet, Lower, Upper, Ridge
    - X-axis: distance from stream (m)
    - Left panel: elevation above stream
    - Right panel: hillslope width at each position
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import xarray as xr
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless systems
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Constants
# =============================================================================

# Hillslope aspect styling (consistent across all plotting scripts)
ASPECTS = {
    1: {"name": "North", "color": "#3498db"},
    2: {"name": "East", "color": "#e74c3c"},
    3: {"name": "South", "color": "#f39c12"},
    4: {"name": "West", "color": "#2ecc71"},
}

N_HILLSLOPE_COLS = 16  # Exclude stream column 16

# =============================================================================
# Main plotting function
# =============================================================================


def plot_overlay_profiles(input_file: str, output_file: str) -> None:
    """
    Plot elevation and width profiles with all 4 hillslopes overlaid.

    Parameters
    ----------
    input_file : str
        Path to h1 NetCDF file containing hillslope geometry variables
    output_file : str
        Path for output PNG file

    Returns
    -------
    None
        Saves plot to output_file
    """

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file)

    # Extract hillslope geometry (exclude stream column 16)
    # Squeeze to handle both 1D and 3D arrays (e.g., shape (16,) or (16,1,1))
    hillslope_idx = ds["hillslope_index"].values[0:N_HILLSLOPE_COLS].squeeze()

    # Support both variable names (hillslope_elevation in new files, hillslope_elev in old)
    elev_var = (
        "hillslope_elevation" if "hillslope_elevation" in ds else "hillslope_elev"
    )
    elevation = ds[elev_var].values[0:N_HILLSLOPE_COLS].squeeze()

    width = ds["hillslope_width"].values[0:N_HILLSLOPE_COLS].squeeze()
    distance = ds["hillslope_distance"].values[0:N_HILLSLOPE_COLS].squeeze()

    ds.close()

    # -------------------------------------------------------------------------
    # Create figure with side-by-side panels
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------------------------------------------------------
    # Plot each hillslope
    # -------------------------------------------------------------------------
    for hillslope_num in range(1, 5):
        # Get styling for this aspect
        style = ASPECTS[hillslope_num]

        # Filter columns for this hillslope
        mask = hillslope_idx == hillslope_num
        elev = elevation[mask]
        w = width[mask]
        dist = distance[mask]

        # Sort by distance (ridge to outlet, descending)
        sort_idx = np.argsort(dist)[::-1]
        elev = elev[sort_idx]
        w = w[sort_idx]
        dist = dist[sort_idx]

        # Plot elevation profile (left panel)
        ax1.plot(
            dist,
            elev,
            marker="o",
            color=style["color"],
            linewidth=2,
            markersize=8,
            label=style["name"],
            alpha=0.85,
        )

        # Plot width profile (right panel)
        ax2.plot(
            dist,
            w,
            marker="o",
            color=style["color"],
            linewidth=2,
            markersize=8,
            label=style["name"],
            alpha=0.85,
        )

    # -------------------------------------------------------------------------
    # Configure left panel (elevation)
    # -------------------------------------------------------------------------
    ax1.set_xlabel("Distance from Stream (m)", fontsize=12)
    ax1.set_ylabel("Elevation above Stream (m)", fontsize=12)
    ax1.set_title("Elevation Profiles", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax1.set_ylim(0, elevation.max() * 1.15)

    # -------------------------------------------------------------------------
    # Configure right panel (width)
    # -------------------------------------------------------------------------
    ax2.set_xlabel("Distance from Stream (m)", fontsize=12)
    ax2.set_ylabel("Hillslope Width (m)", fontsize=12)
    ax2.set_title("Width Profiles", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim(0, width.max() * 1.15)

    # -------------------------------------------------------------------------
    # Overall title and save
    # -------------------------------------------------------------------------
    fig.suptitle(
        "Hillslope Geometry: All Aspects Overlaid", fontsize=14, fontweight="bold"
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
        description="Plot hillslope elevation and width profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1.nc plots/elevation_width_overlay.png
        """,
    )
    parser.add_argument("input_file", help="h1 NetCDF file with hillslope geometry")
    parser.add_argument("output_file", help="Output PNG filename")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print()

    plot_overlay_profiles(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
