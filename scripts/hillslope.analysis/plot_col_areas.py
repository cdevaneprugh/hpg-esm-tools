#!/usr/bin/env python3
"""
plot_col_areas.py - Bar chart of hillslope column areas

PURPOSE:
    Visualize the area distribution across hillslope columns. Shows how
    much of the gridcell each column represents, grouped by hillslope
    aspect (North, East, South, West).

USAGE:
    python3 plot_col_areas.py <input_file> <output_file>
    python3 plot_col_areas.py -h

ARGUMENTS:
    input_file      h1 stream NetCDF file with column geometry data
    output_file     Output PNG filename

EXAMPLE:
    python3 plot_col_areas.py data/combined_h1.nc plots/column_areas.png

NOTES:
    - Uses hillslope columns 0-15 (excludes stream column 16)
    - Colors are assigned by hillslope aspect (North=blue, East=red, etc.)
    - Percentage labels show each column's fraction of total hillslope area
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

# Hillslope aspect colors (consistent across all plotting scripts)
ASPECT_COLORS = {
    1: "#3498db",  # North - blue
    2: "#e74c3c",  # East - red
    3: "#f39c12",  # South - orange
    4: "#2ecc71",  # West - green
}

ASPECT_NAMES = {1: "North", 2: "East", 3: "South", 4: "West"}

# Column layout: 4 hillslopes x 4 positions = 16 columns
# Columns 0-3: North (Outlet, Lower, Upper, Ridge)
# Columns 4-7: East
# Columns 8-11: South
# Columns 12-15: West
N_HILLSLOPE_COLS = 16

# =============================================================================
# Main plotting function
# =============================================================================


def plot_column_areas(input_file: str, output_file: str) -> None:
    """
    Create bar chart of hillslope column areas.

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
    areas = ds["hillslope_area"].values[0:N_HILLSLOPE_COLS].squeeze()

    ds.close()

    # -------------------------------------------------------------------------
    # Calculate statistics
    # -------------------------------------------------------------------------
    total_area = areas.sum()
    percentages = (areas / total_area) * 100

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    # Assign colors by hillslope aspect
    bar_colors = [ASPECT_COLORS[h] for h in hillslope_idx]

    # Create bar chart
    x_pos = np.arange(len(areas))
    bars = ax.bar(x_pos, areas, color=bar_colors, edgecolor="black", linewidth=1)

    # -------------------------------------------------------------------------
    # Add percentage labels on bars
    # -------------------------------------------------------------------------
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + total_area * 0.01,  # Small offset above bar
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # -------------------------------------------------------------------------
    # Add hillslope dividers and labels
    # -------------------------------------------------------------------------
    # Vertical lines between hillslope groups
    for x in [3.5, 7.5, 11.5]:
        ax.axvline(x, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Hillslope labels at top of plot
    y_top = ax.get_ylim()[1] * 0.95
    for aspect_idx, x_center in [(1, 1.5), (2, 5.5), (3, 9.5), (4, 13.5)]:
        ax.text(
            x_center,
            y_top,
            ASPECT_NAMES[aspect_idx],
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=ASPECT_COLORS[aspect_idx],
        )

    # -------------------------------------------------------------------------
    # Axis labels and formatting
    # -------------------------------------------------------------------------
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Col {i}" for i in range(len(areas))], rotation=0, fontsize=9)
    ax.set_xlabel("Column Index", fontsize=12)
    ax.set_ylabel("Column Area (m²)", fontsize=12)
    ax.set_title("Hillslope Column Areas", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

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
        description="Bar chart of hillslope column areas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1.nc plots/column_areas.png
        """,
    )
    parser.add_argument("input_file", help="h1 NetCDF file with column geometry")
    parser.add_argument("output_file", help="Output PNG filename")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print()

    plot_column_areas(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
