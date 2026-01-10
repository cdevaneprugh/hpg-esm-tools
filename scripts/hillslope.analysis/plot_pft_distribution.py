#!/usr/bin/env python3
"""
plot_pft_distribution.py - Plant Functional Type distribution

PURPOSE:
    Visualize the PFT (Plant Functional Type) distribution for hillslope
    columns. Creates a pie chart showing the fractional coverage of each
    vegetation type present in the simulation.

USAGE:
    python3 plot_pft_distribution.py <input_file> <output_file>
    python3 plot_pft_distribution.py -h

ARGUMENTS:
    input_file      h1 stream NetCDF file (e.g., combined_h1.nc)
    output_file     Output PNG filename

EXAMPLE:
    python3 plot_pft_distribution.py data/combined_h1.nc plots/pft_distribution.png

OUTPUT:
    Pie chart showing fractional distribution of PFTs present in the gridcell.
    Percentages indicate column-level PFT weights (pfts1d_wtcol).

NOTES:
    - All hillslope columns have the same PFT distribution, so only first
      column is used for the plot
    - PFT names are from CTSM parameter file (ctsm60_params.c250311.nc)
    - pfts1d_wtcol gives the fractional coverage within each column
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

# PFT names from CTSM parameter file
# Source: /blue/gerber/earth_models/inputdata/lnd/clm2/paramdata/ctsm60_params.c250311.nc
PFT_NAMES = {
    0: "Not vegetated",
    1: "Needleleaf evergreen temperate tree",
    2: "Needleleaf evergreen boreal tree",
    3: "Needleleaf deciduous boreal tree",
    4: "Broadleaf evergreen tropical tree",
    5: "Broadleaf evergreen temperate tree",
    6: "Broadleaf deciduous tropical tree",
    7: "Broadleaf deciduous temperate tree",
    8: "Broadleaf deciduous boreal tree",
    9: "Broadleaf evergreen shrub",
    10: "Broadleaf deciduous temperate shrub",
    11: "Broadleaf deciduous boreal shrub",
    12: "C3 arctic grass",
    13: "C3 non-arctic grass",
    14: "C4 grass",
    15: "C3 crop",
    16: "C3 irrigated",
    17: "Temperate corn",
    18: "Irrigated temperate corn",
    19: "Spring wheat",
    20: "Irrigated spring wheat",
    21: "Winter wheat",
    22: "Irrigated winter wheat",
    23: "Temperate soybean",
    24: "Irrigated temperate soybean",
    25: "Barley",
    26: "Irrigated barley",
    27: "Winter barley",
    28: "Irrigated winter barley",
    29: "Rye",
    30: "Irrigated rye",
    31: "Winter rye",
    32: "Irrigated winter rye",
    33: "Cassava",
    34: "Irrigated cassava",
    35: "Citrus",
    36: "Irrigated citrus",
    37: "Cocoa",
    38: "Irrigated cocoa",
    39: "Coffee",
    40: "Irrigated coffee",
    41: "Cotton",
    42: "Irrigated cotton",
    43: "Date palm",
    44: "Irrigated date palm",
    45: "Fodder grass",
    46: "Irrigated fodder grass",
    47: "Grapes",
    48: "Irrigated grapes",
    49: "Groundnuts",
    50: "Irrigated groundnuts",
    51: "Millet",
    52: "Irrigated millet",
    53: "Oil palm",
    54: "Irrigated oil palm",
    55: "Potatoes",
    56: "Irrigated potatoes",
    57: "Pulses",
    58: "Irrigated pulses",
    59: "Rapeseed",
    60: "Irrigated rapeseed",
    61: "Rice",
    62: "Irrigated rice",
    63: "Sorghum",
    64: "Irrigated sorghum",
    65: "Sugar beet",
    66: "Irrigated sugar beet",
    67: "Sugarcane",
    68: "Irrigated sugarcane",
    69: "Sunflower",
    70: "Irrigated sunflower",
    71: "Miscanthus",
    72: "Irrigated miscanthus",
    73: "Switchgrass",
    74: "Irrigated switchgrass",
    75: "Tropical corn",
    76: "Irrigated tropical corn",
    77: "Tropical soybean",
    78: "Irrigated tropical soybean",
}

# Colormap for pie chart
PIE_COLORS = plt.cm.Set3(np.arange(12))

# =============================================================================
# Main plotting function
# =============================================================================


def plot_pft_distribution(input_file: str, output_file: str) -> None:
    """
    Create pie chart of PFT distribution.

    Parameters
    ----------
    input_file : str
        Path to h1 NetCDF file containing PFT variables
    output_file : str
        Path for output PNG file

    Returns
    -------
    None
        Saves plot to output_file
    """

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file)

    # Validate required variables exist
    required_vars = ["pfts1d_ci", "pfts1d_itype_veg", "pfts1d_wtcol"]
    for var in required_vars:
        if var not in ds:
            print(f"ERROR: Required variable '{var}' not found in {input_file}")
            ds.close()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Extract PFT data for first column
    # -------------------------------------------------------------------------
    pft_column = ds["pfts1d_ci"].values  # Which column each PFT belongs to
    pft_types = ds["pfts1d_itype_veg"].values  # PFT type index
    pft_weights = ds["pfts1d_wtcol"].values  # Fractional weight within column

    ds.close()

    # All hillslope columns have same PFT distribution, so use first column
    col_mask = pft_column == 1
    col_types = pft_types[col_mask]
    col_weights = pft_weights[col_mask]

    # Filter out zero-weight PFTs
    nonzero_mask = col_weights > 0
    col_types = col_types[nonzero_mask]
    col_weights = col_weights[nonzero_mask]

    # -------------------------------------------------------------------------
    # Create pie chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate labels from PFT names
    labels = [PFT_NAMES.get(t, f"PFT {t}") for t in col_types]

    # Create pie with colors cycling through colormap
    colors = PIE_COLORS[: len(col_types) % len(PIE_COLORS)]
    wedges, texts, autotexts = ax.pie(
        col_weights, labels=labels, colors=colors, autopct="%1.2f%%", startangle=90
    )

    # Improve text visibility
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(9)
        autotext.set_weight("bold")

    for text in texts:
        text.set_fontsize(10)

    # -------------------------------------------------------------------------
    # Title and save
    # -------------------------------------------------------------------------
    ax.set_title("Hillslope PFT Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")

    # Print summary
    print()
    print("PFT distribution:")
    for pft_type, weight in zip(col_types, col_weights):
        name = PFT_NAMES.get(pft_type, f"PFT {pft_type}")
        print(f"  {name}: {weight * 100:.2f}%")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot PFT distribution pie chart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1.nc plots/pft_distribution.png
        """,
    )
    parser.add_argument("input_file", help="h1 NetCDF file with PFT data")
    parser.add_argument("output_file", help="Output PNG filename")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print()

    plot_pft_distribution(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
