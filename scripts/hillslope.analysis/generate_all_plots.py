#!/usr/bin/env python3
"""
generate_all_plots.py - Generate all hillslope analysis plots

PURPOSE:
    Batch generation of all standard hillslope analysis plots from
    preprocessed NetCDF data. Intended for quick visualization after
    running bin_temporal.sh on raw history files.

USAGE:
    python3 generate_all_plots.py

PREREQUISITES:
    Data files in data/ directory:
    - combined_h0_20yr.nc  (gridcell-level, 20-year bins)
    - combined_h1_1yr.nc   (column-level, annual bins)
    - combined_h1_20yr.nc  (column-level, 20-year bins)

OUTPUT:
    All plots saved to plots/ directory.
"""

# =============================================================================
# Imports
# =============================================================================
import os

from plot_timeseries_full import plot_timeseries_full
from plot_timeseries_last20 import plot_timeseries_last20
from plot_zwt_hillslope_profile import plot_zwt_profile
from plot_hillslope_cross_section import plot_cross_section
from plot_tai_heatmap import plot_tai_heatmap
from plot_carbon_water_coupling import plot_carbon_water

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data"
PLOT_DIR = "plots"

H0_20YR = os.path.join(DATA_DIR, "combined_h0_20yr.nc")
H1_1YR = os.path.join(DATA_DIR, "combined_h1_1yr.nc")
H1_20YR = os.path.join(DATA_DIR, "combined_h1_20yr.nc")

# Variables for timeseries plots
VARIABLES = ["GPP", "TOTECOSYSC", "ZWT"]

# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all standard hillslope analysis plots."""

    os.makedirs(PLOT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Full simulation timeseries (gridcell-level, 20-year bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Full simulation timeseries (h0, 20yr bins)...")
    print("=" * 60)
    for var in VARIABLES:
        output = os.path.join(PLOT_DIR, f"{var}_full.png")
        plot_timeseries_full(H0_20YR, output, var)
        print()

    # -------------------------------------------------------------------------
    # Recent period timeseries by HAND zone (column-level, annual bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Recent timeseries by HAND zone (h1, 1yr bins)...")
    print("=" * 60)
    for var in VARIABLES:
        output = os.path.join(PLOT_DIR, f"{var}_last20.png")
        plot_timeseries_last20(H1_1YR, output, var, n_years=20)
        print()

    # -------------------------------------------------------------------------
    # Water table profile (column-level, 20-year bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Water table profile (h1, 20yr bins)...")
    print("=" * 60)
    output = os.path.join(PLOT_DIR, "ZWT_hillslope_profile.png")
    plot_zwt_profile(H1_20YR, output)
    print()

    # -------------------------------------------------------------------------
    # Hillslope cross-section (column-level, 20-year bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Hillslope cross-section (h1, 20yr bins)...")
    print("=" * 60)
    output = os.path.join(PLOT_DIR, "cross_section.png")
    plot_cross_section(H1_20YR, output)
    print()

    # -------------------------------------------------------------------------
    # TAI heatmaps (column-level, annual bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TAI heatmaps (h1, 1yr bins)...")
    print("=" * 60)
    for var in ["ZWT", "GPP"]:
        output = os.path.join(PLOT_DIR, f"{var}_heatmap.png")
        plot_tai_heatmap(H1_1YR, output, var)
        print()

    # -------------------------------------------------------------------------
    # Carbon-water coupling (column-level, annual bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Carbon-water coupling (h1, 1yr bins)...")
    print("=" * 60)
    output = os.path.join(PLOT_DIR, "carbon_water_coupling.png")
    plot_carbon_water(H1_1YR, output, "TOTSOMC")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
