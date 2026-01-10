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
    - combined_h1.nc       (column-level, full resolution)
    - combined_h1_1yr.nc   (column-level, annual bins)
    - combined_h1_20yr.nc  (column-level, 20-year bins)

OUTPUT:
    All plots saved to plots/ directory:
    - {VAR}_full.png           Full simulation timeseries (h0, 20yr bins)
    - {VAR}_last20.png         Last 20 years by hillslope group (h1, 1yr bins)
    - ZWT_hillslope_profile.png  Water table vs elevation profile
    - elevation_width_overlay.png  Hillslope geometry
    - column_areas.png         Column area distribution
    - pft_distribution.png     PFT pie chart

NOTES:
    - Modify VARIABLES list to change which variables are plotted
    - Requires all input files to exist before running
    - Creates plots/ directory if it doesn't exist
"""

# =============================================================================
# Imports
# =============================================================================
import os

from plot_timeseries_full import plot_timeseries_full
from plot_timeseries_last20 import plot_timeseries_last20
from plot_zwt_hillslope_profile import plot_zwt_profile
from plot_elevation_width_overlay import plot_overlay_profiles
from plot_col_areas import plot_column_areas
from plot_pft_distribution import plot_pft_distribution

# =============================================================================
# Configuration
# =============================================================================

# Data file paths (relative to script directory)
DATA_DIR = "data"
PLOT_DIR = "plots"

H0_20YR = os.path.join(DATA_DIR, "combined_h0_20yr.nc")
H1_RAW = os.path.join(DATA_DIR, "combined_h1.nc")
H1_1YR = os.path.join(DATA_DIR, "combined_h1_1yr.nc")
H1_20YR = os.path.join(DATA_DIR, "combined_h1_20yr.nc")

# Variables for timeseries plots
VARIABLES = ["GPP", "TOTECOSYSC", "ZWT"]

# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all standard hillslope analysis plots."""

    # Ensure plot directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Full simulation timeseries (gridcell-level, 20-year bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Generating full simulation timeseries plots...")
    print("=" * 60)
    for var in VARIABLES:
        output = os.path.join(PLOT_DIR, f"{var}_full.png")
        plot_timeseries_full(H0_20YR, output, var)
        print()

    # -------------------------------------------------------------------------
    # Recent period timeseries (column-level, annual bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Generating last 20 years timeseries plots...")
    print("=" * 60)
    for var in VARIABLES:
        output = os.path.join(PLOT_DIR, f"{var}_last20.png")
        plot_timeseries_last20(H1_1YR, output, var, n_years=20)
        print()

    # -------------------------------------------------------------------------
    # Water table profile (column-level, 20-year bins)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Generating water table profile plot...")
    print("=" * 60)
    output = os.path.join(PLOT_DIR, "ZWT_hillslope_profile.png")
    plot_zwt_profile(H1_20YR, output, "North")
    print()

    # -------------------------------------------------------------------------
    # Hillslope geometry plots (column-level, raw)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Generating geometry plots...")
    print("=" * 60)

    output = os.path.join(PLOT_DIR, "elevation_width_overlay.png")
    plot_overlay_profiles(H1_RAW, output)
    print()

    output = os.path.join(PLOT_DIR, "column_areas.png")
    plot_column_areas(H1_RAW, output)
    print()

    # -------------------------------------------------------------------------
    # PFT distribution (column-level, raw)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Generating PFT distribution plot...")
    print("=" * 60)
    output = os.path.join(PLOT_DIR, "pft_distribution.png")
    plot_pft_distribution(H1_RAW, output)
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
