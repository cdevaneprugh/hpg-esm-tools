#!/usr/bin/env python3
"""
plot_carbon_water_coupling.py - Water-carbon coupling across hillslope zones

PURPOSE:
    Show how water table position drives carbon cycling at different
    hillslope positions. Selects three representative columns (TAI,
    Transition, Upland) and plots ZWT alongside a carbon variable
    on twin axes for each.

USAGE:
    python3 plot_carbon_water_coupling.py <input_file> <output_file> [--carbon-var=TOTSOMC]
    python3 plot_carbon_water_coupling.py -h

ARGUMENTS:
    input_file      Annual binned h1 NetCDF file (e.g., combined_h1_1yr.nc)
    output_file     Output PNG filename
    --carbon-var    Carbon variable to plot (default: TOTSOMC)

EXAMPLE:
    python3 plot_carbon_water_coupling.py data/combined_h1_1yr.nc plots/coupling.png
    python3 plot_carbon_water_coupling.py data/combined_h1_1yr.nc plots/gpp.png --carbon-var=GPP

OUTPUT:
    3-row figure. Each row is one hillslope zone (TAI, Transition, Upland)
    with ZWT on the left axis and the carbon variable on the right axis.
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


def pick_representative_columns(elevations):
    """Pick TAI (lowest), Transition (~median), Upland (highest)."""
    n = len(elevations)
    picks = {
        "TAI": 0,
        "Transition": n // 2,
        "Upland": n - 1,
    }
    return picks


# =============================================================================
# Main plotting function
# =============================================================================


def plot_carbon_water(
    input_file: str, output_file: str, carbon_var: str = "TOTSOMC"
) -> None:
    """
    Plot water-carbon coupling at 3 representative hillslope positions.

    Parameters
    ----------
    input_file : str
        Path to annually binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    carbon_var : str
        Carbon variable to plot alongside ZWT (default: TOTSOMC)
    """

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(input_file, decode_times=False)

    for var in ["ZWT", carbon_var]:
        if var not in ds:
            print(f"ERROR: Variable '{var}' not found in {input_file}")
            print(f"Available: {[v for v in ds.data_vars if 'column' in ds[v].dims]}")
            ds.close()
            sys.exit(1)

    h_cols = detect_hillslope_columns(ds)
    elevations = ds["hillslope_elev"].values[h_cols]

    zwt = ds["ZWT"].values[:, h_cols]
    carbon = ds[carbon_var].values[:, h_cols]
    mcdate = ds["mcdate"].values
    c_units = ds[carbon_var].attrs.get("units", "")

    ds.close()

    years = mcdate // 10000
    picks = pick_representative_columns(elevations)

    # -------------------------------------------------------------------------
    # Create 3-row figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    zone_colors = {
        "TAI": ("#2166ac", "#1b7837"),
        "Transition": ("#b2182b", "#762a83"),
        "Upland": ("#e08214", "#542788"),
    }

    for ax, (zone_name, col_idx) in zip(axes, picks.items()):
        elev = elevations[col_idx]
        zwt_col = zwt[:, col_idx]
        carbon_col = carbon[:, col_idx]

        zwt_color, carbon_color = zone_colors[zone_name]

        # ZWT on left axis
        ax.plot(years, zwt_col, linewidth=2, color=zwt_color, alpha=0.9)
        ax.set_ylabel("ZWT (m)", fontsize=11, color=zwt_color)
        ax.tick_params(axis="y", labelcolor=zwt_color)
        ax.invert_yaxis()  # ZWT: 0 at top (saturated), deep at bottom

        # Carbon on right axis
        ax2 = ax.twinx()
        ax2.plot(
            years, carbon_col, linewidth=2, color=carbon_color, alpha=0.9, linestyle="-"
        )
        c_label = f"{carbon_var} ({c_units})" if c_units else carbon_var
        ax2.set_ylabel(c_label, fontsize=11, color=carbon_color)
        ax2.tick_params(axis="y", labelcolor=carbon_color)

        ax.set_title(
            f"{zone_name} — HAND {elev:.1f}m (col {col_idx})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.2, linestyle="--")

    axes[-1].set_xlabel("Simulation Year", fontsize=12)

    fig.suptitle(
        f"Water-Carbon Coupling: ZWT vs {carbon_var}\n"
        f"{len(elevations)}-column hillslope, Years {years[0]}-{years[-1]}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")
    for zone_name, col_idx in picks.items():
        print(
            f"  {zone_name}: HAND {elevations[col_idx]:.1f}m, "
            f"ZWT {zwt[-1, col_idx]:.2f}m, "
            f"{carbon_var} {carbon[-1, col_idx]:.1f}"
        )


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot water-carbon coupling across hillslope zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_1yr.nc plots/coupling.png
  %(prog)s data/combined_h1_1yr.nc plots/gpp_coupling.png --carbon-var=GPP
        """,
    )
    parser.add_argument("input_file", help="Annual binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument(
        "--carbon-var", default="TOTSOMC", help="Carbon variable (default: TOTSOMC)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:      {args.input_file}")
    print(f"Output:     {args.output_file}")
    print(f"Carbon var: {args.carbon_var}")
    print()

    plot_carbon_water(args.input_file, args.output_file, args.carbon_var)


if __name__ == "__main__":
    main()
