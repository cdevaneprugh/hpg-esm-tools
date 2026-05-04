#!/usr/bin/env python3
"""
plot_timeseries_last20.py - Recent period time series by HAND zone

PURPOSE:
    Plot last N years of column-level data, grouped by HAND elevation zone
    (TAI / Transition / Upland). Shows how different parts of the hillslope
    contribute to the gridcell average.

    Supports any single-aspect hillslope layout (1xN columns).

USAGE:
    python3 plot_timeseries_last20.py <input_file> <output_file> <variable> [--years=N]
    python3 plot_timeseries_last20.py -h

ARGUMENTS:
    input_file      Annual binned h1 NetCDF file (e.g., combined_h1_1yr.nc)
    output_file     Output PNG filename
    variable        Variable name to plot (e.g., GPP, ZWT, TOTECOSYSC)
    --years=N       Number of recent years to plot (default: 20)

EXAMPLE:
    python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/GPP_last20.png GPP
    python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/ZWT.png ZWT --years=50

OUTPUT:
    2-panel figure:
    - Top: Variable by HAND zone (TAI / Transition / Upland)
    - Bottom: Individual column traces colored by HAND elevation

NOTES:
    - Uses weighted averaging based on column area fractions (cols1d_wtgcell)
    - HAND zones: TAI (<0.5m), Transition (0.5-5m), Upland (>=5m)
    - Bareground column excluded from zone grouping, included in gridcell avg
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
import matplotlib.cm as cm
import numpy as np

# =============================================================================
# Constants
# =============================================================================

# HAND elevation zone boundaries (meters)
TAI_THRESHOLD = 0.5
TRANSITION_THRESHOLD = 5.0

ZONE_STYLES = {
    "TAI (<0.5m)": {"color": "#2166ac"},
    "Transition (0.5-5m)": {"color": "#b2182b"},
    "Upland (>=5m)": {"color": "#4daf4a"},
}


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


def classify_zones(elevations):
    """Classify columns into TAI / Transition / Upland by HAND elevation."""
    zones = {}
    tai = np.where(elevations < TAI_THRESHOLD)[0]
    trans = np.where(
        (elevations >= TAI_THRESHOLD) & (elevations < TRANSITION_THRESHOLD)
    )[0]
    upland = np.where(elevations >= TRANSITION_THRESHOLD)[0]

    if len(tai) > 0:
        zones["TAI (<0.5m)"] = tai
    if len(trans) > 0:
        zones["Transition (0.5-5m)"] = trans
    if len(upland) > 0:
        zones["Upland (>=5m)"] = upland
    return zones


# =============================================================================
# Main plotting function
# =============================================================================


def plot_timeseries_last20(
    input_file: str, output_file: str, variable: str, n_years: int = 20
) -> None:
    """
    Plot last N years of column data by HAND zone and individual columns.

    Parameters
    ----------
    input_file : str
        Path to annual binned h1 NetCDF file
    output_file : str
        Path for output PNG file
    variable : str
        Name of variable to plot
    n_years : int
        Number of recent years to plot (default: 20)
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

    var_data = ds[variable].values
    if var_data.ndim != 2:
        print(f"ERROR: Expected 2D (time, column), got shape {var_data.shape}")
        ds.close()
        sys.exit(1)

    units = ds[variable].attrs.get("units", "")

    # -------------------------------------------------------------------------
    # Detect columns
    # -------------------------------------------------------------------------
    h_cols = detect_hillslope_columns(ds)
    n_hillslope = len(h_cols)
    n_total = var_data.shape[1]

    elevations = ds["hillslope_elev"].values[h_cols]
    weights_all = ds["cols1d_wtgcell"].values[:n_total]

    ds.close()

    # -------------------------------------------------------------------------
    # Extract last N years
    # -------------------------------------------------------------------------
    total_years = var_data.shape[0]
    start_idx = max(0, total_years - n_years)
    var_data = var_data[start_idx:]
    actual_years = var_data.shape[0]

    # -------------------------------------------------------------------------
    # Gridcell average (all columns including bareground)
    # -------------------------------------------------------------------------
    gridcell_avg = (var_data * weights_all).sum(axis=1)

    # -------------------------------------------------------------------------
    # Zone-grouped weighted averages
    # -------------------------------------------------------------------------
    zones = classify_zones(elevations)
    zone_data = {}
    for zone_name, zone_indices in zones.items():
        # zone_indices are relative to h_cols; convert to absolute
        abs_cols = h_cols[zone_indices]
        zone_weights = weights_all[abs_cols]
        w_sum = zone_weights.sum()
        if w_sum > 0:
            renorm = zone_weights / w_sum
            zone_data[zone_name] = (var_data[:, abs_cols] * renorm).sum(axis=1)

    # -------------------------------------------------------------------------
    # Select representative columns for bottom panel
    # -------------------------------------------------------------------------
    if n_hillslope <= 8:
        rep_indices = np.arange(n_hillslope)
    else:
        rep_indices = np.linspace(0, n_hillslope - 1, 6, dtype=int)
        rep_indices = np.unique(rep_indices)

    # -------------------------------------------------------------------------
    # Create 2-panel figure
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    time = np.arange(actual_years)
    ylabel = f"{variable} ({units})" if units else variable

    # -------------------------------------------------------------------------
    # Top panel: By HAND zone
    # -------------------------------------------------------------------------
    ax1.plot(
        time,
        gridcell_avg,
        linewidth=2,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Gridcell Avg",
        zorder=1,
    )

    for zone_name, data in zone_data.items():
        style = ZONE_STYLES[zone_name]
        ax1.plot(
            time,
            data,
            linewidth=2,
            color=style["color"],
            alpha=0.85,
            label=zone_name,
            zorder=2,
        )

    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(
        f"{variable} (Annual) - Last {actual_years} Years\nBy HAND Zone",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="best", fontsize=10)

    # -------------------------------------------------------------------------
    # Bottom panel: Individual column traces
    # -------------------------------------------------------------------------
    ax2.plot(
        time,
        gridcell_avg,
        linewidth=2,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Gridcell Avg",
        zorder=1,
    )

    cmap = cm.viridis
    elev_min, elev_max = elevations[0], elevations[-1]
    elev_range = elev_max - elev_min if elev_max > elev_min else 1.0

    for i in rep_indices:
        abs_col = h_cols[i]
        elev = elevations[i]
        norm_elev = (elev - elev_min) / elev_range
        color = cmap(norm_elev)
        ax2.plot(
            time,
            var_data[:, abs_col],
            linewidth=1.5,
            color=color,
            alpha=0.8,
            label=f"HAND {elev:.1f}m",
            zorder=2,
        )

    ax2.set_xlabel(f"Year (Last {actual_years})", fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(
        "Individual Columns (by HAND elevation)", fontsize=13, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="best", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")
    print(
        f"\n  Columns: {n_hillslope} hillslope + "
        f"{n_total - n_hillslope} other = {n_total} total"
    )
    for zone_name, zone_indices in zones.items():
        print(f"  {zone_name}: {len(zone_indices)} columns")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot recent time series by HAND zone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_1yr.nc plots/GPP_last20.png GPP
  %(prog)s data/combined_h1_1yr.nc plots/ZWT_last50.png ZWT --years=50
        """,
    )
    parser.add_argument("input_file", help="Annual binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument("variable", help="Variable name to plot")
    parser.add_argument(
        "--years",
        type=int,
        default=20,
        help="Number of recent years to plot (default: 20)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Variable: {args.variable}")
    print(f"Years:    {args.years}")
    print()

    plot_timeseries_last20(args.input_file, args.output_file, args.variable, args.years)


if __name__ == "__main__":
    main()
