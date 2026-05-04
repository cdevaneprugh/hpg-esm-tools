#!/usr/bin/env python3
"""
plot_hillslope_cross_section.py - Filled hillslope cross-section

PURPOSE:
    Visualize the hillslope as a filled cross-section showing land surface,
    water table position, surface ponding (if H2OSFC present), and
    saturated/unsaturated zones. Two panels compare early vs late periods.

USAGE:
    python3 plot_hillslope_cross_section.py <input_file> <output_file>
    python3 plot_hillslope_cross_section.py -h

ARGUMENTS:
    input_file      Binned h1 NetCDF file (e.g., combined_h1_20yr.nc)
    output_file     Output PNG filename

EXAMPLE:
    python3 plot_hillslope_cross_section.py data/combined_h1_20yr.nc plots/cross_section.png

OUTPUT:
    2-panel figure:
    - Top: Early simulation period
    - Bottom: Recent period
    Each panel shows a filled cross-section with surface topography,
    water table, and zone shading.
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
from matplotlib.patches import Patch
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


def plot_cross_section(input_file: str, output_file: str) -> None:
    """
    Plot filled hillslope cross-section for early and late periods.

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

    h_cols = detect_hillslope_columns(ds)
    n_cols = len(h_cols)

    zwt = ds["ZWT"].values
    h_elev = ds["hillslope_elev"].values[h_cols]
    h_dist = ds["hillslope_distance"].values[h_cols]
    mcdate = ds["mcdate"].values

    has_h2osfc = "H2OSFC" in ds
    h2osfc = ds["H2OSFC"].values if has_h2osfc else None

    ds.close()

    # -------------------------------------------------------------------------
    # Time periods
    # -------------------------------------------------------------------------
    n_times = zwt.shape[0]
    time_indices = [0, n_times - 1]
    years = mcdate // 10000
    time_labels = [f"Year {years[0]} (Early)", f"Year {years[-1]} (Recent)"]

    # -------------------------------------------------------------------------
    # Interpolation points for smooth fills
    # -------------------------------------------------------------------------
    dist_smooth = np.linspace(h_dist[0], h_dist[-1], 200)
    elev_smooth = np.interp(dist_smooth, h_dist, h_elev)

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, t_idx, label in zip(axes, time_indices, time_labels):
        wt_elev = h_elev - zwt[t_idx, h_cols]
        wt_smooth = np.interp(dist_smooth, h_dist, wt_elev)

        # Clamp water table to not exceed surface
        wt_clamped = np.minimum(wt_smooth, elev_smooth)

        # Y-axis range
        y_min = min(np.min(wt_smooth), 0) - 1
        y_max = np.max(h_elev) * 1.2

        # Saturated zone: below water table
        ax.fill_between(
            dist_smooth, y_min, wt_clamped, alpha=0.35, color="#4393c3", linewidth=0
        )

        # Unsaturated zone: between water table and surface
        ax.fill_between(
            dist_smooth,
            wt_clamped,
            elev_smooth,
            alpha=0.35,
            color="#d6b06c",
            linewidth=0,
        )

        # Surface ponding (where WT is above ground surface)
        if has_h2osfc:
            pond_mm = h2osfc[t_idx, h_cols]
            # Convert mm to meters of depth above surface
            pond_m = pond_mm / 1000.0
            pond_top = h_elev + pond_m
            pond_smooth = np.interp(dist_smooth, h_dist, pond_top)
            # Only show where ponding exists
            ponded = pond_smooth > elev_smooth + 0.001
            if np.any(ponded):
                ax.fill_between(
                    dist_smooth,
                    elev_smooth,
                    pond_smooth,
                    where=ponded,
                    alpha=0.5,
                    color="#2166ac",
                    linewidth=0,
                )

        # Land surface line
        ax.plot(dist_smooth, elev_smooth, "-", linewidth=2.5, color="#8B4513", zorder=3)

        # Water table line
        ax.plot(dist_smooth, wt_smooth, "-", linewidth=2, color="#2166ac", zorder=2)

        # Column markers on surface
        ax.plot(
            h_dist,
            h_elev,
            "|",
            markersize=8,
            color="#8B4513",
            markeredgewidth=1.5,
            zorder=4,
        )

        # Stream level
        ax.axhline(0, color="darkblue", linestyle=":", linewidth=1, alpha=0.5)

        # Annotate a few ZWT values
        if n_cols <= 10:
            indices = range(n_cols)
        else:
            indices = np.linspace(0, n_cols - 1, 5, dtype=int)

        for i in indices:
            zwt_val = zwt[t_idx, h_cols[i]]
            mid_y = (h_elev[i] + wt_elev[i]) / 2
            ax.annotate(
                f"ZWT={zwt_val:.1f}m",
                xy=(h_dist[i], mid_y),
                fontsize=7,
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

        ax.set_ylabel("Elevation (m)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.2, linestyle="--")

        # Legend via patches
        legend_elements = [
            Patch(facecolor="#8B4513", alpha=0.8, label="Land Surface"),
            Patch(facecolor="#2166ac", alpha=0.5, label="Water Table"),
            Patch(facecolor="#4393c3", alpha=0.35, label="Saturated Zone"),
            Patch(facecolor="#d6b06c", alpha=0.35, label="Unsaturated Zone"),
        ]
        if has_h2osfc:
            legend_elements.append(
                Patch(facecolor="#2166ac", alpha=0.5, label="Surface Ponding")
            )
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Distance from Stream (m)", fontsize=11)

    fig.suptitle(
        f"Hillslope Cross-Section ({n_cols} columns)\nYears {years[0]}-{years[-1]}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")
    if has_h2osfc:
        print("  (includes surface ponding from H2OSFC)")


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot filled hillslope cross-section",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/combined_h1_20yr.nc plots/cross_section.png
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

    plot_cross_section(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
