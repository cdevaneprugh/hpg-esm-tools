#!/usr/bin/env python3
"""
Plot OSBS 1m hillslope profiles against Swenson MERIT 90m reference.

Two-panel line plot: elevation and width vs distance from stream.
Swenson's 4-aspect data is pre-averaged into a single profile (4 bins).
Our 1x8 data plotted directly (8 bins).

Usage:
    python scripts/osbs/plot_hillslope_comparison.py [input_nc] [output_png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

# =============================================================================
# Swenson MERIT reference (aspect-averaged from hillslopes_0.9x1.25_c240416.nc)
#
# Source: data/reference/hillslopes_0.9x1.25_c240416.nc at lsmlat=127, lsmlon=222
# 4 aspects (N, E, S, W) x 4 equal-area bins, averaged per bin position.
# Resolution: 90m MERIT DEM, global dataset.
# =============================================================================
SWENSON_ELEVATION = np.array([0.18, 1.24, 2.80, 8.10])  # meters
SWENSON_DISTANCE = np.array([71.42, 208.59, 345.03, 527.19])  # meters
SWENSON_WIDTH = np.array([700.58, 586.62, 492.32, 374.60])  # meters

# =============================================================================
# Defaults
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = (
    BASE_DIR
    / "output"
    / "osbs"
    / "2026-04-09_production"
    / "hillslopes_osbs_production_c260409.nc"
)
DEFAULT_OUTPUT = BASE_DIR / "output" / "plots" / "hillslope_comparison.png"


def plot_comparison(input_file: str, output_file: str) -> None:
    """Plot OSBS 1m vs Swenson MERIT hillslope profiles."""

    ds = xr.open_dataset(input_file)

    elev = ds["hillslope_elevation"].values.squeeze()
    dist = ds["hillslope_distance"].values.squeeze()
    width = ds["hillslope_width"].values.squeeze()

    ds.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: Elevation ---
    ax1.plot(
        dist,
        elev,
        marker="o",
        color="#e74c3c",
        linewidth=2,
        markersize=7,
        label="OSBS 1m (1x8, this work)",
        alpha=0.9,
    )
    ax1.plot(
        SWENSON_DISTANCE,
        SWENSON_ELEVATION,
        marker="s",
        color="#3498db",
        linewidth=2,
        markersize=7,
        label="Swenson MERIT 90m (4x4 avg)",
        alpha=0.9,
        linestyle="--",
    )

    ax1.set_xlabel("Distance from Stream (m)", fontsize=12)
    ax1.set_ylabel("Elevation above Stream (m)", fontsize=12)
    ax1.set_title("Elevation Profile", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.set_xlim(0, max(dist.max(), SWENSON_DISTANCE.max()) * 1.1)
    ax1.set_ylim(0, max(elev.max(), SWENSON_ELEVATION.max()) * 1.15)

    # --- Right panel: Width ---
    ax2.plot(
        dist,
        width,
        marker="o",
        color="#e74c3c",
        linewidth=2,
        markersize=7,
        label="OSBS 1m (1x8, this work)",
        alpha=0.9,
    )
    ax2.plot(
        SWENSON_DISTANCE,
        SWENSON_WIDTH,
        marker="s",
        color="#3498db",
        linewidth=2,
        markersize=7,
        label="Swenson MERIT 90m (4x4 avg)",
        alpha=0.9,
        linestyle="--",
    )

    ax2.set_xlabel("Distance from Stream (m)", fontsize=12)
    ax2.set_ylabel("Hillslope Width (m)", fontsize=12)
    ax2.set_title("Width Profile", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(0, max(dist.max(), SWENSON_DISTANCE.max()) * 1.1)
    ax2.set_ylim(0, max(width.max(), SWENSON_WIDTH.max()) * 1.15)

    fig.suptitle(
        "Hillslope Geometry: OSBS 1m LIDAR vs Swenson MERIT 90m",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot OSBS 1m vs Swenson MERIT hillslope profiles"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Our hillslope NetCDF (default: latest production run)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help="Output PNG (default: output/plots/hillslope_comparison.png)",
    )
    args = parser.parse_args()

    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    plot_comparison(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
