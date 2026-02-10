#!/usr/bin/env python
"""
Stage 9: Accumulation Threshold Sensitivity Analysis

Test different accumulation thresholds to determine if the remaining area
fraction correlation gap (0.82 vs ~1.0) can be improved.

Current state (Stage 8):
- Height:   0.9999 (perfect)
- Distance: 0.9982 (perfect)
- Slope:    0.9966 (perfect)
- Aspect:   0.9999 (perfect)
- Width:    0.9597 (excellent)
- Area:     0.8200 (good, ~18% unexplained)

This script tests thresholds: 20, 34 (current), 50, 100, 200 cells
and reports the area fraction correlation for each.

Expected runtime: ~20-30 minutes (4-5 threshold values × ~5 min each)
"""

import os
import sys
import time
import json
import numpy as np

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

# Add parent directory for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysheds.pgrid import Grid

# Local modules
from spatial_scale import DTR, RE

try:
    import rasterio
    from rasterio.windows import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Configuration
MERIT_DEM_PATH = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif"
PUBLISHED_NC = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/reference/hillslopes_0.9x1.25_c240416.nc"
OUTPUT_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage9"

# Thresholds to test
THRESHOLDS = [20, 34, 50, 100, 200]

# Processing parameters
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0

# Target gridcell (same as stage3)
TARGET_GRIDCELL = {
    "lon_min": -93.1250,
    "lon_max": -91.8750,
    "lat_min": 32.0419,
    "lat_max": 32.9843,
    "center_lon": -92.5000,
    "center_lat": 32.5131,
}

EXPANSION_FACTOR = 1.5

ASPECT_BINS = [
    (315, 45),  # North
    (45, 135),  # East
    (135, 225),  # South
    (225, 315),  # West
]
ASPECT_NAMES = ["North", "East", "South", "West"]


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def compute_gridcell_indices(
    expanded_transform: rasterio.Affine,
    expanded_shape: tuple,
    gridcell: dict,
) -> tuple:
    """Compute row/col slices for extracting gridcell from expanded region."""
    col_min = int((gridcell["lon_min"] - expanded_transform.c) / expanded_transform.a)
    col_max = int((gridcell["lon_max"] - expanded_transform.c) / expanded_transform.a)
    row_min = int((gridcell["lat_max"] - expanded_transform.f) / expanded_transform.e)
    row_max = int((gridcell["lat_min"] - expanded_transform.f) / expanded_transform.e)

    col_min = max(0, col_min)
    col_max = min(expanded_shape[1], col_max)
    row_min = max(0, row_min)
    row_max = min(expanded_shape[0], row_max)

    return slice(row_min, row_max), slice(col_min, col_max)


def get_aspect_mask(aspect: np.ndarray, aspect_bin: tuple) -> np.ndarray:
    """Create mask for pixels within an aspect bin."""
    lower, upper = aspect_bin
    if lower > upper:
        return (aspect >= lower) | (aspect < upper)
    else:
        return (aspect >= lower) & (aspect < upper)


def compute_hand_bins(
    hand: np.ndarray,
    aspect: np.ndarray,
    aspect_bins: list,
    bin1_max: float = 2.0,
    min_aspect_fraction: float = 0.01,
) -> np.ndarray:
    """Compute HAND bin boundaries following Swenson's method."""
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]
    aspect_valid = aspect[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size

    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    if initial_q25 > bin1_max:
        adjusted_bin1_max = bin1_max

        for asp_idx, (asp_low, asp_high) in enumerate(aspect_bins):
            if asp_low > asp_high:
                asp_mask = (aspect_valid >= asp_low) | (aspect_valid < asp_high)
            else:
                asp_mask = (aspect_valid >= asp_low) & (aspect_valid < asp_high)

            hand_asp = hand_valid[asp_mask]
            if hand_asp.size > 0:
                hand_asp_sorted = np.sort(hand_asp)
                below_threshold = (
                    np.sum(hand_asp_sorted <= bin1_max) / hand_asp_sorted.size
                )

                if below_threshold < min_aspect_fraction:
                    idx_1pct = max(
                        0, int(min_aspect_fraction * hand_asp_sorted.size) - 1
                    )
                    bmin = hand_asp_sorted[idx_1pct]
                    adjusted_bin1_max = max(adjusted_bin1_max, bmin)

        above_bin1 = hand_sorted[hand_sorted > adjusted_bin1_max]
        if above_bin1.size > 0:
            n_above = above_bin1.size
            b33 = (
                above_bin1[int(0.33 * n_above) - 1]
                if n_above > 0
                else adjusted_bin1_max * 2
            )
            b66 = (
                above_bin1[int(0.66 * n_above) - 1]
                if n_above > 0
                else adjusted_bin1_max * 3
            )
            bounds = np.array([0, adjusted_bin1_max, b33, b66, 1e6])
        else:
            bounds = np.array(
                [
                    0,
                    adjusted_bin1_max,
                    adjusted_bin1_max * 2,
                    adjusted_bin1_max * 4,
                    1e6,
                ]
            )
    else:
        quartiles = [0.25, 0.5, 0.75, 1.0]
        bounds = [0]
        for q in quartiles:
            idx = max(0, int(q * n) - 1)
            bounds.append(hand_sorted[idx])
        bounds[-1] = 1e6
        bounds = np.array(bounds)

    return bounds


def compute_pixel_areas(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Compute pixel areas using spherical coordinates (Swenson method)."""
    phi = DTR * lon
    theta = DTR * (90.0 - lat)
    dphi = np.abs(phi[1] - phi[0])
    dtheta = np.abs(theta[0] - theta[1])
    sin_theta = np.sin(theta)
    ncols = len(lon)
    area = np.tile(sin_theta.reshape(-1, 1), (1, ncols))
    area = area * dtheta * dphi * RE**2
    return area


def compute_area_fractions_for_threshold(
    grid,
    accum_threshold: int,
    gc_row_slice,
    gc_col_slice,
    lon_center,
    lat_center,
    dirmap,
) -> np.ndarray:
    """
    Compute hillslope area fractions for a given accumulation threshold.

    Returns 16-element array of area fractions (4 aspects × 4 elevation bins).
    """
    acc = grid.acc

    # Stream mask using accumulation threshold
    acc_mask = acc > accum_threshold

    # Create channel mask
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=dirmap, routing="d8")

    stream_cells = np.sum(grid.channel_mask > 0)
    if stream_cells == 0:
        print(f"    Warning: No stream cells with threshold {accum_threshold}")
        return np.ones(16) / 16

    # Compute HAND and DTND
    grid.compute_hand(
        "fdir", "dem", grid.channel_mask, grid.channel_id, dirmap=dirmap, routing="d8"
    )
    hand = grid.hand

    # Get slope and aspect using pgrid method
    grid.slope_aspect("dem")
    aspect = np.array(grid.aspect)

    # Extract gridcell region
    hand_center = np.array(hand)[gc_row_slice, gc_col_slice]
    aspect_center = aspect[gc_row_slice, gc_col_slice]

    # Compute pixel areas
    pixel_areas = compute_pixel_areas(lon_center, lat_center)

    # Flatten arrays
    hand_flat = hand_center.flatten()
    aspect_flat = aspect_center.flatten()
    area_flat = pixel_areas.flatten()

    # Filter valid data
    valid = (hand_flat >= 0) & np.isfinite(hand_flat)

    # Compute HAND bin boundaries
    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )

    # Compute area fractions for each aspect-elevation bin
    area_fractions = []
    total_area = np.sum(area_flat[valid])

    for asp_idx, asp_bin in enumerate(ASPECT_BINS):
        asp_mask = get_aspect_mask(aspect_flat, asp_bin) & valid

        for h_idx in range(N_HAND_BINS):
            h_lower = hand_bounds[h_idx]
            h_upper = hand_bounds[h_idx + 1]

            hand_mask = (hand_flat >= h_lower) & (hand_flat < h_upper)
            bin_mask = asp_mask & hand_mask

            if np.any(bin_mask):
                bin_area = np.sum(area_flat[bin_mask])
                frac = bin_area / total_area
            else:
                frac = 0.0

            area_fractions.append(frac)

    return np.array(area_fractions)


def load_published_area_fractions() -> np.ndarray:
    """Load and compute area fractions from published dataset."""
    if not HAS_XARRAY:
        raise RuntimeError("xarray required")

    ds = xr.open_dataset(PUBLISHED_NC)

    # Find matching gridcell
    target_lon = 267.5
    target_lat = 32.5

    longxy = ds["LONGXY"].values
    latixy = ds["LATIXY"].values

    dist = np.sqrt((longxy - target_lon) ** 2 + (latixy - target_lat) ** 2)
    min_idx = np.unravel_index(dist.argmin(), dist.shape)
    lat_idx, lon_idx = min_idx

    # Extract published areas
    pub_areas = ds["hillslope_area"].values[:, lat_idx, lon_idx]
    ds.close()

    # Compute fractions
    pub_total = np.sum(pub_areas)
    if pub_total > 0:
        return pub_areas / pub_total
    else:
        return pub_areas


def main():
    start_time = time.time()

    print_section("Stage 9: Accumulation Threshold Sensitivity Analysis")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load published data
    print("Loading published area fractions...")
    pub_area_frac = load_published_area_fractions()
    print(f"  Published area fractions: {pub_area_frac}")

    # Load and prepare DEM
    print_section("Loading DEM and Computing Base Flow Routing")

    gc = TARGET_GRIDCELL
    lon_width = gc["lon_max"] - gc["lon_min"]
    lat_height = gc["lat_max"] - gc["lat_min"]

    expanded_bounds = {
        "lon_min": gc["center_lon"] - EXPANSION_FACTOR * lon_width / 2,
        "lon_max": gc["center_lon"] + EXPANSION_FACTOR * lon_width / 2,
        "lat_min": gc["center_lat"] - EXPANSION_FACTOR * lat_height / 2,
        "lat_max": gc["center_lat"] + EXPANSION_FACTOR * lat_height / 2,
    }

    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required")

    with rasterio.open(MERIT_DEM_PATH) as src:
        src_nodata = src.nodata
        window = from_bounds(
            expanded_bounds["lon_min"],
            expanded_bounds["lat_min"],
            expanded_bounds["lon_max"],
            expanded_bounds["lat_max"],
            src.transform,
        )
        dem_data = src.read(1, window=window)
        expanded_transform = src.window_transform(window)

    print(f"  DEM shape: {dem_data.shape}")

    # Get gridcell extraction indices
    gc_row_slice, gc_col_slice = compute_gridcell_indices(
        expanded_transform, dem_data.shape, gc
    )

    # Compute coordinate arrays
    nrows, ncols = dem_data.shape
    lon = np.array(
        [expanded_transform.c + expanded_transform.a * (i + 0.5) for i in range(ncols)]
    )
    lat = np.array(
        [expanded_transform.f + expanded_transform.e * (j + 0.5) for j in range(nrows)]
    )
    lon_center = lon[gc_col_slice]
    lat_center = lat[gc_row_slice]

    # Create pysheds grid
    from pyproj import Proj as PyprojProj

    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=expanded_transform,
        crs=PyprojProj("EPSG:4326"),
        nodata=src_nodata if src_nodata is not None else -9999,
    )

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Condition DEM (only need to do this once)
    print("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")
    grid.resolve_flats("flooded", out_name="inflated")

    # Flow direction (only need to do this once)
    print("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=dirmap, routing="d8")

    # Flow accumulation (only need to do this once)
    print("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=dirmap, routing="d8")

    print_section("Testing Accumulation Thresholds")

    results = {
        "thresholds": [],
        "correlations": [],
        "area_fractions": {},
        "published_area_fractions": pub_area_frac.tolist(),
    }

    for threshold in THRESHOLDS:
        print(f"\n  Testing threshold = {threshold} cells...")
        t0 = time.time()

        our_area_frac = compute_area_fractions_for_threshold(
            grid, threshold, gc_row_slice, gc_col_slice, lon_center, lat_center, dirmap
        )

        # Compute correlation
        valid = np.isfinite(our_area_frac) & np.isfinite(pub_area_frac)
        if np.std(our_area_frac[valid]) > 0 and np.std(pub_area_frac[valid]) > 0:
            correlation = np.corrcoef(our_area_frac[valid], pub_area_frac[valid])[0, 1]
        else:
            correlation = np.nan

        elapsed = time.time() - t0

        print(f"    Area fraction correlation: {correlation:.4f}")
        print(f"    Time: {elapsed:.1f} seconds")

        results["thresholds"].append(threshold)
        results["correlations"].append(float(correlation))
        results["area_fractions"][str(threshold)] = our_area_frac.tolist()

    print_section("Results Summary")

    print(f"{'Threshold':<12} {'Correlation':<15} {'Change from 34':<15}")
    print("-" * 42)

    baseline_corr = None
    for threshold, corr in zip(results["thresholds"], results["correlations"]):
        if threshold == 34:
            baseline_corr = corr
            change_str = "(baseline)"
        elif baseline_corr is not None:
            change = corr - baseline_corr
            change_str = f"{change:+.4f}"
        else:
            change_str = "N/A"

        print(f"{threshold:<12} {corr:<15.4f} {change_str:<15}")

    # Find best threshold
    best_idx = np.argmax(results["correlations"])
    best_threshold = results["thresholds"][best_idx]
    best_corr = results["correlations"][best_idx]

    print(f"\nBest threshold: {best_threshold} cells (correlation: {best_corr:.4f})")

    if best_threshold != 34:
        improvement = best_corr - baseline_corr
        print(f"Improvement over current (34): {improvement:+.4f}")
    else:
        print("Current threshold (34) is optimal or tied for best.")

    # Determine recommendation
    if best_corr > 0.85:
        recommendation = "Significant improvement possible"
    elif best_corr > baseline_corr + 0.02:
        recommendation = "Marginal improvement possible"
    else:
        recommendation = "Threshold sensitivity is low - proceed to OSBS implementation"

    results["summary"] = {
        "best_threshold": best_threshold,
        "best_correlation": float(best_corr),
        "baseline_threshold": 34,
        "baseline_correlation": float(baseline_corr) if baseline_corr else None,
        "recommendation": recommendation,
    }

    print(f"\nRecommendation: {recommendation}")

    # Save results
    print_section("Saving Results")

    json_path = os.path.join(OUTPUT_DIR, "stage9_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save summary text
    summary_path = os.path.join(OUTPUT_DIR, "stage9_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 9: Accumulation Threshold Sensitivity Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write("Purpose: Test if different accumulation thresholds can improve\n")
        f.write("the area fraction correlation (currently 0.82).\n\n")

        f.write("Results:\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'Threshold':<12} {'Correlation':<15}\n")
        f.write("-" * 42 + "\n")
        for threshold, corr in zip(results["thresholds"], results["correlations"]):
            marker = " *" if threshold == best_threshold else ""
            f.write(f"{threshold:<12} {corr:<15.4f}{marker}\n")
        f.write("-" * 42 + "\n")
        f.write("* Best threshold\n\n")

        f.write(f"Best threshold: {best_threshold} cells\n")
        f.write(f"Best correlation: {best_corr:.4f}\n")
        f.write(f"Baseline (34 cells): {baseline_corr:.4f}\n\n")

        f.write(f"Recommendation: {recommendation}\n\n")

        f.write(f"Total processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # Generate plot if matplotlib available
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Correlation vs threshold
        ax = axes[0]
        ax.plot(
            results["thresholds"],
            results["correlations"],
            "bo-",
            markersize=10,
            linewidth=2,
        )
        ax.axhline(
            y=baseline_corr,
            color="r",
            linestyle="--",
            label=f"Baseline (34 cells): {baseline_corr:.4f}",
        )
        ax.scatter(
            [best_threshold],
            [best_corr],
            color="green",
            s=200,
            zorder=5,
            marker="*",
            label=f"Best: {best_threshold} cells",
        )
        ax.set_xlabel("Accumulation Threshold (cells)", fontsize=12)
        ax.set_ylabel("Area Fraction Correlation", fontsize=12)
        ax.set_title("Threshold Sensitivity Analysis", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(results["correlations"]) - 0.05, 1.0])

        # Plot 2: Area fractions comparison for best threshold
        ax = axes[1]
        x = np.arange(16)
        width = 0.35

        best_fracs = results["area_fractions"][str(best_threshold)]

        ax.bar(x - width / 2, pub_area_frac, width, label="Published", alpha=0.7)
        ax.bar(
            x + width / 2,
            best_fracs,
            width,
            label=f"Ours (threshold={best_threshold})",
            alpha=0.7,
        )

        ax.set_xlabel("Hillslope Element Index", fontsize=12)
        ax.set_ylabel("Area Fraction", fontsize=12)
        ax.set_title(f"Area Fractions (r={best_corr:.4f})", fontsize=14)
        ax.legend()
        ax.set_xticks(x)

        # Add aspect labels
        for i, name in enumerate(ASPECT_NAMES):
            ax.axvline(x=i * 4 - 0.5, color="gray", linestyle=":", alpha=0.5)
            ax.text(
                i * 4 + 1.5, ax.get_ylim()[1] * 0.95, name, ha="center", fontsize=10
            )

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "stage9_sensitivity_analysis.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")

    print_section("Stage 9 Complete")

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )


if __name__ == "__main__":
    main()
