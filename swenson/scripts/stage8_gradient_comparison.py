#!/usr/bin/env python
"""
Stage 8: Gradient/Aspect Calculation Comparison

Compare our gradient calculation method to Swenson's pgrid Horn 1981 method
to identify if this is the root cause of the ~2.5% East/South boundary
classification discrepancy.

Key finding from investigation:
- ~2.5% of pixels near the 135° boundary (E/S) are classified differently
- East aspect is +2.4% (too high), South is -2.6% (too low)
- Total area is correct (0.07% error), so this is a classification issue

Methods compared:
1. Our method: np.gradient() with custom averaging and uniform dx
2. Swenson's method: Horn 1981 8-neighbor stencil with per-pixel dx

Data paths:
- MERIT DEM: swenson/data/MERIT_DEM_sample/n30w095_dem.tif
- Output: swenson/output/stage8/

Expected runtime: ~5 minutes on 4 cores
"""

import os
import sys
import time
import json
import numpy as np

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid

# Local modules (for our method)
from spatial_scale import calc_gradient, DTR

try:
    import rasterio
    from rasterio.windows import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configuration
MERIT_DEM_PATH = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/MERIT_DEM_sample/n30w095_dem.tif"
OUTPUT_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/stage8"

# Target gridcell boundaries (same as stage3)
TARGET_GRIDCELL = {
    "lon_min": -93.1250,
    "lon_max": -91.8750,
    "lat_min": 32.0419,
    "lat_max": 32.9843,
    "center_lon": -92.5000,
    "center_lat": 32.5131,
}

# Aspect bin definitions
ASPECT_BINS = [
    (315, 45),   # North
    (45, 135),   # East
    (135, 225),  # South
    (225, 315),  # West
]
ASPECT_NAMES = ["North", "East", "South", "West"]


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def compute_aspect_our_method(
    dem: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """
    Compute aspect using our method (np.gradient with averaging).

    Returns aspect in degrees (0-360, clockwise from North).
    """
    dzdx, dzdy = calc_gradient(dem, lon, lat)
    aspect = np.arctan2(-dzdx, -dzdy) / DTR
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    return aspect


def compute_aspect_pgrid_method(grid: Grid) -> np.ndarray:
    """
    Compute aspect using pgrid's Horn 1981 method.

    Returns aspect in degrees (0-360, clockwise from North).
    """
    grid.slope_aspect("dem")
    return np.array(grid.aspect)


def classify_aspect(aspect: np.ndarray) -> np.ndarray:
    """
    Classify aspect into bins (0=North, 1=East, 2=South, 3=West).
    """
    classification = np.full(aspect.shape, -1, dtype=int)

    for idx, (lower, upper) in enumerate(ASPECT_BINS):
        if lower > upper:  # North wraps around
            mask = (aspect >= lower) | (aspect < upper)
        else:
            mask = (aspect >= lower) & (aspect < upper)
        classification[mask] = idx

    return classification


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("Stage 8: Gradient Calculation Comparison")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not HAS_RASTERIO:
        print("ERROR: rasterio required")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 1: Load DEM for target gridcell
    # -------------------------------------------------------------------------
    print_section("Step 1: Loading DEM for Target Gridcell")

    gc = TARGET_GRIDCELL

    with rasterio.open(MERIT_DEM_PATH) as src:
        src_nodata = src.nodata

        # Get window for gridcell
        window = from_bounds(
            gc["lon_min"],
            gc["lat_min"],
            gc["lon_max"],
            gc["lat_max"],
            src.transform,
        )
        dem_data = src.read(1, window=window)
        transform = src.window_transform(window)

    nrows, ncols = dem_data.shape
    lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
    lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    print(f"  DEM shape: {dem_data.shape}")
    print(f"  Lon range: [{lon[0]:.4f}, {lon[-1]:.4f}]")
    print(f"  Lat range: [{lat[0]:.4f}, {lat[-1]:.4f}]")

    # -------------------------------------------------------------------------
    # Step 2: Compute aspect using BOTH methods
    # -------------------------------------------------------------------------
    print_section("Step 2: Computing Aspect Using Both Methods")

    # Method A: Our implementation
    t0 = time.time()
    aspect_ours = compute_aspect_our_method(dem_data, lon, lat)
    time_ours = time.time() - t0
    print(f"  Our method: {time_ours:.2f}s")
    print(f"    Range: [{np.nanmin(aspect_ours):.1f}, {np.nanmax(aspect_ours):.1f}]")

    # Method B: pgrid Horn 1981
    t0 = time.time()
    from pyproj import Proj as PyprojProj

    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=transform,
        crs=PyprojProj("EPSG:4326"),
        nodata=src_nodata if src_nodata is not None else -9999,
    )
    aspect_pgrid = compute_aspect_pgrid_method(grid)
    time_pgrid = time.time() - t0
    print(f"  pgrid method: {time_pgrid:.2f}s")
    print(f"    Range: [{np.nanmin(aspect_pgrid):.1f}, {np.nanmax(aspect_pgrid):.1f}]")

    # -------------------------------------------------------------------------
    # Step 3: Compare aspect values
    # -------------------------------------------------------------------------
    print_section("Step 3: Comparing Aspect Values")

    # Create valid mask (exclude edges that pgrid doesn't compute)
    valid_pgrid = aspect_pgrid != 0  # pgrid sets edges to 0
    valid_ours = np.isfinite(aspect_ours)
    valid = valid_pgrid & valid_ours

    # Compute aspect difference (handling wraparound)
    diff = aspect_ours - aspect_pgrid
    # Handle 360/0 wraparound
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)

    print(f"  Valid pixels: {np.sum(valid)} / {valid.size}")
    print("  Aspect difference statistics (our - pgrid):")
    print(f"    Mean: {np.nanmean(diff[valid]):.4f}°")
    print(f"    Std:  {np.nanstd(diff[valid]):.4f}°")
    print(f"    Min:  {np.nanmin(diff[valid]):.4f}°")
    print(f"    Max:  {np.nanmax(diff[valid]):.4f}°")
    print(f"    Median: {np.nanmedian(diff[valid]):.4f}°")

    # -------------------------------------------------------------------------
    # Step 4: Analyze aspect classification differences
    # -------------------------------------------------------------------------
    print_section("Step 4: Analyzing Classification Differences")

    class_ours = classify_aspect(aspect_ours)
    class_pgrid = classify_aspect(aspect_pgrid)

    # Count pixels in each aspect bin
    print("\n  Aspect classification counts:")
    print(f"  {'Aspect':<10} {'Ours':<12} {'pgrid':<12} {'Diff':<10} {'%Diff':<10}")
    print(f"  {'-'*54}")

    total_valid = np.sum(valid)
    classification_diff = {}

    for idx, name in enumerate(ASPECT_NAMES):
        count_ours = np.sum((class_ours == idx) & valid)
        count_pgrid = np.sum((class_pgrid == idx) & valid)
        diff_count = count_ours - count_pgrid
        pct_diff = 100 * diff_count / total_valid

        classification_diff[name] = {
            "ours": int(count_ours),
            "pgrid": int(count_pgrid),
            "diff": int(diff_count),
            "pct_diff": float(pct_diff),
            "ours_pct": 100 * count_ours / total_valid,
            "pgrid_pct": 100 * count_pgrid / total_valid,
        }

        print(
            f"  {name:<10} {count_ours:<12} {count_pgrid:<12} {diff_count:<+10} {pct_diff:<+10.2f}%"
        )

    # -------------------------------------------------------------------------
    # Step 5: Identify boundary effects
    # -------------------------------------------------------------------------
    print_section("Step 5: Analyzing Boundary Classification Changes")

    # Find pixels where classification changed
    changed = (class_ours != class_pgrid) & valid
    n_changed = np.sum(changed)
    pct_changed = 100 * n_changed / total_valid

    print(f"  Pixels with changed classification: {n_changed} ({pct_changed:.2f}%)")

    # Break down by from->to transitions
    print("\n  Classification transitions (our -> pgrid):")
    transitions = {}
    for from_idx, from_name in enumerate(ASPECT_NAMES):
        for to_idx, to_name in enumerate(ASPECT_NAMES):
            if from_idx == to_idx:
                continue
            mask = changed & (class_ours == from_idx) & (class_pgrid == to_idx)
            count = np.sum(mask)
            if count > 0:
                key = f"{from_name}->{to_name}"
                pct = 100 * count / total_valid
                transitions[key] = {"count": int(count), "pct": float(pct)}
                print(f"    {key}: {count} ({pct:.3f}%)")

    # Focus on E/S boundary (135°)
    print("\n  Focus on East/South boundary (135°):")

    # Pixels near the boundary
    boundary_margin = 5  # degrees
    near_es_boundary = (
        (aspect_ours >= 135 - boundary_margin) & (aspect_ours < 135 + boundary_margin)
    ) | (
        (aspect_pgrid >= 135 - boundary_margin) & (aspect_pgrid < 135 + boundary_margin)
    )
    near_es_boundary &= valid

    n_near_boundary = np.sum(near_es_boundary)
    changed_near_boundary = np.sum(changed & near_es_boundary)

    print(f"    Pixels within ±{boundary_margin}° of 135°: {n_near_boundary}")
    print(f"    Changed classification near boundary: {changed_near_boundary}")
    if n_near_boundary > 0:
        print(
            f"    % of boundary pixels changed: {100 * changed_near_boundary / n_near_boundary:.1f}%"
        )

    # -------------------------------------------------------------------------
    # Step 6: Analyze aspect differences near boundaries
    # -------------------------------------------------------------------------
    print_section("Step 6: Aspect Differences by Region")

    # Compute mean difference for each aspect bin
    print("  Mean aspect difference by classification (our method):")
    for idx, name in enumerate(ASPECT_NAMES):
        mask = (class_ours == idx) & valid
        if np.sum(mask) > 0:
            mean_diff = np.nanmean(diff[mask])
            std_diff = np.nanstd(diff[mask])
            print(f"    {name}: mean={mean_diff:+.3f}°, std={std_diff:.3f}°")

    # -------------------------------------------------------------------------
    # Step 7: Generate comparison plots
    # -------------------------------------------------------------------------
    if HAS_MATPLOTLIB:
        print_section("Step 7: Generating Comparison Plots")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Our aspect
        ax = axes[0, 0]
        im = ax.imshow(aspect_ours, cmap="hsv", vmin=0, vmax=360)
        ax.set_title("Aspect (Our Method)")
        plt.colorbar(im, ax=ax, label="Degrees from N")

        # Plot 2: pgrid aspect
        ax = axes[0, 1]
        im = ax.imshow(aspect_pgrid, cmap="hsv", vmin=0, vmax=360)
        ax.set_title("Aspect (pgrid Horn 1981)")
        plt.colorbar(im, ax=ax, label="Degrees from N")

        # Plot 3: Difference
        ax = axes[0, 2]
        vmax = max(abs(np.nanmin(diff[valid])), abs(np.nanmax(diff[valid])))
        vmax = min(vmax, 10)  # Cap at 10 degrees
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title("Aspect Difference (Our - pgrid)")
        plt.colorbar(im, ax=ax, label="Degrees")

        # Plot 4: Our classification
        ax = axes[1, 0]
        cmap = plt.cm.get_cmap("Set1", 4)
        im = ax.imshow(class_ours, cmap=cmap, vmin=-0.5, vmax=3.5)
        ax.set_title("Classification (Our Method)")
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(ASPECT_NAMES)

        # Plot 5: pgrid classification
        ax = axes[1, 1]
        im = ax.imshow(class_pgrid, cmap=cmap, vmin=-0.5, vmax=3.5)
        ax.set_title("Classification (pgrid)")
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(ASPECT_NAMES)

        # Plot 6: Classification differences
        ax = axes[1, 2]
        diff_map = np.where(changed, 1, 0)
        im = ax.imshow(diff_map, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"Classification Changed ({pct_changed:.2f}% of pixels)")
        plt.colorbar(im, ax=ax, label="Changed (1=yes)")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "stage8_aspect_comparison.png"), dpi=150)
        plt.close()

        # Histogram of aspect differences
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(diff[valid].flatten(), bins=100, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Aspect Difference (degrees)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Aspect Differences")

        # Scatter: aspect near boundary
        ax = axes[1]
        es_mask = near_es_boundary
        ax.scatter(
            aspect_ours[es_mask],
            aspect_pgrid[es_mask],
            alpha=0.3,
            s=1,
            label="Near 135° boundary",
        )
        ax.plot([100, 170], [100, 170], "r--", linewidth=2, label="1:1 line")
        ax.axhline(135, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(135, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Our Aspect (degrees)")
        ax.set_ylabel("pgrid Aspect (degrees)")
        ax.set_title("Aspect Near East/South Boundary (135°)")
        ax.legend()
        ax.set_xlim(100, 170)
        ax.set_ylim(100, 170)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "stage8_aspect_histogram.png"), dpi=150)
        plt.close()

        print(f"  Saved plots to {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------------
    print_section("Step 8: Saving Results")

    results = {
        "summary": {
            "dem_shape": list(dem_data.shape),
            "valid_pixels": int(np.sum(valid)),
            "total_pixels": int(valid.size),
        },
        "aspect_difference_stats": {
            "mean": float(np.nanmean(diff[valid])),
            "std": float(np.nanstd(diff[valid])),
            "min": float(np.nanmin(diff[valid])),
            "max": float(np.nanmax(diff[valid])),
            "median": float(np.nanmedian(diff[valid])),
        },
        "classification_comparison": classification_diff,
        "transitions": transitions,
        "changed_pixels": {
            "count": int(n_changed),
            "percent": float(pct_changed),
        },
        "boundary_analysis": {
            "margin_degrees": boundary_margin,
            "pixels_near_135": int(n_near_boundary),
            "changed_near_135": int(changed_near_boundary),
        },
        "hypothesis_validation": {
            "expected": "~2.5% of pixels at E/S boundary classified differently",
            "observed_E_diff_pct": classification_diff["East"]["pct_diff"],
            "observed_S_diff_pct": classification_diff["South"]["pct_diff"],
            "total_changed_pct": float(pct_changed),
            "hypothesis_supported": abs(classification_diff["East"]["pct_diff"]) > 1.0,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "stage8_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "stage8_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 8: Gradient Calculation Comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write("Purpose: Validate hypothesis that gradient calculation differences\n")
        f.write("cause the ~2.5% East/South boundary classification discrepancy.\n\n")

        f.write("Key Finding:\n")
        f.write("-" * 40 + "\n")

        if results["hypothesis_validation"]["hypothesis_supported"]:
            f.write("HYPOTHESIS SUPPORTED\n\n")
            f.write(
                f"East aspect difference: {classification_diff['East']['pct_diff']:+.2f}%\n"
            )
            f.write(
                f"South aspect difference: {classification_diff['South']['pct_diff']:+.2f}%\n"
            )
            f.write(f"Total pixels changed: {pct_changed:.2f}%\n\n")
            f.write("Recommendation: Switch to pgrid's Horn 1981 method in stage3.\n")
        else:
            f.write("HYPOTHESIS NOT SUPPORTED\n\n")
            f.write(
                f"Classification differences are small ({pct_changed:.2f}% changed).\n"
            )
            f.write("The area correlation issue may have a different root cause.\n")

        f.write(f"\nTotal processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # -------------------------------------------------------------------------
    # Conclusion
    # -------------------------------------------------------------------------
    print_section("Stage 8 Complete")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.1f} seconds")

    print("\n" + "=" * 60)
    print("  HYPOTHESIS VALIDATION")
    print("=" * 60)

    if results["hypothesis_validation"]["hypothesis_supported"]:
        print("\n  RESULT: HYPOTHESIS SUPPORTED")
        print(
            f"\n  East aspect: {classification_diff['East']['ours_pct']:.1f}% (ours) vs {classification_diff['East']['pgrid_pct']:.1f}% (pgrid)"
        )
        print(
            f"  South aspect: {classification_diff['South']['ours_pct']:.1f}% (ours) vs {classification_diff['South']['pgrid_pct']:.1f}% (pgrid)"
        )
        print(f"\n  Total pixels with different classification: {pct_changed:.2f}%")
        print("\n  RECOMMENDATION: Proceed with Phase 2 - fix stage3 to use pgrid method")
    else:
        print("\n  RESULT: HYPOTHESIS NOT SUPPORTED")
        print(f"\n  Classification differences are small ({pct_changed:.2f}%)")
        print("  The area correlation issue may have a different root cause.")


if __name__ == "__main__":
    main()
