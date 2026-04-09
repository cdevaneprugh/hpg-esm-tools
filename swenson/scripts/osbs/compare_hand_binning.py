#!/usr/bin/env python3
"""
Compare HAND binning strategies for 1x8 hillslope configuration.

Runs pipeline Steps 1-4 once to produce filtered HAND/DTND arrays, then tests
5 binning strategies on the same data:
  - Baseline: Q1/Q99 log-spaced (current, known broken)
  - A: Log-spaced with 0.25m floor
  - B: Equal-count on HAND > 0.25m
  - C: Fixed boundaries from domain knowledge
  - D: Hybrid (5 TAI bins 0-2m + 3 ridge bins 2m+)

Results written to output/osbs/binning_comparison/.

Usage:
    python scripts/osbs/compare_hand_binning.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import rasterio
from pyproj import Proj as PyprojProj

# pysheds fork
PYSHEDS_FORK = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, PYSHEDS_FORK)
from pysheds.pgrid import Grid  # noqa: E402

# Shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hillslope_params import (  # noqa: E402
    circular_mean_aspect,
    compute_hand_bins_log,
    fit_trapezoidal_width,
    get_aspect_mask,
    quadratic,
    tail_index,
)
from spatial_scale import identify_spatial_scale_laplacian_dem  # noqa: E402
from dem_processing import identify_basins  # noqa: E402

# =============================================================================
# Constants (mirror run_pipeline.py)
# =============================================================================
PIXEL_SIZE = 1.0
MIN_WAVELENGTH = 20
NODATA_VALUE = -9999
NODATA_THRESHOLD = -9000
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
N_BINS = 8
HAND_FLOOR = 0.25  # meters — below this is DEM conditioning noise

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent
MOSAIC_DIR = BASE_DIR / "data" / "mosaics" / "production"
MOSAIC_PATH = MOSAIC_DIR / "dtm.tif"
SLOPE_MOSAIC_PATH = MOSAIC_DIR / "slope.tif"
ASPECT_MOSAIC_PATH = MOSAIC_DIR / "aspect.tif"
WATER_MASK_PATH = MOSAIC_DIR / "water_mask.tif"
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / "binning_comparison"
CACHE_PATH = OUTPUT_DIR / "filtered_arrays.npz"


def print_progress(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# =============================================================================
# Binning strategies
# =============================================================================


def bins_baseline(hand: np.ndarray) -> np.ndarray:
    """Q1/Q99 log-spaced (current, known broken)."""
    return compute_hand_bins_log(hand, n_bins=N_BINS)


def bins_a_log_floor(hand: np.ndarray) -> np.ndarray:
    """Log-spaced with 0.25m floor."""
    valid = (hand > 0) & np.isfinite(hand)
    q99 = float(np.percentile(hand[valid], 99))
    internal = np.geomspace(HAND_FLOOR, q99, N_BINS - 1)
    return np.concatenate([[0], internal, [1e6]])


def bins_b_equal_count(hand: np.ndarray) -> np.ndarray:
    """Equal-count bins on HAND > floor."""
    above = hand[hand > HAND_FLOOR]
    if len(above) < N_BINS:
        return np.concatenate([[0], np.linspace(HAND_FLOOR, 25, N_BINS - 1), [1e6]])
    pcts = np.linspace(0, 100, N_BINS + 1)
    boundaries = np.percentile(above, pcts)
    boundaries[0] = 0
    boundaries[-1] = 1e6
    return boundaries


def bins_c_fixed(hand: np.ndarray) -> np.ndarray:
    """Fixed boundaries from domain knowledge."""
    return np.array([0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e6])


def bins_d_hybrid(hand: np.ndarray) -> np.ndarray:
    """Hybrid: 5 equal-area TAI bins (0-2m) + 3 equal-area ridge bins (2m+)."""
    n_tai = 5
    n_ridge = 3
    tai = hand[(hand > 0) & (hand <= 2.0)]
    ridge = hand[hand > 2.0]

    if len(tai) < n_tai:
        tai_bounds = np.linspace(0, 2.0, n_tai + 1)
    else:
        tai_bounds = np.percentile(tai, np.linspace(0, 100, n_tai + 1))

    if len(ridge) < n_ridge:
        ridge_bounds = np.linspace(2.0, 25.0, n_ridge + 1)
    else:
        ridge_bounds = np.percentile(ridge, np.linspace(0, 100, n_ridge + 1))

    bounds = np.concatenate([[0], tai_bounds[1:], ridge_bounds[1:-1], [1e6]])
    return bounds


def bins_a2_log_floor_01_q95(hand: np.ndarray) -> np.ndarray:
    """Log-spaced with 0.1m floor and Q95 upper endpoint."""
    valid = (hand > 0) & np.isfinite(hand)
    q95 = float(np.percentile(hand[valid], 95))
    internal = np.geomspace(0.1, q95, N_BINS - 1)
    return np.concatenate([[0], internal, [1e6]])


STRATEGIES = {
    "Baseline (Q1/Q99)": bins_baseline,
    "A: Log floor (0.25m)": bins_a_log_floor,
    "A2: Log floor (0.1m) + Q95": bins_a2_log_floor_01_q95,
    "B: Equal-count (>0.25m)": bins_b_equal_count,
    "C: Fixed boundaries": bins_c_fixed,
    "D: Hybrid (5 TAI + 3 ridge)": bins_d_hybrid,
}


# =============================================================================
# Parameter computation (extracted from run_pipeline.py Step 5c)
# =============================================================================


def compute_params_for_bins(
    hand_bounds: np.ndarray,
    hand_flat: np.ndarray,
    dtnd_flat: np.ndarray,
    slope_flat: np.ndarray,
    aspect_flat: np.ndarray,
    area_flat: np.ndarray,
    drainage_id_flat: np.ndarray,
) -> list[dict]:
    """Compute 6 hillslope parameters for given HAND bin boundaries."""
    n_bins = len(hand_bounds) - 1
    asp_mask = get_aspect_mask(aspect_flat, (0, 360))
    asp_indices = np.where(asp_mask)[0]
    n_hillslopes = max(len(np.unique(drainage_id_flat[asp_indices])), 1)

    trap = fit_trapezoidal_width(
        dtnd_flat[asp_indices],
        area_flat[asp_indices],
        n_hillslopes,
        min_dtnd=PIXEL_SIZE,
    )
    trap_slope = trap["slope"]
    trap_width = trap["width"]
    trap_area = trap["area"]

    # First pass: raw areas per bin
    bin_raw_areas = []
    bin_indices_list = []
    for h_idx in range(n_bins):
        h_lower = hand_bounds[h_idx]
        h_upper = hand_bounds[h_idx + 1]
        hand_mask = (hand_flat >= h_lower) & (hand_flat < h_upper)
        bin_mask = asp_mask & hand_mask
        indices = np.where(bin_mask)[0]
        bin_raw_areas.append(
            float(np.sum(area_flat[indices])) if len(indices) > 0 else 0
        )
        bin_indices_list.append(indices if len(indices) > 0 else None)

    total_raw = sum(bin_raw_areas)
    area_fractions = (
        [a / total_raw for a in bin_raw_areas]
        if total_raw > 0
        else [1.0 / n_bins] * n_bins
    )
    fitted_areas = [trap_area * frac for frac in area_fractions]

    # Second pass: compute parameters
    elements = []
    for h_idx in range(n_bins):
        indices = bin_indices_list[h_idx]
        if indices is None or np.mean(hand_flat[indices]) <= 0:
            elements.append(
                {
                    "bin": h_idx + 1,
                    "hand_lower": float(hand_bounds[h_idx]),
                    "hand_upper": float(hand_bounds[h_idx + 1]),
                    "height": 0,
                    "distance": 0,
                    "width": 0,
                    "area": fitted_areas[h_idx] if indices is not None else 0,
                    "slope": 0,
                    "aspect": 0,
                    "n_pixels": len(indices) if indices is not None else 0,
                }
            )
            continue

        mean_hand = float(np.mean(hand_flat[indices]))
        mean_slope = float(np.nanmean(slope_flat[indices]))
        mean_aspect = circular_mean_aspect(aspect_flat[indices])
        dtnd_sorted = np.sort(dtnd_flat[indices])
        median_dtnd = float(dtnd_sorted[len(dtnd_sorted) // 2])

        # Width at lower edge
        da_width = sum(fitted_areas[:h_idx]) if h_idx > 0 else 0
        if trap_slope != 0:
            try:
                le = quadratic([trap_slope, trap_width, -da_width])
                width = trap_width + 2 * trap_slope * le
            except RuntimeError:
                width = trap_width * (1 - 0.15 * h_idx)
        else:
            width = trap_width
        width = max(float(width), 1)

        # Distance: trapezoid-derived midpoint
        da_dist = sum(fitted_areas[: h_idx + 1]) - fitted_areas[h_idx] / 2
        if trap_slope != 0:
            try:
                distance = float(quadratic([trap_slope, trap_width, -da_dist]))
            except RuntimeError:
                distance = median_dtnd
        else:
            distance = median_dtnd

        elements.append(
            {
                "bin": h_idx + 1,
                "hand_lower": float(hand_bounds[h_idx]),
                "hand_upper": float(hand_bounds[h_idx + 1]),
                "height": mean_hand,
                "distance": distance,
                "width": width,
                "area": fitted_areas[h_idx],
                "slope": mean_slope,
                "aspect": mean_aspect,
                "n_pixels": len(indices),
            }
        )

    return elements


# =============================================================================
# Pipeline Steps 1-4 + 5a (identical to run_pipeline.py)
# =============================================================================


def run_pipeline_through_filtering():
    """Run Steps 1-4 + 5a to produce filtered arrays for binning comparison."""
    t_start = time.time()

    # --- Step 1: Load mosaics ---
    print_progress("Step 1: Loading mosaics")

    with rasterio.open(MOSAIC_PATH) as src:
        dem = src.read(1)
        rasterio_crs = src.crs
        affine = src.transform

    with rasterio.open(SLOPE_MOSAIC_PATH) as src:
        neon_slope_deg = src.read(1)
    slope = np.tan(np.deg2rad(neon_slope_deg.astype(np.float64)))
    slope[neon_slope_deg <= NODATA_THRESHOLD] = 0.0

    with rasterio.open(ASPECT_MOSAIC_PATH) as src:
        aspect = src.read(1).astype(np.float64)
    aspect[aspect <= NODATA_THRESHOLD] = 0.0

    with rasterio.open(WATER_MASK_PATH) as src:
        water_mask = src.read(1)

    print_progress(
        f"  Shape: {dem.shape}, Water: {np.sum(water_mask > 0):,} px "
        f"({100 * np.sum(water_mask > 0) / water_mask.size:.1f}%)"
    )

    # --- Step 2: FFT ---
    print_progress("Step 2: FFT spatial scale")
    fft_result = identify_spatial_scale_laplacian_dem(
        dem,
        pixel_size=PIXEL_SIZE,
        min_wavelength=MIN_WAVELENGTH,
        blend_edges_n=50,
        zero_edges_n=50,
        verbose=False,
    )
    lc_px = fft_result["spatialScale"]
    accum_threshold = int(0.5 * lc_px**2)
    print_progress(f"  Lc: {lc_px:.0f} px, A_thresh: {accum_threshold}")

    # --- Step 3: DEM processing ---
    print_progress("Step 3: DEM processing")

    crs = PyprojProj(rasterio_crs.to_proj4())

    basin_pre_mask = identify_basins(dem, nodata=NODATA_VALUE)
    dem[basin_pre_mask > 0] = NODATA_VALUE

    grid = Grid()
    grid.add_gridded_data(
        dem, data_name="dem", affine=affine, crs=crs, nodata=NODATA_VALUE
    )

    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")

    flooded_arr = np.array(grid.flooded)
    flooded_arr[water_mask > 0] -= 0.1
    grid.add_gridded_data(
        flooded_arr, data_name="flooded", affine=affine, crs=crs, nodata=NODATA_VALUE
    )

    print_progress("  Resolving flats...")
    try:
        grid.resolve_flats("flooded", out_name="inflated")
    except ValueError:
        print_progress("  WARNING: resolve_flats failed, falling back to flooded DEM")
        grid.add_gridded_data(
            np.array(grid.flooded),
            data_name="inflated",
            affine=affine,
            crs=crs,
            nodata=NODATA_VALUE,
        )

    print_progress("  Flow direction + accumulation...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    max_acc = np.nanmax(np.array(grid.acc))
    if max_acc < accum_threshold:
        accum_threshold = int(max_acc / 100)

    acc_mask = (grid.acc > accum_threshold) & np.isfinite(np.array(grid.inflated))
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

    print_progress(f"  Stream cells: {np.sum(grid.channel_mask > 0):,}")

    # --- Step 4: HAND/DTND ---
    print_progress("Step 4: HAND/DTND (wide mask)")

    wide_channel_mask = (np.array(grid.channel_mask) > 0) | (water_mask > 0)
    wide_channel_id = np.array(grid.channel_id).copy()
    water_not_stream = (water_mask > 0) & (np.array(grid.channel_mask) == 0)
    wide_channel_id[water_not_stream] = 0

    grid.compute_hand(
        "fdir",
        "inflated",
        wide_channel_mask,
        wide_channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    hand = np.array(grid.hand)
    dtnd = np.array(grid.dtnd)

    print_progress("  Hillslope classification...")
    grid.compute_hillslope(
        "fdir", "channel_mask", "bank_mask", dirmap=DIRMAP, routing="d8"
    )

    # Catchment-level aspect averaging
    from hillslope_params import catchment_mean_aspect  # noqa: E402

    aspect = catchment_mean_aspect(
        np.array(grid.drainage_id), aspect, np.array(grid.hillslope)
    )

    print_progress(f"  Steps 1-4: {time.time() - t_start:.0f}s")

    # --- Step 5a: Filtering ---
    print_progress("Step 5a: Filtering")

    hand_flat = hand.flatten()
    dtnd_flat = dtnd.flatten()
    slope_flat = slope.flatten()
    aspect_flat = aspect.flatten()
    area_flat = np.full(hand_flat.shape, PIXEL_SIZE * PIXEL_SIZE)

    water_mask_flat = water_mask.flatten()
    land_finite = np.isfinite(hand_flat) & (water_mask_flat == 0)
    tail_ind = tail_index(dtnd_flat[land_finite], hand_flat[land_finite])
    land_indices = np.where(land_finite)[0]
    keep_tail = np.zeros(hand_flat.shape, dtype=bool)
    keep_tail[land_indices[tail_ind]] = True

    smallest_dtnd = 1.0
    dtnd_flat[dtnd_flat < smallest_dtnd] = smallest_dtnd

    valid = keep_tail.copy()
    valid = valid & (water_mask_flat == 0)

    basin_pre_flat = basin_pre_mask.flatten()
    n_basin = int(np.sum(basin_pre_flat > 0))
    if n_basin > 0:
        non_flat_fraction = np.sum(basin_pre_flat == 0) / basin_pre_flat.size
        if non_flat_fraction > 0.01:
            valid = valid & (basin_pre_flat == 0)

    drainage_id_flat = np.array(grid.drainage_id).flatten()

    hand_flat = hand_flat[valid]
    dtnd_flat = dtnd_flat[valid]
    slope_flat = slope_flat[valid]
    aspect_flat = aspect_flat[valid]
    area_flat = area_flat[valid]
    drainage_id_flat = drainage_id_flat[valid]

    print_progress(f"  Valid pixels: {len(hand_flat):,}")

    # HAND distribution stats
    hand_pos = hand_flat[hand_flat > 0]
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print_progress(f"  HAND Q{p:02d}: {np.percentile(hand_pos, p):.4f} m")

    return (
        hand_flat,
        dtnd_flat,
        slope_flat,
        aspect_flat,
        area_flat,
        drainage_id_flat,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    t_start = time.time()
    recompute = "--recompute" in sys.argv

    if CACHE_PATH.exists() and not recompute:
        print_progress(f"Loading cached arrays: {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        hand_flat = data["hand"]
        dtnd_flat = data["dtnd"]
        slope_flat = data["slope"]
        aspect_flat = data["aspect"]
        area_flat = data["area"]
        drainage_id_flat = data["drainage_id"]
        print_progress(f"  Valid pixels: {len(hand_flat):,}")
        hand_pos = hand_flat[hand_flat > 0]
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print_progress(f"  HAND Q{p:02d}: {np.percentile(hand_pos, p):.4f} m")
    else:
        arrays = run_pipeline_through_filtering()
        hand_flat, dtnd_flat, slope_flat, aspect_flat, area_flat, drainage_id_flat = (
            arrays
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            CACHE_PATH,
            hand=hand_flat,
            dtnd=dtnd_flat,
            slope=slope_flat,
            aspect=aspect_flat,
            area=area_flat,
            drainage_id=drainage_id_flat,
        )
        print_progress(f"  Saved cache: {CACHE_PATH}")

    # --- Test all strategies ---
    print()
    print("=" * 70)
    print("  Binning Strategy Comparison")
    print("=" * 70)

    all_results = {}

    for name, strategy_fn in STRATEGIES.items():
        print(f"\n--- {name} ---")

        t0 = time.time()
        bounds = strategy_fn(hand_flat)
        elements = compute_params_for_bins(
            bounds,
            hand_flat,
            dtnd_flat,
            slope_flat,
            aspect_flat,
            area_flat,
            drainage_id_flat,
        )
        elapsed = time.time() - t0

        # Summary
        n_zero = sum(1 for e in elements if e["height"] == 0)
        n_bins_below_2m = sum(1 for i in range(len(bounds) - 1) if bounds[i + 1] <= 2.0)
        lowest_nonzero = min(
            (e["height"] for e in elements if e["height"] > 0), default=0
        )

        print(
            f"  Bins: {len(bounds) - 1}, Zero-height: {n_zero}, "
            f"Below 2m: {n_bins_below_2m}, Time: {elapsed:.2f}s"
        )
        print(f"  Boundaries: {[f'{b:.4f}' for b in bounds]}")

        header = f"  {'Bin':>3s} {'HAND range':>18s} {'h(m)':>7s} {'d(m)':>7s} {'w(m)':>7s} {'pixels':>10s}"
        print(header)
        for e in elements:
            h_lo = e["hand_lower"]
            h_hi = e["hand_upper"]
            if h_hi > 999:
                rng = f"{h_lo:.2f} - max"
            else:
                rng = f"{h_lo:.2f} - {h_hi:.2f}"
            print(
                f"  {e['bin']:>3d} {rng:>18s} {e['height']:>7.1f} "
                f"{e['distance']:>7.0f} {e['width']:>7.0f} {e['n_pixels']:>10,d}"
            )

        all_results[name] = {
            "bounds": [float(b) for b in bounds],
            "elements": elements,
            "n_zero_height": n_zero,
            "n_bins_below_2m": n_bins_below_2m,
            "lowest_nonzero_height": lowest_nonzero,
        }

    # --- Save results ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_json = {
        "metadata": {
            "description": "HAND binning strategy comparison (1x8, NWI water-masked)",
            "n_bins": N_BINS,
            "hand_floor": HAND_FLOOR,
            "n_valid_pixels": int(len(hand_flat)),
            "timestamp": datetime.now().isoformat(),
        },
        "strategies": all_results,
    }

    json_path = OUTPUT_DIR / "binning_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- Comparison plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {
        "Baseline (Q1/Q99)": "#888888",
        "A: Log floor (0.25m)": "#e74c3c",
        "A2: Log floor (0.1m) + Q95": "#e67e22",
        "B: Equal-count (>0.25m)": "#3498db",
        "C: Fixed boundaries": "#2ecc71",
        "D: Hybrid (5 TAI + 3 ridge)": "#9b59b6",
    }

    for name, result in all_results.items():
        elems = result["elements"]
        bins = list(range(1, len(elems) + 1))
        color = colors[name]

        axes[0].plot(
            bins,
            [e["height"] for e in elems],
            "o-",
            color=color,
            label=name,
            markersize=4,
        )
        axes[1].plot(
            bins,
            [e["distance"] for e in elems],
            "o-",
            color=color,
            label=name,
            markersize=4,
        )
        axes[2].plot(
            bins,
            [e["width"] for e in elems],
            "o-",
            color=color,
            label=name,
            markersize=4,
        )

    axes[0].set_ylabel("Height (m)")
    axes[1].set_ylabel("Distance (m)")
    axes[2].set_ylabel("Width (m)")

    for ax in axes:
        ax.set_xlabel("Bin")
        ax.grid(True, alpha=0.3)

    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("HAND Binning Strategy Comparison (1x8, NWI water-masked)")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "binning_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    print(f"\nTotal: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
