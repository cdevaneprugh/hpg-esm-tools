#!/usr/bin/env python
"""
R6C10 UTM Smoke Test — Phase A Validation

Validates the UTM CRS code path in the pysheds fork (feature/utm-crs-support)
using the standard R6C10 smoke test tile (1000x1000, 1m, EPSG:32617).

Tests every CRS-dependent pysheds function that Phase A modified:
  - fill_pits, fill_depressions, resolve_flats (DEM conditioning)
  - flowdir (D8 routing with UTM pixel spacing)
  - accumulation (flow accumulation)
  - create_channel_mask (stream network delineation)
  - compute_hand (HAND + DTND with Euclidean distance)
  - slope_aspect (Horn 1981 with uniform pixel spacing)

Also compares pysheds DTND to scipy EDT baseline and optionally computes
full hillslope parameters (4 aspects x 4 HAND bins).

Usage:
    python scripts/smoke_tests/run_r6c10_utm.py [--subsample N] [--skip-hillslope]

Output:
    output/osbs/smoke_tests/r6c10_utm/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

# pysheds fork (expects PYTHONPATH or PYSHEDS_FORK env var)
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid  # noqa: E402

# Add parent directory for local imports (scripts/)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# CRS-independent hillslope functions from stage3
from merit_validation.stage3_hillslope_params import (  # noqa: E402
    circular_mean_aspect,
    compute_hand_bins,
    fit_trapezoidal_width,
    get_aspect_mask,
    quadratic,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEM_PATH = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
    "/data/neon/dtm/NEON_D03_OSBS_DP3_404000_3286000_DTM.tif"
)
OUTPUT_DIR = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/osbs/smoke_tests/r6c10_utm"
)

# Phase C working Lc (~300m).  A_thresh = 0.5 * Lc^2, but units depend on
# pixel size: the threshold is in *cell count* because pysheds accumulation
# counts cells, not area.  At 1m resolution, 1 cell = 1 m^2, so
# A_thresh_cells = A_thresh_m2 = 0.5 * 300^2 = 45000.
LC_M = 300.0

# Hillslope parameters
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0  # Maximum HAND for lowest bin (meters)
ASPECT_BINS = [
    (315, 45),  # North
    (45, 135),  # East
    (135, 225),  # South
    (225, 315),  # West
]
ASPECT_NAMES = ["North", "East", "South", "West"]

# D8 direction map (pysheds convention)
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def check(name: str, passed: bool, detail: str, results: list) -> None:
    """Record a validation check result."""
    passed = bool(passed)  # ensure Python bool (not numpy bool_)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    results.append({"name": name, "passed": passed, "detail": detail})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="R6C10 UTM smoke test")
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor (default: 1 = full resolution)",
    )
    parser.add_argument(
        "--skip-hillslope",
        action="store_true",
        help="Skip hillslope parameter computation",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results: list[dict] = []
    timings: dict[str, float] = {}
    t_total = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Load DEM
    # -----------------------------------------------------------------------
    section("Step 1: Load DEM")

    with rasterio.open(DEM_PATH) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata_val = src.nodata
        pixel_size = transform.a  # meters (should be 1.0 for this tile)

    print(f"  File: {os.path.basename(DEM_PATH)}")
    print(f"  CRS: {crs}")
    print(f"  Shape: {dem_data.shape}")
    print(f"  Pixel size: {pixel_size} m")
    print(f"  Nodata value: {nodata_val}")

    # Handle nodata
    if nodata_val is not None:
        nodata_mask = dem_data == nodata_val
        dem_data = dem_data.astype(np.float64)
        dem_data[nodata_mask] = np.nan
        n_nodata = int(np.sum(nodata_mask))
        print(f"  Nodata pixels: {n_nodata}")
    else:
        dem_data = dem_data.astype(np.float64)
        n_nodata = 0

    # Subsample if requested
    if args.subsample > 1:
        dem_data = dem_data[:: args.subsample, :: args.subsample]
        pixel_size *= args.subsample
        # Update transform for subsampled grid
        transform = rasterio.Affine(
            transform.a * args.subsample,
            transform.b,
            transform.c,
            transform.d,
            transform.e * args.subsample,
            transform.f,
        )
        print(
            f"  Subsampled {args.subsample}x -> shape {dem_data.shape}, "
            f"pixel_size {pixel_size} m"
        )

    elev_valid = dem_data[np.isfinite(dem_data)]
    print(f"  Elevation range: {np.min(elev_valid):.2f} - {np.max(elev_valid):.2f} m")
    print(f"  Elevation median: {np.median(elev_valid):.2f} m")

    # A_thresh in cells = 0.5 * Lc^2 / pixel_size^2
    accum_threshold = int(0.5 * LC_M**2 / pixel_size**2)
    print(
        f"  Lc = {LC_M} m -> A_thresh = {accum_threshold} cells "
        f"(at {pixel_size}m pixels)"
    )

    # -----------------------------------------------------------------------
    # Step 2: Initialize pysheds Grid and check CRS detection
    # -----------------------------------------------------------------------
    section("Step 2: Initialize pysheds Grid")

    from pyproj import Proj as PyprojProj

    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=transform,
        crs=PyprojProj(crs),
        nodata=np.nan,
    )

    is_geographic = grid._crs_is_geographic()
    print(f"  grid._crs_is_geographic() = {is_geographic}")
    check(
        "CRS detection",
        not is_geographic,
        f"Expected False for UTM, got {is_geographic}",
        results,
    )

    # -----------------------------------------------------------------------
    # Step 3: DEM conditioning + flow routing
    # -----------------------------------------------------------------------
    section("Step 3: DEM Conditioning + Flow Routing")

    t0 = time.time()

    print("  Filling pits...")
    grid.fill_pits("dem", out_name="pit_filled")

    print("  Filling depressions...")
    grid.fill_depressions("pit_filled", out_name="flooded")

    print("  Resolving flats...")
    grid.resolve_flats("flooded", out_name="inflated")

    print("  Computing flow direction (D8)...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")

    print("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")
    acc = grid.acc

    timings["conditioning_and_routing"] = time.time() - t0
    print(f"  Time: {timings['conditioning_and_routing']:.1f}s")
    print(f"  Max accumulation: {np.nanmax(acc):.0f} cells")

    # Create stream network
    acc_mask = acc > accum_threshold
    n_stream_pixels = int(np.sum(acc_mask))
    print(f"  Stream pixels (acc > {accum_threshold}): {n_stream_pixels}")

    check(
        "Stream network exists",
        n_stream_pixels > 0,
        f"{n_stream_pixels} stream pixels",
        results,
    )

    print("  Creating channel mask...")
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")
    stream_mask = grid.channel_mask
    channel_id = grid.channel_id

    n_channels = len(np.unique(channel_id[channel_id > 0]))
    print(f"  Unique channel segments: {n_channels}")

    # -----------------------------------------------------------------------
    # Step 4: HAND + DTND (Phase A core test)
    # -----------------------------------------------------------------------
    section("Step 4: HAND + DTND (Phase A Core Test)")

    t0 = time.time()

    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    hand = np.array(grid.hand, dtype=np.float64)
    dtnd = np.array(grid.dtnd, dtype=np.float64)

    timings["hand_dtnd"] = time.time() - t0
    print(f"  Time: {timings['hand_dtnd']:.1f}s")

    # Mask invalid values
    valid_mask = np.isfinite(hand) & np.isfinite(dtnd)
    hand_valid = hand[valid_mask]
    dtnd_valid = dtnd[valid_mask]

    print(f"  Valid pixels: {np.sum(valid_mask)} / {hand.size}")
    print(f"  HAND range: [{np.min(hand_valid):.2f}, {np.max(hand_valid):.2f}] m")
    print(
        f"  HAND mean: {np.mean(hand_valid):.2f} m, median: {np.median(hand_valid):.2f} m"
    )
    print(f"  DTND range: [{np.min(dtnd_valid):.2f}, {np.max(dtnd_valid):.2f}] m")
    print(
        f"  DTND mean: {np.mean(dtnd_valid):.2f} m, median: {np.median(dtnd_valid):.2f} m"
    )

    # Validation checks
    # Small negative HAND is expected: DEM conditioning (fill_depressions)
    # raises some pixels above original elevation, so HAND against the
    # original DEM can go slightly negative near conditioned boundaries.
    check(
        "HAND >= -1m",
        bool(np.all(hand_valid >= -1.0)),
        f"min = {np.min(hand_valid):.4f} (small negatives from DEM conditioning)",
        results,
    )
    check(
        "HAND max <= 20m",
        bool(np.max(hand_valid) <= 20),
        f"max = {np.max(hand_valid):.2f} m (tile relief = 19.55m)",
        results,
    )
    check(
        "DTND >= 0",
        bool(np.all(dtnd_valid >= 0)),
        f"min = {np.min(dtnd_valid):.4f}",
        results,
    )
    check(
        "DTND max <= 750m",
        bool(np.max(dtnd_valid) <= 750),
        f"max = {np.max(dtnd_valid):.2f} m (tile is 1000m across, Lc ~300m)",
        results,
    )

    # Stream pixels should have HAND = 0 and DTND = 0
    stream_flat = np.array(stream_mask, dtype=bool).flatten()
    hand_flat = hand.flatten()
    dtnd_flat = dtnd.flatten()

    stream_hand = hand_flat[stream_flat & np.isfinite(hand_flat)]
    stream_dtnd = dtnd_flat[stream_flat & np.isfinite(dtnd_flat)]
    if len(stream_hand) > 0:
        check(
            "Stream HAND = 0",
            bool(np.allclose(stream_hand, 0, atol=1e-6)),
            f"max stream HAND = {np.max(np.abs(stream_hand)):.6f}",
            results,
        )
    if len(stream_dtnd) > 0:
        check(
            "Stream DTND = 0",
            bool(np.allclose(stream_dtnd, 0, atol=1e-6)),
            f"max stream DTND = {np.max(np.abs(stream_dtnd)):.6f}",
            results,
        )

    # HAND/DTND ratio should approximate mean terrain slope
    nonstream = valid_mask.flatten() & ~stream_flat
    if np.sum(nonstream) > 0:
        hand_ns = hand_flat[nonstream]
        dtnd_ns = dtnd_flat[nonstream]
        # Avoid division by zero
        valid_ratio = dtnd_ns > 0
        if np.sum(valid_ratio) > 0:
            ratio = np.mean(hand_ns[valid_ratio] / dtnd_ns[valid_ratio])
            # R6C10 has a large lake/flat area pulling the ratio below
            # mean terrain slope, so we use a wider tolerance.
            check(
                "HAND/DTND ratio ~ terrain slope",
                0.01 <= ratio <= 0.10,
                f"mean ratio = {ratio:.4f} (expected 0.01-0.10)",
                results,
            )

    # -----------------------------------------------------------------------
    # Step 5: Slope + Aspect (Phase A core test)
    # -----------------------------------------------------------------------
    section("Step 5: Slope + Aspect (Phase A Core Test)")

    t0 = time.time()

    grid.slope_aspect("dem")
    slope = np.array(grid.slope, dtype=np.float64)
    aspect = np.array(grid.aspect, dtype=np.float64)

    timings["slope_aspect"] = time.time() - t0
    print(f"  Time: {timings['slope_aspect']:.1f}s")

    slope_valid = slope[np.isfinite(slope)]
    aspect_valid = aspect[np.isfinite(aspect)]

    print(f"  Slope range: [{np.min(slope_valid):.4f}, {np.max(slope_valid):.4f}] m/m")
    print(
        f"  Slope mean: {np.mean(slope_valid):.4f}, median: {np.median(slope_valid):.4f}"
    )
    print(
        f"  Aspect range: [{np.min(aspect_valid):.1f}, {np.max(aspect_valid):.1f}] deg"
    )

    check(
        "Slope max < 2.0",
        bool(np.max(slope_valid) < 2.0),
        f"max = {np.max(slope_valid):.4f} (tile max ~1.08)",
        results,
    )
    check(
        "Slope mean in 0.03-0.10",
        0.03 <= np.mean(slope_valid) <= 0.10,
        f"mean = {np.mean(slope_valid):.4f} (expected ~0.065)",
        results,
    )

    # Aspect distribution: all 4 quadrants should have >= 10% of pixels
    aspect_flat_valid = aspect_valid.flatten()
    n_total = len(aspect_flat_valid)
    quadrant_fracs = {}
    for name, (lo, hi) in zip(ASPECT_NAMES, ASPECT_BINS):
        mask = get_aspect_mask(aspect_flat_valid, (lo, hi))
        frac = np.sum(mask) / n_total
        quadrant_fracs[name] = float(frac)

    min_frac = min(quadrant_fracs.values())
    frac_str = ", ".join(f"{k}: {v:.1%}" for k, v in quadrant_fracs.items())
    check(
        "Aspect: all quadrants >= 10%",
        min_frac >= 0.10,
        frac_str,
        results,
    )

    # -----------------------------------------------------------------------
    # Step 6: DTND comparison (pysheds vs EDT baseline)
    # -----------------------------------------------------------------------
    section("Step 6: DTND Comparison (pysheds vs EDT)")

    t0 = time.time()

    # EDT distance: Euclidean distance to nearest stream pixel regardless of
    # drainage connectivity
    stream_bool = np.array(stream_mask, dtype=bool)
    edt_dtnd = distance_transform_edt(~stream_bool) * pixel_size

    timings["edt_dtnd"] = time.time() - t0

    # Compare where both are valid and non-stream
    compare_mask = valid_mask & ~stream_bool
    pysheds_dtnd_cmp = dtnd[compare_mask]
    edt_dtnd_cmp = edt_dtnd[compare_mask]

    if len(pysheds_dtnd_cmp) > 0 and np.std(pysheds_dtnd_cmp) > 0:
        corr = float(np.corrcoef(pysheds_dtnd_cmp, edt_dtnd_cmp)[0, 1])
        max_diff = float(np.max(np.abs(pysheds_dtnd_cmp - edt_dtnd_cmp)))
        mean_diff = float(np.mean(np.abs(pysheds_dtnd_cmp - edt_dtnd_cmp)))

        print(f"  Correlation: {corr:.4f}")
        print(f"  Max absolute difference: {max_diff:.1f} m")
        print(f"  Mean absolute difference: {mean_diff:.1f} m")
        print("  (Differences expected where drainage divides matter)")

        check(
            "pysheds vs EDT DTND correlation > 0.90",
            corr > 0.90,
            f"r = {corr:.4f}",
            results,
        )
        check(
            "pysheds vs EDT DTND not identical",
            corr < 0.9999,
            f"r = {corr:.4f} (should differ at drainage divides)",
            results,
        )
    else:
        print("  WARNING: Could not compute DTND comparison")

    # -----------------------------------------------------------------------
    # Step 7: Hillslope parameters (optional)
    # -----------------------------------------------------------------------
    hillslope_params = None

    if not args.skip_hillslope:
        section("Step 7: Hillslope Parameters")

        t0 = time.time()

        # Flatten arrays for hillslope computation
        hand_f = hand.flatten()
        dtnd_f = dtnd.flatten()
        slope_f = slope.flatten()
        aspect_f = aspect.flatten()
        area_f = np.full_like(hand_f, pixel_size**2)  # uniform pixel area for UTM

        valid_f = (hand_f >= 0) & np.isfinite(hand_f) & np.isfinite(aspect_f)
        print(
            f"  Valid pixels: {np.sum(valid_f)} ({100 * np.sum(valid_f) / valid_f.size:.1f}%)"
        )

        # Compute HAND bins
        hand_bounds = compute_hand_bins(
            hand_f, aspect_f, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
        )
        print(f"  HAND bin boundaries: {hand_bounds}")

        elements = []
        for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
            asp_mask = get_aspect_mask(aspect_f, asp_bin) & valid_f
            asp_indices = np.where(asp_mask)[0]

            if len(asp_indices) == 0:
                print(f"  {asp_name}: no pixels")
                for h_idx in range(N_HAND_BINS):
                    elements.append(
                        {
                            "aspect_name": asp_name,
                            "aspect_bin": asp_idx,
                            "hand_bin": h_idx,
                            "height": 0,
                            "distance": 0,
                            "area": 0,
                            "slope": 0,
                            "aspect": 0,
                            "width": 0,
                        }
                    )
                continue

            hillslope_frac = np.sum(area_f[asp_indices]) / np.sum(area_f[valid_f])

            # n_hillslopes from drainage_id
            if hasattr(grid, "drainage_id"):
                n_hillslopes = len(np.unique(grid.drainage_id.flatten()[asp_indices]))
            else:
                n_hillslopes = 1
            n_hillslopes = max(n_hillslopes, 1)

            # Fit trapezoidal width model
            trap = fit_trapezoidal_width(
                dtnd_f[asp_indices],
                area_f[asp_indices],
                n_hillslopes,
                min_dtnd=pixel_size,
            )
            trap_slope = trap["slope"]
            trap_width = trap["width"]
            trap_area = trap["area"]

            print(
                f"  {asp_name}: {len(asp_indices)} px ({hillslope_frac:.1%}), "
                f"trap slope={trap_slope:.4f}, width={trap_width:.0f} m, "
                f"area={trap_area:.0f} m²"
            )

            # Compute area fractions per HAND bin
            bin_raw_areas = []
            bin_data = []
            for h_idx in range(N_HAND_BINS):
                h_lower = hand_bounds[h_idx]
                h_upper = hand_bounds[h_idx + 1]
                hand_mask = (hand_f >= h_lower) & (hand_f < h_upper)
                bin_mask = asp_mask & hand_mask
                bin_indices = np.where(bin_mask)[0]
                if len(bin_indices) == 0:
                    bin_raw_areas.append(0)
                    bin_data.append(
                        {"indices": None, "h_lower": h_lower, "h_upper": h_upper}
                    )
                else:
                    bin_raw_areas.append(float(np.sum(area_f[bin_indices])))
                    bin_data.append(
                        {"indices": bin_indices, "h_lower": h_lower, "h_upper": h_upper}
                    )

            total_raw = sum(bin_raw_areas)
            area_fractions = (
                [a / total_raw for a in bin_raw_areas]
                if total_raw > 0
                else [0.25] * N_HAND_BINS
            )
            fitted_areas = [trap_area * frac for frac in area_fractions]

            # Compute parameters per HAND bin
            for h_idx in range(N_HAND_BINS):
                data = bin_data[h_idx]
                bin_indices = data["indices"]
                h_lower = data["h_lower"]
                h_upper = data["h_upper"]

                if bin_indices is None:
                    elements.append(
                        {
                            "aspect_name": asp_name,
                            "aspect_bin": asp_idx,
                            "hand_bin": h_idx,
                            "height": float((h_lower + h_upper) / 2),
                            "distance": 0,
                            "area": 0,
                            "slope": 0,
                            "aspect": float((asp_bin[0] + asp_bin[1]) / 2 % 360),
                            "width": 0,
                        }
                    )
                    continue

                mean_hand = float(np.mean(hand_f[bin_indices]))
                mean_slope = float(np.nanmean(slope_f[bin_indices]))
                mean_aspect = circular_mean_aspect(aspect_f[bin_indices])

                dtnd_sorted = np.sort(dtnd_f[bin_indices])
                median_dtnd = float(dtnd_sorted[len(dtnd_sorted) // 2])

                fitted_area = fitted_areas[h_idx]

                # Width from trapezoidal model
                da = sum(fitted_areas[:h_idx]) if h_idx > 0 else 0
                if trap_slope != 0:
                    try:
                        le = quadratic([trap_slope, trap_width, -da])
                        width = trap_width + 2 * trap_slope * le
                    except RuntimeError:
                        width = trap_width * (1 - 0.15 * h_idx)
                else:
                    width = trap_width

                width = max(float(width), 1)

                elements.append(
                    {
                        "aspect_name": asp_name,
                        "aspect_bin": asp_idx,
                        "hand_bin": h_idx,
                        "height": mean_hand,
                        "distance": median_dtnd,
                        "area": fitted_area,
                        "slope": mean_slope,
                        "aspect": mean_aspect,
                        "width": width,
                    }
                )

        hillslope_params = {
            "metadata": {
                "n_aspect_bins": N_ASPECT_BINS,
                "n_hand_bins": N_HAND_BINS,
                "aspect_bins": ASPECT_BINS,
                "aspect_names": ASPECT_NAMES,
                "hand_bounds": hand_bounds.tolist(),
                "accum_threshold": accum_threshold,
                "lc_m": LC_M,
                "pixel_size_m": pixel_size,
                "dem_shape": list(dem_data.shape),
                "subsample": args.subsample,
            },
            "elements": elements,
        }

        timings["hillslope_params"] = time.time() - t0
        print(f"\n  Time: {timings['hillslope_params']:.1f}s")
        print(f"  Elements computed: {len(elements)}")

        # Save hillslope params
        params_path = os.path.join(OUTPUT_DIR, "r6c10_hillslope_params.json")
        with open(params_path, "w") as f:
            json.dump(hillslope_params, f, indent=2)
        print(f"  Saved: {params_path}")

    # -----------------------------------------------------------------------
    # Step 8: Diagnostic plots
    # -----------------------------------------------------------------------
    section("Step 8: Diagnostic Plots")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # --- 6-panel terrain map ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"R6C10 UTM Smoke Test — {dem_data.shape[0]}x{dem_data.shape[1]} @ {pixel_size}m",
        fontsize=14,
    )

    # DEM
    ax = axes[0, 0]
    im = ax.imshow(dem_data, cmap="terrain")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    ax.set_title("DEM")

    # Stream network
    ax = axes[0, 1]
    ax.imshow(dem_data, cmap="terrain", alpha=0.5)
    stream_overlay = np.ma.masked_where(~stream_bool, np.ones_like(dem_data))
    ax.imshow(stream_overlay, cmap="Blues", alpha=0.8)
    ax.set_title(f"Stream Network (A_thresh={accum_threshold})")

    # HAND
    ax = axes[0, 2]
    hand_plot = np.where(np.isfinite(hand), hand, np.nan)
    im = ax.imshow(hand_plot, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="HAND (m)")
    ax.set_title("Height Above Nearest Drainage")

    # DTND
    ax = axes[1, 0]
    dtnd_plot = np.where(np.isfinite(dtnd), dtnd, np.nan)
    im = ax.imshow(dtnd_plot, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="DTND (m)")
    ax.set_title("Distance To Nearest Drainage")

    # Slope
    ax = axes[1, 1]
    slope_plot = np.where(np.isfinite(slope), slope, np.nan)
    im = ax.imshow(slope_plot, cmap="magma", norm=Normalize(vmin=0, vmax=0.3))
    plt.colorbar(im, ax=ax, label="Slope (m/m)")
    ax.set_title("Slope (Horn 1981)")

    # Aspect
    ax = axes[1, 2]
    aspect_plot = np.where(np.isfinite(aspect), aspect, np.nan)
    im = ax.imshow(aspect_plot, cmap="hsv", norm=Normalize(vmin=0, vmax=360))
    plt.colorbar(im, ax=ax, label="Aspect (deg)")
    ax.set_title("Aspect (Horn 1981)")

    plt.tight_layout()
    terrain_path = os.path.join(OUTPUT_DIR, "r6c10_terrain_maps.png")
    fig.savefig(terrain_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {terrain_path}")

    # --- DTND comparison plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "DTND Comparison: pysheds (hydrological) vs EDT (Euclidean)", fontsize=13
    )

    ax = axes[0]
    im = ax.imshow(dtnd_plot, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="DTND (m)")
    ax.set_title("pysheds DTND (flow-linked)")

    ax = axes[1]
    im = ax.imshow(edt_dtnd, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="DTND (m)")
    ax.set_title("EDT DTND (nearest)")

    ax = axes[2]
    diff = dtnd - edt_dtnd
    diff_plot = np.where(np.isfinite(diff), diff, np.nan)
    vmax = np.nanpercentile(np.abs(diff_plot), 99)
    im = ax.imshow(diff_plot, cmap="RdBu_r", norm=Normalize(vmin=-vmax, vmax=vmax))
    plt.colorbar(im, ax=ax, label="Difference (m)")
    ax.set_title("pysheds - EDT (positive = farther by flow)")

    plt.tight_layout()
    dtnd_path = os.path.join(OUTPUT_DIR, "r6c10_dtnd_comparison.png")
    fig.savefig(dtnd_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dtnd_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("Results Summary")

    timings["total"] = time.time() - t_total

    n_pass = sum(1 for r in results if r["passed"])
    n_fail = sum(1 for r in results if not r["passed"])

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['detail']}")

    print(f"\n  {n_pass} passed, {n_fail} failed, {len(results)} total")
    print(f"  Total time: {timings['total']:.1f}s")

    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n  RESULT: {overall}")

    # Save validation JSON
    validation = {
        "result": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "checks": results,
        "timings": timings,
        "config": {
            "dem": os.path.basename(DEM_PATH),
            "crs": str(crs),
            "pixel_size_m": pixel_size,
            "subsample": args.subsample,
            "lc_m": LC_M,
            "accum_threshold": accum_threshold,
            "dem_shape": list(dem_data.shape),
        },
        "stats": {
            "hand_min": float(np.min(hand_valid)),
            "hand_max": float(np.max(hand_valid)),
            "hand_mean": float(np.mean(hand_valid)),
            "dtnd_min": float(np.min(dtnd_valid)),
            "dtnd_max": float(np.max(dtnd_valid)),
            "dtnd_mean": float(np.mean(dtnd_valid)),
            "slope_min": float(np.min(slope_valid)),
            "slope_max": float(np.max(slope_valid)),
            "slope_mean": float(np.mean(slope_valid)),
            "aspect_quadrant_fracs": quadrant_fracs,
            "n_stream_pixels": n_stream_pixels,
            "n_channels": n_channels,
        },
    }

    val_path = os.path.join(OUTPUT_DIR, "r6c10_validation.json")
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"  Saved: {val_path}")

    # Save text summary
    summary_path = os.path.join(OUTPUT_DIR, "r6c10_summary.txt")
    with open(summary_path, "w") as f:
        f.write("R6C10 UTM Smoke Test Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Result: {overall}\n")
        f.write(f"Checks: {n_pass} pass, {n_fail} fail\n\n")
        f.write(f"DEM: {os.path.basename(DEM_PATH)}\n")
        f.write(f"CRS: {crs} (geographic={is_geographic})\n")
        f.write(f"Shape: {dem_data.shape}, pixel_size: {pixel_size}m\n")
        f.write(f"Lc: {LC_M}m, A_thresh: {accum_threshold} cells\n\n")
        f.write("Terrain Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"HAND:   [{np.min(hand_valid):.2f}, {np.max(hand_valid):.2f}] m, "
            f"mean={np.mean(hand_valid):.2f}\n"
        )
        f.write(
            f"DTND:   [{np.min(dtnd_valid):.2f}, {np.max(dtnd_valid):.2f}] m, "
            f"mean={np.mean(dtnd_valid):.2f}\n"
        )
        f.write(
            f"Slope:  [{np.min(slope_valid):.4f}, {np.max(slope_valid):.4f}] m/m, "
            f"mean={np.mean(slope_valid):.4f}\n"
        )
        f.write(f"Aspect: {frac_str}\n\n")
        f.write("Timings\n")
        f.write("-" * 40 + "\n")
        for k, v in timings.items():
            f.write(f"  {k}: {v:.1f}s\n")
    print(f"  Saved: {summary_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
