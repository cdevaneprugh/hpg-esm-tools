#!/usr/bin/env python
"""
5x5 UTM Smoke Test — Phase A Validation at Scale

Validates the UTM CRS code path in the pysheds fork on 25 NEON tiles
(R6-R10, C7-C11, 5000x5000 at 1m, 25M pixels, 0% nodata).

Tests the same CRS-dependent pysheds functions as run_r6c10_utm.py but
on a domain 25x larger — confirms UTM handling scales correctly and
catches any issues that only appear with larger/more complex terrain.

Usage:
    python scripts/smoke_tests/run_5x5_utm.py [--subsample N] [--skip-hillslope]

Output:
    output/osbs/smoke_tests/5x5_utm/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.merge import merge

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

# Tile grid (from run_pipeline.py)
TILE_GRID_ORIGIN_EASTING = 394000
TILE_GRID_ORIGIN_NORTHING = 3292000
TILE_SIZE = 1000  # meters per tile

DTM_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/neon/dtm"
OUTPUT_DIR = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/osbs/smoke_tests/5x5_utm"
)

# 5x5 tile block: R6-R10, C7-C11
TILE_ROWS = range(6, 11)  # 6, 7, 8, 9, 10
TILE_COLS = range(7, 12)  # 7, 8, 9, 10, 11
EXPECTED_SHAPE = (5000, 5000)

# Phase C working Lc (~300m)
LC_M = 300.0

# Hillslope parameters
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0
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


def tile_filepath(row: int, col: int) -> str:
    """Convert (row, col) to tile filepath."""
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
    return os.path.join(DTM_DIR, f"NEON_D03_OSBS_DP3_{easting}_{northing}_DTM.tif")


def section(title: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {'=' * 56}")
    print(f"[{ts}]   {title}")
    print(f"[{ts}] {'=' * 56}\n")
    sys.stdout.flush()


def progress(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def check(name: str, passed: bool, detail: str, results: list) -> None:
    """Record a validation check result."""
    passed = bool(passed)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    results.append({"name": name, "passed": passed, "detail": detail})


# Lc sanity check thresholds (from validate_lc_physical.py)
LC_PASS_LO, LC_PASS_HI = 0.5, 2.0
LC_MARGINAL_LO, LC_MARGINAL_HI = 0.3, 3.0


def lc_verdict(ratio: float) -> str:
    """Classify an Lc ratio as PASS / MARGINAL / FAIL."""
    if LC_PASS_LO <= ratio <= LC_PASS_HI:
        return "PASS"
    if LC_MARGINAL_LO <= ratio <= LC_MARGINAL_HI:
        return "MARGINAL"
    return "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="5x5 UTM smoke test")
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

    # -------------------------------------------------------------------
    # Step 1: Load and merge 25 tiles
    # -------------------------------------------------------------------
    section("Step 1: Load 5x5 Tile Block (R6-R10, C7-C11)")

    tile_paths = []
    for r in TILE_ROWS:
        for c in TILE_COLS:
            fp = tile_filepath(r, c)
            if not os.path.exists(fp):
                print(f"  MISSING: {os.path.basename(fp)}")
                print("FATAL: Cannot proceed without all 25 tiles.")
                return 1
            tile_paths.append(fp)
    progress(f"Found all {len(tile_paths)} tiles")

    t0 = time.time()
    datasets = [rasterio.open(p) for p in tile_paths]
    dem_data, transform = merge(datasets)[:2]
    for ds in datasets:
        ds.close()

    dem_data = dem_data.squeeze()
    timings["tile_merge"] = time.time() - t0
    progress(f"Merged shape: {dem_data.shape} (expected {EXPECTED_SHAPE})")
    progress(f"Merge time: {timings['tile_merge']:.1f}s")

    if dem_data.shape != EXPECTED_SHAPE:
        print(
            f"  WARNING: Shape mismatch — expected {EXPECTED_SHAPE}, "
            f"got {dem_data.shape}"
        )

    # Read CRS from first tile
    with rasterio.open(tile_paths[0]) as src:
        crs = src.crs
        pixel_size = src.transform.a

    # Handle nodata
    dem_data = dem_data.astype(np.float64)
    dem_data[dem_data == -9999.0] = np.nan
    n_nodata = int(np.sum(~np.isfinite(dem_data)))

    progress(f"CRS: {crs}, pixel size: {pixel_size} m")
    progress(f"Nodata pixels: {n_nodata} ({100 * n_nodata / dem_data.size:.2f}%)")

    if n_nodata > 0:
        progress(f"WARNING: Expected 0 nodata for this interior block, got {n_nodata}")

    # Subsample if requested
    if args.subsample > 1:
        dem_data = dem_data[:: args.subsample, :: args.subsample]
        pixel_size *= args.subsample
        transform = rasterio.Affine(
            transform.a * args.subsample,
            transform.b,
            transform.c,
            transform.d,
            transform.e * args.subsample,
            transform.f,
        )
        progress(
            f"Subsampled {args.subsample}x -> shape {dem_data.shape}, "
            f"pixel_size {pixel_size} m"
        )

    elev_valid = dem_data[np.isfinite(dem_data)]
    progress(f"Elevation range: [{np.min(elev_valid):.2f}, {np.max(elev_valid):.2f}] m")
    progress(f"Elevation mean: {np.mean(elev_valid):.2f} m")

    # A_thresh in cells
    accum_threshold = int(0.5 * LC_M**2 / pixel_size**2)
    progress(
        f"Lc = {LC_M} m -> A_thresh = {accum_threshold} cells (at {pixel_size}m pixels)"
    )

    # -------------------------------------------------------------------
    # Step 2: Initialize pysheds Grid and check CRS detection
    # -------------------------------------------------------------------
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
    progress(f"grid._crs_is_geographic() = {is_geographic}")
    check(
        "CRS detection",
        not is_geographic,
        f"Expected False for UTM, got {is_geographic}",
        results,
    )

    # -------------------------------------------------------------------
    # Step 3: DEM conditioning + flow routing
    # -------------------------------------------------------------------
    section("Step 3: DEM Conditioning + Flow Routing")

    t0 = time.time()

    progress("Filling pits...")
    grid.fill_pits("dem", out_name="pit_filled")

    progress("Filling depressions...")
    grid.fill_depressions("pit_filled", out_name="flooded")

    progress("Resolving flats...")
    grid.resolve_flats("flooded", out_name="inflated")

    progress("Computing flow direction (D8)...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")

    progress("Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")
    acc = grid.acc

    timings["conditioning_and_routing"] = time.time() - t0
    progress(f"Time: {timings['conditioning_and_routing']:.1f}s")
    progress(f"Max accumulation: {np.nanmax(acc):.0f} cells")

    # Create stream network
    acc_mask = acc > accum_threshold
    n_stream_pixels = int(np.sum(acc_mask))
    progress(f"Stream pixels (acc > {accum_threshold}): {n_stream_pixels}")

    check(
        "Stream network exists",
        n_stream_pixels > 0,
        f"{n_stream_pixels} stream pixels",
        results,
    )

    progress("Creating channel mask...")
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")
    stream_mask = grid.channel_mask
    channel_id = grid.channel_id

    n_channels = len(np.unique(channel_id[channel_id > 0]))
    progress(f"Unique channel segments: {n_channels}")

    # -------------------------------------------------------------------
    # Step 4: HAND + DTND (Phase A core test)
    # -------------------------------------------------------------------
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
    progress(f"Time: {timings['hand_dtnd']:.1f}s")

    # Mask invalid values
    valid_mask = np.isfinite(hand) & np.isfinite(dtnd)
    hand_valid = hand[valid_mask]
    dtnd_valid = dtnd[valid_mask]

    progress(f"Valid pixels: {np.sum(valid_mask)} / {hand.size}")
    progress(f"HAND range: [{np.min(hand_valid):.2f}, {np.max(hand_valid):.2f}] m")
    progress(f"HAND mean: {np.mean(hand_valid):.2f} m")
    progress(f"DTND range: [{np.min(dtnd_valid):.2f}, {np.max(dtnd_valid):.2f}] m")
    progress(f"DTND mean: {np.mean(dtnd_valid):.2f} m")

    # Validation checks (relaxed thresholds for larger domain)
    check(
        "HAND >= -1m",
        bool(np.all(hand_valid >= -1.0)),
        f"min = {np.min(hand_valid):.4f}",
        results,
    )
    check(
        "HAND max <= 25m",
        bool(np.max(hand_valid) <= 25),
        f"max = {np.max(hand_valid):.2f} m",
        results,
    )
    check(
        "DTND >= 0",
        bool(np.all(dtnd_valid >= 0)),
        f"min = {np.min(dtnd_valid):.4f}",
        results,
    )
    check(
        "DTND max <= 1500m",
        bool(np.max(dtnd_valid) <= 1500),
        f"max = {np.max(dtnd_valid):.2f} m",
        results,
    )

    # Stream pixels should have HAND = 0 and DTND = 0
    stream_bool = np.array(stream_mask, dtype=bool)
    stream_flat = stream_bool.flatten()
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

    # -------------------------------------------------------------------
    # Step 5: Slope + Aspect (Phase A core test)
    # -------------------------------------------------------------------
    section("Step 5: Slope + Aspect (Phase A Core Test)")

    t0 = time.time()

    grid.slope_aspect("dem")
    slope = np.array(grid.slope, dtype=np.float64)
    aspect = np.array(grid.aspect, dtype=np.float64)

    timings["slope_aspect"] = time.time() - t0
    progress(f"Time: {timings['slope_aspect']:.1f}s")

    slope_valid = slope[np.isfinite(slope)]
    aspect_valid = aspect[np.isfinite(aspect)]

    progress(f"Slope range: [{np.min(slope_valid):.4f}, {np.max(slope_valid):.4f}] m/m")
    progress(f"Slope mean: {np.mean(slope_valid):.4f}")

    check(
        "Slope mean in 0.03-0.10",
        0.03 <= np.mean(slope_valid) <= 0.10,
        f"mean = {np.mean(slope_valid):.4f}",
        results,
    )

    # Aspect distribution: all 4 quadrants should have >= 10%
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

    # -------------------------------------------------------------------
    # Step 6: Lc Sanity Checks (Swenson Section 2.4)
    # -------------------------------------------------------------------
    section("Step 6: Lc Sanity Checks (Swenson Section 2.4)")

    t0 = time.time()

    # --- Check 1: max(DTND) / Lc ---
    nonstream = valid_mask & ~stream_bool
    dtnd_ns = dtnd[nonstream]

    max_dtnd = float(np.max(dtnd_ns))
    p99_dtnd = float(np.percentile(dtnd_ns, 99))
    p95_dtnd = float(np.percentile(dtnd_ns, 95))
    mean_dtnd_ns = float(np.mean(dtnd_ns))

    ratio_max = max_dtnd / LC_M
    ratio_p99 = p99_dtnd / LC_M

    verdict_max = lc_verdict(ratio_max)
    verdict_p99 = lc_verdict(ratio_p99)

    progress("DTND stats (non-stream):")
    progress(
        f"  max     = {max_dtnd:.1f} m  (max/Lc = {ratio_max:.3f})  [{verdict_max}]"
    )
    progress(
        f"  P99     = {p99_dtnd:.1f} m  (P99/Lc = {ratio_p99:.3f})  [{verdict_p99}]"
    )
    progress(f"  P95     = {p95_dtnd:.1f} m")
    progress(f"  mean    = {mean_dtnd_ns:.1f} m")
    progress("  Swenson calibration: max(DTND) ~ Lc (ratio ~ 1.0)")

    check(
        "Lc Check 1: max(DTND)/Lc",
        verdict_max in ("PASS", "MARGINAL"),
        f"ratio = {ratio_max:.3f} [{verdict_max}]",
        results,
    )
    check(
        "Lc Check 1 (P99): P99(DTND)/Lc",
        verdict_p99 in ("PASS", "MARGINAL"),
        f"ratio = {ratio_p99:.3f} [{verdict_p99}] (supplementary — less outlier-prone)",
        results,
    )

    # --- Check 2: mean(catchment area) / Lc^2 ---
    drainage_id = np.array(grid.drainage_id, dtype=np.int64)
    valid_drain = drainage_id[np.isfinite(hand) & (drainage_id > 0)]
    unique_ids, counts = np.unique(valid_drain, return_counts=True)
    catchment_areas = counts.astype(np.float64) * pixel_size**2

    n_total_catchments = len(unique_ids)

    # Exclude edge catchments (touching any domain boundary)
    edge_pixels = np.concatenate(
        [
            drainage_id[0, :],  # top row
            drainage_id[-1, :],  # bottom row
            drainage_id[:, 0],  # left column
            drainage_id[:, -1],  # right column
        ]
    )
    edge_ids = set(np.unique(edge_pixels[edge_pixels > 0]))
    n_edge = len(edge_ids)

    interior_mask_c = np.array([uid not in edge_ids for uid in unique_ids])
    interior_areas = catchment_areas[interior_mask_c]
    n_interior = int(np.sum(interior_mask_c))

    lc_sq = LC_M**2
    if n_interior > 0:
        mean_area_interior = float(np.mean(interior_areas))
        ratio_area = mean_area_interior / lc_sq
        verdict_area = lc_verdict(ratio_area)
    else:
        mean_area_interior = 0.0
        ratio_area = 0.0
        verdict_area = "FAIL"

    mean_area_all = float(np.mean(catchment_areas))
    ratio_area_all = mean_area_all / lc_sq

    progress(f"Total catchments: {n_total_catchments}")
    progress(f"Edge catchments: {n_edge}")
    progress(f"Interior catchments: {n_interior}")
    progress(
        f"  mean interior area = {mean_area_interior:.0f} m^2  "
        f"(mean/Lc^2 = {ratio_area:.3f})  [{verdict_area}]"
    )
    progress(
        f"  mean all area      = {mean_area_all:.0f} m^2  "
        f"(mean/Lc^2 = {ratio_area_all:.3f})"
    )
    progress(f"  Lc^2 = {lc_sq:.0f} m^2")
    progress("  Swenson calibration: mean(catchment area) / Lc^2 ~ 0.94")

    check(
        "Lc Check 2: mean(catchment area)/Lc^2",
        verdict_area in ("PASS", "MARGINAL"),
        f"ratio = {ratio_area:.3f} [{verdict_area}] (interior catchments only)",
        results,
    )

    timings["lc_checks"] = time.time() - t0

    # Store Lc check results for JSON summary
    lc_check_results = {
        "lc_m": LC_M,
        "lc_squared": lc_sq,
        "check1_max_dtnd": max_dtnd,
        "check1_p99_dtnd": p99_dtnd,
        "check1_p95_dtnd": p95_dtnd,
        "check1_mean_dtnd": mean_dtnd_ns,
        "check1_max_dtnd_over_lc": ratio_max,
        "check1_p99_dtnd_over_lc": ratio_p99,
        "check1_verdict": verdict_max,
        "check2_n_total_catchments": n_total_catchments,
        "check2_n_edge_catchments": n_edge,
        "check2_n_interior_catchments": n_interior,
        "check2_mean_area_interior": mean_area_interior,
        "check2_mean_area_all": mean_area_all,
        "check2_mean_area_over_lc2": ratio_area,
        "check2_verdict": verdict_area,
    }

    # -------------------------------------------------------------------
    # Step 7: Hillslope parameters (optional)
    # -------------------------------------------------------------------
    hillslope_params = None

    if not args.skip_hillslope:
        section("Step 7: Hillslope Parameters")

        t0 = time.time()

        # Flatten arrays
        hand_hp = hand.flatten()
        dtnd_hp = dtnd.flatten()
        slope_hp = slope.flatten()
        aspect_hp = aspect.flatten()
        area_hp = np.full_like(hand_hp, pixel_size**2)

        valid_hp = (hand_hp >= 0) & np.isfinite(hand_hp) & np.isfinite(aspect_hp)
        progress(
            f"Valid pixels: {np.sum(valid_hp)} "
            f"({100 * np.sum(valid_hp) / valid_hp.size:.1f}%)"
        )

        # HAND bins
        hand_bounds = compute_hand_bins(
            hand_hp, aspect_hp, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
        )
        progress(f"HAND bin boundaries: {hand_bounds}")

        elements = []
        for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
            asp_mask = get_aspect_mask(aspect_hp, asp_bin) & valid_hp
            asp_indices = np.where(asp_mask)[0]

            if len(asp_indices) == 0:
                progress(f"{asp_name}: no pixels")
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

            hillslope_frac = np.sum(area_hp[asp_indices]) / np.sum(area_hp[valid_hp])

            if hasattr(grid, "drainage_id"):
                n_hillslopes = len(np.unique(grid.drainage_id.flatten()[asp_indices]))
            else:
                n_hillslopes = 1
            n_hillslopes = max(n_hillslopes, 1)

            # Fit trapezoidal width model
            trap = fit_trapezoidal_width(
                dtnd_hp[asp_indices],
                area_hp[asp_indices],
                n_hillslopes,
                min_dtnd=pixel_size,
            )
            trap_slope = trap["slope"]
            trap_width = trap["width"]
            trap_area = trap["area"]

            progress(
                f"{asp_name}: {len(asp_indices)} px ({hillslope_frac:.1%}), "
                f"trap slope={trap_slope:.4f}, width={trap_width:.0f} m, "
                f"area={trap_area:.0f} m^2"
            )

            # Area fractions per HAND bin
            bin_raw_areas = []
            bin_data = []
            for h_idx in range(N_HAND_BINS):
                h_lower = hand_bounds[h_idx]
                h_upper = hand_bounds[h_idx + 1]
                hand_mask = (hand_hp >= h_lower) & (hand_hp < h_upper)
                bin_mask = asp_mask & hand_mask
                bin_indices = np.where(bin_mask)[0]
                if len(bin_indices) == 0:
                    bin_raw_areas.append(0)
                    bin_data.append(
                        {"indices": None, "h_lower": h_lower, "h_upper": h_upper}
                    )
                else:
                    bin_raw_areas.append(float(np.sum(area_hp[bin_indices])))
                    bin_data.append(
                        {
                            "indices": bin_indices,
                            "h_lower": h_lower,
                            "h_upper": h_upper,
                        }
                    )

            total_raw = sum(bin_raw_areas)
            area_fractions = (
                [a / total_raw for a in bin_raw_areas]
                if total_raw > 0
                else [0.25] * N_HAND_BINS
            )
            fitted_areas = [trap_area * frac for frac in area_fractions]

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

                mean_hand = float(np.mean(hand_hp[bin_indices]))
                mean_slope = float(np.nanmean(slope_hp[bin_indices]))
                mean_aspect = circular_mean_aspect(aspect_hp[bin_indices])
                dtnd_sorted = np.sort(dtnd_hp[bin_indices])
                median_dtnd = float(dtnd_sorted[len(dtnd_sorted) // 2])
                fitted_area = fitted_areas[h_idx]

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

        n_elements = len(elements)
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
                "tiles": f"R{min(TILE_ROWS)}-R{max(TILE_ROWS)}, "
                f"C{min(TILE_COLS)}-C{max(TILE_COLS)}",
            },
            "elements": elements,
        }

        timings["hillslope_params"] = time.time() - t0
        progress(f"Elements computed: {n_elements}")
        progress(f"Time: {timings['hillslope_params']:.1f}s")

        check(
            "16 hillslope elements computed",
            n_elements == 16,
            f"{n_elements} elements",
            results,
        )

        # Save hillslope params
        params_path = os.path.join(OUTPUT_DIR, "5x5_hillslope_params.json")
        with open(params_path, "w") as f:
            json.dump(hillslope_params, f, indent=2)
        progress(f"Saved: {params_path}")

    # -------------------------------------------------------------------
    # Step 8: Diagnostic plots
    # -------------------------------------------------------------------
    section("Step 8: Diagnostic Plots")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # --- 6-panel terrain map ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"5x5 UTM Smoke Test — {dem_data.shape[0]}x{dem_data.shape[1]} "
        f"@ {pixel_size}m, R6-R10 C7-C11",
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
    terrain_path = os.path.join(OUTPUT_DIR, "5x5_terrain_maps.png")
    fig.savefig(terrain_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {terrain_path}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
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
            "tiles": f"R{min(TILE_ROWS)}-R{max(TILE_ROWS)}, "
            f"C{min(TILE_COLS)}-C{max(TILE_COLS)}",
            "n_tiles": len(tile_paths),
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
        "lc_checks": lc_check_results,
    }

    val_path = os.path.join(OUTPUT_DIR, "5x5_validation.json")
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    progress(f"Saved: {val_path}")

    # Save text summary
    summary_path = os.path.join(OUTPUT_DIR, "5x5_summary.txt")
    with open(summary_path, "w") as f:
        f.write("5x5 UTM Smoke Test Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Result: {overall}\n")
        f.write(f"Checks: {n_pass} pass, {n_fail} fail\n\n")
        f.write(
            f"Tiles: R{min(TILE_ROWS)}-R{max(TILE_ROWS)}, "
            f"C{min(TILE_COLS)}-C{max(TILE_COLS)} ({len(tile_paths)} tiles)\n"
        )
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
        f.write("Lc Sanity Checks (Swenson Section 2.4)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Check 1: max(DTND)/Lc = {ratio_max:.3f}  [{verdict_max}]\n")
        f.write(f"         P99(DTND)/Lc = {ratio_p99:.3f}  [{verdict_p99}]\n")
        f.write(
            f"Check 2: mean(catchment area)/Lc^2 = {ratio_area:.3f}  [{verdict_area}]\n"
        )
        f.write(
            f"         Interior catchments: {n_interior} / {n_total_catchments}\n\n"
        )
        f.write("Timings\n")
        f.write("-" * 40 + "\n")
        for k, v in timings.items():
            f.write(f"  {k}: {v:.1f}s\n")
    progress(f"Saved: {summary_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
