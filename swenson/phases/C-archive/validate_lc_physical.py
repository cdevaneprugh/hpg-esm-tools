#!/usr/bin/env python
"""
Lc Physical Validation — Phase C Final Check

Validates the working Lc ~300m against two physical criteria from
Swenson & Lawrence (2025) Section 2.4:

  1. Lc ~ max(DTND):  Largest ridge-to-channel distance ≈ Lc
  2. Lc^2 ~ mean(catchment area):  Mean catchment area ≈ Lc^2

Test region: 5x5 tile block (R6-R10, C7-C11), 5000x5000 pixels at 1m.
All 25 tiles are 0% nodata (deep interior of contiguous block).

Loops over Lc = {285, 300, 356} m to test both endpoints of the
Phase C range plus the round working value.

Usage:
    python scripts/smoke_tests/validate_lc_physical.py

Output:
    output/osbs/smoke_tests/lc_physical_validation/
"""

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tile grid (from run_pipeline.py)
TILE_GRID_ORIGIN_EASTING = 394000
TILE_GRID_ORIGIN_NORTHING = 3292000
TILE_SIZE = 1000  # meters per tile

DTM_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/neon/dtm"
OUTPUT_DIR = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
    "/output/osbs/smoke_tests/lc_physical_validation"
)

# 5x5 tile block: R6-R10, C7-C11
TILE_ROWS = range(6, 11)  # 6, 7, 8, 9, 10
TILE_COLS = range(7, 11 + 1)  # 7, 8, 9, 10, 11
EXPECTED_SHAPE = (5000, 5000)  # 5 rows x 5 cols at 1000px each

# Lc values to test (Phase C range endpoints + working value)
LC_VALUES = [285.0, 300.0, 356.0]

# D8 direction map (pysheds convention)
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# Pass/fail thresholds
PASS_LO, PASS_HI = 0.5, 2.0
MARGINAL_LO, MARGINAL_HI = 0.3, 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def verdict(ratio: float) -> str:
    """Classify a ratio as PASS / MARGINAL / FAIL."""
    if PASS_LO <= ratio <= PASS_HI:
        return "PASS"
    if MARGINAL_LO <= ratio <= MARGINAL_HI:
        return "MARGINAL"
    return "FAIL"


def tile_filepath(row: int, col: int) -> str:
    """Convert (row, col) to tile filepath."""
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
    return os.path.join(DTM_DIR, f"NEON_D03_OSBS_DP3_{easting}_{northing}_DTM.tif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    # merge returns (count, rows, cols) — squeeze to 2D
    dem_data = dem_data.squeeze()
    timings["tile_merge"] = time.time() - t0
    progress(f"Merged shape: {dem_data.shape} (expected {EXPECTED_SHAPE})")
    progress(f"Merge time: {timings['tile_merge']:.1f}s")

    if dem_data.shape != EXPECTED_SHAPE:
        print(
            f"  WARNING: Shape mismatch — expected {EXPECTED_SHAPE}, got {dem_data.shape}"
        )

    # Read CRS from first tile
    with rasterio.open(tile_paths[0]) as src:
        crs = src.crs
        pixel_size = src.transform.a
    progress(f"CRS: {crs}, pixel size: {pixel_size} m")

    # Handle nodata
    nodata_count = int(np.sum(dem_data == -9999.0))
    nan_count = int(np.sum(~np.isfinite(dem_data)))
    progress(f"Nodata pixels (-9999): {nodata_count}")
    progress(f"Non-finite pixels: {nan_count}")

    dem_data = dem_data.astype(np.float64)
    dem_data[dem_data == -9999.0] = np.nan
    n_nodata = int(np.sum(~np.isfinite(dem_data)))
    n_valid = int(np.sum(np.isfinite(dem_data)))
    progress(
        f"Total nodata after conversion: {n_nodata} ({100 * n_nodata / dem_data.size:.2f}%)"
    )

    if n_nodata > 0:
        print(f"  WARNING: Expected 0 nodata for this interior block, got {n_nodata}")

    elev_valid = dem_data[np.isfinite(dem_data)]
    progress(f"Elevation range: [{np.min(elev_valid):.2f}, {np.max(elev_valid):.2f}] m")
    progress(
        f"Elevation mean: {np.mean(elev_valid):.2f} m, std: {np.std(elev_valid):.2f} m"
    )
    progress(
        f"Domain: {dem_data.shape[1] * pixel_size / 1000:.0f} km x {dem_data.shape[0] * pixel_size / 1000:.0f} km"
    )

    # -------------------------------------------------------------------
    # Step 2: DEM conditioning + flow routing (once, reused for all Lc)
    # -------------------------------------------------------------------
    section("Step 2: DEM Conditioning + Flow Routing")

    t0 = time.time()

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
    progress(f"CRS detection: geographic={is_geographic}")
    if is_geographic:
        print("  FATAL: Expected projected CRS (UTM), got geographic.")
        return 1

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

    # -------------------------------------------------------------------
    # Step 3: Loop over Lc values
    # -------------------------------------------------------------------
    section("Step 3: Physical Validation Across Lc Values")

    lc_results = {}

    for lc in LC_VALUES:
        a_thresh = int(0.5 * lc**2 / pixel_size**2)
        progress(f"\n--- Lc = {lc} m, A_thresh = {a_thresh} cells ---")

        t0 = time.time()

        # Stream network
        acc_mask = acc > a_thresh
        n_stream = int(np.sum(acc_mask))
        progress(f"  Stream pixels: {n_stream}")

        if n_stream == 0:
            print(f"  FATAL: No stream pixels at A_thresh={a_thresh}")
            lc_results[lc] = {"error": "no stream pixels"}
            continue

        progress("  Creating channel mask...")
        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

        progress("  Computing HAND + DTND...")
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
        drainage_id = np.array(grid.drainage_id, dtype=np.int64)

        t_lc = time.time() - t0
        progress(f"  HAND/DTND time: {t_lc:.1f}s")
        timings[f"hand_dtnd_lc{int(lc)}"] = t_lc

        # --- Check 1: max(DTND) / Lc ---
        stream_bool = np.array(grid.channel_mask, dtype=bool)
        nonstream = np.isfinite(dtnd) & ~stream_bool
        dtnd_ns = dtnd[nonstream]

        max_dtnd = float(np.max(dtnd_ns))
        p99_dtnd = float(np.percentile(dtnd_ns, 99))
        p95_dtnd = float(np.percentile(dtnd_ns, 95))
        mean_dtnd = float(np.mean(dtnd_ns))
        median_dtnd = float(np.median(dtnd_ns))

        ratio_max = max_dtnd / lc
        ratio_p99 = p99_dtnd / lc
        verdict_max = verdict(ratio_max)

        progress("  DTND stats (non-stream):")
        progress(
            f"    max     = {max_dtnd:.1f} m  (max/Lc = {ratio_max:.3f})  [{verdict_max}]"
        )
        progress(f"    P99     = {p99_dtnd:.1f} m  (P99/Lc = {ratio_p99:.3f})")
        progress(f"    P95     = {p95_dtnd:.1f} m")
        progress(f"    mean    = {mean_dtnd:.1f} m")
        progress(f"    median  = {median_dtnd:.1f} m")

        # --- Check 2: mean(catchment area) / Lc^2 ---
        # Identify unique catchments from drainage_id
        # drainage_id maps every pixel to the channel segment it drains to
        valid_drain = drainage_id[np.isfinite(hand) & (drainage_id > 0)]
        unique_ids, counts = np.unique(valid_drain, return_counts=True)
        # Area in m^2 (at 1m pixels, count = area)
        catchment_areas = counts.astype(np.float64) * pixel_size**2

        n_total_catchments = len(unique_ids)
        progress(f"  Total catchments: {n_total_catchments}")

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

        interior_mask = np.array([uid not in edge_ids for uid in unique_ids])
        interior_areas = catchment_areas[interior_mask]
        n_interior = int(np.sum(interior_mask))

        progress(f"  Edge catchments: {n_edge}")
        progress(f"  Interior catchments: {n_interior}")
        progress(f"  Edge fraction: {n_edge / max(n_total_catchments, 1):.1%}")

        lc_sq = lc**2
        if n_interior > 0:
            mean_area_interior = float(np.mean(interior_areas))
            median_area_interior = float(np.median(interior_areas))
            std_area_interior = float(np.std(interior_areas))
            ratio_area = mean_area_interior / lc_sq
            verdict_area = verdict(ratio_area)
        else:
            mean_area_interior = 0.0
            median_area_interior = 0.0
            std_area_interior = 0.0
            ratio_area = 0.0
            verdict_area = "FAIL"

        mean_area_all = float(np.mean(catchment_areas))
        ratio_area_all = mean_area_all / lc_sq

        progress("  Catchment area stats (interior only):")
        progress(
            f"    mean    = {mean_area_interior:.0f} m^2  (mean/Lc^2 = {ratio_area:.3f})  [{verdict_area}]"
        )
        progress(f"    median  = {median_area_interior:.0f} m^2")
        progress(f"    std     = {std_area_interior:.0f} m^2")
        progress("  Catchment area stats (all):")
        progress(
            f"    mean    = {mean_area_all:.0f} m^2  (mean/Lc^2 = {ratio_area_all:.3f})"
        )
        progress(f"    Lc^2    = {lc_sq:.0f} m^2")

        lc_results[lc] = {
            "a_thresh": a_thresh,
            "n_stream_pixels": n_stream,
            "n_total_catchments": n_total_catchments,
            "n_edge_catchments": n_edge,
            "n_interior_catchments": n_interior,
            "edge_fraction": n_edge / max(n_total_catchments, 1),
            "dtnd": {
                "max": max_dtnd,
                "p99": p99_dtnd,
                "p95": p95_dtnd,
                "mean": mean_dtnd,
                "median": median_dtnd,
            },
            "check1_max_dtnd_over_lc": ratio_max,
            "check1_p99_dtnd_over_lc": ratio_p99,
            "check1_verdict": verdict_max,
            "catchment_area": {
                "mean_interior": mean_area_interior,
                "median_interior": median_area_interior,
                "std_interior": std_area_interior,
                "mean_all": mean_area_all,
            },
            "lc_squared": lc_sq,
            "check2_mean_area_over_lc2": ratio_area,
            "check2_mean_area_over_lc2_all": ratio_area_all,
            "check2_verdict": verdict_area,
        }

    # -------------------------------------------------------------------
    # Step 4: Overall verdict
    # -------------------------------------------------------------------
    section("Step 4: Overall Verdict")

    # Use Lc=300 as the primary result
    primary = lc_results.get(300.0, {})
    v1 = primary.get("check1_verdict", "FAIL")
    v2 = primary.get("check2_verdict", "FAIL")

    if v1 == "PASS" and v2 == "PASS":
        overall = "PASS"
    elif v1 == "FAIL" or v2 == "FAIL":
        overall = "FAIL"
    else:
        overall = "MARGINAL"

    progress("Primary Lc = 300m:")
    progress(
        f"  Check 1 (max DTND / Lc):          {primary.get('check1_max_dtnd_over_lc', 'N/A'):.3f}  [{v1}]"
    )
    progress(
        f"  Check 2 (mean catchment / Lc^2):   {primary.get('check2_mean_area_over_lc2', 'N/A'):.3f}  [{v2}]"
    )
    progress(f"  Overall: {overall}")

    print("\n  Comparison across Lc values:")
    print(
        f"  {'Lc (m)':>8}  {'A_thresh':>10}  {'max DTND/Lc':>14}  {'V1':>8}  {'mean area/Lc^2':>16}  {'V2':>8}"
    )
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 14}  {'-' * 8}  {'-' * 16}  {'-' * 8}")
    for lc in LC_VALUES:
        r = lc_results.get(lc, {})
        if "error" in r:
            print(f"  {lc:8.0f}  {'ERROR':>10}")
            continue
        print(
            f"  {lc:8.0f}  {r['a_thresh']:10d}  "
            f"{r['check1_max_dtnd_over_lc']:14.3f}  {r['check1_verdict']:>8}  "
            f"{r['check2_mean_area_over_lc2']:16.3f}  {r['check2_verdict']:>8}"
        )

    # Swenson calibration point for reference
    print(
        "\n  Swenson Section 2.4 calibration (low-relief): max(DTND)/Lc = 0.90, mean(catch)/Lc^2 = 0.94"
    )

    # -------------------------------------------------------------------
    # Step 5: Diagnostic plots
    # -------------------------------------------------------------------
    section("Step 5: Diagnostic Plots")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Plot 1: Terrain overview with stream network at Lc=300 ---
    # Recompute stream network at Lc=300 for plotting
    a_thresh_300 = int(0.5 * 300.0**2 / pixel_size**2)
    acc_mask_300 = acc > a_thresh_300
    grid.create_channel_mask("fdir", mask=acc_mask_300, dirmap=DIRMAP, routing="d8")
    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    hand_300 = np.array(grid.hand, dtype=np.float64)
    dtnd_300 = np.array(grid.dtnd, dtype=np.float64)
    drain_300 = np.array(grid.drainage_id, dtype=np.int64)
    stream_300 = np.array(grid.channel_mask, dtype=bool)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"Lc Physical Validation — {dem_data.shape[0]}x{dem_data.shape[1]} @ {pixel_size}m, "
        f"R6-R10 C7-C11",
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
    stream_overlay = np.ma.masked_where(~stream_300, np.ones_like(dem_data))
    ax.imshow(stream_overlay, cmap="Blues", alpha=0.8)
    n_str = int(np.sum(stream_300))
    ax.set_title(f"Stream Network (Lc=300, A={a_thresh_300}, {n_str} px)")

    # HAND
    ax = axes[0, 2]
    hand_plot = np.where(np.isfinite(hand_300), hand_300, np.nan)
    im = ax.imshow(hand_plot, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="HAND (m)")
    ax.set_title("Height Above Nearest Drainage (Lc=300)")

    # DTND
    ax = axes[1, 0]
    dtnd_plot = np.where(np.isfinite(dtnd_300), dtnd_300, np.nan)
    im = ax.imshow(dtnd_plot, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="DTND (m)")
    ax.set_title("Distance To Nearest Drainage (Lc=300)")

    # Catchment map
    ax = axes[1, 1]
    drain_plot = np.where(drain_300 > 0, drain_300, np.nan)
    # Use a shuffled colormap for visual distinction
    n_ids = len(np.unique(drain_300[drain_300 > 0]))
    cmap_catch = plt.cm.tab20
    im = ax.imshow(drain_plot % 20, cmap=cmap_catch, interpolation="nearest")
    ax.set_title(f"Catchments (n={n_ids}, Lc=300)")

    # Empty panel -> summary text
    ax = axes[1, 2]
    ax.axis("off")
    r300 = lc_results.get(300.0, {})
    summary_lines = [
        "Physical Validation Summary (Lc=300m)",
        "",
        f"Check 1: max(DTND)/Lc = {r300.get('check1_max_dtnd_over_lc', 0):.3f}  [{r300.get('check1_verdict', '?')}]",
        f"  max DTND = {r300.get('dtnd', {}).get('max', 0):.1f} m",
        f"  P99 DTND = {r300.get('dtnd', {}).get('p99', 0):.1f} m",
        "",
        f"Check 2: mean(catch)/Lc^2 = {r300.get('check2_mean_area_over_lc2', 0):.3f}  [{r300.get('check2_verdict', '?')}]",
        f"  mean area = {r300.get('catchment_area', {}).get('mean_interior', 0):.0f} m^2",
        f"  Lc^2 = {r300.get('lc_squared', 0):.0f} m^2",
        f"  n interior = {r300.get('n_interior_catchments', 0)}",
        "",
        f"Overall: {overall}",
        "",
        "Swenson calibration (low-relief):",
        "  max(DTND)/Lc = 0.90",
        "  mean(catch)/Lc^2 = 0.94",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    terrain_path = os.path.join(OUTPUT_DIR, "terrain_overview.png")
    fig.savefig(terrain_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {terrain_path}")

    # --- Plot 2: DTND histogram with Lc lines ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DTND Distribution vs Lc", fontsize=13)

    for idx, lc in enumerate(LC_VALUES):
        ax = axes[idx]
        r = lc_results.get(lc, {})
        if "error" in r:
            ax.text(0.5, 0.5, "ERROR", ha="center", va="center", transform=ax.transAxes)
            continue

        # Recompute DTND for this Lc to get the histogram data
        a_thresh_lc = int(0.5 * lc**2 / pixel_size**2)
        acc_mask_lc = acc > a_thresh_lc
        grid.create_channel_mask("fdir", mask=acc_mask_lc, dirmap=DIRMAP, routing="d8")
        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
            routing="d8",
        )
        dtnd_lc = np.array(grid.dtnd, dtype=np.float64)
        stream_lc = np.array(grid.channel_mask, dtype=bool)
        dtnd_ns = dtnd_lc[np.isfinite(dtnd_lc) & ~stream_lc]

        ax.hist(dtnd_ns, bins=100, color="steelblue", alpha=0.7, density=True)
        ax.axvline(lc, color="red", ls="--", lw=2, label=f"Lc = {lc:.0f} m")
        ax.axvline(
            r["dtnd"]["max"],
            color="orange",
            ls="-",
            lw=1.5,
            label=f"max = {r['dtnd']['max']:.0f} m",
        )
        ax.axvline(
            r["dtnd"]["p99"],
            color="green",
            ls=":",
            lw=1.5,
            label=f"P99 = {r['dtnd']['p99']:.0f} m",
        )
        ax.set_xlabel("DTND (m)")
        ax.set_ylabel("Density")
        ax.set_title(f"Lc = {lc:.0f} m  [max/Lc = {r['check1_max_dtnd_over_lc']:.3f}]")
        ax.legend(fontsize=8)

    plt.tight_layout()
    dtnd_path = os.path.join(OUTPUT_DIR, "dtnd_histograms.png")
    fig.savefig(dtnd_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {dtnd_path}")

    # --- Plot 3: Catchment area histogram with Lc^2 lines ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Catchment Area Distribution vs Lc^2", fontsize=13)

    for idx, lc in enumerate(LC_VALUES):
        ax = axes[idx]
        r = lc_results.get(lc, {})
        if "error" in r:
            ax.text(0.5, 0.5, "ERROR", ha="center", va="center", transform=ax.transAxes)
            continue

        # Recompute catchment areas for this Lc
        a_thresh_lc = int(0.5 * lc**2 / pixel_size**2)
        acc_mask_lc = acc > a_thresh_lc
        grid.create_channel_mask("fdir", mask=acc_mask_lc, dirmap=DIRMAP, routing="d8")
        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
            routing="d8",
        )
        drain_lc = np.array(grid.drainage_id, dtype=np.int64)
        hand_lc = np.array(grid.hand, dtype=np.float64)

        valid_drain = drain_lc[np.isfinite(hand_lc) & (drain_lc > 0)]
        unique_ids, counts = np.unique(valid_drain, return_counts=True)
        areas = counts.astype(np.float64) * pixel_size**2

        # Exclude edge catchments
        edge_px = np.concatenate(
            [
                drain_lc[0, :],
                drain_lc[-1, :],
                drain_lc[:, 0],
                drain_lc[:, -1],
            ]
        )
        edge_set = set(np.unique(edge_px[edge_px > 0]))
        interior = np.array([uid not in edge_set for uid in unique_ids])
        interior_areas = areas[interior]

        lc_sq = lc**2
        if len(interior_areas) > 0:
            # Log-scale histogram for area
            log_areas = np.log10(interior_areas[interior_areas > 0])
            ax.hist(log_areas, bins=50, color="steelblue", alpha=0.7)
            ax.axvline(
                np.log10(lc_sq),
                color="red",
                ls="--",
                lw=2,
                label=f"Lc^2 = {lc_sq:.0f} m^2",
            )
            ax.axvline(
                np.log10(np.mean(interior_areas)),
                color="orange",
                ls="-",
                lw=1.5,
                label=f"mean = {np.mean(interior_areas):.0f} m^2",
            )
            ax.axvline(
                np.log10(np.median(interior_areas)),
                color="green",
                ls=":",
                lw=1.5,
                label=f"median = {np.median(interior_areas):.0f} m^2",
            )
            ax.set_xlabel("log10(Catchment Area m^2)")
            ax.set_ylabel("Count")
        ax.set_title(
            f"Lc = {lc:.0f} m  [mean/Lc^2 = {r['check2_mean_area_over_lc2']:.3f}]"
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    area_path = os.path.join(OUTPUT_DIR, "catchment_area_histograms.png")
    fig.savefig(area_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {area_path}")

    # --- Plot 4: Sensitivity comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sensitivity of Physical Checks to Lc", fontsize=13)

    lc_arr = np.array(LC_VALUES)
    check1_arr = np.array(
        [
            lc_results.get(lc, {}).get("check1_max_dtnd_over_lc", np.nan)
            for lc in LC_VALUES
        ]
    )
    check2_arr = np.array(
        [
            lc_results.get(lc, {}).get("check2_mean_area_over_lc2", np.nan)
            for lc in LC_VALUES
        ]
    )

    ax = axes[0]
    ax.plot(lc_arr, check1_arr, "o-", color="steelblue", markersize=8)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Ideal (1.0)")
    ax.axhline(0.90, color="green", ls="--", alpha=0.5, label="Swenson cal. (0.90)")
    ax.axhspan(PASS_LO, PASS_HI, alpha=0.1, color="green", label="PASS range")
    ax.set_xlabel("Lc (m)")
    ax.set_ylabel("max(DTND) / Lc")
    ax.set_title("Check 1: max(DTND) / Lc")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(3.0, np.nanmax(check1_arr) * 1.2))

    ax = axes[1]
    ax.plot(lc_arr, check2_arr, "o-", color="steelblue", markersize=8)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Ideal (1.0)")
    ax.axhline(0.94, color="green", ls="--", alpha=0.5, label="Swenson cal. (0.94)")
    ax.axhspan(PASS_LO, PASS_HI, alpha=0.1, color="green", label="PASS range")
    ax.set_xlabel("Lc (m)")
    ax.set_ylabel("mean(catchment area) / Lc^2")
    ax.set_title("Check 2: mean(catchment area) / Lc^2")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(3.0, np.nanmax(check2_arr) * 1.2))

    plt.tight_layout()
    sens_path = os.path.join(OUTPUT_DIR, "sensitivity_comparison.png")
    fig.savefig(sens_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {sens_path}")

    # -------------------------------------------------------------------
    # Step 6: Save results
    # -------------------------------------------------------------------
    section("Step 6: Save Results")

    timings["total"] = time.time() - t_total

    output = {
        "result": overall,
        "primary_lc": 300.0,
        "lc_values_tested": LC_VALUES,
        "domain": {
            "tiles": f"R{min(TILE_ROWS)}-R{max(TILE_ROWS)}, C{min(TILE_COLS)}-C{max(TILE_COLS)}",
            "shape": list(dem_data.shape),
            "pixel_size_m": pixel_size,
            "crs": str(crs),
            "n_valid_pixels": n_valid,
            "n_nodata_pixels": n_nodata,
            "elev_min": float(np.min(elev_valid)),
            "elev_max": float(np.max(elev_valid)),
            "elev_mean": float(np.mean(elev_valid)),
        },
        "pass_criteria": {
            "pass_range": [PASS_LO, PASS_HI],
            "marginal_range": [MARGINAL_LO, MARGINAL_HI],
            "swenson_calibration": {
                "max_dtnd_over_lc": 0.90,
                "mean_catch_over_lc2": 0.94,
            },
        },
        "results_by_lc": {str(k): v for k, v in lc_results.items()},
        "timings": timings,
    }

    json_path = os.path.join(OUTPUT_DIR, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    progress(f"Saved: {json_path}")

    # Text summary
    summary_path = os.path.join(OUTPUT_DIR, "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Lc Physical Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Result: {overall}\n\n")
        f.write(
            f"Domain: R{min(TILE_ROWS)}-R{max(TILE_ROWS)}, C{min(TILE_COLS)}-C{max(TILE_COLS)}\n"
        )
        f.write(f"Shape: {dem_data.shape}, pixel size: {pixel_size}m\n")
        f.write(
            f"Elevation: [{np.min(elev_valid):.2f}, {np.max(elev_valid):.2f}] m\n\n"
        )

        f.write(
            f"{'Lc (m)':>8}  {'A_thresh':>10}  {'max DTND/Lc':>14}  {'V1':>8}  {'mean area/Lc^2':>16}  {'V2':>8}\n"
        )
        f.write(
            f"{'-' * 8}  {'-' * 10}  {'-' * 14}  {'-' * 8}  {'-' * 16}  {'-' * 8}\n"
        )
        for lc in LC_VALUES:
            r = lc_results.get(lc, {})
            if "error" in r:
                f.write(f"  {lc:8.0f}  ERROR\n")
                continue
            f.write(
                f"{lc:8.0f}  {r['a_thresh']:10d}  "
                f"{r['check1_max_dtnd_over_lc']:14.3f}  {r['check1_verdict']:>8}  "
                f"{r['check2_mean_area_over_lc2']:16.3f}  {r['check2_verdict']:>8}\n"
            )

        f.write(
            "\nSwenson calibration (low-relief): max(DTND)/Lc = 0.90, mean(catch)/Lc^2 = 0.94\n"
        )
        f.write("\nTimings:\n")
        for k, v in timings.items():
            f.write(f"  {k}: {v:.1f}s\n")
    progress(f"Saved: {summary_path}")

    progress(f"\nTotal time: {timings['total']:.1f}s")
    progress(f"RESULT: {overall}")

    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
