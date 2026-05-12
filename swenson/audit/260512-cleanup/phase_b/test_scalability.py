#!/usr/bin/env python3
"""
Phase B — Scalability Test

Tests whether pysheds resolve_flats() can complete on the full contiguous
interior region (R4-R12, C5-C14, ~90M pixels at 1m) at various memory
allocations. Runs DEM conditioning + flow routing + HAND and logs timing
and peak RSS at each step.

Does NOT compute hillslope parameters, FFT, slope/aspect, or plots.
That is Script 2 (test_resolution_comparison.py).

Usage:
    sbatch --mem=64gb scripts/phase_b/test_scalability.sh
    sbatch --mem=128gb scripts/phase_b/test_scalability.sh
    sbatch --mem=256gb scripts/phase_b/test_scalability.sh

Output:
    output/osbs/phase_b/scalability_<mem>gb.json
"""

import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.merge import merge

# pysheds fork
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pyproj import Proj as PyprojProj  # noqa: E402
from pysheds.pgrid import Grid  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
OUTPUT_DIR = os.path.join(BASE_DIR, "output/osbs/phase_b")

# Tile grid (from run_pipeline.py)
TILE_GRID_ORIGIN_EASTING = 394000
TILE_GRID_ORIGIN_NORTHING = 3292000
TILE_SIZE = 1000  # meters per tile
DTM_DIR = os.path.join(BASE_DIR, "data/neon/dtm")

# Contiguous interior region: R4-R12, C5-C14 (90 tiles, 9x10 km, 0 nodata)
TILE_ROWS = range(4, 13)  # rows 4-12 inclusive (9 rows)
TILE_COLS = range(5, 15)  # cols 5-14 inclusive (10 cols)
EXPECTED_SHAPE = (9000, 10000)

# Lc = 300m, A_thresh = 0.5 * Lc^2 / pixel_size^2 = 0.5 * 300^2 / 1^2
LC_M = 300.0
ACCUM_THRESHOLD = 45000  # cells (at 1m, 1 cell = 1 m^2)

# D8 direction map (pysheds convention)
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_peak_rss_gb() -> float:
    """Read peak RSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    # VmHWM is peak resident set size in kB
                    return float(line.split()[1]) / 1e6  # kB -> GB
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return -1.0


def get_current_rss_gb() -> float:
    """Read current RSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1e6
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return -1.0


def progress(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    rss = get_current_rss_gb()
    print(f"[{ts}] [RSS {rss:.1f} GB] {msg}")
    sys.stdout.flush()


def tile_filepath(row: int, col: int) -> str:
    """Convert (row, col) to tile filepath."""
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
    return os.path.join(DTM_DIR, f"NEON_D03_OSBS_DP3_{easting}_{northing}_DTM.tif")


def timed_step(name: str, func, steps: dict) -> object:
    """Run func(), record timing and peak RSS in steps dict, return result."""
    progress(f"Starting: {name}")
    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        peak = get_peak_rss_gb()
        steps[name] = {"seconds": round(elapsed, 1), "peak_rss_gb": round(peak, 2)}
        progress(f"  Done: {name} ({elapsed:.1f}s, peak RSS {peak:.1f} GB)")
        return result
    except MemoryError:
        elapsed = time.time() - t0
        peak = get_peak_rss_gb()
        steps[name] = {
            "seconds": round(elapsed, 1),
            "peak_rss_gb": round(peak, 2),
            "error": "MemoryError",
        }
        progress(f"  OOM: {name} after {elapsed:.1f}s (peak RSS {peak:.1f} GB)")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine memory label from SLURM or command line
    mem_gb = os.environ.get("SLURM_MEM_PER_NODE", "unknown")
    if mem_gb != "unknown":
        # SLURM reports in MB
        try:
            mem_gb = int(int(mem_gb) / 1024)
        except ValueError:
            pass
    mem_label = str(mem_gb)

    print("=" * 60)
    print(f"Phase B Scalability Test — {mem_label} GB")
    print("=" * 60)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Tiles: R{min(TILE_ROWS)}-R{max(TILE_ROWS)}, C{min(TILE_COLS)}-C{max(TILE_COLS)}"
    )
    print(f"Expected shape: {EXPECTED_SHAPE}")
    print(f"Lc: {LC_M} m, A_thresh: {ACCUM_THRESHOLD} cells")
    print(f"pysheds fork: {pysheds_fork}")
    sys.stdout.flush()

    steps: dict = {}
    status = "PASS"
    t_total = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load DEM (merge tiles on the fly)
    # ------------------------------------------------------------------
    def load_dem():
        tile_paths = []
        for r in TILE_ROWS:
            for c in TILE_COLS:
                fp = tile_filepath(r, c)
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"Missing tile: {os.path.basename(fp)}")
                tile_paths.append(fp)
        progress(f"Found all {len(tile_paths)} tiles")

        datasets = [rasterio.open(p) for p in tile_paths]
        dem_arr, xform = merge(datasets)[:2]
        for ds in datasets:
            ds.close()

        # merge returns (bands, rows, cols) — squeeze to 2D
        dem_arr = dem_arr.squeeze()

        # Read CRS and pixel size from first tile
        with rasterio.open(tile_paths[0]) as src:
            tile_crs = src.crs
            px_size = src.transform.a

        return dem_arr, xform, tile_crs, px_size

    try:
        dem, transform, crs, pixel_size = timed_step("load_dem", load_dem, steps)
    except FileNotFoundError as exc:
        progress(f"FATAL: {exc}")
        status = "ERROR"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    progress(f"DEM shape: {dem.shape}, dtype: {dem.dtype}")
    progress(f"CRS: {crs}, pixel size: {pixel_size} m")

    if dem.shape != EXPECTED_SHAPE:
        progress(f"WARNING: shape mismatch, expected {EXPECTED_SHAPE}")

    # Handle nodata
    dem = dem.astype(np.float64)
    n_nodata = int(np.sum(dem == -9999.0)) + int(np.sum(~np.isfinite(dem)))
    dem[dem == -9999.0] = np.nan
    progress(f"Nodata pixels: {n_nodata} ({100 * n_nodata / dem.size:.2f}%)")

    elev_valid = dem[np.isfinite(dem)]
    progress(
        f"Elevation: [{np.min(elev_valid):.2f}, {np.max(elev_valid):.2f}] m, "
        f"mean {np.mean(elev_valid):.2f} m"
    )

    # ------------------------------------------------------------------
    # Step 2: Create pysheds Grid
    # ------------------------------------------------------------------
    def create_grid():
        g = Grid()
        g.add_gridded_data(
            dem,
            data_name="dem",
            affine=transform,
            crs=PyprojProj(crs),
            nodata=np.nan,
        )
        return g

    try:
        grid = timed_step("create_grid", create_grid, steps)
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    is_geographic = grid._crs_is_geographic()
    progress(f"CRS detection: geographic={is_geographic}")
    if is_geographic:
        progress("ERROR: Expected projected CRS (UTM), got geographic")
        status = "ERROR"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    # ------------------------------------------------------------------
    # Step 3: DEM conditioning chain
    # ------------------------------------------------------------------
    try:
        timed_step(
            "fill_pits", lambda: grid.fill_pits("dem", out_name="pit_filled"), steps
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    try:
        timed_step(
            "fill_depressions",
            lambda: grid.fill_depressions("pit_filled", out_name="flooded"),
            steps,
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    try:
        timed_step(
            "resolve_flats",
            lambda: grid.resolve_flats("flooded", out_name="inflated"),
            steps,
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    # ------------------------------------------------------------------
    # Step 4: Flow routing
    # ------------------------------------------------------------------
    try:
        timed_step(
            "flowdir",
            lambda: grid.flowdir(
                "inflated", out_name="fdir", dirmap=DIRMAP, routing="d8"
            ),
            steps,
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    try:
        timed_step(
            "accumulation",
            lambda: grid.accumulation(
                "fdir", out_name="acc", dirmap=DIRMAP, routing="d8"
            ),
            steps,
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    # ------------------------------------------------------------------
    # Step 5: Stream network + HAND/DTND
    # ------------------------------------------------------------------
    acc = grid.acc
    max_acc = float(np.nanmax(acc))
    progress(f"Max accumulation: {max_acc:.0f} cells")

    acc_mask = acc > ACCUM_THRESHOLD
    n_stream = int(np.sum(acc_mask))
    progress(f"Stream pixels: {n_stream} ({100 * n_stream / dem.size:.3f}%)")

    if n_stream == 0:
        progress("ERROR: No stream pixels — A_thresh too high")
        status = "ERROR"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    try:
        timed_step(
            "create_channel_mask",
            lambda: grid.create_channel_mask(
                "fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8"
            ),
            steps,
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    try:
        timed_step(
            "compute_hand",
            lambda: grid.compute_hand(
                "fdir",
                "dem",
                grid.channel_mask,
                grid.channel_id,
                dirmap=DIRMAP,
                routing="d8",
            ),
            steps,
        )
    except MemoryError:
        status = "OOM"
        _write_result(mem_label, steps, status, time.time() - t_total)
        return 1

    # ------------------------------------------------------------------
    # Step 6: Summary stats
    # ------------------------------------------------------------------
    hand = np.array(grid.hand, dtype=np.float64)
    dtnd = np.array(grid.dtnd, dtype=np.float64)

    nonstream = np.isfinite(hand) & ~np.array(grid.channel_mask, dtype=bool)
    hand_ns = hand[nonstream]
    dtnd_ns = dtnd[nonstream]

    progress(f"HAND: mean={np.mean(hand_ns):.2f}, max={np.max(hand_ns):.2f} m")
    progress(f"DTND: mean={np.mean(dtnd_ns):.2f}, max={np.max(dtnd_ns):.2f} m")

    total_seconds = time.time() - t_total
    peak_rss = get_peak_rss_gb()

    progress(f"\nTotal time: {total_seconds:.1f}s ({total_seconds / 60:.1f} min)")
    progress(f"Peak RSS: {peak_rss:.1f} GB")
    progress(f"Status: {status}")

    _write_result(
        mem_label,
        steps,
        status,
        total_seconds,
        extra={
            "max_accumulation": max_acc,
            "n_stream_pixels": n_stream,
            "stream_fraction": n_stream / dem.size,
            "hand_mean": float(np.mean(hand_ns)),
            "hand_max": float(np.max(hand_ns)),
            "hand_p95": float(np.percentile(hand_ns, 95)),
            "dtnd_mean": float(np.mean(dtnd_ns)),
            "dtnd_max": float(np.max(dtnd_ns)),
            "dtnd_p95": float(np.percentile(dtnd_ns, 95)),
        },
    )

    return 0


def _write_result(
    mem_label: str,
    steps: dict,
    status: str,
    total_seconds: float,
    extra: dict | None = None,
):
    """Write JSON results file."""
    result = {
        "domain": "R4-R12_C5-C14",
        "pixels": list(EXPECTED_SHAPE),
        "total_pixels": EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1],
        "resolution_m": 1,
        "mem_requested_gb": mem_label,
        "lc_m": LC_M,
        "accum_threshold": ACCUM_THRESHOLD,
        "steps": steps,
        "total_seconds": round(total_seconds, 1),
        "peak_rss_gb": round(get_peak_rss_gb(), 2),
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        result.update(extra)

    outpath = os.path.join(OUTPUT_DIR, f"scalability_{mem_label}gb.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    progress(f"Results written to: {outpath}")


if __name__ == "__main__":
    sys.exit(main())
