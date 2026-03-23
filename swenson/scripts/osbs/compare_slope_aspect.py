#!/usr/bin/env python3
"""
Compare pipeline slope/aspect (pgrid Horn 1981) to NEON reference data (DP3.30025.001).

Both use Horn 1981, but NEON applies a 3x3 pre-filter to the DTM (reduces TIN
interpolation noise) and computes with a 20m tile-edge buffer. Our pipeline runs
Horn 1981 on the raw DTM with no pre-smoothing.

Per-tile comparison across the 90 production tiles (R4C5-R12C14):
  - Slope: Pearson r, MAE, RMSE, median absolute difference
  - Aspect: circular correlation, mean angular difference

Output:
  - JSON with per-tile and aggregate stats
  - Multi-panel diagnostic figure
  - Per-tile correlation bar charts

Usage:
    python scripts/osbs/compare_slope_aspect.py

Environment variables:
    PYSHEDS_FORK: Path to pysheds fork (default: /blue/gerber/cdevaneprugh/pysheds_fork)
    TILE_RANGES: Override tile selection (default: R4C5-R12C14)
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import stats

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid  # noqa: E402

# =============================================================================
# Constants
# =============================================================================

NODATA_VALUE = -9999.0

# Tile grid parameters (must match run_pipeline.py)
TILE_GRID_ORIGIN_EASTING = 394000
TILE_GRID_ORIGIN_NORTHING = 3292000
TILE_SIZE = 1000  # meters per tile

# Data directories
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"
DTM_DIR = DATA_DIR / "neon" / "dtm"
SLOPE_DIR = DATA_DIR / "neon" / "slope"
ASPECT_DIR = DATA_DIR / "neon" / "aspect"

# Output
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / "slope_aspect_comparison"

# Production tiles: R4C5-R12C14 (90 tiles, 0 nodata)
DEFAULT_TILE_RANGE = "R4C5-R12C14"


# =============================================================================
# Tile utilities (duplicated from run_pipeline.py to keep this standalone)
# =============================================================================


def parse_tile_range(range_str: str) -> list[tuple[int, int]]:
    """Parse 'R4C5-R12C14' into list of (row, col) tuples."""
    import re

    single = re.match(r"^R(\d+)C(\d+)$", range_str)
    if single:
        return [(int(single.group(1)), int(single.group(2)))]

    rng = re.match(r"^R(\d+)C(\d+)-R(\d+)C(\d+)$", range_str)
    if rng:
        r1, c1, r2, c2 = map(int, rng.groups())
        return [
            (r, c)
            for r in range(min(r1, r2), max(r1, r2) + 1)
            for c in range(min(c1, c2), max(c1, c2) + 1)
        ]
    raise ValueError(f"Invalid tile range: {range_str}")


def tile_to_coords(row: int, col: int) -> tuple[int, int]:
    """Convert (row, col) to (easting, northing)."""
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
    return easting, northing


def tile_paths(row: int, col: int) -> dict[str, Path]:
    """Return DTM, slope, aspect file paths for a tile."""
    e, n = tile_to_coords(row, col)
    prefix = f"NEON_D03_OSBS_DP3_{e}_{n}"
    return {
        "dtm": DTM_DIR / f"{prefix}_DTM.tif",
        "slope": SLOPE_DIR / f"{prefix}_Slope.tif",
        "aspect": ASPECT_DIR / f"{prefix}_Aspect.tif",
    }


# =============================================================================
# Statistics
# =============================================================================


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum angular difference handling 0/360 wrap. Returns [0, 180]."""
    diff = np.abs(a - b)
    return np.minimum(diff, 360.0 - diff)


def circular_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Circular correlation coefficient (Fisher & Lee, 1983).

    Equivalent to Pearson r applied to sin/cos components.
    Range: [-1, 1], where 1 = perfect agreement.
    """
    a_rad = np.deg2rad(a)
    b_rad = np.deg2rad(b)

    sin_a = np.sin(a_rad - np.mean(a_rad))
    sin_b = np.sin(b_rad - np.mean(b_rad))

    num = np.sum(sin_a * sin_b)
    denom = np.sqrt(np.sum(sin_a**2) * np.sum(sin_b**2))

    if denom == 0:
        return 0.0
    return float(num / denom)


def compute_tile_stats(
    our_slope: np.ndarray,
    our_aspect: np.ndarray,
    neon_slope: np.ndarray,
    neon_aspect: np.ndarray,
) -> dict:
    """Compute comparison statistics for a single tile.

    All inputs should be trimmed (no border) and masked to valid pixels only.
    Slope values in m/m for both.
    """
    n_pixels = len(our_slope)

    # Slope statistics
    slope_r, slope_p = stats.pearsonr(our_slope, neon_slope)
    slope_diff = our_slope - neon_slope
    slope_mae = float(np.mean(np.abs(slope_diff)))
    slope_rmse = float(np.sqrt(np.mean(slope_diff**2)))
    slope_median_abs = float(np.median(np.abs(slope_diff)))
    slope_bias = float(np.mean(slope_diff))

    # Aspect statistics
    asp_circ_r = circular_correlation(our_aspect, neon_aspect)
    ang_diff = angular_difference(our_aspect, neon_aspect)
    asp_mean_diff = float(np.mean(ang_diff))
    asp_median_diff = float(np.median(ang_diff))
    asp_p90_diff = float(np.percentile(ang_diff, 90))

    return {
        "n_pixels": n_pixels,
        "slope_r": float(slope_r),
        "slope_p": float(slope_p),
        "slope_mae": slope_mae,
        "slope_rmse": slope_rmse,
        "slope_median_abs_diff": slope_median_abs,
        "slope_bias": slope_bias,
        "aspect_circular_r": asp_circ_r,
        "aspect_mean_angular_diff": asp_mean_diff,
        "aspect_median_angular_diff": asp_median_diff,
        "aspect_p90_angular_diff": asp_p90_diff,
    }


# =============================================================================
# Per-tile processing
# =============================================================================


def process_tile(row: int, col: int) -> dict | None:
    """Process a single tile. Returns stats dict or None if files missing."""
    paths = tile_paths(row, col)

    for name, p in paths.items():
        if not p.exists():
            print(f"  SKIP R{row}C{col}: missing {name} ({p.name})")
            return None

    # Load DTM and compute slope/aspect via pgrid
    grid = Grid.from_raster(str(paths["dtm"]), "dem")
    grid.slope_aspect("dem", nodata_in_dem=NODATA_VALUE)
    our_slope = grid.slope.copy()  # m/m
    our_aspect = grid.aspect.copy()  # degrees, 0-360

    # Load NEON products
    with rasterio.open(paths["slope"]) as src:
        neon_slope_deg = src.read(1)
    with rasterio.open(paths["aspect"]) as src:
        neon_aspect = src.read(1)

    # Convert NEON slope from degrees to m/m
    neon_slope = np.tan(np.deg2rad(neon_slope_deg))

    # Trim 1px border (pgrid sets border to 0; NEON uses 20m buffer so has
    # valid border data, but the outer ring may differ due to edge effects)
    our_slope = our_slope[1:-1, 1:-1]
    our_aspect = our_aspect[1:-1, 1:-1]
    neon_slope = neon_slope[1:-1, 1:-1]
    neon_aspect = neon_aspect[1:-1, 1:-1]

    # Valid mask: exclude nodata and zero-slope pixels (flat → aspect undefined)
    valid = (
        (our_slope > 0)
        & (neon_slope > 0)
        & (neon_slope_deg[1:-1, 1:-1] != NODATA_VALUE)
        & (neon_aspect != NODATA_VALUE)
    )
    n_valid = int(valid.sum())
    n_total = int(valid.size)

    if n_valid < 100:
        print(f"  SKIP R{row}C{col}: only {n_valid} valid pixels")
        return None

    tile_stats = compute_tile_stats(
        our_slope[valid],
        our_aspect[valid],
        neon_slope[valid],
        neon_aspect[valid],
    )
    tile_stats["tile"] = f"R{row}C{col}"
    tile_stats["n_total"] = n_total
    tile_stats["pct_valid"] = round(100.0 * n_valid / n_total, 2)

    # Store full arrays for R6C10 spatial maps
    if row == 6 and col == 10:
        tile_stats["_slope_diff_map"] = (our_slope - neon_slope).astype(np.float32)
        tile_stats["_aspect_diff_map"] = angular_difference(
            our_aspect, neon_aspect
        ).astype(np.float32)
        tile_stats["_valid_mask"] = valid
        # Store flattened arrays for scatter plots
        tile_stats["_our_slope"] = our_slope[valid].astype(np.float32)
        tile_stats["_neon_slope"] = neon_slope[valid].astype(np.float32)
        tile_stats["_our_aspect"] = our_aspect[valid].astype(np.float32)
        tile_stats["_neon_aspect"] = neon_aspect[valid].astype(np.float32)

    return tile_stats


# =============================================================================
# Plotting
# =============================================================================


def plot_diagnostics(
    tile_results: list[dict], r6c10: dict | None, output_dir: Path
) -> None:
    """Create multi-panel diagnostic figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Collect all scatter data from R6C10 (representative tile)
    if r6c10 and "_our_slope" in r6c10:
        our_s = r6c10["_our_slope"]
        neon_s = r6c10["_neon_slope"]
        our_a = r6c10["_our_aspect"]
        neon_a = r6c10["_neon_aspect"]

        # Subsample for scatter plots
        rng = np.random.default_rng(42)
        n = len(our_s)
        idx = rng.choice(n, size=min(20000, n), replace=False)

        # (0,0) Slope scatter
        ax = axes[0, 0]
        ax.scatter(neon_s[idx], our_s[idx], s=1, alpha=0.3, c="steelblue")
        lims = [0, max(our_s[idx].max(), neon_s[idx].max()) * 1.05]
        ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
        ax.set_xlabel("NEON slope (m/m)")
        ax.set_ylabel("pgrid slope (m/m)")
        ax.set_title(f"Slope: R6C10 (r={r6c10['slope_r']:.4f})")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(loc="upper left")

        # (0,1) Aspect scatter
        ax = axes[0, 1]
        ax.scatter(neon_a[idx], our_a[idx], s=1, alpha=0.3, c="coral")
        ax.plot([0, 360], [0, 360], "k--", lw=0.8, label="1:1")
        ax.set_xlabel("NEON aspect (deg)")
        ax.set_ylabel("pgrid aspect (deg)")
        ax.set_title(f"Aspect: R6C10 (circ r={r6c10['aspect_circular_r']:.4f})")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.legend(loc="upper left")

        # (1,0) Slope difference histogram (all valid pixels, not subsampled)
        ax = axes[1, 0]
        slope_diff = our_s - neon_s
        ax.hist(slope_diff, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvline(
            np.mean(slope_diff),
            color="red",
            lw=1,
            ls="-",
            label=f"mean={np.mean(slope_diff):.5f}",
        )
        ax.set_xlabel("Slope difference (pgrid - NEON, m/m)")
        ax.set_ylabel("Count")
        ax.set_title("Slope difference: R6C10")
        ax.legend()

        # (1,1) Aspect angular difference histogram
        ax = axes[1, 1]
        ang_diff = angular_difference(our_a, neon_a)
        ax.hist(ang_diff, bins=100, color="coral", alpha=0.7, edgecolor="none")
        ax.axvline(
            np.median(ang_diff),
            color="red",
            lw=1,
            ls="-",
            label=f"median={np.median(ang_diff):.1f}°",
        )
        ax.set_xlabel("Angular difference (deg)")
        ax.set_ylabel("Count")
        ax.set_title("Aspect angular difference: R6C10")
        ax.legend()

        # (2,0) Slope difference spatial map
        if "_slope_diff_map" in r6c10:
            ax = axes[2, 0]
            smap = np.ma.array(
                r6c10["_slope_diff_map"],
                mask=~r6c10["_valid_mask"],
            )
            vmax = float(np.percentile(np.abs(smap.compressed()), 95))
            im = ax.imshow(smap, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, label="m/m")
            ax.set_title("Slope difference map: R6C10")

        # (2,1) Aspect difference spatial map
        if "_aspect_diff_map" in r6c10:
            ax = axes[2, 1]
            amap = np.ma.array(
                r6c10["_aspect_diff_map"],
                mask=~r6c10["_valid_mask"],
            )
            im = ax.imshow(amap, cmap="hot_r", vmin=0, vmax=90)
            plt.colorbar(im, ax=ax, label="degrees")
            ax.set_title("Aspect angular difference map: R6C10")
    else:
        for i in range(3):
            for j in range(2):
                axes[i, j].text(
                    0.5,
                    0.5,
                    "R6C10 data not available",
                    transform=axes[i, j].transAxes,
                    ha="center",
                )

    fig.suptitle(
        "Pipeline (pgrid Horn 1981) vs NEON DP3.30025.001",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_panels.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Per-tile correlation bar charts
    tiles = [r["tile"] for r in tile_results]
    slope_rs = [r["slope_r"] for r in tile_results]
    aspect_rs = [r["aspect_circular_r"] for r in tile_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(tiles) * 0.2), 10))

    x = np.arange(len(tiles))
    ax1.bar(x, slope_rs, color="steelblue", alpha=0.8)
    ax1.axhline(
        np.mean(slope_rs), color="red", ls="--", label=f"mean={np.mean(slope_rs):.4f}"
    )
    ax1.set_ylabel("Pearson r")
    ax1.set_title("Slope correlation per tile")
    ax1.set_ylim(min(0.9, min(slope_rs) - 0.01), 1.001)
    ax1.legend()

    ax2.bar(x, aspect_rs, color="coral", alpha=0.8)
    ax2.axhline(
        np.mean(aspect_rs), color="red", ls="--", label=f"mean={np.mean(aspect_rs):.4f}"
    )
    ax2.set_ylabel("Circular r")
    ax2.set_title("Aspect circular correlation per tile")
    ax2.set_ylim(min(0.9, min(aspect_rs) - 0.01), 1.001)
    ax2.legend()

    # Label every 5th tile to avoid clutter
    label_step = max(1, len(tiles) // 20)
    ax2.set_xticks(x[::label_step])
    ax2.set_xticklabels(tiles[::label_step], rotation=90, fontsize=7)
    ax1.set_xticks([])

    plt.tight_layout()
    plt.savefig(output_dir / "per_tile_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    t0 = time.time()
    print("Slope/aspect comparison: pipeline vs NEON")
    print(f"{'=' * 56}")

    # Parse tile selection
    tile_range_str = os.environ.get("TILE_RANGES", DEFAULT_TILE_RANGE)
    tile_coords = parse_tile_range(tile_range_str)
    print(f"Tiles: {tile_range_str} ({len(tile_coords)} tiles)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Process tiles
    tile_results = []
    r6c10_data = None

    for i, (row, col) in enumerate(tile_coords):
        t1 = time.time()
        result = process_tile(row, col)
        dt = time.time() - t1

        if result is None:
            continue

        # Print concise per-tile summary
        print(
            f"  [{i + 1:3d}/{len(tile_coords)}] {result['tile']}: "
            f"slope r={result['slope_r']:.4f}  "
            f"aspect circ_r={result['aspect_circular_r']:.4f}  "
            f"({dt:.1f}s)"
        )

        # Stash R6C10 arrays for plotting, strip from JSON result
        if row == 6 and col == 10:
            r6c10_data = result.copy()

        # Remove numpy arrays before appending to JSON-serializable list
        json_result = {k: v for k, v in result.items() if not k.startswith("_")}
        tile_results.append(json_result)

    if not tile_results:
        print("ERROR: No tiles processed successfully.")
        sys.exit(1)

    # Aggregate statistics
    print(f"\n{'=' * 56}")
    print(f"Aggregate ({len(tile_results)} tiles)")
    print(f"{'=' * 56}")

    agg = {
        "n_tiles": len(tile_results),
        "slope_r_mean": float(np.mean([r["slope_r"] for r in tile_results])),
        "slope_r_min": float(np.min([r["slope_r"] for r in tile_results])),
        "slope_r_max": float(np.max([r["slope_r"] for r in tile_results])),
        "slope_mae_mean": float(np.mean([r["slope_mae"] for r in tile_results])),
        "slope_bias_mean": float(np.mean([r["slope_bias"] for r in tile_results])),
        "aspect_circular_r_mean": float(
            np.mean([r["aspect_circular_r"] for r in tile_results])
        ),
        "aspect_circular_r_min": float(
            np.min([r["aspect_circular_r"] for r in tile_results])
        ),
        "aspect_circular_r_max": float(
            np.max([r["aspect_circular_r"] for r in tile_results])
        ),
        "aspect_mean_angular_diff_mean": float(
            np.mean([r["aspect_mean_angular_diff"] for r in tile_results])
        ),
    }

    print(
        f"  Slope Pearson r:  {agg['slope_r_mean']:.4f} "
        f"[{agg['slope_r_min']:.4f}, {agg['slope_r_max']:.4f}]"
    )
    print(f"  Slope MAE:        {agg['slope_mae_mean']:.6f} m/m")
    print(f"  Slope bias:       {agg['slope_bias_mean']:.6f} m/m (+ = ours steeper)")
    print(
        f"  Aspect circ r:    {agg['aspect_circular_r_mean']:.4f} "
        f"[{agg['aspect_circular_r_min']:.4f}, {agg['aspect_circular_r_max']:.4f}]"
    )
    print(f"  Aspect mean diff: {agg['aspect_mean_angular_diff_mean']:.1f} deg")

    # Save JSON
    output = {"aggregate": agg, "per_tile": tile_results}
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {json_path}")

    # Plots
    print("Generating plots...")
    plot_diagnostics(tile_results, r6c10_data, OUTPUT_DIR)
    print("  diagnostic_panels.png")
    print("  per_tile_correlations.png")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
