#!/usr/bin/env python3
"""
Phase B — Resolution Comparison

Compares hillslope parameters at 1m, 2m, and 4m on two domains:
  - 5x5 tile block (R6-R10, C7-C11, 25M pixels at 1m)
  - Full contiguous region (R4-R12, C5-C14, 90M pixels at 1m)

For each domain x resolution combination: DEM conditioning, flow routing,
HAND/DTND, slope/aspect (pgrid Horn 1981, Phase A UTM-aware), aspect
binning, HAND binning, trapezoidal fit, and all 6 hillslope parameters
for all 16 columns.

Generates comparison JSON, text summary, and plots.

Usage:
    sbatch scripts/phase_b/test_resolution_comparison.sh

Output:
    output/osbs/phase_b/resolution_comparison/
"""

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.stats import expon

# pysheds fork
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pyproj import Proj as PyprojProj  # noqa: E402
from pysheds.pgrid import Grid  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTR = np.pi / 180.0

BASE_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
DTM_DIR = os.path.join(BASE_DIR, "data/neon/dtm")
OUTPUT_DIR = os.path.join(BASE_DIR, "output/osbs/phase_b/resolution_comparison")

# Tile grid
TILE_GRID_ORIGIN_EASTING = 394000
TILE_GRID_ORIGIN_NORTHING = 3292000
TILE_SIZE = 1000  # meters per tile

# Domains to test
DOMAINS = {
    "5x5": {
        "tile_rows": range(6, 11),
        "tile_cols": range(7, 12),
        "expected_shape": (5000, 5000),
        "label": "R6-R10_C7-C11",
    },
    "full": {
        "tile_rows": range(4, 13),
        "tile_cols": range(5, 15),
        "expected_shape": (9000, 10000),
        "label": "R4-R12_C5-C14",
    },
}

# Resolutions to test
RESOLUTIONS = [1, 2, 4]

# Hillslope parameters
LC_M = 300.0
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0
ASPECT_BINS = [(315, 45), (45, 135), (135, 225), (225, 315)]
ASPECT_NAMES = ["North", "East", "South", "West"]
PARAM_NAMES = ["height", "distance", "area", "slope", "aspect", "width"]

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------
# Hillslope computation functions (inlined from merit_regression.py)
# Phase D will extract these into a shared module.
# ---------------------------------------------------------------------------


def get_aspect_mask(aspect, aspect_bin):
    """Create boolean mask for pixels within an aspect bin."""
    lower, upper = aspect_bin
    if lower > upper:
        return (aspect >= lower) | (aspect < upper)
    return (aspect >= lower) & (aspect < upper)


def compute_hand_bins(
    hand, aspect, aspect_bins, bin1_max=2.0, min_aspect_fraction=0.01
):
    """Compute HAND bin boundaries following Swenson's SpecifyHandBounds()."""
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size
    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    if initial_q25 > bin1_max:
        for _asp_idx, (asp_low, asp_high) in enumerate(aspect_bins):
            if asp_low > asp_high:
                asp_mask = (aspect >= asp_low) | (aspect < asp_high)
            else:
                asp_mask = (aspect >= asp_low) & (aspect < asp_high)

            hand_asp_sorted = np.sort(hand[asp_mask])
            if hand_asp_sorted.size > 0:
                bmin = hand_asp_sorted[
                    int(min_aspect_fraction * hand_asp_sorted.size - 1)
                ]
            else:
                bmin = bin1_max

            if bmin > bin1_max:
                bin1_max = bmin

        above_bin1 = hand_sorted[hand_sorted > bin1_max]
        if above_bin1.size > 0:
            n_above = above_bin1.size
            b33 = above_bin1[int(0.33 * n_above - 1)]
            b66 = above_bin1[int(0.66 * n_above - 1)]
            if b33 == b66:
                b66 = 2 * b33 - bin1_max
            bounds = np.array([0, bin1_max, b33, b66, 1e6])
        else:
            bounds = np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])
    else:
        quartiles = [0.25, 0.5, 0.75, 1.0]
        bounds = [0.0]
        for q in quartiles:
            idx = max(0, int(q * n) - 1)
            bounds.append(hand_sorted[idx])
        bounds = np.array(bounds)

    return bounds


def fit_trapezoidal_width(dtnd, area, n_hillslopes, min_dtnd=1.0, n_bins=10):
    """Fit trapezoidal plan form following Swenson Eq. (4)."""
    if np.max(dtnd) <= min_dtnd:
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }

    dtnd_bins = np.linspace(min_dtnd, np.max(dtnd) + 1, n_bins + 1)
    d = np.zeros(n_bins)
    A_cumsum = np.zeros(n_bins)

    for k in range(n_bins):
        mask = dtnd >= dtnd_bins[k]
        d[k] = dtnd_bins[k]
        A_cumsum[k] = np.sum(area[mask])

    A_cumsum /= n_hillslopes

    if min_dtnd > 0:
        d = np.concatenate([[0], d])
        A_cumsum = np.concatenate([[np.sum(area) / n_hillslopes], A_cumsum])

    try:
        weights = A_cumsum
        G = np.column_stack([np.ones_like(d), d, d**2])
        W = np.diag(weights)
        GtWG = G.T @ W @ G
        GtWy = G.T @ W @ A_cumsum
        coeffs = np.linalg.solve(GtWG, GtWy)

        trap_slope = -coeffs[2]
        trap_width = -coeffs[1]
        trap_area = coeffs[0]

        if trap_slope < 0:
            Atri = -(trap_width**2) / (4 * trap_slope)
            if Atri < trap_area:
                trap_width = np.sqrt(-4 * trap_slope * trap_area)

        return {"slope": trap_slope, "width": max(trap_width, 1), "area": trap_area}
    except Exception:
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }


def quadratic(coefs, root=0, eps=1e-6):
    """Solve quadratic equation ax^2 + bx + c = 0."""
    ak, bk, ck = coefs
    discriminant = bk**2 - 4 * ak * ck

    if discriminant < 0:
        if abs(discriminant) < eps:
            ck = bk**2 / (4 * ak) * (1 - eps)
            discriminant = bk**2 - 4 * ak * ck
        else:
            raise RuntimeError(
                f"Cannot solve quadratic: discriminant={discriminant:.2f}"
            )

    roots = [
        (-bk + np.sqrt(discriminant)) / (2 * ak),
        (-bk - np.sqrt(discriminant)) / (2 * ak),
    ]
    return roots[root]


def circular_mean_aspect(aspects):
    """Compute circular mean of aspect values (degrees)."""
    sin_sum = np.mean(np.sin(DTR * aspects))
    cos_sum = np.mean(np.cos(DTR * aspects))
    mean_aspect = np.arctan2(sin_sum, cos_sum) / DTR
    if mean_aspect < 0:
        mean_aspect += 360
    return mean_aspect


def catchment_mean_aspect(drainage_id, aspect, hillslope, chunksize=500):
    """Replace per-pixel aspect with catchment-side circular mean."""
    valid_drain = np.isfinite(drainage_id) & (drainage_id > 0)
    uid = np.unique(drainage_id[valid_drain])
    hillslope_types = np.unique(hillslope[hillslope > 0]).astype(int)

    out = np.zeros(aspect.shape)

    if uid.size == 0:
        return out
    valid_aspect = np.isfinite(aspect.flat)

    nchunks = int(max(1, int(uid.size // chunksize)))
    cs = int(min(chunksize, uid.size - 1))

    for n in range(nchunks):
        n1, n2 = int(n * cs), int((n + 1) * cs)
        if n == nchunks - 1:
            n2 = uid.size - 1
        if n1 == n2:
            cind = np.where(valid_aspect & (drainage_id.flat == uid[n1]))[0]
        else:
            cind = np.where(
                valid_aspect
                & (drainage_id.flat >= uid[n1])
                & (drainage_id.flat < uid[n2])
            )[0]

        for did in uid[n1 : n2 + 1]:
            dind = cind[drainage_id.flat[cind] == did]
            for ht in hillslope_types[: hillslope_types.size - 1]:
                sel = (hillslope.flat[dind] == 4) | (hillslope.flat[dind] == ht)
                ind = dind[sel]
                if ind.size > 0:
                    mean_asp = (
                        np.arctan2(
                            np.mean(np.sin(DTR * aspect.flat[ind])),
                            np.mean(np.cos(DTR * aspect.flat[ind])),
                        )
                        / DTR
                    )
                    if mean_asp < 0:
                        mean_asp += 360.0
                    out.flat[ind] = mean_asp

    return out


def tail_index(dtnd, hand, npdf_bins=5000, hval=0.05):
    """Return indices of pixels with DTND below the tail threshold."""
    positive_hand = hand > 0
    if np.sum(positive_hand) == 0:
        return np.arange(dtnd.size)

    dtnd_pos = dtnd[positive_hand]
    std_dtnd = np.std(dtnd_pos)
    if std_dtnd == 0:
        return np.arange(dtnd.size)

    fit_loc, fit_beta = expon.fit(dtnd_pos / std_dtnd)
    rv = expon(loc=fit_loc, scale=fit_beta)

    pbins = np.linspace(0, np.max(dtnd), npdf_bins)
    rvpdf = rv.pdf(pbins / std_dtnd)
    r1 = np.argmin(np.abs(rvpdf - hval * np.max(rvpdf)))
    return np.where(dtnd < pbins[r1])[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_peak_rss_gb():
    """Read peak RSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1e6
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return -1.0


def progress(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def tile_filepath(row, col):
    """Convert (row, col) to NEON DTM tile filepath."""
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
    return os.path.join(DTM_DIR, f"NEON_D03_OSBS_DP3_{easting}_{northing}_DTM.tif")


def block_average(dem, factor):
    """Subsample DEM by block averaging (not decimation)."""
    nrows, ncols = dem.shape
    # Trim to multiple of factor
    nrows_trim = (nrows // factor) * factor
    ncols_trim = (ncols // factor) * factor
    trimmed = dem[:nrows_trim, :ncols_trim]
    # Reshape and mean
    return trimmed.reshape(
        nrows_trim // factor, factor, ncols_trim // factor, factor
    ).mean(axis=(1, 3))


def compute_hillslope_params_utm(
    dem,
    transform,
    crs,
    pixel_size,
    resolution_label,
):
    """
    Full pipeline: conditioning -> flow routing -> HAND/DTND -> slope/aspect
    -> binning -> 6 hillslope parameters for 16 columns.

    Returns dict with elements list, timing, and diagnostic info.
    """
    timings = {}
    t_total = time.time()

    a_thresh = int(0.5 * LC_M**2 / pixel_size**2)
    progress(
        f"  [{resolution_label}] A_thresh = {a_thresh} cells (pixel_size={pixel_size}m)"
    )

    # --- pysheds Grid ---
    t0 = time.time()
    grid = Grid()
    grid.add_gridded_data(
        dem,
        data_name="dem",
        affine=transform,
        crs=PyprojProj(crs),
        nodata=np.nan,
    )
    timings["create_grid"] = round(time.time() - t0, 1)

    # --- DEM conditioning ---
    t0 = time.time()
    grid.fill_pits("dem", out_name="pit_filled")
    timings["fill_pits"] = round(time.time() - t0, 1)

    t0 = time.time()
    grid.fill_depressions("pit_filled", out_name="flooded")
    timings["fill_depressions"] = round(time.time() - t0, 1)

    t0 = time.time()
    try:
        grid.resolve_flats("flooded", out_name="inflated")
    except ValueError:
        progress(
            f"  [{resolution_label}] WARNING: resolve_flats failed, fallback to raw DEM"
        )
        grid.add_gridded_data(
            np.array(grid.dem),
            data_name="inflated",
            affine=grid.affine,
            crs=grid.crs,
            nodata=grid.nodata,
        )
    timings["resolve_flats"] = round(time.time() - t0, 1)

    # --- Flow routing ---
    t0 = time.time()
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")
    timings["flowdir"] = round(time.time() - t0, 1)

    t0 = time.time()
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")
    timings["accumulation"] = round(time.time() - t0, 1)

    acc = grid.acc
    max_acc = float(np.nanmax(acc))
    progress(f"  [{resolution_label}] Max accumulation: {max_acc:.0f}")

    # --- Stream network + HAND/DTND ---
    acc_mask = acc > a_thresh
    n_stream = int(np.sum(acc_mask))
    progress(f"  [{resolution_label}] Stream pixels: {n_stream}")

    if n_stream == 0:
        progress(f"  [{resolution_label}] ERROR: No stream pixels")
        return None

    t0 = time.time()
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")
    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    timings["hand_dtnd"] = round(time.time() - t0, 1)

    hand = np.array(grid.hand, dtype=np.float64)
    dtnd = np.array(grid.dtnd, dtype=np.float64)
    drainage_id = np.array(grid.drainage_id, dtype=np.float64)

    # --- Hillslope classification ---
    t0 = time.time()
    grid.compute_hillslope(
        "fdir", "channel_mask", "bank_mask", dirmap=DIRMAP, routing="d8"
    )
    hillslope = np.array(grid.hillslope)
    timings["compute_hillslope"] = round(time.time() - t0, 1)

    # --- Slope/aspect (pgrid Horn 1981, Phase A UTM-aware) ---
    t0 = time.time()
    grid.slope_aspect("dem")
    slope = np.array(grid.slope)
    aspect = np.array(grid.aspect)
    timings["slope_aspect"] = round(time.time() - t0, 1)

    # --- Catchment-level aspect averaging ---
    t0 = time.time()
    aspect = catchment_mean_aspect(drainage_id, aspect, hillslope)
    timings["catchment_aspect"] = round(time.time() - t0, 1)

    # --- Flatten and filter ---
    hand_flat = hand.flatten()
    dtnd_flat = dtnd.flatten()
    slope_flat = slope.flatten()
    aspect_flat = aspect.flatten()
    # UTM: uniform pixel area
    area_flat = np.full(hand_flat.shape, pixel_size**2)
    drainage_id_flat = drainage_id.flatten()

    # Clip DTND minimum
    dtnd_flat[dtnd_flat < 1.0] = 1.0

    # Valid mask
    valid = np.isfinite(hand_flat)

    # DTND tail removal
    tail_ind = tail_index(dtnd_flat[valid], hand_flat[valid])
    valid_indices = np.where(valid)[0]
    keep = np.zeros(valid.shape, dtype=bool)
    keep[valid_indices[tail_ind]] = True
    n_tail_removed = int(np.sum(valid) - np.sum(keep))
    progress(f"  [{resolution_label}] DTND tail removal: {n_tail_removed} pixels")
    valid = keep

    hand_flat = hand_flat[valid]
    dtnd_flat = dtnd_flat[valid]
    slope_flat = slope_flat[valid]
    aspect_flat = aspect_flat[valid]
    area_flat = area_flat[valid]
    drainage_id_flat = drainage_id_flat[valid]

    # --- HAND bins ---
    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )
    progress(f"  [{resolution_label}] HAND bounds: {hand_bounds}")

    # --- Compute 16 elements ---
    t0 = time.time()
    elements = []
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        asp_mask = get_aspect_mask(aspect_flat, asp_bin)
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            for _ in range(N_HAND_BINS):
                elements.append({p: 0.0 for p in PARAM_NAMES})
            continue

        n_hillslopes = max(len(np.unique(drainage_id_flat[asp_indices])), 1)

        # Trapezoidal fit
        trap = fit_trapezoidal_width(
            dtnd_flat[asp_indices],
            area_flat[asp_indices],
            n_hillslopes,
            min_dtnd=pixel_size,
        )
        trap_slope = trap["slope"]
        trap_width = trap["width"]
        trap_area = trap["area"]

        # First pass: raw areas per HAND bin
        bin_raw_areas = []
        bin_data = []
        for h_idx in range(N_HAND_BINS):
            h_lower = hand_bounds[h_idx]
            h_upper = hand_bounds[h_idx + 1]
            hand_mask = (hand_flat >= h_lower) & (hand_flat < h_upper)
            bin_mask = asp_mask & hand_mask
            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) == 0:
                bin_raw_areas.append(0)
                bin_data.append(
                    {"indices": None, "h_lower": h_lower, "h_upper": h_upper}
                )
            else:
                bin_raw_areas.append(float(np.sum(area_flat[bin_indices])))
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

        # Second pass: compute parameters
        for h_idx in range(N_HAND_BINS):
            data = bin_data[h_idx]
            bin_indices = data["indices"]

            if bin_indices is None:
                elements.append(
                    {
                        "height": float((data["h_lower"] + data["h_upper"]) / 2),
                        "distance": 0.0,
                        "area": 0.0,
                        "slope": 0.0,
                        "aspect": float((asp_bin[0] + asp_bin[1]) / 2 % 360),
                        "width": 0.0,
                    }
                )
                continue

            if np.mean(hand_flat[bin_indices]) <= 0:
                elements.append({p: 0.0 for p in PARAM_NAMES})
                continue

            mean_hand = float(np.mean(hand_flat[bin_indices]))
            mean_slope = float(np.nanmean(slope_flat[bin_indices]))
            mean_aspect = circular_mean_aspect(aspect_flat[bin_indices])
            dtnd_sorted = np.sort(dtnd_flat[bin_indices])
            median_dtnd = float(dtnd_sorted[len(dtnd_sorted) // 2])
            fitted_area = fitted_areas[h_idx]

            # Width: solve quadratic at lower edge of bin
            da_width = sum(fitted_areas[:h_idx]) if h_idx > 0 else 0
            if trap_slope != 0:
                try:
                    le = quadratic([trap_slope, trap_width, -da_width])
                    width = trap_width + 2 * trap_slope * le
                except RuntimeError:
                    width = trap_width * (1 - 0.15 * h_idx)
            else:
                width = trap_width

            width = max(float(width), 1.0)

            # Distance: trapezoid-derived midpoint distance
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
                    "height": mean_hand,
                    "distance": distance,
                    "area": fitted_area,
                    "slope": mean_slope,
                    "aspect": mean_aspect,
                    "width": width,
                }
            )

    timings["compute_params"] = round(time.time() - t0, 1)
    timings["total"] = round(time.time() - t_total, 1)

    # Count catchments
    valid_drain = drainage_id[np.isfinite(hand) & (drainage_id > 0)]
    unique_ids, counts = np.unique(valid_drain, return_counts=True)

    return {
        "resolution_m": pixel_size,
        "dem_shape": list(dem.shape),
        "a_thresh": a_thresh,
        "n_stream_pixels": n_stream,
        "stream_fraction": n_stream / dem.size,
        "n_catchments": len(unique_ids),
        "mean_catchment_area_m2": float(np.mean(counts * pixel_size**2))
        if len(counts) > 0
        else 0,
        "hand_bounds": hand_bounds.tolist(),
        "n_valid_pixels": int(np.sum(valid)),
        "n_tail_removed": n_tail_removed,
        "elements": elements,
        "timings": timings,
        "peak_rss_gb": get_peak_rss_gb(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_comparison_plots(results, output_dir, domain_name=""):
    """Generate 6-panel parameter comparison plot."""
    resolutions = sorted(results.keys())
    n_cols = 16
    domain_label = DOMAINS.get(domain_name, {}).get("label", domain_name)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Phase B: Hillslope Parameter Comparison Across Resolutions\n"
        f"{domain_name} ({domain_label})",
        fontsize=13,
    )

    col_labels = []
    for asp_name in ASPECT_NAMES:
        for h_idx in range(N_HAND_BINS):
            col_labels.append(f"{asp_name[0]}{h_idx + 1}")

    for p_idx, param in enumerate(PARAM_NAMES):
        ax = axes[p_idx // 3, p_idx % 3]

        for res in resolutions:
            r = results[res]
            if r is None:
                continue
            values = [r["elements"][c][param] for c in range(n_cols)]
            ax.plot(
                range(n_cols),
                values,
                "o-",
                markersize=4,
                label=f"{res}m",
            )

        ax.set_title(param.capitalize())
        ax.set_xlabel("Column")
        ax.set_ylabel(param)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"parameter_comparison_{domain_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {path}")

    # --- Timing comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    step_names = [
        "fill_pits",
        "fill_depressions",
        "resolve_flats",
        "flowdir",
        "accumulation",
        "hand_dtnd",
        "compute_hillslope",
        "slope_aspect",
        "catchment_aspect",
        "compute_params",
    ]
    x = np.arange(len(step_names))
    width = 0.25

    for i, res in enumerate(resolutions):
        r = results[res]
        if r is None:
            continue
        times = [r["timings"].get(s, 0) for s in step_names]
        ax.bar(x + i * width, times, width, label=f"{res}m")

    ax.set_xlabel("Pipeline Step")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Pipeline Step Timing by Resolution — {domain_name}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(step_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"timing_comparison_{domain_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    progress(f"Saved: {path}")


def write_summary(results, output_dir, domain_name=""):
    """Write human-readable text summary."""
    resolutions = sorted(results.keys())
    domain_label = DOMAINS.get(domain_name, {}).get("label", domain_name)
    lines = []
    lines.append(f"Phase B: Resolution Comparison Summary — {domain_name}")
    lines.append("=" * 60)
    lines.append(f"Domain: {domain_name} ({domain_label})")
    lines.append(f"Lc: {LC_M} m")
    lines.append("")

    # Overview table
    lines.append(
        f"{'Res (m)':>8}  {'Shape':>14}  {'A_thresh':>10}  {'Streams':>10}  "
        f"{'Catchments':>12}  {'Time (s)':>10}  {'RSS (GB)':>10}"
    )
    lines.append("-" * 90)
    for res in resolutions:
        r = results[res]
        if r is None:
            lines.append(f"{res:>8}  {'FAILED':>14}")
            continue
        shape_str = f"{r['dem_shape'][0]}x{r['dem_shape'][1]}"
        lines.append(
            f"{res:>8}  {shape_str:>14}  {r['a_thresh']:>10}  {r['n_stream_pixels']:>10}  "
            f"{r['n_catchments']:>12}  {r['timings']['total']:>10.1f}  {r['peak_rss_gb']:>10.1f}"
        )
    lines.append("")

    # Per-column parameter comparison
    col_labels = []
    for asp_name in ASPECT_NAMES:
        for h_idx in range(N_HAND_BINS):
            col_labels.append(f"{asp_name[0]}{h_idx + 1}")

    for param in PARAM_NAMES:
        lines.append(f"\n--- {param.upper()} ---")
        header = f"{'Col':>5}"
        for res in resolutions:
            header += f"  {res}m:>12"
        # Simple format
        lines.append(
            f"{'Col':>5}  {'1m':>12}  {'2m':>12}  {'4m':>12}  {'1m-4m diff':>12}"
        )
        lines.append("-" * 60)

        for c in range(16):
            label = col_labels[c]
            vals = {}
            for res in resolutions:
                r = results[res]
                if r is not None:
                    vals[res] = r["elements"][c][param]
                else:
                    vals[res] = float("nan")

            diff = vals.get(1, 0) - vals.get(4, 0)
            lines.append(
                f"{label:>5}  {vals.get(1, 0):>12.4f}  {vals.get(2, 0):>12.4f}  "
                f"{vals.get(4, 0):>12.4f}  {diff:>12.4f}"
            )

    # Correlation summary
    lines.append("\n\n--- PARAMETER CORRELATIONS ACROSS RESOLUTIONS ---")
    for param in PARAM_NAMES:
        vec_1m = []
        vec_2m = []
        vec_4m = []
        for c in range(16):
            for res, vec in [(1, vec_1m), (2, vec_2m), (4, vec_4m)]:
                r = results.get(res)
                if r is not None:
                    vec.append(r["elements"][c][param])
                else:
                    vec.append(0)

        vec_1m = np.array(vec_1m)
        vec_2m = np.array(vec_2m)
        vec_4m = np.array(vec_4m)

        # Only compute correlation if variance exists
        corr_1_2 = "N/A"
        corr_1_4 = "N/A"
        if np.std(vec_1m) > 0 and np.std(vec_2m) > 0:
            corr_1_2 = f"{np.corrcoef(vec_1m, vec_2m)[0, 1]:.4f}"
        if np.std(vec_1m) > 0 and np.std(vec_4m) > 0:
            corr_1_4 = f"{np.corrcoef(vec_1m, vec_4m)[0, 1]:.4f}"

        lines.append(
            f"  {param:>10}: r(1m,2m) = {corr_1_2:>8}  r(1m,4m) = {corr_1_4:>8}"
        )

    path = os.path.join(output_dir, f"summary_{domain_name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    progress(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_domain_tiles(domain_cfg):
    """Load and merge tiles for a domain, return (dem, transform, crs)."""
    tile_rows = domain_cfg["tile_rows"]
    tile_cols = domain_cfg["tile_cols"]
    expected = domain_cfg["expected_shape"]

    tile_paths = []
    for r in tile_rows:
        for c in tile_cols:
            fp = tile_filepath(r, c)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing tile: {os.path.basename(fp)}")
            tile_paths.append(fp)
    progress(f"  Found all {len(tile_paths)} tiles")

    t0 = time.time()
    datasets = [rasterio.open(p) for p in tile_paths]
    dem_1m, transform_1m = merge(datasets)[:2]
    for ds in datasets:
        ds.close()

    dem_1m = dem_1m.squeeze().astype(np.float64)
    dem_1m[dem_1m == -9999.0] = np.nan

    with rasterio.open(tile_paths[0]) as src:
        crs = src.crs

    progress(f"  Merged shape: {dem_1m.shape} ({time.time() - t0:.1f}s)")
    progress(f"  CRS: {crs}")

    if dem_1m.shape != expected:
        progress(f"  WARNING: expected {expected}, got {dem_1m.shape}")

    n_nodata = int(np.sum(~np.isfinite(dem_1m)))
    if n_nodata > 0:
        progress(f"  WARNING: {n_nodata} nodata pixels")

    return dem_1m, transform_1m, crs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total_start = time.time()

    print("=" * 60)
    print("Phase B — Resolution Comparison")
    print("=" * 60)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Domains: {list(DOMAINS.keys())}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Lc: {LC_M} m")
    sys.stdout.flush()

    # Results keyed by (domain_name, resolution)
    all_results: dict[str, dict] = {}

    for domain_name, domain_cfg in DOMAINS.items():
        progress(f"\n{'#' * 60}")
        progress(f"Domain: {domain_name} ({domain_cfg['label']})")
        progress(f"{'#' * 60}")

        # Load tiles for this domain
        try:
            dem_1m, transform_1m, crs = load_domain_tiles(domain_cfg)
        except FileNotFoundError as exc:
            progress(f"FATAL: {exc}")
            return 1

        domain_results = {}

        for res in RESOLUTIONS:
            run_label = f"{domain_name}/{res}m"
            progress(f"\n{'=' * 40}")
            progress(f"{run_label}")
            progress(f"{'=' * 40}")

            if res == 1:
                dem_res = dem_1m.copy()
                pixel_size = 1.0
                t = transform_1m
            else:
                progress(f"  Block averaging {res}x{res}...")
                dem_res = block_average(dem_1m, res)
                pixel_size = float(res)
                t = rasterio.Affine(
                    transform_1m.a * res,
                    transform_1m.b,
                    transform_1m.c,
                    transform_1m.d,
                    transform_1m.e * res,
                    transform_1m.f,
                )

            progress(f"  DEM shape: {dem_res.shape}, pixel size: {pixel_size}m")
            progress(f"  Total pixels: {dem_res.size:,}")

            result = compute_hillslope_params_utm(
                dem_res, t, crs, pixel_size, run_label
            )
            domain_results[res] = result

            if result is not None:
                progress(
                    f"  [{run_label}] Total time: {result['timings']['total']:.1f}s"
                )
                progress(f"  [{run_label}] Peak RSS: {result['peak_rss_gb']:.1f} GB")
            else:
                progress(f"  [{run_label}] FAILED")

        all_results[domain_name] = domain_results

        # Free 1m DEM before loading next domain
        del dem_1m

    # ------------------------------------------------------------------
    # Generate outputs
    # ------------------------------------------------------------------
    progress("\nGenerating outputs...")

    # JSON results
    json_out = {}
    for domain_name, domain_results in all_results.items():
        json_out[domain_name] = {}
        for res, r in domain_results.items():
            if r is not None:
                json_out[domain_name][str(res)] = r
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    progress(f"Saved: {json_path}")

    # Text summary and plots per domain
    for domain_name, domain_results in all_results.items():
        write_summary(domain_results, OUTPUT_DIR, domain_name)
        make_comparison_plots(domain_results, OUTPUT_DIR, domain_name)

    total_time = time.time() - t_total_start
    progress(f"\nTotal elapsed: {total_time:.1f}s ({total_time / 60:.1f} min)")
    progress("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
