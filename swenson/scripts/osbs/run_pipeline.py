#!/usr/bin/env python3
"""
Full Mosaic Pipeline for OSBS 1m LIDAR Data

Run the Swenson hillslope methodology on the full 19x17 km OSBS mosaic.

This script adapts run_smoke_test_pipeline.py for the full mosaic with:
- Memory-efficient processing
- Progress logging for long-running jobs
- Same methodology as the 4x4km smoke test

Creates:
- output/full_mosaic/lc_spectral_analysis.png
- output/full_mosaic/stream_network.png
- output/full_mosaic/hand_map.png
- output/full_mosaic/hillslope_params.json
- output/full_mosaic/full_mosaic_summary.txt

Expected runtime: ~30-60 minutes on 4 cores with 64GB RAM
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for batch jobs
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage, optimize, signal
from scipy.ndimage import distance_transform_edt, label

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid  # noqa: E402

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"

# Output directory uses timestamp + descriptor (set via environment or default)
OUTPUT_DESCRIPTOR = os.environ.get("OUTPUT_DESCRIPTOR", "full")
OUTPUT_TIMESTAMP = time.strftime("%Y-%m-%d")
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / f"{OUTPUT_TIMESTAMP}_{OUTPUT_DESCRIPTOR}"

MOSAIC_PATH = DATA_DIR / "mosaics" / "OSBS_full.tif"

# Tile selection mode: "all" uses full mosaic, "interior" uses selected tiles
TILE_SELECTION_MODE = os.environ.get("TILE_SELECTION_MODE", "all")

# Interior tile selection (excludes edge tiles to avoid boundary effects)
# Format: list of "R#C#-R#C#" range strings
INTERIOR_TILE_RANGES = [
    "R1C11-R1C13",  # 3 tiles
    "R2C11-R2C13",  # 3 tiles
    "R3C11-R3C13",  # 3 tiles
    "R4C5-R4C14",  # 10 tiles
    "R5C1-R5C14",  # 14 tiles
    "R6C1-R6C14",  # 14 tiles
    "R7C1-R7C14",  # 14 tiles
    "R8C1-R8C14",  # 14 tiles
    "R9C1-R9C14",  # 14 tiles
    "R10C1-R10C16",  # 16 tiles
    "R11C5-R11C16",  # 12 tiles
    "R12C5-R12C16",  # 12 tiles
    "R13C5-R13C11",  # 7 tiles
    "R14C5-R14C11",  # 7 tiles
    "R15C5-R15C11",  # 7 tiles
]

# Tile grid parameters (from tile_grid.md)
TILE_GRID_ORIGIN_EASTING = 394000  # UTM easting for column 0
TILE_GRID_ORIGIN_NORTHING = 3292000  # UTM northing for row 0
TILE_SIZE = 1000  # meters per tile

# OSBS center coordinates (from reference file)
OSBS_CENTER_LAT = 29.689282
OSBS_CENTER_LON_360 = 278.006569  # 0-360 convention (= -81.993431 + 360)

# Analysis parameters
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0  # meters
MAX_HILLSLOPE_LENGTH = 2000  # meters
NLAMBDA = 30
MIN_LC_PIXELS = 100  # 100m minimum at 1m resolution

# Aspect bin definitions (degrees from North, clockwise)
ASPECT_BINS = [
    (315, 45),  # North
    (45, 135),  # East
    (135, 225),  # South
    (225, 315),  # West
]
ASPECT_NAMES = ["North", "East", "South", "West"]


def print_section(title: str) -> None:
    """Print a section header with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] {'=' * 56}")
    print(f"[{timestamp}]   {title}")
    print(f"[{timestamp}] {'=' * 56}\n")
    sys.stdout.flush()


def print_progress(msg: str) -> None:
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


# =============================================================================
# Tile Selection and Mosaic Creation
# =============================================================================


def parse_tile_range(range_str: str) -> list[tuple[int, int]]:
    """
    Parse a tile range string into (row, col) tuples.

    Supported formats:
    - "R5C7" -> [(5, 7)]
    - "R5C7-R5C9" -> [(5, 7), (5, 8), (5, 9)]
    - "R4C5-R4C14" -> [(4, 5), (4, 6), ..., (4, 14)]
    """
    # Single tile: R#C#
    single_match = re.match(r"^R(\d+)C(\d+)$", range_str)
    if single_match:
        return [(int(single_match.group(1)), int(single_match.group(2)))]

    # Range: R#C#-R#C#
    range_match = re.match(r"^R(\d+)C(\d+)-R(\d+)C(\d+)$", range_str)
    if range_match:
        r1, c1, r2, c2 = map(int, range_match.groups())
        tiles = []
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                tiles.append((r, c))
        return tiles

    raise ValueError(f"Invalid tile range format: {range_str}")


def parse_all_tile_ranges(ranges: list[str]) -> list[tuple[int, int]]:
    """Parse all tile ranges and return unique (row, col) tuples."""
    tiles = []
    for range_str in ranges:
        tiles.extend(parse_tile_range(range_str))
    # Remove duplicates while preserving order
    seen = set()
    unique_tiles = []
    for tile in tiles:
        if tile not in seen:
            seen.add(tile)
            unique_tiles.append(tile)
    return unique_tiles


def tile_to_filepath(row: int, col: int) -> Path:
    """
    Convert (row, col) to tile filepath.

    Tile naming: NEON_D03_OSBS_DP3_{easting}_{northing}_DTM.tif
    """
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
    filename = f"NEON_D03_OSBS_DP3_{easting}_{northing}_DTM.tif"
    return DATA_DIR / "tiles" / filename


def create_custom_mosaic(
    tile_coords: list[tuple[int, int]], output_path: Path
) -> tuple[Path, int]:
    """
    Create mosaic from selected tiles using rasterio.merge.

    Returns (mosaic_path, tile_count).
    """
    from rasterio.merge import merge

    # Check which tiles exist
    existing_tiles = []
    missing_tiles = []

    for row, col in tile_coords:
        filepath = tile_to_filepath(row, col)
        if filepath.exists():
            existing_tiles.append((row, col, filepath))
        else:
            missing_tiles.append((row, col))

    if missing_tiles:
        print_progress(f"  Warning: {len(missing_tiles)} tiles not found:")
        for row, col in missing_tiles[:10]:  # Show first 10
            print_progress(f"    R{row}C{col}: {tile_to_filepath(row, col).name}")
        if len(missing_tiles) > 10:
            print_progress(f"    ... and {len(missing_tiles) - 10} more")

    if not existing_tiles:
        raise RuntimeError("No tiles found for mosaic creation")

    print_progress(f"  Found {len(existing_tiles)} of {len(tile_coords)} tiles")

    # Reuse existing mosaic if present
    if output_path.exists():
        print_progress(f"  Reusing existing mosaic: {output_path}")
        return output_path, len(existing_tiles)

    # Open all tiles
    print_progress(f"  Opening {len(existing_tiles)} tiles...")
    src_files = []
    for row, col, filepath in existing_tiles:
        src_files.append(rasterio.open(filepath))

    # Merge
    print_progress("  Merging tiles...")
    mosaic, mosaic_transform = merge(src_files)

    # Get profile from first file
    profile = src_files[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        compress="lzw",
    )

    # Write mosaic
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_progress(f"  Writing mosaic to {output_path}...")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    # Close source files
    for src in src_files:
        src.close()

    print_progress(f"  Mosaic created: {mosaic.shape[1]} x {mosaic.shape[2]} pixels")
    return output_path, len(existing_tiles)


def generate_mosaic_heatmap(mosaic_path: Path, output_path: Path) -> None:
    """Generate elevation heatmap of mosaic for tile selection verification."""
    print_progress(f"  Generating mosaic heatmap: {output_path.name}")

    with rasterio.open(mosaic_path) as src:
        dem = src.read(1)
        bounds = src.bounds

    # Mask nodata
    dem_masked = np.ma.masked_less_equal(dem, -9000)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        dem_masked,
        cmap="terrain",
        aspect="equal",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
    )
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    ax.set_title(f"Interior Mosaic Elevation\n{dem.shape[1]} x {dem.shape[0]} pixels")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Spatial Scale Analysis (UTM-adapted) - Same as smoke test
# =============================================================================


def calc_gradient_utm(z: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculate gradient of elevation field in UTM coordinates."""
    dzdy, dzdx = np.gradient(z)

    # Horn 1981 averaging
    dzdy2, dzdx2 = np.zeros(dzdy.shape), np.zeros(dzdx.shape)

    eind = np.asarray([0, 0, 1])
    dzdx2[0, :] = np.mean(dzdx[eind, :], axis=0)
    dzdy2[:, 0] = np.mean(dzdy[:, eind], axis=1)

    eind = np.asarray([-2, -1, -1])
    dzdx2[-1, :] = np.mean(dzdx[eind, :], axis=0)
    dzdy2[:, -1] = np.mean(dzdy[:, eind], axis=1)

    ind = np.asarray([-1, 0, 0, 1])
    for n in range(1, dzdx.shape[0] - 1):
        dzdx2[n, :] = np.mean(dzdx[n + ind, :], axis=0)
    for n in range(1, dzdy.shape[1] - 1):
        dzdy2[:, n] = np.mean(dzdy[:, n + ind], axis=1)

    return (dzdx2 / dx, dzdy2 / dx)


def blend_edges(ifld: np.ndarray, n: int = 10) -> np.ndarray:
    """Blend the edges of a 2D array to reduce spectral leakage."""
    fld = np.copy(ifld)
    jm, im = fld.shape

    tmp = np.zeros((jm, 2 * n))
    for i in range(n):
        w = n - i
        ind = np.arange(-w, (w + 1), 1, dtype=int)
        tmp[:, n + i] = np.sum(fld[:, ind + i], axis=1) / ind.size
        tmp[:, n - (i + 1)] = np.sum(fld[:, ind - (i + 1)], axis=1) / ind.size

    ind = np.arange(-n, n, 1, dtype=int)
    fld[:, ind] = tmp

    tmp = np.zeros((2 * n, im))
    for j in range(n):
        w = n - j
        ind = np.arange(-w, (w + 1), 1, dtype=int)
        tmp[n + j, :] = np.sum(fld[ind + j, :], axis=0) / ind.size
        tmp[n - (j + 1), :] = np.sum(fld[ind - (j + 1), :], axis=0) / ind.size

    ind = np.arange(-n, n, 1, dtype=int)
    fld[ind, :] = tmp

    return fld


def fit_planar_surface_utm(elev: np.ndarray) -> np.ndarray:
    """Fit a planar surface to elevation data for detrending."""
    nrows, ncols = elev.shape
    x = np.arange(ncols)
    y = np.arange(nrows)
    x2d, y2d = np.meshgrid(x, y)

    g = np.column_stack([y2d.ravel(), x2d.ravel(), np.ones(elev.size)])
    gtd = np.dot(g.T, elev.ravel())
    gtg = np.dot(g.T, g)
    coefs = np.linalg.solve(gtg, gtd)

    return y2d * coefs[0] + x2d * coefs[1] + coefs[2]


def bin_amplitude_spectrum(
    amp_fft: np.ndarray, wavelength: np.ndarray, nlambda: int = 20
) -> dict:
    """Bin amplitude spectrum into wavelength bins."""
    logLambda = np.zeros(wavelength.shape)
    logLambda[wavelength > 0] = np.log10(wavelength[wavelength > 0])

    lambda_bounds = np.linspace(0, np.max(logLambda), num=nlambda + 1)
    amp_1d = np.zeros(nlambda)
    lambda_1d = np.zeros(nlambda)

    for n in range(nlambda):
        l1 = np.logical_and(
            logLambda > lambda_bounds[n], logLambda <= lambda_bounds[n + 1]
        )
        if np.any(l1):
            lambda_1d[n] = np.mean(wavelength[l1])
            amp_1d[n] = np.mean(amp_fft[l1])

    ind = np.where(lambda_1d > 0)[0]
    return {"amp": amp_1d[ind], "lambda": lambda_1d[ind]}


def log_normal(
    x: np.ndarray, amp: float, sigma: float, mu: float, shift: float = 0
) -> np.ndarray:
    """Log-normal distribution function."""
    f = np.zeros(x.size)
    if sigma > 0:
        mask = x > shift
        f[mask] = amp * np.exp(
            -((np.log(x[mask] - shift) - mu) ** 2) / (2 * (sigma**2))
        )
    return f


def fit_peak_lognormal(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a log-normal function to locate a peak."""
    meansig = np.mean(y)
    pheight = (meansig, None)
    pwidth = (0, 0.75 * x.size)
    pprom = (0.2 * meansig, None)

    peaks, props = signal.find_peaks(
        y, height=pheight, prominence=pprom, width=pwidth, rel_height=0.5
    )

    if peaks.size == 0:
        pprom = (0.1 * meansig, None)
        peaks, props = signal.find_peaks(
            y, height=pheight, prominence=pprom, width=pwidth, rel_height=0.5
        )

    if peaks.size > 0:
        peaks = np.append(peaks, 0)
        props["widths"] = np.append(props["widths"], np.max(props["widths"]))

    peak_sharp = []
    peak_coefs = []
    peak_gof = []

    for ip in range(peaks.size):
        p = peaks[ip]
        minw = 3
        pw = max(minw, int(0.5 * props["widths"][ip]))
        i1, i2 = max(0, p - pw), min(x.size - 1, p + pw + 1)

        gsigma = np.mean([np.abs(x[p] - x[i1]), np.abs(x[i2] - x[p])])
        amp = np.mean(y[i1 : i2 + 1])
        center = x[p]
        mu = np.log(center) if center > 0 else 0

        try:
            p0 = [amp, gsigma, mu]
            popt, _ = optimize.curve_fit(
                log_normal, x[i1 : i2 + 1], y[i1 : i2 + 1], p0=p0
            )
            ln_peak = np.exp(popt[2])
            pdist = np.abs(center - ln_peak)
            if pdist > popt[1]:
                popt = [0, 0, 1]
        except Exception:
            popt = [0, 0, 1]

        peak_coefs.append(popt)
        peak_gof.append(
            np.mean(np.power(y[i1 : i2 + 1] - log_normal(x[i1 : i2 + 1], *popt), 2))
        )

        if peak_gof[-1] < 1e6 and popt[0] > meansig:
            lnvar = np.sqrt(
                (np.exp(popt[1] ** 2) - 1) * (np.exp(2 * popt[2] + popt[1] ** 2))
            )
            rwid = lnvar / (x[-1] - x[0])
            ramp = popt[0] / np.max(y)
            peak_sharp.append(ramp / rwid)
        else:
            peak_sharp.append(0)

    if len(peak_sharp) > 0:
        pmax = np.argmax(np.asarray(peak_sharp))
        return {
            "coefs": peak_coefs[pmax],
            "psharp": peak_sharp[pmax],
            "gof": peak_gof[pmax],
        }
    return {"coefs": [0, 0, 1], "psharp": 0, "gof": 0}


def identify_spatial_scale_utm(
    elev: np.ndarray,
    pixel_size: float = 1.0,
    max_hillslope_length: float = 2000,
    min_lc_pixels: float = 100,
    subsample: int = 1,
) -> dict:
    """
    Identify characteristic spatial scale for UTM-projected DEM.

    For large DEMs, use subsample > 1 to reduce memory usage.
    """
    print_progress(f"  FFT: Input shape {elev.shape}, subsample={subsample}")

    # Subsample if requested
    if subsample > 1:
        elev = elev[::subsample, ::subsample].copy()
        pixel_size *= subsample
        print_progress(f"  FFT: Subsampled to {elev.shape}")

    nrows, ncols = elev.shape
    max_wavelength = 2 * max_hillslope_length / pixel_size

    # Mask nodata
    valid_mask = elev > -9000
    land_frac = np.sum(valid_mask) / valid_mask.size
    print_progress(f"  FFT: Valid data fraction: {land_frac:.2%}")

    # Fill nodata with mean
    elev_mean = np.mean(elev[valid_mask])
    elev[~valid_mask] = elev_mean

    # Remove planar trend
    print_progress("  FFT: Removing planar trend...")
    elev_planar = fit_planar_surface_utm(elev)
    elev = elev - elev_planar

    # Blend edges
    edge_blend = 50 // subsample if subsample > 1 else 50
    print_progress(f"  FFT: Blending edges (window={edge_blend})...")
    elev = blend_edges(elev, n=edge_blend)

    # Calculate Laplacian
    print_progress("  FFT: Computing Laplacian...")
    grad = calc_gradient_utm(elev, dx=pixel_size)
    x = calc_gradient_utm(grad[0], dx=pixel_size)
    laplac = x[0]
    x = calc_gradient_utm(grad[1], dx=pixel_size)
    laplac += x[1]

    # Zero edges
    edge_zero = 50 // subsample if subsample > 1 else 50
    laplac[:edge_zero, :] = 0
    laplac[:, :edge_zero] = 0
    laplac[-edge_zero:, :] = 0
    laplac[:, -edge_zero:] = 0

    # Compute 2D FFT
    print_progress("  FFT: Computing 2D FFT...")
    laplac_fft = np.fft.rfft2(laplac, norm="ortho")
    laplac_amp_fft = np.abs(laplac_fft)

    # Compute wavelength grid
    rowfreq = np.fft.fftfreq(nrows)
    colfreq = np.fft.rfftfreq(ncols)

    ny, nx = laplac_fft.shape
    radialfreq = np.sqrt(
        np.tile(colfreq * colfreq, (ny, 1)) + np.tile(rowfreq * rowfreq, (nx, 1)).T
    )

    wavelength = np.zeros((ny, nx))
    wavelength[radialfreq > 0] = 1 / radialfreq[radialfreq > 0]
    wavelength[0, 0] = 2 * np.max(wavelength)

    # Bin amplitude spectrum
    x = bin_amplitude_spectrum(laplac_amp_fft, wavelength, nlambda=NLAMBDA)
    lambda_1d, laplac_amp_1d = x["lambda"], x["amp"]

    # Locate peak
    x_ln = fit_peak_lognormal(np.log10(lambda_1d), laplac_amp_1d)

    if x_ln["psharp"] > 1.0:
        model = "lognormal"
        ln_peak = np.exp(x_ln["coefs"][2])
        spatialScale = min(10**ln_peak, max_wavelength)
    else:
        model = "maximum"
        max_idx = np.argmax(laplac_amp_1d)
        spatialScale = lambda_1d[max_idx]

    # Scale back if subsampled
    if subsample > 1:
        spatialScale *= subsample
        lambda_1d *= subsample
        pixel_size /= subsample

    # Enforce minimum
    if spatialScale < min_lc_pixels:
        print_progress(
            f"  FFT: Lc {spatialScale:.1f} px below minimum {min_lc_pixels} px, using minimum"
        )
        spatialScale = min_lc_pixels
        model = f"{model}_constrained"

    print_progress(
        f"  FFT: Model={model}, Lc={spatialScale:.0f} px ({spatialScale * pixel_size:.0f} m)"
    )

    return {
        "valid": True,
        "model": model,
        "spatialScale": spatialScale,
        "spatialScale_m": spatialScale * pixel_size,
        "res": pixel_size,
        "lambda_1d": lambda_1d,
        "laplac_amp_1d": laplac_amp_1d,
    }


# =============================================================================
# Hillslope Parameter Functions (same as smoke test)
# =============================================================================


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
    """Compute HAND bin boundaries following Swenson's methodology."""
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]
    aspect_valid = aspect[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size

    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0
    print_progress(f"    Q25: {initial_q25:.2f} m, bin1_max: {bin1_max:.2f} m")

    if initial_q25 > bin1_max:
        print_progress(f"    Applying mandatory {bin1_max}m constraint")
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
        bounds = [0]
        for q in [0.25, 0.5, 0.75, 1.0]:
            idx = max(0, int(q * n) - 1)
            bounds.append(hand_sorted[idx])
        bounds[-1] = 1e6
        bounds = np.array(bounds)

    print_progress(f"    HAND bins: {bounds[:4]}")
    return bounds


def fit_trapezoidal_width(
    dtnd: np.ndarray,
    area: np.ndarray,
    n_hillslopes: int,
    min_dtnd: float = 1.0,
    n_bins: int = 10,
) -> dict:
    """Fit trapezoidal plan form to hillslope."""
    if np.max(dtnd) < min_dtnd:
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
        Gw = G * weights[:, np.newaxis]
        coeffs = np.linalg.lstsq(Gw, A_cumsum * weights, rcond=None)[0]

        trap_slope = -coeffs[2]
        trap_width = -coeffs[1]
        trap_area = coeffs[0]

        if trap_slope < 0 and trap_width > 0:
            Atri = -(trap_width**2) / (4 * trap_slope)
            if Atri < trap_area:
                trap_width = np.sqrt(-4 * trap_slope * trap_area)

        return {"slope": trap_slope, "width": max(trap_width, 1), "area": trap_area}

    except Exception as e:
        print_progress(f"  Warning: Trapezoidal fit failed: {e}")
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }


def quadratic(coefs, root=0, eps=1e-6):
    """Solve quadratic equation."""
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


def circular_mean_aspect(aspects: np.ndarray) -> float:
    """Compute circular mean of aspect values."""
    sin_sum = np.mean(np.sin(np.radians(aspects)))
    cos_sum = np.mean(np.cos(np.radians(aspects)))
    mean_aspect = np.degrees(np.arctan2(sin_sum, cos_sum))
    if mean_aspect < 0:
        mean_aspect += 360
    return mean_aspect


# =============================================================================
# Plotting Functions
# =============================================================================


def create_spectral_plot(result: dict, output_path: Path) -> None:
    """Create spectral analysis plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lambda_1d = result["lambda_1d"]
    laplac_amp_1d = result["laplac_amp_1d"]
    Lc = result["spatialScale"]
    Lc_m = result["spatialScale_m"]

    ax.semilogy(lambda_1d, laplac_amp_1d, "b.-", linewidth=1.5, markersize=4)
    ax.axvline(
        Lc,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Lc = {Lc:.0f} px ({Lc_m:.0f} m)",
    )

    ax.set_xlabel("Wavelength (pixels)")
    ax.set_ylabel("Laplacian Amplitude")
    ax.set_title(
        f"OSBS Full Mosaic: Spatial Scale Analysis\n"
        f"Model: {result['model']}, Lc = {Lc_m:.0f} m"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_stream_network_plot(
    dem: np.ndarray, stream_mask: np.ndarray, bounds: dict, output_path: Path
) -> None:
    """Create stream network overlay plot."""
    # Downsample for plotting if too large
    max_plot_size = 4000
    downsample = max(1, max(dem.shape) // max_plot_size)

    if downsample > 1:
        dem_plot = dem[::downsample, ::downsample]
        stream_plot = stream_mask[::downsample, ::downsample]
    else:
        dem_plot = dem
        stream_plot = stream_mask

    fig, ax = plt.subplots(figsize=(12, 10))

    dem_masked = np.ma.masked_less_equal(dem_plot, -9000)

    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]
    im = ax.imshow(dem_masked, cmap="terrain", extent=extent, aspect="equal")

    stream_display = np.where(stream_plot > 0, 1, np.nan)
    ax.imshow(stream_display, cmap="Blues", alpha=0.7, extent=extent, aspect="equal")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation (m)")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")
    ax.set_title(
        f"OSBS Full Mosaic: Stream Network\n"
        f"Stream cells: {np.sum(stream_mask > 0):,} "
        f"({100 * np.sum(stream_mask > 0) / stream_mask.size:.2f}%)"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hand_map_plot(hand: np.ndarray, bounds: dict, output_path: Path) -> None:
    """Create HAND map plot."""
    # Downsample for plotting if too large
    max_plot_size = 4000
    downsample = max(1, max(hand.shape) // max_plot_size)

    if downsample > 1:
        hand_plot = hand[::downsample, ::downsample]
    else:
        hand_plot = hand

    fig, ax = plt.subplots(figsize=(12, 10))

    hand_display = np.where(hand_plot < 0, np.nan, hand_plot)
    vmax = np.nanpercentile(hand_display, 95)

    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]
    im = ax.imshow(
        hand_display, cmap="viridis", vmin=0, vmax=vmax, extent=extent, aspect="equal"
    )

    plt.colorbar(im, ax=ax, shrink=0.8, label="HAND (m)")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")

    hand_valid = hand[hand > 0]
    ax.set_title(
        f"OSBS Full Mosaic: Height Above Nearest Drainage\n"
        f"Range: 0 - {np.max(hand_valid):.1f} m, "
        f"Median: {np.median(hand_valid):.1f} m"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hillslope_params_plot(params: dict, output_path: Path) -> None:
    """Create hillslope parameters summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, 4))

    elements = params["elements"]

    ax = axes[0, 0]
    for i, asp in enumerate(ASPECT_NAMES):
        heights = [elements[j]["height"] for j in range(i * 4, (i + 1) * 4)]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins], heights, width=0.18, label=asp, color=colors[i]
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Mean HAND (m)")
    ax.set_title("Height Above Nearest Drainage by Aspect")
    ax.legend()

    ax = axes[0, 1]
    for i, asp in enumerate(ASPECT_NAMES):
        distances = [elements[j]["distance"] for j in range(i * 4, (i + 1) * 4)]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins],
            distances,
            width=0.18,
            label=asp,
            color=colors[i],
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Mean DTND (m)")
    ax.set_title("Distance to Nearest Drainage by Aspect")
    ax.legend()

    ax = axes[1, 0]
    for i, asp in enumerate(ASPECT_NAMES):
        areas = [elements[j]["area"] / 1e6 for j in range(i * 4, (i + 1) * 4)]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins], areas, width=0.18, label=asp, color=colors[i]
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Area (km²)")
    ax.set_title("Hillslope Element Area by Aspect")
    ax.legend()

    ax = axes[1, 1]
    for i, asp in enumerate(ASPECT_NAMES):
        widths = [elements[j]["width"] for j in range(i * 4, (i + 1) * 4)]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins], widths, width=0.18, label=asp, color=colors[i]
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Width (m)")
    ax.set_title("Lower Edge Width by Aspect")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# NetCDF Output for CTSM
# =============================================================================


def write_hillslope_netcdf(
    params: dict,
    output_path: Path,
    center_lon: float,
    center_lat: float,
    total_area_km2: float,
) -> None:
    """
    Write hillslope parameters to CTSM-compatible NetCDF file.

    Parameters
    ----------
    params : dict
        Hillslope parameters from pipeline
    output_path : Path
        Output NetCDF file path
    center_lon : float
        Center longitude in 0-360 convention
    center_lat : float
        Center latitude in degrees north
    total_area_km2 : float
        Total area in km^2
    """
    import netCDF4 as nc

    elements = params["elements"]

    # Extract arrays from elements (16 elements: 4 aspects x 4 bins)
    n_columns = 16
    n_aspects = 4
    n_bins = 4

    # Initialize arrays
    elevation = np.zeros(n_columns)
    distance = np.zeros(n_columns)
    width = np.zeros(n_columns)
    area = np.zeros(n_columns)
    slope = np.zeros(n_columns)
    aspect = np.zeros(n_columns)  # will convert to radians
    hillslope_index = np.zeros(n_columns, dtype=np.int32)
    column_index = np.zeros(n_columns, dtype=np.int32)
    downhill_column_index = np.zeros(n_columns, dtype=np.int32)

    # Fill arrays from elements
    # Elements are ordered: N bin0, N bin1, N bin2, N bin3, E bin0, ...
    for i, elem in enumerate(elements):
        elevation[i] = elem["height"]
        distance[i] = elem["distance"]
        width[i] = elem["width"]
        area[i] = elem["area"]
        slope[i] = elem["slope"]
        aspect[i] = elem["aspect"] * np.pi / 180  # Convert degrees to radians

        # Aspect index (1-4 for N, E, S, W)
        asp_idx = i // n_bins
        hillslope_index[i] = asp_idx + 1

        # Column index (1-16)
        column_index[i] = i + 1

        # Downhill column index
        # For lowest bin (bin 0), set to -9999 (no downhill neighbor)
        # For higher bins, point to the bin below
        bin_idx = i % n_bins
        if bin_idx == 0:
            downhill_column_index[i] = -9999
        else:
            downhill_column_index[i] = i  # Points to i (which is column i+1 - 1 = i)

    # Calculate pct_hillslope (percent of area per aspect)
    pct_hillslope = np.zeros(n_aspects)
    total_area_m2 = sum(elem["area"] for elem in elements)
    for asp_idx in range(n_aspects):
        asp_area = sum(elements[asp_idx * n_bins + j]["area"] for j in range(n_bins))
        if total_area_m2 > 0:
            pct_hillslope[asp_idx] = 100.0 * asp_area / total_area_m2
        else:
            pct_hillslope[asp_idx] = 25.0  # Default equal distribution

    # Stream channel parameters (estimates)
    # Stream depth: typical small stream depth ~0.3m
    stream_depth = 0.3
    # Stream width: typical small stream ~5m
    stream_width = 5.0
    # Stream slope: use mean of lowest-bin slopes * 0.5
    lowest_bin_slopes = [
        elements[asp_idx * n_bins]["slope"] for asp_idx in range(n_aspects)
    ]
    stream_slope = np.mean(lowest_bin_slopes) * 0.5 if any(lowest_bin_slopes) else 0.002

    # Bedrock depth: use large placeholder value (effectively infinite)
    bedrock_depth = np.full(n_columns, 1e6)

    # Create NetCDF file
    print_progress(f"  Writing NetCDF: {output_path.name}")

    with nc.Dataset(output_path, "w", format="NETCDF4") as ds:
        # Global attributes
        ds.creation_date = datetime.now().strftime("%Y-%m-%d")
        ds.source = "OSBS 1m NEON LIDAR processed with hpg-esm-tools/swenson pipeline"
        ds.conventions = "CF-1.6"

        # Create dimensions
        ds.createDimension("lsmlat", 1)
        ds.createDimension("lsmlon", 1)
        ds.createDimension("nhillslope", n_aspects)
        ds.createDimension("nmaxhillcol", n_columns)

        # Coordinate variables
        var_lsmlat = ds.createVariable("lsmlat", "f8", ("lsmlat",))
        var_lsmlat[:] = center_lat

        var_lsmlon = ds.createVariable("lsmlon", "f8", ("lsmlon",))
        var_lsmlon[:] = center_lon

        # 2D coordinate variables
        var_latixy = ds.createVariable("LATIXY", "f8", ("lsmlat", "lsmlon"))
        var_latixy.units = "degrees north"
        var_latixy.long_name = "latitude"
        var_latixy[:] = center_lat

        var_longxy = ds.createVariable("LONGXY", "f8", ("lsmlat", "lsmlon"))
        var_longxy.units = "degrees east"
        var_longxy.long_name = "longitude"
        var_longxy[:] = center_lon

        # Area
        var_area = ds.createVariable("AREA", "f8", ("lsmlat", "lsmlon"))
        var_area.units = "km^2"
        var_area.long_name = "area"
        var_area[:] = total_area_km2

        # Number of hillslope columns
        var_nhillcol = ds.createVariable("nhillcolumns", "i4", ("lsmlat", "lsmlon"))
        var_nhillcol.units = "unitless"
        var_nhillcol.long_name = "number of columns per landunit"
        var_nhillcol[:] = n_columns

        # Percent hillslope
        var_pcthillslope = ds.createVariable(
            "pct_hillslope", "f8", ("nhillslope", "lsmlat", "lsmlon")
        )
        var_pcthillslope.units = "per cent"
        var_pcthillslope.long_name = "percent hillslope of landunit"
        var_pcthillslope[:, 0, 0] = pct_hillslope

        # Hillslope index
        var_hillndx = ds.createVariable(
            "hillslope_index", "i4", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_hillndx.units = "unitless"
        var_hillndx.long_name = "hillslope_index"
        var_hillndx[:, 0, 0] = hillslope_index

        # Column index
        var_colndx = ds.createVariable(
            "column_index", "i4", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_colndx.units = "unitless"
        var_colndx.long_name = "column index"
        var_colndx[:, 0, 0] = column_index

        # Downhill column index
        var_dcolndx = ds.createVariable(
            "downhill_column_index", "i4", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_dcolndx.units = "unitless"
        var_dcolndx.long_name = "downhill column index"
        var_dcolndx[:, 0, 0] = downhill_column_index

        # Hillslope elevation (HAND)
        var_elev = ds.createVariable(
            "hillslope_elevation", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_elev.units = "m"
        var_elev.long_name = "hillslope elevation above channel"
        var_elev[:, 0, 0] = elevation

        # Hillslope distance (DTND)
        var_dist = ds.createVariable(
            "hillslope_distance", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_dist.units = "m"
        var_dist.long_name = "hillslope distance from channel"
        var_dist[:, 0, 0] = distance

        # Hillslope width
        var_width = ds.createVariable(
            "hillslope_width", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_width.units = "m"
        var_width.long_name = "hillslope width"
        var_width[:, 0, 0] = width

        # Hillslope area
        var_harea = ds.createVariable(
            "hillslope_area", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_harea.units = "m2"
        var_harea.long_name = "hillslope area"
        var_harea[:, 0, 0] = area

        # Hillslope slope
        var_slope = ds.createVariable(
            "hillslope_slope", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_slope.units = "m/m"
        var_slope.long_name = "hillslope slope"
        var_slope[:, 0, 0] = slope

        # Hillslope aspect (in radians)
        var_aspect = ds.createVariable(
            "hillslope_aspect", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_aspect.units = "radians"
        var_aspect.long_name = "hillslope aspect (clockwise from North)"
        var_aspect[:, 0, 0] = aspect

        # Hillslope bedrock depth
        var_bedrock = ds.createVariable(
            "hillslope_bedrock_depth", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_bedrock.units = "meters"
        var_bedrock.long_name = "hillslope bedrock depth"
        var_bedrock[:, 0, 0] = bedrock_depth

        # Stream channel parameters
        var_sdepth = ds.createVariable(
            "hillslope_stream_depth", "f8", ("lsmlat", "lsmlon")
        )
        var_sdepth.units = "meters"
        var_sdepth.long_name = "stream channel bankfull depth"
        var_sdepth[:] = stream_depth

        var_swidth = ds.createVariable(
            "hillslope_stream_width", "f8", ("lsmlat", "lsmlon")
        )
        var_swidth.units = "meters"
        var_swidth.long_name = "stream channel bankfull width"
        var_swidth[:] = stream_width

        var_sslope = ds.createVariable(
            "hillslope_stream_slope", "f8", ("lsmlat", "lsmlon")
        )
        var_sslope.units = "m/m"
        var_sslope.long_name = "stream channel slope"
        var_sslope[:] = stream_slope

    print_progress(f"  NetCDF written: {output_path}")

    # Print summary
    print_progress("  NetCDF summary:")
    print_progress(f"    Location: {center_lat:.4f}N, {center_lon:.4f}E (0-360)")
    print_progress(f"    Area: {total_area_km2:.2f} km^2")
    print_progress(f"    pct_hillslope: {pct_hillslope}")
    print_progress(
        f"    Stream: depth={stream_depth:.2f}m, width={stream_width:.1f}m, slope={stream_slope:.6f}"
    )


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("OSBS Hillslope Pipeline")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 0: Determine Mosaic Source
    # -------------------------------------------------------------------------
    print_section("Step 0: Mosaic Selection")

    print_progress(f"  Tile selection mode: {TILE_SELECTION_MODE}")

    if TILE_SELECTION_MODE == "interior":
        # Create custom mosaic from interior tiles
        print_progress("  Parsing interior tile selection...")
        tile_coords = parse_all_tile_ranges(INTERIOR_TILE_RANGES)
        print_progress(f"  Selected {len(tile_coords)} tiles")

        mosaic_path = DATA_DIR / "mosaics" / "OSBS_interior.tif"
        mosaic_path, tile_count = create_custom_mosaic(tile_coords, mosaic_path)

        # Generate verification heatmap
        heatmap_path = OUTPUT_DIR / "mosaic_heatmap.png"
        if not heatmap_path.exists():
            generate_mosaic_heatmap(mosaic_path, heatmap_path)
            print_progress(f"  Saved: {heatmap_path}")
    else:
        # Use full mosaic
        mosaic_path = MOSAIC_PATH
        if not mosaic_path.exists():
            print(f"ERROR: Mosaic file not found: {mosaic_path}")
            sys.exit(1)
        print_progress(f"  Using full mosaic: {mosaic_path}")

    # -------------------------------------------------------------------------
    # Step 1: Load DEM
    # -------------------------------------------------------------------------
    print_section("Step 1: Loading DEM")

    with rasterio.open(mosaic_path) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    bounds_dict = {
        "west": bounds.left,
        "east": bounds.right,
        "south": bounds.bottom,
        "north": bounds.top,
    }

    pixel_size = abs(transform.a)
    dem_valid = dem[dem > -9000]

    print_progress(f"  Shape: {dem.shape}")
    print_progress(f"  CRS: {crs}")
    print_progress(f"  Pixel size: {pixel_size} m")
    print_progress(f"  Bounds: {bounds_dict}")
    print_progress(
        f"  Elevation range: {dem_valid.min():.1f} - {dem_valid.max():.1f} m"
    )
    print_progress(f"  Memory: {dem.nbytes / 1e9:.2f} GB")

    # -------------------------------------------------------------------------
    # Step 2: Spatial Scale Analysis
    # -------------------------------------------------------------------------
    print_section("Step 2: Spatial Scale Analysis (FFT)")

    t0 = time.time()

    # For large DEMs, subsample to reduce FFT memory
    # 19000x17000 is ~323 million pixels - subsample by 4 to ~20 million
    subsample_factor = 4

    spatial_result = identify_spatial_scale_utm(
        dem,
        pixel_size=pixel_size,
        max_hillslope_length=MAX_HILLSLOPE_LENGTH,
        min_lc_pixels=MIN_LC_PIXELS,
        subsample=subsample_factor,
    )

    Lc = spatial_result["spatialScale"]
    Lc_m = spatial_result["spatialScale_m"]
    accum_threshold = int(0.5 * Lc**2)

    print_progress(f"  Characteristic length (Lc): {Lc:.0f} px ({Lc_m:.0f} m)")
    print_progress(f"  Accumulation threshold: {accum_threshold} cells")
    print_progress(f"  FFT time: {time.time() - t0:.1f} seconds")

    create_spectral_plot(spatial_result, OUTPUT_DIR / "lc_spectral_analysis.png")
    print_progress(f"  Saved: {OUTPUT_DIR / 'lc_spectral_analysis.png'}")

    # -------------------------------------------------------------------------
    # Step 3: Extract Largest Connected Region and Flow Routing
    # -------------------------------------------------------------------------
    print_section("Step 3: Flow Routing")

    t0 = time.time()

    from pyproj import Proj as PyprojProj

    # Find the largest connected component of valid data
    # This avoids flow fragmentation across nodata gaps in the mosaic
    valid_mask = dem > -9000
    valid_frac = np.sum(valid_mask) / valid_mask.size
    print_progress(f"  Valid data fraction: {valid_frac:.2%}")

    if valid_frac < 1.0:
        print_progress("  Finding largest connected component of valid data...")
        labeled_array, num_features = label(valid_mask)
        print_progress(f"  Found {num_features} connected components")

        # Find the largest component
        component_sizes = ndimage.sum(
            valid_mask, labeled_array, range(1, num_features + 1)
        )
        largest_label = np.argmax(component_sizes) + 1
        largest_size = component_sizes[largest_label - 1]
        print_progress(
            f"  Largest component: {largest_size:,.0f} pixels "
            f"({100 * largest_size / valid_mask.size:.1f}% of total)"
        )

        # Create mask for largest component only
        largest_mask = labeled_array == largest_label

        # Find bounding box of largest component
        rows = np.any(largest_mask, axis=1)
        cols = np.any(largest_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(dem.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(dem.shape[1] - 1, cmax + pad)

        print_progress(
            f"  Extracting region: rows [{rmin}:{rmax}], cols [{cmin}:{cmax}]"
        )
        print_progress(f"  Region size: {rmax - rmin} x {cmax - cmin} pixels")

        # Extract the region
        dem_region = dem[rmin:rmax, cmin:cmax].copy()
        valid_mask = largest_mask[rmin:rmax, cmin:cmax]

        # Subsample for flow routing to avoid pysheds scaling issues
        # The resolve_flats step has O(n^2) or worse complexity with large flat regions
        subsample = 4  # Process at 4m resolution instead of 1m
        print_progress(f"  Subsampling by {subsample}x for flow routing...")

        dem_sub = dem_region[::subsample, ::subsample]
        valid_mask_sub = largest_mask[rmin:rmax:subsample, cmin:cmax:subsample]

        # Trim nodata-only edges - pysheds fails when all edges are nodata
        rows_valid = np.where(np.any(valid_mask_sub, axis=1))[0]
        cols_valid = np.where(np.any(valid_mask_sub, axis=0))[0]
        tr1, tr2 = rows_valid[0], rows_valid[-1] + 1
        tc1, tc2 = cols_valid[0], cols_valid[-1] + 1

        dem_sub = dem_sub[tr1:tr2, tc1:tc2]
        valid_mask_sub = valid_mask_sub[tr1:tr2, tc1:tc2]

        nodata_count = (~valid_mask_sub).sum()
        print_progress(
            f"  Trimmed nodata edges, new shape: {dem_sub.shape}, nodata: {nodata_count:,}"
        )

        # Use subsampled valid_mask for routing
        valid_mask = valid_mask_sub
        dem_for_routing = dem_sub

        # Adjust region bounds for trimmed edges
        cmin += tc1 * subsample
        cmax = cmin + (tc2 - tc1) * subsample
        rmin += tr1 * subsample
        rmax = rmin + (tr2 - tr1) * subsample

        # Update transform for the subsampled region
        region_transform = rasterio.Affine(
            transform.a * subsample,
            transform.b,
            transform.c + cmin * transform.a,
            transform.d,
            transform.e * subsample,
            transform.f + rmin * transform.e,
        )

        # Update bounds (same as original region)
        bounds_dict = {
            "west": transform.c + cmin * transform.a,
            "east": transform.c + cmax * transform.a,
            "south": transform.f + rmax * transform.e,
            "north": transform.f + rmin * transform.e,
        }
        print_progress(f"  Updated bounds: {bounds_dict}")

        transform_for_routing = region_transform
        pixel_size = pixel_size * subsample  # Update for subsampled data

        # Recalculate accumulation threshold for subsampled resolution
        Lc_sub = Lc / subsample  # Lc in subsampled pixels
        accum_threshold = int(0.5 * Lc_sub**2)
        print_progress(
            f"  Adjusted Lc: {Lc_sub:.0f} px at {pixel_size}m, threshold: {accum_threshold} cells"
        )
    else:
        dem_for_routing = dem
        transform_for_routing = transform

    grid = Grid()
    grid.add_gridded_data(
        dem_for_routing,
        data_name="dem",
        affine=transform_for_routing,
        crs=PyprojProj(crs.to_proj4()),
        nodata=-9999,
    )

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    print_progress("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    print_progress("  Filling depressions...")
    grid.fill_depressions("pit_filled", out_name="flooded")
    print_progress("  Resolving flats...")
    grid.resolve_flats("flooded", out_name="inflated")

    print_progress("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=dirmap, routing="d8")

    print_progress("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=dirmap, routing="d8")
    acc = grid.acc

    print_progress(f"  Max accumulation: {np.max(acc):.0f} cells")
    print_progress(f"  Flow routing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 4: Stream Network and HAND/DTND
    # -------------------------------------------------------------------------
    print_section("Step 4: Stream Network and HAND/DTND")

    t0 = time.time()

    acc_mask = acc > accum_threshold
    print_progress("  Creating channel mask...")
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=dirmap, routing="d8")

    stream_mask = grid.channel_mask
    channel_id = grid.channel_id

    stream_cells = np.sum(stream_mask > 0)
    stream_frac = stream_cells / stream_mask.size
    num_channels = int(np.nanmax(channel_id)) if np.any(~np.isnan(channel_id)) else 0

    print_progress(f"  Accumulation threshold: {accum_threshold} cells")
    print_progress(f"  Stream cells: {stream_cells:,} ({stream_frac:.2%})")
    print_progress(f"  Number of stream segments: {num_channels}")

    print_progress("  Computing HAND...")
    grid.compute_hand(
        "fdir", "dem", grid.channel_mask, grid.channel_id, dirmap=dirmap, routing="d8"
    )
    hand = np.array(grid.hand)

    print_progress("  Computing DTND (Euclidean for UTM)...")
    stream_binary = stream_mask > 0
    dtnd_pixels = distance_transform_edt(~stream_binary)
    dtnd = dtnd_pixels * pixel_size

    # Apply valid_mask to exclude filled nodata regions from statistics
    hand_valid = hand[(hand > 0) & np.isfinite(hand) & valid_mask]
    dtnd_valid = dtnd[(dtnd > 0) & valid_mask]

    print_progress(
        f"  HAND range: 0 - {np.max(hand_valid):.1f} m, median: {np.median(hand_valid):.1f} m"
    )
    print_progress(
        f"  DTND range: 0 - {np.max(dtnd_valid):.0f} m, median: {np.median(dtnd_valid):.0f} m"
    )
    print_progress(f"  HAND/DTND time: {time.time() - t0:.1f} seconds")

    print_progress("  Creating stream network plot...")
    create_stream_network_plot(
        dem_for_routing, stream_mask, bounds_dict, OUTPUT_DIR / "stream_network.png"
    )
    print_progress(f"  Saved: {OUTPUT_DIR / 'stream_network.png'}")

    print_progress("  Creating HAND map plot...")
    create_hand_map_plot(hand, bounds_dict, OUTPUT_DIR / "hand_map.png")
    print_progress(f"  Saved: {OUTPUT_DIR / 'hand_map.png'}")

    # -------------------------------------------------------------------------
    # Step 5: Slope and Aspect
    # -------------------------------------------------------------------------
    print_section("Step 5: Slope and Aspect")

    t0 = time.time()

    print_progress("  Computing slope and aspect...")
    grid.slope_aspect("dem")
    slope = np.array(grid.slope)
    aspect = np.array(grid.aspect)

    print_progress(f"  Slope range: {np.nanmin(slope):.4f} - {np.nanmax(slope):.4f}")
    print_progress(
        f"  Aspect range: {np.nanmin(aspect):.1f} - {np.nanmax(aspect):.1f} degrees"
    )
    print_progress(f"  Slope/aspect time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 6: Hillslope Parameter Computation
    # -------------------------------------------------------------------------
    print_section("Step 6: Hillslope Parameters (16 elements)")

    t0 = time.time()

    hand_flat = hand.flatten()
    dtnd_flat = dtnd.flatten()
    slope_flat = slope.flatten()
    aspect_flat = aspect.flatten()

    pixel_area = pixel_size * pixel_size
    area_flat = np.full(hand_flat.shape, pixel_area)

    # Include original valid_mask to exclude filled nodata regions
    valid_mask_flat = valid_mask.flatten()
    valid = (hand_flat >= 0) & np.isfinite(hand_flat) & valid_mask_flat
    print_progress(
        f"  Valid pixels: {np.sum(valid):,} ({100 * np.sum(valid) / valid.size:.1f}%)"
    )

    print_progress("  Computing HAND bins...")
    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )

    params = {
        "metadata": {
            "n_aspect_bins": N_ASPECT_BINS,
            "n_hand_bins": N_HAND_BINS,
            "aspect_bins": ASPECT_BINS,
            "aspect_names": ASPECT_NAMES,
            "hand_bounds": hand_bounds.tolist(),
            "accum_threshold": accum_threshold,
            "spatial_scale_px": float(Lc),
            "spatial_scale_m": float(Lc_m),
            "region_shape": list(dem_for_routing.shape),
            "bounds": bounds_dict,
            "pixel_size_m": pixel_size,
        },
        "elements": [],
    }

    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        print_progress(f"  Processing {asp_name} aspect...")

        asp_mask = get_aspect_mask(aspect_flat, asp_bin) & valid
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            print_progress(f"    No pixels in {asp_name} aspect")
            for _ in range(N_HAND_BINS):
                params["elements"].append(
                    {
                        "aspect_name": asp_name,
                        "aspect_bin": asp_idx,
                        "hand_bin": 0,
                        "height": 0,
                        "distance": 0,
                        "area": 0,
                        "slope": 0,
                        "aspect": 0,
                        "width": 0,
                    }
                )
            continue

        n_hillslopes = max(
            1,
            len(
                np.unique(
                    grid.drainage_id.flatten()[asp_indices]
                    if hasattr(grid, "drainage_id")
                    else [1]
                )
            ),
        )

        hillslope_frac = np.sum(area_flat[asp_indices]) / np.sum(area_flat[valid])
        print_progress(
            f"    Pixels: {len(asp_indices):,}, Fraction: {hillslope_frac:.1%}"
        )

        trap = fit_trapezoidal_width(
            dtnd_flat[asp_indices], area_flat[asp_indices], n_hillslopes, min_dtnd=1.0
        )

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
                raw_area = float(np.sum(area_flat[bin_indices]))
                bin_raw_areas.append(raw_area)
                bin_data.append(
                    {"indices": bin_indices, "h_lower": h_lower, "h_upper": h_upper}
                )

        total_raw = sum(bin_raw_areas)
        area_fractions = (
            [a / total_raw for a in bin_raw_areas]
            if total_raw > 0
            else [0.25] * N_HAND_BINS
        )
        fitted_areas = [trap["area"] * frac for frac in area_fractions]

        for h_idx in range(N_HAND_BINS):
            data = bin_data[h_idx]
            bin_indices = data["indices"]

            if bin_indices is None:
                params["elements"].append(
                    {
                        "aspect_name": asp_name,
                        "aspect_bin": asp_idx,
                        "hand_bin": h_idx,
                        "height": float((data["h_lower"] + data["h_upper"]) / 2),
                        "distance": 0,
                        "area": 0,
                        "slope": 0,
                        "aspect": float((asp_bin[0] + asp_bin[1]) / 2 % 360),
                        "width": 0,
                    }
                )
                continue

            mean_hand = float(np.mean(hand_flat[bin_indices]))
            mean_slope = float(np.nanmean(slope_flat[bin_indices]))
            mean_aspect = circular_mean_aspect(aspect_flat[bin_indices])
            dtnd_sorted = np.sort(dtnd_flat[bin_indices])
            median_dtnd = float(dtnd_sorted[len(dtnd_sorted) // 2])
            fitted_area = fitted_areas[h_idx]

            da = sum(fitted_areas[:h_idx]) if h_idx > 0 else 0
            if trap["slope"] != 0:
                try:
                    le = quadratic([trap["slope"], trap["width"], -da])
                    width = trap["width"] + 2 * trap["slope"] * le
                except RuntimeError:
                    width = trap["width"] * (1 - 0.15 * h_idx)
            else:
                width = trap["width"]

            width = max(float(width), 1)

            params["elements"].append(
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

            print_progress(
                f"    Bin {h_idx + 1}: h={mean_hand:.1f}m, d={median_dtnd:.0f}m, w={width:.0f}m"
            )

    print_progress(f"  Hillslope param time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 7: Save Results
    # -------------------------------------------------------------------------
    print_section("Step 7: Saving Results")

    json_path = OUTPUT_DIR / "hillslope_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print_progress(f"  Saved: {json_path}")

    create_hillslope_params_plot(params, OUTPUT_DIR / "hillslope_params.png")
    print_progress(f"  Saved: {OUTPUT_DIR / 'hillslope_params.png'}")

    # Write CTSM-compatible NetCDF file
    timetag = datetime.now().strftime("%y%m%d")
    nc_filename = f"hillslopes_osbs_{OUTPUT_DESCRIPTOR}_c{timetag}.nc"
    nc_path = OUTPUT_DIR / nc_filename

    # Calculate total area from elements (in km^2)
    total_area_m2 = sum(elem["area"] for elem in params["elements"])
    total_area_km2 = total_area_m2 / 1e6

    write_hillslope_netcdf(
        params=params,
        output_path=nc_path,
        center_lon=OSBS_CENTER_LON_360,
        center_lat=OSBS_CENTER_LAT,
        total_area_km2=total_area_km2,
    )

    # Write summary text file
    summary_path = OUTPUT_DIR / f"{OUTPUT_DESCRIPTOR}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"OSBS Hillslope Pipeline Summary ({OUTPUT_DESCRIPTOR})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Tile selection mode: {TILE_SELECTION_MODE}\n")
        f.write(f"Input: {mosaic_path}\n")
        f.write(
            f"Processed region shape: {dem_for_routing.shape} pixels "
            f"({dem_for_routing.shape[0] * pixel_size / 1000:.1f} x {dem_for_routing.shape[1] * pixel_size / 1000:.1f} km)\n"
        )
        f.write(f"Resolution: {pixel_size} m\n\n")

        f.write("Spatial Scale Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model: {spatial_result['model']}\n")
        f.write(f"  Characteristic length: {Lc:.0f} px ({Lc_m:.0f} m)\n")
        f.write(f"  Accumulation threshold: {accum_threshold} cells\n\n")

        f.write("Stream Network:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Stream cells: {stream_cells:,} ({stream_frac:.2%})\n")
        f.write(f"  Stream segments: {num_channels}\n\n")

        f.write("HAND Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Range: 0 - {np.max(hand_valid):.1f} m\n")
        f.write(f"  Median: {np.median(hand_valid):.1f} m\n")
        f.write(f"  HAND bins: {hand_bounds.tolist()}\n\n")

        f.write("Hillslope Elements (16 total):\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Aspect':<10} {'Bin':<5} {'Height':<8} {'Distance':<10} "
            f"{'Area':<10} {'Width':<8}\n"
        )
        f.write("-" * 60 + "\n")

        for elem in params["elements"]:
            f.write(
                f"{elem['aspect_name']:<10} {elem['hand_bin'] + 1:<5} "
                f"{elem['height']:<8.1f} {elem['distance']:<10.0f} "
                f"{elem['area'] / 1e6:<10.3f} {elem['width']:<8.0f}\n"
            )

        f.write(f"\nNetCDF output: {nc_filename}\n")
        f.write(f"Total processing time: {time.time() - start_time:.1f} seconds\n")

    print_progress(f"  Saved: {summary_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Pipeline Complete")

    total_time = time.time() - start_time
    print_progress(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    print_progress(f"\nOutputs saved to: {OUTPUT_DIR}")
    print_progress("\nKey results:")
    print_progress(f"  Characteristic length (Lc): {Lc_m:.0f} m")
    print_progress(f"  Accumulation threshold: {accum_threshold} cells")
    print_progress(f"  Stream coverage: {stream_frac:.2%}")
    print_progress(f"  HAND range: 0 - {np.max(hand_valid):.1f} m")
    print_progress(f"\nCTSM-compatible NetCDF: {nc_path}")


if __name__ == "__main__":
    main()
