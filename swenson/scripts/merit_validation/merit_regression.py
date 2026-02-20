#!/usr/bin/env python
"""
MERIT Geographic Regression Test

Validates the pysheds fork's geographic CRS code path by:
1. Computing characteristic length scale (Lc) via FFT
2. Computing 6 hillslope parameters for a known MERIT gridcell
3. Comparing to Swenson's published data

Expected runtime: ~50-90 min, 48GB RAM.
Exit code: 0 on PASS, 1 on FAIL.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import rasterio
import xarray as xr
from pyproj import Proj as PyprojProj
from pysheds.pgrid import Grid
from rasterio.windows import from_bounds

# Add parent directory for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_scale import DTR, RE, identify_spatial_scale_laplacian_dem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SWENSON_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
MERIT_DEM = os.path.join(SWENSON_DIR, "data/merit/n30w095_dem.tif")
PUBLISHED_NC = os.path.join(
    SWENSON_DIR, "data/reference/hillslopes_0.9x1.25_c240416.nc"
)
OUTPUT_DIR = os.path.join(SWENSON_DIR, "output/merit_validation")

# Target gridcell (0.9 x 1.25 degree)
TARGET_LON = (-93.125, -91.875)
TARGET_LAT = (32.0419, 32.9843)
TARGET_CENTER_LON = -92.5000
TARGET_CENTER_LAT = 32.5131

# FFT regions: 3 center crops + full tile at native resolution
FFT_REGION_SIZES = [500, 1000, 3000]

# Expected correlations (tolerance 0.01)
# Width and area_fraction updated after fixing w^2→w^1 polynomial weighting
# to match Swenson's _fit_polynomial (see area_fraction_research.md, Test I).
# Height, slope, width, area_fraction updated after switching compute_hand
# to use inflated DEM (matches Swenson rh:1685). See dem_conditioning_todo.md item 3.
EXPECTED = {
    "height": 0.9977,
    "distance": 0.9986,
    "slope": 0.9827,
    "aspect": 0.9999,
    "width": 0.9471,
    "area_fraction": 0.8284,
}
TOLERANCE = 0.01
EXPECTED_LC_M = 763.0
LC_TOLERANCE_PCT = 5.0

# Flow routing
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
EXPANSION_FACTOR = 1.5

# FFT parameters
MAX_HILLSLOPE_LENGTH = 10000
NLAMBDA = 30

# Binning
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0
ASPECT_BINS = [(315, 45), (45, 135), (135, 225), (225, 315)]
ASPECT_NAMES = ["North", "East", "South", "West"]


# ---------------------------------------------------------------------------
# Lc computation (from stage 2)
# ---------------------------------------------------------------------------
def load_dem_with_coords(
    filepath: str,
    subsample: int | None = None,
    center_region: int | None = None,
) -> dict:
    """Load DEM and compute coordinate arrays."""
    with rasterio.open(filepath) as src:
        elev = src.read(1)
        nrows, ncols = elev.shape
        transform = src.transform
        lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
        lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    if center_region is not None:
        cy, cx = nrows // 2, ncols // 2
        half = center_region // 2
        elev = elev[cy - half : cy + half, cx - half : cx + half]
        lon = lon[cx - half : cx + half]
        lat = lat[cy - half : cy + half]

    if subsample is not None and subsample > 1:
        elev = elev[::subsample, ::subsample]
        lon = lon[::subsample]
        lat = lat[::subsample]

    return {"elev": elev, "lon": lon, "lat": lat}


def compute_lc(dem_path: str) -> dict:
    """
    Run FFT on 3 center regions + full tile at native resolution.

    Returns dict with all 4 Lc values and the median.
    """
    regions = []

    for size in FFT_REGION_SIZES:
        data = load_dem_with_coords(dem_path, center_region=size)
        result = identify_spatial_scale_laplacian_dem(
            elev=data["elev"],
            elon=data["lon"],
            elat=data["lat"],
            max_hillslope_length=MAX_HILLSLOPE_LENGTH,
            land_threshold=0.75,
            min_land_elevation=0,
            detrend_elevation=True,
            blend_edges_flag=True,
            zero_edges=True,
            nlambda=NLAMBDA,
            verbose=False,
        )
        lc_px = result["spatialScale"]
        ares = result["res"]
        regions.append(
            {
                "label": f"{size}x{size} center",
                "lc_px": round(float(lc_px), 1),
                "lc_m": round(float(lc_px * ares), 0),
                "model": result["model"],
            }
        )

    # Full tile (subsample 4x to keep memory reasonable, scale result back)
    subsample = 4
    data = load_dem_with_coords(dem_path, subsample=subsample)
    result = identify_spatial_scale_laplacian_dem(
        elev=data["elev"],
        elon=data["lon"],
        elat=data["lat"],
        max_hillslope_length=MAX_HILLSLOPE_LENGTH,
        land_threshold=0.75,
        min_land_elevation=0,
        detrend_elevation=True,
        blend_edges_flag=True,
        zero_edges=True,
        nlambda=NLAMBDA,
        verbose=False,
    )
    lc_px = result["spatialScale"] * subsample
    ares = result["res"] / subsample
    regions.append(
        {
            "label": "full tile",
            "lc_px": round(float(lc_px), 1),
            "lc_m": round(float(lc_px * ares), 0),
            "model": result["model"],
        }
    )

    median_lc_m = float(np.median([r["lc_m"] for r in regions]))
    return {"regions": regions, "median_lc_m": median_lc_m}


# ---------------------------------------------------------------------------
# Hillslope parameter helpers (from stage 3)
# ---------------------------------------------------------------------------
def get_aspect_mask(aspect: np.ndarray, aspect_bin: tuple) -> np.ndarray:
    """Create boolean mask for pixels within an aspect bin."""
    lower, upper = aspect_bin
    if lower > upper:
        return (aspect >= lower) | (aspect < upper)
    return (aspect >= lower) & (aspect < upper)


def compute_hand_bins(
    hand: np.ndarray,
    aspect: np.ndarray,
    aspect_bins: list,
    bin1_max: float = 2.0,
    min_aspect_fraction: float = 0.01,
) -> np.ndarray:
    """Compute HAND bin boundaries following Swenson's SpecifyHandBounds()."""
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size
    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    if initial_q25 > bin1_max:
        # Ensure at least min_aspect_fraction of pixels exist below bin1_max
        # for each aspect bin (Swenson SpecifyHandBounds lines 365-395).
        # Aspect mask applied to full arrays (including zeros) to match Swenson.
        for asp_idx, (asp_low, asp_high) in enumerate(aspect_bins):
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
        bounds[-1] = 1e6
        bounds = np.array(bounds)

    return bounds


def fit_trapezoidal_width(
    dtnd: np.ndarray,
    area: np.ndarray,
    n_hillslopes: int,
    min_dtnd: float = 90,
    n_bins: int = 10,
) -> dict:
    """
    Fit trapezoidal plan form following Swenson Eq. (4).

    Uses Swenson's _fit_polynomial weighting: W = diag(weights),
    solving (G^T W G) coefs = G^T W y. This minimizes
    sum_i w_i * (residual_i)^2 (w^1 weighting).
    """
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


def circular_mean_aspect(aspects: np.ndarray) -> float:
    """Compute circular mean of aspect values (degrees)."""
    sin_sum = np.mean(np.sin(DTR * aspects))
    cos_sum = np.mean(np.cos(DTR * aspects))
    mean_aspect = np.arctan2(sin_sum, cos_sum) / DTR
    if mean_aspect < 0:
        mean_aspect += 360
    return mean_aspect


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


# ---------------------------------------------------------------------------
# Basin / open water detection (from Swenson geospatial_utils.py)
# ---------------------------------------------------------------------------
def _four_point_laplacian(mask: np.ndarray) -> np.ndarray:
    """4-neighbor Laplacian on a 0/1 mask. Returns abs value per pixel."""
    jm = mask.shape[0]
    laplacian = -4.0 * np.copy(mask)
    laplacian += mask * np.roll(mask, 1, axis=1) + mask * np.roll(mask, -1, axis=1)
    temp = np.roll(mask, 1, axis=0)
    temp[0, :] = mask[1, :]
    laplacian += mask * temp
    temp = np.roll(mask, -1, axis=0)
    temp[jm - 1, :] = mask[jm - 2, :]
    laplacian += mask * temp
    return np.abs(laplacian)


def _expand_mask_buffer(mask: np.ndarray, buf: int = 1) -> np.ndarray:
    """Spatial dilation: set pixel to 1 if any neighbor within buf is 1."""
    omask = np.copy(mask)
    offset = mask.shape[1]
    # Identify interior pixels (exclude buf-width border)
    a = np.arange(mask.size)
    top = []
    for i in range(buf):
        top.extend((i * offset + np.arange(offset)[buf:-buf]).tolist())
    top = np.array(top, dtype=int)
    bottom = mask.size - 1 - top
    left = []
    for i in range(buf):
        left.extend(np.arange(i, mask.size, offset))
    left = np.array(left, dtype=int)
    right = mask.size - 1 - left
    exclude = np.unique(np.concatenate([top, left, right, bottom]))
    inside = np.delete(a, exclude)

    lmask = np.where(_four_point_laplacian(mask) > 0, 1, 0)
    ind = inside[(lmask.flat[inside] > 0)]
    for k in range(-buf, buf + 1):
        if k != 0:
            omask.flat[ind + k] = 1
        for j in range(buf):
            j1 = j + 1
            omask.flat[ind + k + j1 * offset] = 1
            omask.flat[ind + k - j1 * offset] = 1
    return omask


def erode_dilate_mask(mask: np.ndarray, buf: int = 1, niter: int = 10) -> np.ndarray:
    """Morphological open: erode niter times, dilate niter+1, intersect with original."""
    x = np.copy(mask)
    for _ in range(niter):
        x = 1 - _expand_mask_buffer(1 - x, buf=buf)
    for _ in range(niter + 1):
        x = _expand_mask_buffer(x, buf=buf)
    return np.where((x > 0) & (mask > 0), 1, 0)


def identify_basins(
    dem: np.ndarray,
    basin_thresh: float = 0.25,
    niter: int = 10,
    buf: int = 1,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Detect flat basin floors in DEM via elevation histogram.

    Any elevation value occurring in >basin_thresh fraction of pixels
    is considered a basin floor. Morphological cleanup removes noise.
    Returns 0/1 mask (1 = basin pixel).
    """
    imask = np.zeros(dem.shape)

    if nodata is not None:
        udem, ucnt = np.unique(dem[dem != nodata], return_counts=True)
    else:
        udem, ucnt = np.unique(dem, return_counts=True)
    ufrac = ucnt / dem.size
    ind = np.where(ufrac > basin_thresh)[0]

    if ind.size > 0:
        for i in ind:
            eps = 1e-2
            if np.abs(udem[i]) < eps:
                eps = 1e-6
            imask[np.abs(dem - udem[i]) < eps] = 1

        for _ in range(niter):
            imask = _expand_mask_buffer(imask, buf=buf)
            imask[_four_point_laplacian(1 - imask) >= 3] = 0

    return imask


def identify_open_water(
    slope: np.ndarray, max_slope: float = 1e-4, niter: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect coherent open water from slope field.

    Returns (basin_boundary, basin_mask) where basin_boundary is a 2-pixel
    ring around each water body.
    """
    basin_mask = erode_dilate_mask(np.where(slope < max_slope, 1, 0), niter=niter)
    sup_basin_mask = _expand_mask_buffer(basin_mask, buf=2)
    basin_boundary = sup_basin_mask - basin_mask
    return basin_boundary, basin_mask


# ---------------------------------------------------------------------------
# Hillslope parameter computation (from stage 3)
# ---------------------------------------------------------------------------
def compute_hillslope_params(dem_path: str, accum_threshold: int) -> dict:
    """
    Run flow routing + HAND/DTND + slope/aspect + binning + width fitting.

    Returns dict with 16 element dicts containing the 6 parameters.
    """
    # Load expanded region for flow routing
    gc_lon_min, gc_lon_max = TARGET_LON
    gc_lat_min, gc_lat_max = TARGET_LAT
    lon_width = gc_lon_max - gc_lon_min
    lat_height = gc_lat_max - gc_lat_min

    exp_lon_min = TARGET_CENTER_LON - EXPANSION_FACTOR * lon_width / 2
    exp_lon_max = TARGET_CENTER_LON + EXPANSION_FACTOR * lon_width / 2
    exp_lat_min = TARGET_CENTER_LAT - EXPANSION_FACTOR * lat_height / 2
    exp_lat_max = TARGET_CENTER_LAT + EXPANSION_FACTOR * lat_height / 2

    with rasterio.open(dem_path) as src:
        src_nodata = src.nodata
        window = from_bounds(
            exp_lon_min, exp_lat_min, exp_lon_max, exp_lat_max, src.transform
        )
        dem_data = src.read(1, window=window)
        expanded_transform = src.window_transform(window)

    # Gridcell extraction indices
    col_min = int((gc_lon_min - expanded_transform.c) / expanded_transform.a)
    col_max = int((gc_lon_max - expanded_transform.c) / expanded_transform.a)
    row_min = int((gc_lat_max - expanded_transform.f) / expanded_transform.e)
    row_max = int((gc_lat_min - expanded_transform.f) / expanded_transform.e)
    col_min = max(0, col_min)
    col_max = min(dem_data.shape[1], col_max)
    row_min = max(0, row_min)
    row_max = min(dem_data.shape[0], row_max)
    gc_row_slice = slice(row_min, row_max)
    gc_col_slice = slice(col_min, col_max)

    nodata_val = src_nodata if src_nodata is not None else -9999

    print(f"  Expanded DEM: {dem_data.shape}")
    print(f"  Gridcell extraction: rows={gc_row_slice}, cols={gc_col_slice}")

    # --- Basin detection (pre-pysheds) ---
    print("  Detecting basins in raw DEM...")
    basin_pre_mask = identify_basins(dem_data, nodata=nodata_val)
    n_basin_pre = int(np.sum(basin_pre_mask > 0))
    print(f"    Pre-conditioning basin pixels: {n_basin_pre}")
    dem_data[basin_pre_mask > 0] = nodata_val

    # --- pysheds Grid creation ---
    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=expanded_transform,
        crs=PyprojProj("EPSG:4326"),
        nodata=nodata_val,
    )

    # --- Standard conditioning ---
    print("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")

    # --- Slope/aspect on original DEM (for water detection + final output) ---
    print("  Computing slope/aspect...")
    grid.slope_aspect("dem")
    slope = np.array(grid.slope)
    aspect = np.array(grid.aspect)

    # --- Open water detection ---
    print("  Detecting open water from slope field...")
    basin_boundary, basin_mask = identify_open_water(slope)
    n_water = int(np.sum(basin_mask > 0))
    n_boundary = int(np.sum(basin_boundary > 0))
    print(f"    Open water pixels: {n_water}, boundary pixels: {n_boundary}")

    # --- Lower basin pixels in flooded DEM to force flow through them ---
    flooded_arr = np.array(grid.flooded)
    flooded_arr[basin_mask > 0] -= 0.1
    grid.add_gridded_data(
        flooded_arr,
        data_name="flooded",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )

    # --- Resolve flats (on modified flooded DEM) ---
    try:
        grid.resolve_flats("flooded", out_name="inflated")
    except ValueError:
        print("  WARNING: resolve_flats failed, using flooded DEM as inflated")
        grid.add_gridded_data(
            np.array(grid.flooded),
            data_name="inflated",
            affine=grid.affine,
            crs=grid.crs,
            nodata=grid.nodata,
        )

    # --- Flow direction ---
    print("  Computing flow direction + accumulation...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")

    # --- Re-mask basins after flowdir (before accumulation) ---
    inflated_arr = np.array(grid.inflated)
    inflated_arr[basin_mask > 0] = np.nan
    grid.add_gridded_data(
        inflated_arr,
        data_name="inflated",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )

    # --- Accumulation ---
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    # --- Force basin boundaries into stream network ---
    acc_arr = np.array(grid.acc)
    acc_arr[basin_boundary > 0] = accum_threshold + 1
    grid.add_gridded_data(
        acc_arr,
        data_name="acc",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )

    # --- A_thresh safety valve ---
    if np.nanmax(acc_arr) < accum_threshold:
        old_thresh = accum_threshold
        accum_threshold = int(np.nanmax(acc_arr) / 100)
        print(f"    A_thresh reduced: {old_thresh} -> {accum_threshold}")

    # --- Stream network + HAND/DTND (using modified acc) ---
    acc_mask = (grid.acc > accum_threshold) & np.isfinite(np.array(grid.inflated))
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

    print("  Computing HAND/DTND...")
    grid.compute_hand(
        "fdir",
        "inflated",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    hand = grid.hand
    dtnd = grid.dtnd

    # --- Extract gridcell ---
    nrows, ncols = np.array(grid.dem).shape
    transform = grid.affine
    lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
    lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    hand_gc = np.array(hand)[gc_row_slice, gc_col_slice]
    dtnd_gc = np.array(dtnd)[gc_row_slice, gc_col_slice]
    slope_gc = slope[gc_row_slice, gc_col_slice]
    aspect_gc = aspect[gc_row_slice, gc_col_slice]
    lon_gc = lon[gc_col_slice]
    lat_gc = lat[gc_row_slice]

    pixel_areas = compute_pixel_areas(lon_gc, lat_gc)
    res_m = np.abs(lat_gc[0] - lat_gc[1]) * RE * np.pi / 180

    # --- Flood filter: mark basin pixels with low HAND as invalid ---
    basin_mask_gc = basin_mask[gc_row_slice, gc_col_slice].flatten()
    n_flooded_gc = int(np.sum(basin_mask_gc > 0))
    if n_flooded_gc > 0:
        hand_gc_flat_temp = hand_gc.flatten()
        for ft in np.linspace(0, 20, 50):
            flooded_low_hand = (basin_mask_gc > ft) & (hand_gc_flat_temp < 2.0)
            if np.sum(flooded_low_hand) / n_flooded_gc < 0.95:
                hand_gc_flat_temp[flooded_low_hand] = -1
                print(
                    f"    Flood filter: threshold={ft:.2f}, "
                    f"marked {int(np.sum(flooded_low_hand))} pixels invalid"
                )
                hand_gc = hand_gc_flat_temp.reshape(hand_gc.shape)
                break

    # --- Flatten ---
    hand_flat = hand_gc.flatten()
    dtnd_flat = dtnd_gc.flatten()
    slope_flat = slope_gc.flatten()
    aspect_flat = aspect_gc.flatten()
    area_flat = pixel_areas.flatten()
    valid = (hand_flat >= 0) & np.isfinite(hand_flat)

    # HAND bins
    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )

    # Compute 16 elements
    elements = []
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        asp_mask = get_aspect_mask(aspect_flat, asp_bin) & valid
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            for _ in range(N_HAND_BINS):
                elements.append(
                    {
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
            len(
                np.unique(
                    grid.drainage_id.flatten()[asp_indices]
                    if hasattr(grid, "drainage_id")
                    else [1]
                )
            ),
            1,
        )

        # Trapezoidal fit
        trap = fit_trapezoidal_width(
            dtnd_flat[asp_indices], area_flat[asp_indices], n_hillslopes, min_dtnd=res_m
        )
        trap_slope = trap["slope"]
        trap_width = trap["width"]
        trap_area = trap["area"]

        # First pass: raw areas per bin
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

            width = max(float(width), 1)

            # Distance: trapezoid-derived midpoint distance
            # (Swenson representative_hillslope.py:847-859)
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

    return {"elements": elements}


# ---------------------------------------------------------------------------
# Comparison to published data (from stages 4+5)
# ---------------------------------------------------------------------------
def compute_circular_correlation(our_rad: np.ndarray, pub_rad: np.ndarray) -> float:
    """Compute circular correlation for aspect (radians)."""
    valid = np.isfinite(our_rad) & np.isfinite(pub_rad)
    our_v = our_rad[valid]
    pub_v = pub_rad[valid]
    if len(our_v) == 0:
        return float("nan")

    our_sin, our_cos = np.sin(our_v), np.cos(our_v)
    pub_sin, pub_cos = np.sin(pub_v), np.cos(pub_v)

    sin_corr = (
        np.corrcoef(our_sin, pub_sin)[0, 1]
        if np.std(our_sin) > 0 and np.std(pub_sin) > 0
        else 1.0
    )
    cos_corr = (
        np.corrcoef(our_cos, pub_cos)[0, 1]
        if np.std(our_cos) > 0 and np.std(pub_cos) > 0
        else 1.0
    )
    return float((sin_corr + cos_corr) / 2)


def compare_to_published(our_params: dict, published_path: str) -> dict:
    """
    Load published data, extract matching gridcell, compute correlations.

    Returns dict of {param_name: correlation}.
    """
    ds = xr.open_dataset(published_path)

    # Find matching gridcell (published uses 0-360 longitude)
    target_lon_360 = TARGET_CENTER_LON + 360  # -92.5 -> 267.5
    longxy = ds["LONGXY"].values
    latixy = ds["LATIXY"].values
    dist = np.sqrt((longxy - target_lon_360) ** 2 + (latixy - TARGET_CENTER_LAT) ** 2)
    lat_idx, lon_idx = np.unravel_index(dist.argmin(), dist.shape)

    print(
        f"  Published gridcell: lon={longxy[lat_idx, lon_idx]:.2f}, "
        f"lat={latixy[lat_idx, lon_idx]:.2f}"
    )

    # Extract published 16-element vectors
    pub = {
        "height": ds["hillslope_elevation"].values[:, lat_idx, lon_idx],
        "distance": ds["hillslope_distance"].values[:, lat_idx, lon_idx],
        "area": ds["hillslope_area"].values[:, lat_idx, lon_idx],
        "slope": ds["hillslope_slope"].values[:, lat_idx, lon_idx],
        "aspect": ds["hillslope_aspect"].values[:, lat_idx, lon_idx],
        "width": ds["hillslope_width"].values[:, lat_idx, lon_idx],
    }
    ds.close()

    # Extract our 16-element vectors
    our = {name: np.array([e[name] for e in our_params["elements"]]) for name in pub}

    correlations = {}

    # Standard Pearson for height, distance, slope, width
    for name in ("height", "distance", "slope", "width"):
        our_v = our[name]
        pub_v = pub[name]
        valid = (our_v > 0) & (pub_v > 0) & np.isfinite(our_v) & np.isfinite(pub_v)
        if np.sum(valid) > 1:
            correlations[name] = float(np.corrcoef(our_v[valid], pub_v[valid])[0, 1])
        else:
            correlations[name] = float("nan")

    # Area: compare as fractions (different absolute units)
    our_area_frac = (
        our["area"] / our["area"].sum() if our["area"].sum() > 0 else our["area"]
    )
    pub_area_frac = (
        pub["area"] / pub["area"].sum() if pub["area"].sum() > 0 else pub["area"]
    )
    valid = np.isfinite(our_area_frac) & np.isfinite(pub_area_frac)
    if np.sum(valid) > 1:
        correlations["area_fraction"] = float(
            np.corrcoef(our_area_frac[valid], pub_area_frac[valid])[0, 1]
        )
    else:
        correlations["area_fraction"] = float("nan")

    # Aspect: circular correlation in radians
    our_aspect_rad = our["aspect"] * np.pi / 180
    pub_aspect_rad = pub["aspect"]  # already radians in published data
    correlations["aspect"] = compute_circular_correlation(
        our_aspect_rad, pub_aspect_rad
    )

    return correlations


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def get_pysheds_commit() -> str:
    """Get short commit hash of pysheds fork."""
    pysheds_fork = os.environ.get(
        "PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork"
    )
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pysheds_fork,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_results(lc_results: dict, correlations: dict, output_dir: str) -> dict:
    """Write results.json and summary.txt. Returns the full results dict."""
    os.makedirs(output_dir, exist_ok=True)

    pysheds_commit = get_pysheds_commit()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Lc pass/fail
    lc_delta_pct = abs(lc_results["median_lc_m"] - EXPECTED_LC_M) / EXPECTED_LC_M * 100
    lc_pass = lc_delta_pct <= LC_TOLERANCE_PCT

    # Parameter pass/fail
    params_results = {}
    any_fail = not lc_pass
    for name, expected in EXPECTED.items():
        actual = correlations.get(name, float("nan"))
        delta = actual - expected
        status = "PASS" if abs(delta) <= TOLERANCE else "FAIL"
        if status == "FAIL":
            any_fail = True
        params_results[name] = {
            "expected": expected,
            "actual": round(actual, 4),
            "delta": round(delta, 4),
            "status": status,
        }

    overall = "FAIL" if any_fail else "PASS"

    results = {
        "timestamp": timestamp,
        "pysheds_commit": pysheds_commit,
        "lc": {
            "regions": lc_results["regions"],
            "median_lc_m": lc_results["median_lc_m"],
            "expected_lc_m": EXPECTED_LC_M,
            "tolerance_pct": LC_TOLERANCE_PCT,
            "status": "PASS" if lc_pass else "FAIL",
        },
        "parameters": params_results,
        "result": overall,
    }

    # Write JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Write summary text
    summary_lines = [
        "MERIT Geographic Regression Test",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"pysheds commit: {pysheds_commit}",
        "",
        "Characteristic length scale (Lc):",
    ]
    for r in lc_results["regions"]:
        summary_lines.append(
            f"  {r['label']:<20s} {r['lc_m']:>4.0f} m ({r['lc_px']:.1f} px, {r['model']})"
        )
    summary_lines.append(
        f"  Median: {lc_results['median_lc_m']:.1f} m "
        f"(expected: {EXPECTED_LC_M:.1f} m, tolerance: {LC_TOLERANCE_PCT:.1f}%)  "
        f"{'PASS' if lc_pass else 'FAIL'}"
    )
    summary_lines.append("")
    summary_lines.append(
        f"{'Parameter':<22s} {'Expected':>10s} {'Actual':>10s} {'Delta':>10s}  Status"
    )
    summary_lines.append("-" * 62)

    display_names = {
        "height": "Height (HAND)",
        "distance": "Distance (DTND)",
        "slope": "Slope",
        "aspect": "Aspect (circular)",
        "width": "Width",
        "area_fraction": "Area fraction",
    }
    for name in EXPECTED:
        pr = params_results[name]
        summary_lines.append(
            f"{display_names[name]:<22s} {pr['expected']:>10.4f} {pr['actual']:>10.4f} "
            f"{pr['delta']:>+10.4f}  {pr['status']}"
        )

    summary_lines.append("")
    summary_lines.append(f"RESULT: {overall}")

    summary_text = "\n".join(summary_lines)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    # Print same summary to stdout
    print(summary_text)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

    # Verify inputs
    for path, label in [(MERIT_DEM, "MERIT DEM"), (PUBLISHED_NC, "Published data")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # Step 1: Compute Lc
    print("=== Step 1: Characteristic Length Scale (FFT) ===")
    t0 = time.time()
    lc_results = compute_lc(MERIT_DEM)
    print(f"  Median Lc: {lc_results['median_lc_m']:.0f} m")
    for r in lc_results["regions"]:
        print(f"    {r['label']}: {r['lc_m']:.0f} m ({r['model']})")
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 2: Compute hillslope params
    print("\n=== Step 2: Hillslope Parameters ===")
    t0 = time.time()
    accum_threshold = int(0.5 * (lc_results["median_lc_m"] / 92.3) ** 2)
    # 92.3m is approximate MERIT pixel size at this latitude
    # But we need to use the pixel-based threshold from Lc_px
    # Use the median Lc in pixels to compute threshold
    lc_px_values = [r["lc_px"] for r in lc_results["regions"]]
    median_lc_px = float(np.median(lc_px_values))
    accum_threshold = int(0.5 * median_lc_px**2)
    print(f"  Lc = {median_lc_px:.1f} px, A_thresh = {accum_threshold} cells")
    hillslope_params = compute_hillslope_params(MERIT_DEM, accum_threshold)
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 3: Compare to published
    print("\n=== Step 3: Compare to Published Data ===")
    t0 = time.time()
    correlations = compare_to_published(hillslope_params, PUBLISHED_NC)
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 4: Write summary + PASS/FAIL
    print("\n=== Results ===")
    results = write_results(lc_results, correlations, OUTPUT_DIR)

    total = time.time() - start_time
    print(f"\nTotal time: {total:.0f}s ({total / 60:.1f} min)")

    sys.exit(0 if results["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
