#!/usr/bin/env python
"""
Area Fraction Diagnostic Investigation

Investigates why the MERIT area fraction correlation is only 0.8157 by testing
four pixel-filtering steps present in Swenson's code but missing from ours:

  A. DTND tail removal (TailIndex exponential fit)
  B. DTND minimum clipping (< 1.0 m → 1.0 m)
  C. Flooded region handling (HAND = -1 for heavily-flooded low-HAND pixels)
  D. Mean-HAND ≤ 0 bin skip (skip bins where mean HAND ≤ 0)
  E. All filters combined

Also prints detailed intermediates and a sensitivity map showing which of the
16 elements drives the correlation gap.

Expected runtime: ~50-90 min, 48GB RAM (same as merit_regression).
"""

import os
import sys
import time

import numpy as np
import rasterio
import xarray as xr
from pyproj import Proj as PyprojProj
from pysheds.pgrid import Grid
from rasterio.windows import from_bounds
from scipy.stats import expon

# Add parent directory for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_scale import DTR, RE, identify_spatial_scale_laplacian_dem

# ---------------------------------------------------------------------------
# Constants (shared with merit_regression.py)
# ---------------------------------------------------------------------------
SWENSON_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
MERIT_DEM = os.path.join(SWENSON_DIR, "data/merit/n30w095_dem.tif")
PUBLISHED_NC = os.path.join(
    SWENSON_DIR, "data/reference/hillslopes_0.9x1.25_c240416.nc"
)

# Target gridcell (0.9 x 1.25 degree)
TARGET_LON = (-93.125, -91.875)
TARGET_LAT = (32.0419, 32.9843)
TARGET_CENTER_LON = -92.5000
TARGET_CENTER_LAT = 32.5131

# FFT regions
FFT_REGION_SIZES = [500, 1000, 3000]
MAX_HILLSLOPE_LENGTH = 10000
NLAMBDA = 30

# Flow routing
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
EXPANSION_FACTOR = 1.5

# Binning
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0
ASPECT_BINS = [(315, 45), (45, 135), (135, 225), (225, 315)]
ASPECT_NAMES = ["North", "East", "South", "West"]


# ---------------------------------------------------------------------------
# Swenson open water mask (ported from geospatial_utils.py)
# ---------------------------------------------------------------------------
def _four_point_laplacian(mask):
    """Discrete Laplacian on a 2D binary mask."""
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


def _inside_indices_buffer(data, buf=1, mask=None):
    """Return indices of non-edge points (outside of buf)."""
    if mask is None:
        mask = np.array([]).astype(int)
    a = np.arange(data.size)
    offset = int(data.shape[1])

    top = []
    for i in range(buf):
        top.extend((i * offset + np.arange(offset)[buf:-buf]).tolist())
    top = np.array(top, dtype=int)

    bottom = data.size - 1 - top

    left = []
    for i in range(buf):
        left.extend(np.arange(i, data.size, offset))
    left = np.array(left, dtype=int)

    right = data.size - 1 - left

    exclude = np.unique(np.concatenate([top, left, right, bottom, mask]))
    inside = np.delete(a, exclude)

    return inside


def _expand_mask_buffer(mask, buf=1):
    """Expand a binary mask by buf pixels."""
    omask = np.copy(mask)
    inside = _inside_indices_buffer(mask, buf=buf)
    lmask = np.where(_four_point_laplacian(mask) > 0, 1, 0)
    ind = inside[(lmask.flat[inside] > 0)]
    offset = mask.shape[1]
    for k in range(-buf, buf + 1):
        if k != 0:
            omask.flat[ind + k] = 1
        for j in range(buf):
            j1 = j + 1
            omask.flat[ind + k + j1 * offset] = 1
            omask.flat[ind + k - j1 * offset] = 1

    return omask


def erode_dilate_mask(mask, buf=1, niter=10):
    """Morphological erosion then dilation of a binary mask."""
    x = np.copy(mask)
    for _ in range(niter):
        x = 1 - _expand_mask_buffer(1 - x, buf=buf)
    for _ in range(niter + 1):
        x = _expand_mask_buffer(x, buf=buf)
    return np.where(np.logical_and(x > 0, mask > 0), 1, 0)


def identify_open_water(slope, max_slope=1e-4, niter=15):
    """Create open water mask from slope field. Returns [basin_boundary, basin_mask]."""
    basin_mask = erode_dilate_mask(np.where(slope < max_slope, 1, 0), niter=niter)
    sup_basin_mask = _expand_mask_buffer(basin_mask, buf=2)
    basin_boundary = sup_basin_mask - basin_mask
    return [basin_boundary, basin_mask]


def identify_basins(dem, basin_thresh=0.25, niter=10, buf=1, nodata=None):
    """
    Port of Swenson's identify_basins (representative_hillslope.py:263-296).

    Creates a basin mask (1 = basin, 0 = outside) for large flat regions.
    Flat areas produce spurious HAND/DTND values due to flow paths in the
    flooded/inflated DEM. Masking them before pysheds conditioning prevents this.

    Finds elevation values that occupy > basin_thresh fraction of the grid,
    marks those pixels, then iteratively expands and prunes the mask.
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
            for i in ind:
                eps = 1e-2
                if np.abs(udem[i]) < eps:
                    eps = 1e-6
                imask[_four_point_laplacian(1 - imask) >= 3] = 0

    return imask


# ---------------------------------------------------------------------------
# Lc computation (same as merit_regression.py)
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
    """Run FFT on 3 center regions + full tile (4x sub + native). Returns median Lc in pixels."""
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
        regions.append(
            {
                "lc_px": float(result["spatialScale"]),
                "res": float(result["res"]),
                "label": f"center_{size}",
            }
        )

    # Full tile (4x subsample)
    data = load_dem_with_coords(dem_path, subsample=4)
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
    regions.append(
        {
            "lc_px": float(result["spatialScale"]) * 4,
            "res": float(result["res"]) / 4,
            "label": "full_4x",
        }
    )

    # Full tile (native resolution — no subsampling)
    data = load_dem_with_coords(dem_path)
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
    regions.append(
        {
            "lc_px": float(result["spatialScale"]),
            "res": float(result["res"]),
            "label": "full_native",
        }
    )

    median_lc_px = float(np.median([r["lc_px"] for r in regions]))
    return {"median_lc_px": median_lc_px, "regions": regions}


# ---------------------------------------------------------------------------
# Flow routing: setup (A_thresh-independent) + full routing
# ---------------------------------------------------------------------------
def setup_flow_routing(dem_path: str) -> dict:
    """
    Load DEM, condition, compute fdir + acc + slope/aspect.

    Returns the live grid object and A_thresh-independent arrays for the
    gridcell extent. Channel mask, HAND/DTND, and drainage_id are
    A_thresh-dependent and must be computed by the caller.
    """
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

    print(f"  Expanded DEM: {dem_data.shape}")
    print(f"  Gridcell extraction: rows={gc_row_slice}, cols={gc_col_slice}")

    # pysheds flow routing
    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=expanded_transform,
        crs=PyprojProj("EPSG:4326"),
        nodata=src_nodata if src_nodata is not None else -9999,
    )

    print("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")
    grid.resolve_flats("flooded", out_name="inflated")

    print("  Computing flow direction + accumulation...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    # Slope/aspect using pgrid's Horn 1981 (A_thresh-independent)
    print("  Computing slope/aspect...")
    grid.slope_aspect("dem")

    # Build coordinate arrays
    nrows, ncols = np.array(grid.dem).shape
    transform = grid.affine
    lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
    lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    # Extract A_thresh-independent gridcell arrays
    slope_gc = np.array(grid.slope)[gc_row_slice, gc_col_slice]
    aspect_gc = np.array(grid.aspect)[gc_row_slice, gc_col_slice]
    lon_gc = lon[gc_col_slice]
    lat_gc = lat[gc_row_slice]

    # Flood field: slope-based open water mask (Swenson's identify_open_water)
    print("  Computing open water mask (identify_open_water)...")
    slope_full = np.array(grid.slope)
    _, basin_mask = identify_open_water(slope_full)
    fflood_gc = basin_mask[gc_row_slice, gc_col_slice]

    # Pixel areas
    phi = DTR * lon_gc
    theta = DTR * (90.0 - lat_gc)
    dphi = np.abs(phi[1] - phi[0])
    dtheta = np.abs(theta[0] - theta[1])
    sin_theta = np.sin(theta)
    pixel_areas = np.tile(sin_theta.reshape(-1, 1), (1, len(lon_gc)))
    pixel_areas = pixel_areas * dtheta * dphi * RE**2

    res_m = np.abs(lat_gc[0] - lat_gc[1]) * RE * np.pi / 180

    return {
        "grid": grid,
        "gc_row_slice": gc_row_slice,
        "gc_col_slice": gc_col_slice,
        "slope_gc": slope_gc,
        "aspect_gc": aspect_gc,
        "pixel_areas": pixel_areas,
        "fflood_gc": fflood_gc,
        "res_m": res_m,
    }


def run_flow_routing(dem_path: str, accum_threshold: int) -> dict:
    """
    Run DEM conditioning + flow routing + HAND/DTND + slope/aspect.

    Returns all intermediate arrays needed for diagnostic tests, including
    the raw and conditioned DEMs for flood detection.
    """
    setup = setup_flow_routing(dem_path)
    grid = setup["grid"]
    gc_row_slice = setup["gc_row_slice"]
    gc_col_slice = setup["gc_col_slice"]

    # Stream network + HAND/DTND (A_thresh-dependent)
    acc_mask = grid.acc > accum_threshold
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

    print("  Computing HAND/DTND...")
    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )

    # Extract A_thresh-dependent gridcell arrays
    hand_gc = np.array(grid.hand)[gc_row_slice, gc_col_slice]
    dtnd_gc = np.array(grid.dtnd)[gc_row_slice, gc_col_slice]
    drainage_id_gc = np.array(grid.drainage_id)[gc_row_slice, gc_col_slice]

    return {
        "hand": hand_gc.flatten(),
        "dtnd": dtnd_gc.flatten(),
        "slope": setup["slope_gc"].flatten(),
        "aspect": setup["aspect_gc"].flatten(),
        "area": setup["pixel_areas"].flatten(),
        "fflood": setup["fflood_gc"].flatten(),
        "drainage_id": drainage_id_gc.flatten(),
        "res_m": setup["res_m"],
    }


# ---------------------------------------------------------------------------
# Flow routing with full Swenson DEM conditioning (Test N)
# ---------------------------------------------------------------------------
def run_flow_routing_swenson(dem_path: str, accum_threshold: int) -> dict:
    """
    Run Swenson's full DEM conditioning + flow routing + HAND/DTND + slope/aspect.

    Adds 5 missing steps from Swenson's CalcLandscapeCharacteristicsPysheds
    (representative_hillslope.py:1457-1754):

      1. identify_basins() — mask large flat regions as nodata before pysheds
      2. identify_open_water(slope) — find low-slope regions after fill_depressions
      3. Lower flooded areas by 0.1m before resolve_flats
      4. Re-mask basin pixels after flowdir, before accumulation
      5. Force basin boundary pixels into stream network

    Returns same dict structure as run_flow_routing() for direct comparison,
    plus extra diagnostic fields about the conditioning.
    """
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

    fill_value = src_nodata if src_nodata is not None else -9999

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

    print(f"  Expanded DEM: {dem_data.shape}")
    print(f"  Gridcell extraction: rows={gc_row_slice}, cols={gc_col_slice}")

    # --- Step 1: Pre-conditioning basin identification (Swenson lines 1514-1519) ---
    print("  [Swenson step 1] Identifying basins (flat regions)...")
    basin_mask_pre = identify_basins(dem_data, nodata=fill_value)
    n_basin_pre = int(np.sum(basin_mask_pre > 0))
    print(
        f"    Masked {n_basin_pre} pixels "
        f"({100 * n_basin_pre / dem_data.size:.2f}%) as nodata"
    )
    dem_data[basin_mask_pre > 0] = fill_value

    # pysheds grid setup
    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=expanded_transform,
        crs=PyprojProj("EPSG:4326"),
        nodata=fill_value,
    )

    # Standard conditioning
    print("  Conditioning DEM (fill_pits → fill_depressions)...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")

    # --- Steps 2-3: Open water identification + depression lowering ---
    print("  [Swenson step 2-3] Open water detection + 0.1m lowering...")
    grid.slope_aspect("dem")
    slope_full = np.array(grid.slope)
    basin_boundary, basin_mask_water = identify_open_water(slope_full)
    n_water = int(np.sum(basin_mask_water > 0))
    n_boundary = int(np.sum(basin_boundary > 0))
    print(f"    Open water pixels: {n_water}, boundary pixels: {n_boundary}")

    # Lower flooded DEM at water pixels by 0.1m (ensures flow drains away)
    grid.flooded[basin_mask_water > 0] -= 0.1

    print("  Resolving flats...")
    grid.resolve_flats("flooded", out_name="inflated")

    print("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")

    # --- Step 4: Re-mask basin pixels after flowdir, before accumulation ---
    print("  [Swenson step 4] Re-masking basin pixels from conditioned DEMs...")
    grid.flooded[basin_mask_water > 0] = fill_value
    grid.inflated[basin_mask_water > 0] = fill_value

    print("  Computing accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    # --- Step 5: Force basin boundary into stream network ---
    print("  [Swenson step 5] Forcing basin boundary into stream network...")
    n_stream_before = int(np.sum(grid.acc > accum_threshold))
    grid.acc[basin_boundary > 0] = accum_threshold + 1
    n_stream_after = int(np.sum(grid.acc > accum_threshold))
    n_stream_added = n_stream_after - n_stream_before
    print(
        f"    Stream pixels: {n_stream_before} before → {n_stream_after} after "
        f"(+{n_stream_added} from basin boundary)"
    )

    # Stream network + HAND/DTND
    acc_mask = grid.acc > accum_threshold

    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

    print("  Computing HAND/DTND (using inflated DEM)...")
    grid.compute_hand(
        "fdir",
        "inflated",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )

    # Slope/aspect from original (basin-masked) DEM — already computed above
    # grid.slope and grid.aspect are already set from the slope_aspect("dem") call

    # Build coordinate arrays
    nrows, ncols = np.array(grid.dem).shape
    transform = grid.affine
    lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
    lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    # Extract gridcell arrays
    hand_gc = np.array(grid.hand)[gc_row_slice, gc_col_slice]
    dtnd_gc = np.array(grid.dtnd)[gc_row_slice, gc_col_slice]
    slope_gc = np.array(grid.slope)[gc_row_slice, gc_col_slice]
    aspect_gc = np.array(grid.aspect)[gc_row_slice, gc_col_slice]
    lon_gc = lon[gc_col_slice]
    lat_gc = lat[gc_row_slice]

    # Flood field from open water mask (within gridcell)
    fflood_gc = basin_mask_water[gc_row_slice, gc_col_slice]

    # Pixel areas
    phi = DTR * lon_gc
    theta = DTR * (90.0 - lat_gc)
    dphi = np.abs(phi[1] - phi[0])
    dtheta = np.abs(theta[0] - theta[1])
    sin_theta = np.sin(theta)
    pixel_areas = np.tile(sin_theta.reshape(-1, 1), (1, len(lon_gc)))
    pixel_areas = pixel_areas * dtheta * dphi * RE**2

    res_m = np.abs(lat_gc[0] - lat_gc[1]) * RE * np.pi / 180

    # Drainage ID for n_hillslopes computation
    drainage_id_gc = np.array(grid.drainage_id)[gc_row_slice, gc_col_slice]

    return {
        "hand": hand_gc.flatten(),
        "dtnd": dtnd_gc.flatten(),
        "slope": slope_gc.flatten(),
        "aspect": aspect_gc.flatten(),
        "area": pixel_areas.flatten(),
        "fflood": fflood_gc.flatten(),
        "drainage_id": drainage_id_gc.flatten(),
        "res_m": res_m,
        # Extra diagnostics
        "n_basin_pre": n_basin_pre,
        "n_water": n_water,
        "n_boundary": n_boundary,
        "n_stream_before": n_stream_before,
        "n_stream_after": n_stream_after,
    }


# ---------------------------------------------------------------------------
# Swenson filters
# ---------------------------------------------------------------------------
def tail_index(fdtnd: np.ndarray, fhand: np.ndarray, hval: float = 0.05) -> np.ndarray:
    """
    Port of Swenson's TailIndex (terrain_utils.py:286-296).

    Returns indices of pixels to KEEP (tail removed).
    Uses exponential fit to the DTND distribution to find
    the cutoff where PDF drops below hval * max(PDF).
    """
    mask = fhand > 0
    if np.sum(mask) < 10:
        return np.arange(fdtnd.size)

    std_dtnd = np.std(fdtnd[mask])
    if std_dtnd == 0:
        return np.arange(fdtnd.size)

    normalized = fdtnd[mask] / std_dtnd
    fit_loc, fit_beta = expon.fit(normalized)
    rv = expon(loc=fit_loc, scale=fit_beta)

    pbins = np.linspace(0, np.max(fdtnd), 5000)
    rvpdf = rv.pdf(pbins / std_dtnd)
    r1 = np.argmin(np.abs(rvpdf - hval * np.max(rvpdf)))

    return np.where(fdtnd < pbins[r1])[0]


def apply_flood_filter(
    fhand: np.ndarray, fflood: np.ndarray, hand_threshold: float = 2.0
) -> np.ndarray:
    """
    Port of Swenson's flooded region handling (representative_hillslope.py:678-697).

    Sets HAND = -1 for heavily-flooded low-HAND pixels.
    Returns modified copy of fhand.
    """
    fhand_out = fhand.copy()
    num_flooded_pts = np.sum(np.abs(fflood[fhand < hand_threshold]) > 0)

    if num_flooded_pts == 0:
        return fhand_out

    # Find flood threshold: smallest value where < 95% of flooded pixels remain
    flood_thresh = 0
    for ft in np.linspace(0, 20, 50):
        ratio = np.sum(np.abs(fflood[fhand < hand_threshold]) > ft) / num_flooded_pts
        if ratio < 0.95:
            flood_thresh = ft
            break

    mask = (np.abs(fflood) > flood_thresh) & (fhand < hand_threshold)
    fhand_out[mask] = -1

    return fhand_out


# ---------------------------------------------------------------------------
# Hillslope parameter helpers (same as merit_regression.py)
# ---------------------------------------------------------------------------
def get_aspect_mask(aspect: np.ndarray, aspect_bin: tuple) -> np.ndarray:
    lower, upper = aspect_bin
    if lower > upper:
        return (aspect >= lower) | (aspect < upper)
    return (aspect >= lower) & (aspect < upper)


def compute_hand_bins(
    hand: np.ndarray,
    aspect: np.ndarray,
    aspect_bins: list,
    bin1_max: float | None = 2.0,
    min_aspect_fraction: float = 0.01,
) -> np.ndarray:
    """Compute HAND bin boundaries following Swenson's SpecifyHandBounds().

    If bin1_max is None, skip the forced branch and always use quartile bounds.
    """
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]

    if hand_valid.size == 0:
        fallback = bin1_max if bin1_max is not None else 2.0
        return np.array([0, fallback, fallback * 2, fallback * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size
    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    if bin1_max is not None and initial_q25 > bin1_max:
        for asp_low, asp_high in aspect_bins:
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


def _fit_polynomial_w1(x, y, ncoefs, weights=None):
    """
    Port of Swenson's _fit_polynomial (representative_hillslope.py:113-136).

    Weighted normal equations: (G^T W G) coefs = G^T W y
    where W = diag(weights). Minimizes sum_i w_i * (residual_i)^2.
    """
    g = np.column_stack([np.power(x, n) for n in range(ncoefs)])

    if weights is None:
        gtg = g.T @ g
        gtd = g.T @ y
    else:
        W = np.diag(weights)
        gtg = g.T @ W @ g
        gtd = g.T @ W @ y

    return np.linalg.solve(gtg, gtd)


def fit_trapezoidal_width(
    dtnd: np.ndarray,
    area: np.ndarray,
    n_hillslopes: int,
    min_dtnd: float = 90,
    n_bins: int = 10,
) -> dict:
    """Fit trapezoidal plan form following Swenson Eq. (4). Uses w^2 weighting (baseline)."""
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


def fit_trapezoidal_width_w1(
    dtnd: np.ndarray,
    area: np.ndarray,
    n_hillslopes: int,
    min_dtnd: float = 90,
    n_bins: int = 10,
) -> dict:
    """Fit trapezoidal plan form using Swenson's w^1 weighting (corrected)."""
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
        coeffs = _fit_polynomial_w1(d, A_cumsum, ncoefs=3, weights=A_cumsum)

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
    sin_sum = np.mean(np.sin(DTR * aspects))
    cos_sum = np.mean(np.cos(DTR * aspects))
    mean_aspect = np.arctan2(sin_sum, cos_sum) / DTR
    if mean_aspect < 0:
        mean_aspect += 360
    return mean_aspect


# ---------------------------------------------------------------------------
# Compute 16-element hillslope params from pre-filtered arrays
# ---------------------------------------------------------------------------
def compute_elements(
    hand_flat: np.ndarray,
    dtnd_flat: np.ndarray,
    slope_flat: np.ndarray,
    aspect_flat: np.ndarray,
    area_flat: np.ndarray,
    drainage_id_flat: np.ndarray,
    res_m: float,
    skip_zero_hand_bins: bool = False,
    swenson_style: bool = False,
    use_w1_fit: bool = False,
    bin1_max_override: float | None = LOWEST_BIN_MAX,
) -> dict:
    """
    Compute 16 hillslope elements from flat arrays.

    If skip_zero_hand_bins=True, bins where mean(HAND) <= 0 get zero area
    (matching Swenson line 819: `if np.mean(fhand[cind]) <= 0: continue`).

    If swenson_style=True, the aspect population includes negative-HAND pixels
    (matching Swenson line 653: only NaN removed). This affects n_hillslopes,
    the trapezoidal fit, and area fraction denominators. Bins still only
    contain HAND >= 0 pixels (bin bounds start at 0).

    If use_w1_fit=True, uses Swenson's w^1 polynomial weighting instead of
    the baseline w^2 lstsq weighting.

    bin1_max_override controls the lowest bin upper bound. Default is
    LOWEST_BIN_MAX (2.0). Pass None for pure quartile binning.

    Returns dict with 'elements' list and diagnostic intermediates.
    """
    valid = (hand_flat >= 0) & np.isfinite(hand_flat)

    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=bin1_max_override
    )

    elements = []
    diag = {
        "hand_bounds": hand_bounds.tolist(),
        "aspect_pixel_counts": [],
        "aspect_pop_counts": [],
        "neg_hand_counts": [],
        "n_hillslopes": [],
        "trap_params": [],
        "bin_raw_areas": [],
        "bin_pixel_counts": [],
        "skipped_bins": [],
    }

    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        if swenson_style:
            # Swenson: include negative-HAND in aspect population
            asp_mask = get_aspect_mask(aspect_flat, asp_bin) & np.isfinite(hand_flat)
        else:
            # Current: exclude negative-HAND from everything
            asp_mask = get_aspect_mask(aspect_flat, asp_bin) & valid
        asp_indices = np.where(asp_mask)[0]

        # Count negative-HAND pixels in this aspect (for diagnostics)
        asp_geo_mask = get_aspect_mask(aspect_flat, asp_bin) & np.isfinite(hand_flat)
        neg_hand_in_asp = int(np.sum((hand_flat[asp_geo_mask] < 0)))
        diag["neg_hand_counts"].append(neg_hand_in_asp)
        diag["aspect_pop_counts"].append(int(np.sum(asp_geo_mask)))
        diag["aspect_pixel_counts"].append(int(len(asp_indices)))

        if len(asp_indices) == 0:
            diag["n_hillslopes"].append(0)
            diag["trap_params"].append(None)
            diag["bin_raw_areas"].append([0] * N_HAND_BINS)
            diag["bin_pixel_counts"].append([0] * N_HAND_BINS)
            diag["skipped_bins"].append([False] * N_HAND_BINS)
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
            len(np.unique(drainage_id_flat[asp_indices])),
            1,
        )
        diag["n_hillslopes"].append(n_hillslopes)

        # Trapezoidal fit
        fit_fn = fit_trapezoidal_width_w1 if use_w1_fit else fit_trapezoidal_width
        trap = fit_fn(
            dtnd_flat[asp_indices],
            area_flat[asp_indices],
            n_hillslopes,
            min_dtnd=res_m,
        )
        trap_slope = trap["slope"]
        trap_width = trap["width"]
        trap_area = trap["area"]
        diag["trap_params"].append(
            {"slope": trap_slope, "width": trap_width, "area": trap_area}
        )

        # First pass: raw areas per bin
        bin_raw_areas = []
        bin_pixel_counts = []
        bin_data = []
        skipped = []
        for h_idx in range(N_HAND_BINS):
            h_lower = hand_bounds[h_idx]
            h_upper = hand_bounds[h_idx + 1]
            hand_mask = (hand_flat >= h_lower) & (hand_flat < h_upper)
            bin_mask = asp_mask & hand_mask
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) == 0:
                bin_raw_areas.append(0)
                bin_pixel_counts.append(0)
                skipped.append(False)
                bin_data.append(
                    {"indices": None, "h_lower": h_lower, "h_upper": h_upper}
                )
            elif skip_zero_hand_bins and np.mean(hand_flat[bin_indices]) <= 0:
                # Swenson line 819: skip bins where mean HAND <= 0
                bin_raw_areas.append(0)
                bin_pixel_counts.append(int(len(bin_indices)))
                skipped.append(True)
                bin_data.append(
                    {"indices": None, "h_lower": h_lower, "h_upper": h_upper}
                )
            else:
                bin_raw_areas.append(float(np.sum(area_flat[bin_indices])))
                bin_pixel_counts.append(int(len(bin_indices)))
                skipped.append(False)
                bin_data.append(
                    {"indices": bin_indices, "h_lower": h_lower, "h_upper": h_upper}
                )

        diag["bin_raw_areas"].append(bin_raw_areas)
        diag["bin_pixel_counts"].append(bin_pixel_counts)
        diag["skipped_bins"].append(skipped)

        if swenson_style:
            # Swenson: denominator is full aspect population area (incl. neg HAND)
            total_raw = float(np.sum(area_flat[asp_indices]))
        else:
            # Current: denominator is sum of binned areas only
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

            # Width at lower edge of bin
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

    return {"elements": elements, "diagnostics": diag}


# ---------------------------------------------------------------------------
# Published data loading and comparison
# ---------------------------------------------------------------------------
def load_published() -> dict:
    """Load the 16-element published vectors for our target gridcell."""
    ds = xr.open_dataset(PUBLISHED_NC)

    target_lon_360 = TARGET_CENTER_LON + 360
    longxy = ds["LONGXY"].values
    latixy = ds["LATIXY"].values
    dist = np.sqrt((longxy - target_lon_360) ** 2 + (latixy - TARGET_CENTER_LAT) ** 2)
    lat_idx, lon_idx = np.unravel_index(dist.argmin(), dist.shape)

    pub = {
        "height": ds["hillslope_elevation"].values[:, lat_idx, lon_idx],
        "distance": ds["hillslope_distance"].values[:, lat_idx, lon_idx],
        "area": ds["hillslope_area"].values[:, lat_idx, lon_idx],
        "slope": ds["hillslope_slope"].values[:, lat_idx, lon_idx],
        "aspect": ds["hillslope_aspect"].values[:, lat_idx, lon_idx],
        "width": ds["hillslope_width"].values[:, lat_idx, lon_idx],
    }
    ds.close()
    return pub


def compute_area_fraction_corr(our_elements: list, pub: dict) -> float:
    """Compute area fraction correlation between our elements and published."""
    our_area = np.array([e["area"] for e in our_elements])
    pub_area = pub["area"]

    our_frac = our_area / our_area.sum() if our_area.sum() > 0 else our_area
    pub_frac = pub_area / pub_area.sum() if pub_area.sum() > 0 else pub_area

    valid = np.isfinite(our_frac) & np.isfinite(pub_frac)
    if np.sum(valid) > 1:
        return float(np.corrcoef(our_frac[valid], pub_frac[valid])[0, 1])
    return float("nan")


def compute_all_correlations(our_elements: list, pub: dict) -> dict:
    """Compute all 6 parameter correlations."""
    our = {name: np.array([e[name] for e in our_elements]) for name in pub}
    correlations = {}

    for name in ("height", "distance", "slope", "width"):
        our_v = our[name]
        pub_v = pub[name]
        valid = (our_v > 0) & (pub_v > 0) & np.isfinite(our_v) & np.isfinite(pub_v)
        if np.sum(valid) > 1:
            correlations[name] = float(np.corrcoef(our_v[valid], pub_v[valid])[0, 1])
        else:
            correlations[name] = float("nan")

    # Area fraction
    correlations["area_fraction"] = compute_area_fraction_corr(our_elements, pub)

    # Aspect: circular correlation
    our_asp_rad = our["aspect"] * np.pi / 180
    pub_asp_rad = pub["aspect"]  # already radians
    valid = np.isfinite(our_asp_rad) & np.isfinite(pub_asp_rad)
    our_v = our_asp_rad[valid]
    pub_v = pub_asp_rad[valid]
    if len(our_v) > 0:
        sin_corr = (
            np.corrcoef(np.sin(our_v), np.sin(pub_v))[0, 1]
            if np.std(np.sin(our_v)) > 0 and np.std(np.sin(pub_v)) > 0
            else 1.0
        )
        cos_corr = (
            np.corrcoef(np.cos(our_v), np.cos(pub_v))[0, 1]
            if np.std(np.cos(our_v)) > 0 and np.std(np.cos(pub_v)) > 0
            else 1.0
        )
        correlations["aspect"] = float((sin_corr + cos_corr) / 2)
    else:
        correlations["aspect"] = float("nan")

    return correlations


# ===========================================================================
# Part 1: Visibility — detailed intermediates
# ===========================================================================
def print_visibility(our_elements: list, pub: dict, diag: dict) -> None:
    """Print side-by-side comparison and intermediates."""
    our_area = np.array([e["area"] for e in our_elements])
    pub_area = pub["area"]
    our_frac = our_area / our_area.sum() if our_area.sum() > 0 else our_area
    pub_frac = pub_area / pub_area.sum() if pub_area.sum() > 0 else pub_area

    print("\n" + "=" * 78)
    print("PART 1: VISIBILITY — Side-by-side area fraction comparison")
    print("=" * 78)

    # 16-element table
    print(
        f"\n{'Elem':>4s}  {'Aspect':<6s}  {'Bin':>3s}  "
        f"{'Ours':>8s}  {'Published':>9s}  {'Delta':>8s}  {'Rank':>4s}"
    )
    print("-" * 60)

    deltas = np.abs(our_frac - pub_frac)
    rank_order = np.argsort(-deltas)  # largest delta first
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(1, len(rank_order) + 1)

    for i in range(16):
        asp_idx = i // N_HAND_BINS
        h_idx = i % N_HAND_BINS
        print(
            f"{i:>4d}  {ASPECT_NAMES[asp_idx]:<6s}  {h_idx:>3d}  "
            f"{our_frac[i]:>8.4f}  {pub_frac[i]:>9.4f}  "
            f"{our_frac[i] - pub_frac[i]:>+8.4f}  {ranks[i]:>4d}"
        )

    # Per-aspect sub-correlations
    print(f"\n{'Aspect':<8s}  {'Corr':>8s}  {'Our sum':>8s}  {'Pub sum':>8s}")
    print("-" * 40)
    for asp_idx, name in enumerate(ASPECT_NAMES):
        sl = slice(asp_idx * N_HAND_BINS, (asp_idx + 1) * N_HAND_BINS)
        o = our_frac[sl]
        p = pub_frac[sl]
        if np.std(o) > 0 and np.std(p) > 0:
            corr = float(np.corrcoef(o, p)[0, 1])
        else:
            corr = float("nan")
        print(f"{name:<8s}  {corr:>8.4f}  {np.sum(o):>8.4f}  {np.sum(p):>8.4f}")

    overall_corr = compute_area_fraction_corr(our_elements, pub)
    print(f"\nOverall area fraction correlation: {overall_corr:.4f}")

    # Intermediates
    print(f"\nHAND bin bounds: {diag['hand_bounds']}")

    print(
        f"\n{'Aspect':<8s}  {'Pixels':>8s}  {'n_hills':>7s}  "
        f"{'trap_a':>10s}  {'trap_w':>10s}  {'trap_s':>10s}"
    )
    print("-" * 65)
    for asp_idx, name in enumerate(ASPECT_NAMES):
        tp = diag["trap_params"][asp_idx]
        if tp is not None:
            print(
                f"{name:<8s}  {diag['aspect_pixel_counts'][asp_idx]:>8d}  "
                f"{diag['n_hillslopes'][asp_idx]:>7d}  "
                f"{tp['area']:>10.1f}  {tp['width']:>10.2f}  {tp['slope']:>10.6f}"
            )
        else:
            print(f"{name:<8s}  {diag['aspect_pixel_counts'][asp_idx]:>8d}  0")

    # Raw pixel counts per bin
    print(
        f"\n{'Elem':>4s}  {'Aspect':<6s}  {'Bin':>3s}  "
        f"{'Pixels':>8s}  {'Raw area':>12s}"
    )
    print("-" * 45)
    for asp_idx in range(N_ASPECT_BINS):
        for h_idx in range(N_HAND_BINS):
            i = asp_idx * N_HAND_BINS + h_idx
            print(
                f"{i:>4d}  {ASPECT_NAMES[asp_idx]:<6s}  {h_idx:>3d}  "
                f"{diag['bin_pixel_counts'][asp_idx][h_idx]:>8d}  "
                f"{diag['bin_raw_areas'][asp_idx][h_idx]:>12.1f}"
            )


# ===========================================================================
# Part 1b: Negative-HAND and aspect boundary diagnostics
# ===========================================================================
def print_negative_hand_diagnostics(arrays: dict) -> None:
    """Print negative-HAND pixel distribution and aspect boundary analysis."""
    hand = arrays["hand"]
    aspect = arrays["aspect"]
    area = arrays["area"]
    drainage_id = arrays["drainage_id"]

    finite_mask = np.isfinite(hand)
    neg_mask = (hand < 0) & finite_mask
    valid_mask = (hand >= 0) & finite_mask

    n_total = hand.size
    n_finite = int(np.sum(finite_mask))
    n_neg = int(np.sum(neg_mask))
    n_valid = int(np.sum(valid_mask))

    print("\n" + "=" * 78)
    print("PART 1b: NEGATIVE-HAND AND ASPECT BOUNDARY DIAGNOSTICS")
    print("=" * 78)

    # --- Negative-HAND distribution by aspect ---
    print(
        f"\nTotal pixels: {n_total}, finite: {n_finite}, "
        f"HAND<0: {n_neg} ({100 * n_neg / n_finite:.2f}%), "
        f"HAND>=0: {n_valid}"
    )

    print(
        f"\n{'Aspect':<8s}  {'HAND<0':>8s}  {'HAND>=0':>8s}  {'Total':>8s}  "
        f"{'%neg/asp':>8s}  {'%neg/all':>8s}  "
        f"{'Area HAND<0':>12s}  {'Area HAND>=0':>12s}"
    )
    print("-" * 100)

    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        asp_mask = get_aspect_mask(aspect, asp_bin) & finite_mask
        asp_neg = asp_mask & neg_mask
        asp_valid = asp_mask & valid_mask

        n_asp = int(np.sum(asp_mask))
        n_asp_neg = int(np.sum(asp_neg))
        n_asp_valid = int(np.sum(asp_valid))

        area_neg = float(np.sum(area[asp_neg])) if np.any(asp_neg) else 0.0
        area_valid = float(np.sum(area[asp_valid])) if np.any(asp_valid) else 0.0

        pct_neg_of_asp = 100 * n_asp_neg / n_asp if n_asp > 0 else 0
        pct_neg_of_all = 100 * n_asp_neg / n_neg if n_neg > 0 else 0

        print(
            f"{asp_name:<8s}  {n_asp_neg:>8d}  {n_asp_valid:>8d}  {n_asp:>8d}  "
            f"{pct_neg_of_asp:>7.2f}%  {pct_neg_of_all:>7.2f}%  "
            f"{area_neg:>12.1f}  {area_valid:>12.1f}"
        )

    # --- n_hillslopes comparison ---
    print(
        f"\n{'Aspect':<8s}  {'n_hills(>=0)':>13s}  {'n_hills(finite)':>15s}  {'Delta':>6s}"
    )
    print("-" * 50)
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        asp_valid_mask = get_aspect_mask(aspect, asp_bin) & valid_mask
        asp_finite_mask = get_aspect_mask(aspect, asp_bin) & finite_mask

        ids_valid = np.unique(drainage_id[asp_valid_mask])
        ids_finite = np.unique(drainage_id[asp_finite_mask])

        n_v = len(ids_valid)
        n_f = len(ids_finite)
        print(f"{asp_name:<8s}  {n_v:>13d}  {n_f:>15d}  {n_f - n_v:>+6d}")

    # --- Aspect histogram near bin boundaries ---
    boundaries = [45, 135, 225, 315]
    half_width = 2.0  # degrees

    print(f"\nAspect boundary analysis (±{half_width}° bands):")
    print(
        f"{'Boundary':>8s}  {'Lower asp':>9s}  {'Upper asp':>9s}  "
        f"{'In band':>8s}  {'Below':>8s}  {'Above':>8s}  {'Ratio B/A':>9s}"
    )
    print("-" * 70)

    for bdry in boundaries:
        band_lo = bdry - half_width
        band_hi = bdry + half_width

        # Handle wraparound for 315
        if band_lo < 0:
            in_band = ((aspect >= (360 + band_lo)) | (aspect < band_hi)) & finite_mask
            below = ((aspect >= (360 + band_lo)) & (aspect < bdry)) & finite_mask
            above = ((aspect >= bdry) & (aspect < band_hi)) & finite_mask
        else:
            in_band = ((aspect >= band_lo) & (aspect < band_hi)) & finite_mask
            below = ((aspect >= band_lo) & (aspect < bdry)) & finite_mask
            above = ((aspect >= bdry) & (aspect < band_hi)) & finite_mask

        n_band = int(np.sum(in_band))
        n_below = int(np.sum(below))
        n_above = int(np.sum(above))
        ratio = n_below / n_above if n_above > 0 else float("inf")

        # Identify which aspects are on either side
        asp_names_at_bdry = {
            45: ("North", "East"),
            135: ("East", "South"),
            225: ("South", "West"),
            315: ("West", "North"),
        }
        lower_asp, upper_asp = asp_names_at_bdry[bdry]

        print(
            f"{bdry:>8.0f}°  {lower_asp:>9s}  {upper_asp:>9s}  "
            f"{n_band:>8d}  {n_below:>8d}  {n_above:>8d}  {ratio:>9.3f}"
        )


# ===========================================================================
# Part 2: Hypothesis testing
# ===========================================================================
def run_test(
    label: str,
    hand: np.ndarray,
    dtnd: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    area: np.ndarray,
    drainage_id: np.ndarray,
    res_m: float,
    pub: dict,
    skip_zero_hand_bins: bool = False,
    swenson_style: bool = False,
    use_w1_fit: bool = False,
    bin1_max_override: float | None = LOWEST_BIN_MAX,
) -> dict:
    """Run hillslope computation on filtered arrays and report correlation."""
    n_pixels = hand.size
    n_valid = int(np.sum((hand >= 0) & np.isfinite(hand)))
    n_neg = int(np.sum((hand < 0) & np.isfinite(hand)))

    result = compute_elements(
        hand,
        dtnd,
        slope,
        aspect,
        area,
        drainage_id,
        res_m,
        skip_zero_hand_bins=skip_zero_hand_bins,
        swenson_style=swenson_style,
        use_w1_fit=use_w1_fit,
        bin1_max_override=bin1_max_override,
    )
    corrs = compute_all_correlations(result["elements"], pub)

    print(f"\n  {label}")
    print(
        f"    Pixels: {n_pixels} total, {n_valid} valid (hand >= 0 & finite), "
        f"{n_neg} negative HAND"
    )
    print(f"    Area fraction correlation: {corrs['area_fraction']:.4f}")
    print("    All correlations: ", end="")
    for name in ("height", "distance", "slope", "aspect", "width", "area_fraction"):
        print(f"{name}={corrs[name]:.4f}  ", end="")
    print()

    return {"correlations": corrs, "elements": result["elements"]}


def run_hypothesis_tests(arrays: dict, pub: dict) -> dict:
    """Run all hypothesis tests A-E and return results."""
    hand = arrays["hand"]
    dtnd = arrays["dtnd"]
    slope = arrays["slope"]
    aspect = arrays["aspect"]
    area = arrays["area"]
    fflood = arrays["fflood"]
    did = arrays["drainage_id"]
    res_m = arrays["res_m"]

    print("\n" + "=" * 78)
    print("PART 2: HYPOTHESIS TESTING — Incremental filter application")
    print("=" * 78)

    results = {}

    # Baseline (no filters — same as merit_regression.py)
    results["baseline"] = run_test(
        "BASELINE (no filters)", hand, dtnd, slope, aspect, area, did, res_m, pub
    )

    # Test A: DTND tail removal
    keep_idx = tail_index(dtnd, hand)
    n_removed = hand.size - keep_idx.size
    print("\n  --- Test A: DTND tail removal ---")
    print(
        f"    TailIndex removed {n_removed} pixels ({100 * n_removed / hand.size:.2f}%)"
    )
    results["A_tail"] = run_test(
        "Test A: DTND tail removal",
        hand[keep_idx],
        dtnd[keep_idx],
        slope[keep_idx],
        aspect[keep_idx],
        area[keep_idx],
        did[keep_idx],
        res_m,
        pub,
    )

    # Test B: DTND minimum clipping
    dtnd_clipped = dtnd.copy()
    n_clipped = int(np.sum(dtnd_clipped < 1.0))
    dtnd_clipped[dtnd_clipped < 1.0] = 1.0
    print("\n  --- Test B: DTND min clipping ---")
    print(
        f"    Clipped {n_clipped} pixels with DTND < 1.0 m ({100 * n_clipped / hand.size:.2f}%)"
    )
    results["B_clip"] = run_test(
        "Test B: DTND min clipping",
        hand,
        dtnd_clipped,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
    )

    # Test C: Flooded region handling
    hand_flood = apply_flood_filter(hand, fflood)
    n_flooded = int(np.sum(hand_flood == -1))
    print("\n  --- Test C: Flooded region handling ---")
    print(
        f"    fflood stats: min={fflood.min():.4f}, max={fflood.max():.4f}, "
        f"mean={fflood.mean():.4f}, nonzero={np.sum(np.abs(fflood) > 0)}"
    )
    print(
        f"    Marked {n_flooded} pixels as HAND=-1 ({100 * n_flooded / hand.size:.2f}%)"
    )
    results["C_flood"] = run_test(
        "Test C: Flooded region handling",
        hand_flood,
        dtnd,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
    )

    # Test D: Mean-HAND <= 0 bin skip
    print("\n  --- Test D: Mean-HAND <= 0 bin skip ---")
    results["D_skip"] = run_test(
        "Test D: Mean-HAND <= 0 bin skip",
        hand,
        dtnd,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
        skip_zero_hand_bins=True,
    )

    # Test E: All filters combined
    print("\n  --- Test E: All filters combined ---")
    # Apply in Swenson's order: NaN removal → tail removal → flood → DTND clipping
    # Step 1: NaN removal (already done by valid mask in compute_elements)
    # Step 2: Tail removal
    keep_idx_e = tail_index(dtnd, hand)
    h_e = hand[keep_idx_e].copy()
    d_e = dtnd[keep_idx_e].copy()
    s_e = slope[keep_idx_e]
    a_e = aspect[keep_idx_e]
    ar_e = area[keep_idx_e]
    did_e = did[keep_idx_e]
    ff_e = fflood[keep_idx_e]

    # Step 3: Flood filter
    h_e = apply_flood_filter(h_e, ff_e)

    # Step 4: DTND clipping
    d_e[d_e < 1.0] = 1.0

    print(
        f"    After all filters: {h_e.size} pixels "
        f"(removed {hand.size - h_e.size} by tail, "
        f"{int(np.sum(h_e == -1))} marked flood)"
    )
    results["E_all"] = run_test(
        "Test E: All filters combined (A+B+C+D)",
        h_e,
        d_e,
        s_e,
        a_e,
        ar_e,
        did_e,
        res_m,
        pub,
        skip_zero_hand_bins=True,
    )

    # Test F: Swenson-style valid mask (np.isfinite only, includes negative HAND)
    n_neg_hand = int(np.sum((hand < 0) & np.isfinite(hand)))
    n_finite = int(np.sum(np.isfinite(hand)))
    print("\n  --- Test F: Swenson-style valid mask (include negative HAND) ---")
    print(
        f"    Negative-HAND pixels: {n_neg_hand} "
        f"({100 * n_neg_hand / hand.size:.2f}% of total, "
        f"{100 * n_neg_hand / n_finite:.2f}% of finite)"
    )
    results["F_swenson_mask"] = run_test(
        "Test F: Swenson-style valid mask (isfinite only)",
        hand,
        dtnd,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
        swenson_style=True,
    )

    # Test G: Swenson valid mask + DTND tail removal
    print("\n  --- Test G: Swenson mask + DTND tail removal ---")
    results["G_swenson_tail"] = run_test(
        "Test G: Swenson mask + DTND tail removal",
        hand[keep_idx],
        dtnd[keep_idx],
        slope[keep_idx],
        aspect[keep_idx],
        area[keep_idx],
        did[keep_idx],
        res_m,
        pub,
        swenson_style=True,
    )

    # Test H: Full Swenson pipeline with corrected fflood
    # Order: tail removal → flood filter (binary mask) → DTND clipping
    # + swenson_style valid mask + skip zero-HAND bins
    print("\n  --- Test H: Full Swenson pipeline (corrected fflood) ---")
    keep_idx_h = tail_index(dtnd, hand)
    h_h = hand[keep_idx_h].copy()
    d_h = dtnd[keep_idx_h].copy()
    s_h = slope[keep_idx_h]
    a_h = aspect[keep_idx_h]
    ar_h = area[keep_idx_h]
    did_h = did[keep_idx_h]
    ff_h = fflood[keep_idx_h]

    h_h = apply_flood_filter(h_h, ff_h)
    d_h[d_h < 1.0] = 1.0

    n_flood_h = int(np.sum(h_h == -1))
    print(
        f"    After all filters: {h_h.size} pixels "
        f"(removed {hand.size - h_h.size} by tail, "
        f"{n_flood_h} marked flood)"
    )
    results["H_full_swenson"] = run_test(
        "Test H: Full Swenson pipeline (corrected fflood)",
        h_h,
        d_h,
        s_h,
        a_h,
        ar_h,
        did_h,
        res_m,
        pub,
        skip_zero_hand_bins=True,
        swenson_style=True,
    )

    # Test I: Fix polynomial fit weighting (w^1 instead of w^2)
    print("\n  --- Test I: Corrected polynomial fit weighting (w^1) ---")
    print("    Swenson's _fit_polynomial uses W=diag(w) → minimizes sum w_i * r_i^2")
    print("    Our lstsq uses G*w, y*w → minimizes sum w_i^2 * r_i^2")
    results["I_w1_fit"] = run_test(
        "Test I: Corrected polynomial fit weighting (w^1)",
        hand,
        dtnd,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
        use_w1_fit=True,
    )

    return results


# ===========================================================================
# Part 3: Sensitivity map
# ===========================================================================
def print_sensitivity_map(our_elements: list, pub: dict) -> None:
    """
    For each of the 16 elements, compute how much perturbing its area fraction
    changes the overall correlation. Identifies which elements drive the gap.
    """
    print("\n" + "=" * 78)
    print("PART 3: SENSITIVITY MAP — Which elements drive the correlation gap?")
    print("=" * 78)

    our_area = np.array([e["area"] for e in our_elements])
    pub_area = pub["area"]
    our_frac = our_area / our_area.sum() if our_area.sum() > 0 else our_area
    pub_frac = pub_area / pub_area.sum() if pub_area.sum() > 0 else pub_area

    baseline_corr = float(np.corrcoef(our_frac, pub_frac)[0, 1])

    # Test: what if we replace each element with published value?
    print(
        f"\n{'Elem':>4s}  {'Aspect':<6s}  {'Bin':>3s}  "
        f"{'Corr if fixed':>13s}  {'Delta':>8s}  {'Impact':>6s}"
    )
    print("-" * 55)

    impacts = []
    for i in range(16):
        modified = our_frac.copy()
        modified[i] = pub_frac[i]
        # Renormalize the rest proportionally
        remaining_our = 1 - our_frac[i]
        remaining_new = 1 - pub_frac[i]
        if remaining_our > 0:
            for j in range(16):
                if j != i:
                    modified[j] = our_frac[j] * remaining_new / remaining_our

        new_corr = float(np.corrcoef(modified, pub_frac)[0, 1])
        delta = new_corr - baseline_corr
        asp_idx = i // N_HAND_BINS
        h_idx = i % N_HAND_BINS
        impacts.append(delta)
        print(
            f"{i:>4d}  {ASPECT_NAMES[asp_idx]:<6s}  {h_idx:>3d}  "
            f"{new_corr:>13.4f}  {delta:>+8.4f}  "
            f"{'***' if delta > 0.02 else '**' if delta > 0.01 else '*' if delta > 0.005 else ''}"
        )

    print(f"\nBaseline correlation: {baseline_corr:.4f}")
    print(
        f"Max single-element improvement: {max(impacts):+.4f} "
        f"(element {np.argmax(impacts)})"
    )
    top3 = np.argsort(impacts)[-3:][::-1]
    print(
        f"Top 3 impactful elements: {top3.tolist()} "
        f"(deltas: {[f'{impacts[i]:+.4f}' for i in top3]})"
    )


# ===========================================================================
# Summary table
# ===========================================================================
def print_summary(test_results: dict) -> None:
    """Print compact summary of all test correlations."""
    print("\n" + "=" * 78)
    print("SUMMARY: Area fraction correlation by test")
    print("=" * 78)

    print(f"\n{'Test':<40s}  {'Area frac':>9s}  {'Delta':>8s}")
    print("-" * 62)

    baseline = test_results["baseline"]["correlations"]["area_fraction"]
    for key, label in [
        ("baseline", "Baseline (no filters)"),
        ("A_tail", "A: DTND tail removal"),
        ("B_clip", "B: DTND min clipping"),
        ("C_flood", "C: Flooded region handling"),
        ("D_skip", "D: Mean-HAND <= 0 bin skip"),
        ("E_all", "E: All filters combined"),
        ("F_swenson_mask", "F: Swenson valid mask (incl neg HAND)"),
        ("G_swenson_tail", "G: Swenson mask + tail removal"),
        ("H_full_swenson", "H: Full Swenson (corrected fflood)"),
        ("I_w1_fit", "I: Corrected poly fit weighting (w^1)"),
    ]:
        corr = test_results[key]["correlations"]["area_fraction"]
        delta = corr - baseline
        print(f"{label:<40s}  {corr:>9.4f}  {delta:>+8.4f}")

    # Full correlation table: baseline vs key tests
    print(
        f"\n{'Parameter':<18s}  {'Baseline':>8s}  {'All (E)':>8s}  "
        f"{'Swen (F)':>8s}  {'S+tail(G)':>9s}  {'Full(H)':>8s}  {'w1(I)':>8s}"
    )
    print("-" * 88)
    for name in ("height", "distance", "slope", "aspect", "width", "area_fraction"):
        b = test_results["baseline"]["correlations"][name]
        e = test_results["E_all"]["correlations"][name]
        f = test_results["F_swenson_mask"]["correlations"][name]
        g = test_results["G_swenson_tail"]["correlations"][name]
        h = test_results["H_full_swenson"]["correlations"][name]
        i = test_results["I_w1_fit"]["correlations"][name]
        print(
            f"{name:<18s}  {b:>8.4f}  {e:>8.4f}  {f:>8.4f}  "
            f"{g:>9.4f}  {h:>8.4f}  {i:>8.4f}"
        )


# ===========================================================================
# Part 4: A_cumsum(d) curve diagnostics (Test J)
# ===========================================================================
def print_acumsum_diagnostics(arrays: dict) -> None:
    """
    Test J: For each aspect, print the raw A_cumsum(d) data, the fitted
    quadratic (both w^2 and w^1), per-point residuals, and R^2.
    """
    hand = arrays["hand"]
    dtnd = arrays["dtnd"]
    aspect = arrays["aspect"]
    area = arrays["area"]
    drainage_id = arrays["drainage_id"]
    res_m = arrays["res_m"]

    valid = (hand >= 0) & np.isfinite(hand)

    print("\n" + "=" * 78)
    print("TEST J: A_cumsum(d) CURVE DIAGNOSTICS")
    print("=" * 78)

    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        asp_mask = get_aspect_mask(aspect, asp_bin) & valid
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            print(f"\n  {asp_name}: no valid pixels")
            continue

        n_hillslopes = max(len(np.unique(drainage_id[asp_indices])), 1)

        # Build A_cumsum(d) curve (same as fit_trapezoidal_width)
        asp_dtnd = dtnd[asp_indices]
        asp_area = area[asp_indices]

        if np.max(asp_dtnd) < res_m:
            print(f"\n  {asp_name}: max DTND < min_dtnd ({res_m:.1f}m), skipped")
            continue

        n_bins = 10
        dtnd_bins = np.linspace(res_m, np.max(asp_dtnd) + 1, n_bins + 1)
        d = np.zeros(n_bins)
        A_cumsum = np.zeros(n_bins)

        for k in range(n_bins):
            mask = asp_dtnd >= dtnd_bins[k]
            d[k] = dtnd_bins[k]
            A_cumsum[k] = np.sum(asp_area[mask])

        A_cumsum /= n_hillslopes

        if res_m > 0:
            d = np.concatenate([[0], d])
            A_cumsum = np.concatenate([[np.sum(asp_area) / n_hillslopes], A_cumsum])

        # Fit w^2 (baseline lstsq)
        weights = A_cumsum
        G = np.column_stack([np.ones_like(d), d, d**2])
        Gw = G * weights[:, np.newaxis]
        coeffs_w2 = np.linalg.lstsq(Gw, A_cumsum * weights, rcond=None)[0]

        # Fit w^1 (Swenson's method)
        coeffs_w1 = _fit_polynomial_w1(d, A_cumsum, ncoefs=3, weights=A_cumsum)

        # Evaluate fits
        fitted_w2 = coeffs_w2[0] + coeffs_w2[1] * d + coeffs_w2[2] * d**2
        fitted_w1 = coeffs_w1[0] + coeffs_w1[1] * d + coeffs_w1[2] * d**2

        # R^2 for each
        ss_tot = np.sum((A_cumsum - np.mean(A_cumsum)) ** 2)
        ss_res_w2 = np.sum((A_cumsum - fitted_w2) ** 2)
        ss_res_w1 = np.sum((A_cumsum - fitted_w1) ** 2)
        r2_w2 = 1 - ss_res_w2 / ss_tot if ss_tot > 0 else float("nan")
        r2_w1 = 1 - ss_res_w1 / ss_tot if ss_tot > 0 else float("nan")

        print(
            f"\n  {asp_name} (n_hillslopes={n_hillslopes}, {len(asp_indices)} pixels)"
        )
        print(
            f"    w^2 coeffs: a0={coeffs_w2[0]:.1f}, a1={coeffs_w2[1]:.4f}, "
            f"a2={coeffs_w2[2]:.8f}  R^2={r2_w2:.6f}"
        )
        print(
            f"    w^1 coeffs: a0={coeffs_w1[0]:.1f}, a1={coeffs_w1[1]:.4f}, "
            f"a2={coeffs_w1[2]:.8f}  R^2={r2_w1:.6f}"
        )

        print(
            f"    {'d(m)':>8s}  {'A_cumsum':>10s}  "
            f"{'fit_w2':>10s}  {'res_w2':>10s}  {'%err_w2':>8s}  "
            f"{'fit_w1':>10s}  {'res_w1':>10s}  {'%err_w1':>8s}"
        )
        print("    " + "-" * 84)

        for j in range(len(d)):
            res_w2 = A_cumsum[j] - fitted_w2[j]
            res_w1 = A_cumsum[j] - fitted_w1[j]
            pct_w2 = 100 * res_w2 / A_cumsum[j] if A_cumsum[j] != 0 else 0
            pct_w1 = 100 * res_w1 / A_cumsum[j] if A_cumsum[j] != 0 else 0
            print(
                f"    {d[j]:>8.1f}  {A_cumsum[j]:>10.1f}  "
                f"{fitted_w2[j]:>10.1f}  {res_w2:>+10.1f}  {pct_w2:>+7.2f}%  "
                f"{fitted_w1[j]:>10.1f}  {res_w1:>+10.1f}  {pct_w1:>+7.2f}%"
            )


# ===========================================================================
# Part 5: HAND CDF at bin-0 boundary (Test K)
# ===========================================================================
def print_hand_cdf_diagnostics(arrays: dict) -> None:
    """
    Test K: For each aspect, compute pixel density near the 2.0m HAND
    bin boundary. High density means small HAND shifts move large numbers
    of pixels across the boundary.
    """
    hand = arrays["hand"]
    aspect = arrays["aspect"]

    valid = (hand >= 0) & np.isfinite(hand)

    print("\n" + "=" * 78)
    print("TEST K: HAND CDF AT BIN-0 BOUNDARY (2.0m)")
    print("=" * 78)

    # Band widths to check
    bands = [(1.0, 3.0), (1.5, 2.5), (1.8, 2.2)]

    for band_lo, band_hi in bands:
        print(
            f"\n  HAND band [{band_lo:.1f}, {band_hi:.1f}]m "
            f"(±{(band_hi - band_lo) / 2:.1f}m around 2.0m)"
        )
        print(
            f"    {'Aspect':<8s}  {'In band':>8s}  {'Total':>8s}  "
            f"{'% in band':>9s}  {'density':>12s}"
        )
        print("    " + "-" * 55)

        for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
            asp_mask = get_aspect_mask(aspect, asp_bin) & valid
            n_asp = int(np.sum(asp_mask))

            in_band = asp_mask & (hand >= band_lo) & (hand < band_hi)
            n_band = int(np.sum(in_band))

            pct = 100 * n_band / n_asp if n_asp > 0 else 0
            # Density: pixels per meter of HAND (normalized by total aspect pixels)
            density = n_band / (band_hi - band_lo) / n_asp if n_asp > 0 else 0

            print(
                f"    {asp_name:<8s}  {n_band:>8d}  {n_asp:>8d}  "
                f"{pct:>8.2f}%  {density:>12.4f} /m"
            )

    # CDF slope at exactly 2.0m (approximate with narrow band)
    eps = 0.1
    print(f"\n  CDF slope approximation (HAND in [{2 - eps:.1f}, {2 + eps:.1f}]m):")
    print(
        f"    {'Aspect':<8s}  {'Pixels':>8s}  {'Total':>8s}  "
        f"{'CDF slope':>12s}  {'Interpretation'}"
    )
    print("    " + "-" * 65)

    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        asp_mask = get_aspect_mask(aspect, asp_bin) & valid
        n_asp = int(np.sum(asp_mask))

        narrow = asp_mask & (hand >= 2.0 - eps) & (hand < 2.0 + eps)
        n_narrow = int(np.sum(narrow))

        # CDF slope = fraction of pixels per unit HAND at this point
        cdf_slope = n_narrow / (2 * eps) / n_asp if n_asp > 0 else 0

        if cdf_slope > 0.10:
            interp = "HIGH sensitivity"
        elif cdf_slope > 0.05:
            interp = "moderate sensitivity"
        else:
            interp = "low sensitivity"

        print(
            f"    {asp_name:<8s}  {n_narrow:>8d}  {n_asp:>8d}  "
            f"{cdf_slope:>12.4f} /m  {interp}"
        )


# ===========================================================================
# Part 6: Q25 diagnostic and bin1_max sensitivity (Test L)
# ===========================================================================
def print_q25_diagnostic(arrays: dict) -> None:
    """Print Q25 of hand[hand > 0] and which branch compute_hand_bins takes."""
    hand = arrays["hand"]
    aspect = arrays["aspect"]

    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]
    n_stream = int(np.sum(hand == 0))
    n_valid = hand_valid.size

    print("\n" + "=" * 78)
    print("Q25 DIAGNOSTIC: HAND distribution and bin branch")
    print("=" * 78)

    if n_valid == 0:
        print("  No valid HAND > 0 pixels!")
        return

    hand_sorted = np.sort(hand_valid)
    q25 = hand_sorted[int(0.25 * n_valid) - 1]
    q50 = hand_sorted[int(0.50 * n_valid) - 1]
    q75 = hand_sorted[int(0.75 * n_valid) - 1]

    print(f"  HAND > 0 pixels: {n_valid}")
    print(f"  HAND == 0 pixels (stream): {n_stream}")
    print(f"  HAND < 0 pixels: {int(np.sum((hand < 0) & np.isfinite(hand)))}")
    print(f"  Q25 = {q25:.4f} m")
    print(f"  Q50 = {q50:.4f} m")
    print(f"  Q75 = {q75:.4f} m")
    print(f"  LOWEST_BIN_MAX = {LOWEST_BIN_MAX} m")

    if q25 > LOWEST_BIN_MAX:
        print(f"  Branch: FORCED (Q25={q25:.4f} > bin1_max={LOWEST_BIN_MAX})")
        print(
            "    → Uses min-aspect-fraction + 33rd/66th percentile of hand > bin1_max"
        )
    else:
        print(f"  Branch: QUARTILE (Q25={q25:.4f} <= bin1_max={LOWEST_BIN_MAX})")
        print(f"    → Uses quartile bounds [0, {q25:.4f}, {q50:.4f}, {q75:.4f}, 1e6]")

    # Show what bounds each branch produces
    bounds_forced = compute_hand_bins(
        hand, aspect, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )
    bounds_quartile = compute_hand_bins(hand, aspect, ASPECT_BINS, bin1_max=None)

    print(f"\n  Forced branch bounds:   {bounds_forced.tolist()}")
    print(f"  Quartile branch bounds: {bounds_quartile.tolist()}")


def run_bin1max_sweep(arrays: dict, pub: dict) -> dict:
    """
    Test L: Sweep bin1_max values and report correlations at each.

    Tests whether the forced 2m HAND bin constraint is causing the area
    fraction gap by trying alternative bin1_max values.
    """
    hand = arrays["hand"]
    dtnd = arrays["dtnd"]
    slope = arrays["slope"]
    aspect = arrays["aspect"]
    area = arrays["area"]
    did = arrays["drainage_id"]
    res_m = arrays["res_m"]

    sweep_values: list[float | None] = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, None]

    print("\n" + "=" * 78)
    print("TEST L: bin1_max SENSITIVITY SWEEP")
    print("=" * 78)
    print("  Sweeping bin1_max to test whether the forced 2m HAND constraint")
    print("  is causing the area fraction gap. All runs use w^1 fit.")

    results = {}

    for val in sweep_values:
        label = f"bin1_max={val}" if val is not None else "bin1_max=None (quartiles)"

        result = compute_elements(
            hand,
            dtnd,
            slope,
            aspect,
            area,
            did,
            res_m,
            use_w1_fit=True,
            bin1_max_override=val,
        )
        corrs = compute_all_correlations(result["elements"], pub)
        diag = result["diagnostics"]

        # Per-aspect sub-correlations
        our_area = np.array([e["area"] for e in result["elements"]])
        pub_area = pub["area"]
        our_frac = our_area / our_area.sum() if our_area.sum() > 0 else our_area
        pub_frac = pub_area / pub_area.sum() if pub_area.sum() > 0 else pub_area

        asp_corrs = []
        for asp_idx in range(N_ASPECT_BINS):
            sl = slice(asp_idx * N_HAND_BINS, (asp_idx + 1) * N_HAND_BINS)
            o = our_frac[sl]
            p = pub_frac[sl]
            if np.std(o) > 0 and np.std(p) > 0:
                asp_corrs.append(float(np.corrcoef(o, p)[0, 1]))
            else:
                asp_corrs.append(float("nan"))

        print(f"\n  --- {label} ---")
        print(f"    HAND bounds: {diag['hand_bounds']}")
        print(f"    Area fraction corr: {corrs['area_fraction']:.4f}")
        print(
            f"    Per-aspect: N={asp_corrs[0]:.4f}  E={asp_corrs[1]:.4f}  "
            f"S={asp_corrs[2]:.4f}  W={asp_corrs[3]:.4f}"
        )
        print("    All params: ", end="")
        for name in ("height", "distance", "slope", "aspect", "width", "area_fraction"):
            print(f"{name}={corrs[name]:.4f}  ", end="")
        print()

        results[str(val)] = {
            "bin1_max": val,
            "hand_bounds": diag["hand_bounds"],
            "correlations": corrs,
            "aspect_corrs": asp_corrs,
        }

    # Summary table
    print("\n  --- Summary ---")
    print(
        f"  {'bin1_max':>12s}  {'Area frac':>9s}  "
        f"{'North':>7s}  {'East':>7s}  {'South':>7s}  {'West':>7s}  "
        f"{'Height':>7s}  {'Width':>7s}"
    )
    print("  " + "-" * 82)

    for val in sweep_values:
        key = str(val)
        r = results[key]
        label = f"{val}" if val is not None else "None(Q)"
        print(
            f"  {label:>12s}  {r['correlations']['area_fraction']:>9.4f}  "
            f"{r['aspect_corrs'][0]:>7.4f}  {r['aspect_corrs'][1]:>7.4f}  "
            f"{r['aspect_corrs'][2]:>7.4f}  {r['aspect_corrs'][3]:>7.4f}  "
            f"{r['correlations']['height']:>7.4f}  "
            f"{r['correlations']['width']:>7.4f}"
        )

    return results


# ===========================================================================
# Part 7: Infer Swenson's bin boundaries from published data (Test M)
# ===========================================================================
def infer_published_boundaries(arrays: dict, pub: dict) -> None:
    """
    Infer Swenson's HAND bin boundaries from published mean HAND values.

    Prints raw published areas and mean HAND (never printed before), then
    sweeps bin1_max candidates to find which boundaries best reproduce the
    published per-element mean HAND. Compares inferred boundaries to our
    forced and quartile bounds.

    Not wired into main() — call manually for one-off analysis.
    """
    hand = arrays["hand"]
    aspect = arrays["aspect"]
    area = arrays["area"]

    pub_height = pub["height"]  # mean HAND per element (16 values)
    pub_area_raw = pub["area"]  # raw area per element (16 values)

    valid = (hand > 0) & np.isfinite(hand)

    # Reference bounds from our data
    bounds_forced = compute_hand_bins(
        hand, aspect, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )
    bounds_quartile = compute_hand_bins(hand, aspect, ASPECT_BINS, bin1_max=None)

    print("\n" + "=" * 78)
    print("TEST M: BOUNDARY INFERENCE FROM PUBLISHED DATA")
    print("=" * 78)

    # ----- Part 1: Raw published data -----
    total_pub_area = float(np.sum(pub_area_raw))

    print("\n  --- Raw published data per element ---")
    print(
        f"  {'Elem':>4s}  {'Aspect':<6s}  {'Bin':>3s}  "
        f"{'Pub area (m²)':>14s}  {'Pub HAND (m)':>12s}  {'Pub frac':>8s}"
    )
    print("  " + "-" * 58)

    for i in range(16):
        asp_idx = i // N_HAND_BINS
        h_idx = i % N_HAND_BINS
        frac = pub_area_raw[i] / total_pub_area if total_pub_area > 0 else 0
        print(
            f"  {i:>4d}  {ASPECT_NAMES[asp_idx]:<6s}  {h_idx:>3d}  "
            f"{pub_area_raw[i]:>14.1f}  {pub_height[i]:>12.4f}  {frac:>8.4f}"
        )

    # Per-aspect totals
    print(f"\n  {'Aspect':<8s}  {'Total area (m²)':>16s}  {'Frac':>8s}")
    print("  " + "-" * 36)
    for asp_idx, asp_name in enumerate(ASPECT_NAMES):
        sl = slice(asp_idx * N_HAND_BINS, (asp_idx + 1) * N_HAND_BINS)
        asp_area_sum = float(np.sum(pub_area_raw[sl]))
        asp_frac = asp_area_sum / total_pub_area if total_pub_area > 0 else 0
        print(f"  {asp_name:<8s}  {asp_area_sum:>16.1f}  {asp_frac:>8.4f}")

    print(f"\n  Total published area: {total_pub_area:.1f} m²")

    # Our total area for scale comparison
    our_total_area = float(np.sum(area[valid]))
    print(f"  Our total area (hand>0): {our_total_area:.1f} m²")
    if our_total_area > 0:
        print(f"  Ratio (pub/ours): {total_pub_area / our_total_area:.4f}")

    # ----- Part 2: Compute mean HAND for reference bound sets -----
    def _compute_bin_means(bounds: np.ndarray) -> np.ndarray:
        """Compute mean HAND in each of 16 aspect-bin combinations."""
        means = np.zeros(16)
        for a_idx, asp_bin in enumerate(ASPECT_BINS):
            asp_mask = get_aspect_mask(aspect, asp_bin) & valid
            for h_idx in range(N_HAND_BINS):
                h_lower = bounds[h_idx]
                h_upper = bounds[h_idx + 1]
                bin_mask = asp_mask & (hand >= h_lower) & (hand < h_upper)
                if np.any(bin_mask):
                    means[a_idx * N_HAND_BINS + h_idx] = float(np.mean(hand[bin_mask]))
        return means

    forced_means = _compute_bin_means(bounds_forced)
    quartile_means = _compute_bin_means(bounds_quartile)

    forced_rmse = float(np.sqrt(np.mean((forced_means - pub_height) ** 2)))
    quartile_rmse = float(np.sqrt(np.mean((quartile_means - pub_height) ** 2)))

    # ----- Part 3: Sweep bin1_max candidates -----
    sweep_values = np.arange(0.5, 5.01, 0.01)

    best_rmse = np.inf
    best_bin1max = None
    best_bounds = None
    best_means = None

    # Track RMSE curve for printing
    rmse_curve = []

    for val in sweep_values:
        bounds = compute_hand_bins(hand, aspect, ASPECT_BINS, bin1_max=float(val))
        means = _compute_bin_means(bounds)
        rmse = float(np.sqrt(np.mean((means - pub_height) ** 2)))
        rmse_curve.append((float(val), rmse, bounds.tolist()))

        if rmse < best_rmse:
            best_rmse = rmse
            best_bin1max = float(val)
            best_bounds = bounds.copy()
            best_means = means.copy()

    # ----- Part 4: Results -----
    print("\n  --- Sweep results ---")
    print(f"  Best-fit bin1_max: {best_bin1max:.2f} m (RMSE = {best_rmse:.4f} m)")
    print(f"  Forced (2.0m) RMSE:  {forced_rmse:.4f} m")
    print(f"  Quartile RMSE:       {quartile_rmse:.4f} m")

    # Print RMSE at sampled points
    print("\n  RMSE curve (sampled):")
    print(f"  {'bin1_max':>8s}  {'RMSE':>8s}  {'Branch':>10s}  {'Bounds'}")
    print("  " + "-" * 70)
    # Print at integer + 0.5 steps, plus the best-fit point
    sample_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    hand_sorted = np.sort(hand[valid])
    q25 = hand_sorted[int(0.25 * hand_sorted.size) - 1]
    printed_best = False
    for val, rmse, bounds in rmse_curve:
        if round(val, 2) in sample_vals or (not printed_best and val >= best_bin1max):
            if val >= best_bin1max and not printed_best:
                # Print best-fit row first if it falls between samples
                if round(val, 2) != round(best_bin1max, 2):
                    best_branch = "forced" if q25 > best_bin1max else "quartile"
                    print(
                        f"  {best_bin1max:>8.2f}  {best_rmse:>8.4f}  "
                        f"{best_branch:>10s}  "
                        f"{[f'{b:.2f}' for b in best_bounds.tolist()]}  <-- BEST"
                    )
                printed_best = True
            branch = "forced" if q25 > val else "quartile"
            marker = "  <-- BEST" if round(val, 2) == round(best_bin1max, 2) else ""
            if round(val, 2) in sample_vals:
                print(
                    f"  {val:>8.2f}  {rmse:>8.4f}  {branch:>10s}  "
                    f"{[f'{b:.2f}' for b in bounds]}{marker}"
                )

    # Boundary comparison table
    print("\n  --- Boundary comparison ---")
    print(
        f"  {'Source':<22s}  {'b0':>6s}  {'b1':>6s}  "
        f"{'b2':>6s}  {'b3':>6s}  {'b4':>8s}  {'RMSE':>8s}"
    )
    print("  " + "-" * 68)

    for label, bounds, rmse in [
        ("Forced (2.0m)", bounds_forced, forced_rmse),
        ("Quartile", bounds_quartile, quartile_rmse),
        (f"Best-fit ({best_bin1max:.2f}m)", best_bounds, best_rmse),
    ]:
        print(
            f"  {label:<22s}  {bounds[0]:>6.2f}  {bounds[1]:>6.2f}  "
            f"{bounds[2]:>6.2f}  {bounds[3]:>6.2f}  "
            f"{bounds[4]:>8.0f}  {rmse:>8.4f}"
        )

    # Per-element comparison
    print("\n  --- Per-element mean HAND comparison ---")
    print(
        f"  {'Elem':>4s}  {'Aspect':<6s}  {'Bin':>3s}  "
        f"{'Published':>9s}  {'Forced':>8s}  {'Quartile':>8s}  {'Best-fit':>8s}"
    )
    print("  " + "-" * 60)

    for i in range(16):
        asp_idx = i // N_HAND_BINS
        h_idx = i % N_HAND_BINS
        print(
            f"  {i:>4d}  {ASPECT_NAMES[asp_idx]:<6s}  {h_idx:>3d}  "
            f"{pub_height[i]:>9.4f}  {forced_means[i]:>8.4f}  "
            f"{quartile_means[i]:>8.4f}  {best_means[i]:>8.4f}"
        )

    # Interpretation
    print("\n  --- Interpretation ---")
    print(f"  Our Q25: {q25:.4f} m")
    print(f"  Our forced branch triggers at bin1_max < {q25:.4f}")
    if best_bin1max < q25:
        print(
            f"  Best-fit ({best_bin1max:.2f}) < Q25 ({q25:.2f}) "
            f"→ Swenson likely uses FORCED branch"
        )
    else:
        print(
            f"  Best-fit ({best_bin1max:.2f}) >= Q25 ({q25:.2f}) "
            f"→ Swenson likely uses QUARTILE branch"
        )

    if abs(best_rmse - forced_rmse) < 0.1:
        print(
            f"  Forced RMSE ({forced_rmse:.4f}) ≈ best RMSE ({best_rmse:.4f}) "
            f"→ forced branch is close to optimal"
        )
    elif forced_rmse < quartile_rmse:
        print(
            f"  Forced RMSE ({forced_rmse:.4f}) < quartile RMSE "
            f"({quartile_rmse:.4f}) → forced branch is better"
        )
    else:
        print(
            f"  Quartile RMSE ({quartile_rmse:.4f}) < forced RMSE "
            f"({forced_rmse:.4f}) → quartile branch is better"
        )


# ===========================================================================
# Part 8: Swenson DEM conditioning (Test N)
# ===========================================================================
def run_test_n(accum_threshold: int, pub: dict, arrays_original: dict) -> dict:
    """
    Test N: Re-run flow routing with Swenson's full DEM conditioning and
    compare all 6 correlations to the original (unconditioned) results.

    Tests whether the 5 missing conditioning steps explain the systematic
    HAND offset (0.2-0.4m) seen in Test M.
    """
    print("\n" + "=" * 78)
    print("TEST N: SWENSON DEM CONDITIONING")
    print("=" * 78)
    print("  Adding 5 missing DEM conditioning steps from Swenson's code:")
    print("    1. identify_basins() — mask large flat regions before pysheds")
    print("    2. identify_open_water(slope) — find low-slope regions")
    print("    3. Lower flooded areas by 0.1m before resolve_flats")
    print("    4. Re-mask basin pixels after flowdir, before accumulation")
    print("    5. Force basin boundary pixels into stream network")
    print()

    # Run flow routing with full Swenson conditioning
    t0 = time.time()
    arrays = run_flow_routing_swenson(MERIT_DEM, accum_threshold)
    print(f"  Flow routing complete ({time.time() - t0:.0f}s)")

    hand = arrays["hand"]
    dtnd = arrays["dtnd"]
    slope = arrays["slope"]
    aspect = arrays["aspect"]
    area = arrays["area"]
    fflood = arrays["fflood"]
    did = arrays["drainage_id"]
    res_m = arrays["res_m"]

    # --- Array comparison ---
    hand_orig = arrays_original["hand"]
    dtnd_orig = arrays_original["dtnd"]

    print("\n  --- Array comparison (Swenson-conditioned vs original) ---")

    valid_orig = (hand_orig >= 0) & np.isfinite(hand_orig)
    valid_new = (hand >= 0) & np.isfinite(hand)

    print(f"  {'Metric':<30s}  {'Original':>12s}  {'Swenson':>12s}  {'Delta':>12s}")
    print("  " + "-" * 70)

    for label, orig, new in [
        ("Valid pixels (HAND>=0)", np.sum(valid_orig), np.sum(valid_new)),
        ("Mean HAND (valid)", np.mean(hand_orig[valid_orig]), np.mean(hand[valid_new])),
        (
            "Median HAND (valid)",
            np.median(hand_orig[valid_orig]),
            np.median(hand[valid_new]),
        ),
        ("Mean DTND (valid)", np.mean(dtnd_orig[valid_orig]), np.mean(dtnd[valid_new])),
        (
            "Median DTND (valid)",
            np.median(dtnd_orig[valid_orig]),
            np.median(dtnd[valid_new]),
        ),
    ]:
        delta = float(new) - float(orig)
        print(
            f"  {label:<30s}  {float(orig):>12.4f}  {float(new):>12.4f}  {delta:>+12.4f}"
        )

    # Per-bin HAND comparison
    valid_both = valid_orig & valid_new
    if np.any(valid_both):
        hand_diff = hand[valid_both] - hand_orig[valid_both]
        print(
            f"\n  HAND difference (Swenson - original) where both valid:"
            f"\n    mean={np.mean(hand_diff):+.4f}m, "
            f"median={np.median(hand_diff):+.4f}m, "
            f"std={np.std(hand_diff):.4f}m"
        )

    # --- Baseline (no filters, w1 fit) ---
    results = {}

    print("\n  --- N.baseline: Swenson conditioning, no pixel filters ---")
    results["N_baseline"] = run_test(
        "N.baseline: Swenson conditioning, no filters",
        hand,
        dtnd,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
        use_w1_fit=True,
    )

    # --- With Swenson-style valid mask ---
    print("\n  --- N.swenson_mask: + Swenson valid mask (include neg HAND) ---")
    results["N_swenson_mask"] = run_test(
        "N.swenson_mask: + Swenson valid mask",
        hand,
        dtnd,
        slope,
        aspect,
        area,
        did,
        res_m,
        pub,
        swenson_style=True,
        use_w1_fit=True,
    )

    # --- Full Swenson pipeline (tail + flood + clip + skip + mask + w1) ---
    print("\n  --- N.full: + all pixel filters ---")
    keep_idx = tail_index(dtnd, hand)
    h_n = hand[keep_idx].copy()
    d_n = dtnd[keep_idx].copy()
    s_n = slope[keep_idx]
    a_n = aspect[keep_idx]
    ar_n = area[keep_idx]
    did_n = did[keep_idx]
    ff_n = fflood[keep_idx]

    h_n = apply_flood_filter(h_n, ff_n)
    d_n[d_n < 1.0] = 1.0

    n_tail = hand.size - keep_idx.size
    n_flood = int(np.sum(h_n == -1))
    print(
        f"    After filters: {h_n.size} pixels "
        f"(removed {n_tail} by tail, {n_flood} marked flood)"
    )

    results["N_full"] = run_test(
        "N.full: Swenson conditioning + all filters",
        h_n,
        d_n,
        s_n,
        a_n,
        ar_n,
        did_n,
        res_m,
        pub,
        skip_zero_hand_bins=True,
        swenson_style=True,
        use_w1_fit=True,
    )

    # --- Per-element HAND comparison against published ---
    print("\n  --- Per-element mean HAND: Swenson-conditioned vs published ---")
    baseline_elems = results["N_baseline"]["elements"]
    full_elems = results["N_full"]["elements"]
    pub_height = pub["height"]

    print(
        f"  {'Elem':>4s}  {'Aspect':<6s}  {'Bin':>3s}  "
        f"{'Published':>9s}  {'N.base':>8s}  {'N.full':>8s}  "
        f"{'Offset(b)':>9s}  {'Offset(f)':>9s}"
    )
    print("  " + "-" * 72)

    offsets_base = []
    offsets_full = []
    for i in range(16):
        asp_idx = i // N_HAND_BINS
        h_idx = i % N_HAND_BINS
        h_pub = pub_height[i]
        h_base = baseline_elems[i]["height"]
        h_full = full_elems[i]["height"]
        off_b = h_base - h_pub
        off_f = h_full - h_pub
        offsets_base.append(off_b)
        offsets_full.append(off_f)
        print(
            f"  {i:>4d}  {ASPECT_NAMES[asp_idx]:<6s}  {h_idx:>3d}  "
            f"{h_pub:>9.4f}  {h_base:>8.4f}  {h_full:>8.4f}  "
            f"{off_b:>+9.4f}  {off_f:>+9.4f}"
        )

    print(
        f"\n  Mean HAND offset (baseline): {np.mean(offsets_base):+.4f}m "
        f"(was ~+0.3m before conditioning)"
    )
    print(f"  Mean HAND offset (full):     {np.mean(offsets_full):+.4f}m")

    # --- Summary ---
    print("\n  --- Test N summary ---")
    print(
        f"  {'Config':<45s}  {'Height':>7s}  {'Dist':>7s}  "
        f"{'Slope':>7s}  {'Aspect':>7s}  {'Width':>7s}  {'AreaFr':>7s}"
    )
    print("  " + "-" * 95)
    for key, label in [
        ("N_baseline", "N.baseline (conditioning only, w1)"),
        ("N_swenson_mask", "N.swenson_mask (+ neg-HAND mask)"),
        ("N_full", "N.full (+ all filters)"),
    ]:
        c = results[key]["correlations"]
        print(
            f"  {label:<45s}  {c['height']:>7.4f}  {c['distance']:>7.4f}  "
            f"{c['slope']:>7.4f}  {c['aspect']:>7.4f}  {c['width']:>7.4f}  "
            f"{c['area_fraction']:>7.4f}"
        )

    return results


# ===========================================================================
# Part 9: A_thresh sweep (Test O)
# ===========================================================================
def run_test_o(pub: dict, lc_regions: list) -> dict:
    """
    Test O: Sweep A_thresh to find optimal stream network density.

    Runs DEM conditioning + flow routing once, then for each A_thresh value
    recomputes only the channel mask + HAND/DTND. This avoids redundant
    conditioning (~15 min) per threshold.

    Parameters
    ----------
    pub : dict
        Published 16-element parameter vectors.
    lc_regions : list
        Region results from compute_lc(), for reporting full-tile native Lc.
    """
    print("\n" + "=" * 78)
    print("TEST O: A_THRESH SWEEP")
    print("=" * 78)
    print("  Sweeping accumulation threshold to find which A_thresh maximizes")
    print("  area_fraction correlation. Routing computed once; only channel mask")
    print("  + HAND/DTND recomputed per threshold.")

    # Report full-tile native Lc
    native = [r for r in lc_regions if r["label"] == "full_native"]
    if native:
        r = native[0]
        lc_m = r["lc_px"] * r["res"]
        a_thresh_native = int(0.5 * r["lc_px"] ** 2)
        print(f"\n  Full-tile native FFT: Lc = {r['lc_px']:.1f} px ({lc_m:.0f} m)")
        print(f"  → A_thresh = {a_thresh_native}")

    # Step 1: One-time DEM conditioning + flow routing
    print("\n  --- Setup: DEM conditioning + flow routing ---")
    t0 = time.time()
    setup = setup_flow_routing(MERIT_DEM)
    grid = setup["grid"]
    gc_row_slice = setup["gc_row_slice"]
    gc_col_slice = setup["gc_col_slice"]
    print(f"  Setup complete ({time.time() - t0:.0f}s)")

    # A_thresh-independent arrays (pre-computed, flattened)
    slope_flat = setup["slope_gc"].flatten()
    aspect_flat = setup["aspect_gc"].flatten()
    area_flat = setup["pixel_areas"].flatten()
    res_m = setup["res_m"]
    total_pixels = setup["slope_gc"].size

    # Step 2: Sweep A_thresh values
    sweep_values = [20, 25, 28, 30, 33, 34, 36, 38, 40, 42, 46, 50, 55, 60, 70, 80, 100]

    results = []

    for a_thresh in sweep_values:
        t1 = time.time()

        # Re-compute channel mask + HAND/DTND for this threshold
        acc_mask = grid.acc > a_thresh
        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")
        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
            routing="d8",
        )

        # Extract A_thresh-dependent gridcell arrays
        hand_gc = np.array(grid.hand)[gc_row_slice, gc_col_slice].flatten()
        dtnd_gc = np.array(grid.dtnd)[gc_row_slice, gc_col_slice].flatten()
        drainage_id_gc = np.array(grid.drainage_id)[
            gc_row_slice, gc_col_slice
        ].flatten()

        # Stream pixel count within gridcell
        channel_gc = np.array(grid.channel_mask)[gc_row_slice, gc_col_slice]
        n_stream_gc = int(np.sum(channel_gc))
        stream_pct = 100 * n_stream_gc / total_pixels

        # Compute elements + correlations (w^1 fit to match best known config)
        result = compute_elements(
            hand_gc,
            dtnd_gc,
            slope_flat,
            aspect_flat,
            area_flat,
            drainage_id_gc,
            res_m,
            use_w1_fit=True,
        )
        corrs = compute_all_correlations(result["elements"], pub)

        lc_px = np.sqrt(2 * a_thresh)
        lc_m = lc_px * res_m

        elapsed = time.time() - t1

        results.append(
            {
                "a_thresh": a_thresh,
                "lc_px": lc_px,
                "lc_m": lc_m,
                "n_stream_gc": n_stream_gc,
                "stream_pct": stream_pct,
                "correlations": corrs,
                "elapsed": elapsed,
            }
        )

        print(
            f"  A_thresh={a_thresh:>3d}  "
            f"area_frac={corrs['area_fraction']:.4f}  "
            f"stream={stream_pct:.2f}%  ({elapsed:.0f}s)"
        )

    # Print sweep table
    print("\n  --- Sweep results ---")
    print(
        f"  {'A_thresh':>8s}  {'Lc_px':>6s}  {'Lc_m':>6s}  "
        f"{'stream%':>7s}  {'height':>7s}  {'dist':>7s}  {'slope':>7s}  "
        f"{'aspect':>7s}  {'width':>7s}  {'area_fr':>7s}"
    )
    print("  " + "-" * 88)

    for r in results:
        c = r["correlations"]
        print(
            f"  {r['a_thresh']:>8d}  {r['lc_px']:>6.1f}  {r['lc_m']:>6.0f}  "
            f"{r['stream_pct']:>6.2f}%  {c['height']:>7.4f}  {c['distance']:>7.4f}  "
            f"{c['slope']:>7.4f}  {c['aspect']:>7.4f}  {c['width']:>7.4f}  "
            f"{c['area_fraction']:>7.4f}"
        )

    # Find best A_thresh for area_fraction
    best_idx = max(
        range(len(results)),
        key=lambda i: results[i]["correlations"]["area_fraction"],
    )
    best = results[best_idx]
    best_a = best["a_thresh"]
    best_lc_px = best["lc_px"]
    best_corr = best["correlations"]["area_fraction"]

    # Find current (A_thresh=33 or 34) for comparison
    current_corrs = [
        r["correlations"]["area_fraction"] for r in results if r["a_thresh"] in (33, 34)
    ]
    current_corr = max(current_corrs) if current_corrs else float("nan")

    print(
        f"\n  Best A_thresh: {best_a} "
        f"(Lc_px = {best_lc_px:.1f}, area_frac = {best_corr:.4f})"
    )
    print(f"  Current A_thresh ~33-34 (area_frac = {current_corr:.4f})")
    print(f"  Improvement: {best_corr - current_corr:+.4f}")

    # Full-tile native comparison
    if native:
        native_a = int(0.5 * native[0]["lc_px"] ** 2)
        native_results = [r for r in results if r["a_thresh"] == native_a]
        if native_results:
            nc = native_results[0]["correlations"]["area_fraction"]
            print(f"  Full-tile native A_thresh={native_a} (area_frac = {nc:.4f})")

    # Interpretation
    if abs(best_corr - current_corr) < 0.01:
        print(
            "\n  Conclusion: A_thresh has minimal effect on area_fraction."
            "\n  The 0.82 gap is NOT from Lc differences."
        )
    else:
        print(
            f"\n  Conclusion: A_thresh matters. "
            f"Optimal Lc_px ~ {best_lc_px:.1f} (vs current ~8.25)."
        )
        if best_corr > current_corr:
            print(
                f"  Area fraction improves by {best_corr - current_corr:+.4f} "
                f"at A_thresh={best_a}."
            )

    return {"sweep": results, "best_idx": best_idx}


# ===========================================================================
# Main
# ===========================================================================
def main():
    start_time = time.time()

    for path, label in [(MERIT_DEM, "MERIT DEM"), (PUBLISHED_NC, "Published data")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # Step 1: Compute Lc
    print("=== Step 1: Characteristic Length Scale (FFT) ===")
    t0 = time.time()
    lc = compute_lc(MERIT_DEM)
    accum_threshold = int(0.5 * lc["median_lc_px"] ** 2)
    for r in lc["regions"]:
        lc_m = r["lc_px"] * r["res"]
        print(f"    {r['label']:<15s}: Lc = {r['lc_px']:.1f} px ({lc_m:.0f} m)")
    print(f"  Median Lc: {lc['median_lc_px']:.1f} px, A_thresh = {accum_threshold}")
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 2: Flow routing
    print("\n=== Step 2: Flow Routing + HAND/DTND ===")
    t0 = time.time()
    arrays = run_flow_routing(MERIT_DEM, accum_threshold)
    print(f"  Arrays: {arrays['hand'].size} pixels")
    print(f"  HAND: min={arrays['hand'].min():.2f}, max={arrays['hand'].max():.2f}")
    print(f"  DTND: min={arrays['dtnd'].min():.2f}, max={arrays['dtnd'].max():.2f}")
    print(
        f"  fflood: min={arrays['fflood'].min():.4f}, max={arrays['fflood'].max():.4f}"
    )
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 3: Load published data
    print("\n=== Step 3: Load Published Data ===")
    pub = load_published()

    # Step 3b: Negative-HAND and aspect boundary diagnostics
    print("\n=== Step 3b: Negative-HAND & Aspect Boundary Diagnostics ===")
    print_negative_hand_diagnostics(arrays)

    # Step 4: Baseline computation + visibility
    print("\n=== Step 4: Baseline Computation ===")
    t0 = time.time()
    baseline = compute_elements(
        arrays["hand"],
        arrays["dtnd"],
        arrays["slope"],
        arrays["aspect"],
        arrays["area"],
        arrays["drainage_id"],
        arrays["res_m"],
    )
    print_visibility(baseline["elements"], pub, baseline["diagnostics"])
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 4b: A_cumsum curve diagnostics (Test J)
    print("\n=== Step 4b: A_cumsum(d) Curve Diagnostics (Test J) ===")
    print_acumsum_diagnostics(arrays)

    # Step 4c: HAND CDF at bin boundary (Test K)
    print("\n=== Step 4c: HAND CDF at Bin-0 Boundary (Test K) ===")
    print_hand_cdf_diagnostics(arrays)

    # Step 5: Hypothesis testing
    print("\n=== Step 5: Hypothesis Testing ===")
    t0 = time.time()
    test_results = run_hypothesis_tests(arrays, pub)
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 6: Sensitivity map
    print("\n=== Step 6: Sensitivity Analysis ===")
    print_sensitivity_map(baseline["elements"], pub)

    # Summary
    print_summary(test_results)

    # Step 7: Boundary inference from published data (Test M) — TEMPORARY
    print("\n=== Step 7: Boundary Inference (Test M) ===")
    t0 = time.time()
    infer_published_boundaries(arrays, pub)
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 8: Swenson DEM conditioning (Test N)
    print("\n=== Step 8: Swenson DEM Conditioning (Test N) ===")
    t0 = time.time()
    run_test_n(accum_threshold, pub, arrays)
    print(f"  ({time.time() - t0:.0f}s)")

    # Step 9: A_thresh sweep (Test O)
    print("\n=== Step 9: A_thresh Sweep (Test O) ===")
    t0 = time.time()
    run_test_o(pub, lc["regions"])
    print(f"  ({time.time() - t0:.0f}s)")

    total = time.time() - start_time
    print(f"\nTotal time: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
