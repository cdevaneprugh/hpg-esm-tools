#!/usr/bin/env python3
"""
OSBS Hillslope Pipeline

Computes representative hillslope parameters from 1m NEON LIDAR data for OSBS
using the Swenson & Lawrence (2025) methodology. Produces CTSM-compatible NetCDF
output with 8 hillslope columns (1 aspect x 8 log-spaced elevation bins).

The processing algorithm follows merit_regression.py (the validated source of
truth), adapted for UTM CRS using shared modules:
  - spatial_scale.py: FFT-based characteristic length scale
  - hillslope_params.py: Binning, trapezoidal fit, width computation
  - dem_processing.py: Basin detection, open water identification

Phase decisions baked in:
  - Phase A: pgrid slope_aspect() and compute_hand() are UTM-aware
  - Phase B: Full 1m resolution, no subsampling (64GB sufficient)
  - Phase C: min_wavelength=20m filters k^2 micro-topographic artifact in FFT

Configuration:
    Set MOSAIC_PATH below to select the input DEM:
      - Production (default): data/mosaics/OSBS_production.tif (90 tiles, R4C5-R12C14)
      - Smoke test: data/neon/dtm/NEON_D03_OSBS_DP3_404000_3286000_DTM.tif (R6C10)
    The mosaic must be contiguous (no nodata pixels).

Usage:
    python scripts/osbs/run_pipeline.py

Environment variables:
    OUTPUT_DESCRIPTOR: Label for output directory (default: "production")
    PYSHEDS_FORK: Path to pysheds fork
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for batch jobs
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Proj as PyprojProj

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid  # noqa: E402

# Add parent directory for shared module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dem_processing import identify_basins, identify_open_water  # noqa: E402
from hillslope_params import (  # noqa: E402
    catchment_mean_aspect,
    circular_mean_aspect,
    compute_hand_bins_log,
    fit_trapezoidal_width,
    get_aspect_mask,
    quadratic,
    tail_index,
)
from spatial_scale import identify_spatial_scale_laplacian_dem  # noqa: E402


# =============================================================================
# Constants
# =============================================================================

# Pipeline configuration
PIXEL_SIZE = 1.0  # meters (Phase B: full 1m resolution, no subsampling)
MIN_WAVELENGTH = 20  # meters (Phase C: filters k^2 micro-topographic artifact)
NODATA_VALUE = -9999  # standardized nodata sentinel
NODATA_THRESHOLD = -9000  # threshold for valid data detection in GeoTIFFs
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # D8 flow direction mapping

# Hillslope binning (1 aspect x 8 log-spaced HAND bins)
# See docs/hillslope-binning-rationale.md for justification.
N_ASPECT_BINS = 1
N_HAND_BINS = 8
LOWEST_BIN_MAX = (
    2.0  # meters (Swenson constraint, satisfied by design with log spacing)
)
ASPECT_BINS = [(0, 360)]
ASPECT_NAMES = ["All"]

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"

# Mosaic path — change for smoke test vs production
# Smoke test (R6C10): DATA_DIR / "neon" / "dtm" / "NEON_D03_OSBS_DP3_404000_3286000_DTM.tif"
# Production (R4C5-R12C14): DATA_DIR / "mosaics" / "OSBS_production.tif"
MOSAIC_PATH = DATA_DIR / "mosaics" / "OSBS_production.tif"

# Output
OUTPUT_DESCRIPTOR = os.environ.get("OUTPUT_DESCRIPTOR", "production")
OUTPUT_TIMESTAMP = time.strftime("%Y-%m-%d")
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / f"{OUTPUT_TIMESTAMP}_{OUTPUT_DESCRIPTOR}"

# OSBS center coordinates (from reference file)
OSBS_CENTER_LAT = 29.689282
OSBS_CENTER_LON_360 = 278.006569  # 0-360 convention


# =============================================================================
# Utility Functions
# =============================================================================


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
        f"OSBS Spatial Scale Analysis ({OUTPUT_DESCRIPTOR})\n"
        f"Model: {result['model']}, Lc = {Lc_m:.0f} m"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_stream_network_plot(
    dem: np.ndarray,
    stream_mask: np.ndarray,
    bounds: dict,
    output_path: Path,
) -> None:
    """Create stream network overlay plot."""
    max_plot_size = 4000
    downsample = max(1, max(dem.shape) // max_plot_size)

    if downsample > 1:
        dem_plot = dem[::downsample, ::downsample]
        stream_plot = stream_mask[::downsample, ::downsample]
    else:
        dem_plot = dem
        stream_plot = stream_mask

    fig, ax = plt.subplots(figsize=(12, 10))

    dem_masked = np.ma.masked_less_equal(dem_plot, NODATA_THRESHOLD)

    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]
    im = ax.imshow(dem_masked, cmap="terrain", extent=extent, aspect="equal")

    stream_display = np.where(stream_plot > 0, 1, np.nan)
    ax.imshow(stream_display, cmap="Blues", alpha=0.7, extent=extent, aspect="equal")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation (m)")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")
    ax.set_title(
        f"OSBS Stream Network ({OUTPUT_DESCRIPTOR})\n"
        f"Stream cells: {np.sum(stream_mask > 0):,} "
        f"({100 * np.sum(stream_mask > 0) / stream_mask.size:.2f}%)"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hand_map_plot(hand: np.ndarray, bounds: dict, output_path: Path) -> None:
    """Create HAND map plot."""
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
        hand_display,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )

    plt.colorbar(im, ax=ax, shrink=0.8, label="HAND (m)")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")

    hand_valid = hand[hand > 0]
    ax.set_title(
        f"OSBS HAND ({OUTPUT_DESCRIPTOR})\n"
        f"Range: 0 - {np.max(hand_valid):.1f} m, "
        f"Median: {np.median(hand_valid):.1f} m"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hillslope_params_plot(params: dict, output_path: Path) -> None:
    """Create hillslope parameters summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    elements = params["elements"]
    n_bins = params["metadata"]["n_hand_bins"]
    n_aspects = params["metadata"]["n_aspect_bins"]
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_aspects, 2)))

    aspect_names = params["metadata"]["aspect_names"]
    bin_labels = range(1, n_bins + 1)
    bar_width = 0.7 / n_aspects

    for panel_idx, (ax, key, ylabel, title) in enumerate(
        [
            (axes[0, 0], "height", "Mean HAND (m)", "Height Above Nearest Drainage"),
            (axes[0, 1], "distance", "Mean DTND (m)", "Distance to Nearest Drainage"),
            (axes[1, 0], "area", "Area (km^2)", "Hillslope Element Area"),
            (axes[1, 1], "width", "Width (m)", "Lower Edge Width"),
        ]
    ):
        for i, asp in enumerate(aspect_names):
            start = i * n_bins
            values = [elements[j][key] for j in range(start, start + n_bins)]
            if key == "area":
                values = [v / 1e6 for v in values]
            ax.bar(
                [b + i * bar_width for b in bin_labels],
                values,
                width=bar_width * 0.9,
                label=asp,
                color=colors[i],
            )
        ax.set_xlabel("Elevation Bin")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if n_aspects > 1:
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
    stream_depth: float,
    stream_width: float,
    stream_slope: float,
    lc_m: float | None = None,
    accum_threshold: int | None = None,
    min_wavelength: float | None = None,
) -> None:
    """
    Write hillslope parameters to CTSM-compatible NetCDF file.

    Parameters
    ----------
    params : dict
        Hillslope parameters with "elements" list
    output_path : Path
        Output NetCDF file path
    center_lon : float
        Center longitude in 0-360 convention
    center_lat : float
        Center latitude in degrees north
    total_area_km2 : float
        Total area in km^2
    stream_depth : float
        Stream channel bankfull depth (m)
    stream_width : float
        Stream channel bankfull width (m)
    stream_slope : float
        Stream channel slope (m/m)
    lc_m : float, optional
        Characteristic length scale (m) for metadata
    accum_threshold : int, optional
        Flow accumulation threshold for metadata
    min_wavelength : float, optional
        FFT minimum wavelength filter (m) for metadata
    """
    import netCDF4 as nc

    elements = params["elements"]

    n_aspects = params["metadata"]["n_aspect_bins"]
    n_bins = params["metadata"]["n_hand_bins"]
    n_columns = n_aspects * n_bins

    elevation = np.zeros(n_columns)
    distance = np.zeros(n_columns)
    width = np.zeros(n_columns)
    area = np.zeros(n_columns)
    slope = np.zeros(n_columns)
    aspect = np.zeros(n_columns)
    hillslope_index = np.zeros(n_columns, dtype=np.int32)
    column_index = np.zeros(n_columns, dtype=np.int32)
    downhill_column_index = np.zeros(n_columns, dtype=np.int32)

    # NOTE: This loop assumes all bins are populated (no empty columns).
    # Swenson's pipeline has a compression step (rh:971-1012) that removes
    # empty bins, renumbers column_index/downhill_column_index, and sets
    # nhillcolumns to the actual count. We omit this because at OSBS with
    # 90M pixels and a single-aspect configuration, empty bins are not
    # possible. If this pipeline is applied to smaller domains or sites
    # with sparse topography, a compression step would be needed to avoid
    # writing zero-area columns that CTSM would try to route flow through.
    # See: docs/osbs-pipeline-divergence-audit-260316.md, issue #1.
    for i, elem in enumerate(elements):
        elevation[i] = elem["height"]
        distance[i] = elem["distance"]
        width[i] = elem["width"]
        area[i] = elem["area"]
        slope[i] = elem["slope"]
        aspect[i] = elem["aspect"] * np.pi / 180

        asp_idx = i // n_bins
        hillslope_index[i] = asp_idx + 1
        column_index[i] = i + 1

        bin_idx = i % n_bins
        if bin_idx == 0:
            downhill_column_index[i] = -9999
        else:
            downhill_column_index[i] = i  # 1-indexed column_index of previous column

    pct_hillslope = np.zeros(n_aspects)
    total_area_m2 = sum(elem["area"] for elem in elements)
    for asp_idx in range(n_aspects):
        asp_area = sum(elements[asp_idx * n_bins + j]["area"] for j in range(n_bins))
        if total_area_m2 > 0:
            pct_hillslope[asp_idx] = 100.0 * asp_area / total_area_m2
        else:
            pct_hillslope[asp_idx] = 100.0 / n_aspects

    # Placeholder — matches Swenson (all zeros). Under the current osbs2
    # config (hillslope_soil_profile_method='Uniform'), CTSM never reads
    # this field. Only matters if switched to 'FromFile'. See audit issue #2.
    bedrock_depth = np.zeros(n_columns)

    print_progress(f"  Writing NetCDF: {output_path.name}")

    with nc.Dataset(output_path, "w", format="NETCDF4") as ds:
        # Global attributes
        ds.creation_date = datetime.now().strftime("%Y-%m-%d")
        ds.source = "OSBS 1m NEON LIDAR processed with hpg-esm-tools/swenson pipeline"
        ds.conventions = "CF-1.6"
        ds.pixel_size_m = PIXEL_SIZE
        if lc_m is not None:
            ds.characteristic_length_m = float(lc_m)
        if accum_threshold is not None:
            ds.accumulation_threshold = int(accum_threshold)
        if min_wavelength is not None:
            ds.fft_min_wavelength_m = float(min_wavelength)

        # Dimensions
        ds.createDimension("lsmlat", 1)
        ds.createDimension("lsmlon", 1)
        ds.createDimension("nhillslope", n_aspects)
        ds.createDimension("nmaxhillcol", n_columns)

        # Coordinate variables
        var_lsmlat = ds.createVariable("lsmlat", "f8", ("lsmlat",))
        var_lsmlat[:] = center_lat

        var_lsmlon = ds.createVariable("lsmlon", "f8", ("lsmlon",))
        var_lsmlon[:] = center_lon

        var_latixy = ds.createVariable("LATIXY", "f8", ("lsmlat", "lsmlon"))
        var_latixy.units = "degrees north"
        var_latixy.long_name = "latitude"
        var_latixy[:] = center_lat

        var_longxy = ds.createVariable("LONGXY", "f8", ("lsmlat", "lsmlon"))
        var_longxy.units = "degrees east"
        var_longxy.long_name = "longitude"
        var_longxy[:] = center_lon

        var_area = ds.createVariable("AREA", "f8", ("lsmlat", "lsmlon"))
        var_area.units = "km^2"
        var_area.long_name = "area"
        var_area[:] = total_area_km2

        var_nhillcol = ds.createVariable("nhillcolumns", "i4", ("lsmlat", "lsmlon"))
        var_nhillcol.units = "unitless"
        var_nhillcol.long_name = "number of columns per landunit"
        var_nhillcol[:] = n_columns

        var_pcthillslope = ds.createVariable(
            "pct_hillslope", "f8", ("nhillslope", "lsmlat", "lsmlon")
        )
        var_pcthillslope.units = "per cent"
        var_pcthillslope.long_name = "percent hillslope of landunit"
        var_pcthillslope[:, 0, 0] = pct_hillslope

        var_hillndx = ds.createVariable(
            "hillslope_index", "i4", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_hillndx.units = "unitless"
        var_hillndx.long_name = "hillslope_index"
        var_hillndx[:, 0, 0] = hillslope_index

        var_colndx = ds.createVariable(
            "column_index", "i4", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_colndx.units = "unitless"
        var_colndx.long_name = "column index"
        var_colndx[:, 0, 0] = column_index

        var_dcolndx = ds.createVariable(
            "downhill_column_index",
            "i4",
            ("nmaxhillcol", "lsmlat", "lsmlon"),
        )
        var_dcolndx.units = "unitless"
        var_dcolndx.long_name = "downhill column index"
        var_dcolndx[:, 0, 0] = downhill_column_index

        var_elev = ds.createVariable(
            "hillslope_elevation",
            "f8",
            ("nmaxhillcol", "lsmlat", "lsmlon"),
        )
        var_elev.units = "m"
        var_elev.long_name = "hillslope elevation above channel"
        var_elev[:, 0, 0] = elevation

        var_dist = ds.createVariable(
            "hillslope_distance",
            "f8",
            ("nmaxhillcol", "lsmlat", "lsmlon"),
        )
        var_dist.units = "m"
        var_dist.long_name = "hillslope distance from channel"
        var_dist[:, 0, 0] = distance

        var_width = ds.createVariable(
            "hillslope_width", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_width.units = "m"
        var_width.long_name = "hillslope width"
        var_width[:, 0, 0] = width

        var_harea = ds.createVariable(
            "hillslope_area", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_harea.units = "m2"
        var_harea.long_name = "hillslope area"
        var_harea[:, 0, 0] = area

        var_slope = ds.createVariable(
            "hillslope_slope", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_slope.units = "m/m"
        var_slope.long_name = "hillslope slope"
        var_slope[:, 0, 0] = slope

        var_aspect = ds.createVariable(
            "hillslope_aspect", "f8", ("nmaxhillcol", "lsmlat", "lsmlon")
        )
        var_aspect.units = "radians"
        var_aspect.long_name = "hillslope aspect (clockwise from North)"
        var_aspect[:, 0, 0] = aspect

        var_bedrock = ds.createVariable(
            "hillslope_bedrock_depth",
            "f8",
            ("nmaxhillcol", "lsmlat", "lsmlon"),
        )
        var_bedrock.units = "meters"
        var_bedrock.long_name = "hillslope bedrock depth"
        var_bedrock[:, 0, 0] = bedrock_depth

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
    print_progress("  NetCDF summary:")
    print_progress(f"    Location: {center_lat:.4f}N, {center_lon:.4f}E (0-360)")
    print_progress(f"    Area: {total_area_km2:.2f} km^2")
    print_progress(f"    pct_hillslope: {pct_hillslope}")
    print_progress(
        f"    Stream: depth={stream_depth:.3f}m, "
        f"width={stream_width:.1f}m, slope={stream_slope:.6f}"
    )


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    """Run the OSBS hillslope pipeline."""
    start_time = time.time()

    print_section("OSBS Hillslope Pipeline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load DEM
    # =========================================================================
    print_section("Step 1: Loading DEM")

    if not MOSAIC_PATH.exists():
        print(f"ERROR: Mosaic not found: {MOSAIC_PATH}")
        sys.exit(1)

    print_progress(f"  Mosaic: {MOSAIC_PATH}")

    with rasterio.open(MOSAIC_PATH) as src:
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
    dem_valid = dem[dem > NODATA_THRESHOLD]

    print_progress(f"  Shape: {dem.shape}")
    print_progress(f"  CRS: {crs}")
    print_progress(f"  Pixel size: {pixel_size} m")
    print_progress(
        f"  Elevation range: {dem_valid.min():.1f} - {dem_valid.max():.1f} m"
    )
    print_progress(f"  Memory: {dem.nbytes / 1e9:.2f} GB")

    # =========================================================================
    # Step 2: Spatial Scale Analysis (FFT)
    # =========================================================================
    print_section("Step 2: Spatial Scale Analysis (FFT)")

    t0 = time.time()

    fft_result = identify_spatial_scale_laplacian_dem(
        dem,
        pixel_size=PIXEL_SIZE,
        min_wavelength=MIN_WAVELENGTH,
        blend_edges_n=50,
        zero_edges_n=50,
        verbose=True,
    )

    if not fft_result.get("validDEM", False):
        print("ERROR: FFT found no valid DEM data")
        sys.exit(1)

    lc_m = fft_result["spatialScale_m"]
    lc_px = fft_result["spatialScale"]
    accum_threshold = int(0.5 * lc_px**2)

    print_progress(f"  Lc: {lc_m:.0f} m ({lc_px:.1f} px)")
    print_progress(f"  Accumulation threshold: {accum_threshold} cells")
    print_progress(f"  FFT time: {time.time() - t0:.1f} seconds")

    create_spectral_plot(fft_result, OUTPUT_DIR / "lc_spectral_analysis.png")
    print_progress(f"  Saved: {OUTPUT_DIR / 'lc_spectral_analysis.png'}")

    # =========================================================================
    # Step 3: Region Extraction + DEM Processing
    # =========================================================================
    print_section("Step 3: DEM Processing")

    t0 = time.time()

    # --- 3a: Validate DEM (no nodata allowed) ---
    valid_data_mask = dem > NODATA_THRESHOLD
    valid_frac = np.sum(valid_data_mask) / valid_data_mask.size
    print_progress(f"  Valid data fraction: {valid_frac:.2%}")

    if valid_frac < 1.0:
        nodata_count = int(np.sum(~valid_data_mask))
        print(
            f"ERROR: Mosaic contains {nodata_count:,} nodata pixels "
            f"({1 - valid_frac:.2%}). Pipeline requires a contiguous mosaic "
            f"with no nodata."
        )
        sys.exit(1)

    dem_for_routing = dem.copy()
    valid_mask = np.ones(dem.shape, dtype=bool)
    region_transform = transform

    print_progress(
        f"  Processing region: {dem_for_routing.shape} "
        f"({dem_for_routing.size:,} pixels)"
    )

    # --- 3b: Basin detection (pre-conditioning) ---
    print_progress("  Detecting basins in raw DEM...")
    basin_pre_mask = identify_basins(dem_for_routing, nodata=NODATA_VALUE)
    n_basin_pre = int(np.sum(basin_pre_mask > 0))
    print_progress(f"    Pre-conditioning basin pixels: {n_basin_pre}")
    dem_for_routing[basin_pre_mask > 0] = NODATA_VALUE

    # --- 3c: pysheds Grid creation ---
    grid = Grid()
    grid.add_gridded_data(
        dem_for_routing,
        data_name="dem",
        affine=region_transform,
        crs=PyprojProj(crs.to_proj4()),
        nodata=NODATA_VALUE,
    )

    # --- 3d: DEM conditioning ---
    print_progress("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")

    # Slope/aspect on original DEM (for water detection + final output)
    # Uses pgrid's Horn 1981 method, now UTM-aware via Phase A
    print_progress("  Computing slope/aspect (pgrid Horn 1981)...")
    grid.slope_aspect("dem")
    slope = np.array(grid.slope)
    aspect = np.array(grid.aspect)

    # Open water detection from slope field
    print_progress("  Detecting open water from slope field...")
    basin_boundary, basin_mask = identify_open_water(slope)
    n_water = int(np.sum(basin_mask > 0))
    n_boundary = int(np.sum(basin_boundary > 0))
    print_progress(f"    Open water: {n_water} px, boundary: {n_boundary} px")

    # Lower basin pixels in flooded DEM to force flow through them
    flooded_arr = np.array(grid.flooded)
    flooded_arr[basin_mask > 0] -= 0.1
    grid.add_gridded_data(
        flooded_arr,
        data_name="flooded",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )

    # Resolve flats with fallback to original DEM
    print_progress("  Resolving flats...")
    try:
        grid.resolve_flats("flooded", out_name="inflated")
    except ValueError:
        # Fall back to flooded DEM (post-fill_pits, post-fill_depressions).
        # Better than raw DEM: has connected drainage but unresolved flats.
        # Raw DEM has millions of LIDAR noise pits that fragment the stream network.
        print_progress("  WARNING: resolve_flats failed, falling back to flooded DEM")
        grid.add_gridded_data(
            np.array(grid.flooded),
            data_name="inflated",
            affine=grid.affine,
            crs=grid.crs,
            nodata=grid.nodata,
        )

    # --- 3e: Flow routing ---
    print_progress("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")

    # Re-mask basins after flowdir, before accumulation
    inflated_arr = np.array(grid.inflated)
    inflated_arr[basin_mask > 0] = np.nan
    grid.add_gridded_data(
        inflated_arr,
        data_name="inflated",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )

    print_progress("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    # Force basin boundaries into stream network
    acc_arr = np.array(grid.acc)
    acc_arr[basin_boundary > 0] = accum_threshold + 1
    grid.add_gridded_data(
        acc_arr,
        data_name="acc",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )

    # A_thresh safety valve
    max_acc = np.nanmax(acc_arr)
    if max_acc < accum_threshold:
        old_thresh = accum_threshold
        accum_threshold = int(max_acc / 100)
        print_progress(f"    A_thresh reduced: {old_thresh} -> {accum_threshold}")

    print_progress(f"  Max accumulation: {max_acc:.0f} cells")

    # --- 3f: Channel mask + stream network ---
    acc_mask = (grid.acc > accum_threshold) & np.isfinite(np.array(grid.inflated))
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

    stream_mask = grid.channel_mask
    stream_cells = np.sum(stream_mask > 0)
    stream_frac = stream_cells / stream_mask.size
    print_progress(f"  Stream cells: {stream_cells:,} ({stream_frac:.2%})")

    # Stream network extraction
    print_progress("  Extracting river network...")
    try:
        branches = grid.extract_river_network(
            "fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8"
        )
        print_progress(f"    Stream reaches: {len(branches['features'])}")
    except MemoryError:
        import resource

        max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print_progress(
            f"  WARNING: MemoryError in extract_river_network "
            f"(RSS: {max_rss_gb:.1f} GB)"
        )
        branches = None

    # Stream network length + slope
    grid.inflated_dem = grid.inflated  # alias needed by pgrid
    net_stats = None
    print_progress("  Computing network length and slope...")
    try:
        net_stats = grid.river_network_length_and_slope(
            "fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8"
        )
        print_progress(f"    Network length: {net_stats['length']:.0f} m")
        print_progress(f"    Mean network slope: {net_stats['slope']:.6f} m/m")
        print_progress(f"    Main channel length: {net_stats['mch_length']:.0f} m")
        print_progress(f"    Main channel slope: {net_stats['mch_slope']:.6f} m/m")
    except MemoryError:
        import resource

        max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print_progress(
            f"  WARNING: MemoryError in river_network_length_and_slope "
            f"(RSS: {max_rss_gb:.1f} GB)"
        )

    print_progress(f"  DEM processing time: {time.time() - t0:.1f} seconds")

    # =========================================================================
    # Step 4: HAND/DTND + Hillslope Classification
    # =========================================================================
    print_section("Step 4: HAND/DTND + Hillslope Classification")

    t0 = time.time()

    # HAND/DTND from inflated DEM (hydrological DTND, Phase A UTM-aware)
    print_progress("  Computing HAND/DTND...")
    grid.compute_hand(
        "fdir",
        "inflated",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    hand = np.array(grid.hand)
    dtnd = np.array(grid.dtnd)

    hand_positive = hand[(hand > 0) & np.isfinite(hand) & valid_mask]
    dtnd_positive = dtnd[(dtnd > 0) & valid_mask]
    print_progress(
        f"  HAND range: 0 - {np.max(hand_positive):.1f} m, "
        f"median: {np.median(hand_positive):.1f} m"
    )
    print_progress(
        f"  DTND range: 0 - {np.max(dtnd_positive):.0f} m, "
        f"median: {np.median(dtnd_positive):.0f} m"
    )

    # Hillslope classification (Swenson rh:1692)
    print_progress("  Computing hillslope classification...")
    grid.compute_hillslope(
        "fdir", "channel_mask", "bank_mask", dirmap=DIRMAP, routing="d8"
    )

    # Catchment-level aspect averaging (Swenson rh:1725-1751).
    # Still applied with 1-aspect binning: replaces per-pixel aspect noise with
    # catchment-side circular means, improving the per-element mean_aspect stored
    # in the NetCDF (used by CTSM for insolation correction via shr_orb_cosinc).
    print_progress("  Averaging aspect within catchment sides...")
    aspect = catchment_mean_aspect(
        np.array(grid.drainage_id), aspect, np.array(grid.hillslope)
    )

    print_progress(f"  HAND/DTND time: {time.time() - t0:.1f} seconds")

    # Plots
    create_stream_network_plot(
        dem_for_routing,
        stream_mask,
        bounds_dict,
        OUTPUT_DIR / "stream_network.png",
    )
    print_progress(f"  Saved: {OUTPUT_DIR / 'stream_network.png'}")

    create_hand_map_plot(hand, bounds_dict, OUTPUT_DIR / "hand_map.png")
    print_progress(f"  Saved: {OUTPUT_DIR / 'hand_map.png'}")

    # =========================================================================
    # Step 5: Hillslope Parameter Computation
    # =========================================================================
    print_section(
        f"Step 5: Hillslope Parameters ({N_ASPECT_BINS * N_HAND_BINS} elements)"
    )

    t0 = time.time()

    # --- 5a: Filtering ---

    # Flood filter: mark basin pixels with low HAND as invalid
    basin_mask_flat = basin_mask.flatten()
    n_flooded = int(np.sum(basin_mask_flat > 0))
    if n_flooded > 0:
        hand_flat_temp = hand.flatten().copy()
        for ft in np.linspace(0, 20, 50):
            flooded_low_hand = (basin_mask_flat > ft) & (hand_flat_temp < 2.0)
            if np.sum(flooded_low_hand) / n_flooded < 0.95:
                hand_flat_temp[flooded_low_hand] = -1
                n_marked = int(np.sum(flooded_low_hand))
                print_progress(
                    f"    Flood filter: threshold={ft:.2f}, "
                    f"marked {n_marked} px invalid"
                )
                hand = hand_flat_temp.reshape(hand.shape)
                break

    # Flatten all fields
    hand_flat = hand.flatten()
    dtnd_flat = dtnd.flatten()
    slope_flat = slope.flatten()
    aspect_flat = aspect.flatten()
    valid_mask_flat = valid_mask.flatten()

    pixel_area = PIXEL_SIZE * PIXEL_SIZE
    area_flat = np.full(hand_flat.shape, pixel_area)

    # DTND tail removal — applied BEFORE basin masking and DTND clip, matching
    # Swenson's order (rh:666-676). Basin pixels with long flow paths should be
    # included in the exponential fit so the tail threshold accounts for them.
    # This matters once synthetic lake bottoms / basin detection is active.
    finite_hand = np.isfinite(hand_flat)
    tail_ind = tail_index(dtnd_flat[finite_hand], hand_flat[finite_hand])
    finite_indices = np.where(finite_hand)[0]
    keep_tail = np.zeros(hand_flat.shape, dtype=bool)
    keep_tail[finite_indices[tail_ind]] = True
    n_removed = int(np.sum(finite_hand) - np.sum(keep_tail))
    print_progress(f"    DTND tail removal: {n_removed} px removed")

    # DTND minimum clip (Swenson rh:699-700)
    smallest_dtnd = 1.0  # meters — Swenson's fixed minimum
    dtnd_flat[dtnd_flat < smallest_dtnd] = smallest_dtnd

    # Valid mask: tail-filtered + isfinite(hand) + valid_mask for nodata exclusion
    valid = keep_tail & valid_mask_flat

    # Basin masking: remove pre-conditioning basin pixels
    basin_pre_flat = basin_pre_mask.flatten()
    n_basin = int(np.sum(basin_pre_flat > 0))
    if n_basin > 0:
        non_flat_fraction = np.sum(basin_pre_flat == 0) / basin_pre_flat.size
        if non_flat_fraction > 0.01:
            valid = valid & (basin_pre_flat == 0)
            print_progress(f"    Basin masking: {n_basin} px removed")
        else:
            print_progress(
                f"  WARNING: Region too flat (non_flat={non_flat_fraction:.4f})"
            )

    # Extract drainage_id before applying valid filter
    drainage_id_flat = np.array(grid.drainage_id).flatten()

    # Apply valid filter to all arrays
    print_progress(
        f"  Valid pixels: {np.sum(valid):,} ({100 * np.sum(valid) / valid.size:.1f}%)"
    )

    hand_flat = hand_flat[valid]
    dtnd_flat = dtnd_flat[valid]
    slope_flat = slope_flat[valid]
    aspect_flat = aspect_flat[valid]
    area_flat = area_flat[valid]
    drainage_id_flat = drainage_id_flat[valid]

    # --- 5b: HAND bins ---
    print_progress("  Computing HAND bins...")
    hand_bounds = compute_hand_bins_log(hand_flat, n_bins=N_HAND_BINS)
    print_progress(f"    HAND bins ({len(hand_bounds) - 1}): {hand_bounds}")

    # --- 5c: Per-aspect parameter computation ---
    params = {
        "metadata": {
            "n_aspect_bins": N_ASPECT_BINS,
            "n_hand_bins": N_HAND_BINS,
            "aspect_bins": ASPECT_BINS,
            "aspect_names": ASPECT_NAMES,
            "hand_bounds": hand_bounds.tolist(),
            "accum_threshold": accum_threshold,
            "spatial_scale_m": float(lc_m),
            "min_wavelength_m": float(MIN_WAVELENGTH),
            "pixel_size_m": PIXEL_SIZE,
            "region_shape": list(dem_for_routing.shape),
            "bounds": bounds_dict,
        },
        "elements": [],
    }

    zero_element = {
        "height": 0,
        "distance": 0,
        "area": 0,
        "slope": 0,
        "aspect": 0,
        "width": 0,
    }

    # For 1x8 config: loops once with asp_bin=(0,360), capturing all pixels.
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        print_progress(f"  Processing {asp_name} aspect...")

        asp_mask = get_aspect_mask(aspect_flat, asp_bin)
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            print_progress(f"    No pixels in {asp_name} aspect")
            for h_idx in range(N_HAND_BINS):
                params["elements"].append(
                    {
                        "aspect_name": asp_name,
                        "aspect_bin": asp_idx,
                        "hand_bin": h_idx,
                        **zero_element,
                    }
                )
            continue

        # Count unique drainage IDs per aspect bin
        n_hillslopes = max(len(np.unique(drainage_id_flat[asp_indices])), 1)

        hillslope_frac = np.sum(area_flat[asp_indices]) / np.sum(area_flat)
        print_progress(
            f"    Pixels: {len(asp_indices):,}, "
            f"Fraction: {hillslope_frac:.1%}, "
            f"Hillslopes: {n_hillslopes}"
        )

        # Trapezoidal fit (min_dtnd = PIXEL_SIZE for UTM)
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
                    {
                        "indices": None,
                        "h_lower": h_lower,
                        "h_upper": h_upper,
                    }
                )
            else:
                bin_raw_areas.append(float(np.sum(area_flat[bin_indices])))
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
            else [1.0 / N_HAND_BINS] * N_HAND_BINS
        )
        fitted_areas = [trap_area * frac for frac in area_fractions]

        # Second pass: compute parameters
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
                        "aspect": float(
                            (asp_bin[0] + asp_bin[1]) / 2 % 360
                        ),  # 180 for (0,360)
                        "width": 0,
                    }
                )
                continue

            # HAND <= 0 bin skip (merit_regression.py:706-717)
            if np.mean(hand_flat[bin_indices]) <= 0:
                params["elements"].append(
                    {
                        "aspect_name": asp_name,
                        "aspect_bin": asp_idx,
                        "hand_bin": h_idx,
                        **zero_element,
                    }
                )
                continue

            mean_hand = float(np.mean(hand_flat[bin_indices]))
            mean_slope = float(np.nanmean(slope_flat[bin_indices]))
            # With 1-aspect, averages all pixel aspects in this HAND bin.
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

            # Distance: trapezoid-derived midpoint (merit_regression.py:739-748)
            da_dist = sum(fitted_areas[: h_idx + 1]) - fitted_areas[h_idx] / 2
            if trap_slope != 0:
                try:
                    distance = float(quadratic([trap_slope, trap_width, -da_dist]))
                except RuntimeError:
                    distance = median_dtnd
            else:
                distance = median_dtnd

            params["elements"].append(
                {
                    "aspect_name": asp_name,
                    "aspect_bin": asp_idx,
                    "hand_bin": h_idx,
                    "height": mean_hand,
                    "distance": distance,
                    "area": fitted_area,
                    "slope": mean_slope,
                    "aspect": mean_aspect,
                    "width": width,
                }
            )

            print_progress(
                f"    Bin {h_idx + 1}: h={mean_hand:.1f}m, "
                f"d={distance:.0f}m, w={width:.0f}m"
            )

    print_progress(f"  Parameter computation time: {time.time() - t0:.1f} seconds")

    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    print_section("Step 6: Saving Results")

    # Stream parameters
    total_area_m2 = sum(elem["area"] for elem in params["elements"])

    # Stream slope from network (validated approach)
    if net_stats is not None:
        stream_slope_val = net_stats["slope"]
    else:
        # Fallback: mean of lowest-bin slopes * 0.5
        lowest_bin_slopes = [
            params["elements"][asp_idx * N_HAND_BINS]["slope"]
            for asp_idx in range(N_ASPECT_BINS)
        ]
        stream_slope_val = (
            np.mean(lowest_bin_slopes) * 0.5 if any(lowest_bin_slopes) else 0.002
        )
        print_progress(
            f"  WARNING: Using fallback stream slope: {stream_slope_val:.6f}"
        )

    # INTERIM power laws for depth/width (Swenson rh:1104-1114)
    # Phase E will research OSBS-specific empirical relationships
    stream_depth = 0.001 * total_area_m2**0.4
    stream_width = 0.001 * total_area_m2**0.6

    print_progress(
        f"  Stream: depth={stream_depth:.3f}m, "
        f"width={stream_width:.1f}m, slope={stream_slope_val:.6f}"
    )

    # JSON output
    json_path = OUTPUT_DIR / "hillslope_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print_progress(f"  Saved: {json_path}")

    # Plots
    create_hillslope_params_plot(params, OUTPUT_DIR / "hillslope_params.png")
    print_progress(f"  Saved: {OUTPUT_DIR / 'hillslope_params.png'}")

    # NetCDF
    timetag = datetime.now().strftime("%y%m%d")
    nc_filename = f"hillslopes_osbs_{OUTPUT_DESCRIPTOR}_c{timetag}.nc"
    nc_path = OUTPUT_DIR / nc_filename
    total_area_km2 = total_area_m2 / 1e6

    write_hillslope_netcdf(
        params=params,
        output_path=nc_path,
        center_lon=OSBS_CENTER_LON_360,
        center_lat=OSBS_CENTER_LAT,
        total_area_km2=total_area_km2,
        stream_depth=stream_depth,
        stream_width=stream_width,
        stream_slope=stream_slope_val,
        lc_m=lc_m,
        accum_threshold=accum_threshold,
        min_wavelength=MIN_WAVELENGTH,
    )

    # Summary text
    summary_path = OUTPUT_DIR / f"{OUTPUT_DESCRIPTOR}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"OSBS Hillslope Pipeline Summary ({OUTPUT_DESCRIPTOR})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {MOSAIC_PATH}\n")
        f.write(f"Region: {dem_for_routing.shape} pixels\n")
        f.write(f"Resolution: {PIXEL_SIZE} m (Phase B: no subsampling)\n\n")

        f.write("Spatial Scale Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model: {fft_result['model']}\n")
        f.write(f"  Lc: {lc_m:.0f} m ({lc_px:.1f} px)\n")
        f.write(f"  A_thresh: {accum_threshold}\n")
        f.write(f"  min_wavelength: {MIN_WAVELENGTH} m\n\n")

        f.write("Stream Network:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Stream cells: {stream_cells:,} ({stream_frac:.2%})\n")
        if net_stats is not None:
            f.write(f"  Network length: {net_stats['length']:.0f} m\n")
            f.write(f"  Network slope: {net_stats['slope']:.6f} m/m\n")
        f.write(f"  Stream depth: {stream_depth:.3f} m (INTERIM power law)\n")
        f.write(f"  Stream width: {stream_width:.1f} m (INTERIM power law)\n")
        f.write(f"  Stream slope: {stream_slope_val:.6f} m/m\n\n")

        f.write("HAND:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Range: 0 - {np.max(hand_positive):.1f} m\n")
        f.write(f"  Median: {np.median(hand_positive):.1f} m\n")
        f.write(f"  HAND bins: {hand_bounds.tolist()}\n\n")

        f.write(f"Hillslope Elements ({N_ASPECT_BINS * N_HAND_BINS} total):\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Aspect':<10} {'Bin':<5} {'Height':<8} "
            f"{'Distance':<10} {'Area':<10} {'Width':<8}\n"
        )
        f.write("-" * 60 + "\n")

        for elem in params["elements"]:
            f.write(
                f"{elem['aspect_name']:<10} {elem['hand_bin'] + 1:<5} "
                f"{elem['height']:<8.1f} {elem['distance']:<10.0f} "
                f"{elem['area'] / 1e6:<10.3f} {elem['width']:<8.0f}\n"
            )

        f.write(f"\nNetCDF: {nc_filename}\n")
        f.write(f"Total time: {time.time() - start_time:.1f} seconds\n")

    print_progress(f"  Saved: {summary_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("Pipeline Complete")

    total_time = time.time() - start_time
    print_progress(f"Total: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print_progress(f"Lc: {lc_m:.0f} m, A_thresh: {accum_threshold}")
    print_progress(f"Streams: {stream_frac:.2%}")
    print_progress(f"HAND: 0 - {np.max(hand_positive):.1f} m")
    print_progress(f"Output: {OUTPUT_DIR}")
    print_progress(f"NetCDF: {nc_path}")


if __name__ == "__main__":
    main()
