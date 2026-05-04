#!/usr/bin/env python3
"""
OSBS Hillslope Pipeline

Computes representative hillslope parameters from 1m NEON LIDAR data for OSBS
using the Swenson & Lawrence (2025) methodology. Produces a CTSM-compatible
NetCDF with 25 hillslope columns: 1 lake column at chain index 1 + 24 land
bins (1 aspect x 24 HAND bins, raw-HAND scheme with Q01/Q99 trim, locked
Phase E.5 design 2026-05-04).

The processing algorithm follows merit_regression.py (the validated source of
truth), adapted for UTM CRS using shared modules:
  - spatial_scale.py: FFT-based characteristic length scale
  - hillslope_params.py: Binning, trapezoidal fit, width computation
  - dem_processing.py: Basin detection

Phase decisions baked in:
  - Phase A: pgrid compute_hand() is UTM-aware
  - Phase B: Full 1m resolution, no subsampling (64GB sufficient)
  - Phase C: min_wavelength=20m filters k^2 micro-topographic artifact in FFT
  - Phase E: NEON DP3.30025.001 slope/aspect replaces pgrid computation
  - Phase E.5: Q01/Q99 raw-HAND trim with true discard, 24-bin TAI-focused
    scheme on raw_hand = hand - (flooded_orig - pit_filled), explicit
    channel/water-mask filter, lake column at chain index 1.

Configuration:
    Set MOSAIC_DIR below to select the input mosaics (DTM, slope, aspect):
      - Production (default): data/mosaics/production/ (90 tiles, R4C5-R12C14)
      - Smoke test: point individual paths at single NEON tiles (R6C10)
    The mosaics must be contiguous (no nodata pixels).

References:
    phases/E.5-bin-redesign.md  - Outlier strategy, bin layout, LIDAR error
                                  budget, working scheme rationale.
    docs/lake-column-ctsm-audit.md  - Lake column parameters (Sections 5.1-5.5,
                                      5.2.1) and pixel filter recipe (6.9).

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

from hillslope_params import (  # noqa: E402
    catchment_mean_aspect,
    circular_mean_aspect,
    compute_hand_bins_tai_focused,
    fit_trapezoidal_width,
    get_aspect_mask,
    quadratic,
    tail_index,
)
from scipy.ndimage import binary_erosion  # noqa: E402
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
SMALLEST_DTND_M = 1.0  # Swenson rh:699-700 — minimum DTND clamp

# Hillslope binning (Phase E.5: 24 land bins + 1 lake column = 25 total)
#
# Land bins: 1 aspect x 24 raw-HAND bins, hand-tuned TAI-focused scheme
# (12 FZ + 12 upland, 0.25 m floor in TAI core, smooth 2x width
# progression outward). Outermost edges set dynamically to Q01/Q99 of
# the cleaned land population per run; expected ~-6.34 m / +17.46 m on
# the production domain.
#
# Lake column: prepended at chain index 1 with hill_elev = -6.0 m
# (chain-bookkeeping value, see audit Section 5.2.1). Land columns
# shift up: former col 1 (lowest land bin) becomes col 2, ..., col 24
# becomes col 25.
#
# See:
#   - phases/E.5-bin-redesign.md "Working bin scheme" subsection
#   - docs/lake-column-ctsm-audit.md Sections 5.1-5.5 + 5.2.1 + 6.9
#   - hillslope_params.compute_hand_bins_tai_focused() for the edge list
N_ASPECT_BINS = 1
N_LAND_BINS = 24
N_LAKE_COLUMNS = 1
N_TOTAL_COLUMNS = N_ASPECT_BINS * N_LAND_BINS + N_LAKE_COLUMNS  # 25
ASPECT_BINS = [(0, 360)]
ASPECT_NAMES = ["All"]

# Lake column parameters (locked 2026-05-04 — see audit Sections 5.1-5.5).
# Inert under current osbs2 config (use_hillslope_routing=.false. and
# tdepth=0 clamp the lake-to-stream subsurface gradient to zero), so
# distance/width/slope/aspect/bedrock are mathematically inconsequential
# but must be present and finite. Lake column is at chain index 1.
LAKE_HILL_ELEV_M = -6.0  # PI direction; deeper than deepest land bin mean (-5.13 m)
LAKE_HILL_DISTANCE_M = 5.0  # PI direction; ~stream width
LAKE_HILL_SLOPE = 0.0  # PI direction; "lake bottom" framing (water surface horizontal)
LAKE_HILL_ASPECT_DEG = 0.0  # inconsequential for flat lake column
LAKE_HILL_BEDROCK_DEPTH_M = 0.0  # inert under Uniform soil profile method

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"

# Mosaic paths — change all three for smoke test vs production
# Smoke test (R6C10):
#   MOSAIC_DIR = DATA_DIR / "neon"
#   MOSAIC_PATH = MOSAIC_DIR / "dtm" / "NEON_D03_OSBS_DP3_404000_3286000_DTM.tif"
#   SLOPE_MOSAIC_PATH = MOSAIC_DIR / "slope" / "NEON_D03_OSBS_DP3_404000_3286000_Slope.tif"
#   ASPECT_MOSAIC_PATH = MOSAIC_DIR / "aspect" / "NEON_D03_OSBS_DP3_404000_3286000_Aspect.tif"
# Production (R4C5-R12C14):
MOSAIC_DIR = DATA_DIR / "mosaics" / "production"
MOSAIC_PATH = MOSAIC_DIR / "dtm.tif"
SLOPE_MOSAIC_PATH = MOSAIC_DIR / "slope.tif"
ASPECT_MOSAIC_PATH = MOSAIC_DIR / "aspect.tif"
WATER_MASK_PATH = MOSAIC_DIR / "water_mask.tif"

# Output
OUTPUT_DESCRIPTOR = os.environ.get("OUTPUT_DESCRIPTOR", "production")
OUTPUT_TIMESTAMP = time.strftime("%Y-%m-%d")
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / f"{OUTPUT_TIMESTAMP}_{OUTPUT_DESCRIPTOR}"

# Diagnostic mode: when SAVE_DIAGNOSTICS=1, save intermediate conditioning
# arrays (dem, pit_filled, flooded_orig, hand, dtnd, raw_hand, water_mask,
# wide_channel_mask) to OUTPUT_DIR/diagnostics/ for ad-hoc post-run debugging
# via numpy.load. Off by default for production runs.
SAVE_DIAGNOSTICS = os.environ.get("SAVE_DIAGNOSTICS", "0") == "1"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

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


def compute_lake_perimeter(water_mask: np.ndarray, pixel_size: float) -> float:
    """Boundary-pixel approximation of total NWI lake perimeter (meters).

    A boundary pixel is a water pixel with at least one non-water 4-neighbor.
    Approach: subtract the eroded mask from the original to isolate boundary
    pixels, then multiply count by pixel_size.

    Underestimates the true polygon perimeter by ~5-10% (diagonals not
    counted). Adequate for the lake column's `hill_width` field, which is
    mathematically inert under the current osbs2 configuration (routing off,
    lake-to-stream subsurface gradient clamped to zero by the empty-stream
    guard in SoilHydrologyMod). See docs/lake-column-ctsm-audit.md Section
    5.3 for the full discussion. If routing is ever enabled, switch to a
    shapefile-based polygon perimeter (geopandas + shapely).
    """
    eroded = binary_erosion(water_mask > 0)
    boundary = (water_mask > 0) & ~eroded
    return float(np.sum(boundary)) * pixel_size


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
    water_mask: np.ndarray | None = None,
) -> None:
    """Create stream network overlay on hillshade background."""
    from matplotlib.colors import LightSource
    from scipy.ndimage import binary_dilation

    max_plot_size = 4000
    downsample = max(1, max(dem.shape) // max_plot_size)

    if downsample > 1:
        dem_plot = dem[::downsample, ::downsample]
        stream_plot = stream_mask[::downsample, ::downsample]
        water_plot = (
            water_mask[::downsample, ::downsample] if water_mask is not None else None
        )
    else:
        dem_plot = dem
        stream_plot = stream_mask
        water_plot = water_mask

    # Dilate streams by 1px so single-pixel channels survive downsampling
    stream_dilated = binary_dilation(stream_plot > 0, iterations=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]

    # Hillshade background (gray, high contrast for overlays)
    dem_filled = np.where(dem_plot <= NODATA_THRESHOLD, np.nan, dem_plot)
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.nan_to_num(dem_filled, nan=0), vert_exag=3)
    ax.imshow(hillshade, cmap="gray", extent=extent, aspect="equal")

    # Water mask (light blue, semi-transparent)
    if water_plot is not None:
        water_display = np.where(water_plot > 0, 1.0, np.nan)
        ax.imshow(
            water_display,
            cmap="Blues",
            vmin=0,
            vmax=1.5,
            alpha=0.4,
            extent=extent,
            aspect="equal",
        )

    # Streams (red, high contrast against gray hillshade)
    from matplotlib.colors import ListedColormap

    stream_cmap = ListedColormap(["red"])
    stream_display = np.where(stream_dilated, 1.0, np.nan)
    ax.imshow(
        stream_display,
        cmap=stream_cmap,
        alpha=0.9,
        extent=extent,
        aspect="equal",
    )

    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")

    n_stream = int(np.sum(stream_mask > 0))
    n_water = int(np.sum(water_mask > 0)) if water_mask is not None else 0
    title = (
        f"OSBS Stream Network ({OUTPUT_DESCRIPTOR})\n"
        f"Stream: {n_stream:,} px ({100 * n_stream / stream_mask.size:.2f}%)"
    )
    if n_water > 0:
        title += (
            f"  |  Water (NWI): {n_water:,} px ({100 * n_water / water_mask.size:.1f}%)"
        )
    ax.set_title(title)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hand_map_plot(
    hand: np.ndarray,
    bounds: dict,
    output_path: Path,
    water_mask: np.ndarray | None = None,
) -> None:
    """Create HAND map plot. Water pixels masked out (not in HAND bins)."""
    max_plot_size = 4000
    downsample = max(1, max(hand.shape) // max_plot_size)

    if downsample > 1:
        hand_plot = hand[::downsample, ::downsample]
        water_plot = (
            water_mask[::downsample, ::downsample] if water_mask is not None else None
        )
    else:
        hand_plot = hand
        water_plot = water_mask

    fig, ax = plt.subplots(figsize=(12, 10))

    hand_display = np.where(hand_plot < 0, np.nan, hand_plot)
    if water_plot is not None:
        hand_display = np.where(water_plot > 0, np.nan, hand_display)
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
    """Create hillslope parameters summary plot.

    Phase E.5: elements[0] is the lake column; elements[1..N] are the land
    columns. Bars show the 24 land bins. The lake's hill_elev and DTND sit
    at the lower edge of their respective panels and are useful as a visual
    reference line — those panels get a navy axhline for the lake value.
    The lake's hill_area (~11 km^2) and hill_width (~48 km) are far off-
    scale relative to land bins, so an axhline on those panels would crush
    the land detail; the Area and Width panels intentionally omit the lake
    line. Lake values are still reported in the summary text and NetCDF
    metadata for cross-reference.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    elements = params["elements"]
    n_land_bins = params["metadata"]["n_land_bins"]
    n_aspects = params["metadata"]["n_aspect_bins"]
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_aspects, 2)))

    aspect_names = params["metadata"]["aspect_names"]
    bin_labels = range(1, n_land_bins + 1)
    bar_width = 0.7 / n_aspects

    # Skip lake column (element[0]) — bars are land bins only.
    land_elements = [e for e in elements if not e.get("is_lake")]
    lake_elements = [e for e in elements if e.get("is_lake")]

    # Per-panel config: (ax, key, ylabel, title, show_lake_line).
    # show_lake_line=False on Area and Width panels because the lake's
    # values are wildly off-scale and would crush the bar detail.
    panel_configs = [
        (
            axes[0, 0],
            "height",
            "Mean raw HAND (m)",
            "Height Above Nearest Drainage",
            True,
        ),
        (axes[0, 1], "distance", "Mean DTND (m)", "Distance to Nearest Drainage", True),
        (axes[1, 0], "area", "Area (km^2)", "Hillslope Element Area", False),
        (axes[1, 1], "width", "Width (m)", "Lower Edge Width", False),
    ]

    for ax, key, ylabel, title, show_lake_line in panel_configs:
        for i, asp in enumerate(aspect_names):
            start = i * n_land_bins
            values = [land_elements[j][key] for j in range(start, start + n_land_bins)]
            if key == "area":
                values = [v / 1e6 for v in values]
            ax.bar(
                [b + i * bar_width for b in bin_labels],
                values,
                width=bar_width * 0.9,
                label=asp,
                color=colors[i],
            )

        if show_lake_line and lake_elements:
            lake = lake_elements[0]  # single-aspect: at most one lake column
            v = lake[key] / 1e6 if key == "area" else lake[key]
            ax.axhline(
                v,
                color="navy",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"Lake: {v:.2f}",
            )

        ax.set_xlabel("Land bin (chain order, deepest -> ridge)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

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
    q01: float | None = None,
    q99: float | None = None,
    n_pre_trim: int | None = None,
    n_post_trim: int | None = None,
) -> None:
    """
    Write hillslope parameters to CTSM-compatible NetCDF file.

    Phase E.5 element ordering convention: elements[0] is the lake column
    (chain index 1, terminal sink); elements[1..N] are the land columns in
    chain order (lowest land at index 1, ridge at the end). All columns share
    `hillslope_index = 1` (single-aspect setup). The chain topology is:

        col 25 (ridge) -> col 24 -> ... -> col 3 -> col 2 (lowest land)
                       -> col 1 (lake) -> [-9999 stream sentinel]

    `downhill_column_index` is `-9999` for the lake (terminal) and `i` for
    every land column at chain index `i+1` (drains to the previous chain
    index). See docs/lake-column-ctsm-audit.md Sections 5.1 + 7.1 for
    rationale.

    Parameters
    ----------
    params : dict
        Hillslope parameters with "elements" list (lake at index 0, land
        bins at indices 1..N).
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
    q01, q99 : float, optional
        Phase E.5 outlier cutoffs in raw HAND (m). Stored as global attrs.
    n_pre_trim, n_post_trim : int, optional
        Pixel counts before/after the Q01/Q99 trim, for verifiability.
    """
    import netCDF4 as nc

    elements = params["elements"]

    n_aspects = params["metadata"]["n_aspect_bins"]
    n_land_bins = params["metadata"]["n_land_bins"]
    n_lake_columns = params["metadata"].get("n_lake_columns", 0)
    n_columns = len(elements)
    assert n_columns == n_aspects * n_land_bins + n_lake_columns, (
        f"Element count {n_columns} does not match "
        f"{n_aspects} aspects x {n_land_bins} bins + {n_lake_columns} lake"
    )

    elevation = np.zeros(n_columns)
    distance = np.zeros(n_columns)
    width = np.zeros(n_columns)
    area = np.zeros(n_columns)
    slope = np.zeros(n_columns)
    aspect = np.zeros(n_columns)
    hillslope_index = np.zeros(n_columns, dtype=np.int32)
    column_index = np.zeros(n_columns, dtype=np.int32)
    downhill_column_index = np.zeros(n_columns, dtype=np.int32)

    # Chain topology (Phase E.5):
    #   - Lake at i=0 -> column_index=1, downhill=-9999 (terminal sink)
    #   - Lowest land at i=1 -> column_index=2, downhill=1 (drains to lake)
    #   - Subsequent land at i=2..N -> column_index=i+1, downhill=i
    #
    # NOTE: this assumes single-aspect (all columns share hillslope_index=1).
    # For multi-aspect configurations a compression step would be needed to
    # handle empty bins; not relevant at OSBS with 90M pixels and 1 aspect.
    # See docs/osbs-pipeline-divergence-audit-260316.md issue #1.
    for i, elem in enumerate(elements):
        elevation[i] = elem["height"]
        distance[i] = elem["distance"]
        width[i] = elem["width"]
        area[i] = elem["area"]
        slope[i] = elem["slope"]
        aspect[i] = elem["aspect"] * np.pi / 180

        # Single-aspect: all columns (lake + land) share hillslope_index=1.
        hillslope_index[i] = 1
        column_index[i] = i + 1

        if i == 0:
            # Lake at chain index 1 — terminal sink.
            downhill_column_index[i] = -9999
        else:
            # Land columns drain to the previous chain index. Index 1 (lake)
            # receives the lowest land bin's flow at i=1.
            downhill_column_index[i] = i  # 1-indexed col_index of previous

    # pct_hillslope: single-aspect by design (see assert in main()). One
    # hillslope contains all 25 columns (lake + 24 land bins), which means
    # the single aspect's percentage is always 100% — independent of how
    # area is distributed across the columns. Lake's fractional weight
    # within the landunit is computed downstream by CTSM via the per-column
    # hill_area / total ratio (see audit Section 3); we don't pre-compute
    # it here.
    assert n_aspects == 1, "Writer assumes single-aspect (see main() assert)."
    pct_hillslope = np.array([100.0])

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

        # Phase E.5 metadata (locked design 2026-05-04). Recorded as global
        # attrs so the cutoffs and method are reproducible per-run without
        # reading them out of pipeline logs.
        ds.bin_scheme = "Phase E.5 TAI-focused 24-bin (locked 2026-05-04)"
        ds.binning_input = "raw_hand = hand - (flooded_orig - pit_filled)"
        ds.outlier_method = "Q01/Q99 percentile trim, true discard"
        ds.n_land_bins = int(n_land_bins)
        ds.n_lake_columns = int(n_lake_columns)
        if q01 is not None:
            ds.q01_cutoff_m = float(q01)
        if q99 is not None:
            ds.q99_cutoff_m = float(q99)
        if n_pre_trim is not None:
            ds.n_pixels_pre_trim = int(n_pre_trim)
        if n_post_trim is not None:
            ds.n_pixels_post_trim = int(n_post_trim)
        ds.lake_column_at_index = 1
        ds.lake_hill_elev_m = float(LAKE_HILL_ELEV_M)

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

    # Single-aspect lock. The pipeline + writer + lake column placement
    # all assume one hillslope (ASPECT_BINS = [(0, 360)]). At OSBS the
    # N/S/E/W aspect distribution is ~uniform over the small drainage
    # basins, so a 4-aspect split would split few real catchments and
    # produce noisy per-aspect statistics. If multi-aspect is ever
    # needed: revisit lake-at-i=0 placement, restore the per-aspect
    # area-summation logic in write_hillslope_netcdf, and decide which
    # hillslope_index the lake belongs to.
    assert N_ASPECT_BINS == 1, (
        "Pipeline is locked to single-aspect; multi-aspect support requires "
        "redesign of lake column placement and writer pct_hillslope logic."
    )

    print_section("OSBS Hillslope Pipeline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load Mosaics (DEM + NEON slope/aspect)
    # =========================================================================
    print_section("Step 1: Loading Mosaics")

    for path, name in [
        (MOSAIC_PATH, "DTM"),
        (SLOPE_MOSAIC_PATH, "slope"),
        (ASPECT_MOSAIC_PATH, "aspect"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} mosaic not found: {path}")
            sys.exit(1)

    print_progress(f"  DTM: {MOSAIC_PATH}")
    print_progress(f"  Slope: {SLOPE_MOSAIC_PATH}")
    print_progress(f"  Aspect: {ASPECT_MOSAIC_PATH}")

    with rasterio.open(MOSAIC_PATH) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    # Load NEON slope/aspect (DP3.30025.001)
    with rasterio.open(SLOPE_MOSAIC_PATH) as src:
        neon_slope_deg = src.read(1)
    with rasterio.open(ASPECT_MOSAIC_PATH) as src:
        neon_aspect = src.read(1)

    # Convert NEON slope from degrees to m/m
    slope = np.tan(np.deg2rad(neon_slope_deg))
    aspect = neon_aspect.copy()

    # Zero out nodata pixels in slope/aspect
    slope[neon_slope_deg <= NODATA_THRESHOLD] = 0.0
    aspect[neon_aspect <= NODATA_THRESHOLD] = 0.0

    assert dem.shape == slope.shape == aspect.shape, (
        f"Shape mismatch: DTM {dem.shape}, slope {slope.shape}, aspect {aspect.shape}"
    )

    # Load NWI water mask (optional — falls back to no masking if absent)
    if WATER_MASK_PATH.exists():
        with rasterio.open(WATER_MASK_PATH) as src:
            water_mask = src.read(1)
        assert dem.shape == water_mask.shape, (
            f"Shape mismatch: DTM {dem.shape}, water mask {water_mask.shape}"
        )
        n_water_px = int(np.sum(water_mask > 0))
        print_progress(f"  Water mask: {WATER_MASK_PATH.name}")
        print_progress(
            f"    Water pixels: {n_water_px:,} "
            f"({100 * n_water_px / water_mask.size:.1f}%)"
        )
    else:
        print_progress(f"  WARNING: Water mask not found: {WATER_MASK_PATH}")
        print_progress("    Proceeding without water masking")
        water_mask = np.zeros(dem.shape, dtype=np.uint8)

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

    print_progress(f"  Processing region: {dem.shape} ({dem.size:,} pixels)")

    # --- 3b: pysheds Grid creation ---
    grid = Grid()
    grid.add_gridded_data(
        dem,
        data_name="dem",
        affine=transform,
        crs=PyprojProj(crs.to_proj4()),
        nodata=NODATA_VALUE,
    )

    # --- 3c: DEM conditioning ---
    print_progress("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")

    # slope and aspect loaded from NEON mosaics in Step 1 (DP3.30025.001)

    # NWI water mask (loaded in Step 1, replaces slope-based detection)
    n_water = int(np.sum(water_mask > 0))
    print_progress(
        f"  NWI water mask: {n_water:,} px ({100 * n_water / water_mask.size:.1f}%)"
    )

    # Phase E.5: capture pre-water-lowering conditioning state to derive raw HAND.
    # The four-stage conditioning pipeline is: dem -> pit_filled -> flooded ->
    # (water-lowered) -> inflated. We need pit_filled and flooded BEFORE the
    # water-lowering line overwrites grid.flooded, so we materialize them here
    # unconditionally. dep_fill_arr = flooded_orig - pit_filled isolates the
    # depression-fill stage from the other (sub-cm) conditioning stages.
    # See phases/E.5-bin-redesign.md "Pipeline implementation steps" subsection
    # and docs/lake-column-ctsm-audit.md Section 6.2 for the decomposition.
    # Memory cost: ~720 MB peak (3 x ~240 MB float64 for 89M pixels).
    pit_filled_arr = np.array(grid.pit_filled)
    flooded_orig_arr = np.array(grid.flooded)
    dep_fill_arr = flooded_orig_arr - pit_filled_arr

    if SAVE_DIAGNOSTICS:
        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        print_progress(f"  Saving diagnostic arrays to {DIAGNOSTICS_DIR}/")
        np.save(DIAGNOSTICS_DIR / "dem.npy", np.array(grid.dem))
        np.save(DIAGNOSTICS_DIR / "pit_filled.npy", pit_filled_arr)
        np.save(DIAGNOSTICS_DIR / "flooded_orig.npy", flooded_orig_arr)
        np.save(DIAGNOSTICS_DIR / "water_mask.npy", water_mask)

    # Lower water pixels in flooded DEM to force flow through them
    flooded_arr = flooded_orig_arr.copy()
    flooded_arr[water_mask > 0] -= 0.1
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

    if SAVE_DIAGNOSTICS:
        np.save(DIAGNOSTICS_DIR / "inflated.npy", np.array(grid.inflated))

    # --- 3d: Flow routing ---
    print_progress("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")

    # Water pixels stay finite in inflated DEM (no NaN masking).
    # They participate in HAND computation via the wide channel mask.

    print_progress("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    # A_thresh safety valve
    max_acc = np.nanmax(np.array(grid.acc))
    if max_acc < accum_threshold:
        old_thresh = accum_threshold
        accum_threshold = int(max_acc / 100)
        print_progress(f"    A_thresh reduced: {old_thresh} -> {accum_threshold}")

    print_progress(f"  Max accumulation: {max_acc:.0f} cells")

    # --- 3e: Channel mask + stream network ---
    acc_mask = (grid.acc > accum_threshold) & np.isfinite(np.array(grid.inflated))
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8")

    stream_mask = grid.channel_mask
    stream_cells = np.sum(stream_mask > 0)
    stream_frac = stream_cells / stream_mask.size
    print_progress(f"  Stream cells: {stream_cells:,} ({stream_frac:.2%})")

    # Stream network extraction
    print_progress("  Extracting river network...")
    river_net = grid.extract_river_network(
        "fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8"
    )
    print_progress(f"    Stream reaches: {len(river_net['features'])}")

    # Stream network length + slope
    # pgrid.river_network_length_and_slope() accesses self.inflated_dem (pgrid.py:3269),
    # but resolve_flats stored output as grid.inflated (out_name="inflated").
    grid.inflated_dem = grid.inflated
    print_progress("  Computing network length and slope...")
    net_stats = grid.river_network_length_and_slope(
        "fdir", mask=acc_mask, dirmap=DIRMAP, routing="d8"
    )
    print_progress(f"    Network length: {net_stats['length']:.0f} m")
    print_progress(f"    Mean network slope: {net_stats['slope']:.6f} m/m")
    print_progress(f"    Main channel length: {net_stats['mch_length']:.0f} m")
    print_progress(f"    Main channel slope: {net_stats['mch_slope']:.6f} m/m")

    print_progress(f"  DEM processing time: {time.time() - t0:.1f} seconds")

    # =========================================================================
    # Step 4: HAND/DTND + Hillslope Classification
    # =========================================================================
    print_section("Step 4: HAND/DTND + Hillslope Classification")

    t0 = time.time()

    # --- Wide channel mask for HAND (natural streams + lake pixels) ---
    # Natural channel mask defines catchments; wide mask defines HAND reference.
    # Lake pixels are "stream" for HAND so land near lakes gets HAND relative
    # to lake surface. Lake pixels themselves get HAND=0.
    wide_channel_mask = (np.array(grid.channel_mask) > 0) | (water_mask > 0)
    wide_channel_id = np.array(grid.channel_id).copy()
    water_not_stream = (water_mask > 0) & (np.array(grid.channel_mask) == 0)
    wide_channel_id[water_not_stream] = 0  # dummy ID for lake pixels

    n_natural = int(np.sum(np.array(grid.channel_mask) > 0))
    n_wide = int(np.sum(wide_channel_mask > 0))
    n_dummy = int(np.sum(water_not_stream))
    print_progress(f"  Natural stream pixels: {n_natural:,}")
    print_progress(f"  Wide mask pixels (natural + lake): {n_wide:,}")
    print_progress(f"  Lake pixels with dummy channel_id: {n_dummy:,}")

    # HAND/DTND using wide mask (hydrological DTND, Phase A UTM-aware)
    print_progress("  Computing HAND/DTND (wide mask)...")
    grid.compute_hand(
        "fdir",
        "inflated",
        wide_channel_mask,
        wide_channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    hand = np.array(grid.hand)
    dtnd = np.array(grid.dtnd)

    # Phase E.5: derive raw HAND by removing the depression-fill component
    # of conditioning. Pixels in unmapped basins that fill_depressions raised
    # to spill elevation will have raw_hand < 0, reflecting their true depth
    # below stream level — these populate the FZ bins. Pixels never touched
    # by fill_depressions (dep_fill = 0) have raw_hand = hand. This is the
    # binning input for the 24-bin TAI-focused scheme.
    raw_hand = hand - dep_fill_arr

    if SAVE_DIAGNOSTICS:
        np.save(DIAGNOSTICS_DIR / "hand.npy", hand)
        np.save(DIAGNOSTICS_DIR / "dtnd.npy", dtnd)
        np.save(DIAGNOSTICS_DIR / "raw_hand.npy", raw_hand)
        np.save(DIAGNOSTICS_DIR / "wide_channel_mask.npy", wide_channel_mask)

    # HAND diagnostics split by pixel type
    hand_flat_diag = hand.flatten()
    water_flat_diag = water_mask.flatten()
    n_land_pos = int(np.sum((hand_flat_diag > 0) & np.isfinite(hand_flat_diag)))
    n_lake_zero = int(np.sum((water_flat_diag > 0) & (hand_flat_diag == 0)))
    n_nan = int(np.sum(np.isnan(hand_flat_diag)))
    print_progress(
        f"  HAND breakdown: land(>0)={n_land_pos:,}, "
        f"lake(=0)={n_lake_zero:,}, NaN={n_nan:,}"
    )

    hand_positive = hand[(hand > 0) & np.isfinite(hand)]
    dtnd_positive = dtnd[dtnd > 0]
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
        dem,
        stream_mask,
        bounds_dict,
        OUTPUT_DIR / "stream_network.png",
        water_mask=water_mask,
    )
    print_progress(f"  Saved: {OUTPUT_DIR / 'stream_network.png'}")

    create_hand_map_plot(
        hand, bounds_dict, OUTPUT_DIR / "hand_map.png", water_mask=water_mask
    )
    print_progress(f"  Saved: {OUTPUT_DIR / 'hand_map.png'}")

    # =========================================================================
    # Step 5: Hillslope Parameter Computation
    # =========================================================================
    print_section(
        f"Step 5: Hillslope Parameters "
        f"({N_ASPECT_BINS * N_LAND_BINS} land elements + {N_LAKE_COLUMNS} lake)"
    )

    t0 = time.time()

    # --- 5a: Filtering (Phase E.5 recipe) ---
    #
    # See docs/lake-column-ctsm-audit.md Section 6.9 for the pixel-category
    # table and filter rationale. Replaces the prior `(hand > 0)` shortcut,
    # which silently misclassified streams routed through filled depressions
    # (raw_hand < 0) into FZ bins.

    # Flatten all fields
    hand_flat = hand.flatten()
    raw_hand_flat = raw_hand.flatten()
    dtnd_flat = dtnd.flatten().copy()  # we mutate via DTND minimum clip
    slope_flat = slope.flatten()
    aspect_flat = aspect.flatten()
    water_flat = water_mask.flatten()
    channel_flat = (np.array(grid.channel_mask) > 0).flatten()
    pixel_area = PIXEL_SIZE * PIXEL_SIZE
    area_flat = np.full(hand_flat.shape, pixel_area)

    n_water_hand0 = int(np.sum((water_flat > 0) & (hand_flat == 0)))
    print_progress(f"    Water pixels with HAND=0: {n_water_hand0:,}")

    # DTND tail-index outlier removal (Swenson convention via expon fit;
    # tail_index() in hillslope_params.py). Exclude water pixels first
    # (HAND=0, DTND=0) so the 10.7M zero-DTND lake pixels don't skew the
    # exponential fit.
    land_finite_for_tail = np.isfinite(hand_flat) & (water_flat == 0)
    tail_ind = tail_index(
        dtnd_flat[land_finite_for_tail], hand_flat[land_finite_for_tail]
    )
    land_indices = np.where(land_finite_for_tail)[0]
    keep_tail = np.zeros(hand_flat.shape, dtype=bool)
    keep_tail[land_indices[tail_ind]] = True
    n_tail_removed = int(land_finite_for_tail.sum() - keep_tail.sum())
    print_progress(f"    DTND tail removal: {n_tail_removed:,} px removed")

    # DTND minimum clip (Swenson rh:699-700)
    dtnd_flat[dtnd_flat < SMALLEST_DTND_M] = SMALLEST_DTND_M

    # Combined cleanup mask (Phase E.5). Drop lakes (water_mask), all stream
    # channel pixels (channel_mask — catches streams in filled depressions
    # which have raw_hand < 0), NaN, and DTND tail outliers.
    valid_pre_trim = (
        (water_flat == 0) & (channel_flat == 0) & np.isfinite(raw_hand_flat) & keep_tail
    )

    n_pre_trim = int(valid_pre_trim.sum())
    print_progress(f"    Valid pre-Q01/Q99 trim: {n_pre_trim:,}")

    # Phase E.5 outlier strategy (locked 2026-05-02): true-discard Q01/Q99
    # trim on raw HAND. Cutoffs computed dynamically from the cleaned land
    # population each run; expected ~-6.34 m / +17.46 m on the production
    # domain. The Q01/Q99 values double as the outermost edges of the 24-bin
    # TAI-focused scheme. See phases/E.5-bin-redesign.md 2026-05-02 log
    # entry for the formal defense.
    q01 = float(np.percentile(raw_hand_flat[valid_pre_trim], 1))
    q99 = float(np.percentile(raw_hand_flat[valid_pre_trim], 99))
    valid = valid_pre_trim & (raw_hand_flat >= q01) & (raw_hand_flat <= q99)
    n_post_trim = int(valid.sum())
    n_trimmed = n_pre_trim - n_post_trim
    print_progress(
        f"    Q01 = {q01:.4f} m   Q99 = {q99:.4f} m   "
        f"trimmed {n_trimmed:,} ({100 * n_trimmed / n_pre_trim:.2f}%)"
    )
    print_progress(
        f"  Valid pixels (post-trim): {n_post_trim:,} "
        f"({100 * n_post_trim / valid.size:.1f}%)"
    )

    # Extract drainage_id before applying valid filter
    drainage_id_flat = np.array(grid.drainage_id).flatten()

    # Apply valid filter to all arrays. raw_hand_flat is the binning input.
    # The conditioned hand_flat is no longer needed downstream (the n_water_hand0
    # diagnostic above runs on the pre-filter array); the conditioned `hand`
    # 2D array is still saved under SAVE_DIAGNOSTICS for ad-hoc post-run
    # debugging.
    raw_hand_flat = raw_hand_flat[valid]
    dtnd_flat = dtnd_flat[valid]
    slope_flat = slope_flat[valid]
    aspect_flat = aspect_flat[valid]
    area_flat = area_flat[valid]
    drainage_id_flat = drainage_id_flat[valid]

    # --- 5b: HAND bins (Phase E.5: 24-bin TAI-focused, raw HAND) ---
    #
    # The locked scheme (compute_hand_bins_tai_focused) returns 25 hand-tuned
    # edges with q01/q99 as the outermost edges. 12 FZ + 12 upland; 0.25 m
    # floor in the TAI core (LIDAR 2-sigma distinguishability rule). The
    # binning input is raw_hand, so FZ bins occupy raw_hand < 0 and upland
    # bins occupy raw_hand > 0. See phases/E.5-bin-redesign.md "Working
    # bin scheme" subsection for the per-bin density table and rationale.
    print_progress("  Computing HAND bins (Phase E.5 TAI-focused)...")
    hand_bounds = compute_hand_bins_tai_focused(q01, q99)
    assert len(hand_bounds) == N_LAND_BINS + 1, (
        f"Expected {N_LAND_BINS + 1} edges, got {len(hand_bounds)}"
    )
    print_progress(f"    Land bins ({len(hand_bounds) - 1}): {hand_bounds}")

    # --- 5c: Per-aspect parameter computation (land columns only) ---
    # Lake column is constructed in section 5d below and prepended to the
    # elements list.
    params = {
        "metadata": {
            "n_aspect_bins": N_ASPECT_BINS,
            "n_land_bins": N_LAND_BINS,
            "n_lake_columns": N_LAKE_COLUMNS,
            "n_total_columns": N_TOTAL_COLUMNS,
            "aspect_bins": ASPECT_BINS,
            "aspect_names": ASPECT_NAMES,
            "hand_bounds": hand_bounds.tolist(),
            "binning_input": "raw_hand = hand - (flooded_orig - pit_filled)",
            "outlier_method": "Q01/Q99 percentile trim, true discard",
            "q01_cutoff_m": q01,
            "q99_cutoff_m": q99,
            "n_pixels_pre_trim": n_pre_trim,
            "n_pixels_post_trim": n_post_trim,
            "accum_threshold": accum_threshold,
            "spatial_scale_m": float(lc_m),
            "min_wavelength_m": float(MIN_WAVELENGTH),
            "pixel_size_m": PIXEL_SIZE,
            "region_shape": list(dem.shape),
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
        "is_lake": False,
    }

    # For 1x24 config: loops once with asp_bin=(0,360), capturing all pixels.
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        print_progress(f"  Processing {asp_name} aspect...")

        asp_mask = get_aspect_mask(aspect_flat, asp_bin)
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            print_progress(f"    No pixels in {asp_name} aspect")
            for h_idx in range(N_LAND_BINS):
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

        # First pass: raw areas per bin (binning on raw_hand under Phase E.5)
        bin_raw_areas = []
        bin_data = []
        for h_idx in range(N_LAND_BINS):
            h_lower = hand_bounds[h_idx]
            h_upper = hand_bounds[h_idx + 1]
            hand_mask = (raw_hand_flat >= h_lower) & (raw_hand_flat < h_upper)
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
            else [1.0 / N_LAND_BINS] * N_LAND_BINS
        )
        fitted_areas = [trap_area * frac for frac in area_fractions]

        # Second pass: compute parameters.
        # Note (Phase E.5): the Swenson `mean(hand) <= 0` skip guard from
        # merit_regression.py:706-717 is intentionally NOT applied here. That
        # guard discards bins whose mean conditioned-HAND is non-positive — a
        # sensible filter when binning on `hand` (where 0 means stream
        # channel) but it would zero out every FZ bin under raw-HAND binning,
        # defeating the design.
        for h_idx in range(N_LAND_BINS):
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
                        "is_lake": False,
                    }
                )
                continue

            mean_raw_hand = float(np.mean(raw_hand_flat[bin_indices]))
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
                    "height": mean_raw_hand,
                    "distance": distance,
                    "area": fitted_area,
                    "slope": mean_slope,
                    "aspect": mean_aspect,
                    "width": width,
                    "is_lake": False,
                }
            )

            print_progress(
                f"    Bin {h_idx + 1}: raw_h={mean_raw_hand:+.2f}m, "
                f"d={distance:.0f}m, w={width:.0f}m, "
                f"area={fitted_area / 1e6:.3f}km^2"
            )

    print_progress(f"  Parameter computation time: {time.time() - t0:.1f} seconds")

    # =========================================================================
    # Step 5d: Lake Column (Phase E.5 / Phase G integration, locked 2026-05-04)
    # =========================================================================
    # Add one column representing the aggregate of NWI-mapped lake area as a
    # submerged "lake" column at chain index 1. Land columns shift up: former
    # col 1 (lowest land bin) becomes col 2, ..., col 24 becomes col 25.
    #
    # Why col 1: preserves the existing 16-column `wetlandisfull` flag
    # behavior in the PI's SourceMod (the `cold == ispval` column is processed
    # FIRST in SurfaceWaterMod's loop, and subsequent columns reset the flag).
    # Putting the lake at col 25 would accidentally activate that dormant
    # flag, changing CTSM behavior compared to the 16-column baseline. See
    # docs/lake-column-ctsm-audit.md Section 7.1.
    #
    # Why hill_elev = -6.0 m: chain-bookkeeping value, not a physical lake
    # bottom. Constraint: lake hill_elev must be more negative than the
    # deepest land bin's mean (-5.13 m for bin 1 of the 24-bin scheme) to
    # satisfy CTSM chain monotonicity. Empirical lake geometry doesn't reach
    # that depth (NWI mean -2.53 m, Lee 2023 spill 2.64 m), so the value is
    # set 0.87 m below the floor with PI's blessing. Tuning deferred to model
    # output. See docs/lake-column-ctsm-audit.md Section 5.2.1.
    print_section("Step 5d: Lake Column (Phase E.5)")

    lake_n_pixels = int(np.sum(water_mask > 0))
    lake_area_m2 = float(lake_n_pixels) * (PIXEL_SIZE**2)
    lake_perimeter_m = compute_lake_perimeter(water_mask, PIXEL_SIZE)
    lake_hill_width_m = lake_perimeter_m / 2.0

    print_progress(f"  NWI water pixels: {lake_n_pixels:,}")
    print_progress(f"  Lake area: {lake_area_m2 / 1e6:.3f} km^2")
    print_progress(
        f"  Lake perimeter (boundary-pixel approx): {lake_perimeter_m:,.0f} m"
    )
    print_progress(f"  Lake hill_width (1/2 perimeter): {lake_hill_width_m:,.0f} m")
    print_progress(f"  Lake hill_elev (locked 2026-05-04): {LAKE_HILL_ELEV_M} m")

    lake_element = {
        "aspect_name": "Lake",
        "aspect_bin": 0,
        "hand_bin": -1,  # sentinel: lake is not a HAND bin
        "is_lake": True,
        "height": LAKE_HILL_ELEV_M,
        "distance": LAKE_HILL_DISTANCE_M,
        "area": lake_area_m2,
        "slope": LAKE_HILL_SLOPE,
        "aspect": LAKE_HILL_ASPECT_DEG,
        "width": lake_hill_width_m,
    }

    # Prepend lake to elements list — lake gets chain index 1, land columns
    # shift to indices 2..25.
    params["elements"] = [lake_element] + params["elements"]
    params["metadata"]["lake_pixels"] = lake_n_pixels
    params["metadata"]["lake_area_m2"] = lake_area_m2
    params["metadata"]["lake_perimeter_m"] = lake_perimeter_m
    params["metadata"]["lake_hill_elev_m"] = LAKE_HILL_ELEV_M
    params["metadata"]["lake_hill_distance_m"] = LAKE_HILL_DISTANCE_M
    params["metadata"]["lake_hill_width_m"] = lake_hill_width_m

    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    print_section("Step 6: Saving Results")

    # Stream parameters
    total_area_m2 = sum(elem["area"] for elem in params["elements"])

    # Stream slope from network (validated approach)
    stream_slope_val = net_stats["slope"]

    # Stream depth/width: Swenson power laws (rh:1104-1114). Kept as-is per
    # PI decision (#4, 2026-03-30): osbs2 runs with use_hillslope_routing=
    # .false., so these values are never read by CTSM under the current
    # configuration. Refining them is deferred until routing is ever
    # enabled.
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
        q01=q01,
        q99=q99,
        n_pre_trim=n_pre_trim,
        n_post_trim=n_post_trim,
    )

    # Summary text
    summary_path = OUTPUT_DIR / f"{OUTPUT_DESCRIPTOR}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"OSBS Hillslope Pipeline Summary ({OUTPUT_DESCRIPTOR})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {MOSAIC_PATH}\n")
        f.write(f"Region: {dem.shape} pixels\n")
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
        f.write(f"  Network length: {net_stats['length']:.0f} m\n")
        f.write(f"  Network slope: {net_stats['slope']:.6f} m/m\n")
        f.write(
            f"  Stream depth: {stream_depth:.3f} m (Swenson power law; routing off)\n"
        )
        f.write(
            f"  Stream width: {stream_width:.1f} m (Swenson power law; routing off)\n"
        )
        f.write(f"  Stream slope: {stream_slope_val:.6f} m/m\n\n")

        f.write("HAND (Phase E.5: bin on raw_hand = hand - dep_fill):\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Conditioned-HAND positive range: 0 - {np.max(hand_positive):.1f} m\n"
        )
        f.write(f"  Conditioned-HAND median: {np.median(hand_positive):.1f} m\n")
        f.write(f"  Q01 cutoff (raw HAND): {q01:.4f} m\n")
        f.write(f"  Q99 cutoff (raw HAND): {q99:.4f} m\n")
        f.write(f"  Pixels pre-trim:  {n_pre_trim:,}\n")
        f.write(f"  Pixels post-trim: {n_post_trim:,}\n")
        f.write(f"  Land bin edges (24 bins): {hand_bounds.tolist()}\n\n")

        f.write("Lake Column (Phase E.5, chain index 1; locked 2026-05-04):\n")
        f.write("-" * 60 + "\n")
        f.write(f"  hill_elev:     {LAKE_HILL_ELEV_M} m  (chain-bookkeeping value)\n")
        f.write(
            f"  hill_distance: {LAKE_HILL_DISTANCE_M} m  (PI direction; ~stream width)\n"
        )
        f.write(
            f"  hill_area:     {lake_area_m2 / 1e6:.3f} km^2  ({lake_n_pixels:,} NWI water px)\n"
        )
        f.write(f"  hill_width:    {lake_hill_width_m:,.0f} m  (1/2 NWI perimeter)\n")
        f.write(f"  hill_slope:    {LAKE_HILL_SLOPE} (lake-bottom framing)\n\n")

        f.write(
            f"Hillslope Elements ({len(params['elements'])} total: "
            f"{N_LAKE_COLUMNS} lake + {N_ASPECT_BINS * N_LAND_BINS} land):\n"
        )
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Col':<5} {'Type':<8} {'Bin':<5} {'Height':<10} "
            f"{'Distance':<10} {'Area_km2':<10} {'Width':<8}\n"
        )
        f.write("-" * 70 + "\n")

        for col_i, elem in enumerate(params["elements"], start=1):
            label = "Lake" if elem.get("is_lake") else elem["aspect_name"]
            bin_str = "-" if elem.get("is_lake") else str(elem["hand_bin"] + 1)
            f.write(
                f"{col_i:<5} {label:<8} {bin_str:<5} "
                f"{elem['height']:<10.3f} {elem['distance']:<10.0f} "
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
