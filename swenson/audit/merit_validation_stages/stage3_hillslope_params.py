#!/usr/bin/env python
"""
Stage 3: Hillslope Parameter Computation

Compute the 6 geomorphic parameters per hillslope element following
Swenson & Lawrence (2025).

This script:
1. Loads Stage 2 results (spatial scale → accumulation threshold)
2. Processes DEM to compute HAND/DTND with the data-driven threshold
3. Computes slope and aspect from the DEM
4. Bins pixels by aspect (4 bins: N, E, S, W) and elevation (4 bins)
5. Calculates mean parameters per bin
6. Fits trapezoidal width model

The 6 parameters per hillslope element:
- Area (A): Horizontally projected surface area
- Height (h): Mean height above stream channel (HAND)
- Distance (d): Mean distance from channel (DTND)
- Width (w): Width at downslope interface
- Slope (α): Mean topographic slope
- Aspect (β): Azimuthal orientation from North

Structure: 4 aspects × 4 elevation bins = 16 hillslope elements per gridcell

Data paths:
- Stage 2 output: swenson/output/merit_validation/stage2/stage2_results.json
- input: /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif
- Output: swenson/output/merit_validation/stage1/

Expected runtime: ~30-60 minutes on 4 cores with 32GB RAM
"""

import os
import sys
import time
import json
import numpy as np

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

# Add parent directory for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysheds.pgrid import Grid

# Local modules - only need constants, not calc_gradient (use pgrid instead)
from spatial_scale import DTR, RE

try:
    import rasterio
    from rasterio.windows import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Configuration
MERIT_DEM_PATH = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif"
STAGE2_RESULTS = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage2/stage2_results.json"
)
OUTPUT_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage3"

# Processing parameters
N_ASPECT_BINS = 4  # N, E, S, W
N_HAND_BINS = 4  # Elevation bins
LOWEST_BIN_MAX = 2.0  # Maximum HAND for lowest bin (meters)

# Target gridcell boundaries (from published 0.9x1.25 grid)
# Extracted from hillslopes_0.9x1.25_c240416.nc at lon_idx=214, lat_idx=130
# Published uses 0-360°E; MERIT DEM uses -180 to 180° (Western Hemisphere negative)
# 267.5°E = 267.5 - 360 = -92.5°W
TARGET_GRIDCELL = {
    "lon_min": -93.1250,  # 266.875°E converted to Western Hemisphere
    "lon_max": -91.8750,  # 268.125°E converted to Western Hemisphere
    "lat_min": 32.0419,
    "lat_max": 32.9843,
    "center_lon": -92.5000,  # 267.5°E converted
    "center_lat": 32.5131,
}

# Expansion factor for flow routing (process larger, extract to gridcell)
# Flow routing needs buffer to avoid edge effects
EXPANSION_FACTOR = 1.5  # Process 1.5x gridcell size

# Aspect bin definitions (degrees from North, clockwise)
# North: ≥315° or <45°, East: ≥45° and <135°, etc.
ASPECT_BINS = [
    (315, 45),  # North
    (45, 135),  # East
    (135, 225),  # South
    (225, 315),  # West
]
ASPECT_NAMES = ["North", "East", "South", "West"]


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# Note: compute_slope_aspect() removed in Stage 8 fix.
# Now using grid.slope_aspect() which uses Horn 1981 method with correct
# coordinate conventions. Our custom implementation had a Y-axis sign
# inversion that caused North/South aspect swapping.


def compute_gridcell_indices(
    expanded_transform: rasterio.Affine,
    expanded_shape: tuple,
    gridcell: dict,
) -> tuple[slice, slice]:
    """
    Compute row/col slices for extracting gridcell from expanded region.

    Parameters
    ----------
    expanded_transform : Affine
        Affine transform for the expanded region
    expanded_shape : tuple
        Shape (rows, cols) of expanded region
    gridcell : dict
        Gridcell bounds with lon_min, lon_max, lat_min, lat_max

    Returns
    -------
    row_slice, col_slice : tuple of slices
    """
    # Get pixel coordinates for gridcell boundaries
    # For rasterio, pixel (0,0) is at transform.c, transform.f (top-left)
    # col = (lon - transform.c) / transform.a
    # row = (lat - transform.f) / transform.e

    col_min = int((gridcell["lon_min"] - expanded_transform.c) / expanded_transform.a)
    col_max = int((gridcell["lon_max"] - expanded_transform.c) / expanded_transform.a)
    row_min = int(
        (gridcell["lat_max"] - expanded_transform.f) / expanded_transform.e
    )  # Note: lat_max gives row_min
    row_max = int((gridcell["lat_min"] - expanded_transform.f) / expanded_transform.e)

    # Clamp to valid range
    col_min = max(0, col_min)
    col_max = min(expanded_shape[1], col_max)
    row_min = max(0, row_min)
    row_max = min(expanded_shape[0], row_max)

    return slice(row_min, row_max), slice(col_min, col_max)


def get_aspect_mask(aspect: np.ndarray, aspect_bin: tuple) -> np.ndarray:
    """
    Create mask for pixels within an aspect bin.

    Parameters
    ----------
    aspect : array
        Aspect values in degrees (0-360)
    aspect_bin : tuple
        (lower, upper) bounds in degrees

    Returns
    -------
    mask : boolean array
    """
    lower, upper = aspect_bin
    if lower > upper:  # Crosses 0° (e.g., North: 315-45)
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
    """
    Compute HAND bin boundaries following Swenson's SpecifyHandBounds().

    Algorithm:
    1. Start with quartiles (25%, 50%, 75%, 100%)
    2. If Q25 > bin1_max:
       a. Validate per-aspect: ensure each aspect has ≥1% below bin1_max
       b. Adjust bin1_max upward only if necessary
       c. Compute bins 2-4 from points above bin1_max (33%, 66% quantiles)

    Parameters
    ----------
    hand : array
        HAND values (flattened)
    aspect : array
        Aspect values (flattened)
    aspect_bins : list
        List of (lower, upper) tuples for aspect bins
    bin1_max : float
        Maximum HAND for lowest bin (meters) - MANDATORY constraint
    min_aspect_fraction : float
        Minimum fraction of points that must be below bin1_max per aspect

    Returns
    -------
    bounds : array
        N_HAND_BINS + 1 boundary values
    """
    # Filter valid HAND values
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]
    aspect_valid = aspect[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    # Sort HAND values
    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size

    # Initial quartile-based bounds
    quartiles = [0.25, 0.5, 0.75, 1.0]
    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    print(f"    Initial Q25: {initial_q25:.2f} m, bin1_max target: {bin1_max:.2f} m")

    # Check if bin1_max constraint needs to be applied
    if initial_q25 > bin1_max:
        print(
            f"    Q25 ({initial_q25:.2f}m) > bin1_max ({bin1_max}m), applying mandatory 2m constraint"
        )

        # Per-aspect validation
        adjusted_bin1_max = bin1_max

        for asp_idx, (asp_low, asp_high) in enumerate(aspect_bins):
            # Select points for this aspect
            if asp_low > asp_high:  # North wraps around (315-45)
                asp_mask = (aspect_valid >= asp_low) | (aspect_valid < asp_high)
            else:
                asp_mask = (aspect_valid >= asp_low) & (aspect_valid < asp_high)

            hand_asp = hand_valid[asp_mask]
            if hand_asp.size > 0:
                hand_asp_sorted = np.sort(hand_asp)
                # Check fraction below bin1_max
                below_threshold = (
                    np.sum(hand_asp_sorted <= bin1_max) / hand_asp_sorted.size
                )

                if below_threshold < min_aspect_fraction:
                    # Find the value at min_aspect_fraction quantile
                    idx_1pct = max(
                        0, int(min_aspect_fraction * hand_asp_sorted.size) - 1
                    )
                    bmin = hand_asp_sorted[idx_1pct]
                    print(
                        f"    Warning: Aspect {ASPECT_NAMES[asp_idx]} has only "
                        f"{below_threshold:.1%} below {bin1_max}m (need {min_aspect_fraction:.0%})"
                    )
                    adjusted_bin1_max = max(adjusted_bin1_max, bmin)

        if adjusted_bin1_max != bin1_max:
            print(f"    Adjusted bin1_max from {bin1_max}m to {adjusted_bin1_max:.2f}m")

        # Compute remaining bins from points ABOVE adjusted_bin1_max
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

        print(f"    HAND bins (with 2m constraint): {bounds[:4]} + [max]")
    else:
        # Q25 already <= bin1_max, use quartiles directly
        bounds = [0]
        for q in quartiles:
            idx = max(0, int(q * n) - 1)
            bounds.append(hand_sorted[idx])
        bounds[-1] = 1e6  # Upper bound set to large value
        bounds = np.array(bounds)
        print(f"    HAND bins (quartile-based): {bounds[:4]} + [max]")

    return bounds


def fit_trapezoidal_width(
    dtnd: np.ndarray,
    area: np.ndarray,
    n_hillslopes: int,
    min_dtnd: float = 90,
    n_bins: int = 10,
) -> dict:
    """
    Fit trapezoidal plan form to hillslope using area vs distance relationship.

    Following Swenson & Lawrence (2025) Equation (4):
    A_sum(d) = w_base * d + α * d²

    Parameters
    ----------
    dtnd : array
        Distance to nearest drainage (flattened)
    area : array
        Pixel areas (flattened)
    n_hillslopes : int
        Number of individual hillslopes
    min_dtnd : float
        Minimum DTND value to use
    n_bins : int
        Number of distance bins for fitting

    Returns
    -------
    dict with keys: slope (plan form divergence), width (base width), area
    """
    if np.max(dtnd) < min_dtnd:
        # Not enough data for fitting
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }

    # Create distance bins
    dtnd_bins = np.linspace(min_dtnd, np.max(dtnd) + 1, n_bins + 1)

    d = np.zeros(n_bins)
    A_cumsum = np.zeros(n_bins)

    for k in range(n_bins):
        mask = dtnd >= dtnd_bins[k]
        d[k] = dtnd_bins[k]
        A_cumsum[k] = np.sum(area[mask])

    # Normalize by number of hillslopes
    A_cumsum /= n_hillslopes

    # Add d=0, total area point
    if min_dtnd > 0:
        d = np.concatenate([[0], d])
        A_cumsum = np.concatenate([[np.sum(area) / n_hillslopes], A_cumsum])

    # Fit quadratic: A(d) = c0 + c1*d + c2*d²
    # where width = -c1, slope = -c2
    try:
        # Weighted least squares
        weights = A_cumsum
        G = np.column_stack([np.ones_like(d), d, d**2])
        Gw = G * weights[:, np.newaxis]
        coeffs = np.linalg.lstsq(Gw, A_cumsum * weights, rcond=None)[0]

        trap_slope = -coeffs[2]
        trap_width = -coeffs[1]
        trap_area = coeffs[0]

        # Ensure positive width
        if trap_slope < 0 and trap_width > 0:
            # Check if width becomes negative at some distance
            Atri = -(trap_width**2) / (4 * trap_slope)
            if Atri < trap_area:
                trap_width = np.sqrt(-4 * trap_slope * trap_area)

        return {"slope": trap_slope, "width": max(trap_width, 1), "area": trap_area}

    except Exception as e:
        print(f"  Warning: Trapezoidal fit failed: {e}")
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }


def quadratic(coefs, root=0, eps=1e-6):
    """
    Solve quadratic equation ax² + bx + c = 0.

    Port of Swenson's geospatial_utils.quadratic().

    Parameters
    ----------
    coefs : tuple
        (a, b, c) coefficients
    root : int
        Which root to return (0 or 1)
    eps : float
        Tolerance for near-zero discriminant

    Returns
    -------
    float : The selected root
    """
    ak, bk, ck = coefs

    discriminant = bk**2 - 4 * ak * ck

    if discriminant < 0:
        # If negative due to roundoff, adjust c to get zero discriminant
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
    """
    Compute circular mean of aspect values.
    Handles wraparound at 0°/360°.
    """
    sin_sum = np.mean(np.sin(DTR * aspects))
    cos_sum = np.mean(np.cos(DTR * aspects))
    mean_aspect = np.arctan2(sin_sum, cos_sum) / DTR
    if mean_aspect < 0:
        mean_aspect += 360
    return mean_aspect


def compute_pixel_areas(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Compute pixel areas using spherical coordinates (Swenson method).

    Formula: A = R² × dθ × dφ × sin(θ)
    where θ = colatitude (90° - lat), dθ/dφ = grid spacing in radians

    This matches Swenson's representative_hillslope.py lines 1708-1715:
        phi = dtr * lon
        th = dtr * (90.0 - lat)
        dphi = np.abs(phi[1] - phi[0])
        dth = np.abs(th[0] - th[1])
        farea = np.tile(np.sin(th), (im, 1)).T
        self.area = farea * dth * dphi * np.power(re, 2)

    Parameters
    ----------
    lon : 1D array
        Longitude values (degrees)
    lat : 1D array
        Latitude values (degrees)

    Returns
    -------
    area : 2D array (nrows, ncols)
        Pixel areas in m²
    """
    # Convert to radians
    phi = DTR * lon  # longitude in radians
    theta = DTR * (90.0 - lat)  # colatitude in radians

    # Grid spacing in radians
    dphi = np.abs(phi[1] - phi[0])
    dtheta = np.abs(theta[0] - theta[1])

    # sin(colatitude) varies with latitude
    sin_theta = np.sin(theta)

    # Create 2D area grid: each row has same sin(theta), broadcast across columns
    # Shape: (len(lat), len(lon))
    ncols = len(lon)
    area = np.tile(sin_theta.reshape(-1, 1), (1, ncols))
    area = area * dtheta * dphi * RE**2

    return area


def create_diagnostic_plots(
    params: dict,
    dem: np.ndarray,
    hand: np.ndarray,
    dtnd: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    output_dir: str,
) -> None:
    """Generate diagnostic plots for Stage 3."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
        return

    print_section("Generating Diagnostic Plots")

    # Plot 1: Aspect binning
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.imshow(aspect, cmap="hsv", vmin=0, vmax=360)
    ax.set_title("Pixel Aspect (degrees from N)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(slope, cmap="YlOrRd")
    ax.set_title("Pixel Slope (m/m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    hand_plot = np.where(hand < 0, np.nan, hand)
    im = ax.imshow(
        hand_plot, cmap="viridis", vmin=0, vmax=np.nanpercentile(hand_plot, 95)
    )
    ax.set_title("HAND (m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    dtnd_km = np.where(dtnd < 0, np.nan, dtnd / 1000)
    im = ax.imshow(dtnd_km, cmap="magma", vmin=0, vmax=np.nanpercentile(dtnd_km, 95))
    ax.set_title("DTND (km)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage3_terrain_analysis.png"), dpi=150)
    plt.close()

    # Plot 2: Hillslope parameters by aspect
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    aspects = ASPECT_NAMES
    colors = plt.cm.Set2(np.linspace(0, 1, 4))

    # Height by aspect
    ax = axes[0, 0]
    for i, asp in enumerate(aspects):
        heights = [params["elements"][j]["height"] for j in range(i * 4, (i + 1) * 4)]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins], heights, width=0.18, label=asp, color=colors[i]
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Mean HAND (m)")
    ax.set_title("Height Above Nearest Drainage by Aspect")
    ax.legend()

    # Distance by aspect
    ax = axes[0, 1]
    for i, asp in enumerate(aspects):
        distances = [
            params["elements"][j]["distance"] / 1000 for j in range(i * 4, (i + 1) * 4)
        ]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins],
            distances,
            width=0.18,
            label=asp,
            color=colors[i],
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Mean DTND (km)")
    ax.set_title("Distance to Nearest Drainage by Aspect")
    ax.legend()

    # Area by aspect
    ax = axes[1, 0]
    for i, asp in enumerate(aspects):
        areas = [params["elements"][j]["area"] / 1e6 for j in range(i * 4, (i + 1) * 4)]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins], areas, width=0.18, label=asp, color=colors[i]
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Area (km²)")
    ax.set_title("Hillslope Element Area by Aspect")
    ax.legend()

    # Width by aspect
    ax = axes[1, 1]
    for i, asp in enumerate(aspects):
        widths = [
            params["elements"][j]["width"] / 1000 for j in range(i * 4, (i + 1) * 4)
        ]
        bins = range(1, 5)
        ax.bar(
            [b + i * 0.2 for b in bins], widths, width=0.18, label=asp, color=colors[i]
        )
    ax.set_xlabel("Elevation Bin")
    ax.set_ylabel("Width (km)")
    ax.set_title("Lower Edge Width by Aspect")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage3_hillslope_params.png"), dpi=150)
    plt.close()

    print(f"  Saved diagnostic plots to {output_dir}")


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("Stage 3: Hillslope Parameter Computation")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Stage 2 results
    # -------------------------------------------------------------------------
    print_section("Step 1: Loading Stage 2 Results")

    if os.path.exists(STAGE2_RESULTS):
        with open(STAGE2_RESULTS) as f:
            stage2 = json.load(f)
        accum_threshold = stage2["best_estimate"]["accum_threshold_cells"]
        spatial_scale_px = stage2["best_estimate"]["spatial_scale_px"]
        spatial_scale_m = stage2["best_estimate"]["spatial_scale_m"]
        print(f"  Spatial scale: {spatial_scale_px:.1f} px ({spatial_scale_m:.0f} m)")
        print(f"  Accumulation threshold: {accum_threshold:.0f} cells")
    else:
        print(f"  WARNING: Stage 2 results not found at {STAGE2_RESULTS}")
        print("  Using default accumulation threshold of 1000 cells")
        accum_threshold = 1000
        spatial_scale_m = 0

    accum_threshold = int(accum_threshold)

    # -------------------------------------------------------------------------
    # Step 2: Load DEM and compute flow routing
    # -------------------------------------------------------------------------
    print_section("Step 2: Loading DEM and Computing Flow Routing")

    t0 = time.time()

    # Calculate expanded bounds for flow routing (need buffer to avoid edge effects)
    gc = TARGET_GRIDCELL
    lon_width = gc["lon_max"] - gc["lon_min"]
    lat_height = gc["lat_max"] - gc["lat_min"]

    expanded_bounds = {
        "lon_min": gc["center_lon"] - EXPANSION_FACTOR * lon_width / 2,
        "lon_max": gc["center_lon"] + EXPANSION_FACTOR * lon_width / 2,
        "lat_min": gc["center_lat"] - EXPANSION_FACTOR * lat_height / 2,
        "lat_max": gc["center_lat"] + EXPANSION_FACTOR * lat_height / 2,
    }

    print("  Target gridcell:")
    print(f"    Lon: [{gc['lon_min']:.4f}, {gc['lon_max']:.4f}] ({lon_width:.4f}°)")
    print(f"    Lat: [{gc['lat_min']:.4f}, {gc['lat_max']:.4f}] ({lat_height:.4f}°)")
    print("  Expanded region (for flow routing):")
    print(
        f"    Lon: [{expanded_bounds['lon_min']:.4f}, {expanded_bounds['lon_max']:.4f}]"
    )
    print(
        f"    Lat: [{expanded_bounds['lat_min']:.4f}, {expanded_bounds['lat_max']:.4f}]"
    )

    # Load DEM using rasterio window for expanded region
    if HAS_RASTERIO:
        with rasterio.open(MERIT_DEM_PATH) as src:
            src_nodata = src.nodata

            # Get window for expanded bounds
            window = from_bounds(
                expanded_bounds["lon_min"],
                expanded_bounds["lat_min"],
                expanded_bounds["lon_max"],
                expanded_bounds["lat_max"],
                src.transform,
            )
            dem_data = src.read(1, window=window)
            expanded_transform = src.window_transform(window)
            print(f"  Loaded DEM window: {dem_data.shape} pixels")

            # Compute gridcell extraction indices
            gc_row_slice, gc_col_slice = compute_gridcell_indices(
                expanded_transform, dem_data.shape, gc
            )
            print(f"  Gridcell extraction: rows={gc_row_slice}, cols={gc_col_slice}")
    else:
        raise RuntimeError("rasterio required for gridcell-based DEM loading")

    # Create pysheds grid from the array with proper affine transform
    # pgrid expects pyproj.Proj object (not CRS)
    from pyproj import Proj as PyprojProj

    grid = Grid()
    grid.add_gridded_data(
        dem_data,
        data_name="dem",
        affine=expanded_transform,
        crs=PyprojProj("EPSG:4326"),
        nodata=src_nodata if src_nodata is not None else -9999,
    )
    dem = grid.dem

    print(f"  Expanded DEM shape: {dem.shape}")

    # D8 direction map (pysheds convention)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Condition DEM
    print("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")
    grid.resolve_flats("flooded", out_name="inflated")

    # Flow direction
    print("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=dirmap, routing="d8")

    # Flow accumulation
    print("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=dirmap, routing="d8")
    acc = grid.acc

    print(f"  Flow routing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 3: Create stream network and compute HAND/DTND
    # -------------------------------------------------------------------------
    print_section("Step 3: Creating Stream Network and Computing HAND/DTND")

    t0 = time.time()

    # Stream mask using accumulation threshold
    acc_mask = acc > accum_threshold

    # Use pgrid's create_channel_mask method
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=dirmap, routing="d8")

    stream_mask = grid.channel_mask
    channel_id = grid.channel_id

    stream_cells = np.sum(stream_mask > 0)
    num_channels = int(np.nanmax(channel_id)) if np.any(~np.isnan(channel_id)) else 0
    print(f"  Accumulation threshold: {accum_threshold} cells")
    print(
        f"  Stream cells: {stream_cells} ({100 * stream_cells / stream_mask.size:.2f}%)"
    )
    print(f"  Number of stream segments: {num_channels}")

    # Compute HAND and DTND using pgrid method
    print("  Computing HAND and DTND...")
    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=dirmap,
        routing="d8",
    )
    hand = grid.hand
    dtnd = grid.dtnd

    print(f"  HAND/DTND computation time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 4: Compute slope and aspect using pgrid's Horn 1981 method
    # -------------------------------------------------------------------------
    print_section("Step 4: Computing Slope and Aspect")

    t0 = time.time()

    # Get coordinate arrays (still needed for Step 5)
    transform = grid.affine
    nrows, ncols = dem.shape
    lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
    lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    # Compute slope and aspect using pgrid's Horn 1981 method
    # This method uses correct coordinate conventions (fixed Stage 8 N/S swap issue)
    grid.slope_aspect("dem")
    slope = np.array(grid.slope)
    aspect = np.array(grid.aspect)

    print("  Using pgrid.slope_aspect() (Horn 1981 method)")
    print(f"  Slope range: [{np.nanmin(slope):.4f}, {np.nanmax(slope):.4f}]")
    print(f"  Aspect range: [{np.nanmin(aspect):.1f}, {np.nanmax(aspect):.1f}] degrees")
    print(f"  Slope/aspect computation time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 5: Extract gridcell region for parameter computation
    # -------------------------------------------------------------------------
    print_section("Step 5: Extracting Target Gridcell")

    # Extract arrays to the exact gridcell boundaries
    dem_center = np.array(dem)[gc_row_slice, gc_col_slice]
    hand_center = np.array(hand)[gc_row_slice, gc_col_slice]
    dtnd_center = np.array(dtnd)[gc_row_slice, gc_col_slice]
    slope_center = slope[gc_row_slice, gc_col_slice]
    aspect_center = aspect[gc_row_slice, gc_col_slice]
    lon_center = lon[gc_col_slice]
    lat_center = lat[gc_row_slice]

    print(f"  Region shape: {dem_center.shape}")
    print(f"  Lon range: [{lon_center[0]:.4f}, {lon_center[-1]:.4f}]")
    print(f"  Lat range: [{lat_center[0]:.4f}, {lat_center[-1]:.4f}]")

    # Compute pixel areas using spherical coordinates (Swenson method)
    # Each pixel has its own area based on sin(colatitude)
    pixel_areas = compute_pixel_areas(lon_center, lat_center)
    res_m = np.abs(lat_center[0] - lat_center[1]) * RE * np.pi / 180
    print(f"  Pixel resolution: ~{res_m:.1f} m")
    print(
        f"  Pixel area range: [{np.min(pixel_areas):.0f}, {np.max(pixel_areas):.0f}] m²"
    )
    print(f"  Pixel area mean: {np.mean(pixel_areas):.0f} m²")

    # -------------------------------------------------------------------------
    # Step 6: Compute HAND bins
    # -------------------------------------------------------------------------
    print_section("Step 6: Computing HAND Bins")

    # Flatten arrays
    hand_flat = hand_center.flatten()
    dtnd_flat = dtnd_center.flatten()
    slope_flat = slope_center.flatten()
    aspect_flat = aspect_center.flatten()
    area_flat = pixel_areas.flatten()

    # Filter valid data
    valid = (hand_flat >= 0) & np.isfinite(hand_flat)
    print(f"  Valid pixels: {np.sum(valid)} ({100 * np.sum(valid) / valid.size:.1f}%)")

    # Compute HAND bin boundaries (following Swenson's mandatory 2m constraint)
    print(f"  Computing HAND bins with mandatory {LOWEST_BIN_MAX}m constraint...")
    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )
    print(f"  Final HAND bin boundaries: {hand_bounds}")

    # -------------------------------------------------------------------------
    # Step 7: Compute hillslope parameters by aspect and elevation bin
    # -------------------------------------------------------------------------
    print_section("Step 7: Computing Hillslope Parameters")

    t0 = time.time()

    # Initialize output structure
    params = {
        "metadata": {
            "n_aspect_bins": N_ASPECT_BINS,
            "n_hand_bins": N_HAND_BINS,
            "aspect_bins": ASPECT_BINS,
            "aspect_names": ASPECT_NAMES,
            "hand_bounds": hand_bounds.tolist(),
            "accum_threshold": accum_threshold,
            "spatial_scale_m": spatial_scale_m,
            "region_shape": list(dem_center.shape),
            "lon_range": [float(lon_center[0]), float(lon_center[-1])],
            "lat_range": [float(lat_center[0]), float(lat_center[-1])],
            "target_gridcell": TARGET_GRIDCELL,
            "expansion_factor": EXPANSION_FACTOR,
        },
        "elements": [],
    }

    # Process each aspect
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        print(f"\n  Processing {asp_name} aspect ({asp_bin[0]}°-{asp_bin[1]}°)...")

        # Get pixels in this aspect bin
        asp_mask = get_aspect_mask(aspect_flat, asp_bin) & valid
        asp_indices = np.where(asp_mask)[0]

        if len(asp_indices) == 0:
            print(f"    No pixels in {asp_name} aspect")
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

        n_hillslopes = len(
            np.unique(
                grid.drainage_id.flatten()[asp_indices]
                if hasattr(grid, "drainage_id")
                else [1]
            )
        )
        n_hillslopes = max(n_hillslopes, 1)

        # Hillslope fraction
        hillslope_frac = np.sum(area_flat[asp_indices]) / np.sum(area_flat[valid])
        print(f"    Pixels: {len(asp_indices)}, Fraction: {hillslope_frac:.1%}")

        # Fit trapezoidal width model for this aspect
        trap = fit_trapezoidal_width(
            dtnd_flat[asp_indices], area_flat[asp_indices], n_hillslopes, min_dtnd=res_m
        )
        trap_slope = trap["slope"]
        trap_width = trap["width"]
        trap_area = trap["area"]

        print(
            f"    Trapezoidal fit: slope={trap_slope:.4f}, width={trap_width:.0f} m, area={trap_area:.0f} m²"
        )

        # First pass: collect raw pixel areas for each bin to compute fractions
        bin_raw_areas = []
        bin_data = []

        for h_idx in range(N_HAND_BINS):
            h_lower = hand_bounds[h_idx]
            h_upper = hand_bounds[h_idx + 1]

            # Get pixels in this aspect-elevation bin
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
                raw_area = float(np.sum(area_flat[bin_indices]))
                bin_raw_areas.append(raw_area)
                bin_data.append(
                    {
                        "indices": bin_indices,
                        "h_lower": h_lower,
                        "h_upper": h_upper,
                    }
                )

        # Compute area fractions
        total_raw = sum(bin_raw_areas)
        if total_raw > 0:
            area_fractions = [a / total_raw for a in bin_raw_areas]
        else:
            area_fractions = [0.25] * N_HAND_BINS

        # Compute fitted areas using trapezoidal model
        fitted_areas = [trap_area * frac for frac in area_fractions]

        # Second pass: compute parameters using fitted areas for width calculation
        for h_idx in range(N_HAND_BINS):
            data = bin_data[h_idx]
            bin_indices = data["indices"]
            h_lower = data["h_lower"]
            h_upper = data["h_upper"]

            if bin_indices is None:
                params["elements"].append(
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

            # Compute mean parameters for this bin
            mean_hand = float(np.mean(hand_flat[bin_indices]))
            mean_slope = float(np.nanmean(slope_flat[bin_indices]))
            mean_aspect = circular_mean_aspect(aspect_flat[bin_indices])

            # Median distance (more robust than mean)
            dtnd_sorted = np.sort(dtnd_flat[bin_indices])
            median_dtnd = float(dtnd_sorted[len(dtnd_sorted) // 2])

            # Use fitted area for this bin (preserves relative distribution)
            fitted_area = fitted_areas[h_idx]

            # Compute width using Swenson's method with cumulative FITTED areas
            # Cumulative area up to (but not including) this bin
            da = sum(fitted_areas[:h_idx]) if h_idx > 0 else 0

            # Width at lower edge using quadratic solver
            if trap_slope != 0:
                try:
                    # Solve: da = trap_width * le + trap_slope * le²
                    le = quadratic([trap_slope, trap_width, -da])
                    width = trap_width + 2 * trap_slope * le
                except RuntimeError:
                    # Fallback: linear interpolation
                    width = trap_width * (1 - 0.15 * h_idx)
            else:
                width = trap_width

            width = max(float(width), 1)

            params["elements"].append(
                {
                    "aspect_name": asp_name,
                    "aspect_bin": asp_idx,
                    "hand_bin": h_idx,
                    "height": mean_hand,
                    "distance": median_dtnd,
                    "area": fitted_area,  # Use fitted area, not raw pixel area
                    "slope": mean_slope,
                    "aspect": mean_aspect,
                    "width": width,
                }
            )

            print(
                f"    Bin {h_idx + 1} (HAND {h_lower:.1f}-{h_upper:.1f}m): "
                f"h={mean_hand:.1f}m, d={median_dtnd:.0f}m, "
                f"A={fitted_area:.0f}m², w={width:.0f}m"
            )

    print(f"\n  Parameter computation time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 8: Save outputs
    # -------------------------------------------------------------------------
    print_section("Step 8: Saving Outputs")

    # Save JSON results
    json_path = os.path.join(OUTPUT_DIR, "stage3_hillslope_params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save text summary
    summary_path = os.path.join(OUTPUT_DIR, "stage3_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 3: Hillslope Parameter Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Target Gridcell (0.9x1.25 grid):\n")
        f.write(f"  Center: ({gc['center_lon']:.4f}°E, {gc['center_lat']:.4f}°N)\n")
        f.write(f"  Lon bounds: [{gc['lon_min']:.4f}, {gc['lon_max']:.4f}]\n")
        f.write(f"  Lat bounds: [{gc['lat_min']:.4f}, {gc['lat_max']:.4f}]\n\n")
        f.write(
            f"Extracted region: {dem_center.shape[0]}x{dem_center.shape[1]} pixels\n"
        )
        f.write(f"Actual Lon: [{lon_center[0]:.4f}, {lon_center[-1]:.4f}]\n")
        f.write(f"Actual Lat: [{lat_center[0]:.4f}, {lat_center[-1]:.4f}]\n")
        f.write(f"Accumulation threshold: {accum_threshold} cells\n")
        f.write(f"HAND bin boundaries: {hand_bounds}\n\n")

        f.write("Hillslope Elements (16 total):\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Aspect':<10} {'Bin':<5} {'Height':<10} {'Distance':<12} "
            f"{'Area':<12} {'Slope':<10} {'Width':<10}\n"
        )
        f.write("-" * 60 + "\n")

        for elem in params["elements"]:
            f.write(
                f"{elem['aspect_name']:<10} {elem['hand_bin'] + 1:<5} "
                f"{elem['height']:<10.1f} {elem['distance']:<12.0f} "
                f"{elem['area'] / 1e6:<12.2f} {elem['slope']:<10.4f} "
                f"{elem['width']:<10.0f}\n"
            )

        f.write(f"\nTotal processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # Generate diagnostic plots
    create_diagnostic_plots(
        params,
        dem_center,
        hand_center,
        dtnd_center,
        slope_center,
        aspect_center,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Stage 3 Complete")

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\n16 hillslope elements computed (4 aspects × 4 elevation bins)")


if __name__ == "__main__":
    main()
