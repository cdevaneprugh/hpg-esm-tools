#!/usr/bin/env python3
"""
Smoke Test Pipeline for OSBS 1m LIDAR Data

Run the Swenson hillslope methodology on the 4x4km OSBS subset to verify
the pipeline works with 1m resolution data.

This script:
1. Loads the 4x4km UTM subset DEM
2. Runs FFT spatial scale analysis (adapted for UTM/1m)
3. Computes flow direction and accumulation
4. Creates stream network with data-driven threshold
5. Computes HAND and DTND
6. Computes slope and aspect
7. Bins by aspect and elevation (4x4 = 16 elements)
8. Generates diagnostic outputs

Creates:
- output/smoke_test/lc_spectral_analysis.png
- output/smoke_test/stream_network.png
- output/smoke_test/hand_map.png
- output/smoke_test/hillslope_params.json
- output/smoke_test/smoke_test_summary.txt

Expected runtime: ~5-10 minutes on 4 cores with 32GB RAM
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import optimize, signal

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid  # noqa: E402 (must import after sys.path modification)

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / "smoke_test"

SUBSET_PATH = DATA_DIR / "mosaics" / "OSBS_smoke_test_4x4.tif"

# Analysis parameters
N_ASPECT_BINS = 4
N_HAND_BINS = 4
LOWEST_BIN_MAX = 2.0  # meters
MAX_HILLSLOPE_LENGTH = 2000  # meters (smaller for 1m res, wetland terrain)
NLAMBDA = 30

# Aspect bin definitions (degrees from North, clockwise)
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
    print("=" * 60 + "\n")


# =============================================================================
# Spatial Scale Analysis (UTM-adapted)
# =============================================================================


def calc_gradient_utm(z: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient of elevation field in UTM coordinates.

    For UTM projection, dx = dy = pixel size (1m for NEON LIDAR).

    Parameters
    ----------
    z : 2D array
        Elevation values
    dx : float
        Pixel size in meters

    Returns
    -------
    tuple of (dz/dx, dz/dy) in physical units (m/m)
    """
    dzdy, dzdx = np.gradient(z)

    # Horn 1981 averaging
    dzdy2, dzdx2 = np.zeros(dzdy.shape), np.zeros(dzdx.shape)

    # Edges: use 3 points
    eind = np.asarray([0, 0, 1])
    dzdx2[0, :] = np.mean(dzdx[eind, :], axis=0)
    dzdy2[:, 0] = np.mean(dzdy[:, eind], axis=1)

    eind = np.asarray([-2, -1, -1])
    dzdx2[-1, :] = np.mean(dzdx[eind, :], axis=0)
    dzdy2[:, -1] = np.mean(dzdy[:, eind], axis=1)

    # Interior: use 4 points
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

    # j axis (columns)
    tmp = np.zeros((jm, 2 * n))
    for i in range(n):
        w = n - i
        ind = np.arange(-w, (w + 1), 1, dtype=int)
        tmp[:, n + i] = np.sum(fld[:, ind + i], axis=1) / ind.size
        tmp[:, n - (i + 1)] = np.sum(fld[:, ind - (i + 1)], axis=1) / ind.size

    ind = np.arange(-n, n, 1, dtype=int)
    fld[:, ind] = tmp

    # i axis (rows)
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


def gaussian_no_norm(x: np.ndarray, amp: float, cen: float, sigma: float) -> np.ndarray:
    """Unnormalized Gaussian function."""
    return amp * np.exp(-((x - cen) ** 2) / (2 * (sigma**2)))


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
    min_lc_pixels: float = 50,  # Minimum Lc in pixels (50m at 1m res)
    detrend: bool = True,
    blend_edges_flag: bool = True,
    zero_edges: bool = True,
    edge_blend_window: int = 50,  # 50m at 1m resolution
    edge_zero_margin: int = 50,  # 50m at 1m resolution
    nlambda: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Identify characteristic spatial scale for UTM-projected DEM.

    Adapted from Swenson's method for geographic coordinates.

    Parameters
    ----------
    elev : 2D array
        Elevation values
    pixel_size : float
        Pixel size in meters
    max_hillslope_length : float
        Maximum hillslope length in meters
    detrend : bool
        Whether to remove planar trend
    blend_edges_flag : bool
        Whether to blend edges
    zero_edges : bool
        Whether to zero edges of Laplacian
    edge_blend_window : int
        Window size for edge blending (pixels)
    edge_zero_margin : int
        Margin for zeroing edges (pixels)
    nlambda : int
        Number of wavelength bins
    verbose : bool
        Print diagnostic info

    Returns
    -------
    dict with spatial scale results
    """
    elev = np.copy(elev)
    nrows, ncols = elev.shape

    max_wavelength = 2 * max_hillslope_length / pixel_size

    if verbose:
        print(f"  DEM shape: {elev.shape}")
        print(f"  Resolution: {pixel_size:.1f} m")
        print(f"  Max wavelength: {max_wavelength:.1f} pixels")

    # Mask nodata
    valid_mask = elev > -9000
    land_frac = np.sum(valid_mask) / valid_mask.size

    if verbose:
        print(f"  Valid data fraction: {land_frac:.2%}")

    if land_frac < 0.5:
        print("  WARNING: Less than 50% valid data")

    # Fill nodata with mean
    elev_mean = np.mean(elev[valid_mask])
    elev[~valid_mask] = elev_mean

    # Remove planar trend
    if detrend:
        elev_planar = fit_planar_surface_utm(elev)
        elev = elev - elev_planar
        if verbose:
            print("  Planar surface removed")

    # Blend edges
    if blend_edges_flag:
        elev = blend_edges(elev, n=edge_blend_window)
        if verbose:
            print(f"  Edges blended (window={edge_blend_window} pixels)")

    # Calculate Laplacian
    if verbose:
        print("  Computing Laplacian...")
    grad = calc_gradient_utm(elev, dx=pixel_size)
    x = calc_gradient_utm(grad[0], dx=pixel_size)
    laplac = x[0]
    x = calc_gradient_utm(grad[1], dx=pixel_size)
    laplac += x[1]

    # Zero edges
    if zero_edges:
        n = edge_zero_margin
        laplac[:n, :] = 0
        laplac[:, :n] = 0
        laplac[-n:, :] = 0
        laplac[:, -n:] = 0

    # Compute 2D FFT
    if verbose:
        print("  Computing 2D FFT...")
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
    x = bin_amplitude_spectrum(laplac_amp_fft, wavelength, nlambda=nlambda)
    lambda_1d, laplac_amp_1d = x["lambda"], x["amp"]

    # Locate peak using log-normal fit
    x_ln = fit_peak_lognormal(np.log10(lambda_1d), laplac_amp_1d)

    if x_ln["psharp"] > 1.0:
        model = "lognormal"
        ln_peak = np.exp(x_ln["coefs"][2])
        spatialScale = min(10**ln_peak, max_wavelength)
    else:
        # Fallback: find maximum
        model = "maximum"
        max_idx = np.argmax(laplac_amp_1d)
        spatialScale = lambda_1d[max_idx]

    # Enforce minimum Lc constraint
    # At 1m resolution, very small Lc values are likely noise
    if spatialScale < min_lc_pixels:
        if verbose:
            print(
                f"  Spatial scale {spatialScale:.1f} px below minimum {min_lc_pixels} px, using minimum"
            )
        spatialScale = min_lc_pixels
        model = f"{model}_constrained"

    if verbose:
        print(f"  Model: {model}")
        print(
            f"  Spatial scale: {spatialScale:.1f} pixels ({spatialScale * pixel_size:.0f} m)"
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
# Hillslope Parameter Computation
# =============================================================================


def get_aspect_mask(aspect: np.ndarray, aspect_bin: tuple) -> np.ndarray:
    """Create mask for pixels within an aspect bin."""
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
    Compute HAND bin boundaries following Swenson's methodology.

    The lowest bin must have upper bound <= bin1_max (mandatory per paper).
    """
    # Filter valid HAND values
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]
    aspect_valid = aspect[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size

    # Initial Q25
    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    print(f"    Initial Q25: {initial_q25:.2f} m, bin1_max target: {bin1_max:.2f} m")

    if initial_q25 > bin1_max:
        print(f"    Q25 > bin1_max, applying mandatory {bin1_max}m constraint")

        # Per-aspect validation
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

        # Compute remaining bins from points above bin1_max
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

        print(f"    HAND bins: {bounds[:4]} + [max]")
    else:
        # Use quartiles
        bounds = [0]
        for q in [0.25, 0.5, 0.75, 1.0]:
            idx = max(0, int(q * n) - 1)
            bounds.append(hand_sorted[idx])
        bounds[-1] = 1e6
        bounds = np.array(bounds)
        print(f"    HAND bins (quartile-based): {bounds[:4]} + [max]")

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
        print(f"  Warning: Trapezoidal fit failed: {e}")
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


def compute_dtnd_euclidean(
    hndx: np.ndarray,
    transform: rasterio.Affine,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """
    Compute Distance To Nearest Drainage using Euclidean distance for UTM data.

    pysheds uses haversine (great-circle) distance which assumes geographic
    coordinates. For UTM data, we need Euclidean distance instead.

    Parameters
    ----------
    hndx : 2D array
        Index of nearest drainage cell for each pixel (from pysheds HAND calculation)
        Value of -1 indicates no drainage found
    transform : rasterio.Affine
        Affine transform for the grid
    pixel_size : float
        Pixel size in meters

    Returns
    -------
    dtnd : 2D array
        Distance to nearest drainage in meters
    """
    nrows, ncols = hndx.shape

    # Create coordinate grids
    col_idx, row_idx = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # Flatten the index array
    hndx_flat = hndx.ravel()

    # Get row/col of each cell and its nearest drainage cell
    src_row = row_idx.ravel()
    src_col = col_idx.ravel()

    # Get row/col of nearest drainage (from flattened index)
    drn_row = hndx_flat // ncols
    drn_col = hndx_flat % ncols

    # Calculate Euclidean distance in pixels
    delta_row = src_row - drn_row
    delta_col = src_col - drn_col
    dist_pixels = np.sqrt(delta_row**2 + delta_col**2)

    # Convert to meters
    dtnd = dist_pixels * pixel_size

    # Handle invalid cells (hndx == -1)
    dtnd[hndx_flat == -1] = 0

    return dtnd.reshape(nrows, ncols)


# =============================================================================
# Diagnostic Plots
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
        f"OSBS Smoke Test: Spatial Scale Analysis\n"
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
    fig, ax = plt.subplots(figsize=(10, 10))

    dem_masked = np.ma.masked_less_equal(dem, -9000)

    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]
    im = ax.imshow(dem_masked, cmap="terrain", extent=extent, aspect="equal")

    # Overlay stream network
    stream_display = np.where(stream_mask > 0, 1, np.nan)
    ax.imshow(stream_display, cmap="Blues", alpha=0.7, extent=extent, aspect="equal")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation (m)")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")
    ax.set_title(
        f"OSBS Smoke Test: Stream Network\n"
        f"Stream cells: {np.sum(stream_mask > 0):,} "
        f"({100 * np.sum(stream_mask > 0) / stream_mask.size:.2f}%)"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hand_map_plot(hand: np.ndarray, bounds: dict, output_path: Path) -> None:
    """Create HAND map plot."""
    fig, ax = plt.subplots(figsize=(10, 10))

    hand_plot = np.where(hand < 0, np.nan, hand)
    vmax = np.nanpercentile(hand_plot, 95)

    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]
    im = ax.imshow(
        hand_plot, cmap="viridis", vmin=0, vmax=vmax, extent=extent, aspect="equal"
    )

    plt.colorbar(im, ax=ax, shrink=0.8, label="HAND (m)")
    ax.set_xlabel("Easting (m UTM 17N)")
    ax.set_ylabel("Northing (m UTM 17N)")
    ax.set_title(
        f"OSBS Smoke Test: Height Above Nearest Drainage\n"
        f"Range: 0 - {np.nanmax(hand_plot):.1f} m, "
        f"Median: {np.nanmedian(hand_plot):.1f} m"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_hillslope_params_plot(params: dict, output_path: Path) -> None:
    """Create hillslope parameters summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, 4))

    elements = params["elements"]

    # Height by aspect
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

    # Distance by aspect
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

    # Area by aspect
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

    # Width by aspect
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
# Main Pipeline
# =============================================================================


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("OSBS Smoke Test Pipeline")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUBSET_PATH.exists():
        print(f"ERROR: Subset file not found: {SUBSET_PATH}")
        print("Run extract_smoke_test_subset.py first.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 1: Load DEM
    # -------------------------------------------------------------------------
    print_section("Step 1: Loading DEM")

    with rasterio.open(SUBSET_PATH) as src:
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

    print(f"  Shape: {dem.shape}")
    print(f"  CRS: {crs}")
    print(f"  Pixel size: {pixel_size} m")
    print(f"  Bounds: {bounds_dict}")
    print(f"  Elevation range: {dem_valid.min():.1f} - {dem_valid.max():.1f} m")

    # -------------------------------------------------------------------------
    # Step 2: Spatial Scale Analysis
    # -------------------------------------------------------------------------
    print_section("Step 2: Spatial Scale Analysis (FFT)")

    t0 = time.time()

    # For 1m LIDAR, FFT often picks up high-frequency noise.
    # Set minimum Lc to 100m (100 pixels) to get meaningful hillslope structure.
    MIN_LC_PIXELS = 100  # 100m at 1m resolution

    spatial_result = identify_spatial_scale_utm(
        dem,
        pixel_size=pixel_size,
        max_hillslope_length=MAX_HILLSLOPE_LENGTH,
        min_lc_pixels=MIN_LC_PIXELS,
        detrend=True,
        blend_edges_flag=True,
        zero_edges=True,
        edge_blend_window=50,
        edge_zero_margin=50,
        nlambda=NLAMBDA,
        verbose=True,
    )

    Lc = spatial_result["spatialScale"]
    Lc_m = spatial_result["spatialScale_m"]
    accum_threshold = int(0.5 * Lc**2)

    print("\n  Results:")
    print(f"    Model: {spatial_result['model']}")
    print(f"    Characteristic length (Lc): {Lc:.0f} pixels ({Lc_m:.0f} m)")
    print(f"    Accumulation threshold: {accum_threshold} cells")
    print(f"    Processing time: {time.time() - t0:.1f} seconds")

    # Create spectral plot
    create_spectral_plot(spatial_result, OUTPUT_DIR / "lc_spectral_analysis.png")
    print(f"  Saved: {OUTPUT_DIR / 'lc_spectral_analysis.png'}")

    # -------------------------------------------------------------------------
    # Step 3: Flow Routing with pysheds
    # -------------------------------------------------------------------------
    print_section("Step 3: Flow Routing")

    t0 = time.time()

    from pyproj import Proj as PyprojProj

    grid = Grid()
    grid.add_gridded_data(
        dem,
        data_name="dem",
        affine=transform,
        crs=PyprojProj(crs.to_proj4()),
        nodata=-9999,
    )

    # D8 direction map
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    print("  Conditioning DEM...")
    grid.fill_pits("dem", out_name="pit_filled")
    grid.fill_depressions("pit_filled", out_name="flooded")
    grid.resolve_flats("flooded", out_name="inflated")

    print("  Computing flow direction...")
    grid.flowdir("inflated", out_name="fdir", dirmap=dirmap, routing="d8")

    print("  Computing flow accumulation...")
    grid.accumulation("fdir", out_name="acc", dirmap=dirmap, routing="d8")
    acc = grid.acc

    print(f"  Max accumulation: {np.max(acc):.0f} cells")
    print(f"  Processing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 4: Stream Network and HAND/DTND
    # -------------------------------------------------------------------------
    print_section("Step 4: Stream Network and HAND/DTND")

    t0 = time.time()

    # Create stream mask
    acc_mask = acc > accum_threshold
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=dirmap, routing="d8")

    stream_mask = grid.channel_mask
    channel_id = grid.channel_id

    stream_cells = np.sum(stream_mask > 0)
    stream_frac = stream_cells / stream_mask.size
    num_channels = int(np.nanmax(channel_id)) if np.any(~np.isnan(channel_id)) else 0

    print(f"  Accumulation threshold: {accum_threshold} cells")
    print(f"  Stream cells: {stream_cells:,} ({stream_frac:.2%})")
    print(f"  Number of stream segments: {num_channels}")

    # Compute HAND - first get the index to nearest drainage
    print("  Computing HAND...")
    grid.compute_hand(
        "fdir", "dem", grid.channel_mask, grid.channel_id, dirmap=dirmap, routing="d8"
    )
    hand = np.array(grid.hand)

    # NOTE: pysheds DTND uses haversine formula which requires geographic coordinates.
    # For UTM data, we need to compute DTND using Euclidean distance instead.
    print("  Computing DTND (Euclidean for UTM)...")

    # Get the nearest drainage index by running compute_hand with return_index=True
    # But this returns the hand index directly, so we need to extract it differently
    # The drainage_id attribute should have been set - let's use an alternative approach

    # Alternative: Compute DTND from the drainage_id by finding the nearest stream cell
    # Actually, let's compute it directly from HAND index
    # pysheds stores the index in an internal variable during compute_hand

    # Workaround: Re-run to get index, or compute from stream mask directly
    # For simplicity, compute DTND from stream_mask using scipy distance transform
    from scipy.ndimage import distance_transform_edt

    # Create binary mask (stream = True)
    stream_binary = stream_mask > 0

    # Compute distance transform (distance to nearest True cell in pixels)
    dtnd_pixels = distance_transform_edt(~stream_binary)
    dtnd = dtnd_pixels * pixel_size  # Convert to meters

    hand_valid = hand[(hand > 0) & np.isfinite(hand)]
    dtnd_valid = dtnd[dtnd > 0]

    print(
        f"  HAND range: 0 - {np.max(hand_valid):.1f} m, median: {np.median(hand_valid):.1f} m"
    )
    print(
        f"  DTND range: 0 - {np.max(dtnd_valid):.0f} m, median: {np.median(dtnd_valid):.0f} m"
    )
    print(f"  Processing time: {time.time() - t0:.1f} seconds")

    # Create stream network plot
    create_stream_network_plot(
        dem, stream_mask, bounds_dict, OUTPUT_DIR / "stream_network.png"
    )
    print(f"  Saved: {OUTPUT_DIR / 'stream_network.png'}")

    # Create HAND map
    create_hand_map_plot(hand, bounds_dict, OUTPUT_DIR / "hand_map.png")
    print(f"  Saved: {OUTPUT_DIR / 'hand_map.png'}")

    # -------------------------------------------------------------------------
    # Step 5: Slope and Aspect
    # -------------------------------------------------------------------------
    print_section("Step 5: Slope and Aspect")

    t0 = time.time()

    grid.slope_aspect("dem")
    slope = np.array(grid.slope)
    aspect = np.array(grid.aspect)

    print(f"  Slope range: {np.nanmin(slope):.4f} - {np.nanmax(slope):.4f}")
    print(f"  Aspect range: {np.nanmin(aspect):.1f} - {np.nanmax(aspect):.1f} degrees")
    print(f"  Processing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 6: Hillslope Parameter Computation
    # -------------------------------------------------------------------------
    print_section("Step 6: Hillslope Parameters (16 elements)")

    t0 = time.time()

    # Flatten arrays
    hand_flat = hand.flatten()
    dtnd_flat = dtnd.flatten()
    slope_flat = slope.flatten()
    aspect_flat = aspect.flatten()

    # For UTM, pixel areas are uniform (1 m² at 1m resolution)
    pixel_area = pixel_size * pixel_size
    area_flat = np.full(hand_flat.shape, pixel_area)

    # Valid mask
    valid = (hand_flat >= 0) & np.isfinite(hand_flat)
    print(
        f"  Valid pixels: {np.sum(valid):,} ({100 * np.sum(valid) / valid.size:.1f}%)"
    )

    # Compute HAND bins
    print("  Computing HAND bins...")
    hand_bounds = compute_hand_bins(
        hand_flat, aspect_flat, ASPECT_BINS, bin1_max=LOWEST_BIN_MAX
    )

    # Initialize output
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
            "region_shape": list(dem.shape),
            "bounds": bounds_dict,
            "pixel_size_m": pixel_size,
        },
        "elements": [],
    }

    # Process each aspect
    for asp_idx, (asp_bin, asp_name) in enumerate(zip(ASPECT_BINS, ASPECT_NAMES)):
        print(f"\n  Processing {asp_name} aspect...")

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
        print(f"    Pixels: {len(asp_indices):,}, Fraction: {hillslope_frac:.1%}")

        # Fit trapezoidal width
        trap = fit_trapezoidal_width(
            dtnd_flat[asp_indices], area_flat[asp_indices], n_hillslopes, min_dtnd=1.0
        )

        # First pass: collect raw areas
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

        # Area fractions
        total_raw = sum(bin_raw_areas)
        area_fractions = (
            [a / total_raw for a in bin_raw_areas]
            if total_raw > 0
            else [0.25] * N_HAND_BINS
        )
        fitted_areas = [trap["area"] * frac for frac in area_fractions]

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

            # Width calculation
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

            print(
                f"    Bin {h_idx + 1}: h={mean_hand:.1f}m, d={median_dtnd:.0f}m, "
                f"A={fitted_area / 1e6:.3f}km², w={width:.0f}m"
            )

    print(f"\n  Processing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 7: Save Results
    # -------------------------------------------------------------------------
    print_section("Step 7: Saving Results")

    # Save JSON
    json_path = OUTPUT_DIR / "hillslope_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {json_path}")

    # Create hillslope params plot
    create_hillslope_params_plot(params, OUTPUT_DIR / "hillslope_params.png")
    print(f"  Saved: {OUTPUT_DIR / 'hillslope_params.png'}")

    # Save text summary
    summary_path = OUTPUT_DIR / "smoke_test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("OSBS Smoke Test Pipeline Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {SUBSET_PATH}\n")
        f.write(
            f"Shape: {dem.shape} pixels ({dem.shape[0] * pixel_size / 1000:.1f} x {dem.shape[1] * pixel_size / 1000:.1f} km)\n"
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

        f.write(f"\nTotal processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Smoke Test Complete")

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey results:")
    print(f"  Characteristic length (Lc): {Lc_m:.0f} m")
    print(f"  Accumulation threshold: {accum_threshold} cells")
    print(f"  Stream coverage: {stream_frac:.2%}")
    print(f"  HAND range: 0 - {np.max(hand_valid):.1f} m")
    print("\nNext steps:")
    print("  1. Review diagnostic plots to verify results are reasonable")
    print("  2. If Lc seems too small/large, adjust MAX_HILLSLOPE_LENGTH")
    print("  3. Compare stream network to known drainage patterns")


if __name__ == "__main__":
    main()
