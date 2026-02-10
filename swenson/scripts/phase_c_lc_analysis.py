#!/usr/bin/env python3
"""
Phase C: Characteristic Length Scale (Lc) Analysis

Computes Lc at full 1m resolution on the OSBS interior mosaic, using Swenson's
full model selection logic (Gaussian vs Lognormal vs Linear vs Flat). Then runs
a parameter sensitivity sweep to determine whether Lc is stable or sensitive
to FFT preprocessing parameters.

Output:
  - Baseline Lc with model identification, peak sharpness, and all fit details
  - Sensitivity table: (parameter, value, Lc, model, psharp) for each test
  - Diagnostic plot: amplitude spectrum with fitted peak

Usage:
  python scripts/phase_c_lc_analysis.py
  python scripts/phase_c_lc_analysis.py --plot-dir output/osbs/phase_c
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import optimize, signal


# ============================================================================
# Configuration defaults
# ============================================================================

MOSAIC_PATH = Path(__file__).parent.parent / "data" / "mosaics" / "OSBS_interior.tif"
DEFAULT_PLOT_DIR = Path(__file__).parent.parent / "output" / "osbs" / "phase_c"

# Baseline parameters (matching current pipeline defaults)
DEFAULTS = {
    "blend_edges": 50,
    "zero_edges": 50,
    "nlambda": 30,
    "max_hillslope_length": 2000,  # meters
    "detrend": True,
}

# Sensitivity sweep: one parameter varied at a time
SENSITIVITY_TESTS = {
    "A_blend_edges": {"param": "blend_edges", "values": [4, 25, 50, 100, 200]},
    "B_zero_edges": {"param": "zero_edges", "values": [5, 20, 50, 100]},
    "C_nlambda": {"param": "nlambda", "values": [20, 30, 50, 75]},
    "D_max_hillslope_length": {
        "param": "max_hillslope_length",
        "values": [500, 1000, 2000, 10000],
    },
    "E_detrend": {"param": "detrend", "values": [True, False]},
}


# ============================================================================
# Logging
# ============================================================================


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# FFT preprocessing functions (from run_pipeline.py, UTM-adapted)
# ============================================================================


def calc_gradient_utm(z: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Gradient with Horn 1981 averaging for UTM coordinates."""
    dzdy, dzdx = np.gradient(z)

    dzdy2 = np.zeros(dzdy.shape)
    dzdx2 = np.zeros(dzdx.shape)

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
    """Blend edges of a 2D array to reduce spectral leakage."""
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


def fit_planar_surface(elev: np.ndarray) -> np.ndarray:
    """Fit a planar surface for detrending."""
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
    log_lambda = np.zeros(wavelength.shape)
    log_lambda[wavelength > 0] = np.log10(wavelength[wavelength > 0])

    lambda_bounds = np.linspace(0, np.max(log_lambda), num=nlambda + 1)
    amp_1d = np.zeros(nlambda)
    lambda_1d = np.zeros(nlambda)

    for n in range(nlambda):
        mask = np.logical_and(
            log_lambda > lambda_bounds[n], log_lambda <= lambda_bounds[n + 1]
        )
        if np.any(mask):
            lambda_1d[n] = np.mean(wavelength[mask])
            amp_1d[n] = np.mean(amp_fft[mask])

    ind = np.where(lambda_1d > 0)[0]
    return {"amp": amp_1d[ind], "lambda": lambda_1d[ind]}


# ============================================================================
# Peak fitting functions
# ============================================================================


def log_normal(
    x: np.ndarray, amp: float, sigma: float, mu: float, shift: float = 0
) -> np.ndarray:
    """Log-normal distribution."""
    f = np.zeros(x.size)
    if sigma > 0:
        mask = x > shift
        f[mask] = amp * np.exp(
            -((np.log(x[mask] - shift) - mu) ** 2) / (2 * (sigma**2))
        )
    return f


def gaussian(x: np.ndarray, amp: float, cen: float, sigma: float) -> np.ndarray:
    """Gaussian distribution (unnormalized)."""
    return amp * np.exp(-((x - cen) ** 2) / (2 * (sigma**2)))


def _find_peaks_with_fallback(
    y: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Find peaks with prominence fallback, adding edge peak candidate."""
    meansig = np.mean(y)
    pheight = (meansig, None)
    pwidth = (0, 0.75 * y.size)
    pprom = (0.2 * meansig, None)

    peaks, props = signal.find_peaks(
        y, height=pheight, prominence=pprom, width=pwidth, rel_height=0.5
    )

    if peaks.size == 0:
        pprom = (0.1 * meansig, None)
        peaks, props = signal.find_peaks(
            y, height=pheight, prominence=pprom, width=pwidth, rel_height=0.5
        )

    # find_peaks misses edge peaks — add index 0 as candidate
    if peaks.size > 0:
        peaks = np.append(peaks, 0)
        props["widths"] = np.append(props["widths"], np.max(props["widths"]))

    return peaks, props


def _fit_single_peak(
    x: np.ndarray,
    y: np.ndarray,
    peaks: np.ndarray,
    props: dict,
    fit_func: callable,
    make_p0: callable,
    extract_center: callable,
    compute_sharpness: callable,
) -> dict:
    """Generic peak fitting loop used by both Gaussian and Lognormal fitters."""
    meansig = np.mean(y)
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

        try:
            p0 = make_p0(amp, center, gsigma)
            popt, _ = optimize.curve_fit(
                fit_func, x[i1 : i2 + 1], y[i1 : i2 + 1], p0=p0
            )
            fitted_center = extract_center(popt)
            pdist = np.abs(center - fitted_center)
            # Gaussian: sigma is popt[2] (== popt[-1])
            # Lognormal: sigma is popt[1]
            sigma = popt[-1] if fit_func is gaussian else popt[1]
            if pdist > sigma:
                popt = make_p0(0, 0, 1)  # sentinel
        except Exception:
            popt = make_p0(0, 0, 1)

        peak_coefs.append(list(popt))
        gof = np.mean(np.power(y[i1 : i2 + 1] - fit_func(x[i1 : i2 + 1], *popt), 2))
        peak_gof.append(gof)

        if gof < 1e6 and popt[0] > meansig:
            peak_sharp.append(compute_sharpness(popt, x, y))
        else:
            peak_sharp.append(0)

    if len(peak_sharp) > 0:
        pmax = np.argmax(np.asarray(peak_sharp))
        return {
            "coefs": peak_coefs[pmax],
            "psharp": peak_sharp[pmax],
            "gof": peak_gof[pmax],
        }
    return {"coefs": make_p0(0, 0, 1), "psharp": 0, "gof": 0}


def fit_peak_gaussian(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit Gaussian to locate peak (Swenson _fit_peak_gaussian)."""
    peaks, props = _find_peaks_with_fallback(y)
    if peaks.size == 0:
        return {"coefs": [0, 0, 1], "psharp": 0, "gof": 0}

    def sharpness(popt, x, y):
        rwid = popt[2] / (x[-1] - x[0])
        ramp = popt[0] / np.max(y)
        return ramp / rwid

    return _fit_single_peak(
        x,
        y,
        peaks,
        props,
        fit_func=gaussian,
        make_p0=lambda amp, cen, sig: [amp, cen, sig],
        extract_center=lambda popt: popt[1],
        compute_sharpness=sharpness,
    )


def fit_peak_lognormal(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit log-normal to locate peak (Swenson _fit_peak_lognormal)."""
    peaks, props = _find_peaks_with_fallback(y)
    if peaks.size == 0:
        return {"coefs": [0, 0, 1], "psharp": 0, "gof": 0}

    def sharpness(popt, x, y):
        lnvar = np.sqrt(
            (np.exp(popt[1] ** 2) - 1) * (np.exp(2 * popt[2] + popt[1] ** 2))
        )
        rwid = lnvar / (x[-1] - x[0])
        ramp = popt[0] / np.max(y)
        return ramp / rwid

    def make_p0(amp, cen, sig):
        mu = np.log(cen) if cen > 0 else 0
        return [amp, sig, mu]

    return _fit_single_peak(
        x,
        y,
        peaks,
        props,
        fit_func=log_normal,
        make_p0=make_p0,
        extract_center=lambda popt: np.exp(popt[2]),
        compute_sharpness=sharpness,
    )


# ============================================================================
# Polynomial helpers (from Swenson spatial_scale.py)
# ============================================================================


def fit_polynomial(x: np.ndarray, y: np.ndarray, ncoefs: int) -> np.ndarray:
    """Fit polynomial coefficients via least squares."""
    if x.size < ncoefs:
        raise RuntimeError(f"Not enough data to fit {ncoefs} coefficients")

    g = np.column_stack([np.power(x, n) for n in range(ncoefs)])
    gtd = np.dot(g.T, y)
    gtg = np.dot(g.T, g)
    covm = np.linalg.inv(gtg)
    return np.dot(covm, gtd)


def synth_polynomial(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Reconstruct polynomial from coefficients."""
    return sum(coefs[n] * np.power(x, n) for n in range(len(coefs)))


# ============================================================================
# Full model selection (Swenson _LocatePeak)
# ============================================================================

PSHARP_THRESHOLD = 1.5
TSCORE_THRESHOLD = 2.0


def locate_peak(
    lambda_1d: np.ndarray,
    amp_1d: np.ndarray,
    max_wavelength: float = 1e6,
) -> dict:
    """
    Full Swenson model selection: Gaussian vs Lognormal vs Linear vs Flat.

    Operates on log10(lambda) vs amplitude, matching Swenson's _LocatePeak.

    Returns dict with: model, spatialScale, selection, coefs, psharp_ga,
    psharp_ln, gof_ga, gof_ln, tscore.
    """
    log_lambda = np.log10(lambda_1d)
    min_wavelength = np.min(lambda_1d)

    # Fit peaked models
    x_ga = fit_peak_gaussian(log_lambda, amp_1d)
    x_ln = fit_peak_lognormal(log_lambda, amp_1d)

    psharp_ga = x_ga["psharp"]
    psharp_ln = x_ln["psharp"]
    gof_ga = x_ga["gof"]
    gof_ln = x_ln["gof"]

    # Fit linear model for fallback
    lcoefs = fit_polynomial(log_lambda, amp_1d, 2)

    # T-score for linear fit significance
    n_pts = log_lambda.size
    residuals = amp_1d - synth_polynomial(log_lambda, lcoefs)
    num = (1 / max(1, n_pts - 2)) * np.sum(np.power(residuals, 2))
    den = np.sum(np.power(log_lambda - np.mean(log_lambda), 2))
    se = np.sqrt(num / den) if den > 0 else 1e6
    tscore = np.abs(lcoefs[1]) / se if se > 0 else 0

    # Model selection
    if psharp_ga >= PSHARP_THRESHOLD or psharp_ln >= PSHARP_THRESHOLD:
        if gof_ga < gof_ln:
            model = "gaussian"
            spatial_scale = np.clip(
                10 ** x_ga["coefs"][1], min_wavelength, max_wavelength
            )
            selection = 1
            coefs = x_ga["coefs"]
        else:
            model = "lognormal"
            ln_peak = np.exp(x_ln["coefs"][2])
            spatial_scale = np.clip(10**ln_peak, min_wavelength, max_wavelength)
            selection = 2
            coefs = x_ln["coefs"]
    elif tscore > TSCORE_THRESHOLD:
        model = "linear"
        if lcoefs[1] > 0:
            spatial_scale = max_wavelength
            selection = 3
        else:
            spatial_scale = min_wavelength
            selection = 4
        coefs = list(lcoefs)
    else:
        model = "flat"
        spatial_scale = min_wavelength
        selection = 5
        coefs = [1]

    return {
        "model": model,
        "spatialScale": spatial_scale,
        "selection": selection,
        "coefs": coefs,
        "psharp_ga": psharp_ga,
        "psharp_ln": psharp_ln,
        "gof_ga": gof_ga,
        "gof_ln": gof_ln,
        "tscore": tscore,
    }


# ============================================================================
# Core Lc computation
# ============================================================================


def compute_lc(
    elev: np.ndarray,
    pixel_size: float = 1.0,
    blend_edges_n: int = 50,
    zero_edges_n: int = 50,
    nlambda: int = 30,
    max_hillslope_length: float = 2000,
    detrend: bool = True,
    use_laplacian: bool = True,
) -> dict:
    """
    Compute characteristic length scale from DEM.

    When use_laplacian=True (default), computes the FFT of the Laplacian of
    elevation — Swenson's standard method. The Laplacian amplifies frequency
    by k², which boosts short-wavelength content.

    When use_laplacian=False, computes the FFT of the detrended elevation
    directly. This shows the actual energy distribution across scales without
    k² weighting.

    Returns dict with Lc in pixels and meters, model info, and spectrum data.
    """
    nrows, ncols = elev.shape
    max_wavelength = 2 * max_hillslope_length / pixel_size

    # Mask nodata
    valid_mask = elev > -9000
    land_frac = np.sum(valid_mask) / valid_mask.size
    log(f"  Valid data fraction: {land_frac:.2%}")

    elev_work = elev.copy()
    elev_mean = np.mean(elev_work[valid_mask])
    elev_work[~valid_mask] = elev_mean

    # Detrend
    if detrend:
        log("  Removing planar trend...")
        elev_work = elev_work - fit_planar_surface(elev_work)

    # Blend edges
    if blend_edges_n > 0:
        log(f"  Blending edges (window={blend_edges_n})...")
        elev_work = blend_edges(elev_work, n=blend_edges_n)

    if use_laplacian:
        # Laplacian via gradient
        log("  Computing Laplacian...")
        grad = calc_gradient_utm(elev_work, dx=pixel_size)
        gx = calc_gradient_utm(grad[0], dx=pixel_size)
        gy = calc_gradient_utm(grad[1], dx=pixel_size)
        fft_input = gx[0] + gy[1]
    else:
        log("  Skipping Laplacian (raw elevation spectrum)...")
        fft_input = elev_work

    # Zero edges (suppress boundary artifacts in FFT)
    if zero_edges_n > 0:
        fft_input[:zero_edges_n, :] = 0
        fft_input[:, :zero_edges_n] = 0
        fft_input[-zero_edges_n:, :] = 0
        fft_input[:, -zero_edges_n:] = 0

    # 2D FFT
    log("  Computing 2D FFT...")
    spec_fft = np.fft.rfft2(fft_input, norm="ortho")
    spec_amp = np.abs(spec_fft)

    # Wavelength grid
    rowfreq = np.fft.fftfreq(nrows)
    colfreq = np.fft.rfftfreq(ncols)
    ny, nx = spec_fft.shape
    radialfreq = np.sqrt(
        np.tile(colfreq * colfreq, (ny, 1)) + np.tile(rowfreq * rowfreq, (nx, 1)).T
    )
    wavelength = np.zeros((ny, nx))
    wavelength[radialfreq > 0] = 1 / radialfreq[radialfreq > 0]
    wavelength[0, 0] = 2 * np.max(wavelength)

    # Bin amplitude spectrum
    binned = bin_amplitude_spectrum(spec_amp, wavelength, nlambda=nlambda)
    lambda_1d = binned["lambda"]
    amp_1d = binned["amp"]

    # Full model selection
    log("  Fitting peak models...")
    peak_result = locate_peak(lambda_1d, amp_1d, max_wavelength=max_wavelength)

    lc_pixels = peak_result["spatialScale"]
    lc_meters = lc_pixels * pixel_size

    log(
        f"  Result: model={peak_result['model']}, "
        f"Lc={lc_pixels:.1f} px ({lc_meters:.1f} m), "
        f"selection={peak_result['selection']}"
    )

    return {
        "lc_pixels": lc_pixels,
        "lc_meters": lc_meters,
        "pixel_size": pixel_size,
        "model": peak_result["model"],
        "selection": peak_result["selection"],
        "coefs": peak_result["coefs"],
        "psharp_ga": peak_result["psharp_ga"],
        "psharp_ln": peak_result["psharp_ln"],
        "gof_ga": peak_result["gof_ga"],
        "gof_ln": peak_result["gof_ln"],
        "tscore": peak_result["tscore"],
        "lambda_1d": lambda_1d,
        "amp_1d": amp_1d,
        "use_laplacian": use_laplacian,
        "params": {
            "blend_edges": blend_edges_n,
            "zero_edges": zero_edges_n,
            "nlambda": nlambda,
            "max_hillslope_length": max_hillslope_length,
            "detrend": detrend,
        },
    }


# ============================================================================
# Plotting
# ============================================================================


def plot_spectrum(result: dict, title: str, save_path: Path) -> None:
    """Plot amplitude spectrum with fitted peak."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lambda_1d = result["lambda_1d"]
    amp_1d = result["amp_1d"]
    log_lambda = np.log10(lambda_1d)

    ax.plot(lambda_1d, amp_1d, "k-o", ms=4, label="Binned amplitude")

    # Overlay fitted model
    model = result["model"]
    coefs = result["coefs"]
    x_fine = np.linspace(log_lambda[0], log_lambda[-1], 200)
    lambda_fine = 10**x_fine

    if model == "gaussian":
        y_fit = gaussian(x_fine, *coefs)
        peak_wl = 10 ** coefs[1]
        ax.plot(
            lambda_fine, y_fit, "r-", lw=2, label=f"Gaussian (peak={peak_wl:.1f} px)"
        )
        ax.axvline(peak_wl, color="r", ls="--", alpha=0.5)
    elif model == "lognormal":
        y_fit = log_normal(x_fine, *coefs)
        peak_wl = 10 ** np.exp(coefs[2])
        ax.plot(
            lambda_fine, y_fit, "b-", lw=2, label=f"Lognormal (peak={peak_wl:.1f} px)"
        )
        ax.axvline(peak_wl, color="b", ls="--", alpha=0.5)
    elif model == "linear":
        y_fit = synth_polynomial(x_fine, np.array(coefs))
        ax.plot(lambda_fine, y_fit, "g-", lw=2, label="Linear trend")

    ax.set_xscale("log")
    ax.set_xlabel("Wavelength (pixels)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()

    # Annotation with key metrics
    info_text = (
        f"Lc = {result['lc_pixels']:.1f} px ({result['lc_meters']:.1f} m)\n"
        f"Model: {model} (selection={result['selection']})\n"
        f"psharp: Ga={result['psharp_ga']:.2f}, Ln={result['psharp_ln']:.2f}\n"
        f"GoF: Ga={result['gof_ga']:.2e}, Ln={result['gof_ln']:.2e}\n"
        f"T-score: {result['tscore']:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        fontfamily="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"  Saved plot: {save_path}")


def plot_sensitivity(results: list[dict], save_path: Path) -> None:
    """Plot Lc sensitivity across all parameter tests."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Group results by test
    tests_grouped: dict[str, list[dict]] = {}
    for r in results:
        test_name = r["test_name"]
        if test_name not in tests_grouped:
            tests_grouped[test_name] = []
        tests_grouped[test_name].append(r)

    for idx, (test_name, test_results) in enumerate(sorted(tests_grouped.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]
        values = [r["param_value"] for r in test_results]
        lcs = [r["lc_meters"] for r in test_results]
        models = [r["model"] for r in test_results]

        # Color by model type
        model_colors = {
            "gaussian": "red",
            "lognormal": "blue",
            "linear": "green",
            "flat": "gray",
        }
        colors = [model_colors.get(m, "black") for m in models]

        x_labels = [str(v) for v in values]
        ax.bar(range(len(values)), lcs, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Lc (m)")
        ax.set_title(test_name)

        # Mark default value
        param = test_results[0]["param_name"]
        default_val = DEFAULTS.get(param)
        if default_val is not None and default_val in values:
            default_idx = values.index(default_val)
            ax.get_children()[default_idx].set_edgecolor("orange")
            ax.get_children()[default_idx].set_linewidth(3)

    # Hide unused subplot
    for idx in range(len(tests_grouped), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Gaussian"),
        Patch(facecolor="blue", alpha=0.7, label="Lognormal"),
        Patch(facecolor="green", alpha=0.7, label="Linear"),
        Patch(facecolor="gray", alpha=0.7, label="Flat"),
        Patch(facecolor="white", edgecolor="orange", linewidth=3, label="Default"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.suptitle("Phase C: Lc Sensitivity to FFT Parameters", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"  Saved sensitivity plot: {save_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Phase C: Lc Analysis")
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=MOSAIC_PATH,
        help="Path to interior mosaic GeoTIFF",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory for output plots",
    )
    args = parser.parse_args()

    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load mosaic ----
    log(f"Reading mosaic: {args.mosaic}")
    with rasterio.open(args.mosaic) as src:
        elev = src.read(1).astype(np.float64)
        pixel_size = src.res[0]  # meters (should be 1.0 for 1m LIDAR)
        crs = src.crs
        nodata = src.nodata

    log(f"  Shape: {elev.shape}, pixel_size: {pixel_size}m, CRS: {crs}")
    log(f"  Nodata value: {nodata}")

    if nodata is not None:
        elev[elev == nodata] = -9999

    # ---- Baseline run ----
    log("")
    log("=" * 60)
    log("BASELINE: Full-resolution FFT with default parameters")
    log("=" * 60)

    baseline = compute_lc(
        elev,
        pixel_size=pixel_size,
        blend_edges_n=DEFAULTS["blend_edges"],
        zero_edges_n=DEFAULTS["zero_edges"],
        nlambda=DEFAULTS["nlambda"],
        max_hillslope_length=DEFAULTS["max_hillslope_length"],
        detrend=DEFAULTS["detrend"],
    )

    # Print baseline summary
    print("\n" + "=" * 60)
    print("BASELINE RESULT")
    print("=" * 60)
    print(
        f"  Lc:          {baseline['lc_pixels']:.1f} px ({baseline['lc_meters']:.1f} m)"
    )
    print(f"  Model:       {baseline['model']} (selection={baseline['selection']})")
    print(f"  Threshold:   {0.5 * baseline['lc_meters'] ** 2:.0f} m^2")
    thresh_cells = 0.5 * baseline["lc_pixels"] ** 2
    print(f"               {thresh_cells:.0f} cells (at {pixel_size}m)")
    print(f"  psharp(Ga):  {baseline['psharp_ga']:.3f}")
    print(f"  psharp(Ln):  {baseline['psharp_ln']:.3f}")
    print(f"  GoF(Ga):     {baseline['gof_ga']:.4e}")
    print(f"  GoF(Ln):     {baseline['gof_ln']:.4e}")
    print(f"  T-score:     {baseline['tscore']:.3f}")
    print(f"  Coefs:       {baseline['coefs']}")

    # Save baseline plot
    plot_spectrum(
        baseline,
        "Phase C Baseline: Full-Resolution Lc Analysis (OSBS Interior)",
        plot_dir / "baseline_spectrum.png",
    )

    # ---- Sensitivity sweep ----
    log("")
    log("=" * 60)
    log("SENSITIVITY SWEEP")
    log("=" * 60)

    all_results = []

    for test_name, test_config in sorted(SENSITIVITY_TESTS.items()):
        param_name = test_config["param"]
        values = test_config["values"]

        log(f"\nTest {test_name}: varying {param_name}")

        for val in values:
            # Start from defaults, override one parameter
            params = dict(DEFAULTS)
            params[param_name] = val

            log(f"  {param_name}={val}...")
            result = compute_lc(
                elev,
                pixel_size=pixel_size,
                blend_edges_n=params["blend_edges"],
                zero_edges_n=params["zero_edges"],
                nlambda=params["nlambda"],
                max_hillslope_length=params["max_hillslope_length"],
                detrend=params["detrend"],
            )
            result["test_name"] = test_name
            result["param_name"] = param_name
            result["param_value"] = val
            all_results.append(result)

    # Print sensitivity table
    print("\n" + "=" * 80)
    print("SENSITIVITY RESULTS")
    print("=" * 80)
    print(
        f"{'Test':<25} {'Param':<22} {'Value':<8} "
        f"{'Lc(px)':<8} {'Lc(m)':<8} {'Model':<10} {'psharp_ga':<10} {'psharp_ln':<10}"
    )
    print("-" * 101)

    for r in all_results:
        is_default = r["param_value"] == DEFAULTS.get(r["param_name"])
        marker = " *" if is_default else ""
        print(
            f"{r['test_name']:<25} {r['param_name']:<22} {str(r['param_value']):<8} "
            f"{r['lc_pixels']:<8.1f} {r['lc_meters']:<8.1f} {r['model']:<10} "
            f"{r['psharp_ga']:<10.3f} {r['psharp_ln']:<10.3f}{marker}"
        )

    print("\n* = default parameter value")

    # Lc range summary
    lc_values = [r["lc_meters"] for r in all_results]
    print(f"\nLc range across all tests: {min(lc_values):.1f} - {max(lc_values):.1f} m")
    print(f"Baseline Lc: {baseline['lc_meters']:.1f} m")

    # Save sensitivity plot
    plot_sensitivity(all_results, plot_dir / "sensitivity_sweep.png")

    log("\nPhase C analysis complete.")


if __name__ == "__main__":
    main()
