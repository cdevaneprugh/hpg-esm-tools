"""
Spatial Scale Analysis Module

Adapted from Swenson & Lawrence (2025) Representative Hillslopes codebase.
Identifies the characteristic spatial scale at which a DEM exhibits
the largest divergence/convergence of topographic gradient.

This is a self-contained module that includes only the functions needed
for spatial scale analysis, avoiding external dependencies on Swenson's
full codebase.

Key function: identify_spatial_scale_laplacian_dem()
"""

import numpy as np
from scipy import optimize, signal

# Constants
DTR = np.pi / 180.0  # degrees to radians
RE = 6.371e6  # Earth radius in meters


def calc_gradient(
    z: np.ndarray, lon: np.ndarray, lat: np.ndarray, method: str = "Horn1981"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient of elevation field in geographic coordinates.

    Parameters
    ----------
    z : 2D array
        Elevation values
    lon : 1D array
        Longitude values (columns)
    lat : 1D array
        Latitude values (rows)
    method : str
        "Horn1981" for smoothed gradient or "O1" for simple gradient

    Returns
    -------
    tuple of (dz/dx, dz/dy) in physical units (m/m)
    """
    if method not in ["Horn1981", "O1"]:
        raise ValueError("method must be either Horn1981 or O1")

    if method == "O1":
        dzdy2, dzdx2 = np.gradient(z)
    else:  # Horn1981
        dzdy, dzdx = np.gradient(z)
        dzdy2, dzdx2 = np.zeros(dzdy.shape), np.zeros(dzdx.shape)

        # Average [-1,0,0,1] gradient values at each point
        # At edges, use 3 points instead of 4
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

    # Calculate spacing in meters
    dx = RE * DTR * np.abs(lon[0] - lon[1])
    dy = RE * DTR * np.abs(lat[0] - lat[1])

    dx2d = dx * np.tile(np.cos(DTR * lat), (lon.size, 1)).T
    dy2d = dy * np.ones((lat.size, lon.size))

    return (dzdx2 / dx2d, dzdy2 / dy2d)


def smooth_2d_array(
    elev: np.ndarray, land_frac: float = 1.0, scalar: float = 1.0
) -> np.ndarray:
    """Smooth a 2D elevation array using FFT-based filter."""
    hw = scalar / (land_frac**2 * np.min(elev.shape))
    elev_fft = np.fft.rfft2(elev, norm="ortho")
    ny, nx = elev_fft.shape
    rowfreq = np.fft.fftfreq(elev.shape[0])
    colfreq = np.fft.rfftfreq(elev.shape[1])
    radialfreq = np.sqrt(
        np.tile(colfreq * colfreq, (ny, 1)) + np.tile(rowfreq * rowfreq, (nx, 1)).T
    )
    wl = np.exp(-radialfreq / hw)
    return np.fft.irfft2(wl * elev_fft, norm="ortho", s=elev.shape)


def fit_planar_surface(
    elev: np.ndarray, elon: np.ndarray, elat: np.ndarray
) -> np.ndarray:
    """Fit a planar surface to elevation data for detrending."""
    elon2d = np.tile(elon, (elat.size, 1))
    elat2d = np.tile(elat, (elon.size, 1)).T
    ncoef = 3
    g = np.zeros((elon2d.size, ncoef))
    g[:, 0] = elat2d.flat
    g[:, 1] = elon2d.flat
    g[:, 2] = 1
    gtd = np.dot(np.transpose(g), elev.flat)
    gtg = np.dot(np.transpose(g), g)
    covm = np.linalg.inv(gtg)
    coefs = np.dot(covm, gtd)
    return elat2d * coefs[0] + elon2d * coefs[1] + coefs[2]


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


def _fit_polynomial(
    x: np.ndarray, y: np.ndarray, ncoefs: int, weights: np.ndarray = None
) -> np.ndarray:
    """Fit polynomial coefficients using least squares."""
    im = x.size
    if im < ncoefs:
        raise RuntimeError(f"not enough data to fit {ncoefs} coefficients")

    g = np.zeros((im, ncoefs), dtype=np.float64)
    for n in range(ncoefs):
        g[:, n] = np.power(x, n)

    if weights is None:
        gtd = np.dot(np.transpose(g), y)
        gtg = np.dot(np.transpose(g), g)
    else:
        gtd = np.dot(np.transpose(g), np.dot(np.diag(weights), y))
        gtg = np.dot(np.transpose(g), np.dot(np.diag(weights), g))

    covm = np.linalg.inv(gtg)
    return np.dot(covm, gtd)


def _synth_polynomial(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Reconstruct polynomial from coefficients."""
    y = np.zeros(x.size, dtype=np.float64)
    for n in range(coefs.size):
        y += coefs[n] * np.power(x, n)
    return y


def _bin_amplitude_spectrum(
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


def _gaussian_no_norm(
    x: np.ndarray, amp: float, cen: float, sigma: float
) -> np.ndarray:
    """Unnormalized Gaussian function."""
    return amp * np.exp(-((x - cen) ** 2) / (2 * (sigma**2)))


def _log_normal(
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


def _fit_peak_gaussian(x: np.ndarray, y: np.ndarray, verbose: bool = False) -> dict:
    """Fit a Gaussian to locate a peak in the amplitude spectrum."""
    meansig = np.mean(y)
    pheight = (meansig, None)
    pwidth = (0, 0.75 * x.size)
    pprom = (0.2 * meansig, None)

    peaks, props = signal.find_peaks(
        y, height=pheight, prominence=pprom, width=pwidth, rel_height=0.5
    )

    # If no peak found, try reducing prominence
    if peaks.size == 0:
        pprom = (0.1 * meansig, None)
        peaks, props = signal.find_peaks(
            y, height=pheight, prominence=pprom, width=pwidth, rel_height=0.5
        )

    # Add edge peak test
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
        sigma = gsigma

        try:
            p0 = [amp, center, sigma]
            popt, _ = optimize.curve_fit(
                _gaussian_no_norm, x[i1 : i2 + 1], y[i1 : i2 + 1], p0=p0
            )
            pdist = np.abs(center - popt[1])
            if pdist > popt[2]:
                popt = [0, 0, 1]
        except Exception:
            popt = [0, 0, 1]

        peak_coefs.append(popt)
        peak_gof.append(
            np.mean(
                np.power(y[i1 : i2 + 1] - _gaussian_no_norm(x[i1 : i2 + 1], *popt), 2)
            )
        )

        if peak_gof[-1] < 1e6 and popt[0] > meansig:
            rwid = popt[2] / (x[-1] - x[0])
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


def _fit_peak_lognormal(x: np.ndarray, y: np.ndarray, verbose: bool = False) -> dict:
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
                _log_normal, x[i1 : i2 + 1], y[i1 : i2 + 1], p0=p0
            )
            ln_peak = np.exp(popt[2])
            pdist = np.abs(center - ln_peak)
            if pdist > popt[1]:
                popt = [0, 0, 1]
        except Exception:
            popt = [0, 0, 1]

        peak_coefs.append(popt)
        peak_gof.append(
            np.mean(np.power(y[i1 : i2 + 1] - _log_normal(x[i1 : i2 + 1], *popt), 2))
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


def _locate_peak(
    lambda_1d: np.ndarray,
    ratio_var_to_lambda: np.ndarray,
    max_wavelength: float = 1e6,
    min_wavelength: float = 1,
    verbose: bool = False,
) -> dict:
    """
    Fit ratio of variance to wavelength using linear and peaked models.
    Select best model to determine characteristic wavelength.
    """
    lmax = np.argmin(np.abs(lambda_1d - max_wavelength))
    lmin = np.argmin(np.abs(lambda_1d - min_wavelength))

    logLambda = np.log10(lambda_1d)

    # Linear fit
    lcoefs = _fit_polynomial(
        logLambda[lmin : lmax + 1], ratio_var_to_lambda[lmin : lmax + 1], 2
    )

    # Gaussian fit
    x_ga = _fit_peak_gaussian(
        logLambda[lmin : lmax + 1],
        ratio_var_to_lambda[lmin : lmax + 1],
        verbose=verbose,
    )
    pgauss = x_ga["coefs"]
    psharp_ga = x_ga["psharp"]
    gof_ga = x_ga["gof"]

    # Log-normal fit
    x_ln = _fit_peak_lognormal(
        logLambda[lmin : lmax + 1],
        ratio_var_to_lambda[lmin : lmax + 1],
        verbose=verbose,
    )
    plognorm = x_ln["coefs"]
    psharp_ln = x_ln["psharp"]
    gof_ln = x_ln["gof"]

    # Calculate t-score for linear fit
    num = (1 / (lmax - 2)) * np.sum(
        np.power(
            ratio_var_to_lambda[lmin : lmax + 1]
            - _synth_polynomial(logLambda[lmin : lmax + 1], lcoefs),
            2,
        )
    )
    den = np.sum(
        np.power(logLambda[lmin : lmax + 1] - np.mean(logLambda[lmin : lmax + 1]), 2)
    )
    se = np.sqrt(num / den) if den > 0 else 1e10
    tscore = np.abs(lcoefs[1]) / se if se > 0 else 0

    # Select best model
    psharp_threshold = 1.5
    tscore_threshold = 2
    model = "None"

    if psharp_ga >= psharp_threshold or psharp_ln >= psharp_threshold:
        if gof_ga < gof_ln:
            model = "gaussian"
            spatialScale = min(10 ** pgauss[1], max_wavelength)
            spatialScale = max(spatialScale, min_wavelength)
            selection = 1
            ocoefs = pgauss
        else:
            model = "lognormal"
            ln_peak = np.exp(plognorm[2])
            spatialScale = min(10**ln_peak, max_wavelength)
            spatialScale = max(spatialScale, min_wavelength)
            selection = 2
            ocoefs = plognorm
    else:
        if tscore > tscore_threshold:
            model = "linear"
            if lcoefs[1] > 0:
                spatialScale = max_wavelength
                selection = 3
            else:
                spatialScale = min_wavelength
                selection = 4
            ocoefs = lcoefs
        else:
            model = "flat"
            spatialScale = min_wavelength
            selection = 5
            ocoefs = [1]

    if verbose:
        print(f"  Model selected: {model}")
        print(f"  Peak sharpness (gaussian): {psharp_ga:.2f}")
        print(f"  Peak sharpness (lognormal): {psharp_ln:.2f}")
        print(f"  T-score (linear): {tscore:.2f}")

    return {
        "model": model,
        "spatialScale": spatialScale,
        "selection": selection,
        "coefs": ocoefs,
    }


def identify_spatial_scale_laplacian_dem(
    elev: np.ndarray,
    elon: np.ndarray,
    elat: np.ndarray,
    max_hillslope_length: float = 10000,
    land_threshold: float = 0.75,
    min_land_elevation: float = 0,
    detrend_elevation: bool = False,
    blend_edges_flag: bool = True,
    zero_edges: bool = True,
    nlambda: int = 30,
    verbose: bool = False,
) -> dict:
    """
    Identify the spatial scale at which the input DEM exhibits the
    largest divergence/convergence of topographic gradient.

    This is done by computing the Laplacian of the DEM, taking the 2D FFT,
    and finding the wavelength with maximum amplitude in the spectrum.

    Parameters
    ----------
    elev : 2D array
        Elevation values (rows=lat, cols=lon)
    elon : 1D array
        Longitude values (degrees)
    elat : 1D array
        Latitude values (degrees)
    max_hillslope_length : float
        Maximum hillslope length in meters (used to set max wavelength)
    land_threshold : float
        Fraction of land required (0-1)
    min_land_elevation : float
        Minimum elevation considered as land
    detrend_elevation : bool
        Whether to remove a planar trend
    blend_edges_flag : bool
        Whether to blend edges to reduce spectral leakage
    zero_edges : bool
        Whether to zero edges of Laplacian
    nlambda : int
        Number of wavelength bins for spectral analysis
    verbose : bool
        Print diagnostic information

    Returns
    -------
    dict with keys:
        - validDEM: bool
        - model: str (gaussian, lognormal, linear, flat)
        - spatialScale: float (characteristic wavelength in pixels)
        - res: float (resolution in meters)
        - lambda_1d: array (wavelength bins)
        - laplac_amp_1d: array (amplitude in each bin)
    """
    elev = np.copy(elev)
    ejm, eim = elev.shape

    # Approximate resolution in meters
    ares = np.abs(elat[0] - elat[1]) * (RE * np.pi / 180)
    max_wavelength = 2 * max_hillslope_length / ares

    if verbose:
        print(f"  DEM shape: {elev.shape}")
        print(f"  Resolution: {ares:.1f} m")
        print(f"  Max wavelength: {max_wavelength:.1f} pixels")

    # Create land/ocean mask
    lmask = np.where(elev > min_land_elevation, 1, 0)
    land_frac = np.sum(lmask) / lmask.size

    if verbose:
        print(f"  Land fraction: {land_frac:.2%}")

    min_land_fraction = 0.01
    if land_frac <= min_land_fraction:
        return {"validDEM": False}

    # If land fraction is low, remove smoothed elevation
    if land_frac < land_threshold:
        if verbose:
            print("  Removing smoothed elevation (coastal adjustment)")
        smooth_elev = smooth_2d_array(elev, land_frac=land_frac)
        elev -= smooth_elev

    # Remove planar trend if requested
    if detrend_elevation:
        elev_planar = fit_planar_surface(elev, elon, elat)
        elev -= elev_planar
        if verbose:
            print("  Planar surface removed")

    # Blend edges to reduce spectral leakage
    if blend_edges_flag:
        win = 4
        elev = blend_edges(elev, n=win)
        if verbose:
            print(f"  Edges blended (window={win})")

    # Calculate Laplacian
    grad = calc_gradient(elev, elon, elat)
    x = calc_gradient(grad[0], elon, elat)
    laplac = x[0]
    x = calc_gradient(grad[1], elon, elat)
    laplac += x[1]

    # Zero edges if requested
    if zero_edges:
        n = 5
        laplac[:n, :] = 0
        laplac[:, :n] = 0
        laplac[-n:, :] = 0
        laplac[:, -n:] = 0

    # Compute 2D FFT
    laplac_fft = np.fft.rfft2(laplac, norm="ortho")
    laplac_amp_fft = np.abs(laplac_fft)

    if verbose:
        print("  2D FFT computed")

    # Compute wavelength grid
    rowfreq = np.fft.fftfreq(ejm)
    colfreq = np.fft.rfftfreq(eim)

    ny, nx = laplac_fft.shape
    radialfreq = np.sqrt(
        np.tile(colfreq * colfreq, (ny, 1)) + np.tile(rowfreq * rowfreq, (nx, 1)).T
    )

    wavelength = np.zeros((ny, nx))
    wavelength[radialfreq > 0] = 1 / radialfreq[radialfreq > 0]
    wavelength[0, 0] = 2 * np.max(wavelength)

    # Bin amplitude spectrum
    x = _bin_amplitude_spectrum(laplac_amp_fft, wavelength, nlambda=nlambda)
    lambda_1d, laplac_amp_1d = x["lambda"], x["amp"]

    # Locate peak
    x = _locate_peak(
        lambda_1d, laplac_amp_1d, max_wavelength=max_wavelength, verbose=verbose
    )

    model = x["model"]
    spatialScale = x["spatialScale"]
    selection = x["selection"]

    # Set minimum wavelength
    min_wavelength = np.min(lambda_1d)
    spatialScale = max(spatialScale, min_wavelength)

    if verbose:
        print(f"  Spatial scale: {spatialScale:.1f} pixels")
        print(f"  Spatial scale: {spatialScale * ares:.0f} m")

    return {
        "validDEM": True,
        "model": model,
        "spatialScale": spatialScale,
        "selection": selection,
        "res": ares,
        "lambda_1d": lambda_1d,
        "laplac_amp_1d": laplac_amp_1d,
    }
