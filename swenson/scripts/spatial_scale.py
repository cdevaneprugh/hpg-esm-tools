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
    z: np.ndarray,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    pixel_size: float | None = None,
    method: str = "Horn1981",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient of elevation field.

    Supports two CRS modes:
    - **Geographic** (lon/lat provided): Haversine-based spacing that varies
      with latitude. This is the original Swenson implementation.
    - **UTM** (pixel_size provided): Uniform spacing in meters. Use for
      projected coordinate systems where pixels have equal physical size.

    Parameters
    ----------
    z : 2D array
        Elevation values (rows=y, cols=x)
    lon : 1D array, optional
        Longitude values (columns). Required for geographic mode.
    lat : 1D array, optional
        Latitude values (rows). Required for geographic mode.
    pixel_size : float, optional
        Pixel size in meters. Required for UTM mode.
    method : str
        "Horn1981" for smoothed gradient or "O1" for simple gradient

    Returns
    -------
    tuple of (dz/dx, dz/dy) in physical units (m/m)

    Raises
    ------
    ValueError
        If neither (lon, lat) nor pixel_size is provided, or if both are
        provided. Note: providing only one of lon/lat (without the other)
        counts as "no geographic coordinates" — UTM mode is used if
        pixel_size is also provided, otherwise ValueError is raised.

    Notes
    -----
    The Horn 1981 averaging block operates on raw np.gradient output (pixel
    units) before physical spacing is applied. It averages the [-1,0,0,1]
    stencil at each point to smooth the gradient. This averaging is purely
    index-based and CRS-independent — the CRS only enters when converting
    from pixel gradients to physical gradients (m/m) at the end.

    For the Laplacian (d2z/dx2 + d2z/dy2), this function is called three
    times: once on elevation, then once on each gradient component. The
    Laplacian differentiates each axis twice, so sign conventions from
    np.gradient cancel out. This is fundamentally different from aspect
    computation (STATUS.md #4) where first-derivative sign matters.
    """
    if method not in ["Horn1981", "O1"]:
        raise ValueError("method must be either Horn1981 or O1")

    # Validate CRS arguments: exactly one of (lon+lat) or pixel_size
    has_geographic = lon is not None and lat is not None
    has_utm = pixel_size is not None
    if has_geographic == has_utm:
        raise ValueError(
            "Provide either (lon, lat) for geographic CRS "
            "or pixel_size for UTM CRS, not both or neither."
        )

    # --- Gradient computation (CRS-independent) ---
    # np.gradient returns finite differences in array index units.
    # axis 0 = rows (y direction), axis 1 = columns (x direction).
    if method == "O1":
        dzdy2, dzdx2 = np.gradient(z)
    else:  # Horn1981
        dzdy, dzdx = np.gradient(z)
        dzdy2, dzdx2 = np.zeros(dzdy.shape), np.zeros(dzdx.shape)

        # Average [-1,0,0,1] gradient values at each point.
        # At edges, use 3 points instead of 4.
        # This smoothing operates on array indices only — no CRS involved.
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

    # --- Physical spacing conversion (CRS-dependent) ---
    if has_utm:
        # UTM: uniform spacing in both axes.
        # pixel_size is in meters, so dzdx2/pixel_size gives m/m.
        return (dzdx2 / pixel_size, dzdy2 / pixel_size)
    else:
        # Geographic: spacing varies with latitude due to converging meridians.
        # dx shrinks toward poles by cos(lat), dy is constant.
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
    elev: np.ndarray,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit a planar surface to elevation data for detrending.

    Parameters
    ----------
    elev : 2D array
        Elevation values (rows=y, cols=x)
    x_coords : 1D array, optional
        Column coordinates. If None, uses np.arange(ncols).
    y_coords : 1D array, optional
        Row coordinates. If None, uses np.arange(nrows).

    Returns
    -------
    2D array of fitted planar surface, same shape as elev

    Notes
    -----
    Planar detrending only needs monotonic coordinates — degrees, meters,
    or pixel indices all produce identical residuals. When coordinates are
    None, pixel indices are used, which is appropriate for UTM data where
    uniform spacing makes explicit coordinates unnecessary.

    Backward compatible: positional calls ``fit_planar_surface(elev, lon, lat)``
    still work because the old positional args (elon, elat) map to
    (x_coords, y_coords).
    """
    nrows, ncols = elev.shape
    if x_coords is None:
        x_coords = np.arange(ncols, dtype=np.float64)
    if y_coords is None:
        y_coords = np.arange(nrows, dtype=np.float64)

    x2d = np.tile(x_coords, (y_coords.size, 1))
    y2d = np.tile(y_coords, (x_coords.size, 1)).T
    ncoef = 3
    g = np.zeros((x2d.size, ncoef))
    g[:, 0] = y2d.flat
    g[:, 1] = x2d.flat
    g[:, 2] = 1
    gtd = np.dot(np.transpose(g), elev.flat)
    gtg = np.dot(np.transpose(g), g)
    covm = np.linalg.inv(gtg)
    coefs = np.dot(covm, gtd)
    return y2d * coefs[0] + x2d * coefs[1] + coefs[2]


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
        "psharp_ga": psharp_ga,
        "psharp_ln": psharp_ln,
        "gof_ga": gof_ga,
        "gof_ln": gof_ln,
        "tscore": tscore,
    }


def identify_spatial_scale_laplacian_dem(
    elev: np.ndarray,
    elon: np.ndarray | None = None,
    elat: np.ndarray | None = None,
    max_hillslope_length: float = 10000,
    land_threshold: float = 0.75,
    min_land_elevation: float = 0,
    detrend_elevation: bool = False,
    blend_edges_flag: bool = True,
    zero_edges: bool = True,
    nlambda: int = 30,
    verbose: bool = False,
    pixel_size: float | None = None,
    blend_edges_n: int | None = None,
    zero_edges_n: int | None = None,
    min_wavelength: float | None = None,
) -> dict:
    """
    Identify the spatial scale at which the input DEM exhibits the
    largest divergence/convergence of topographic gradient.

    This is done by computing the Laplacian of the DEM, taking the 2D FFT,
    and finding the wavelength with maximum amplitude in the spectrum.

    Supports two CRS modes:
    - **Geographic** (elon/elat provided): Original Swenson path for MERIT
      and other lat/lon DEMs. Resolution derived from lat spacing via
      haversine. Default blend/zero windows are small (4/5 pixels) because
      geographic DEMs are typically ~90m resolution.
    - **UTM** (pixel_size provided): For projected DEMs like NEON LIDAR.
      Resolution is pixel_size directly. Default blend/zero windows are
      larger (50 pixels) because UTM DEMs are typically 1m resolution, so
      50 pixels = 50m — comparable to the geographic defaults in physical
      distance.

    Parameters
    ----------
    elev : 2D array
        Elevation values (rows=y, cols=x)
    elon : 1D array, optional
        Longitude values (degrees). Required for geographic mode.
    elat : 1D array, optional
        Latitude values (degrees). Required for geographic mode.
    max_hillslope_length : float
        Maximum hillslope length in meters (used to set max wavelength)
    land_threshold : float
        Fraction of land required (0-1). Only used in geographic mode
        for coastal gridcells; UTM DEMs are assumed to be fully land.
    min_land_elevation : float
        Minimum elevation considered as land. In geographic mode, 0 is
        appropriate (ocean). In UTM mode with LIDAR data, set this below
        the nodata sentinel (e.g., -9999) to correctly identify valid pixels.
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
    pixel_size : float, optional
        Pixel size in meters. Triggers UTM mode when provided.
    blend_edges_n : int, optional
        Edge blend window size in pixels. Defaults: 4 (geographic), 50 (UTM).
    zero_edges_n : int, optional
        Zero-edge window size in pixels. Defaults: 5 (geographic), 50 (UTM).
    min_wavelength : float, optional
        Minimum wavelength in pixels passed to _locate_peak. Default: 1.

    Returns
    -------
    dict with keys:
        - validDEM: bool
        - model: str (gaussian, lognormal, linear, flat)
        - spatialScale: float (characteristic wavelength in pixels)
        - spatialScale_m: float (characteristic wavelength in meters)
        - res: float (resolution in meters)
        - lambda_1d: array (wavelength bins)
        - laplac_amp_1d: array (amplitude in each bin)
        - selection: int (model selection code)
        - psharp_ga, psharp_ln: float (peak sharpness for each model)
        - gof_ga, gof_ln: float (goodness of fit for each model)
        - tscore: float (t-score for linear fit)
    """
    # --- CRS detection ---
    has_geographic = elon is not None and elat is not None
    has_utm = pixel_size is not None
    if has_geographic == has_utm:
        raise ValueError(
            "Provide either (elon, elat) for geographic CRS "
            "or pixel_size for UTM CRS, not both or neither."
        )

    elev = np.copy(elev)
    ejm, eim = elev.shape

    # --- Resolution and max wavelength ---
    if has_utm:
        # UTM: pixel_size is the resolution directly, in meters.
        ares = pixel_size
    else:
        # Geographic: approximate resolution from latitude spacing.
        # abs(dlat) * RE * DTR converts degree spacing to meters.
        ares = np.abs(elat[0] - elat[1]) * (RE * np.pi / 180)

    max_wavelength = 2 * max_hillslope_length / ares

    if verbose:
        crs_label = "UTM" if has_utm else "Geographic"
        print(f"  CRS: {crs_label}")
        print(f"  DEM shape: {elev.shape}")
        print(f"  Resolution: {ares:.1f} m")
        print(f"  Max wavelength: {max_wavelength:.1f} pixels")

    # --- Land/ocean mask ---
    lmask = np.where(elev > min_land_elevation, 1, 0)
    land_frac = np.sum(lmask) / lmask.size

    if verbose:
        print(f"  Land fraction: {land_frac:.2%}")

    min_land_fraction = 0.01
    if land_frac <= min_land_fraction:
        return {"validDEM": False}

    # --- Nodata handling ---
    if has_utm:
        # UTM path: fill nodata with mean elevation before FFT.
        # UTM DEMs (e.g., NEON LIDAR) use nodata for gaps rather than
        # ocean, so smoothed-elevation subtraction doesn't apply.
        valid_mask = elev > min_land_elevation
        if not np.all(valid_mask):
            elev_mean = np.mean(elev[valid_mask])
            elev[~valid_mask] = elev_mean
            if verbose:
                print(f"  Nodata filled with mean elevation ({elev_mean:.1f} m)")
    else:
        # Geographic path: if land fraction is low (coastal gridcell),
        # remove smoothed elevation to reduce ocean-land boundary artifacts.
        if land_frac < land_threshold:
            if verbose:
                print("  Removing smoothed elevation (coastal adjustment)")
            smooth_elev = smooth_2d_array(elev, land_frac=land_frac)
            elev -= smooth_elev

    # --- Planar detrending ---
    if detrend_elevation:
        if has_utm:
            # UTM: use pixel indices (uniform spacing makes explicit
            # coordinates unnecessary for planar detrending).
            elev_planar = fit_planar_surface(elev)
        else:
            # Geographic: use lon/lat coordinates.
            elev_planar = fit_planar_surface(elev, elon, elat)
        elev -= elev_planar
        if verbose:
            print("  Planar surface removed")

    # --- Edge blending ---
    # Default window sizes differ by CRS because pixel sizes differ:
    # Geographic ~90m pixels → 4 pixels ≈ 360m physical window.
    # UTM 1m pixels → 50 pixels = 50m physical window.
    if blend_edges_flag:
        if blend_edges_n is not None:
            win = blend_edges_n
        else:
            win = 50 if has_utm else 4
        elev = blend_edges(elev, n=win)
        if verbose:
            print(f"  Edges blended (window={win})")

    # --- Laplacian ---
    # The Laplacian is d2z/dx2 + d2z/dy2. Each axis is differentiated
    # twice, so the sign convention of np.gradient cancels out on each
    # axis. This is why there is no N/S sign issue here (unlike aspect,
    # which uses first derivatives — see STATUS.md #4).
    if has_utm:
        grad = calc_gradient(elev, pixel_size=pixel_size)
        x = calc_gradient(grad[0], pixel_size=pixel_size)
        laplac = x[0]
        x = calc_gradient(grad[1], pixel_size=pixel_size)
        laplac += x[1]
    else:
        grad = calc_gradient(elev, elon, elat)
        x = calc_gradient(grad[0], elon, elat)
        laplac = x[0]
        x = calc_gradient(grad[1], elon, elat)
        laplac += x[1]

    # --- Zero edges ---
    if zero_edges:
        if zero_edges_n is not None:
            n = zero_edges_n
        else:
            n = 50 if has_utm else 5
        laplac[:n, :] = 0
        laplac[:, :n] = 0
        laplac[-n:, :] = 0
        laplac[:, -n:] = 0
        if verbose:
            print(f"  Edges zeroed (window={n})")

    # --- 2D FFT ---
    laplac_fft = np.fft.rfft2(laplac, norm="ortho")
    laplac_amp_fft = np.abs(laplac_fft)

    if verbose:
        print("  2D FFT computed")

    # --- Wavelength grid ---
    # Wavelength is in pixel units. Multiply by ares to get meters.
    rowfreq = np.fft.fftfreq(ejm)
    colfreq = np.fft.rfftfreq(eim)

    ny, nx = laplac_fft.shape
    radialfreq = np.sqrt(
        np.tile(colfreq * colfreq, (ny, 1)) + np.tile(rowfreq * rowfreq, (nx, 1)).T
    )

    wavelength = np.zeros((ny, nx))
    wavelength[radialfreq > 0] = 1 / radialfreq[radialfreq > 0]
    wavelength[0, 0] = 2 * np.max(wavelength)

    # --- Bin amplitude spectrum ---
    x = _bin_amplitude_spectrum(laplac_amp_fft, wavelength, nlambda=nlambda)
    lambda_1d, laplac_amp_1d = x["lambda"], x["amp"]

    # --- Locate peak ---
    peak_min_wl = min_wavelength if min_wavelength is not None else 1
    x = _locate_peak(
        lambda_1d,
        laplac_amp_1d,
        max_wavelength=max_wavelength,
        min_wavelength=peak_min_wl,
        verbose=verbose,
    )

    model = x["model"]
    spatialScale = x["spatialScale"]
    selection = x["selection"]

    # Enforce minimum wavelength from the binned spectrum
    lambda_min = np.min(lambda_1d)
    spatialScale = max(spatialScale, lambda_min)

    if verbose:
        print(f"  Spatial scale: {spatialScale:.1f} pixels")
        print(f"  Spatial scale: {spatialScale * ares:.0f} m")

    return {
        "validDEM": True,
        "model": model,
        "spatialScale": spatialScale,
        "spatialScale_m": spatialScale * ares,
        "selection": selection,
        "res": ares,
        "lambda_1d": lambda_1d,
        "laplac_amp_1d": laplac_amp_1d,
        "psharp_ga": x["psharp_ga"],
        "psharp_ln": x["psharp_ln"],
        "gof_ga": x["gof_ga"],
        "gof_ln": x["gof_ln"],
        "tscore": x["tscore"],
    }
