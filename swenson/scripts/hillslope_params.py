"""
Hillslope Parameter Functions

Shared functions for computing the 6 geomorphic parameters (height, distance,
area, slope, aspect, width) from HAND/DTND/slope/aspect fields. Used by both
the MERIT validation pipeline and the OSBS pipeline.

Extracted from merit_regression.py (the validated source of truth).
Cross-referenced against Swenson's originals:
  - representative_hillslope.py: calc_width_parameters, _fit_polynomial
  - terrain_utils.py: SpecifyHandBounds, set_aspect_to_hillslope_mean_serial, TailIndex
  - geospatial_utils.py: quadratic
"""

import numpy as np
from scipy.stats import expon

# Constants — defined independently from spatial_scale.py because this module
# serves both geographic and UTM pipelines.
DTR = np.pi / 180.0  # degrees to radians
RE = 6.371e6  # Earth radius (meters)


def get_aspect_mask(aspect: np.ndarray, aspect_bin: tuple) -> np.ndarray:
    """Create boolean mask for pixels within an aspect bin.

    Aspect bins are defined as (lower, upper) degree ranges. Most bins use
    simple AND logic (lower <= aspect < upper). The North bin (315, 45)
    crosses the 0/360 boundary and requires OR logic: aspect >= 315 OR
    aspect < 45. This is detected by lower > upper.
    """
    lower, upper = aspect_bin
    if lower > upper:
        # Wrapping bin (e.g. North: 315-360 and 0-45)
        return (aspect >= lower) | (aspect < upper)
    return (aspect >= lower) & (aspect < upper)


def compute_hand_bins(
    hand: np.ndarray,
    aspect: np.ndarray,
    aspect_bins: list,
    bin1_max: float = 2.0,
    min_aspect_fraction: float = 0.01,
) -> np.ndarray:
    """Compute HAND bin boundaries following Swenson's SpecifyHandBounds().

    Creates 4 HAND bins (5 boundaries including 0 and 1e6) with approximately
    equal pixel counts, subject to the constraint that the lowest bin upper
    bound is <= bin1_max (default 2m). This constraint ensures the soil column
    extends below stream channel elevation for realistic water table dynamics.

    Two branches:
      - Q25 <= bin1_max (common case): simple quartile boundaries.
      - Q25 > bin1_max (high-relief terrain): force lowest bin to bin1_max,
        split remaining pixels into thirds.

    Swenson reference: terrain_utils.py:299-412, "fastsort" method.

    Parameters
    ----------
    hand : 1D array of HAND values (pre-filtered to valid pixels)
    aspect : 1D array of aspect values (same size as hand)
    aspect_bins : list of (lower, upper) degree tuples for each aspect bin
    bin1_max : maximum upper bound for the lowest HAND bin (meters)
    min_aspect_fraction : minimum fraction of pixels per aspect bin that must
        fall below bin1_max (prevents empty lowland bins)

    Returns
    -------
    1D array of 5 bin boundaries: [0, b1, b2, b3, 1e6]
    """
    # Filter to non-channel pixels with finite HAND. HAND=0 means the pixel
    # IS the stream channel; those don't belong in any hillslope bin.
    # Matches Swenson's fhand[fhand > 0] (tu:307, 353).
    valid = (hand > 0) & np.isfinite(hand)
    hand_valid = hand[valid]

    if hand_valid.size == 0:
        return np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])

    hand_sorted = np.sort(hand_valid)
    n = hand_sorted.size
    # Q25 of positive HAND values. Assumes n >= 4 (always true in practice
    # since we're processing gridcells with thousands of pixels).
    initial_q25 = hand_sorted[int(0.25 * n) - 1] if n > 0 else 0

    if initial_q25 > bin1_max:
        # HIGH-RELIEF BRANCH (Swenson tu:364-408):
        # Q25 exceeds bin1_max, so we can't use simple quartiles. Instead:
        # 1. Expand bin1_max if needed so each aspect bin has at least
        #    min_aspect_fraction of its pixels below the threshold.
        # 2. Split the above-bin1_max pixels into equal thirds.
        for asp_idx, (asp_low, asp_high) in enumerate(aspect_bins):
            if asp_low > asp_high:
                asp_mask = (aspect >= asp_low) | (aspect < asp_high)
            else:
                asp_mask = (aspect >= asp_low) & (aspect < asp_high)

            # Find the HAND value at the min_aspect_fraction percentile
            # for this aspect bin. If it exceeds bin1_max, raise the
            # threshold to ensure non-empty lowland bins.
            hand_asp_sorted = np.sort(hand[asp_mask])
            if hand_asp_sorted.size > 0:
                bmin = hand_asp_sorted[
                    int(min_aspect_fraction * hand_asp_sorted.size - 1)
                ]
            else:
                bmin = bin1_max

            if bmin > bin1_max:
                bin1_max = bmin

        # Split pixels above bin1_max into equal thirds for bins 2-4.
        above_bin1 = hand_sorted[hand_sorted > bin1_max]
        if above_bin1.size > 0:
            n_above = above_bin1.size
            b33 = above_bin1[int(0.33 * n_above - 1)]
            b66 = above_bin1[int(0.66 * n_above - 1)]
            # Guard: if b33 == b66 (degenerate distribution), shift b66
            # upward to create distinct bins. Swenson (tu:405-407):
            # "just shift b66 for now".
            if b33 == b66:
                b66 = 2 * b33 - bin1_max
            bounds = np.array([0, bin1_max, b33, b66, 1e6])
        else:
            bounds = np.array([0, bin1_max, bin1_max * 2, bin1_max * 4, 1e6])
    else:
        # COMMON CASE (Swenson tu:350-361): Q25 <= bin1_max.
        # Simple quartile boundaries produce approximately equal-area bins.
        # Last bound is hand_sorted[-1] = max(hand), so the last bin is
        # [Q75, max(hand)) — excludes pixels at exactly max(hand).
        quartiles = [0.25, 0.5, 0.75, 1.0]
        bounds = [0.0]
        for q in quartiles:
            idx = max(0, int(q * n) - 1)
            bounds.append(hand_sorted[idx])
        bounds = np.array(bounds)

    return bounds


def fit_trapezoidal_width(
    dtnd: np.ndarray,
    area: np.ndarray,
    n_hillslopes: int,
    min_dtnd: float = 1.0,
    n_bins: int = 10,
) -> dict:
    """
    Fit trapezoidal plan form following Swenson Eq. (4).

    Models cumulative hillslope area as a function of distance from channel:

        A(d) = A_trap - w_base * d - slope * d²

    where A_trap is total per-hillslope area, w_base is base width at the
    channel, and slope controls convergence (positive) or divergence (negative)
    of the hillslope plan form. The polynomial coefficients are fit to the
    empirical cumulative area curve A_cumsum(d) using weighted least squares.

    Uses Swenson's _fit_polynomial weighting (rh:113-136): W = diag(weights),
    solving (G^T W G) coefs = G^T W y. This minimizes
    sum_i w_i * (residual_i)^2 (w^1 weighting, where weights = A_cumsum).

    Swenson reference: calc_width_parameters (rh:54-96).

    Parameters
    ----------
    dtnd : 1D array of DTND values for one aspect bin
    area : 1D array of pixel areas (same size as dtnd)
    n_hillslopes : number of independent hillslopes (unique drainage IDs)
        in this aspect bin — used to normalize areas to per-hillslope values.
        Swenson divides at the call site (rh:768); we divide internally.
    min_dtnd : minimum DTND to include in binning (meters). Callers pass
        the DEM resolution (e.g. 90m for MERIT, pixel_size for UTM).
    n_bins : number of distance bins for the cumulative area curve

    Returns
    -------
    dict with keys: slope, width, area (the trapezoidal fit parameters)
    """
    # Degenerate case: all DTND values at or below minimum threshold.
    # Return fallback with slope=0 (parallel-sided), width as a small
    # fraction of total area (heuristic), and total per-hillslope area.
    if np.max(dtnd) <= min_dtnd:
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }

    # Build cumulative area curve: A_cumsum[k] = total area of pixels
    # with DTND >= d[k]. This is a decreasing function of distance —
    # fewer pixels remain as you move away from the channel.
    dtnd_bins = np.linspace(min_dtnd, np.max(dtnd) + 1, n_bins + 1)
    d = np.zeros(n_bins)
    A_cumsum = np.zeros(n_bins)

    for k in range(n_bins):
        mask = dtnd >= dtnd_bins[k]
        d[k] = dtnd_bins[k]
        A_cumsum[k] = np.sum(area[mask])

    # Normalize to per-hillslope area. Swenson does this at the call site
    # (rh:768: farea[aind] / number_of_hillslopes[asp_ndx]); we do it here.
    A_cumsum /= n_hillslopes

    # Prepend d=0 intercept: total per-hillslope area (the cumulative area
    # at distance 0 is the entire hillslope).
    if min_dtnd > 0:
        d = np.concatenate([[0], d])
        A_cumsum = np.concatenate([[np.sum(area) / n_hillslopes], A_cumsum])

    try:
        # Weighted least squares: fit A(d) = c0 + c1*d + c2*d²
        # G = Vandermonde matrix [1, d, d²]
        # W = diagonal weight matrix (weights = A_cumsum for w^1 weighting)
        # Solve normal equations: (G^T W G) c = G^T W y
        # Equivalent to Swenson's _fit_polynomial (rh:113-136) but uses
        # np.linalg.solve instead of np.linalg.inv (more numerically stable).
        weights = A_cumsum
        G = np.column_stack([np.ones_like(d), d, d**2])
        W = np.diag(weights)
        GtWG = G.T @ W @ G
        GtWy = G.T @ W @ A_cumsum
        coeffs = np.linalg.solve(GtWG, GtWy)

        # Extract trapezoidal parameters from polynomial coefficients.
        # A(d) = c0 + c1*d + c2*d², and the trapezoidal model is
        # A(d) = A_trap - w_base*d - slope*d², so:
        #   A_trap = c0, w_base = -c1, slope = -c2
        trap_slope = -coeffs[2]
        trap_width = -coeffs[1]
        trap_area = coeffs[0]

        # Width adjustment for convergent hillslopes (Swenson rh:87-94):
        # If slope is negative (convergent), the triangle formed by width
        # alone (Atri = w²/4|slope|) may have less area than the total
        # fitted area. In that case, increase width so the trapezoid can
        # contain the full area: w = sqrt(4 * |slope| * A_trap).
        if trap_slope < 0:
            Atri = -(trap_width**2) / (4 * trap_slope)
            if Atri < trap_area:
                trap_width = np.sqrt(-4 * trap_slope * trap_area)

        return {"slope": trap_slope, "width": max(trap_width, 1), "area": trap_area}
    except Exception:
        # Fallback for singular/ill-conditioned matrices.
        # Same heuristic as the degenerate early-return case.
        return {
            "slope": 0,
            "width": np.sum(area) / n_hillslopes / 100,
            "area": np.sum(area) / n_hillslopes,
        }


def quadratic(coefs, root=0, eps=1e-6):
    """Solve quadratic equation ax^2 + bx + c = 0.

    Used to find distance along the trapezoidal hillslope given an accumulated
    area. For width-at-bin-edge: solve slope*d² + width*d - area_below = 0.
    For distance-to-bin-midpoint: same equation with area_below = midpoint area.

    Swenson reference: geospatial_utils.py:168-188.

    Parameters
    ----------
    coefs : (a, b, c) tuple of quadratic coefficients
    root : 0 for the + root, 1 for the - root
    eps : tolerance for near-zero negative discriminants (floating-point fix)
    """
    ak, bk, ck = coefs
    discriminant = bk**2 - 4 * ak * ck

    if discriminant < 0:
        # Near-zero negative discriminant from floating-point roundoff:
        # adjust c to make discriminant exactly zero (tangent solution).
        # Larger negative discriminants indicate genuinely inconsistent
        # fit parameters — no real solution exists.
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
    """Compute circular mean of aspect values (degrees).

    Standard circular mean: convert to unit vectors on the circle,
    average the sin and cos components, convert back to angle.
    """
    sin_sum = np.mean(np.sin(DTR * aspects))
    cos_sum = np.mean(np.cos(DTR * aspects))
    mean_aspect = np.arctan2(sin_sum, cos_sum) / DTR
    if mean_aspect < 0:
        mean_aspect += 360
    return mean_aspect


def catchment_mean_aspect(
    drainage_id: np.ndarray,
    aspect: np.ndarray,
    hillslope: np.ndarray,
    chunksize: int = 500,
) -> np.ndarray:
    """Replace per-pixel aspect with catchment-side circular mean.

    For each catchment (unique drainage_id), for each hillslope side
    (headwater=1, right bank=2, left bank=3), compute the circular mean
    aspect of all pixels in that group plus channel pixels (type 4).
    Assign the mean back to every pixel in the group.

    This ensures all pixels on the same side of a catchment share one
    aspect value, preventing local noise from splitting catchments across
    aspect bins during binning.

    Follows Swenson's set_aspect_to_hillslope_mean_serial
    (terrain_utils.py:236-279).

    Parameters
    ----------
    drainage_id : 2D ndarray — catchment ID per pixel (from compute_hand)
    aspect : 2D ndarray — per-pixel aspect in degrees
    hillslope : 2D ndarray — hillslope classification
        1=headwater, 2=right bank, 3=left bank, 4=channel

    Returns
    -------
    2D ndarray — aspect with catchment-side means replacing pixel values
    """
    valid_drain = np.isfinite(drainage_id) & (drainage_id > 0)
    uid = np.unique(drainage_id[valid_drain])
    hillslope_types = np.unique(hillslope[hillslope > 0]).astype(int)

    out = np.zeros(aspect.shape)

    # Swenson (tu:184-185): guard against empty drainage_id.
    # Can occur if all drainage_id values are NaN or zero.
    if uid.size == 0:
        return out
    valid_aspect = np.isfinite(aspect.flat)

    # Process drainage IDs in chunks to reduce the cost of np.where searches
    # on large flat arrays. Each chunk pre-selects a range of drainage IDs,
    # then iterates over individual IDs within that range.
    # cs = min(chunksize, uid.size - 1) ensures chunk size doesn't exceed
    # the number of unique IDs; the last chunk extends to cover the remainder.
    nchunks = int(max(1, int(uid.size // chunksize)))
    cs = int(min(chunksize, uid.size - 1))

    for n in range(nchunks):
        n1, n2 = int(n * cs), int((n + 1) * cs)
        if n == nchunks - 1:
            n2 = uid.size - 1
        # Single-drainage-ID optimization: use exact equality instead of
        # range comparison when the chunk contains exactly one ID.
        if n1 == n2:
            cind = np.where(valid_aspect & (drainage_id.flat == uid[n1]))[0]
        else:
            cind = np.where(
                valid_aspect
                & (drainage_id.flat >= uid[n1])
                & (drainage_id.flat < uid[n2])
            )[0]

        for did in uid[n1 : n2 + 1]:
            dind = cind[drainage_id.flat[cind] == did]
            # Loop over non-channel hillslope types (1, 2, 3). Skip type 4
            # (channel) because channels are combined WITH each other type,
            # not processed alone. hillslope_types is sorted, so type 4 is
            # last — slicing off the last element excludes it.
            for ht in hillslope_types[: hillslope_types.size - 1]:
                # Union this hillslope type with channel pixels (type 4)
                sel = (hillslope.flat[dind] == 4) | (hillslope.flat[dind] == ht)
                ind = dind[sel]
                if ind.size > 0:
                    mean_asp = (
                        np.arctan2(
                            np.mean(np.sin(DTR * aspect.flat[ind])),
                            np.mean(np.cos(DTR * aspect.flat[ind])),
                        )
                        / DTR
                    )
                    if mean_asp < 0:
                        mean_asp += 360.0
                    out.flat[ind] = mean_asp

    return out


def compute_pixel_areas(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Compute pixel areas on a sphere using geographic coordinates.

    Uses the spherical surface element: dA = R² sin(θ) dθ dφ
    where θ = colatitude (90° - latitude), φ = longitude (radians).
    Area varies by latitude (sin(θ) term) but is uniform across longitude.

    Geographic CRS only. UTM callers use uniform pixel_size² instead.

    Swenson reference: representative_hillslope.py:1708-1715.

    Parameters
    ----------
    lon : 1D array of longitude values (degrees)
    lat : 1D array of latitude values (degrees)

    Returns
    -------
    2D array of pixel areas (m²), shape (len(lat), len(lon))
    """
    phi = DTR * lon
    theta = DTR * (90.0 - lat)
    dphi = np.abs(phi[1] - phi[0])
    dtheta = np.abs(theta[0] - theta[1])
    sin_theta = np.sin(theta)
    ncols = len(lon)
    area = np.tile(sin_theta.reshape(-1, 1), (1, ncols))
    area = area * dtheta * dphi * RE**2
    return area


def tail_index(
    dtnd: np.ndarray, hand: np.ndarray, npdf_bins: int = 5000, hval: float = 0.05
) -> np.ndarray:
    """
    Return indices of pixels with DTND below the tail threshold.

    Swenson (tu:286-296): fits exponential distribution to DTND (where
    HAND > 0), normalized by its standard deviation. Finds the DTND value
    where the fitted PDF drops to hval (5%) of its maximum. Pixels beyond
    that threshold are considered outliers from DEM flooding/inflation.

    **Important:** Input arrays must be pre-filtered to valid pixels only.
    Returned indices are into the input arrays, not the original full arrays.
    Callers must map back if needed (see merit_regression.py usage).

    Parameters
    ----------
    dtnd : 1D array of DTND values (pre-filtered to valid pixels)
    hand : 1D array of HAND values (same size, pre-filtered)
    npdf_bins : number of bins for PDF evaluation
    hval : fraction of max PDF to use as tail cutoff (0.05 = 5%)

    Returns
    -------
    1D array of indices (into input arrays) where dtnd < tail cutoff
    """
    positive_hand = hand > 0
    if np.sum(positive_hand) == 0:
        return np.arange(dtnd.size)

    dtnd_pos = dtnd[positive_hand]
    # Population std dev (ddof=0), matching Swenson's custom std_dev()
    # function (tu:282-283).
    std_dtnd = np.std(dtnd_pos)
    if std_dtnd == 0:
        return np.arange(dtnd.size)

    # Normalize DTND by std before fitting exponential — improves numerical
    # stability of the MLE fit for distributions with large absolute values.
    fit_loc, fit_beta = expon.fit(dtnd_pos / std_dtnd)
    rv = expon(loc=fit_loc, scale=fit_beta)

    # Evaluate fitted PDF on a fine grid and find where it drops to hval
    # (5%) of its peak value. Pixels with DTND beyond this threshold are
    # considered tail outliers from DEM flooding/inflation artifacts.
    pbins = np.linspace(0, np.max(dtnd), npdf_bins)
    rvpdf = rv.pdf(pbins / std_dtnd)
    r1 = np.argmin(np.abs(rvpdf - hval * np.max(rvpdf)))
    return np.where(dtnd < pbins[r1])[0]
