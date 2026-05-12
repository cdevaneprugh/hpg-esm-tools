"""
DEM Processing Functions for OSBS Pipeline

Basin detection and open water identification for DEM conditioning.

Copied from merit_regression.py (the validated source of truth) for
intentional decoupling: merit_regression.py stays frozen as a regression
test while this module evolves for 1m-specific DEM conditioning
(e.g., synthetic lake bottoms, other preprocessing).

Swenson reference: geospatial_utils.py
"""

import numpy as np


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
