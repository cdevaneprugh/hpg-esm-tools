# MERIT Geographic Regression Test

Validates the pysheds fork's geographic CRS code path by computing hillslope
parameters for a known MERIT DEM gridcell and comparing to Swenson's published
data. Run this after any changes to `$PYSHEDS_FORK/pysheds/pgrid.py`.

## Input Data

| File | Description |
|------|-------------|
| `data/merit/n30w095_dem.tif` | MERIT DEM tile (90m, geographic CRS) |
| `data/reference/hillslopes_0.9x1.25_c240416.nc` | Swenson published hillslope parameters |

## What It Does

```
MERIT DEM tile
    |
    v
Extract 0.9x1.25 degree gridcell (32.0N, -93.1W to 33.0N, -91.9W)
    |
    v
FFT of Laplacian --> characteristic length scale (Lc)
    |                   (expected: 763m, tolerance: 5%)
    v
pysheds flow routing (resolve flats, compute flow direction/accumulation)
    |
    v
Stream network delineation (A_thresh = 0.5 * Lc^2)
    |
    v
HAND, DTND, slope, aspect per pixel
    |
    v
Bin by aspect (N/E/S/W) x elevation (4 bins) --> 16 hillslope elements
    |
    v
Compute 6 parameters per element: height, distance, slope, aspect, width, area
    |
    v
Correlate against published data (tolerance: 0.01)
    |
    v
PASS / FAIL
```

## Pass Criteria

| Parameter | Expected correlation |
|-----------|---------------------|
| Height (HAND) | 0.9979 |
| Distance (DTND) | 0.9992 |
| Slope | 0.9839 |
| Aspect (circular) | 1.0000 |
| Width | 0.9919 |
| Area fraction | 0.9244 |

All 6 must be within 0.01 of expected, and Lc within 5% of 763m.

## Usage

```bash
cd $SWENSON
sbatch scripts/merit_validation/merit_regression.sh
```

Expected runtime: ~10-20 min, 48GB RAM.

## Output

Written to `scripts/merit_validation/output/`:

| File | Content |
|------|---------|
| `results.json` | Machine-readable results with pass/fail per parameter |
| `summary.txt` | Human-readable summary table |
| `merit_regression_<jobid>.log` | SLURM job output |
