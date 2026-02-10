# Phase C: Establish Trustworthy Characteristic Length Scale (Lc)

Status: Not started
Depends on: Phase B (for final resolution decision, but FFT itself is independent)
Blocks: Phase D

## Problem

Everything downstream depends on Lc — accumulation threshold, stream network density, HAND, DTND, all 6 hillslope parameters (STATUS.md #3). The current Lc values are not trustworthy:

| Dataset | Lc source | Lc value | Threshold | Stream coverage |
|---------|-----------|----------|-----------|-----------------|
| MERIT | FFT peak | 763m | 34 cells | 2.17% |
| OSBS (full, 4x sub) | Forced minimum | 100m | 312 cells | 2.32% |
| OSBS (interior, 4x sub) | FFT peak (4m res) | 166m | 864 cells | 1.44% |

The full-resolution FFT on the OSBS interior has never been run. numpy handles arrays of this size trivially.

Additionally, FFT parameters (#8) were copied from Swenson's 90m defaults without validation at 1m. At 1m resolution, several parameters have qualitatively different effects (blend_edges covers 7x less geographic area, zero_edges covers 90x less, NLAMBDA spans 3+ orders of magnitude instead of 1.5).

## Tasks

- [ ] Run full-resolution (1m) FFT on interior mosaic (decoupled from flow routing)
- [ ] Run FFT parameter sensitivity tests (one variable at a time on a representative region):

| Test | Variable | Values |
|------|----------|--------|
| A | blend_edges window | 4, 25, 50, 100, 200 |
| B | zero_edges margin | 5, 20, 50, 100 |
| C | NLAMBDA | 20, 30, 50, 75 |
| D | MAX_HILLSLOPE_LENGTH | 500m, 1km, 2km, 10km |
| E | detrend_elevation | True, False |
| F | Region size | 2000x2000, 5000x5000, 8000x8000 |

- [ ] Determine whether Lc is stable or sensitive to parameters
- [ ] Set final Lc value with justification (stable = pick it and move on; sensitive = identify which parameters need calibration)

## Deliverable

Lc value with error bounds or sensitivity analysis. Clear record of what was tested and what the results show.

## Log

