# Hillslope Binning Rationale: 1x8 Log-Spaced HAND Bins

Date: 2026-03-19

## Decision

Replace the 4-aspect x 4-equal-area-bin hillslope structure (16 columns) with a 1-aspect x
8-log-spaced-bin structure (8 columns) for the OSBS pipeline.

## Why Drop 4 Aspects

OSBS is a low-relief wetlandscape with slopes of 0.01-0.06 m/m. The 4-aspect split (N, E, S, W)
is wasteful for three reasons:

1. **Uniform aspect distribution.** Aspect is nearly uniformly distributed (~25% per quadrant)
   across the production domain. The 4 aspects represent redundant copies of the same hillslope
   profile, not physically distinct hillslopes.

2. **Negligible insolation correction.** Aspect enters CTSM through `shr_orb_cosinc()` in
   `SurfaceAlbedoMod.F90:264`, which adjusts solar incidence angle by
   `slope * cos(aspect - solar_azimuth)`. At OSBS slopes, the maximum correction is 3-6% of the
   cosine of the solar zenith angle. This does not meaningfully affect water or carbon cycling.

3. **Wasted columns.** On the R6C10 single tile, North and South aspects are completely empty
   (zero area in all 4 bins) because the tile's drainage runs east-west. This is 8 of 16 columns
   producing no useful output. Even at tier 3 scale, the 4 aspects contain essentially the same
   information repeated 4 times.

Aspect does NOT enter the lateral flow equations directly (`SoilHydrologyMod.F90:2260-2372`).
The hydrological physics that matters for TAI dynamics is driven by `hill_elev`, `hill_slope`,
`hill_distance`, and `hill_width` -- not aspect.

## Why 8 Bins, Not 4 or 16

**Not 4:** The current 4 equal-area bins give one bin in the 0-0.3m HAND range and one covering
0.3-3.8m. The TAI transition zone (where the water table intersects the surface) spans roughly
0-2m HAND at OSBS. With only 1-2 bins in this zone, the pipeline cannot resolve the
wetland-upland gradient that drives the bulk of carbon exchange.

**Not 16:** Log spacing with 16 bins creates near-stream bins narrower than the DEM conditioning
noise floor. `fill_pits` and `fill_depressions` introduce cm-scale artifacts in HAND values.
With 16 log-spaced bins, the smallest bins near the stream would have widths of a few cm in HAND,
which is indistinguishable from conditioning noise. Additionally, on smaller domains (single tiles
with ~500K valid pixels), the smallest bins could contain only a few thousand pixels, producing
noisy statistics for the trapezoidal width fit and per-bin parameter averages.

**8 is the sweet spot:** With log spacing, roughly 4-5 of the 8 bins fall in the 0-3m TAI zone,
giving sub-meter HAND resolution where it matters. At tier 3 (~800K valid pixels), each bin
contains ~50-100K pixels even with unequal log-spaced areas. The bins are wide enough to be
well above the DEM conditioning noise floor.

Computationally, 8 columns is half the soil physics calculations per timestep compared to 16.
Over multi-century spinups, this is a meaningful savings.

## Why Log Spacing, Not Equal-Area

Equal-area binning (Swenson's default) assigns each bin approximately the same number of pixels.
At OSBS, this means:

- Q25 of positive HAND is only 0.31m → the first bin covers 0-0.31m (stream-adjacent to 31cm)
- Q50 is 3.85m → the second bin covers 0.31-3.85m
- Q75 is 8.70m → the third bin covers 3.85-8.70m

The second bin (0.31-3.85m) spans the entire TAI transition zone as a single bin. This is
insufficient resolution for the physics we want to capture.

Log spacing concentrates bins where the hydrological gradient is steepest -- near the stream.
A 30cm HAND difference near the stream (water table crosses the surface) drives more
hydrological change than a 3m difference near the ridge. CTSM's lateral subsurface flow depends
on the hydraulic gradient dh/dd between adjacent columns. Finer bins near the stream produce
smoother gradient representation in the zone where most lateral flow occurs.

## Percentile-Based Bin Edges

Rather than log-spacing over the full [min, max] HAND range (where outliers at either end would
distort the bin layout), we use Q1 (1st percentile) and Q99 (99th percentile) of positive HAND
as the endpoints for `np.geomspace`. This provides:

- Robustness to DEM conditioning artifacts near zero HAND
- Robustness to isolated ridge pixels with anomalously high HAND
- A sentinel bin [0, Q1] catching stream-adjacent pixels
- A sentinel bin [Q99, 1e6] catching ridge extremes

For R6C10 (representative single tile), this produces approximately:

| Bin | HAND range (m) | Zone |
|-----|---------------|------|
| 1 | 0 - ~0.01 | Stream-adjacent |
| 2 | ~0.01 - ~0.05 | Near-stream saturated |
| 3 | ~0.05 - ~0.20 | TAI transition lower |
| 4 | ~0.20 - ~0.75 | TAI transition upper |
| 5 | ~0.75 - ~2.8 | Lower hillslope |
| 6 | ~2.8 - ~6.5 | Mid hillslope |
| 7 | ~6.5 - ~13 | Upper hillslope |
| 8 | ~13 - max | Ridge |

Exact boundaries depend on the HAND distribution of the input domain and will differ between
tier 1 (single tile) and tier 3 (90 tiles). The 2m constraint from Swenson (lowest bin upper
bound <= 2m) is satisfied by design -- the first 4-5 bins are all well below 2m.

## What This Gives Up

1. **Aspect-dependent insolation:** CTSM will compute insolation using the single circular-mean
   aspect value for all 8 columns. At OSBS slopes, this is a <6% effect.

2. **Direct column-by-column comparison with Swenson reference:** The reference file
   (`hillslopes_osbs_c240416.nc`) has 4 aspects x 4 bins. Comparison must use averaged profiles
   or aggregate statistics rather than bin-to-bin matching. The resolution difference (90m MERIT
   vs 1m NEON) already makes bin-to-bin comparison approximate.

3. **Equal-area bins:** With log spacing, near-stream bins have smaller area fractions than ridge
   bins. This is physically correct (the near-stream zone IS small) but means gridcell-mean
   outputs weight near-stream columns less. Per-column (h1 stream) output is unaffected.

## MERIT Validation

The MERIT regression test (`scripts/merit_validation/merit_regression.py`) validates our pysheds
fork against Swenson's published 4-aspect x 4-bin data. It intentionally retains the original
4x4 structure and equal-area binning via `compute_hand_bins()`. The new `compute_hand_bins_log()`
function is separate and does not affect the MERIT validation code path.
