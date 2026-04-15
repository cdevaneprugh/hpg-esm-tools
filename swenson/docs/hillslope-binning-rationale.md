# Hillslope Binning Rationale

Date: 2026-03-19 (last updated 2026-04-14)

**Current scheme (2026-04-14):** 1 aspect × 16 HAND bins, hybrid fixed 10cm + log-spaced tail. See "2026-04-14 update" section below for rationale and permutation testing. The sections above that one describe the evolution of the binning decisions up to the prior 1×8 A2 scheme.

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

## Problem: Q1/Q99 Endpoints Land in Numerical Noise

**Date: 2026-03-30.** The percentile-based approach described above does not work as designed.

### What happened

Production run (2026-03-30, 1x8 log-spaced, NWI water masking active) produced HAND bin
boundaries:

```
[0, 0.00002, 0.00019, 0.0019, 0.018, 0.179, 1.75, 17.0, 1e6]
```

4 of 8 bins have mean height = 0.0m. This is not just wasteful — it defeats the purpose of
the 8-bin configuration.

### Why this matters in CTSM

Each bin becomes a hillslope column in CTSM with `hill_elev` set from the bin's mean HAND.
Lateral subsurface flow between columns is driven by the hydraulic head gradient
(`SoilHydrologyMod.F90` line 2261):

```fortran
head_gradient = (col%hill_elev(c) - zwt(c)) / col%hill_distance(c)
```

If 4 columns all have `hill_elev = 0.0`, there is no elevation gradient between them. They all
have the same hydraulic relationship with the stream, the same water table depth, the same soil
moisture profile, the same decomposition rate (`w_scalar`, `o_scalar`), and the same CH4
production. They are redundant columns producing copies of the same answer.

The scientific purpose of 8 bins is to resolve the 0-2m TAI zone with columns at *different*
elevations — capturing the gradient from saturated (near-stream) to unsaturated (mid-hillslope).
When bins collapse to the same height, you spend 4 of 8 columns on one effective elevation and
have only 3-4 distinct elevations across the full hillslope profile. The 8-bin configuration
offers no advantage over 4-bin equal-area in this state.

Unequal pixel counts per bin are fine — log-spaced bins *should* have fewer pixels near the
stream. The problem is that the bins must produce distinct `hill_elev` values for CTSM to
differentiate. The minimum separation between adjacent bins must exceed the DEM conditioning
noise floor.

### Per-bin results:

| Bin | HAND range (m) | Height (m) | Distance (m) | Width (m) |
|-----|---------------|-----------|--------------|-----------|
| 1 | 0 - 0.00002 | 0.0 | 2 | 569 |
| 2 | 0.00002 - 0.00019 | 0.0 | 12 | 566 |
| 3 | 0.00019 - 0.0019 | 0.0 | 26 | 549 |
| 4 | 0.0019 - 0.018 | 0.0 | 40 | 541 |
| 5 | 0.018 - 0.179 | 0.1 | 58 | 522 |
| 6 | 0.179 - 1.75 | 0.9 | 99 | 508 |
| 7 | 1.75 - 17.0 | 6.2 | 236 | 446 |
| 8 | 17.0 - 1e6 | 19.2 | 383 | 219 |

The first 4 bins span 0 to 0.018m — less than 2cm of elevation. These bins are capturing
resolve_flats micro-gradients, not real topography.

### Root cause

`resolve_flats` assigns micro-gradients (order 1e-5 to 1e-2 m) to all flat regions in the DEM
to enable D8 routing. At OSBS, flat land is pervasive — low-relief sandhills with large areas of
near-zero slope. These flat land pixels get HAND values in the 1e-5 to 1e-2 m range that are
numerical artifacts from DEM conditioning, not real elevation above the stream.

The NWI water masking (2026-03-27) removed the worst offenders — lake pixels had HAND ~ 4e-6m
from resolve_flats on filled lake beds. But the broader population of flat *land* pixels still
has micro-gradient HAND values. Q1 of positive HAND lands squarely in this noise (Q1 ~ 0.00002m).
Q5 and Q10 would be similarly small. Any percentile-based lower endpoint will be dominated by
DEM conditioning artifacts because the noise floor extends across a substantial fraction of
the domain.

The assumption in the "Percentile-Based Bin Edges" section above — that Q1 provides "robustness
to DEM conditioning artifacts near zero HAND" — was wrong. Q1 does not avoid the artifacts; it
samples them.

### Why this didn't appear in the original R6C10 test (2026-03-19)

The original 1x8 log-spaced test on R6C10 (single tile, no water masking) showed 5/8 bins at
h=0.0m. This was attributed to unmasked lake pixels (Q1 ~ 4e-6m from lake surfaces). The fix
was to defer log bins until water masking was implemented. Water masking did clean the lake
pixels, but the flat-land micro-gradient problem remained hidden because it produces slightly
larger HAND values (1e-5 to 1e-2m) that still dominate Q1.

### For comparison: 1x4 equal-area bins (same water masking)

The 2026-03-27 production run with 1x4 equal-area bins and the same water masking produced:

```
[0, 0.27, 2.30, 5.94, 25.2]
```

Equal-area bins avoid the problem because Q25 (~0.27m) is well above the noise floor. The noise
pixels are absorbed into the first bin and averaged with physically meaningful near-stream pixels.

---

## Candidate Solutions

The goal is unchanged: more bins near the stream/shoreline (low HAND) where TAI dynamics
concentrate, fewer bins on the ridge. The Q1/Q99 percentile-based geomspace fails because Q1 is
in the noise floor. Each candidate below replaces the lower endpoint strategy while preserving
log or log-like spacing.

### Solution A: Log-spaced with a minimum HAND floor

Replace Q1 with a physically meaningful minimum (e.g., 0.1m or 0.25m). Geomspace from the floor
to Q99.

```python
hand_min = 0.25  # meters — below this is DEM conditioning noise
q99 = np.percentile(hand_valid, 99)
internal = np.geomspace(hand_min, q99, n_bins - 1)
bounds = np.concatenate([[0], internal, [1e6]])
```

**Pros:** Simplest change (one line). Preserves log spacing where it matters. The lowest bin
[0, 0.25m] absorbs all noise pixels. For 8 bins with floor=0.25m and Q99~17m, boundaries would
be approximately [0, 0.25, 0.55, 1.2, 2.6, 5.6, 12.2, 17, 1e6].

**Cons:** Floor value is a tunable parameter. Must be above the noise but below physically
meaningful near-stream HAND values. The right value depends on the DEM conditioning and terrain.

**How to evaluate:** Run production pipeline, check that all bins have distinct mean heights. The
floor is too high if the lowest bin is too fat (contains too many real near-stream pixels); too
low if lowest bins still show h=0.0m.

### Solution B: Equal-count bins on restricted range

Take only HAND > floor, split into N equal-population bins.

```python
hand_above_floor = hand_valid[hand_valid > 0.25]
boundaries = np.percentile(hand_above_floor, np.linspace(0, 100, n_bins + 1))
boundaries[0] = 0
boundaries[-1] = 1e6
```

**Pros:** Concentrates bins where data density is highest — which at OSBS is the low-HAND zone.
Each bin has roughly equal pixel count, so trapezoidal fits have similar statistical power.

**Cons:** Not strictly log-spaced. The bin layout depends on the HAND distribution, which is
domain-specific. May not concentrate near-stream bins as aggressively as log spacing if the
distribution is skewed toward mid-HAND values.

**How to evaluate:** Compare bin boundaries to the log-spaced result. Check whether the 0-2m
TAI zone gets adequate resolution (3+ bins).

### Solution C: Fixed boundaries from domain knowledge

Define bin boundaries manually using OSBS-specific knowledge of the TAI zone:

```python
bounds = np.array([0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e6])
```

**Pros:** Directly encodes the scientific intent — dense resolution in the 0-2m TAI zone (4 bins),
coarser above (4 bins). No dependence on HAND distribution or percentile noise. Reproducible
across domains.

**Cons:** Not adaptive — boundaries don't respond to the actual terrain. May waste bins on empty
ranges or lump disparate terrain into one bin. Requires domain expertise to set well.

**How to evaluate:** Check pixel counts per bin. Bins with very few pixels (<1% of total) suggest
wasted resolution. Bins with >30% suggest consolidation needed.

### Solution D: Hybrid (equal-area TAI zone + log above)

Pack N_low bins into the 0-2m TAI zone using equal-area spacing, then N_high bins above 2m using
equal-area or log spacing.

```python
tai_hand = hand_valid[hand_valid <= 2.0]
ridge_hand = hand_valid[hand_valid > 2.0]
tai_bounds = np.percentile(tai_hand, np.linspace(0, 100, n_low + 1))
ridge_bounds = np.percentile(ridge_hand, np.linspace(0, 100, n_high + 1))
bounds = np.concatenate([[0], tai_bounds[1:], ridge_bounds[1:-1], [1e6]])
```

**Pros:** Directly addresses the PI's request for more bins near the waterline. The 2m boundary
aligns with Swenson's constraint (lowest bin upper bound <= 2m). Equal-area within each zone
ensures adequate pixel counts.

**Cons:** More complex. The 2m cutoff is a tunable parameter. Bin count allocation (e.g., 5 TAI +
3 ridge vs 4+4) is a design choice.

**How to evaluate:** Check that TAI bins have distinct mean heights and that ridge bins are not
too fat or too thin.

---

## Test Results

All strategies tested on the same filtered arrays (76,092,006 valid land pixels, NWI
water-masked). Comparison script: `scripts/osbs/compare_hand_binning.py`. Full results:
`output/osbs/binning_comparison/`.

### Key evaluation metric: adjacent bin height difference

The number of "bins below 2m" is a rough proxy. The real criterion is whether adjacent columns
have enough `hill_elev` separation for CTSM to compute distinct head gradients. At OSBS, the
water table fluctuates on the order of 0.1-3m, so height differences of at least 0.2m are
needed for meaningful differentiation.

### All strategies compared (2026-04-09)

| Bin | Baseline | A (0.25+Q99) | A2 (0.1+Q95) | B (equal) | C (fixed) | D (hybrid) |
|-----|----------|-------------|-------------|-----------|-----------|------------|
| 1 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 |
| 2 | 0.0 (+0.0) | 0.4 (+0.4) | 0.2 (+0.2) | 1.2 (+1.0) | 0.4 (+0.4) | 0.0 (+0.0) |
| 3 | 0.0 (+0.0) | 0.8 (+0.4) | 0.4 (+0.2) | 2.1 (+0.9) | 0.7 (+0.3) | 0.2 (+0.2) |
| 4 | 0.0 (+0.0) | 1.5 (+0.7) | 0.8 (+0.4) | 3.2 (+1.1) | 1.5 (+0.8) | 0.8 (+0.6) |
| 5 | 0.1 (+0.1) | 3.1 (+1.6) | 1.7 (+0.9) | 4.6 (+1.4) | 2.9 (+1.4) | 1.6 (+0.8) |
| 6 | 0.9 (+0.8) | 6.1 (+3.0) | 3.9 (+2.2) | 6.2 (+1.6) | 5.8 (+2.9) | 3.1 (+1.5) |
| 7 | 6.2 (+5.3) | 11.3 (+5.2) | 8.2 (+4.3) | 8.5 (+2.3) | 10.8 (+5.0) | 5.7 (+2.6) |
| 8 | 19.2 (+13.0) | 19.2 (+7.9) | 15.1 (+6.9) | 13.2 (+4.7) | 18.4 (+7.6) | 11.1 (+5.4) |

**Baseline** (Q1/Q99): 4 consecutive zero-difference transitions. Broken.

**B** (equal-count): Effectively linear — same concept as Swenson's equal-area but 1x8. Does
not concentrate resolution near the stream. Dropped from consideration.

**D** (hybrid): Two zero-difference transitions at the bottom. The equal-area TAI split still
starts from the noise floor because it uses all HAND > 0. Needs a floor to work.

**A** (0.25m floor + Q99), **C** (fixed), and **A2** (0.1m floor + Q95) are the viable options.
A and C produce nearly identical profiles. A2 gives more TAI resolution (4 bins below 2.5m vs
3 for A) and a balanced last bin (5% vs 1%).

### Recommended: A2 (0.1m floor + Q95)

```
Boundaries: [0, 0.10, 0.22, 0.50, 1.11, 2.47, 5.51, 12.28, 1e6]
```

| Bin | HAND range (m) | Height (m) | Dist (m) | Width (m) | Pixels | % | h diff |
|-----|---------------|-----------|---------|---------|--------|-----|--------|
| 1 | 0.00 - 0.10 | 0.0 | 30 | 569 | 16,838,110 | 22.1% | — |
| 2 | 0.10 - 0.22 | 0.2 | 65 | 512 | 1,720,962 | 2.3% | +0.2 |
| 3 | 0.22 - 0.50 | 0.4 | 75 | 506 | 3,433,514 | 4.5% | +0.2 |
| 4 | 0.50 - 1.11 | 0.8 | 95 | 493 | 6,472,400 | 8.5% | +0.4 |
| 5 | 1.11 - 2.47 | 1.7 | 132 | 468 | 10,806,818 | 14.2% | +0.9 |
| 6 | 2.47 - 5.51 | 3.9 | 196 | 424 | 15,995,411 | 21.0% | +2.2 |
| 7 | 5.51 - 12.28 | 8.2 | 291 | 348 | 17,026,308 | 22.4% | +4.3 |
| 8 | 12.28 - max | 15.1 | 369 | 243 | 3,798,483 | 5.0% | +6.9 |

**Why A2 over A:** More TAI resolution (4 bins below 2.5m vs 3), balanced last bin (5% vs 1%),
0.1m floor captures less real topography in the noise bin than 0.25m.

**Why A2 over C:** Adaptive to the terrain — boundaries shift with the HAND distribution rather
than being hardcoded. Similar performance but more generalizable.

### Open items for the recommended configuration

**Bin 1 is 22% of pixels at h=0.0m.** This is a fifth of the land area in a single stream-level
column. Whether this is a problem depends on the spatial distribution of those pixels. If they
cluster around streams and lake shores, the bin correctly represents the near-stream saturated
zone. If they are scattered across the domain on flat upland patches that happen to have
resolve_flats micro-gradients, the 0.1m floor may be misclassifying upland pixels as
stream-adjacent. Checking the spatial distribution of bin 1 pixels against the stream network
and NWI mask would clarify this.

**Bin 2 is thin (2.3% of pixels, 1.7M).** This is the transition from noise floor to real
topography. The trapezoidal width fit and per-bin parameter averages should be checked for
statistical noise in this bin. If the width or slope values are erratic compared to adjacent
bins, the bin may need widening (raise the floor to 0.15m or merge bins 2-3).

**The 0.1m floor is a judgment call.** A more principled approach would examine the HAND
cumulative distribution for an inflection point where noise-dominated values transition to
signal-dominated values. This was not done — the floor was chosen as a round number above the
observed noise range (1e-5 to 1e-2 m) and below the first clearly physical HAND values (~0.25m
at Q25).

**The real test is CTSM.** Binning refinements can be iterated indefinitely. The definitive
evaluation is whether CTSM produces sensible water table dynamics, carbon fluxes, and CH4
production with these parameters (Phase F).

---

## Status update (2026-04-09): Adjacent-height criterion relaxed

PI guidance after the 2026-04-09 meeting: small gradients between low-HAND bins that lead to
"standing water" in CTSM are physically correct for wetland-dominated terrain, not a bug. The
"adjacent-bin height difference >= 0.2m" criterion used to evaluate binning strategies above
no longer applies as a hard requirement.

Implications:

- The A2 configuration (0.1m floor + Q95) remains the current implementation
- Alternative strategies become viable: lower noise floor (e.g., 0.01m), or return to Q1/Q99
  endpoints (the originally "broken" approach that produced 4/8 bins near h=0.0m), or a
  no-floor configuration
- Increasing to 16 bins with focus on the 0-50cm zone is under consideration
- Specific binning strategy will be revisited alongside Phase G implementation

The comparison script (`scripts/osbs/compare_hand_binning.py`) and cached filtered arrays
(`output/osbs/binning_comparison/filtered_arrays.npz`) can test new strategies in seconds
without rerunning the full pipeline. New strategies should be added to the `STRATEGIES` dict
and the script rerun.

---

## 2026-04-14 update: moved to 16-bin hybrid

PI requested higher concentration of bins in the 0-50cm zone (the TAI transition) while keeping
the upland coverage reasonable. A2's 8-bin log-spaced scheme had 3 bins below 50cm; the new
scheme has 5 explicit 10cm bins in that zone.

### Decision

**16 HAND bins:** 5 fixed 10cm bins from 0 to 0.5m + 10 log-spaced bins from 0.5m to Q99 +
1 sentinel bin above Q99.

Final boundaries at OSBS (Q99 = 17.02m):

```
[0, 0.1, 0.2, 0.3, 0.4, 0.5,
 0.711, 1.012, 1.441, 2.050, 2.917, 4.151, 5.906, 8.404, 11.958, 17.016,
 1e6]
```

### Permutation testing

Thirteen variants were tested on the cached filtered land-pixel arrays (76M pixels) from the
03-30 production run. Dimensions varied:

- Fixed-step size: 5cm, 10cm, 15cm
- Fixed-upper: 0.3m, 0.5m, 0.6m
- Log-tail bin count: 5 to 12
- Total bin count: 14 to 18
- Uniform vs. non-uniform fixed spacing

Winning scheme (Case A with 5 fixed + 10 log = 16 bins):

| Bin | HAND range (m) | Mean HAND | Pixels | % land |
|-----|----------------|-----------|--------|--------|
| 1 | 0.00 - 0.10 | 0.02 | 16.84M | 22.13% |
| 2 | 0.10 - 0.20 | 0.15 | 1.43M | 1.87% |
| 3 | 0.20 - 0.30 | 0.25 | 1.29M | 1.69% |
| 4 | 0.30 - 0.40 | 0.35 | 1.27M | 1.67% |
| 5 | 0.40 - 0.50 | 0.45 | 1.20M | 1.58% |
| 6 | 0.50 - 0.71 | 0.60 | 2.40M | 3.15% |
| 7 | 0.71 - 1.01 | 0.86 | 3.11M | 4.09% |
| 8 | 1.01 - 1.44 | 1.22 | 3.95M | 5.20% |
| 9 | 1.44 - 2.05 | 1.74 | 4.87M | 6.39% |
| 10 | 2.05 - 2.92 | 2.47 | 5.73M | 7.53% |
| 11 | 2.92 - 4.15 | 3.52 | 6.79M | 8.93% |
| 12 | 4.15 - 5.91 | 5.00 | 8.08M | 10.62% |
| 13 | 5.91 - 8.40 | 7.06 | 8.25M | 10.84% |
| 14 | 8.40 - 11.96 | 9.95 | 6.70M | 8.81% |
| 15 | 11.96 - 17.02 | 13.83 | 3.42M | 4.50% |
| 16 | 17.02 - max | 19.17 | 0.76M | 1.00% |

### Rejected alternatives

- **5cm fixed bins**: The noise floor extends to ~0.1m; splitting 0-10cm into two 5cm bins
  puts resolve_flats artifacts (HAND 1e-5 to 1e-2 m) in bin 1 and genuinely-near-stream land
  in bin 2, mixing signal with noise. The 10cm floor cleanly absorbs the noise zone.
- **10cm × 3 fixed to 0.3m**: Loses the PI's requested 0.3-0.5m resolution.
- **Non-uniform fixed (e.g. [0, 0.05, 0.1, 0.2, 0.3, 0.5])**: Same noise-bleed issue as 5cm
  variants.
- **Q95 upper endpoint**: Q99 chosen instead for the log tail so the sentinel bin catches a
  cleaner ~1% ridge tail. With PI's "standing water is a feature" framing, the symmetry of
  Q99-only trimming (retain all low-end data, trim only extreme ridge) is preferred.

### Why Q1 is not trimmed

Q99 defines the upper endpoint of the log spacing; trimming above Q99 into the sentinel bin
is meaningful. The lower end has a fixed floor [0, 0.1m], so there is no percentile endpoint
to trim — all low-HAND pixels, including the ~1% below Q1 (0.000017m), fall into bin 1 by
design. Removing them would break area accounting without changing the binning boundaries.
Water pixels are already excluded upstream via the NWI mask.

### Function signature

`compute_hand_bins_hybrid(hand, fixed_upper=0.5, fixed_step=0.1, n_log=10, upper_percentile=99.0)`
in `scripts/hillslope_params.py`. Defaults produce the scheme above; parameters are kept for
future site-specific tuning without code changes.

---

## MERIT Validation

The MERIT regression test (`scripts/merit_validation/merit_regression.py`) validates our pysheds
fork against Swenson's published 4-aspect x 4-bin data. It intentionally retains the original
4x4 structure and equal-area binning via `compute_hand_bins()`. The new `compute_hand_bins_log()`
function is separate and does not affect the MERIT validation code path.
