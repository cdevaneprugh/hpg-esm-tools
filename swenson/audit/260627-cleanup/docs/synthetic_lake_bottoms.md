# Synthetic Lake Bottoms: Brainstorming Notes

Date: 2026-02-20

Status: Brainstorming (not implemented)

Related: STATUS.md problem #5 (DEM conditioning), PI question #1
(depression handling), full_pipeline_audit.md (synthetic lake bottoms
section)

---

## The Problem

NEON LIDAR DTM returns water surface elevation for lakes and ponds -- a
flat constant number, not the actual bathymetry. This creates large flat
regions that:

- Break `resolve_flats()` (performance bottleneck, OOM at 1m resolution)
- Fool `identify_basins()` (histogram-based detection)
- Force DEM conditioning (`fill_pits`, `fill_depressions`) to work harder
- Produce meaningless HAND/DTND values for lake pixels
- Create artificial flow paths across lake surfaces

Standard D8 flow routing requires a depression-free DEM. The current
pipeline fills depressions and resolves flats, but at 1m resolution the
large flat lake surfaces overwhelm the algorithms.

---

## The Two-DEM Problem

The pipeline uses the DEM for two fundamentally different purposes that
need different surfaces:

**Flow routing DEM** -- needs a depression-free, continuous surface. Input
to `fill_pits` -> `fill_depressions` -> `resolve_flats` -> `flowdir` ->
`accumulation` -> `compute_hand`. Its job is to produce correct D8 flow
directions, catchment delineation, and HAND/DTND. Doesn't need to be
physically real -- just needs to route water correctly.

**Physical characteristics DEM** -- supplies slope, aspect, and (indirectly
through HAND) elevation profiles for the 16 hillslope elements. Should
reflect the actual land surface as accurately as possible.

The pipeline already partially separates these: slope/aspect are computed
on the original DEM (`grid.slope_aspect("dem")`), not the conditioned one.
But HAND and DTND come from the conditioned DEM through pysheds, so lake
pixels get routed through the conditioning machinery and produce HAND/DTND
values derived from an artificial surface.

**Synthetic bottoms are about improving the flow routing DEM while leaving
the physical characteristics alone.**

---

## Synthetic Bottom Approaches

### 1. Constant offset below water surface

Drop all lake pixels by a fixed depth (e.g., 2m). Fast but doesn't create
internal gradient -- the lake floor is still flat, just lower. Doesn't help
`resolve_flats` at all.

### 2. Distance-from-shore bowl (recommended starting point)

For each lake pixel, lower it proportional to distance from the shoreline.
Deeper in the center, shallower at edges. Creates a bowl shape with real
gradients everywhere.

```
depth(pixel) = min(alpha * distance_from_shore, max_depth)
```

Where `alpha` comes from surrounding mean slope or empirical area-depth
relationship. Simplest version is linear (cone); parabolic is more
physically plausible.

Advantages:
- Simple geometry, single parameter (alpha or max_depth)
- Creates gradient everywhere -- `resolve_flats` becomes trivial
- Distance-from-shore computed via `scipy.ndimage.distance_transform_edt`
  (already used in the pipeline)

Limitations:
- Needs a max depth estimate
- Same shape for all lakes regardless of geology

### 3. Boundary interpolation (thin plate spline / kriging)

Use shoreline pixel elevations as boundary conditions, interpolate inward.
Produces a smooth extrapolation of surrounding terrain into the lake.

More sophisticated than a cone but more expensive. Can produce artifacts
if shoreline elevations aren't uniform.

### 4. Slope-derived depression

Use mean slope of surrounding terrain (within buffer of shoreline) to
project inward from all sides. If the land slopes at 0.03 m/m, continue
that slope into the lake. Produces a surface consistent with local
geomorphology.

This is essentially what Hollister et al. (2011) formalized -- see
area-depth regressions below.

---

## Area-Depth Regressions from the Literature

### Fractal scaling (Cael et al. 2017, 2022)

Theoretical framework: Earth's topography is self-affine with Hurst
exponent H ~ 0.4. Power law:

```
d_mean ~ c * A^H    where H ~ 0.4
```

Mean depth scales with area to the 0.4 power. For maximum depth, Cael
(2022) extends via fractional Brownian motion theory with fitted
area-lengthscale coefficient of 0.17.

**Limitation:** Global relationship calibrated on lakes spanning orders
of magnitude in size. The coefficient `c` absorbs regional geology.
GLOBathy found surface area alone explains only ~54% of depth variance
(NSE = 0.54). Adding watershed area, shoreline length, and surface
elevation bumps to 97% -- but requires a random forest, not a simple
formula.

For OSBS ponds (10^3 - 10^5 m^2), the power law predicts depths of
roughly 1-3m. Confidence interval is large -- individual lakes can be
5x off.

**References:**
- Cael, B.B., Heathcote, A.J., Seekell, D.A. (2017). The volume and
  mean depth of Earth's lakes. *GRL*, 44, 209-218.
  https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016GL071378
- Cael, B.B. (2022). A theory for the relationship between lake surface
  area and maximum depth. *L&O Letters*, 7, 482-489.
  https://aslopubs.onlinelibrary.wiley.com/doi/10.1002/lol2.10269

### Surrounding topography (Hollister et al. 2011)

More directly applicable:

```
depth(pixel) = distance_from_shore * median_slope_surrounding * correction_factor
```

Correction factors: 0.553 (New England), 0.462 (Mid-Atlantic).
Cross-validated RMSE of 5-6m, correlation 0.69-0.82. Explains 50-60%
of variance in maximum depth.

This is basically the "extend surrounding slope into the lake" approach.
The correction factor ~0.5 is key: actual depth is about half what naive
slope extrapolation predicts.

**Why this matters for us:** We already have the DEM slope around each
water body. A correction factor of ~0.5 (or lower for Florida's flat
terrain) with a max depth cap of 3-5m gives a physically motivated
bowl shape.

**Limitation:** Calibrated for northeastern US. Florida correction
factor unknown and likely different (shallower lakes, different geology).

**Reference:**
- Hollister, J.W., Milstead, W.B., Urrutia, M.A. (2011). Predicting
  maximum lake depth from surrounding topography. *PLOS ONE*, 6(9),
  e25764. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025764

### GLOBathy (Khazaei et al. 2022)

Global dataset of 1.4M lake bathymetries using random forest regression
on shoreline length, surface area, volume, watershed area, and surface
elevation. NSE = 0.97 for max depth.

Surface area and shoreline alone are insufficient (NSE = 0.54).

Could look up OSBS-region lakes in this dataset as a calibration check.

**Reference:**
- Khazaei, B., et al. (2022). GLOBathy, the global lakes bathymetry
  dataset. *Scientific Data*, 9, 36.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8814159/

### Florida-specific data

FWC has bathymetric surveys for selected Florida lakes:
https://geodata.myfwc.com/datasets/bathymetry-of-select-lakes-in-florida/about

USF Water Atlas has county-level bathymetric maps that might include
Alachua/Putnam county lakes near OSBS:
https://orange.wateratlas.usf.edu/shared/learnmore.asp?toolsection=lm_bathymetric

USGS Hydrology of Central Florida Lakes primer (Schiffer, 1998):
https://fl.water.usgs.gov/PDF_files/c1137_schiffer.pdf

OSBS water bodies are sinkhole-influenced, shallow (likely 1-5m), and
seasonally dynamic (large water level fluctuations).

---

## The 2m HAND Bin Threshold

### What the constraint does

Swenson's `SpecifyHandBounds` (tu:299-412) enforces: lowest HAND bin
upper bound <= 2m. This ensures the lowest hillslope column is close
enough to the stream for two-way water exchange.

In CTSM, the lateral flow code (`SoilHydrologyMod.F90:1831-1838`):

```fortran
! flow between channel and lowest column
! bankfull height is defined to be zero
head_gradient = (col%hill_elev(c)-zwt_perched(c)) &
     - max(min((stream_water_depth - stream_channel_depth),0._r8), &
     (col%hill_elev(c)-frost_table(c)))
```

`hill_elev` is mean HAND of the column. Stream bankfull is at elevation
0 by convention. The lowest column's `hill_elev` determines how high
above the stream it sits. CTSM soil columns extend ~8m deep, so a column
with `hill_elev` = 1m has its bottom at -7m relative to the stream. This
enables two-way exchange.

### Why 2m is probably fine for OSBS

**The constraint is likely never active.** It only fires when
Q25(HAND) > 2m. At OSBS with ~20-30m total relief and shallow water
tables, a large fraction of pixels will have HAND < 2m. Q25 is likely
well below 2m, which means the quartile branch fires and the constraint
is moot.

The 2m value was designed for MERIT at 90m resolution where sub-2m HAND
values are noisy. At 1m resolution, HAND values of 0.1m vs 0.5m vs 1.0m
are physically meaningful.

### The real issue: binning resolution in the TAI zone

With quartile-based equal-area binning, 25% of pixels go in each bin.
If the HAND distribution at OSBS is something like
[0, 0.5, 1.2, 3.5, 25m] at quartiles:

| Bin | Range | Zone |
|-----|-------|------|
| 1 | [0, 0.5m] | Near-stream |
| 2 | [0.5, 1.2m] | Lower TAI |
| 3 | [1.2, 3.5m] | Upper TAI |
| 4 | [3.5, 25m] | Uplands |

That's actually reasonable -- quartiles naturally give more resolution
where there are more pixels, which at OSBS is the low-HAND zone.

**But equal-area may not be ideal.** The difference between HAND=0.1m
(standing water) and HAND=0.5m (seasonally saturated) matters enormously
for carbon cycling, but they'd be in the same bin.

### Options (no CTSM code changes required)

1. **Lower the threshold** (0.5m or 1m). Only matters if Q25 > threshold.
   At OSBS, Q25 is likely below 2m anyway, so this probably doesn't fire.

2. **Use 1x8 instead of 4x4.** Gives 8 elevation bins, doubling HAND
   resolution. Combined with equal-area binning: bins like
   [0, 0.25, 0.5, 0.8, 1.2, 2.0, 3.5, 25m]. Much better TAI resolution.

3. **Non-equal-area binning.** Log-spaced or custom bins to concentrate
   resolution near HAND=0. Requires modifying `compute_hand_bins`. Changes
   column weights, which affects gridcell-mean output.

4. **Leave it alone.** The 2m constraint is inactive at OSBS. Bigger gains
   come from fixing upstream problems (UTM pysheds, flow resolution) than
   fine-tuning bin boundaries.

**Assessment:** The 2m threshold is a non-issue for OSBS because Q25 will
be below 2m. The 1x8 configuration (PI question #2 in STATUS.md) is a
much bigger lever for TAI resolution. If 1x8, the quartile logic extends
to octiles straightforwardly.

---

## Pipeline Integration Sketch

```
Original DEM
    |
    |---> identify_basins / identify_open_water ---> lake mask
    |
    |---> slope_aspect(original DEM) ---> slope, aspect [PHYSICAL]
    |
    v
Apply synthetic bottoms (lake mask + distance transform)
    |
    v
Routing DEM (synthetic bottoms applied)
    |
    v
fill_pits -> fill_depressions -> resolve_flats -> flowdir -> accumulation
    |
    v
compute_hand (uses routing DEM) ---> HAND, DTND [ROUTING-DERIVED]
    |
    v
Basin masking (audit item 9) removes lake pixels from hillslope arrays
    |
    v
Hillslope binning + parameters (on non-lake pixels only)
```

Key separation: slope and aspect come from the original surface. HAND and
DTND come from the routing DEM (with synthetic bottoms). Lake pixels are
excluded from hillslope computation by basin masking. The synthetic bottom
never enters the physical characteristics -- it only exists to make flow
routing work.

---

## 2026-02-26: Detection Fails at 1m Resolution

### Evidence

Phase D tier testing (R6C10 single tile, 5x5 block, full 90-tile contiguous
region) produced **zero detections** from both `identify_basins()` and
`identify_open_water()` across all three runs. The R6C10 tile is documented
as containing a lake and swamp — these are known water bodies, not an
absence of water.

```
Tier 1 (R6C10, 1M px):     Pre-conditioning basin pixels: 0, Open water: 0 px
Tier 2 (5x5, 25M px):      Pre-conditioning basin pixels: 0, Open water: 0 px
Tier 3 (90 tiles, 90M px):  Pre-conditioning basin pixels: 0, Open water: 0 px
```

### Why the thresholds fail

Both detection methods were designed for — or at least calibrated against —
coarser DEMs. Neither accounts for what 1m LIDAR does to water surfaces.

**`identify_basins()` (histogram, >25% threshold):** Requires a single
float elevation value to cover >25% of all pixels. At 90m, water surfaces
are smoother and fewer pixels cover a lake, so a dominant elevation can
plausibly hit 25%. At 1m, LIDAR vertical noise (~5-15 cm) means
water-surface returns are spread across many slightly different elevation
values. A lake covering 5% of the domain split across centimeter-level
variations will never spike above 25% in a histogram of exact float values.

**`identify_open_water()` (slope < 1e-4):** A slope of 1e-4 means 0.1 mm
elevation change per 1m pixel. LIDAR vertical accuracy is ~5-15 cm, so
noise alone produces slopes of 0.05-0.15 m/m — three orders of magnitude
above the threshold. Even a perfectly flat water surface will have apparent
slopes of ~0.1 m/m from measurement noise at 1m spacing. The morphological
cleanup (erode/dilate with niter=15) then removes any scattered pixels that
happen to fall below the threshold.

### Consequences for undetected lakes

When neither detection fires, lake pixels pass through the full pipeline
as ordinary terrain:

1. **DEM conditioning.** `fill_depressions` raises the lake floor to the
   pour-point elevation (often a near-no-op if the water surface is already
   at spill level). `resolve_flats` then adds micro-gradients across the
   entire flat surface to enable D8 routing. For large lakes at 1m, this
   means resolving millions of flat pixels — the documented OOM/performance
   risk.

2. **Flow routing.** D8 directions across the resolved-flat surface are
   arbitrary (driven by the micro-gradients from `resolve_flats`, not by
   real bathymetry). Catchment boundaries that pass through lakes become
   artifacts of the flat-resolution algorithm rather than reflections of
   actual drainage structure.

3. **HAND values.** If the lake has accumulation > A_thresh (likely for
   any significant water body), lake pixels are classified as stream and
   get HAND ≈ 0. These enter the lowest HAND bin and dilute near-stream
   hillslope statistics with non-hillslope land. If accumulation <
   A_thresh, lake pixels get HAND values derived from the conditioned
   (not real) surface — meaningless but mixed into the hillslope arrays.

4. **DTND values.** Same bifurcation. Stream-classified lake pixels get
   DTND ≈ 0. Non-stream lake pixels get flow-path distances across the
   resolved-flat surface, which can be artificially long as the path
   meanders before exiting.

5. **Hillslope parameters.** Without basin masking or flood filtering,
   lake pixels enter the aspect binning, HAND binning, trapezoidal width
   fit, and all six parameter calculations. The lowest HAND bin gets
   inflated with pixels that have real slope/aspect (from the original
   DEM's noisy water surface) but meaningless HAND/DTND. Area fractions
   shift toward whichever aspect bin the noise assigns lake pixels to.

### Implications

The current detection is effectively dead code at 1m. The pipeline
produces output without warnings, but any domain containing standing
water has unfiltered lake pixels in the hillslope statistics.

This reinforces that synthetic lake bottoms (or at minimum, a
1m-appropriate detection method) are not optional for production runs.
The question of _how_ to handle detected water (synthetic bottoms vs
simple masking) is secondary to the question of _whether_ water is
detected at all. Right now it isn't.

**Possible 1m-appropriate detection approaches:**

- Raise the slope threshold substantially (e.g., slope < 0.01 m/m
  with area filtering to exclude small flat patches)
- Use elevation variance in a moving window instead of slope
- Use external water body polygons (NHD, NEON aquatic boundaries)
- Classify from multispectral NEON data (not just DTM)

---

## OSBS-Specific Considerations

- **Sinkhole lakes** are karst depressions, not fluvial. Bowl geometry is
  probably correct but the depth-area relationship differs from fluvial
  lakes.
- **Seasonal dynamics** mean water level at LIDAR capture time affects
  which pixels are "lake" vs "wet ground". The lake mask depends on the
  acquisition date.
- **Closed vs through-flow basins.** Some depressions are true closed
  basins (sinkholes); others are through-flow. Synthetic bottoms treat
  them the same (bowl -> flow converges to center -> exits at lowest
  shoreline point). For closed basins this is wrong, but D8 requires an
  exit. Same fundamental tension as `fill_depressions`.
- **Interaction with conditioning.** If applied before `fill_pits` /
  `fill_depressions`, the conditioning chain still runs but has less work.
  If the synthetic bottom is deep enough, `fill_depressions` won't fill it
  back up. But surrounding low points below the synthetic edge could cause
  `fill_depressions` to partially undo the bottom.
- **Wetlands that aren't flat enough to trigger detection.** OSBS has wet
  flatwoods with slopes of 0.001-0.01 m/m. These aren't "basins" by the
  histogram criterion but still have poor drainage. Synthetic bottoms
  won't help here.

---

## Open Questions

1. **What correction factor for Florida?** Hollister's 0.46-0.55 is for
   the northeast. Florida lakes are shallower (flatter terrain, different
   geology). Could calibrate using FWC bathymetry data if any surveyed
   lakes are near OSBS.

2. **Max depth cap?** Florida lakes are typically 2-6m for non-sinkhole
   lakes. Sinkhole lakes (like Kingsley Lake) can be 25m+ but those are
   unusual. A cap of 3-5m seems reasonable for OSBS ponds.

3. **Detection method?** `identify_basins()` (elevation histogram,
   threshold 25%) vs `identify_open_water()` (slope < 1e-4 with
   morphological cleanup). **Update 2026-02-26: Both methods produce
   zero detections at 1m across all tested domains, including tiles
   with known water bodies. Neither threshold is viable at 1m. See
   "Detection Fails at 1m Resolution" section above.**

4. **Should HAND/DTND values from synthetic bottoms be used or masked?**
   Currently basin masking (item 9) removes lake pixels from hillslope
   computation entirely. The synthetic bottom only affects flow routing
   *through* the lake, which influences catchment delineation and HAND
   for lake-adjacent upland pixels. The lake pixels themselves are
   excluded.

5. **1x8 vs 4x4?** Needs PI decision (STATUS.md question #2). Affects
   how much TAI resolution we get regardless of synthetic bottoms.

6. **Ordering: synthetic bottoms before or after `identify_basins`?**
   If before, then `identify_basins` won't detect the lake (it's no
   longer flat). If after, then the detection is on the original DEM
   (correct) and the synthetic bottom is applied to a copy for routing.
   The "after" approach is clearly better -- detect on original, modify
   a routing copy.
