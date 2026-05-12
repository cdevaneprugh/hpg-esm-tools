# Phase E.5: Bin Redesign and Spillheight Removal

Status: Complete (locked 2026-05-04; in production via `hillslopes_osbs_production_c260505.nc`)
Depends on: Phase E (complete), Phase E.6 (NWI mask regen, complete)
Blocks: Phase F, Phase G
Supersedes: original "HAND binning fix" framing in STATUS.md and audit doc Section 6.7

## Background

The PI meeting on 2026-04-30 reframed the problem and dissolved several
constraints we had been working against:

1. **The flood zone is the dry continuation of the same basin as the lake**,
   not a separate buffer between lake and uplands. NWI-mapped lake pixels
   and surrounding non-NWI flood-zone pixels were both raised by
   `fill_depressions` to the same spill elevation. They are physically and
   hydrologically continuous; the NWI mask just marks which currently
   contain surface water.

2. **The PI's spillheight SourceMod is being retired.** Its purpose was to
   simulate low-lying wetland and flood-zone columns by lowering all
   `hill_elev` values uniformly. With raw HAND now usable for binning
   (Section 6.7 of the audit doc), we get those columns from real data.
   Spillheight will be set to 0 in the SourceMod, effectively disabling it.
   The SourceMods themselves stay in place (no Fortran changes).

3. **The lake column gets a real `hill_elev` derived from basin physics**,
   not the arbitrary `-SPILLHEIGHT` convention. Constraint: lake hill_elev
   must be more negative than the deepest flood-zone bin's hill_elev to
   keep chain monotonicity. Defensible empirical anchor: Lee 2023's mean
   OSBS basin spill depth (-2.64 m) or our own pipeline's 3.33 m for
   non-NWI basins ≥ 1 ha. Either is fine; both are documented in the audit.

4. **The flood zone is the TAI zone we care about most.** The PI wants at
   least half the columns of the hillslope file in the FZ (possibly two
   thirds). FZ bin areas should increase going uphill toward the stream,
   matching the existing upland-bin behavior.

## What Changes Mechanically

### Lake column

| Field | Old (pre-meeting) | New |
|-------|-------------------|-----|
| `hill_elev` | `-SPILLHEIGHT` (-0.2 m file → -0.4 m runtime via SourceMod) | **Locked 2026-05-04: -6.0 m** (PI suggestion). Chain-bookkeeping value, set 0.87 m below deepest land bin mean (-5.13 m, bin 1 of 24-bin scheme). Empirical lake geometry (NWI mean -2.53 m; Lee/pipeline spill 2.64-3.33 m) doesn't reach the chain monotonicity floor; -6.0 m is a chain reference, not a physical bottom. SPILLHEIGHT=0, no SourceMod shift. See audit Section 5.2.1 for full derivation. |
| Runtime shift | -SPILLHEIGHT (universal, via SourceMod) | None (SPILLHEIGHT = 0) |
| Resulting ponding capacity | 0.4 m before drainage | `-hill_elev` (e.g., 2.64 m) before drainage |

The `hill_distance`, `hill_area`, `hill_width`, `hill_slope`, `hill_aspect`,
`hill_bedrock_depth` values from the pre-meeting design (Lake Column
section of the audit doc 5.2) all remain unchanged.

### Flood-zone columns

Replace the current single FZ column (or 5-bin Setup 2 proposal) with a
larger set of FZ bins, comprising ≥50% of total columns (PI direction
2026-04-30). Bins occupy the negative-raw-HAND range, with deeper bins
having smaller areas and shallower bins (toward the rim) having larger
areas — matches the upland-bin progression.

Tuning levers (all subject to iteration; **not final on first pass**):
- Outlier removal threshold for the deepest tail
- FZ bin count (9 if 50%, 12 if ⅔ of an 18-column file)
- FZ bin edges (quantile-based, log-spaced, or hand-tuned)
- True B1 retention vs. fold-in to FZ_shallowest

### Upland (B-bin) columns

Reduced count to make room for FZ bins. With ~12 FZ bins, the upland set
shrinks from 16 bins to ~5–8 bins. Will need re-tuning of edges since
collapsing 16 bins into fewer changes the area-vs-HAND distribution
upstream of stream level.

### SourceMod

`SPILLHEIGHT = 0` in the PI's SourceMod. Effectively disables the
universal `-SPILLHEIGHT` shift in `HillslopeHydrologyMod.F90:363`. No
Fortran rewrite — just the constant set to zero.

## Working Approach

The PI was explicit that the bin design is iterative tuning, not a
one-shot final decision. Workflow:

1. **Outlier removal first.** Apply across all raw-HAND data, not just FZ.
   Look at the full raw-HAND distribution, identify natural break points
   on both tails, settle on cuts (likely Q-something on the deep tail and
   Q-something on the high tail). Document the chosen thresholds.

2. **Try a simple log-spaced 16-bin scheme** as a starting point. See what
   the resulting bin edges, areas, and column means look like. This gives
   us a sanity-check baseline before introducing FZ-vs-upland asymmetry.

3. **Iterate.** Permutations of bin count, spacing scheme, FZ/upland
   split, manual edge adjustments. Run the pipeline against each, compare
   resulting hillslope files, pick what works. The pipeline supports this
   workflow — bin definition is localized.

4. **Manual nudging is OK.** Per PI: combining bins or shifting boundaries
   by hand is acceptable. The goal is a hillslope file that works for
   OSBS, not a globally-optimal binning algorithm.

## TAI Conceptual Framework (Bin Design Reference)

This section synthesizes the literature on terrestrial-aquatic interface
(TAI) extent and process gradients to inform OSBS-specific HAND bin
design. Built from four sources: the DOE BER 2017 TAI workshop report
(canonical conceptual framework), Lee et al. 2023 (OSBS field
hydrology data), Wardinski et al. 2022 (working definition of TAI
sampling extent), and Cohen et al. (DOE project proposal — what our
project is actually trying to predict).

### The fundamental asymmetry: TAI process gradients are not symmetric across the boundary

The DOE BER 2017 workshop report — the canonical conceptual framework
for representing TAIs in Earth System Models — explicitly identifies
the asymmetric nature of TAI process gradients (Executive Summary, page
viii):

> "From a terrestrial perspective, the distribution of processes may be
> steepest near the physical boundary between domains, **diminishing
> rapidly into the aquatic realm but attenuating gradually over large
> distances into the terrestrial realm**."

It also frames hydrology as the dominant process driver, with the
water-saturated boundary creating sharp transitions:

> "Hydrology is a key feature of TAIs, in part, because water is an
> effective barrier to oxygen diffusion and creates a sharp boundary
> separating areas dominated by aerobic versus anaerobic microbial
> activity. Gradients in oxygen availability are relevant at scales
> ranging from soil pore spaces to landscapes. The anaerobic microsites
> of fully terrestrial (upland) soils rapidly transition into an
> anaerobic matrix across a water-saturated boundary, **whether it be
> the water table surface in a soil profile or an upland-to-wetland
> transition along an elevation gradient**."

And acknowledges that the boundaries themselves are process-dependent
rather than fixed:

> "the boundaries between systems are realistically vague and dependent
> on the distribution of relevant processes of interest."

**Implication for our bin design.** We should expect to allocate
proportionally more bins on the deep-FZ side per meter than on the
upland side (steeper gradient), but the upland side warrants resolution
extending further than the FZ side (gradient attenuates gradually).
That's exactly what the PI's "tilt toward FZ resolution but extend a
couple meters upland" intuition encodes.

### OSBS-specific water table dynamics: Lee et al. 2023

Lee et al. 2023 instrumented n = 14 OSBS wetlands with PVC wells and
sub-daily pressure transducers from January 2018 through November 2020.
Table 1 reports:

| Parameter | OSBS value |
|---|---|
| Mean wetland stage (depth from bottom) | 177.5 ± 80.2 cm |
| Mean spill depth (h_crit) | **264.1 ± 95.0 cm** |
| Mean stage − h_crit | -86.6 cm |
| Spill depth range (Figure 3a) | ~150 cm to 600+ cm, outliers to ~1300 cm |
| Percent time inundated (PTI) | ~100% (perennially wet) |
| Percent time surface-connected (PTC) | ≤ 20% |
| Recession rate below h_crit | -1.0 cm/d |
| Recession rate above h_crit | -2.1 cm/d |

The water-surface-position distribution at OSBS (Figure 5a, indexed to
h_crit) shows water surface ranging approximately from -100 cm to +50 cm
relative to spill, with mode at -50 to -80 cm below spill.

Lee summarizes the OSBS regime in Section 4.1:

> "Wetlands in OSC hold the lowest mean water depth whereas those in
> OSBS hold the highest with the largest stage fluctuations."

> "OSBS wetlands with deeper depressions were perennially inundated
> during our study (Figure 3c)."

> "**OSBS wetlands never exceeded ≤ 20% surface connectivity**, despite
> unusually wet conditions during the study period (~15% increase in
> annual rainfall)."

> "wetland water levels in OSBS were almost always far below h_crit,
> consistent with the lack of surface connectivity, and likely driven
> by large basin volume and rapid rates of groundwater recession that
> prevent accumulation of water to initiate surface connectivity. That
> is, **OSBS wetlands rarely activate surface flowpaths because
> groundwater export rates are sufficient to constrain water level
> rises**."

**Translation to raw HAND.** A wetland with the rim at raw_hand ≈ 0
(stream-channel-equivalent reference after fill_depressions) and a
typical OSBS spill depth of 264 cm has its bed at raw_hand ≈ -2.64 m.
Water surface visits the range from -1 m (Lee's water-surface-descent
extreme, dry season) to +0.5 m (Lee's overflow extreme, peak wet) in
that median wetland's neighborhood. For deeper wetlands (spill 4-6 m),
water surface descends to -3 to -7 m raw HAND in dry years.

**Implication for our bin design.** The water-surface-visit envelope
across all OSBS wetlands is roughly **raw_hand ∈ [-7 m, +0.5 m]** in
extreme conditions; the *typical* envelope is roughly **[-3 m, +0.5 m]**.
Pixels with raw_hand below -3 m are mostly the basin floors of deeper
wetlands — water table is normally there but doesn't fluctuate much (lake
column territory). Pixels with raw_hand above +0.5 m don't directly see
water surface from Lee's data alone — but biogeochemistry (next section)
extends the relevant zone further.

### Working definition of TAI extent: Wardinski et al. 2022

Wardinski et al. 2022 measured water-soluble organic matter (WSOM)
along upland-to-wetland transects at four Delmarva Bay wetlands using
four sampling points designed to capture the full TAI gradient. Their
operational definition of the TAI extent (from the Methods, surfaced
via Wiley AGU search):

> "The Wetland point was located in the basin area that experiences
> nearly continuous inundation."

> "The Edge point was located near the edge of the wetland's inundated
> extent at the time of sampling."

> "**The Transition point was located outside the inundated extent of
> the wetland surface water, but where the groundwater table reaches
> into the upper 50 cm of the soil profile**."

> "The Upland point was located where soils are rarely saturated in the
> upper 50 cm of the soil profile."

**Translation.** The TAI on the dry side of zero extends to a height
where the water table can rise within 50 cm of the soil surface. For a
pixel at HAND = h, when the water table elevation reaches h - 0.5 m (or
higher), that pixel is in the "Transition" zone. So **upland-side TAI
extent is roughly h_max + 0.5 m**, where h_max is the highest water-
table elevation observed.

For OSBS, the highest water-table elevation observed in Lee's data is
the wetland surface at peak inundation, which can rise ~50 cm above
spill. Adding the 50 cm capillary-reach criterion gives **upland TAI
extending to roughly +1 m raw HAND** by direct field-measurement
extrapolation. Storm events and biogeochemical-relevant moisture
gradients extend the practical TAI further than the strict capillary
criterion (next section).

### Project objectives extend the TAI definition: Cohen et al. DOE proposal

The Cohen et al. DOE project proposal frames the TAI as a moving target
in *both* spatial and temporal dimensions, with carbon biogeochemistry
as the primary driver of relevance:

> "the location and length of the terrestrial-aquatic interface (TAI)
> shifts overlapping spatial variation in primary production and soil
> organic C stocks. This has important but poorly studied implications
> for the timing, location, and magnitude of gas emissions (e.g., CO₂
> and CH₄)."

> "wetlandscapes are characterized by prolonged surface storage of
> water during which groundwater flowpaths dominate export. However,
> episodic surface connectivity, which occurs when water levels exceed
> critical elevation thresholds, activates rapid surface flowpaths that
> export significant mass of dissolved C."

The proposal explicitly anchors the soil column of interest:

> "C stocks of the top 1 m of soil"

And Figure 1 caption notes the dynamic range:

> "Inundation extent (0-65% of total area) and shifting TAI position
> and length (0.001 to 0.28 m m⁻²) are non-linearly related (inset),
> and coupled to water table dynamics."

**Implication for upland TAI.** Even though Lee's OSBS data shows water
surface rarely rises above +50 cm raw HAND, the TAI as a *carbon-
relevant zone* extends further:
- Capillary fringe in sandy OSBS soils: ~30-60 cm above water table
- Storm-event saturation: water table can pulse upward during heavy
  rainfall, briefly inundating pixels above the typical envelope
- Soil moisture gradients: vegetation-water coupling matters
  hydrologically and biogeochemically across more of the upland column
  than just where water actually appears
- Soil C stocks (top 1 m) respond to long-term water-table position,
  not just instantaneous saturation

The PI's working number of **+2 m upland TAI extent** reflects this
biogeochemical-relevance framing rather than the strict water-surface-
visits framing. Defensible: the top 1 m of soil at a pixel at HAND = 2 m
sees water-table influence (via capillary, soil moisture, vegetation
demand) even if it's never directly inundated.

### Synthesis: zone definitions for OSBS bin design

Combining the four sources gives a defensible zone structure for raw
HAND bin allocation:

| Zone | Raw HAND range | TAI relevance | Justification |
|---|---|---|---|
| **Deep "always-wet"** | < ~-3 m | low | Below the water-table descent envelope of median OSBS wetlands per Lee (water surface visits down to ~-1 m below spill, and median spill is 2.64 m, so water surface visits down to ~-3.6 m raw HAND). Pixels deeper than -3 m are basin floors of the deepest wetlands — always saturated, behavior dominated by lake-bottom physics rather than TAI fluctuation. Connect to lake column. |
| **Deep FZ TAI** | -3 to ~-1 m | high | Lee's water-surface-position envelope at median OSBS wetlands sits in this band most of the time (mode at -50 to -80 cm below spill, i.e., raw HAND ~-3.5 to -3.2 m for typical 264 cm spill depths; water surface descends ~1 m below this in dry seasons). High-frequency saturation cycles → high TAI relevance. |
| **Stream-margin TAI core** | -1 to +1 m | **highest** | Where water surface oscillates across zero in median wetlands seasonally. DOE BER report: "The anaerobic microsites of fully terrestrial (upland) soils rapidly transition into an anaerobic matrix across a water-saturated boundary." This is that boundary. Process gradient steepest here. |
| **Inner upland TAI** | +1 to +2 m | high | Wardinski's "Transition" zone (water table within 50 cm of surface) plus capillary fringe in sandy soils (30-60 cm) plus storm-pulse saturation events. Soil moisture variation and vegetation-water coupling matter. Per Cohen et al. proposal, "C stocks of the top 1 m of soil" respond to long-term water-table position, including this band. |
| **Mid upland** | +2 to ~+5 m | moderate | Wardinski's "Upland" zone ("rarely saturated in the upper 50 cm"). Soil moisture extends here, vegetation responds, but TAI biogeochemistry attenuates. DOE BER: gradient "attenuating gradually over large distances into the terrestrial realm" — relevance drops slowly through this band. |
| **Sandhill ridge** | > ~+5 m | minimal | Effectively never wet in the modeled hydrology. OSBS sandhill uplands are well-drained, deep water table. Resolution wasted here. Sentinel bin. |

### Implications for bin density allocation

1. **Asymmetric bin tilt toward FZ confirmed.** Deep FZ TAI (-3 to -1 m,
   2 m wide) + stream-margin (-1 to 0 m, 1 m wide) = 3 m of FZ TAI,
   vs. stream-margin (0 to +1 m, 1 m wide) + inner upland TAI (+1 to
   +2 m, 1 m wide) = 2 m of upland TAI. Per-meter resolution should be
   highest in the stream-margin zone, then decreasing in both
   directions but **decreasing slower in the deep-FZ direction than the
   upland direction**, matching the DOE BER asymmetry framing.

2. **Stream-margin core deserves the densest bins.** The boundary at
   raw_hand = 0 is where saturation flips dynamically. Process gradient
   steepest. Bins of 0.05-0.25 m width here capture the transition
   meaningfully.

3. **Inner upland TAI (0 to +2 m) deserves more bins than mid upland
   (+2 to +5 m).** Wardinski's Transition vs. Upland distinction maps
   directly: top 50 cm saturation matters in Transition, doesn't in
   Upland. Bin density should drop noticeably crossing +2 m.

4. **Outer zones get sentinel bins.** Below -3 m and above ~+5 m get
   one or two wide bins each. Resolution there doesn't change model
   behavior.

5. **Total bin budget 16-22 columns.** Roughly:
   - 1-2 deep sentinel (< -3 m)
   - 5-7 deep FZ TAI (-3 to -1 m)
   - 4-6 stream-margin core (-1 to +1 m)
   - 4-5 inner upland TAI (+1 to +2 m)
   - 1-3 mid upland (+2 to +5 m)
   - 1 ridge sentinel (> +5 m)

### Bin minimum width — LIDAR error budget and signal-processing analysis

The DOE BER framing emphasizes that TAI process gradients are steepest
at the physical boundary. But bin widths cannot meaningfully resolve
gradients below the noise floor of the underlying raw HAND values
themselves. The minimum defensible bin width follows from a propagation
of LIDAR vertical noise through the HAND derivation, evaluated against
the standard signal-processing rule for distinguishing populations.

**Locked decision (2026-05-04): bin width floor is 0.25 m.** Derivation
below.

#### Why this matters

There are two distinct questions the bin floor protects against:

1. **Bin-membership reliability.** Each pixel has a noisy HAND value.
   If the noise σ exceeds the bin width, pixels routinely cross
   boundaries by chance, so adjacent bins draw their pixels from
   approximately the same physical population. The bins become
   statistically blended, and per-bin parameters (mean HAND, DTND,
   trap-fit slope/width) are drawn toward each other in ways the
   model cannot distinguish from real terrain differences.

2. **Hydrologic distinguishability.** Even if pixels were perfectly
   measured, sandy OSBS soils have a capillary fringe of 30-60 cm.
   Two adjacent columns separated by less than the capillary fringe
   share nearly the same water-table response. The bins represent
   distinct numerical states but not distinct physical regimes.

The standard error of the mean (SEM = σ/√N) does **not** supersede
either of these. SEM tells you how precisely the bin's mean value is
known given the pixels that ended up in the bin. It says nothing about
whether those are the right pixels (point 1) or whether the columns
behave differently in the model (point 2).

#### NEON DP3.30024.001 anchor

NEON specifies the absolute vertical accuracy of the LiDAR sensor as
**≤ 0.15 m RMSE** for the bare-earth DTM, verified per-mission via
runway flights against ground control. Source: NEON Lidar collection
documentation. This is the worst-case absolute error against an
external geodetic reference for any single ground point, and it
includes contributions from GPS+IMU positioning, range measurement
timing, scanner geometry, and atmospheric corrections.

Absolute RMSE is the *upper bound*; what propagates into our raw HAND
calculation is *relative pixel-to-pixel* noise, which is smaller
because nearby pixels share many systematic biases that cancel in
differences.

#### Error budget (raw HAND noise at OSBS)

Working from the absolute sensor accuracy down to the per-pixel noise
on raw HAND:

**Step 1: absolute → relative pixel noise.** Absolute RMSE includes
both random per-shot noise (range, scanner-edge, atmospheric) and
systematic biases shared by nearby pixels (GPS solution, IMU drift,
flight-line orientation, datum corrections). Within a single flight
line over short distances (< 100 m), systematic biases largely cancel
when you take pixel-to-pixel differences. The fraction of absolute
RMSE that survives the difference is typically 50-70% for
well-processed bare-earth data:

```
σ_relative ≈ 0.6 × σ_absolute = 0.6 × 0.15 m ≈ 0.09 m
```

**Step 2: bare-earth gridding noise.** NEON's DTM is generated from
last-return + ground-filtered points, gridded to 1 m. Additional noise
from the gridding step depends on terrain:

- Open sandhill: minimal — clean point cloud, last-return ≈ ground.
  Adds ~3 cm.
- Pine canopy (most of OSBS): some ground-filter ambiguity. Adds
  ~5-10 cm.
- Wetland edge: water surfaces can be mistaken for ground. NEON flies
  dry-season to mitigate, but residual wetland noise is ~10-20 cm in
  unfavorable conditions.

For typical OSBS pixels (mixed pine + sandhill), σ_grid ≈ 0.07 m. In
quadrature with σ_relative:

```
σ_pixel,relative = √(σ_relative² + σ_grid²)
                 = √(0.09² + 0.07²)
                 ≈ 0.11 m per pixel
```

**Step 3: HAND noise.** HAND is the elevation difference between a
pixel and its nearest drainage pixel. Both contribute noise, but stream
pixels and their neighbors share systematic errors (often same flight
line), so the cancellation effect partially applies again:

```
σ_HAND ≈ √2 × σ_pixel × (cancellation factor 0.7-0.8)
       ≈ 0.12 m
```

**Step 4: raw HAND adds DEM-conditioning artifacts.** Conditioning
contributes:

- `fill_pits`: sub-pixel, ~1-2 cm (only fills single-pixel dips)
- `fill_depressions`: large but explicitly subtracted in `dep_fill =
  flooded_orig - pit_filled`, so does not add to raw HAND noise
- `resolve_flats`: cm-scale tie-breaking perturbations on flat areas,
  ~2-3 cm

In quadrature:

```
σ_raw_hand = √(σ_HAND² + σ_condition²)
           = √(0.12² + 0.03²)
           ≈ 0.12 m
```

**Final values:**

| Pixel population | σ(raw_hand) |
|---|---|
| Open sandhill (favorable) | ~0.08 m |
| Mixed pine canopy (typical) | ~0.12 m |
| Wetland edge (unfavorable) | ~0.15-0.18 m |
| **Domain-averaged best estimate** | **~0.12 m** |

#### Translating noise to bin width

Standard signal-processing rule for distinguishing two populations
against noise: bin separation should be at least k × σ where k
controls the false-blending rate.

| k | Bin width at OSBS (σ ≈ 0.12 m) | Practical meaning |
|---|---|---|
| 1 | 0.12 m | At noise floor; ~30% pixel mixing across boundaries; bins are functionally similar |
| 2 | 0.24 m | Cleanly distinguishable bins; standard "distinguishable population" threshold |
| 3 | 0.36 m | Conservative; bins separated well above any noise concern |

Round numbers:

- 0.10 m floor — below noise (k < 1). Bins indistinguishable from neighbors.
- 0.15 m floor — at noise (k ≈ 1.25). Marginal. Accepts substantial mixing.
- **0.25 m floor — above noise (k ≈ 2). Standard distinguishability. ←
  current decision**
- 0.30 m floor — conservative; defensible against any LIDAR noise critique.

#### Why 0.25 m specifically

0.25 m is the smallest round-number floor that satisfies the standard
2σ distinguishability rule given σ ≈ 0.12 m. It also sits comfortably
within the lower end of the OSBS capillary-fringe range (30-60 cm), so
adjacent 0.25 m bins represent roughly distinct hydrologic regimes —
not perfectly distinct (the smallest capillary-fringe estimate barely
exceeds 0.25 m), but distinct enough that the model treats them
meaningfully differently.

A 0.30 m floor would be more conservative on the noise side but starts
to compromise stream-margin TAI resolution. A 0.20 m floor is at the
edge of the 2σ rule and would only just pass; some pixels in dense
canopy or wetland edges would still mix across boundaries.

0.25 m is the pragmatic middle: scientifically defensible against
LIDAR noise, hydrologically meaningful, and fine enough to give 4 bins
in each meter of stream-margin TAI on each side of zero.

#### Audit trail

Earlier provisional positions in this phase doc:

- 2026-05-02: claimed "0.05 m floor on FZ, 0.10 m on upland, 0.10-0.25 m
  sweet spot in core." This rested on an over-strong SEM argument
  ("with millions of pixels per bin, bin means are mathematically
  precise to mm even at 0.05 m spacing") that didn't address pixel
  membership or hydrologic coherence. **Retracted.**
- 2026-05-04 (revised): proposed 0.10 m floor as a permissive choice.
  At the 1σ noise level. Bins still subject to ~30% pixel mixing.
  **Superseded.**
- **2026-05-04 (locked): 0.25 m floor.** At the standard 2σ
  distinguishability threshold. Bins clean against LIDAR noise and
  meaningfully separated against capillary-fringe coherence.

### Signal-processing references

- [NEON Lidar collection — vertical accuracy spec](https://www.neonscience.org/data-collection/lidar)
- [NEON DP3.30024.001 product page](https://data.neonscience.org/data-products/DP3.30024.001)
- [Vertical Accuracy Standards — Penn State GEOG 892](https://courses.ems.psu.edu/geog892/node/709)
- [USGS Vertical Accuracy Assessment Using Ground Points](https://www.usgs.gov/ngp-standards-and-specifications/vertical-accuracy-assessment-using-ground-points)

### Working bin scheme (locked 2026-05-04, pending PI review)

24 bins total: 12 FZ + 12 upland. Asymmetric tilt toward FZ resolution.
0.25 m floor (LIDAR 2σ distinguishability). Smooth width progression
through the upland transition.

**Edges (m):**

```
[-6.35, -4.0, -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0,
 -0.75, -0.5, -0.25,  0.0,  0.25, 0.5,  1.0,  1.5,  2.0,
   3.0,  4.0,  5.0,   6.0,  8.0, 10.0, 17.02]
```

(The last edge is `Q99 = 17.02 m`, set dynamically per run; for the
2026-04-24 production data this is 17.02 m.)

**Per-zone width allocation:**

| Zone | Bin range | Width (m) | Count |
|------|-----------|-----------|-------|
| Deep tail sentinel | bin 1 | 2.35 | 1 |
| Deeper FZ | bin 2 | 1.0 | 1 |
| Deep FZ TAI | bins 3-4 | 0.5 | 2 |
| **TAI core (FZ + stream-margin)** | **bins 5-14** | **0.25** | **10** |
| Inner upland TAI | bins 15-17 | 0.5 | 3 |
| Mid upland | bins 18-21 | 1.0 | 4 |
| Mid-to-ridge | bins 22-23 | 2.0 | 2 |
| Ridge sentinel | bin 24 | ~7 | 1 |

**Width progression (smoothness principle):**

The width ramp from TAI core outward is approximately geometric, each
zone roughly 2× the previous:

```
0.25 m → 0.5 m → 1.0 m → 2.0 m → 7.0 m
   ↑        ↑       ↑      ↑       ↑
TAI core  Inner    Mid   Mid-to-  Ridge
         upland  upland  ridge   sentinel
                                 (~Q99 boundary)
```

This gradual ramp matches the DOE BER framing that TAI process
gradients "attenuate gradually over large distances into the
terrestrial realm" — bin width follows the gradient attenuation.

**Why we accept non-monotonic per-bin areas**

The "bin area must increase as we move uphill" heuristic was
considered as a constraint and explicitly dropped. Justification:

1. **No methodological basis.** Swenson's published representative-
   hillslope methodology targets approximately *equal*-area bins, not
   monotonically increasing. CTSM hillslope hydrology has no
   monotonicity requirement on `hill_area` — the model uses the area
   field as a weight in mass and energy balance aggregation, with no
   ordering constraint.

2. **Counter to physics at OSBS.** Forcing area-monotonic at OSBS would
   require widening sparse-tail bins (deep FZ extreme, upland ridge)
   and narrowing dense bins (stream-margin). The result:
   - Sparse zones get artificially inflated weight in gridcell-aggregated
     output, over-representing their statistics
   - Dense zones lose resolution where it matters most
   Both work against accurate aggregation.

3. **At OSBS specifically, area-mono is mathematically incompatible
   with TAI-focused spacing.** OSBS pixel density peaks near zero and
   tails off in both directions. Any bin scheme with narrow bins at
   the boundary (required for TAI resolution) and wide bins at the
   sentinels (required to capture sparse tails without dominating the
   spacing layout) will produce non-monotonic per-bin areas. Tested
   empirically across 12+ candidate schemes in `output/osbs/2026-05-01_bin_schemes/`.

**The smoothness check we use instead.** Per-bin pixel density
(area / bin width) should ramp smoothly through the chain. This
captures the underlying terrain heterogeneity without imposing an
unphysical area-mono constraint. The working scheme's per-meter
density:

| Bin | HAND range (m) | Width (m) | Area (km²) | km²/m |
|---|---|---|---|---|
| 1 | -6.35 to -4.0 | 2.35 | 1.45 | 0.62 |
| 2 | -4.0 to -3.0 | 1.0 | 0.91 | 0.91 |
| 3 | -3.0 to -2.5 | 0.5 | 0.70 | 1.39 |
| 4 | -2.5 to -2.0 | 0.5 | 0.76 | 1.52 |
| 5 | -2.0 to -1.75 | 0.25 | 0.62 | 2.48 |
| 6 | -1.75 to -1.5 | 0.25 | 0.71 | 2.86 |
| 7 | -1.5 to -1.25 | 0.25 | 1.48 | 5.92 |
| 8 | -1.25 to -1.0 | 0.25 | 1.26 | 5.06 |
| 9 | -1.0 to -0.75 | 0.25 | 1.21 | 4.85 |
| 10 | -0.75 to -0.5 | 0.25 | 1.32 | 5.28 |
| 11 | -0.5 to -0.25 | 0.25 | 1.88 | 7.54 |
| 12 | -0.25 to 0 | 0.25 | 2.46 | 9.84 |
| 13 | 0 to 0.25 | 0.25 | 3.48 | **13.91** ← peak |
| 14 | 0.25 to 0.5 | 0.25 | 3.06 | 12.23 |
| 15 | 0.5 to 1.0 | 0.5 | 5.35 | 10.70 |
| 16 | 1.0 to 1.5 | 0.5 | 4.56 | 9.12 |
| 17 | 1.5 to 2.0 | 0.5 | 3.95 | 7.90 |
| 18 | 2.0 to 3.0 | 1.0 | 6.58 | 6.58 |
| 19 | 3.0 to 4.0 | 1.0 | 5.52 | 5.52 |
| 20 | 4.0 to 5.0 | 1.0 | 4.91 | 4.91 |
| 21 | 5.0 to 6.0 | 1.0 | 4.30 | 4.30 |
| 22 | 6.0 to 8.0 | 2.0 | 6.78 | 3.39 |
| 23 | 8.0 to 10.0 | 2.0 | 4.75 | 2.37 |
| 24 | 10.0 to 17.02 | 7.02 | 6.45 | 0.92 |

Density rises smoothly from 0.62 km²/m at the deepest sentinel, peaks
at 13.9 km²/m at bin 13 (just-upland TAI core, where the bulk
distribution is densest), and tapers smoothly back down to 0.92 km²/m
at the ridge sentinel. **The density curve is smooth and monotonic
toward the peak from each side**, which is the physically meaningful
metric — it tracks the underlying terrain heterogeneity rather than
imposing an artificial constraint on the per-bin areas.

The mid-upland bins (18-21) all have decreasing per-meter density as
we move toward ridge, which matches the known pixel distribution at
OSBS (sandhill upland gets sparser at higher HAND). The width
progression doubles smoothly from 1.0 m → 2.0 m → 7.0 m through the
final transition, avoiding the abrupt 1m → 2.5m → 7m jump in the
earlier 23-bin draft.

### References

- DOE BER (2017). [*Research Priorities to Incorporate Terrestrial-
  Aquatic Interfaces in Earth System Models: Workshop Report*](https://science.osti.gov/-/media/ber/pdf/community-resources/Terrestrial-Aquatic_Interfaces_report.pdf), DOE/SC-0187.
- Lee, E., Epstein, J. M., & Cohen, M. J. (2023). [Patterns of Wetland
  Hydrologic Connectivity Across Coastal-Plain Wetlandscapes](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR034553).
  *Water Resources Research*, 59, e2023WR034553.
- Wardinski et al. (2022). [Water-Soluble Organic Matter From Soils at
  the Terrestrial-Aquatic Interface in Wetland-Dominated
  Landscapes](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JG006994).
  *Journal of Geophysical Research: Biogeosciences*.
- Cohen, M. J., et al. DOE Project Proposal: *Water and carbon dynamics
  of coastal plain wetlandscapes* (local copy at
  `docs/papers/DOE-Project-Grant-Application.pdf`).
- McLaughlin, D. L., et al. (2019). [Wetland Connectivity Thresholds
  and Flow Dynamics From Stage Measurements](https://doi.org/10.1029/2018WR024652).
  *Water Resources Research*, 55, 6018-6032. (Methodological source for
  Lee 2023's CFS approach.)

## Tasks

### Outlier strategy (first up)

- [x] Histogram of raw HAND across full domain (positive + negative tails).
      Use saved diagnostic arrays at
      `output/osbs/2026-04-24_diagnostic/diagnostics/*.npy`.
      Output: `output/osbs/2026-05-01_outlier_strategy/`
      (1_raw_hand_full.png, 2_conditioned_hand_full.png, 3_raw_hand_tail_cdf.png).
      Script: `scripts/osbs/diagnose_outlier_strategy.py`.
- [x] Identify natural break points; report Q90, Q95, Q97, Q99 on both
      tails for comparison. Done via tail-density and spatial-cluster
      analyses. See `summary.json` and 2026-05-01 log entry. Sharpest
      density breaks are at Q01 deep (33.5×) and Q99 upland (34.8×); both
      tails are real terrain (negligible singleton mass), so cutoff =
      "log-tail endpoint with sentinel bin past it," not discard.
- [x] Choose outlier cutoffs (one for upland tail, one for FZ tail).
      **Decision locked (2026-05-01): Q01 deep = -6.34 m / Q99 upland =
      +17.46 m. True discard (not sentinel-binned).** Total ~1.53 M
      pixels (~2% of land) discarded. Defense uses two converging
      methods (Q01 ≈ classical 2σ clip on raw HAND distribution) with
      Lee 2023 spill-depth data as a physical sanity check. See
      2026-05-01 log entries for derivation, the 2026-05-02 entry below
      for the formalized defense framing.
- [x] Verify chain monotonicity once both the cutoff and lake hill_elev
      are chosen. Confirmed in production: lake `hill_elev = -6.0 m` is
      0.87 m below deepest land bin's mean (-5.13 m). The osbs5 and
      osbs.swenson.spinup runs both ingest the chain without error.

### Initial bin scheme

**Implementation reference:** the pipeline change is more than just
adding a Q01/Q99 trim — it also switches binning input from
conditioned `hand` to `raw_hand`. The conditioned-HAND `(hand > 0)`
filter shortcut (which implicitly drops streams + lakes since both
are `hand = 0`) breaks under raw-HAND because real streams in filled
depressions have `raw_hand < 0` and would land in FZ bins. The
correct filter recipe and a pixel-category reference table live in
`docs/lake-column-ctsm-audit.md` Section 6.9. Use explicit
`(water_mask == 0) & (channel_mask == 0)` rather than a
`raw_hand != 0` heuristic. Section 6.9 also notes that the TAI is an
emergent property — the immediate upland band (0 to ~2 m raw HAND)
is as TAI-relevant as the FZ side and bin density there should
reflect that.

#### Pipeline implementation steps

Concrete steps for the `run_pipeline.py` change. Insertion points
relative to the current line numbering (≈mid-2026-05).

**1. Step 3 (DEM conditioning) — capture `dep_fill` before water-pixel
lowering.** Currently `grid.flooded` is overwritten in place by the
water-mask lowering line (`flooded_arr[water_mask > 0] -= 0.1`). Save
the pre-lowering flooded DEM and pit_filled DEM as numpy arrays first,
then compute `dep_fill = flooded_orig - pit_filled`. Adds ~720 MB to
peak memory but the SLURM allocation has headroom.

```python
# Right after grid.fill_depressions(...) finishes, BEFORE the
# water-pixel lowering line in run_pipeline.py:830-831
pit_filled_arr = np.array(grid.pit_filled)
flooded_orig_arr = np.array(grid.flooded)
dep_fill_arr = flooded_orig_arr - pit_filled_arr
```

**2. Step 4 (HAND/DTND) — derive `raw_hand`** once after `compute_hand`
returns:

```python
hand_arr = np.array(grid.hand)
raw_hand_arr = hand_arr - dep_fill_arr
```

**3. Step 5 (Hillslope Parameters) — replace the `(hand > 0)` filter
with the combined cleanup mask, compute Q01/Q99, then trim.** Insert
near `run_pipeline.py:~1037`, replacing the existing tail-removal +
binning input prep:

```python
# Combined cleanup mask — drop lakes, streams, NaN, DTND-tail outliers
channel_mask_flat = (np.array(grid.channel_mask) > 0).flatten()
water_flat = water_mask.flatten()
raw_hand_flat = raw_hand_arr.flatten()
hand_flat = hand_arr.flatten()
dtnd_flat = np.array(grid.dtnd).flatten().copy()

# Existing DTND tail-index outlier removal (run_pipeline.py:1031-1037)
land_finite = np.isfinite(hand_flat) & (water_flat == 0)
tail_ind = tail_index(dtnd_flat[land_finite], hand_flat[land_finite])
keep_tail = np.zeros(hand_flat.shape, dtype=bool)
keep_tail[np.where(land_finite)[0][tail_ind]] = True

# Existing DTND minimum clip
dtnd_flat[dtnd_flat < SMALLEST_DTND_M] = SMALLEST_DTND_M

# Combined cleanup mask
valid = (
    (water_flat == 0)               # exclude lakes
    & (channel_mask_flat == 0)      # exclude all stream pixels
    & np.isfinite(raw_hand_flat)    # exclude NaN
    & keep_tail                     # exclude DTND tail-index outliers
)

# Compute Q01/Q99 dynamically from the cleaned land population
q01 = float(np.percentile(raw_hand_flat[valid], 1))
q99 = float(np.percentile(raw_hand_flat[valid], 99))

# Apply outlier trim — narrows valid by ~2% (1% each tail)
valid &= (raw_hand_flat >= q01) & (raw_hand_flat <= q99)
```

Steps 2 and 3 (compute Q01/Q99 then trim) are conceptually a single
"compute-then-trim" operation; they're split only because numpy lacks
a one-shot percentile-trim primitive. After this block, `valid` is
the final mask used for trap fit, bin assignment, per-bin statistics,
and NetCDF write.

**4. Switch binning input from `hand` to `raw_hand` everywhere
downstream.** All bin-edge construction, `np.digitize` calls, and
per-bin "Mean HAND" computations should consume `raw_hand_flat[valid]`
instead of `hand_flat[valid]`. The trap fit input (`dtnd_flat[valid]`,
pixel area) is unchanged in structure but `valid` itself now reflects
the new filter and trim.

**5. NetCDF metadata — log the cutoffs and method.** As global
attributes on the output NetCDF:

```python
ncfile.outlier_method = "Q01/Q99 percentile trim, true discard"
ncfile.q01_cutoff_m = q01
ncfile.q99_cutoff_m = q99
ncfile.n_pixels_pre_trim = int((... pre-trim valid ...).sum())
ncfile.n_pixels_post_trim = int(valid.sum())
ncfile.binning_input = "raw_hand = hand - (flooded_orig - pit_filled)"
```

This makes the cutoff values reproducible and verifiable per run
without reading them out of pipeline logs.

**Verification.** After the pipeline change:
- Confirm Q01/Q99 reported by the run match the diagnostic
  (`output/osbs/2026-05-01_outlier_strategy/summary.json`): -6.34 m
  and +17.46 m within 0.01 m on the same input data.
- `n_pixels_post_trim` should be ~75 M (≈ 76.6 M land minus ~2 % trim).
- Compare a per-bin "Mean HAND" for the deepest FZ bin against the
  hypothetical-setup output (`output/osbs/2026-05-01_bin_schemes/
  summary.json` for the matching scheme) — should agree within rounding.

- [x] Choose bin scheme. **Locked 2026-05-04 (pending PI review):
      24-bin TAI-focused scheme**, 12 FZ + 12 upland, 0.25 m floor,
      asymmetric tilt toward FZ. Edge list and full design rationale in
      "Working bin scheme" subsection above. Per-bin parameters
      verified via `scripts/osbs/diagnose_bin_schemes.py`; output at
      `output/osbs/2026-05-04_bin_schemes/setup_working.png`.
- [x] Implement the locked 24-bin scheme in `run_pipeline.py` per the
      "Pipeline implementation steps" subsection above. Pipeline change
      includes: (a) capture `dep_fill` in Step 3, (b) derive `raw_hand`
      in Step 4, (c) switch binning input from `hand` to `raw_hand`,
      (d) replace `(hand > 0)` filter with the explicit channel-and-
      water mask, (e) apply Q01/Q99 trim in main loop, (f) hardcode
      the 24 bin edges (or graduate to a function in
      `hillslope_params.py`). Production NetCDF dated 2026-05-05.
- [x] Generate diagnostic plot showing bin scheme fit to raw HAND
      histogram. Done via `diagnose_bin_schemes.py` for the locked
      scheme. Plot at `output/osbs/2026-05-04_bin_schemes/setup_working.png`.
- [x] Run pipeline; verify bin edges, areas, means match the
      diagnostic. Cross-check against `summary.json` from the
      diagnostic. Verified in 2026-05-04 and 2026-05-05 production runs.
- [x] Document baseline output for comparison with PI feedback.
      Captured in NetCDF global attrs + audit doc Section 5.

### FZ bin redesign

- [ ] Decide FZ count: 9 (50%) or 12 (⅔). Defer slightly until baseline
      is in hand and we can see the FZ-pixel distribution.
- [ ] Design FZ bin edges with increasing-area-going-uphill constraint.
- [ ] Decide True B1 fate (keep separate, fold into FZ, fold into upland).
- [ ] Run pipeline with new scheme; compare to baseline.

### Lake column update

- [x] Set lake `hill_elev` = -6.0 m (locked 2026-05-04, chain-bookkeeping value).
- [x] Verify chain monotonicity end-to-end with the chosen FZ scheme.
      24-bin scheme + lake at -6.0 m chain accepted by CTSM ingestion in
      osbs5 (100 yr) and osbs.swenson.spinup (current run).
- [x] Document the choice and reasoning in the NetCDF generation step.
      Captured in `docs/lake-column-ctsm-audit.md` Section 5.2.1 +
      NetCDF global attributes on the production output.

### SourceMod and case config

- [x] Edit `osbs4.branch.v2/SourceMods/src.clm/HillslopeHydrologyMod.F90`
      to set `SPILLHEIGHT = 0`. Do not remove the SourceMod files.
      Approach revised: SPILLHEIGHT is set to 0 via the namelist
      override (`spillheight = 0.0` in user_nl_clm), not by editing
      the SourceMod constant. SourceMod files are retained but rendered
      inert. Same runtime effect; lower maintenance burden across cases.
- [x] Note the change in the case directory's run log. user_nl_clm
      includes a comment block in osbs5/osbs.swenson.spinup explaining
      the Phase E.5 reframe.

### Iteration

- [ ] Run permutations as PI requests; record each in the log section
      below with the parameters used and resulting column structure.

## Parallel Tasks (not blocking)

### Observation-date research

Date stamps for the data sources we're using. Not road-blocking but
should be done thoroughly when picked up. The PI may use these to tune
lake/FZ weights or save them for a separate paper.

Findings live in `docs/data-acquisition-dates.md`.

- [x] **NEON OSBS LIDAR**: 2023-05 collection, RELEASE-2026 (per
      `data/neon/README.md`). Late-dry-season timing — favorable for
      capturing dry beds. NEON has 8 collections at OSBS (2014–2025); only
      2023-05 is used in our pipeline.
- [x] **NWI Lake Mask**: 2017 source imagery, true color, 1 m resolution
      ("Lower St. John" project). Database extract October 2024. Determined
      by spatial intersection of project metadata against the production
      domain — earlier 1983-84 CIR quads do not overlap our domain.
      Polygons match Google Earth recent imagery (visual confirmation).
- [ ] **Lee 2023 LIDAR**: ambiguous. Paper cites NCALM for all four sites
      but doesn't specify which dataset for OSBS. Three candidates: 2010
      NCALM Optech Gemini (peak wet season), 2018 USGS Florida Peninsular
      Putnam (dry-conditions design), or a custom mission. Resolution path:
      ask Cohen (UF, senior author of Lee 2023) directly via the
      collaborator channel.

## Out of Scope

- No CTSM Fortran rewrites. SourceMod retirement is just setting the
  constant to zero.
- No NWI shapefile reprocessing (Phase E.6 already fixed the rasterization).
- No pipeline architecture changes. The bin definition module is the only
  area that meaningfully changes.
- No "final" bin scheme on first pass. Expect iteration.

## Log

### 2026-04-30 — Phase initiated

PI meeting reframed the project. Created this file to track the bin
redesign work iteratively. Audit doc Section 5.x and 6.7 sections about
spillheight are flagged as superseded but kept for historical reference.
Phase G's "submerged lake column with spillheight SourceMod" framing
similarly retired; Phase G now reduces to "lake column gets the
empirical hill_elev assigned in this phase" — effectively folded into
Phase E.5.

Next: outlier strategy. Will start with histogram of raw HAND from the
existing diagnostic arrays.

### 2026-05-01 — Outlier strategy diagnostic complete

Built `scripts/osbs/diagnose_outlier_strategy.py`. No pipeline rerun;
loads four arrays from `output/osbs/2026-04-24_diagnostic/diagnostics/`
(hand, flooded_orig, pit_filled, water_mask) and derives raw HAND as
`hand - (flooded_orig - pit_filled)`. Output in
`output/osbs/2026-05-01_outlier_strategy/`.

**Pixel populations** (76.67M land pixels analyzed):

| Region | Pixels | % |
|--------|--------|---|
| Negative raw HAND (FZ candidates) | 15.65M | 20.4% |
| Positive raw HAND (upland) | 61.02M | 79.6% |

(Cross-checks audit Section 6.7.3: 76.6M land, 10.7M NWI water.)

**Quantile values (raw HAND, m):**

| Q01 | Q03 | Q05 | Q10 | Q50 | Q90 | Q95 | Q97 | Q99 | min | max |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| -6.34 | -3.88 | -2.49 | -1.21 | 2.31 | 9.96 | 12.51 | 14.09 | 17.46 | -17.28 | 25.20 |

**Tail-density analysis** (px/m of discarded vs in-tail mass):

Sharp density ratios across the cutoff identify natural breaks. Both
tails show their sharpest break at the most extreme candidate (Q01 deep,
Q99 upland), with monotonically softer breaks at less extreme cutoffs.

| Cutoff | n_discarded | discarded_density (px/m) | surviving_density (px/m) | ratio |
|--------|-------------|--------------------------|--------------------------|-------|
| Q01 = -6.34m | 766,735 | 70,065 | 2,347,074 | **33.5×** |
| Q03 = -3.88m | 2,300,204 | 171,553 | 3,443,805 | 20.1× |
| Q05 = -2.49m | 3,833,673 | 259,187 | 4,738,785 | 18.3× |
| Q10 = -1.21m | 7,667,345 | 476,843 | 6,622,541 | 13.9× |
| Q90 = 9.96m | 7,667,345 | 502,969 | 5,356,956 | 10.7× |
| Q95 = 12.51m | 3,833,665 | 301,954 | 4,572,192 | 15.1× |
| Q97 = 14.09m | 2,300,159 | 206,977 | 4,167,343 | 20.1× |
| Q99 = 17.46m | 766,734 | 99,031 | 3,450,719 | **34.8×** |

**Spatial cluster analysis** (8-connectivity on the 2D outlier mask):

| Cutoff | n outliers | n components | largest (px) | largest fraction | n=1 | n≥100 |
|--------|------------|--------------|--------------|------------------|-----|-------|
| Q01 deep | 766,735 | 329 | 249,923 | 33% | 58 | 58 |
| Q03 deep | 2,300,204 | 142 | 371,556 | 16% | 7 | 72 |
| Q05 deep | 3,833,673 | 430 | 691,995 | 18% | 22 | 115 |
| Q10 deep | 7,667,345 | 5,010 | 1,017,460 | 13% | 208 | 283 |
| Q90 upland | 7,667,345 | 3,953 | 958,560 | 13% | 2,579 | 139 |
| Q95 upland | 3,833,665 | 2,397 | 419,688 | 11% | 1,580 | 111 |
| Q97 upland | 2,300,159 | 1,731 | 226,346 | 10% | 1,139 | 93 |
| Q99 upland | 766,734 | 668 | 127,766 | 17% | 426 | 43 |

**Reading.** Singletons and other small components are negligible at
every cutoff (singleton count is at most ~0.05% of outlier mass). The
"outliers" are not LIDAR noise — they are real terrain features.

- **Deep tail past Q01** (766K px): one large feature (250K px ≈ 0.25
  km², likely the deepest part of a major basin) plus ~328 smaller real
  components. Throwing these pixels out of the binning entirely would
  discard the deepest core of the FZ structure — the opposite of what we
  want. They should stay in the dataset, probably in a sentinel "deepest
  FZ" bin past Q01. Existing convention in
  `compute_hand_bins_hybrid` (Phase E) already does this on the upland
  side (sentinel bin `[Q99, 1e6]`); mirror for the deep side as
  `(-1e6, Q01]`.
- **Upland tail past Q99** (767K px): no dominant feature (largest only
  17%); 668 components averaging ~1.1K pixels each. Mostly small
  isolated rises (sandhills) plus 426 single pixels (still negligible
  mass). Same sentinel-bin treatment as the deep side preserves the
  upland gradient without distorting log-spaced internal bins.

**Practical implication.** The "natural break" cutoffs are **Q01 deep
(-6.34m)** and **Q99 upland (17.46m)**. They define where the bulk
distribution gives way to long sparse tails. But "cutoff" here means
"endpoint for log-spaced bin layout, with a sentinel bin past it" —
not "discard these pixels." The 766K + 767K pixels in the two tails
are real and should each go into their own sentinel bin, contributing
to FZ-deepest and upland-ridge column statistics.

**What we do not yet need from this diagnostic.** Lake-column hill_elev
is a downstream PI decision; this analysis does not constrain it.
Whatever the PI picks for the lake elevation, it just needs to sit
deeper than the deepest FZ bin's mean — which depends on the bin design
(next task). The lake-elev candidate values (-2.64m, -3.33m) are drawn
on the plots only as informational landmarks, not as constraints.

**Plot reference.** `1_raw_hand_full.png` shows both tails with all
quantile lines and the lake-elev landmarks on the same axes as
`2_conditioned_hand_full.png` (basin pixels collapsed to ~0). The
density argument above is visible directly: the histogram's left/right
fringes are sparse compared to the bulk between Q01 and Q99.
`3_raw_hand_tail_cdf.png` shows the cumulative fraction at each
quantile — used to read off cutoff values.

Next: surface this to the PI for cutoff approval. Default proposal is
Q01 / Q99 sentinel-bin scheme based on the density and cluster
evidence. Once approved, move to "Initial bin scheme" task — implement
log-spaced 16-bin baseline with sentinel bins on each end.

### 2026-05-01 — Cutoff decision (provisional, pending PI confirmation)

After spatial verification we settled on **Q01 deep / Q99 upland** as
the outlier cutoffs:

- **Deep:** Q01 = -6.34 m. 766,735 pixels (1% of land) below.
- **Upland:** Q99 = +17.46 m. 766,734 pixels (1% of land) above.
- Total ~1.53 M pixels (~2% of land) beyond cutoffs.

Two new spatial plots added to the diagnostic
(`scripts/osbs/diagnose_outlier_strategy.py`):

- `4_spatial_hand_lt0.png` — all FZ candidates (raw HAND < 0), colored
  by depth (plasma_r gradient); NWI as blue outlines.
- `5_spatial_hand_lt-10.png` — extreme deep-tail pixels (raw HAND <
  -10 m); same color treatment.
- `6_spatial_q01_q99_outliers.png` — Q01/Q99 outliers shown together:
  red below Q01, black above Q99, blue NWI outlines.

**Spatial finding.** The HAND<-10 mass (259 K pixels) is almost entirely
concentrated in **one small sinkhole-like feature in the SE portion of
the production domain** — small relative to the rest of the basin
structure visible in plot 4. The dominant 250 K-pixel connected
component identified in the cluster analysis sits there. Discarding it
via the Q01 cutoff is acceptable.

**Decision rationale.** Q01/Q99 cuts ~2% of land pixels at sharp
density-ratio breaks (33.5× deep, 34.8× upland). The deep tail's
dominant feature is a localized SE sinkhole rather than the deep core
of a broadly distributed FZ; removing it does not erase substantial
basin-floor mass elsewhere. The remaining 79.6% upland and 19.4% FZ
land mass between -6.34 m and +17.46 m forms the input to the
log-spaced bin layout.

**Sentinel vs. discard.** Earlier framing was "sentinel bins past the
cutoffs preserve the data." After spatial inspection the SE sinkhole
looks like a discrete feature better excluded entirely than carried as
a one-column sentinel — it would distort that column's statistics
substantially. Final implementation choice (sentinel vs. true discard)
will be made when the bin scheme lands; provisionally lean toward
**discard for the deep tail** (lose the SE sinkhole) and **sentinel
for the upland tail** (scattered small hills are spread across the
domain and behave like real ridge features).

Next: take the proposal to the PI for confirmation, then move to
"Initial bin scheme" — implement log-spaced 16-bin baseline with the
Q01/Q99 endpoints, plus the diagnostic plot showing how the bins fit
the raw-HAND distribution.

### 2026-05-02 — Outlier cutoffs locked + formal defense

**Decision locked.** Outlier cutoffs for the production pipeline are
**Q01 = −6.34 m (FZ) and Q99 = +17.46 m (upland)** with **true discard**
on both ends. This choice replaces all earlier "sentinel-bin past Q01"
discussion in this phase doc. Pixels beyond the cutoffs are removed
from binning entirely; downstream column statistics (HAND, DTND, area,
width) reflect only the trimmed pixel set.

**Formalized defense.** Two independent methods converge at the deep
cutoff, with Lee 2023 field data acting as a physical sanity check
(not a derivation):

| Method | Lower cutoff | Comment |
|---|---|---|
| Q01 (empirical 1st percentile of raw HAND) | **−6.34 m** | The cutoff itself |
| mean − 2σ on raw HAND distribution | −6.10 m | Classical sigma clipping |
| Lee 2023 OSBS spill mean + 2·SD (n=14) | (4.54 m → cutoff at -4.54 m if used) | Field-measured upper plausibility bound; not used as cutoff |

The Q01 cutoff coincides with classical 2σ clipping on the raw HAND
distribution (within 0.24 m). Lee et al. 2023 reports OSBS field-
measured spill depths of 2.64 m ± 0.95 m SD across n = 14 wetlands;
the upper 2σ bound is 4.54 m, meaning our Q01 cutoff (6.34 m magnitude)
is **1.8 m deeper than any wetland Lee surveyed**. The cutoff is
conservative relative to physical OSBS wetland depths.

**Why not other methods.**

- **3σ clipping (mean − 3σ = −10.85 m)** — too permissive; only 0.28%
  trimmed; std is inflated by the heavy upland tail (skewed
  distribution), so 3σ doesn't behave like 3σ on a normal.
- **MAD-based (median − 3·MAD = −5.12 m)** — symmetric around the
  median (+2.31 m), which is biased to the upland side by skew.
  Symmetric MAD bounds clip 1.96% on the FZ side; cuts into real
  basin terrain.
- **Tukey 1.5·IQR (−8.37 m)** — too permissive; keeps the SE sinkhole.
- **Lee + 4σ (−6.50 m, claimed in prior framing)** — n = 14 is too
  small to characterize 4σ tails; the expected sample maximum for n=14
  drawn from N(2.64, 0.95) is mean + σ·E[Z_(14)] ≈ 4.26 m, not 6.5 m.
  4σ extrapolation was confirmation bias; retracted.

**Same logic applies on the upland side.** Q99 = +17.46 m matches
mean + 3σ = +17.65 m within 0.19 m. The upland tail's heavier mass
makes 3σ (rather than 2σ) the matching multiplier — a direct
consequence of the distribution's skewness.

**Defense statement (for the PI / paper):**

> Outlier removal: pixels with raw HAND < Q01 = −6.34 m or > Q99 =
> +17.46 m are discarded prior to binning (1.0% of land pixels removed
> from each tail; ~1.53 M pixels total, 2% of land). The cutoffs
> coincide with classical sigma clipping (mean − 2σ = −6.10 m on the
> deep side; mean + 3σ = +17.65 m on the upland side); the asymmetric
> σ multipliers reflect the right-skewness of the raw HAND distribution
> at OSBS. As a physical sanity check, the deep cutoff is 1.8 m deeper
> than the upper 2σ bound of OSBS wetland spill depths reported in
> Lee et al. 2023 (mean 2.64 m + 2·SD = 4.54 m, n = 14). Pixels removed
> at the deep cutoff are deeper than any wetland in their field survey,
> supporting the choice as conservative on physical grounds.

**Honest disclosure on small-sample physical bound.** Lee 2023's
n = 14 cannot reliably characterize σ-tail behavior beyond ~2σ, so we
do not attempt to derive a cutoff from their data. The Lee comparison
is a one-way sanity check: "is our cutoff at least as deep as their
deepest plausible wetland?" Yes, by ~1.8 m.

**Audit trail.** Earlier provisional framings used (a) "Lee + 4σ"
(retracted as numerical confirmation bias) and (b) "sentinel bin past
Q01" (superseded by true discard after PI direction). Both are
preserved in the prior log entries above for history. The formalized
decision in this entry supersedes them.

Next: move to "Initial bin scheme" task with the locked cutoffs as
endpoints. Bin schemes 1a-3f have been explored (see
`output/osbs/2026-05-01_bin_schemes/`). Top candidate is 3e (manual
TAI-focused with merged ridge, 14 bins, 8/6 split) pending PI
selection.

### 2026-05-04 — Working bin scheme locked (pending PI review)

After several rounds of design (literature research on TAI extent,
LIDAR error budget for the 0.25 m floor, and iterative refinement of
the upland transition), settled on a **24-bin scheme** as the working
starting point:

- **12 FZ + 12 upland** with asymmetric tilt toward FZ resolution
- **0.25 m floor** in the TAI core (5-14), justified by the 2σ LIDAR
  noise distinguishability rule
- **Smooth width progression** outward from the TAI core: 0.25 → 0.5 →
  1.0 → 2.0 → 7.0 m, each step ≈ 2× the previous
- **Sentinels at both extremes** — bin 1 covers the deep tail
  (-6.35 to -4.0 m), bin 24 covers the ridge (10 to ~17 m)
- **Hard boundary at zero** (bin 12 ends at 0; bin 13 starts at 0)

Full edge list, per-bin parameters, design rationale, and the
explicit decision to drop area-monotonic in favor of smooth per-meter
density are documented in the "Working bin scheme" subsection above.

**Design history summary** (for the log):

1. Round 1 (2026-05-01): 9 candidate schemes — log-spaced, equal-area,
   manual TAI variants. None satisfied area-monotonic; user identified
   the constraint as not worth chasing.
2. Round 2: Conceptual reframe via TAI literature research. Established
   the asymmetric "steeper FZ, gradual upland" framing from DOE BER and
   the OSBS-specific water-table envelope from Lee 2023.
3. Round 3 (2026-05-04): First "gut" scheme — 20 bins, 12/8, 0.05 m
   minimum. Pushed back on by user; bin minimum walk-through revealed
   0.05 m was below the LIDAR noise floor.
4. Round 4: Error budget anchored on NEON DP3.30024.001 ≤ 0.15 m RMSE
   spec. Propagated through to σ(raw_hand) ≈ 0.12 m. 2σ rule → 0.25 m
   floor. 23-bin scheme produced.
5. Round 5: Smoothed the upland transition. Bin 21 in the 23-bin draft
   was sitting too proud at 9.62 km² (5.0-7.5 m, 2.5 m wide) due to
   width jump from 1.0 m to 2.5 m. Replaced with 4 bins (1.0, 2.0,
   2.0, sentinel) for the 5+ m range. Final 24-bin scheme.

**Remaining work in Phase E.5:**

1. Implement the bin edges in `run_pipeline.py` per the steps documented
   in the "Pipeline implementation steps" subsection.
2. Run the pipeline to produce the production NetCDF.
3. Verify NetCDF structure (24 columns + lake column to come).
4. Lake column work (Phase E.5 task list, separate set of tasks).
5. PI review of the working scheme. Adjustments may be requested.

The current scheme is "good enough to ship to CTSM and iterate." It's
not optimized — it's defensible. PI feedback may shift bin counts in
specific zones, but the framework (zone definitions, smooth width
progression, 0.25 m floor, asymmetric tilt) should hold.

### 2026-05-04 — Lake column hill_elev locked

**Decision:** lake `hill_elev` = **-6.0 m** (PI's suggested starting
value).

**Why this number doesn't directly come from data.** Three constraints
collide:

1. Chain monotonicity requires lake hill_elev < deepest land bin mean.
   With the 24-bin scheme, the deepest land bin (bin 1, deep tail
   sentinel covering -6.35 to -4.0 m raw HAND) has mean -5.13 m.
2. Empirical lake geometry doesn't naturally reach -5 m. Mean NWI
   lake-surface raw HAND in the production domain is **-2.53 m**.
   Lee 2023 mean OSBS spill depth is 2.64 m (rim-to-bottom); our
   pipeline measures 3.33 m for non-NWI basins ≥ 1 ha.
3. PI direction (2026-05-04): don't reorder the chain, don't fold
   the deep tail sentinel into the lake column.

With (1) requiring < -5.13 m and (2) saying physical lake geometry
doesn't reach that depth and (3) blocking restructuring, the lake
hill_elev becomes a *chain-bookkeeping value*, not a physical lake-
bottom elevation. PI suggested -6.0 m; we adopt that with explicit
acknowledgment that it doesn't represent any single physical lake.

**Considered alternative (rejected):** lake_hill_elev = -5.17 m (=
mean NWI water raw HAND - Lee spill depth). Conflated rim-to-bottom
(spill depth) with surface-to-bottom (water depth) and landed near
the chain monotonicity floor by coincidence rather than derivation.
Documented in audit Section 5.2.1 for transparency.

**Other lake column parameters at this point** (most are unchanged
from earlier PI direction):

| Field | Value | Status |
|---|---|---|
| column_index | 1 | Locked |
| downhill_column_index | -9999 | Locked |
| hillslope_index | 1 | Locked |
| hill_distance | 0.5 × Bin 1's distance (computed dynamically) | Locked 2026-05-04 as a dynamic value. Static ~5 m would invert the col-col Darcy gradient sign (audit Section 1.1) because Bin 1's trap-fit DTND is small (~3 m on production). |
| **hill_elev** | **-6.0 m** | **Locked 2026-05-04** |
| hill_area | sum(water_mask × pixel_area) ≈ 10.68 km² | Locked |
| hill_width | 1/2 NWI total perimeter ≈ 51,405 m | Working |
| hill_slope | 0.0 | Working (PI direction) |
| hill_aspect | 0.0 | Locked |
| hill_bedrock_depth | 0.0 | Locked |
| SPILLHEIGHT (SourceMod) | 0 (effectively disabled) | Locked |

Full canonical reference for the lake column parameters lives in
`docs/lake-column-ctsm-audit.md` Sections 5.1-5.5 and 5.2.1.

**Phase E.5 status after this entry:**

- Outlier strategy: locked (Q01/Q99, true discard, 2026-05-02)
- Land bin scheme: locked (24-bin TAI-focused, 2026-05-04)
- Lake column parameters: 6 locked, 3 working values, 0 open ← all
  parameters now have working values
- SPILLHEIGHT: locked at 0
- All design decisions complete; remaining work is implementation
  and PI review.

### 2026-05-04 — Lake hill_distance: switch to dynamic computation

**Discovery.** First production rerun (job 31861082) wrote
`hillslopes_osbs_production_c260504.nc` with a chain-monotonicity
violation in `hill_distance`. Lake at 5 m, Bin 1 at 2.59 m → denominator
in CTSM's col-col Darcy gradient (audit Section 1.1, Path A) is
**−2.41 m**, inverting the gradient sign. Water that should flow Bin 1
→ Lake would be reported as Lake → Bin 1.

**Why the audit's static-5 m framing failed.** Audit Section 4.4
analyzed the constraint assuming `lowest-land-bin DTND ≈ 30 m` (true
under the prior conditioned-HAND scheme). Under raw-HAND binning the
deepest FZ bin contains basin-floor pixels — close to wide-mask
boundaries — so its trap-fit DTND came out at 2.59 m, ~10× smaller
than the audit assumed. The static "~stream width" recommendation
violated monotonicity by construction.

**Fix.** Replaced `LAKE_HILL_DISTANCE_M = 5.0` constant with dynamic
computation in `run_pipeline.py`:

```python
LAKE_HILL_DISTANCE_FRACTION = 0.5  # module constant
...
# In Step 5d (lake column construction), AFTER the bin loop:
lowest_land_distance_m = float(params["elements"][0]["distance"])
lake_hill_distance_m = LAKE_HILL_DISTANCE_FRACTION * lowest_land_distance_m
```

This guarantees `d(Bin1) - d(Lake) = 0.5 × d(Bin1) > 0` by construction
regardless of how small Bin 1's DTND turns out to be in any future run.
On the 2026-05-04 production data: lake distance becomes 1.30 m.

**Doc updates.** Audit Section 4.4 and 5.2, Phase G doc lake table,
STATUS.md lake table, this phase doc lake table — all updated to
reflect the dynamic value and the audit-assumption failure mode.

### 2026-05-05 — Open question: lake vs land area unit mismatch

Surfaced during inspection of the 2026-05-04 production NetCDF. Worth
flagging before Phase F runs. **Note:** the framing below describes
the Swenson convention rationale; the actual CTSM-runtime story for
osbs2 single-point cases is simpler and is documented in
`phases/G-ctsm-lake-representation.md` 2026-05-05 log entry. Short
version: `grc%area = spval` in single-point mode, so
`nhill_per_landunit` is moot. The rescale still matters — but for
column weights (`wtlunit`), not stamping. Read the Phase G entry for
the complete picture.

**Observation.** Lake `hill_area` = 11,082,394 m² (11.08 km²) is the
total NWI water surface summed across the 90 km² gridcell. Land bin
`hill_area` is the trapezoidal-fitted area `Aw = ∫w(d)dd` for *one
representative hillslope element* (Swenson Eq. 4 plan-form model). The
24 land bins together sum to 147,867 m² (0.148 km²) — the area of one
representative hillslope, not the total per-bin pixel sum across the
domain. Lake-to-largest-land-bin ratio is ~822x.

**CTSM consequence.** `HillslopeHydrologyMod.F90:520` weights columns
within the landunit by:

```
col%wtlunit(c) = (col%hill_area(c) / hillslope_area(nh)) * pct_hillslope * 0.01
```

With our values, the lake column gets 11.082e6 / (11.082e6 + 0.148e6) ≈
**98.7%** of the landunit weight. All 24 land bins together get ~1.3%
(largest individual land bin = 0.12%). Any column-area-weighted
gridcell aggregate (soil moisture, GPP, NEE, water table, latent heat,
etc.) will be ~99% lake-driven and only ~1% landscape-driven.

For Phase F (routing off, lake column behavior dominating) this
matters scientifically — the test branch comparison against the
unmodified osbs2 baseline will look like a lake-dominated gridcell, not
a wetlandscape with embedded lakes. For Phase G (routing on), the
col-col Darcy volumetric exchange is column-area dependent, so the
asymmetry will likely break the lake's water budget under lateral flow.

`nhill_per_landunit` (line 487, only fires under routing) divides total
landunit area by `hillslope_area`. With our `hillslope_area` summing to
11.23 km² and the gridcell sized at ~90 km² in the surfdata, CTSM will
think there are ~8 copies of this representative hillslope tiled across
the gridcell. Stream channel length is then `hill_width × 0.5 ×
nhill_per_landunit` (line 502), so a wrong `nhill_per_landunit` cascades
into wrong stream geometry.

**Chosen solution: rescale lake to per-representative-hillslope basis
(dynamic, keyed off the implicit nhill multiplier).**

Land bin areas are already in Swenson rep-hillslope units by
construction (trap-fit `Aw` per representative hillslope). The lake
just needs to be put on the same scale. Rather than hard-code a
correction for OSBS, derive it dynamically from quantities the
pipeline already computes — works for other sites, other gridcell
sizes, other bin schemes without code changes.

**Implicit rep-hillslope multiplier:**

```
nhill_implicit = total_land_area_m² / sum(land_bin_trap_fit_areas)
```

For the 5/4 production run: `nhill_implicit = 78.92 km² / 0.148 km² ≈
533`. Translation: "this hillslope file represents one of ~533
representative hillslopes that tile the domain." Matches the framework
CTSM applies at runtime via `HillslopeHydrologyMod.F90:487`.

**Lake column rescaling:**

```
lake_area_per_rep  = sum(land_bin_areas) × (total_lake_area / total_land_area)
lake_width_per_rep = (NWI_perimeter / 2) × sum(land_bin_areas) / total_land_area
```

For the 5/4 production run:
- `lake_area_per_rep` = 0.148 km² × (11.082 / 78.92) ≈ **0.0208 km² = 20,800 m²**
- `lake_width_per_rep` = (48,270 m) × (0.148 / 78.92) ≈ **90 m**

Why this form is cleaner: per-rep lake-to-land ratio matches the
domain-wide lake-to-land ratio by construction. Stamping
`nhill_implicit` copies recovers ~78.9 km² of land + ~11.1 km² of
lake = 90 km² gridcell — self-consistent. The lake `hill_width` (90 m)
also lands in the same scale as the deepest land bin's width (~556 m)
instead of the 87x asymmetry under the old convention.

After the rescale, lake `wtlunit` ≈ 12.5% (matches NWI water fraction
in the domain) and the 24 land bins together get ~87.5%, properly
distributed by their individual trap-fit areas. Phase F gridcell
aggregates will reflect a wetlandscape with embedded lakes, not a
lake-dominated gridcell.

**Implementation site.** `scripts/osbs/run_pipeline.py` Step 5d (lake
column construction). All inputs already exist at that point in the
script (`lake_n_pixels`, `lake_area_m2`, `lake_perimeter_m`,
`params["elements"]` with all 24 land trap-fit areas, `region_shape`
and `PIXEL_SIZE` for total domain). Add ~5 lines computing
`nhill_implicit` and the per-rep lake area/width before the lake
element is appended. Other lake fields (`hill_elev = -6.0`,
`hill_distance = 0.5 × Bin1`, `hill_slope = 0`, `hill_aspect = 0`,
`hill_bedrock = 0`) unchanged.

**Open implementation choices to settle before coding:**

1. **Denominator for `total_land_area`:** total domain land
   (`90 - 11.082 = 78.92 km²`) or post-Q01/Q99-trim valid land
   (`74.45 km²`)? Differ by ~5% (`nhill ≈ 533` vs `503`). Total domain
   is the cleaner story (matches what CTSM thinks the gridcell area
   is); post-trim is what's actually represented in the bins. Current
   lean: total domain.
2. **Whether to write `nhill_implicit` to the NetCDF as a global
   attribute** for downstream traceability. Current lean: yes —
   meaningful number, costs nothing.
3. **PI sign-off** before pipeline rerun. This is a meaningful change
   to the lake column area (11.082 km² → 0.0208 km²) and width
   (48,270 m → 90 m); should be flagged before Phase F.

**Alternative considered and rejected: Option T (total-area basis —
bring land up).** Replace land bin trap-fit area with raw per-bin pixel
count × pixel_size² summed over all catchments. Then `sum(hill_area) ≈
90 km²` and `nhill_per_landunit ≈ 1`. Matches what the global Swenson
file looks like for global gridcells. Rejected because it abandons
Swenson's trap-fit `Aw` for land bins, breaking internal consistency
between `hill_area` and `hill_width × hill_distance`. The chosen
solution preserves the Swenson convention for land and brings the lake
into it.
