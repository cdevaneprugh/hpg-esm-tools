# Water Masking and Lake Representation in the OSBS Hillslope Pipeline

Date: 2026-03-27

## Status (2026-04-09)

**The weir overflow approach described in this document was superseded on 2026-04-09 by a simpler submerged lake column approach.** See `phases/G-ctsm-lake-representation.md` for the current Phase G direction.

This document remains valuable as historical reference:

- The **NWI water masking analysis and dual-mask solution** are implemented and production-ready (see Problems 1-2 and the "Solution" section). This is current.
- The **CTSM source investigation** (stream water cycle, lateral flow, carbon-water coupling, PCT_LAKE interaction, MOSART coupling) is still accurate and useful when reasoning about hillslope hydrology behavior. The fact that negative `hill_elev` is permitted (`ColumnType.F90:76`, no guards) is what makes the new submerged-column approach viable.
- The **modification options A-E** in the "CTSM Modification Options for Lake/Wetland Representation" section are no longer planned implementations.

PI decision (after consulting collaborating scientists): lake bathymetry detail and stream-as-lake repurposing are more complexity than OSBS needs. The submerged-column approach leverages the existing CTSM subgrid machinery without requiring Fortran modifications from our fork — the PI's existing spillheight SourceMod handles the model-side behavior.

## Background

The OSBS hillslope pipeline generates representative hillslope parameters for
CTSM from 1m NEON LIDAR data. Open water bodies (lakes, ponds) cover ~12% of the
production domain (10.7M pixels, 1068 ha across 103 NWI features). These water
pixels contaminate the hillslope statistics — particularly the lowest HAND bins —
because Swenson's slope-based water detection (`identify_open_water()`) produces
zero detections at 1m resolution (see `synthetic_lake_bottoms.md` for details on
why detection fails).

An NWI (National Wetlands Inventory) water mask was created from the Lower St.
Johns Watershed shapefile (HU8_03080103), filtered to Lacustrine (`L*`) and
Palustrine Unconsolidated Bottom (`PUB*`) features, and rasterized onto the
production DTM grid at `data/mosaics/production/water_mask.tif`.

This document describes the problems encountered integrating the mask into the
pipeline, the ideas explored, and the findings from investigating CTSM source code.

---

## Problem 1: Water Boundary Forcing Fragments Catchments

### What happened

Swenson's code forces a 2-pixel boundary ring around detected water bodies into
the stream network by setting those pixels' accumulation above the threshold:

```python
acc_arr[water_boundary > 0] = accum_threshold + 1   # run_pipeline.py line 812
```

At 90m MERIT with sparse detections, this adds a few dozen pixels to the stream
network — negligible. At 1m with 103 NWI polygons, the boundary ring is 275,668
pixels, nearly doubling the stream network (207K natural -> 482K total).

### Why it matters

The stream network defines catchment boundaries. Each distinct stream reach creates
a separate catchment (hillslope). The 275K boundary pixels form closed rings that
intersect natural streams at many points, creating thousands of new junction points
and stream segments. Each segment defines a new catchment.

Result: n_hillslopes exploded from ~800 to 220,575.

### How it affects parameters

The trapezoidal width fit computes A_sum(d) as total area / n_hillslopes. With
220K hillslopes instead of 800:

| Metric | Before (no mask) | After (boundary forcing) |
|--------|-----------------|-------------------------|
| n_hillslopes | ~800 | 220,575 |
| Per-hillslope area | 28,000 m^2 | 86 m^2 |
| Width | 211-381 m | 1 m (minimum) |
| Stream cells | 207,832 | 389,324 |
| Stream depth | 0.141 m | 0.010 m |
| Stream width | 1.7 m | 0.0 m |

Height and distance were unaffected (computed per-pixel, not per-hillslope).
HAND bins improved as intended (lowest boundary moved from 0.00027m to 0.26m).

### Root cause

The boundary ring looks like a circular stream channel with land on both sides —
upland on the outside and lake interior on the inside. The lake interior is treated
as ordinary (lowered) terrain for D8 routing purposes, so it gets carved into
micro-catchments that drain to different segments of the ring. The upland between
the ring and natural streams is also fragmented.

---

## Problem 2: NaN Propagation from Water Masking

### What happened

Swenson's code sets water pixels to NaN in the inflated DEM after flow directions
are computed but before HAND:

```python
inflated_arr[water_mask > 0] = np.nan   # run_pipeline.py line 796
```

### Why it matters at 1m

`compute_hand()` traces each pixel downstream along its D8 flow path until it hits
a stream pixel. HAND = pixel elevation - stream pixel elevation. If any pixel along
the flow path has NaN elevation, the elevation difference cannot be computed —
HAND becomes NaN.

With 12% of the domain being water, many land pixels' flow paths cross lakes before
reaching a natural stream. NaN-masking the lakes propagates NaN to those upstream
land pixels, losing valid hillslope data.

At 90m MERIT, very few pixels were detected as water, so NaN propagation was
minimal. At 1m with comprehensive NWI masking, the effect is significant.

---

## Problem 3: Swenson's Paper Doesn't Address This

The Swenson & Lawrence (2025) paper does not describe open water handling at all.
The -0.1m lowering, NaN masking, boundary forcing, and flood filter are all
implementation details in Swenson's code — not published methodology. The paper's
flowchart (Figure 9) shows a clean pipeline: DEM -> FFT -> pysheds D8 -> HAND ->
binning -> parameters. No water handling step.

The code-level water handling was designed for sparse detections at 90m where it
was a minor perturbation. It does not scale to 1m LIDAR with comprehensive
external water masks.

---

## Solution: Lake-as-Stream with Dual Channel Masks

### Concept

Instead of creating a boundary ring around lakes, classify lake interior pixels as
stream pixels — but only for HAND computation. The natural stream network is
preserved for catchment delineation, stream statistics, and hillslope classification.

### How it works

1. Build the **natural channel mask** from accumulation (the normal A_thresh
   stream network). Use this for `create_channel_mask()`, stream network
   extraction, stream stats, `compute_hillslope()`, and all downstream
   catchment-based calculations.

2. Build a **wide channel mask** = natural channels OR water mask pixels. Assign
   lake pixels a dummy `channel_id = 0`.

3. Call `compute_hand()` with the wide mask. Land pixels near lakes hit a lake
   "stream" pixel at the shore and get HAND measured from the lake surface
   elevation. Lake pixels themselves get HAND = 0 (stream pixels point to
   themselves). No NaN propagation.

4. At binning time, lake pixels (HAND = 0, stream pixels) are automatically
   excluded from hillslope statistics.

### Why it works

- **No catchment fragmentation.** `compute_hillslope()` uses the natural channel
  mask (string reference `"channel_mask"`). n_hillslopes stays at ~800.
- **No NaN propagation.** Lake pixels have finite (lowered) elevation values during
  HAND computation. All flow paths are continuous.
- **HAND measured from lake edges.** Land pixels adjacent to lakes get HAND relative
  to the lake surface — physically correct.
- **Stream stats from natural network.** Stream depth/width/slope computed before
  lake pixels are added to any mask.
- **No pysheds modification needed.** `compute_hand()` accepts direct array
  arguments for channel_mask and channel_id (confirmed in pgrid.py lines 1866-1871).

### The dummy channel_id

All lake pixels get `channel_id = 0`. Land pixels draining to lakes inherit
`drainage_id = 0`. This adds one extra "catchment" to n_hillslopes (~801 instead
of ~800) — negligible impact on the area calculation.

### What stays from Swenson's approach

- **-0.1m lowering** of lake pixels in the flooded DEM (line 752). This guides D8
  routing through lakes to the pour point. Without it, `fill_depressions` raises
  the lake to the pour-point elevation, creating a flat surface where `resolve_flats`
  adds arbitrary micro-gradients.
- **Flood filter** as a safety valve for edge pixels (though with this approach,
  most water pixels have HAND = 0 rather than NaN, so the filter catches fewer).

### What is removed

- **Boundary ring computation** (binary_dilation of water mask)
- **Boundary forcing into accumulation** (the line that caused fragmentation)
- **NaN masking of water pixels** after flow directions (the line that caused
  NaN propagation)

---

## CTSM Source Code Investigation

Investigation of the CTSM Fortran source at `$BLUE/ctsm5.3/src/` to evaluate
proposed lake representation approaches.

### Key findings

#### Stream parameters (`HillslopeHydrologyMod.F90`)

- `stream_slope` is used in `sqrt(stream_slope)` in Manning's equation (line 1012).
  Zero slope gives zero velocity — water accumulates in the stream forever. Not a
  crash, but not physical.
- `stream_width = 0` would cause division by zero, but there is an `active_stream`
  guard (lines 991-996) that prevents entering the calculation if width is zero.
- Stream parameters are per-landunit (one set per gridcell, not per-column).

#### Negative `hillslope_height` is allowed

- No guards, sign checks, or absolute value operations on `hill_elev` anywhere in
  the source (`ColumnType.F90` line 76, `HillslopeHydrologyMod.F90` line 361).
- The lateral flow head gradient equation (line 2261-2263) works with negative
  values — it computes differences, so signs cancel appropriately.
- Negative `hill_elev` means "below stream" — flow from stream to column becomes
  the default. This is physically correct for a basin below the pour point.
- Head gradient is clamped to [-2, +2] regardless (line 2300).

#### Lateral subsurface flow (`SoilHydrologyMod.F90`)

- Driven by Darcy-law hydraulic head gradient between adjacent columns.
- Bidirectional — water can flow uphill if the downhill column has a higher water
  table.
- When the stream is empty, negative gradients are clamped to zero (line 2282-2283)
  — the empty stream can't pull water from nowhere.
- Stream-to-column flow is limited by available stream water volume (line 2362-2366).
- Two methods: Darcy (default, uses water table position) and Kinematic (uses
  topographic slope only).

#### Lake and hillslope land units are completely separate

- Hillslope columns only exist on `istsoil` (vegetated) land units
  (line 257: `lun%itype(l) == istsoil`).
- Lake hydrology runs in a separate code path with a separate filter.
- `PCT_LAKE` is defined in the surface dataset (`fsurdat`), not the hillslope file.
- **There is no mechanism for water exchange between lake and hillslope land units.**
  They are independent boxes. Their outputs combine at the gridcell level based on
  area weights, but there is no lateral coupling.

This means the standard approach (separate lake land unit + separate hillslope)
inherently cannot represent TAI dynamics where lake water interacts with the
adjacent hillslope.

#### Surface water fill-and-spill (`SurfaceWaterMod.F90`)

- Each column has `h2osfc` (surface water depth) and `h2osfc_thresh` (ponding
  threshold). When `h2osfc > h2osfc_thresh`, excess becomes surface runoff.
- This is within-column only — no lateral surface water transfer between columns.
- Surface runoff goes directly to the stream, not to adjacent columns.
- `h2osfc` is designed for shallow ponding (mm to cm), not meters of lake depth.

#### The PI's spillheight SourceMod experiment

Documented in `$TOOLS/docs/SPILLHEIGHT_IMPLEMENTATION.md`. Subtracts a uniform
`spillheight` from all `hill_elev` values, reducing the head gradient for all
columns and slowing subsurface drainage. This is not in the mainline fork — it
is a SourceMod experiment. It represents the closest existing approach to
modifying the hillslope-stream interaction for wetland dynamics.

---

## Ideas Explored and Their Viability

### 1. Shifted coordinate system (lake bottom = h=0)

**Concept:** Redefine the profile so the lake bottom is at h=0 and the spill
height (lake shore) is a positive value. All columns have positive height. The
hillslope profile extends continuously from lake bottom up to ridge.

**CTSM behavior:** The code would run without crashing. Lateral flow would drive
water from upland toward basin columns, which is correct. But:
- Soil columns cannot represent standing water (no deep ponding mechanism)
- No spill mechanism at the shore threshold
- Vegetation and soil processes run on "lake bottom" columns
- Surface runoff bypasses lateral routing — goes directly to stream

**Verdict:** Mechanically possible, physically problematic. Does not produce lake
behavior without CTSM source modifications.

### 2. Negative `hillslope_height` for below-stream lake columns

**Concept:** Keep the stream at h=0 (the lake shore). Add columns below with
negative height representing the lake basin.

**CTSM behavior:** No crashes. Head gradients work with negative values. Flow
direction is correct (stream water flows into below-stream columns). But the same
fundamental limitations apply: soil columns can't represent open water, no spill
mechanism, vegetation runs on submerged columns.

**Verdict:** Technically cleaner than shifted coordinates, same physical limitations.

### 3. Lake as separate land unit with adjusted PCT_LAKE

**Concept:** Keep the hillslope file for land-only characterization. Set
`PCT_LAKE` in `fsurdat` to 12% based on the NWI mask. Lake and hillslope are
independent in CTSM.

**CTSM behavior:** Standard, well-tested approach. Lake gets its own hydrology.
Hillslope parameters are clean (land-only). But no TAI dynamics — lake and
hillslope don't interact.

**Verdict:** Simplest, most defensible. Appropriate for initial validation runs.
TAI dynamics require CTSM development work (separate effort from parameter
generation).

### 4. Spillheight SourceMod approach

**Concept:** Use the PI's existing spillheight SourceMod to modify the
stream-column interaction. Reduce `hill_elev` by a spillheight value, slowing
drainage from lowland columns.

**CTSM behavior:** Tested by the PI. Reduces subsurface drainage, creating
wetter conditions near the stream. Does not create a lake but simulates the
effect of a raised water surface on hillslope hydrology.

**Verdict:** Promising intermediate step. Builds on existing PI work. Does not
require lake basin representation in the hillslope file.

---

## CTSM Stream Water Cycle (Detailed)

Understanding the full stream lifecycle is essential for evaluating how to
repurpose it for lake/wetland representation.

### How water enters the stream

`HillslopeUpdateStreamWater` (`HillslopeHydrologyMod.F90` lines 1053-1146)
iterates over all hillslope columns in the landunit and collects three flux
terms from each:

```fortran
! Lines 1114-1125
qflx_surf_vol = qflx_surf(c) * 1.e-3_r8 * (grc%area(g)*1.e6_r8*col%wtgcell(c))
qflx_drain_perched_vol = qflx_drain_perched(c) * 1.e-3_r8 * (...)
qflx_drain_vol = qflx_drain(c) * 1.e-3_r8 * (...)

stream_water_volume(l) = stream_water_volume(l) &
     + (qflx_drain_perched_vol + qflx_drain_vol + qflx_surf_vol) * dtime
```

Sources:
- `qflx_surf` — surface runoff (overland flow)
- `qflx_drain_perched` — perched water table drainage (lateral flow above frost table)
- `qflx_drain` — saturated zone drainage (lateral flow below water table)

### How stream water is stored

`stream_water_volume` is a **landunit-level** scalar (one bucket per gridcell's
vegetated landunit). Declared in `WaterStateType.F90` line 60. Units: m^3.

`stream_water_depth` is a diagnostic (`HillslopeHydrologyMod.F90` lines 1137-1139):
```fortran
stream_water_depth(l) = stream_water_volume(l) &
     / lun%stream_channel_length(l) / lun%stream_channel_width(l)
```
Assumes rectangular cross-section: depth = volume / (length x width).

### How water leaves the stream (Manning discharge)

`HillslopeStreamOutflow` (`HillslopeHydrologyMod.F90` lines 939-1050):

```fortran
! Lines 999-1012
cross_sectional_area = stream_water_volume(l) / lun%stream_channel_length(l)
stream_depth = cross_sectional_area / lun%stream_channel_width(l)
hydraulic_radius = cross_sectional_area &
     / (lun%stream_channel_width(l) + 2._r8*stream_depth)

flow_velocity = (hydraulic_radius)**manning_exponent &
     * sqrt(lun%stream_channel_slope(l)) / manning_roughness
volumetric_streamflow(l) = cross_sectional_area * flow_velocity
```

Overbank flow (lines 1015-1031): when `stream_depth > stream_channel_depth`, the
default method multiplies streamflow by `stream_depth/stream_channel_depth`,
increasing the effective discharge.

Discharge is clamped to not exceed available volume (line 1040):
```fortran
volumetric_streamflow(l) = max(0._r8, &
     min(volumetric_streamflow(l), stream_water_volume(l)/dtime))
```

### How stream water flows back to hillslope (losing-stream)

The bidirectional exchange occurs in `SubsurfaceLateralFlow` (`SoilHydrologyMod.F90`
lines 2249-2402). For the lowest hillslope column (where `col%cold(c) == ispval`):

```fortran
! Lines 2276-2280
! bankfull height is defined to be zero
head_gradient = (col%hill_elev(c)-zwt(c)) &
     - min((stream_water_depth - stream_channel_depth), 0._r8)
head_gradient = head_gradient / (col%hill_distance(c))
```

The stream's effective water level is `min(stream_water_depth - stream_channel_depth, 0)`:
- Below bankfull: negative value (stream surface below reference plane)
- At or above bankfull: clamped to 0

When `head_gradient < 0` (stream head higher than column head), flow reverses
from stream into the column. The transmissivity for losing-stream flow
(lines 2346-2347):
```fortran
transmis = (1.e-3_r8*ice_imped(c,jwt(c)+1)*hksat(c,jwt(c)+1)) * stream_water_depth
```

An additional vertical drainage term is added (lines 2287-2291):
```fortran
if (head_gradient < 0._r8) then
   head_gradient = head_gradient - 1._r8/k_anisotropic
endif
```

Flow from stream to column is limited by available stream water (lines 2362-2367):
```fortran
if (use_hillslope_routing .and. (qflx_latflow_out_vol(c) < 0._r8)) then
   available_stream_water = stream_water_volume(l) &
        / lun%stream_channel_number(l) / nhillslope
   if (abs(qflx_latflow_out_vol(c))*dtime > available_stream_water) then
      qflx_latflow_out_vol(c) = -available_stream_water/dtime
   endif
endif
```

When the stream is empty, negative gradients are clamped to zero (lines 2282-2283):
```fortran
if (stream_water_depth <= 0._r8) then
   head_gradient = max(head_gradient, 0._r8)
endif
```

### What controls stream water level

Five factors at steady state:
1. Inflow rate from all hillslope columns (drainage + surface runoff)
2. Manning outflow rate (depth^(5/3) * slope^(1/2))
3. Losing-stream recharge to lowest column (negative feedback)
4. Channel geometry constants (depth, width, length, slope)
5. Number of channels (`stream_channel_number`)

---

## CTSM Carbon-Water Coupling

The carbon model responds automatically to water table position through multiple
pathways. This is critical: if the hydrology is correct, the carbon dynamics
follow without biogeochemistry code changes.

### Decomposition rate (w_scalar)

`SoilBiogeochemDecompCascadeBGCMod.F90` lines 753-763: decomposition rate scales
with soil water potential. When soil is saturated (`soilpsi = maxpsi`),
`w_scalar -> 1` (maximum). When dry (`soilpsi < minpsi`), `w_scalar = 0`.

### Oxygen limitation (o_scalar)

`SoilBiogeochemDecompCascadeBGCMod.F90` lines 847-860: when `use_lch4 = .true.`
and `anoxia = .true.`, decomposition is further limited by oxygen stress.
`o_scalar = max(o2stress_unsat, mino2lim)`. Below the water table, O2 diffusion
is ~10,000x slower (liquid vs gas), so `o2stress_unsat` drops sharply. This
suppresses aerobic decomposition in saturated soil.

### Fractional inundated area (finundated)

`ch4FInundatedStreamType.F90` lines 294-316: computed from water table depth:
```fortran
finundated(c) = f0 * exp(-zwt_actual/zwt0) + p3*qflx_surf_lag
```
Shallow water table (small `zwt`) gives high `finundated`, which increases
CH4 production. Alternatively: `finundated(c) = frac_h2osfc(c)` when using
the surface water method.

### CH4 production

`ch4Mod.F90` line 2371-2372: "Production is done below the water table, based
on CN heterotrophic respiration." The CH4 model splits each column into saturated
and unsaturated fractions. In the saturated zone, aerobic decomposition is
suppressed and a fraction of remaining carbon mineralization becomes CH4.

### The complete chain

```
zwt (water table depth)
  |-> soilpsi -> w_scalar -> decomp_k (decomposition rate)
  |-> O2 transport -> o2stress_unsat -> o_scalar -> decomp_k
  |-> finundated -> CH4 production, transport, oxidation
```

**Key implication:** getting the water table dynamics right near lakes/wetlands
(rising water table -> saturated soil -> suppressed aerobic decomposition ->
increased CH4) produces the TAI carbon dynamics automatically. No
biogeochemistry code modifications needed.

---

## CTSM Modification Options for Lake/Wetland Representation

### Option A: Lake-shaped stream (parameter changes only, no code changes)

Set stream parameters to lake-like values:
- `stream_channel_depth` = characteristic lake depth
- `stream_channel_width` = effective lake width (from NWI polygon areas)
- `stream_channel_slope` = very small but nonzero

**What works:**
- Larger width/depth increase storage capacity
- Small slope reduces Manning discharge, making the "stream" sluggish and buffered
- Bidirectional exchange mechanism works unchanged

**What breaks:**
- Manning gives **continuous discharge** even with tiny slope. Q ~ slope^(0.5),
  never zero. A real lake has zero outflow below the pour point.
- Stream geometry is 1D (volume = cross_section x length). Cannot represent 2D
  lake surface area.
- `stream_channel_length` is structurally determined from hillslope geometry
  (`HillslopeHydrologyMod.F90` lines 498-504), not a free parameter.
- Overbank handling (lines 1015-1031) accelerates drainage above bankfull — opposite
  of desired lake behavior.
- Losing-stream transmissivity (line 2347) uses `stream_water_depth` directly.
  Deep lake would vastly overestimate subsurface return flow.

**Verdict:** Rough first-order approximation. Cannot capture threshold discharge
behavior. May be useful for sensitivity testing.

### Option B: Replace Manning with weir overflow (~70 lines Fortran)

Replace Manning's equation in `HillslopeStreamOutflow` with a weir equation:
```
Q = C * L * max(0, h - h_spill)^(3/2)
```

**Zero discharge below pour point.** Water accumulates in the stream bucket.
Water table rises in the lowest column via losing-stream exchange. Soil becomes
anaerobic. CH4 production increases. When water level exceeds the spill height,
discharge begins — the TAI overflow event.

**Code changes required:**

1. `HillslopeStreamOutflow` (`HillslopeHydrologyMod.F90` lines 999-1041):
   replace Manning block with weir equation. ~40 lines.
   ```fortran
   if (stream_depth > lun%stream_spill_height(l)) then
      overflow_depth = stream_depth - lun%stream_spill_height(l)
      volumetric_streamflow(l) = weir_coefficient &
           * lun%stream_spill_width(l) * overflow_depth**(1.5_r8)
   else
      volumetric_streamflow(l) = 0._r8
   endif
   ```

2. `LandunitType.F90`: add `stream_spill_height` and `stream_spill_width`
   variables. ~6 lines.

3. `InitHillslope` (`HillslopeHydrologyMod.F90`): read new variables from
   NetCDF. ~20 lines.

4. Optionally add a `streamflow_method` namelist selector so weir and Manning
   can coexist for different configurations.

**Total: ~70 lines of Fortran**, highly localized to one subroutine plus
initialization. The highest-impact, lowest-complexity option.

**Remaining concern:** Stream geometry is still 1D (depth = volume / length / width).
For a lake, water level should scale with area, not a rectangular channel. But if
width and length are chosen to match lake surface area at the pour point, this is
an acceptable approximation.

**New pipeline parameters needed:**
- `stream_spill_height`: elevation difference between lake surface and pour point rim.
  Computable from DEM + NWI mask (pour point analysis).
- `stream_spill_width`: effective width of the spillway/outflow. Estimated from DEM
  gradient at pour point, or simplified as a constant.

### Option C: Add a wetland storage pool (~200+ lines Fortran)

Add a new state variable (`wetland_water_volume`) between the hillslope and
stream. Water drains from hillslope to wetland, then from wetland to stream
when the wetland overflows.

**Would require:**
- New state variable in `WaterStateType.F90`
- New diagnostic in `WaterDiagnosticBulkType.F90`
- New wetland properties in `LandunitType.F90`
- New subroutines: `HillslopeUpdateWetlandWater`, `HillslopeWetlandOutflow`
- Modify `SubsurfaceLateralFlow` and `PerchedLateralFlow`
- Balance check, restart, and history output modifications

**Verdict:** Cleanest conceptual approach, but ~200+ lines of Fortran across many
files. Would likely require upstream CTSM coordination. Not recommended as first
step.

### Option D: Variable stream geometry (depth-area curve)

Make `stream_channel_width` a function of `stream_water_depth` by reading a
lookup table from the NetCDF.

**Would require changes to:**
- Manning discharge (width enters as denominator)
- Depth-from-volume calculation (width is no longer constant)
- Stream bank length computation
- Lateral flow routines

**Verdict:** Pervasive changes across many subroutines. More complex than Option B
for less clear benefit. Could be a future refinement on top of Option B.

### Option E: Large h2osfc_thresh on lowest column (~5 lines Fortran)

`h2osfc_thresh` is per-column (`SoilHydrologyType.F90` line 30). Computed in
`SoilHydrologyInitTimeConstMod.F90` lines 214-236 from `micro_sigma` (function
of slope). The surface runoff code (`SurfaceWaterMod.F90` lines 484-493):

```fortran
if (h2osfc(c) > h2osfc_thresh(c) .and. h2osfcflag/=0) then
   k_wet = 1e-4_r8 * max(col%hill_slope(c), min_hill_slope)
   qflx_h2osfc_surf(c) = k_wet * frac_infclust * (h2osfc(c) - h2osfc_thresh(c))
endif
```

Override for the lowest hillslope column:
```fortran
if (col%is_hillslope_column(c) .and. col%cold(c) == ispval) then
   soilhydrology_inst%h2osfc_thresh_col(c) = spillheight * 1000._r8  ! m to mm
endif
```

**~5 lines of code.** Surface water ponds up to spill height before overflowing.
Complements Option B by allowing visible surface ponding. `frac_h2osfc` drives
`finundated` in the CH4 model.

**Limitations:**
- `h2osfc` is designed for shallow ponding (mm-cm), not meters of lake depth
- Subsurface exchange still uses stream head gradient, not ponded water level
- Carbon model interaction through `frac_h2osfc` may not be calibrated for
  deep ponding

**Verdict:** Useful complement to Option B. Not sufficient alone.

---

## PCT_LAKE Interaction

### How PCT_LAKE works

`surfrdMod.F90` lines 591-613: `PCT_LAKE` is read from the surface dataset and
determines the weight of the deep lake landunit (`istdlak = 5`):
```fortran
wt_lunit(nl,istdlak) = pctlak(nl) / 100._r8
```

Setting `PCT_LAKE > 0` **steals area from the vegetated landunit** (`istsoil`),
which is where hillslope columns live. The sum of PCT_WETLAND + PCT_LAKE +
PCT_URBAN + PCT_GLACIER + PCT_OCEAN must be < 100%.

### The wetland land unit

The wetland landunit (`istwet = 6`) in `HydrologyDrainageMod.F90` lines 195-204
is extremely simple: all drainage, infiltration, and runoff are set to zero. The
only flux is `qflx_qrgwl` (residual for water balance). Wetland columns are a
bucket receiving precipitation and losing water only by evaporation.

### Design choice: separate lake vs lake-within-hillslope

**Separate lake (`PCT_LAKE > 0`):** Standard approach. Lake gets its own hydrology
(lake temperature model, etc). No TAI coupling with hillslope. Simple, defensible,
no code changes. Appropriate for initial validation.

**Lake-within-hillslope (`PCT_LAKE = 0`):** Novel approach. Lake is represented
through modified stream parameters (Option B weir overflow). The hillslope
occupies 100% of the vegetated landunit. Lake storage and dynamics are implicit
in the stream bucket. Enables TAI coupling. Requires Fortran modifications.

The project's scientific goal (TAI dynamics) argues for the lake-within-hillslope
approach. But the separate-lake approach is the safer starting point for
validation.

---

## MOSART Coupling

Stream discharge reaches MOSART via `lnd2atmMod.F90` lines 343-354:
```fortran
if (use_hillslope_routing) then
   water_inst%waterlnd2atmbulk_inst%qflx_rofliq_stream_grc(g) = &
        water_inst%waterlnd2atmbulk_inst%qflx_rofliq_stream_grc(g) &
        + water_inst%waterfluxbulk_inst%volumetric_streamflow_lun(l) &
        * 1e3_r8 / (grc%area(g)*1.e6_r8)
endif
```

In the coupler (`lnd_import_export.F90` lines 911-921), this is added to
subsurface runoff export (`Flrl_rofsub`) and sent to MOSART.

When hillslope routing is active, hillslope columns are excluded from standard
gridcell-averaged runoff exports (`lnd2atmMod.F90` line 370:
`if (.not. col%is_hillslope_column(c))`) to avoid double-counting.

**Impact of weir overflow (Option B):** MOSART receives zero runoff from the
"lake stream" when below the pour point, and a pulse when it overflows. MOSART
doesn't care about the mechanism — it just receives mm/s. Physically correct for
a lake with a surface outlet.

**Closed basins:** If a lake never overflows, `volumetric_streamflow` is
persistently zero. Water leaves only by evaporation and losing-stream recharge.
This is physically correct for a closed-basin sinkhole lake but may trigger
balance check warnings.

---

## Recommended Path Forward

### Immediate (pipeline fix)

Implement the lake-as-stream dual-mask approach to get clean hillslope parameters:

1. Natural channel mask for catchment delineation and stream stats
2. Wide channel mask (natural + lake) for HAND computation
3. Lake pixels get HAND = 0, excluded from hillslope stats
4. Remove boundary forcing and NaN masking
5. Keep -0.1m lowering for routing continuity

This produces a hillslope file that characterizes the land portion of OSBS with
HAND measured from lake edges. HAND bins are clean. Area/width are correct.

### Short-term (CTSM validation)

Use the land-only hillslope file with `PCT_LAKE` set from the NWI mask. Run
comparison simulations against the osbs2 baseline. This validates the hillslope
characterization without attempting TAI dynamics.

### Medium-term (weir overflow + bathymetry)

1. Implement Option B (weir overflow) in CTSM — ~70 lines of Fortran, localized
   to `HillslopeStreamOutflow` plus initialization.
2. Integrate lake bathymetry data to compute `stream_spill_height` and
   characteristic lake geometry for the stream parameters.
3. Set `PCT_LAKE = 0`, represent lake within the hillslope framework.
4. Optionally add Option E (h2osfc_thresh) for surface ponding (~5 lines).

This enables TAI dynamics: lake fills from hillslope drainage, water table rises
in adjacent soil, aerobic decomposition suppresses, CH4 increases, and overflow
triggers the spill event. Carbon response comes automatically from existing
w_scalar, o_scalar, and finundated pathways — no biogeochemistry changes needed.

### Long-term (refinements)

- Option D (variable stream geometry with depth-area curve) for more realistic
  lake level dynamics
- Option C (explicit wetland storage pool) if the stream-as-lake abstraction
  proves insufficient
- Upstream coordination with CTSM development team for mainline integration

---

## Key CTSM Source Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `src/biogeophys/HillslopeHydrologyMod.F90` | Init, stream outflow, stream water update | Init: 257-504, Outflow: 939-1050, Update: 1053-1146 |
| `src/biogeophys/SoilHydrologyMod.F90` | Lateral flow between columns | Perched: 1703-1987, Subsurface: 2087-2510 |
| `src/biogeophys/HydrologyDrainageMod.F90` | Calling sequence for lateral flow | 139-158 |
| `src/biogeophys/SurfaceWaterMod.F90` | h2osfc fill & spill | 484-493 |
| `src/biogeophys/SoilHydrologyInitTimeConstMod.F90` | h2osfc_thresh initialization | 214-236 |
| `src/biogeophys/SoilHydrologyType.F90` | h2osfc_thresh declaration (per-column) | 30 |
| `src/main/ColumnType.F90` | `hill_elev` definition | 76 |
| `src/main/LandunitType.F90` | Stream channel properties | 57-61 |
| `src/main/TopoMod.F90` | Meteorological downscaling using `hill_elev` | 143-144, 288-291 |
| `src/main/surfrdMod.F90` | `PCT_LAKE` read from surface dataset | 532, 591-613 |
| `src/biogeochem/SoilBiogeochemDecompCascadeBGCMod.F90` | w_scalar, o_scalar decomposition | 753-763, 847-860, 888-903 |
| `src/biogeochem/ch4Mod.F90` | CH4 production below water table | 2371-2372 |
| `src/biogeochem/ch4FInundatedStreamType.F90` | finundated from water table depth | 294-316 |
| `src/cpl/lnd2atmMod.F90` | Stream discharge to MOSART | 343-354, 370 |
| `src/cpl/lnd_import_export.F90` | Coupler export of stream runoff | 911-921 |

---

## References

- Swenson & Lawrence (2025): `docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md`
- Swenson et al. (2019): Intra-hillslope lateral subsurface flow in CLM
- Synthetic lake bottoms: `docs/synthetic_lake_bottoms.md`
- Spillheight implementation: `$TOOLS/docs/SPILLHEIGHT_IMPLEMENTATION.md`
- Phase E tracking: `phases/E-complete-parameters.md`
- NWI data: `data/HU8_03080103_Watershed/`
- Water mask: `data/mosaics/production/water_mask.tif`

### Immediate (pipeline fix)

Implement the lake-as-stream dual-mask approach to get clean hillslope parameters:

1. Natural channel mask for catchment delineation and stream stats
2. Wide channel mask (natural + lake) for HAND computation
3. Lake pixels get HAND = 0, excluded from hillslope stats
4. Remove boundary forcing and NaN masking
5. Keep -0.1m lowering for routing continuity

This produces a hillslope file that characterizes the land portion of OSBS with
HAND measured from lake edges. HAND bins are clean. Area/width are correct.

### Short-term (CTSM validation)

Use the land-only hillslope file with `PCT_LAKE` set from the NWI mask. Run
comparison simulations against the osbs2 baseline. This validates the hillslope
characterization without attempting TAI dynamics.

### Medium-term (lake bathymetry)

Integrate lake bathymetry data from the new paper the PI identified. Characterize
a representative lake bottom profile (depth, shape, area). This data informs the
CTSM development work but does not go into the hillslope file directly under the
current CTSM framework.

### Long-term (CTSM development)

Implementing TAI dynamics in CTSM requires source code modifications:
- Lake-hillslope water exchange (currently zero coupling between land units)
- Surface water routing between columns (currently runoff goes directly to stream)
- Spill-height mechanism (threshold-based overflow from basin to hillslope)

The PI's spillheight SourceMod and the hillslope parameter data from this pipeline
provide the foundation for this work. The characteristic lake profile (from
bathymetry) would inform the design of the CTSM modifications.

---

## Key CTSM Source Files

| File | Purpose |
|------|---------|
| `src/biogeophys/HillslopeHydrologyMod.F90` | Init, stream outflow, stream water update |
| `src/biogeophys/SoilHydrologyMod.F90` | `PerchedLateralFlow` (line 1703), `SubsurfaceLateralFlow` (line 2087) |
| `src/biogeophys/HydrologyDrainageMod.F90` | Calling sequence for lateral flow (line 139-158) |
| `src/biogeophys/SurfaceWaterMod.F90` | h2osfc fill & spill (line 484) |
| `src/main/ColumnType.F90` | `hill_elev` definition (line 76) |
| `src/main/LandunitType.F90` | Stream channel properties (lines 57-61) |
| `src/main/TopoMod.F90` | Meteorological downscaling using `hill_elev` |
| `src/main/surfrdMod.F90` | `PCT_LAKE` read from surface dataset (line 532) |

---

## References

- Swenson & Lawrence (2025): `docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md`
- Synthetic lake bottoms: `docs/synthetic_lake_bottoms.md`
- Spillheight implementation: `$TOOLS/docs/SPILLHEIGHT_IMPLEMENTATION.md`
- Phase E tracking: `phases/E-complete-parameters.md`
- NWI data: `data/HU8_03080103_Watershed/`
- Water mask: `data/mosaics/production/water_mask.tif`
