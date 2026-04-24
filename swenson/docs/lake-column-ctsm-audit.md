# Lake Column CTSM Audit: hill_distance and Implementation Viability

Date: 2026-04-21
Revised: 2026-04-23 (lake at col 1, surface infiltration pathway, bedrock depth
inert under Uniform soil profile, PI items 2 and 6 resolved)
Revised: 2026-04-23 (bathymetric slope from Lee et al. 2023 for `hill_slope`
and `hill_aspect`; spill depth discrepancy flagged for PI)
Revised: 2026-04-23 (verified `tdepth(g) = 0` for osbs4 via
`flds_r2l_stream_channel_depths = .false.` chain; PI item 4 resolved)
Revised: 2026-04-23 (lake-at-col-1 converted from PI item to design decision)
Revised: 2026-04-23 (SPILLHEIGHT=0.2m confirmed; `hill_elev = -SPILLHEIGHT`
formula holds regardless of PI tuning; item 3 resolved, item 7 reframed as
spill-depth tuning)
Revised: 2026-04-23 (Section 6 rewritten: the real HAND contamination source
at OSBS is unmapped wetland basins being filled to rim elevation, not
ridgetop dimples — the latter story was mountainous-terrain intuition and
doesn't apply to OSBS's low relief. Fill_depth diagnostic reframed as
plots-first-then-decide; 1cm threshold retracted.)
Revised: 2026-04-23 (Section 6 expanded with fill-depth decomposition
approach: split the four conditioning stages into separate signals;
`depression_fill_depth > 0` is the crisp contamination marker with no
threshold needed. Added 5-phase diagnostic workflow and secondary signals.)

## Purpose

Assess the viability of adding a 17th "lake" column to the OSBS hillslope NetCDF
file (Phase G). This document catalogs every place `hill_distance` enters CTSM
calculations, identifies failure modes, analyzes three approaches for the lake
column's DTND value, and documents a newly discovered ordering issue in the PI's
spillheight SourceMod.

All line numbers reference the PI's osbs4.branch.v2 SourceMods at
`$CASES/osbs4.branch.v2/SourceMods/src.clm/` unless noted as "upstream" (which
refers to `$BLUE/ctsm5.3/src/`).

---

## 1. Complete Catalog of hill_distance Usage

### 1.1 Darcy Gradient: Column-to-Column (4 occurrences, NO guards)

The Darcy lateral flow calculation computes head gradients between adjacent
hillslope columns. There are two code paths, each appearing twice (once for the
perched water table, once for the main water table):

**Path A: Between two hillslope columns** (`col%cold(c) /= ispval`)

```fortran
head_gradient = (col%hill_elev(c) - zwt(c))
             - (col%hill_elev(col%cold(c)) - zwt(col%cold(c)))
head_gradient = head_gradient / (col%hill_distance(c) - col%hill_distance(col%cold(c)))
```

| Instance | File | Line |
|----------|------|------|
| Perched | SoilHydrologyMod.F90 | 1820-1822 |
| Saturated | SoilHydrologyMod.F90 | 2263-2265 |

**Constraint:** `hill_distance(c) > hill_distance(cold(c))` must hold. If equal,
division by zero. If reversed, the gradient sign flips and water flows uphill.

**Path B: Lowest column to stream** (`col%cold(c) == ispval`)

```fortran
head_gradient = (col%hill_elev(c) - zwt(c))
             - min((stream_water_depth - stream_channel_depth), 0._r8)
head_gradient = head_gradient / (col%hill_distance(c))
```

| Instance | File | Line |
|----------|------|------|
| Perched | SoilHydrologyMod.F90 | 1835-1840 |
| Saturated | SoilHydrologyMod.F90 | 2278-2282 |

**Constraint:** `hill_distance(c) > 0`. Zero causes division by zero.

**Post-division clamp** (saturated path only, line 2302):
```fortran
head_gradient = min(max(head_gradient, -2._r8), 2._r8)
```
Catches extreme values but does not prevent NaN from division by zero.

**Empty-stream guard** (lines 2284-2286):
```fortran
if (stream_water_depth <= 0._r8) then
   head_gradient = max(head_gradient, 0._r8)
endif
```
When the stream is empty (routing off, `tdepth(g) = 0`), negative gradients are
clamped to zero. This blocks water from flowing out of the lowest column to the
stream.

### 1.2 Soil Depth Linear Profile

**Status: NOT A CURRENT CONCERN for osbs4.** Potential concern only if
`hillslope_soil_profile_method` is ever switched away from `Uniform`. See
"Relevance" at the end of this section.

**File:** `HillslopeHydrologyUtilsMod.F90` (upstream, lines 48-84)

```fortran
min_hill_dist = minval(col%hill_distance(lun%coli(l):lun%colf(l)))
max_hill_dist = maxval(col%hill_distance(lun%coli(l):lun%colf(l)))

if (abs(max_hill_dist - min_hill_dist) > toosmall_distance) then  ! 1e-6
   m = (soil_depth_lowland - soil_depth_upland) / (max_hill_dist - min_hill_dist)
else
   m = 0._r8
end if
b = soil_depth_upland

soil_depth_col = m * (max_hill_dist - col%hill_distance(c)) + b
```

This computes a linear soil depth gradient across the hillslope: the column at
`min_hill_dist` gets the deepest soil (lowland), the column at `max_hill_dist`
gets the shallowest (upland).

**Impact of adding a lake column:** The lake's small `hill_distance` becomes the
new `min_hill_dist`, stretching the interpolation range and slightly shifting all
columns' soil depths.

**Guard:** `toosmall_distance = 1e-6` prevents division by zero. If all distances
are equal, `m = 0` and all columns get uniform depth.

**Relevance — NOT ACTIVE for osbs4.** This calculation is only executed when
`soil_profile_method == soil_profile_linear` (method 3). Verified (2026-04-23):
`osbs4/CaseDocs/lnd_in` sets `hillslope_soil_profile_method = 'Uniform'`
(method 1, line 261), under which every hillslope column gets the same soil
thickness regardless of `hill_distance` or `hillslope_bedrock_depth`.

**Potential future concern:** If the configuration is ever switched to
`Linear`, the lake's small `hill_distance` would become the new
`min_hill_dist`, stretching the interpolation range and shifting all land
columns' soil depths by a small amount (~2% for our parameters). Revisit this
section if that method change is ever proposed.

### 1.3 Volumetric Flow and Inflow Routing

**File:** SoilHydrologyMod.F90

Volumetric lateral flow (line 2360):
```fortran
qflx_latflow_out_vol(c) = transmis * col%hill_width(c) * head_gradient
```

Inflow to downhill column (line 2388-2389):
```fortran
qflx_latflow_in(col%cold(c)) = qflx_latflow_in(col%cold(c)) +
     1.e3_r8 * qflx_latflow_out_vol(c) / col%hill_area(col%cold(c))
```

`hill_distance` does not appear directly here, but it controls `head_gradient`
(see 1.1). The lake column's `hill_area` appears as a denominator when receiving
inflow — large area means incoming flux is spread over more surface, producing
a smaller per-area flux. This is physically correct.

### 1.4 Meteorological Downscaling

**File:** `TopoMod.F90` (upstream, lines 254-291)

```fortran
mean_hillslope_elevation(l) += col%hill_elev(c) * col%hill_area(c)
...
topo_col(c) += col%hill_elev(c) - mean_hillslope_elevation(l)
```

Uses `hill_elev` and `hill_area`, not `hill_distance`. A lake column with large
area and deeply negative `hill_elev` pulls down the area-weighted mean elevation,
slightly raising the topo offset of all other columns. Effect is small (~0.05m
shift for a -0.4m lake at 11.9% area fraction).

### 1.5 Stream Channel Length (routing on only)

**File:** HillslopeHydrologyMod.F90, lines 498-504

```fortran
if (col%cold(c) == ispval) then
   stream_channel_length += col%hill_width(c) * 0.5 * nhill_per_landunit(...)
end if
```

The lowest column's `hill_width` enters the stream channel length. Only executed
when `use_hillslope_routing = .true.`. Not relevant for the current osbs2/osbs4
configuration (routing off).

### 1.6 Diagnostic Output

**File:** `histFileMod.F90` (upstream, line 3329)

```fortran
call ncd_io(varname='hillslope_distance', data=col%hill_distance, ...)
```

Written to history files for diagnostics. No calculations involved.

### 1.7 Summary Table

| Location | What uses hill_distance | Guard? | Relevant? |
|----------|------------------------|--------|-----------|
| SoilHydrologyMod:1822 | Perched gradient (col-col) | None | Yes |
| SoilHydrologyMod:1840 | Perched gradient (col-stream) | None | See note |
| SoilHydrologyMod:2265 | Saturated gradient (col-col) | None | Yes |
| SoilHydrologyMod:2282 | Saturated gradient (col-stream) | Empty-stream clamp | See note |
| HillslopeHydrologyUtilsMod:66 | Soil depth linear profile | toosmall_distance | NOT ACTIVE (Uniform method); potential concern if switched to Linear |
| TopoMod:260 | Met downscaling | N/A | Uses hill_elev, not distance |
| HillslopeHydrologyMod:502 | Stream channel length | N/A | Routing off |
| histFileMod:3329 | Diagnostic write | N/A | No |

**Note on col-stream paths:** With routing off and `tdepth(g) = 0`, the gradient
for the lowest column is always clamped to zero (see Section 4.1). The absolute
value of `hill_distance` for the lowest column is irrelevant in this
configuration. However, this assumption depends on MOSART providing zero stream
depth — see Section 7.3.

**Note on hill_width:** Width scales volumetric lateral outflow for ALL columns
(line 2360), not just the lowest. It is always relevant for columns with nonzero
head gradients, regardless of routing. See Section 5.3.

---

## 2. Column Topology

### 2.1 How the Chain Works

CTSM reads `downhill_column_index` from the hillslope NetCDF file. A value of
`-9999` (sentinel, checked as `<= -999`) marks the lowest column, which gets
`col%cold = ispval`. All other columns point to their downstream neighbor.

The uphill neighbor (`col%colu`) is computed by linear search — find the column
whose `cold` points to you. Each column has at most ONE uphill neighbor. The
chain is strictly linear: ridge -> ... -> lowest -> [stream].

**Reference:** HillslopeHydrologyMod.F90, lines 420-444

### 2.2 PFT and Soil Depth Assignment

Lowland vs. upland classification is based on **topology**, not elevation or
distance:

```fortran
if (col%cold(c) == ispval) then
   ! LOWLAND: deeper soil, lowland PFT
else
   ! UPLAND: shallower soil, upland PFT
endif
```

**Reference:** HillslopeHydrologyMod.F90, lines 711-730 (soil depth), 772-777
(PFT)

If the lake column has `cold == ispval`, it automatically gets lowland treatment.

### 2.3 NetCDF Dimension Requirements

- `nmaxhillcol` dimension must be 17 (CTSM reads this dynamically from
  `surfrdMod.F90:167` and allocates arrays accordingly)
- `nhillcolumns` variable must be 17
- All 17 column slots must have valid (non-NaN) data for all fields
- `hillslope_index` can be shared (all 17 columns can share index 1)
- `pct_hillslope` stays at `[100.0]` (single aspect)

---

## 3. Weight and Area Redistribution

The weight calculation (HillslopeHydrologyMod.F90:517-523):

```fortran
hillslope_area(nh) = sum of col%hill_area for all columns in hillslope nh
col%wtlunit(c) = (col%hill_area(c) / hillslope_area(nh)) * (pct_hillslope(l,nh) * 0.01)
```

With `pct_hillslope = 100%`, each column's weight is its fraction of total area.

**Current (16 columns):** Total area = land area only (~79.3 km²)
**With lake (17 columns):** Total area = land + lake (~79.3 + 10.7 = 90.0 km²)

Each existing column's weight decreases by `10.7 / 90.0 = 11.9%`. The lake
column gets `10.7 / 90.0 = 11.9%` weight. This is physically correct — the lake
IS 11.9% of the production domain.

The `AREA` variable in the NetCDF (total gridcell area) should reflect the full
domain.

---

## 4. Analysis of DTND Approaches

### 4.1 Why the Current Pipeline Gives DTND ~ 0 for Lakes

The pipeline computes HAND/DTND against a "wide mask" (natural streams + NWI lake
pixels). Lake pixels are IN the reference mask, so they get DTND near zero —
they're treated as drainage targets. Using `mean(dtnd[water_mask])` from this
computation gives ~0, which causes division by zero in the Darcy gradient.

### 4.2 Approach A: Recompute DTND Against Natural Stream Mask

Run `compute_hand()` a second time using only the ~552 natural stream reaches.
Lake pixels trace D8 flow to the nearest stream and get meaningful DTND values.

**Problem:** OSBS lakes are topographic depressions OFF the drainage axis. Their
DTND against the natural stream mask would reflect the distance from the lake to
the stream network via the flow path. For isolated depressions far from streams,
this could easily be 100-500m — larger than the lowest HAND bin's distance (~30m
in production output).

If `d(lake) > d(col1)`, the column-to-column gradient denominator
`d(col1) - d(lake)` goes **negative**, flipping the gradient sign. Water would
flow FROM the lake INTO the upland — the opposite of intended physics.

Even if verified for the current domain, there's no guarantee the values stay
below 30m for different domain sizes or mask configurations.

**Verdict:** Dangerous. Non-monotonicity risk is real and domain-dependent.

### 4.3 Approach B: Bump All Distances by an Offset

Add a constant delta to all 16 existing columns' distances. Set the lake column
at a small positive value (e.g., 5m).

**Algebra:** For column-to-column gradients, the offset cancels:
```
(d(c) + delta) - (d(cold(c)) + delta) = d(c) - d(cold(c))
```
All inter-column gradients are unchanged. This is exact.

**Where delta does NOT cancel:**
1. Lake-to-stream gradient: uses absolute distance. But this is clamped to zero
   anyway (routing off, empty stream).
2. Soil depth linear profile: `min_hill_dist` and `max_hill_dist` both increase
   by delta, so `max - min` is unchanged. The lake column (smallest distance)
   gets the deepest soil. Other columns' soil depths are unaffected.

**Verdict:** Safe but adds unnecessary complexity. The offset is a mathematical
no-op for inter-column gradients. The only thing that matters is the lake's
absolute distance, which is irrelevant when the lake-to-stream path is clamped.

### 4.4 Approach C: Small Fixed DTND (Recommended)

Set `hill_distance(lake) = hill_distance(lowest_land_column) / 2`.

With the lowest land column's distance at ~30m in the production output, this
gives ~15m. The lake column sits below that column in the chain. (Note on
indexing: in the recommended topology of Section 5.1, the lake is NetCDF
column 1 and the lowest land column is NetCDF column 2. "col1" in the Darcy
equations below refers to the lowest land column, not NetCDF index 1.)

**Why this works:**

1. **Monotonicity guaranteed by construction:** `15 < 30`, so
   `d(lowest_land) - d(lake) = 15m > 0`. No sign flip.

2. **Land-to-lake gradient denominator = 15m:** The gradient is
   `delta_head / 15m`. Physically reasonable for the distance between a lake edge
   and the nearest upland bin.

3. **Lake-to-stream gradient = 0:** With routing off and `tdepth(g) = 0`,
   the gradient is `(-0.4 - zwt) / 15`. This is negative, clamped to 0. **The
   absolute value of the lake's distance is irrelevant for this path.**

4. **No second-order effects:** If `soil_profile_set_lowland_upland` is used
   (topology-based), soil depth is unaffected. Met downscaling uses `hill_elev`,
   not distance. PFT assignment uses topology. Weights use area, not distance.

**The value is physically motivated:** Half the distance of the nearest upland
bin places the lake conceptually between the lowest land and the (inactive)
stream. For OSBS lakes surrounded by low-HAND terrain, this is within the right
order of magnitude (~tens of meters from lake edge to surrounding lowland).

---

## 5. Recommended Topology and Parameter Values

### 5.1 Column Chain

**Design decision (2026-04-23):** Place the lake at column index 1. Land
columns shift up by one — former col 1 becomes col 2, former col 16 becomes
col 17.

```
Col 17 (ridge) -> Col 16 -> ... -> Col 3 -> Col 2 (lowest land) -> Col 1 (lake) -> [stream]
```

- Column 1 (lake) has `downhill_column_index = -9999` (terminal, drains to stream)
- Column 2 (former col 1, lowest land) has `downhill_column_index = 1`
- Columns 3-17 chain as before (each points to index - 1)

**Rationale:** The Fortran loop iterates in column index order. Placing the
lake at col 1 ensures the `cold == ispval` column is processed FIRST in the
`SurfaceWaterMod` first loop. Subsequent columns reset the `wetlandisfull`
flag to `.false.`, preserving the current 16-column behavior (see Section 7.1).
Side benefit: the lake is the first slice along the column axis in the
NetCDF, easier for manual inspection.

### 5.2 Lake Column Parameters

| Field | Value | Rationale |
|-------|-------|-----------|
| `column_index` | 1 | First column; lake at bottom of chain (Section 5.1) |
| `downhill_column_index` | -9999 | Terminal column, drains to stream |
| `hillslope_index` | 1 | Same hillslope as all others |
| `hill_distance` | `hill_distance(lowest_land) / 2` | Monotonic by construction |
| `hill_elev` | `-SPILLHEIGHT` (-0.2m) | Runtime: -0.4m after SourceMod shift |
| `hill_area` | Sum(water_mask * pixel_area) | ~10.68 km² |
| `hill_width` | NWI lake perimeter (see 5.3) | Physical interface length |
| `hill_slope` | ~0.015 (see 5.5) | Bathymetry-derived bowl-shape slope (Lee et al. 2023) |
| `hill_aspect` | `hill_aspect(col2)` | Aggregate lake has no preferred direction; inherit from adjacent land |
| `hill_bedrock_depth` | 0.0 | Inert under Uniform soil profile (Section 5.4) |

### 5.3 hill_width

`hill_width` controls the cross-sectional area of the lateral flow interface at
the downslope edge of each column. For ALL columns (not just the lake), it
directly scales volumetric lateral outflow (line 2360):

```fortran
qflx_latflow_out_vol(c) = transmis * col%hill_width(c) * head_gradient
```

This applies regardless of whether routing is on or off. Width matters for every
column with a nonzero head gradient.

For the **lake column specifically**, the question is whether its gradient is
ever nonzero. The lake-to-stream gradient is clamped to zero only when
`stream_water_depth <= 0` (Section 6.3). If MOSART provides nonzero `tdepth(g)`,
the clamp does not fire and the lake's width enters an active calculation.

**Recommendation:** Use the total perimeter of all NWI lake polygons as the
lake's width — this is the physical interface length between lake and land. The
NWI shapefile has the polygon geometries. As a fallback, 1.0 is a safe
placeholder that produces negligible flow even if the gradient is nonzero, but
it is not physically motivated.

**Note:** Inflow FROM column 1 TO the lake (line 2389) uses column 1's width and
the lake's area. The lake's own width does not control how much water enters it
from uphill.

### 5.4 hill_bedrock_depth and the "Always Submerged" Constraint

The PI requires the lake column to be permanently submerged. Three mechanisms
interact:

**Surface ponding:** The spillheight mechanism (SurfaceWaterMod.F90:516)
prevents surface drainage when `hill_elev + h2osfc*1e-3 < 0`. With
`hill_elev = -0.4m` at runtime, the lake must accumulate >400mm of surface
water before any can drain. Spillheight guarantees the surface stays ponded.

**Subsurface lateral drainage:** The Darcy lateral flow can drain the lake's
soil column into the adjacent land column. If the lowest land column dries out
(zwt drops to 2m), its water table elevation
(`hill_elev - zwt = 0.018 - 2.0 = -1.98m`) is lower than the lake's
(`-0.4 - 0 = -0.4m`), and the gradient drives water FROM the lake INTO the
land column.

**Surface-to-soil infiltration (replenishment):** Ponded surface water
infiltrates down into the soil column via `qflx_h2osfc_drain`
(SoilHydrologyMod.F90:510):
```fortran
qflx_infl(c) = qflx_in_soil_limited(c) + qflx_h2osfc_drain(c)
```
This recharges the soil column from above at the same time the Darcy code may
be draining it laterally. The spillheight SourceMod only blocks *horizontal*
surface outflow, not *vertical* infiltration, so this pathway operates
alongside the surface ponding mechanism. For OSBS sandy soils with high
hydraulic conductivity, this infiltration is likely fast and should keep the
lake's soil saturated even during adjacent-column dry-down.

Whether additional enforcement is needed depends on what "always submerged"
means:
- **Surface always ponded:** Guaranteed by spillheight. No action needed.
- **Soil column always saturated:** Likely achieved by the infiltration vs
  lateral-drainage balance under OSBS conditions. Should verify empirically in
  the Phase G test run by monitoring lake-column `zwt` and `h2osfc`.

**`hill_bedrock_depth` as a saturation knob — not viable under current config.**
CTSM only reads `hillslope_bedrock_depth` from the NetCDF when
`hillslope_soil_profile_method = "FromFile"`. osbs4 uses `'Uniform'` (verified
in lnd_in:261), under which every hillslope column gets the same soil
thickness and `hillslope_bedrock_depth` is ignored. The Swenson reference
file and our production NetCDF both set it to 0 as an inert placeholder.
Switching to `FromFile` to use this knob would require setting bedrock depths
for all 16 land columns too — new complexity without clear payoff, especially
since the infiltration pathway above likely handles the saturation concern
natively.

**Recommendation:** Set `hillslope_bedrock_depth = 0` for the lake (consistent
with all land columns under Uniform). Verify saturation empirically in the
Phase G test. Revisit parameter-based enforcement only if the lake soil
column actually desaturates in practice.

### 5.5 hill_slope from Bathymetry (Lee et al. 2023)

The lake column is a saturated, always-underwater LAND column — the `hill_slope`
parameter describes the slope of the lake bed, not the flat water surface above
it. NEON 1m LIDAR cannot measure lake bed slope directly (water blocks laser
returns), so we need another data source.

**Source:** Lee, E., Epstein, A. & Cohen, M. J. (2023). *Patterns of Wetland
Hydrologic Connectivity Across Coastal-Plain Wetlandscapes*. Water Resources
Research, 59(8), e2023WR034553. Also summarized in the USDA NRCS CEAP
Conservation Insight "Storage and Release of Water in Coastal Plain
Wetlandscapes" (May 2024).

**Key OSBS bathymetry values from the paper (Table 1, OSBS column):**

| Quantity | Value | Notes |
|----------|-------|-------|
| Mean spill depth (basin bottom to rim) | 264.1 ± 95.0 cm | LIDAR-derived via Lane & D'Amico (2010) method |
| Mean wetland stage (standing water depth) | 177.5 ± 80.2 cm | Observed water level above bottom |
| Number of wetlands sampled | 14 | From 67-wetland study across 4 sites |

Qualitative: "Depressional wetlands are typically bowl-shaped with gradual
slopes from the wetland bottom to the spill elevation."

**Derivation of mean slope for the aggregate lake column.** For a bowl-shaped
basin with depth D and area A, the mean slope from rim to bottom is
approximately `D / R` where `R = √(A/π)` is the effective radius.

Using the aggregate of 103 NWI features covering 10.7 km²:
```
mean_area_per_feature = 10.7e6 / 103 ≈ 104,000 m²
effective_radius      = √(104000/π) ≈ 182 m
mean_slope            = 2.64 m / 182 m ≈ 0.0145 (≈ 1.5%)
```

**Recommended value:** `hill_slope(lake) ≈ 0.015`.

**Caveats:**

1. **Aggregate smoothing.** The 103 features span a size range. A small
   10,000 m² depression (R ≈ 56m, same 2.64m depth) gives slope ≈ 0.047. A
   larger 0.5 km² feature (R ≈ 400m, same depth) gives slope ≈ 0.007. The
   single aggregate value hides this variability. 1.5% is the central
   estimate but not the only defensible number.

2. **Sample size.** 14 wetlands sampled from OSBS may not be fully
   representative of all 103 NWI features. The spread (±95 cm on spill
   depth) is wide enough that the mean is the best estimator we have.

3. **Refinement possible.** Per-feature bathymetric estimates using the
   Lane & D'Amico (2010) LIDAR method on the actual NWI polygon set would
   give an area-weighted mean slope. Probably converges to a similar
   value given the wide variability. Worth doing if Phase G test results
   motivate it.

**Comparison to alternatives:**

| Source | Slope | Physical meaning |
|--------|-------|-----------------|
| Bathymetric (recommended) | 0.015 | Lake bed slope from bowl-shape + measured depth |
| Surrounding land (col2) | 0.050 | Upland sandhill terrain, not basin floor |
| Flat water surface | 0.000 | Water surface — wrong quantity for a submerged land column |
| min_hill_slope floor | 0.001 | Placeholder, no physics |

The bathymetric value is preferable because it's the only option with direct
physical support: measured depth + measured area → derived slope. The
surrounding-land value conflates basin and hillslope geology. The flat-water
value was from an incorrect framing (that the lake column represents the
water surface rather than the lake bed).

**Spill depth vs `hill_elev` — a separate concern.** The paper's 2.64m spill
depth describes the actual OSBS wetland storage capacity (basin bottom to
rim). Our current `hill_elev = -0.4m` (runtime, after the SPILLHEIGHT
SourceMod shift) allows only 0.4m of surface ponding before overflow — about
1/7th of the real capacity. For a column that's supposed to behave like an
OSBS wetland, this is a potential physics mismatch. See PI item #7.

---

## 6. HAND Contamination Analysis and Fill-Depth Diagnostic

### 6.1 What Actually Contaminates Low HAND Bins at OSBS

**Superseded framing (2026-04-21, incorrect):** "Ridgetop micro-depressions get
filled and assigned HAND near zero, polluting bin 1."

**Corrected framing (2026-04-23):** At OSBS's low relief (landscape span ~3-5m,
per CEAP paper Figure 2), a ridgetop pixel has real HAND of 3-5m whether or not
it's been pit-filled. Filling a 10cm dimple on the ridge doesn't move that pixel
into bin 1 (0-0.1m HAND) — it stays in the mid-slope bins. The "ridgetop
contamination" story was imported from mountainous terrain intuition and does
not apply here.

**The actual contamination source is unmapped wetland basins.** When a wetland
sits in a depression physically BELOW stream elevation (e.g., basin bottom at
26.3m when the local stream is at 27m), pit_fill / fill_depressions raises all
basin pixels to the basin's spill elevation (say 27.2m). The flow routes from
the spill point to the stream, so:

```
HAND_basin = spill_elevation - stream_elevation
           = 27.2 - 27.0 = 0.2m
```

That pixel is now mathematically "0.2m above drainage" and lands in bin 3, but
physically the pixel is INSIDE a closed basin whose bottom is below the stream.
The pixel's small positive HAND is an artifact of the fill operation, not a
reflection of real near-stream terrain.

**Scale of concern.** NWI captures the larger lakes and wetlands (→ wide mask →
handled separately). The remaining "unmapped" depressions are smaller basins
that the NWI inventory didn't include. Their contribution to bin 2/3 HAND
pixels depends on how many unmapped basins exist on the production domain —
unknown without the diagnostic.

### 6.2 Fill-Depth Decomposition (The Real Contamination Signal)

**Key insight (2026-04-23):** The pipeline produces FOUR distinct conditioning
stages, each with different physical meaning and magnitude. Treating `fill_depth
= inflated − dem` as one scalar conflates them and requires arbitrary
thresholds. Computing them separately gives a crisp, physically-meaningful
signal.

Pipeline conditioning order (`run_pipeline.py:802-839`):
```python
grid.fill_pits("dem", out_name="pit_filled")            # Stage 1
grid.fill_depressions("pit_filled", out_name="flooded") # Stage 2
flooded_arr[water_mask > 0] -= 0.1                      # Stage 3
grid.resolve_flats("flooded", out_name="inflated")      # Stage 4
```

Decomposed per-pixel signals:

| Stage | Quantity | Typical magnitude | Physical meaning |
|-------|----------|-------------------|------------------|
| 1. `fill_pits` | `pit_filled − dem` | mm to cm | LIDAR single-cell noise correction (benign) |
| 2. `fill_depressions` | `flooded_orig − pit_filled` | cm to meters | **Real basin filled to spill. Synthetic drainage. Contamination source.** |
| 3. NWI lowering | −0.1m for water pixels, 0 elsewhere | exactly −0.1m | Our routing trick (deterministic; we know who got what) |
| 4. `resolve_flats` | `inflated − flooded_lowered` | sub-mm | Microgradient tie-breaking (benign) |

**The crisp contamination signal is `depression_fill_depth > 0`** — it's a
yes/no answer to "was this pixel inside a closed basin that got filled to
rim?" No threshold needed at this level. Tiny values of
`depression_fill_depth` (say, < 1cm) could still be filtered as numerical
multi-pixel noise, but now the threshold is principled (it separates
numerical multi-cell noise from real basin fills).

**Pipeline change required for this decomposition.** The current pipeline
overwrites `grid.flooded` with the water-lowered version at line 816
(`grid.add_gridded_data(flooded_arr, data_name="flooded", ...)`). To recover
`flooded_orig`, we need to save a copy BEFORE the overwrite, OR reconstruct it
as `flooded_lowered + 0.1 * water_mask`.

**Pipeline history note:** our pipeline has a try/except around `resolve_flats`
(`run_pipeline.py:826-839`) because large open-water surfaces used to break it.
Since adding the NWI mask + 0.1m lowering, resolve_flats runs successfully
(verified in the 2026-04-15 production log, ~3 min runtime). If it ever falls
back, Stage 4 contributes zero — the decomposition still works cleanly.

### 6.3 Secondary Diagnostic Signals

Beyond fill-depth decomposition, several other signals help cross-check and
localize contamination:

1. **Raw vs. effective HAND discrepancy.** Compute
   `raw_hand = dem − dem[flow_termination_pixel]` alongside the pipeline's
   `hand = inflated − inflated[flow_termination_pixel]`. For unmodified pixels,
   they match. For depression-filled pixels, effective HAND is inflated above
   raw HAND. The difference equals `depression_fill_depth` for the source pixel
   — confirms the decomposition.

2. **HAND/DTND ratio.** A real near-stream pixel has small HAND AND small DTND.
   A filled-basin pixel has small HAND (by construction) but potentially large
   DTND (flow path goes up to spill, then down to stream). In bin 1,
   anomalously large DTND is a contamination red flag.

3. **Flow path length vs. Euclidean distance to nearest wide-mask pixel.**
   If D8 flow path is much longer than the straight-line distance, drainage is
   synthetic (routed the long way via spill point).

4. **Drainage catchment accumulation.** Pixels whose flow terminates in a tiny
   terminal catchment (low accumulation) are candidates for synthetic spill
   drainage. Well-accumulated catchments indicate real streams.

### 6.4 Proposed Diagnostic Workflow

**Plot first, decide later.** Do NOT pre-specify a threshold. Generate
diagnostic plots, look at what the data actually does, then choose how (or
whether) to filter.

**Phase 1: Decompose and plot fill components.** For all non-water land pixels
with finite HAND, produce log-y histograms of:
- `pit_fill_depth` (expect mostly zero with small tail)
- `depression_fill_depth` (expect mostly zero with fat tail — this is the
  primary signal)
- `resolve_flat_depth` (expect narrow sub-mm spike only)

If `depression_fill_depth` has a broad tail with significant pixel counts,
contamination is potentially important. If it's mostly zero, contamination
is minor.

**Phase 2: Stratify the bin 1 population.** Among pixels in bin 1 (0-0.1m
HAND):
- Count pixels with `depression_fill_depth == 0` (clean, real near-stream)
- Count pixels with `depression_fill_depth > 0` (depression-filled)
- Overlay DTND histograms of the two subpopulations

If DTND distributions overlap, the depression-filled pixels are actually close
to streams (benign — they're lowland with internal pits). If the depression-
filled subpopulation has DTND shifted much higher, that's the "basin far from
stream lifted to rim" case.

**Phase 3: Visual inspection of contamination.** Plot the production domain
as a 2D image colored by `depression_fill_depth`. Overlay the NWI water mask.
Expected patterns:
- Warm pixels clustered inside NWI polygons → expected (lakes partly filled)
- Warm pixels clustered OUTSIDE NWI polygons → unmapped wetlands (the real
  contamination)
- Scattered warm noise → random LIDAR artifacts (probably fine)

**Phase 4: Cross-check with HAND/DTND ratio.** For bin 1 pixels, compute
`ratio = DTND / (HAND + 0.01)`. Plot the distribution. A bimodal or
long-tailed shape indicates mixed populations.

**Phase 5: Decide the filter based on evidence.**

| What the diagnostics show | Recommended filter |
|---------------------------|-------------------|
| `depression_fill_depth` is mostly zero; DTND distributions overlap | No filter needed. Col 2 stats are robust. |
| Clear subpopulation has `depression_fill_depth > 0` AND elevated DTND | Filter on `depression_fill_depth == 0` for col 2 stats. Preserves unmodified lowland, removes basin artifacts. |
| Contamination is spatially clustered near unmapped wetlands | Consider a separate "near-unmapped-wetland" column rather than lumping into col 2 or the lake column. |

### 6.5 Why Decomposition Beats a Single Threshold

The original framing ("fill_depth > 0" or "fill_depth > 1cm") conflated all
four conditioning stages into one scalar and then tried to pick a threshold
separating "noise" from "real." That's guessing at signal boundaries.

Decomposition lets us answer the physical question directly: "was this pixel
depression-filled?" is a yes/no from `depression_fill_depth`. No threshold
needed at the first level. Only if we want to exclude numerical multi-cell
noise do we pick a threshold (e.g., `depression_fill_depth < 1cm = treat as
noise`), and that threshold is now physically meaningful because it's inside
a single stage with a clear semantics.

### 6.6 Open Questions

- **Do unmapped wetland basins actually exist in the OSBS production domain?**
  NWI is supposed to be comprehensive, but at 1m resolution there may be small
  seasonally wet depressions that NWI missed. This is partially answered by
  Quintero & Cohen 2019 ("Scale-dependent patterning of wetland depressions in
  a low-relief karst landscape") cited in the CEAP paper — worth following up.
- **Do we need to separately analyze water-adjacent pixels?** NWI water pixels
  get lowered (Stage 3) but may also sit inside basins that got filled (Stage
  2). The interaction could affect land pixels just outside lake polygons.
- **Computational cost of diagnostics.** Fill decomposition is cheap
  (subtraction on 90M-pixel arrays). Histogram/scatter plots are cheap. 2D
  spatial visualization is just an image plot. Runs in minutes.

### 6.5 Sequencing: Diagnostic → Bins → Lake

The recommended order of operations:

1. Generate the diagnostic plots (fill_depth histogram, HAND histograms split
   by fill_depth class, DTND verification)
2. Decide filtering strategy based on what the plots reveal
3. If filtering: rerun pipeline with the filter applied, confirm col 2 stats
   are stable
4. Finalize the binning scheme (how many bins, where the boundaries are)
5. Re-run the pipeline producing the final 16-column NetCDF
6. Append the lake column at col 1 with `DTND(lake) = DTND(col 2) / 2`

If the binning is changed after the lake column is added, the lake's DTND
must be recomputed. Better to stabilize col 2's stats (and bin boundaries)
first.

---

## 7. Other Pitfalls

### 7.1 wetlandisfull Flag Ordering (SurfaceWaterMod.F90:506-557)

**This is the most significant finding of the audit.**

The PI's SourceMod uses a scalar `wetlandisfull` flag to control surface water
redistribution. The flag is reset to `.false.` at the start of EACH column in
the first loop (line 506), and can be set to `.true.` on line 534 if the lowest
column's water surface exceeds the spillheight.

Only the **last column processed** retains its flag value into the second loop.

**Current behavior (16 columns):**
- Columns processed in index order: 1, 2, ..., 16
- Column 1 (the lowest, `cold == ispval`) is processed FIRST
- Even if column 1 sets `wetlandisfull = .true.`, columns 2-16 each reset it
- The flag is **always .false.** by the second loop
- **The flag is effectively dead.** The surface cascade always runs.

**With 17 columns:**
- Column 17 (lake, `cold == ispval`) is processed LAST
- If the lake sets `wetlandisfull = .true.` (requires `h2osfc > 400mm`),
  it persists into the second loop
- **All columns skip the net flux calculation** — the cascade is bypassed
- This is a behavior change that has never been triggered in the 16-column case

**Implications:**
- When the lake accumulates enough surface water (>400mm, which it eventually
  will since it's a permanent depression), ALL hillslope columns stop
  redistributing surface water and instead send everything directly to the stream
- This may suppress the ponding cascade that the PI's spillheight mechanism
  relies on
- OR: the PI may have intended this behavior — "when the wetland is full, stop
  cascading and drain everything to the stream"

**Resolution (2026-04-23):** Lake placed at column index 1 (land columns
shift to 2-17). The `cold == ispval` column is processed FIRST in the loop,
and subsequent columns reset the flag to `.false.`, preserving the current
16-column behavior exactly.

**Alternatives considered and rejected:**
- Lake at col 17: Would accidentally activate the dormant flag, changing
  behavior compared to the current 16-column baseline. Undesirable without
  PI validation that the changed behavior is intended.
- Modify the SourceMod to compute `wetlandisfull` once per landunit: Requires
  Fortran changes, which Phase G aims to avoid.

**Design rationale:** Low-risk pipeline-side fix that preserves the current
behavior without touching Fortran. Side benefit: the lake is the first slice
along the column axis in the NetCDF — easier to manually inspect. See
Section 5.1 for the resulting column chain.

### 7.2 Soil Depth Profile Method — Resolved

Verified (2026-04-23): `osbs4/CaseDocs/lnd_in` sets
`hillslope_soil_profile_method = 'Uniform'` (line 261). Under Uniform, every
hillslope column gets the same soil thickness — neither `hill_distance` nor
`hillslope_bedrock_depth` affects soil depth.

This resolves the concern in Section 1.2: the linear-profile interaction is
not active. The Swenson reference file's `hillslope_bedrock_depth = 0` is
inert under Uniform, and our lake column can use the same value without
effect on soil saturation (see Section 5.4).

### 7.3 MOSART Stream Depth and the Zero-Gradient Assumption — Resolved

The lake-to-stream gradient guard (`max(gradient, 0)` when `stream_water_depth
<= 0`) relies on `tdepth(g) = 0`. Verified (2026-04-23) that this holds for
osbs4.branch.v2 via a 5-step chain:

1. `nuopc.runconfig:111` sets `flds_r2l_stream_channel_depths = .false.`
2. MOSART's `rof_import_export.F90:95-98` makes the `Sr_tdepth`/`Sr_tdepth_max`
   exports conditional on that flag — with it false, the fields are NOT
   advertised to the mediator
3. CTSM's `lnd_import_export.F90:619` calls `fldchk(importState, Sr_tdepth)`
   which returns FALSE
4. CTSM falls through to the `else` branch: `tdepth_grc(:) = 0._r8` (line 624).
   Same for `tdepthmax_grc` (line 630). **Reset to zero every timestep.**
5. In `SoilHydrologyMod.F90:2270`, `stream_water_depth = tdepth(g) = 0`,
   triggering the guard on line 2284: `head_gradient = max(head_gradient, 0)`.
   Negative lake-to-stream gradients are clamped to zero.

**Consequence:** The lake's `hill_distance` value is mathematically irrelevant
for the lake-to-stream path — the gradient is always zero regardless of the
denominator. Water cannot flow from the "stream" into the lake.

**Why this is the case for osbs4:** The `flds_r2l_stream_channel_depths` flag
is documented in the mosart ChangeLog as specifically for the hillslope model
when routing is active. With `use_hillslope_routing = .false.` in osbs4, the
flag is left at its `.false.` default, and MOSART does not send stream depth
to the land model.

**Note on the subsurface Darcy head term.** The subsurface Darcy gradient uses
`hill_elev - zwt` as the head — it does NOT include ponded surface water
(`h2osfc`). A lake column with 400mm of standing water and a saturated soil
column (`zwt = 0`) has the same subsurface head (`-0.4m`) as one with no
standing water. The surface water exerts no hydraulic pressure on the
subsurface lateral flow. Surface and subsurface water are decoupled in CTSM.
This means the "zero gradient" claim only applies to the subsurface path.
Surface water flow is handled separately by the spillheight ponding logic in
SurfaceWaterMod.

Additionally, the subsurface Darcy gradient uses `hill_elev - zwt` as the head
term — it does NOT include ponded surface water (`h2osfc`). A lake column with
400mm of standing water and a saturated soil column (`zwt = 0`) has the same
subsurface head (`-0.4m`) as one with no standing water. The surface water
exerts no hydraulic pressure on the subsurface lateral flow. Surface and
subsurface water are decoupled in CTSM. This means the "zero gradient" claim
only applies to the subsurface path. Surface water flow is handled separately
by the spillheight ponding logic in SurfaceWaterMod.

### 7.4 Transmissivity and the Saturation Balance

When the lake column receives lateral inflow from the lowest land column, the
transmissivity calculation runs normally using the lake column's soil
properties. Under Uniform soil profile, the lake has the same soil thickness
as all other columns — `hillslope_bedrock_depth = 0` is inert and does not
restrict the soil column.

The inflow formula (line 2389) adds volume directly to the lake's
`qflx_latflow_in` — it's the UPHILL column's transmissivity and width that
control flow volume, not the lake's. The lake's transmissivity only matters
for its own outflow.

The lake's subsurface outflow is NOT necessarily zero — the column-to-column
gradient between the lake and the lowest land column can reverse when the
land column dries out. However, the surface-to-soil infiltration pathway
(`qflx_h2osfc_drain`, Section 5.4) provides a replenishment mechanism:
ponded water on the lake surface infiltrates down to refill the soil column.
For OSBS sandy soils, this is likely fast enough to maintain saturation.

The exact balance depends on the relative rates of:
1. Lateral Darcy outflow (lake → adjacent dry column)
2. Vertical infiltration from h2osfc (lake surface → lake soil)

Monitor `zwt(lake)` and `h2osfc(lake)` in the Phase G test run to confirm the
balance works as expected.

---

## 8. Items Requiring PI Clarification

### Active (need PI input)

| # | Question | Why It Matters |
|---|----------|----------------|
| 5 | What does "always submerged" mean? (5.4) | Surface-only (guaranteed by spillheight) vs. full soil saturation (likely maintained by surface→soil infiltration, but should verify in Phase G test) |
| 7 | Spill depth tuning (5.5) | Real OSBS wetlands have mean spill depth 2.64m (Lee et al. 2023); our current `hill_elev = -0.2m` (runtime -0.4m with SPILLHEIGHT=0.2) allows only 0.4m ponding before overflow. The PI may tune SPILLHEIGHT upward to match physical capacity. Our NetCDF convention (`hill_elev = -SPILLHEIGHT`) holds regardless of the tuned value. |

### Resolved during audit

| # | Question | Resolution |
|---|----------|------------|
| 1 | Lake-at-col-1 placement (7.1, 5.1) | **Design decision (2026-04-23):** lake at col 1, land at cols 2-17. Preserves current 16-column `wetlandisfull` behavior without Fortran changes. |
| 2 | `soil_profile_method` for osbs4 | **Uniform** (lnd_in:261). See Section 7.2. No distance/bedrock dependence on soil depth. |
| 3 | SPILLHEIGHT = 0.2m confirmed | Verified current value is 0.2m (HillslopeHydrologyMod.F90:55). The NetCDF convention `hill_elev = -SPILLHEIGHT` is a design choice; the current value produces `hill_elev = -0.2m` (runtime -0.4m). PI may tune SPILLHEIGHT later — see active item #7 for physical-capacity considerations — but the formulaic relationship holds. |
| 4 | `tdepth(g) = 0` for OSBS? | **Yes, guaranteed.** `flds_r2l_stream_channel_depths = .false.` in nuopc.runconfig:111 → MOSART does not export Sr_tdepth → CTSM resets tdepth_grc to 0 every timestep (lnd_import_export.F90:624). Empty-stream guard fires; lake-to-stream gradient clamped to zero. See Section 7.3. |
| 6 | `hill_bedrock_depth` for lake (5.4) | **Moot.** Field is ignored under Uniform soil profile. Set to 0, consistent with all land columns. Saturation enforcement, if needed, must come from elsewhere (infiltration likely handles it). |

---

## 9. References

| Document | Location |
|----------|----------|
| PI's osbs4 SourceMods | `$CASES/osbs4.branch.v2/SourceMods/src.clm/` |
| Upstream HillslopeHydrologyMod | `$BLUE/ctsm5.3/src/biogeophys/HillslopeHydrologyMod.F90` |
| Upstream SoilHydrologyMod | `$BLUE/ctsm5.3/src/biogeophys/SoilHydrologyMod.F90` |
| Upstream HillslopeHydrologyUtilsMod | `$BLUE/ctsm5.3/src/biogeophys/HillslopeHydrologyUtilsMod.F90` |
| Upstream TopoMod | `$BLUE/ctsm5.3/src/main/TopoMod.F90` |
| Phase G plan | `$SWENSON/phases/G-ctsm-lake-representation.md` |
| STATUS.md | `$SWENSON/STATUS.md` |

### Papers

- Lee, E., Epstein, A., & Cohen, M. J. (2023). Patterns of Wetland Hydrologic
  Connectivity Across Coastal-Plain Wetlandscapes. *Water Resources Research*,
  59(8), e2023WR034553. https://doi.org/10.1029/2023WR034553
- USDA NRCS (May 2024). Storage and Release of Water in Coastal Plain
  Wetlandscapes. Conservation Effects Assessment Project (CEAP) Conservation
  Insight. (Summary of Lee et al. 2023 for practitioners.)
  `~/CEAP-Wetlands-Conservation-Insight-WetlandscapeConnectivity-May2024.pdf`
- Lane, C. R., & D'Amico, E. (2010). Calculating the ecosystem service of water
  storage in isolated wetlands using LiDAR in North Central Florida, USA.
  *Wetlands*, 30(5), 967-977. https://doi.org/10.1007/s13157-010-0085-z
  (LIDAR spill-depth method cited by Lee et al.)
