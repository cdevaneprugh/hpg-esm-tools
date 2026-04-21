# Lake Column CTSM Audit: hill_distance and Implementation Viability

Date: 2026-04-21

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

**Relevance:** Only matters if `soil_profile_method == soil_profile_linear`
(method 3). If the PI uses `soil_profile_set_lowland_upland` (method 2), soil
depth is assigned by topology (`cold == ispval` = lowland), not distance, and
this calculation is never called. **Need to confirm which method osbs4 uses.**

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
| HillslopeHydrologyUtilsMod:66 | Soil depth linear profile | toosmall_distance | If linear method |
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

Set `hill_distance(lake) = hill_distance(col1) / 2`.

With column 1's distance at ~30m in the production output, this gives ~15m. The
lake column sits below column 1 in the chain.

**Why this works:**

1. **Monotonicity guaranteed by construction:** `15 < 30`, so
   `d(col1) - d(lake) = 15m > 0`. No sign flip.

2. **Col1-to-lake gradient denominator = 15m:** The gradient is
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

```
Col 16 (ridge) -> ... -> Col 2 -> Col 1 (former lowest) -> Col 17 (lake) -> [stream]
```

- Column 1's `downhill_column_index` changes from -9999 to 17
- Column 17 (lake) gets `downhill_column_index = -9999` (terminal)

### 5.2 Lake Column Parameters

| Field | Value | Rationale |
|-------|-------|-----------|
| `column_index` | 17 | Last column |
| `downhill_column_index` | -9999 | Lowest column, drains to stream |
| `hillslope_index` | 1 | Same hillslope as all others |
| `hill_distance` | `hill_distance(col1) / 2` | Monotonic by construction |
| `hill_elev` | `-SPILLHEIGHT` (-0.2m) | Runtime: -0.4m after SourceMod shift |
| `hill_area` | Sum(water_mask * pixel_area) | ~10.68 km² |
| `hill_width` | NWI lake perimeter (see 5.3) | Physical interface length |
| `hill_slope` | 0.0 | Flat lake; min_hill_slope floor in SurfaceWaterMod |
| `hill_aspect` | 0.0 | Irrelevant for flat surface |
| `hill_bedrock_depth` | TBD (see 5.4) | Affects subsurface saturation |

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

The PI requires the lake column to be permanently submerged. Two mechanisms
contribute to this:

**Surface water:** The spillheight mechanism (SurfaceWaterMod.F90:516) prevents
surface drainage when `hill_elev + h2osfc*1e-3 < 0`. With `hill_elev = -0.4m`,
the lake must accumulate >400mm of surface water before any can drain. This keeps
the surface ponded.

**Subsurface water:** The Darcy lateral flow can drain the lake's soil column
into adjacent columns. If column 1 dries out (zwt drops to 2m), its head
(`hill_elev - zwt = -0.1 - 2.0 = -2.1m`) is lower than the lake's head
(`-0.4 - 0 = -0.4m`), and the gradient drives water FROM the lake INTO column 1.
The lake's soil column can desaturate even while the surface stays ponded.

Whether this matters depends on what "always submerged" means:
- **Surface always ponded:** Guaranteed by spillheight. No parameter choice needed.
- **Entire soil column always saturated:** Not guaranteed by current physics.

`hill_bedrock_depth` controls how much soil the lake column has. A value of 0
gives a paper-thin soil column — there's almost nothing to desaturate, which
effectively forces full saturation by construction. But it also means incoming
lateral flow has nowhere to go except surface ponding, which could cause
numerical issues.

A full lowland depth (8m) gives the lake a normal soil column that can
participate in subsurface exchanges — more physically realistic, but the column
can partially desaturate under dry conditions.

**Recommendation:** Ask the PI. This interacts with the "always submerged"
requirement.

---

## 6. HAND Noise Floor and Fill-Depth Diagnostic

### 6.1 The Problem: Conditioned Pixels Contaminate Low HAND Bins

DEM conditioning (pit fill + resolve_flats) creates artificial near-zero HAND
values. A small depression on a ridgetop gets filled to its spill point, assigned
a micro-gradient by resolve_flats, and ends up with HAND near zero — even though
it's physically at high elevation, far from any stream. Its DTND could be 300m+,
but it gets averaged into the lowest HAND bin alongside genuinely low-lying
terrain at 10-20m DTND.

This produces a mixed population in bin 1: real lowland pixels and misclassified
ridgetop artifacts, with a blended mean DTND (~30m) that represents neither
population accurately. Since the lake column's DTND is derived from column 1's
DTND, this noise propagates directly into the lake's parameters.

### 6.2 Fill-Depth Diagnostic

The pipeline already has both the original DEM (`dem`) and the conditioned DEM
(`flooded`) in memory. The difference `fill_depth = flooded - dem` identifies
which pixels were modified:

- `fill_depth = 0`: Unmodified pixel. Drainage relationship is real.
- `fill_depth > 0`: Conditioned pixel. Drainage relationship is synthetic.

Not all conditioned pixels are noise — real depressions also get filled (lakes,
wetland basins, seasonal ponds). The fill depth distribution should show a heavy
spike near zero (resolve_flats micro-gradients) with a long tail (real
depressions). A fill depth threshold (e.g., 1cm) separates the two:

- **Noise population:** `0 < fill_depth < threshold` (numerical artifacts)
- **Real depressions:** `fill_depth >= threshold` (physical features, already
  handled by NWI mask for the large ones)

### 6.3 Determining the HAND Cutoff

1. Compute `fill_depth = flooded - dem` for all land pixels
2. Partition into unmodified (`fill_depth <= epsilon`) and conditioned
   (`fill_depth > epsilon`, where epsilon ~1mm for float tolerance)
3. Among the noise population (`0 < fill_depth < threshold`), compute the HAND
   distribution
4. The HAND cutoff = a high percentile (95th or 99th) of the noise population's
   HAND values
5. Any HAND bin whose upper bound falls below this cutoff is dominated by
   conditioned pixels and should not be treated as physical signal

This is more defensible than an arbitrary sacrificial bin or a fixed Q1 cutoff
because the threshold is derived from the conditioning itself — it identifies
which pixels' HAND values are trustworthy vs. synthetic.

### 6.4 Recommended Diagnostic Outputs

1. **Fill-depth histogram:** Distribution of `flooded - dem` for all land pixels.
   Expect a spike near zero and a tail. Identify the threshold between
   resolve_flats noise and real depressions.
2. **HAND histogram (full):** Distribution of HAND values for all land pixels,
   with the conditioned (noise) and unmodified populations shown separately.
3. **HAND histogram (0-2m zoom):** Same, zoomed to the TAI zone where bin design
   matters most.
4. **DTND histogram:** Distribution of DTND for the cleaned (noise-removed)
   population in the lowest HAND bins, as a verification that the remaining
   pixels have plausible lowland distances.

These diagnostics should be generated BEFORE finalizing the binning scheme, as
the noise floor determines the bin boundaries, which determine column 1's
parameters, which determine the lake column's DTND.

### 6.5 Sequencing: Bins Before Lake

The recommended order of operations:

1. Run fill-depth diagnostic to establish the HAND noise floor
2. Design the binning scheme (how many bins, where the boundaries are)
3. Run the pipeline with the new bins
4. Verify column 1's DTND is clean (tight distribution, no ridgetop artifacts)
5. Append the lake column with DTND = col1_distance / 2

If the binning is changed after the lake column is added, the lake's DTND must
be recomputed. Better to stabilize the bins first.

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

**Options:**
1. Place the lake at column index 1 (shift everything else up by 1) to preserve
   the current (ineffective) behavior
2. Accept the new behavior and discuss with PI
3. Modify the SourceMod to compute `wetlandisfull` once per landunit instead of
   per-column (requires Fortran changes, which Phase G aims to avoid)

**Recommendation:** Flag to PI. The current 16-column behavior suggests the flag
was not intended to work as designed. The 17-column case accidentally fixes the
ordering — whether that's desired is a PI decision.

### 7.2 Soil Depth Profile Method

If `soil_profile_method == soil_profile_linear` (method 3), the lake column's
small distance becomes the new `min_hill_dist`, changing the soil depth
interpolation slope for ALL columns. The lake gets the deepest soil; other
columns get slightly shallower soil.

If `soil_profile_method == soil_profile_set_lowland_upland` (method 2), no
effect — soil depth is assigned by topology (cold == ispval = lowland).

**Need to verify which method osbs4 uses.**

### 7.3 MOSART Stream Depth and the Zero-Gradient Assumption

The lake-to-stream gradient guard (`max(gradient, 0)` when `stream_water_depth
<= 0`) relies on `tdepth(g) = 0`. If MOSART somehow provides a non-zero stream
depth, the guard doesn't fire and the negative gradient drives water FROM the
stream INTO the lake. With routing off, there's no `stream_water_volume` to limit
this flow — the water is effectively created from nothing.

**Need to verify `tdepth(g) = 0` for the OSBS DATM configuration.**

Additionally, the subsurface Darcy gradient uses `hill_elev - zwt` as the head
term — it does NOT include ponded surface water (`h2osfc`). A lake column with
400mm of standing water and a saturated soil column (`zwt = 0`) has the same
subsurface head (`-0.4m`) as one with no standing water. The surface water
exerts no hydraulic pressure on the subsurface lateral flow. Surface and
subsurface water are decoupled in CTSM. This means the "zero gradient" claim
only applies to the subsurface path. Surface water flow is handled separately
by the spillheight ponding logic in SurfaceWaterMod.

### 7.4 Transmissivity of the Lake Column

When the lake column receives lateral inflow from column 1, the transmissivity
calculation runs normally using the lake column's soil properties. Since the lake
has `hill_bedrock_depth = 0`, the soil column is very shallow. This limits the
saturated thickness available for lateral flow, which could reduce the rate at
which water enters the lake subsurface.

However, the inflow formula (line 2389) adds the volume directly to the lake's
`qflx_latflow_in` — it's the UPHILL column's transmissivity and width that
control the flow volume, not the lake's. The lake's transmissivity only matters
for its own outflow.

Note that the lake's subsurface outflow is NOT necessarily zero — the column-to-
column gradient between the lake and column 1 can reverse when column 1 dries
out. See Section 5.4 for the "always submerged" implications.

---

## 8. Items Requiring PI Clarification

| # | Question | Why It Matters |
|---|----------|----------------|
| 1 | `wetlandisfull` ordering (7.1) | Adding col 17 as the lake accidentally enables a dormant flag. Behavior change. |
| 2 | `soil_profile_method` for osbs4 | Determines if lake's distance affects soil depth of all columns |
| 3 | SPILLHEIGHT = 0.2m? | Confirm SourceMod value matches our NetCDF assumption |
| 4 | `tdepth(g) = 0` for OSBS? | Ensures lake-to-stream gradient guard fires correctly; determines if lake width matters |
| 5 | What does "always submerged" mean? (5.4) | Surface-only (spillheight guarantees) vs. full soil saturation (not guaranteed). Determines bedrock_depth. |
| 6 | `hill_bedrock_depth` for lake (5.4) | 0 = thin soil, enforces saturation by construction but may cause numerical issues. 8m = full lowland soil, can partially desaturate. |

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
