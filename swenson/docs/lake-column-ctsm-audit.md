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
Revised: 2026-04-25 (PI notes incorporated: lake hill_distance ≈ stream
width, hill_slope = 0 with lake-bottom framing, hill_width = 1/2 NWI
perimeter. Items 5 and 7 resolved — don't enforce permanent submersion
parametrically; defer SPILLHEIGHT tuning to model output. Earlier 0.015
bathymetric slope and col2/2 distance superseded.)
Revised: 2026-04-25 (Section 6.7 added: the HAND binning fix proposal.
After diagnostic analysis, identified that compute_hand uses inflated DEM,
producing geometrically incorrect HAND values for ~25% of land pixels at
OSBS. Proposed fix: use raw DEM for HAND, keep inflated for routing.
Generalizes across all bins. Pending PI approval of architectural shift to
flood-zone bins.)
Revised: 2026-04-29 (Section 7.5 added: NWI mask has holes from nested-
polygon rasterization. ~18.5K hole pixels per major lake; ~50-200K total
domain-wide. Pipeline data impact small (88% of holes are NaN-HAND, filtered
out); visualization impact visible. Fix deferred to next pipeline rerun via
NWI re-rasterization with proper nested-polygon handling.)
Revised: 2026-04-30 (Section 6.7.3 corrected: the prior "NWI-interior fill
2.97m matches Lee 2023's 2.64m" framing was wrong — those measure different
quantities (storage above water vs. basin-bottom-to-rim). After reading
McLaughlin 2019 and Lee 2023 in full, the correct empirical comparison is
non-NWI basins ≥ 1 ha (n=107, mean 3.33m vs. Lee's 2.64m, within ~25%).
Section 5.5 and PI item #7 updated accordingly. McLaughlin 2019 added to
references.)

---

> **POST-PI-MEETING REDESIGN (2026-04-30) — sections referencing the
> spillheight convention are superseded.**
>
> The PI meeting reframed the lake column and FZ bin design. Key changes:
>
> 1. The PI's spillheight SourceMod is being retired (SPILLHEIGHT → 0).
>    The universal `-SPILLHEIGHT` shift in `HillslopeHydrologyMod.F90:363`
>    becomes a no-op.
> 2. Lake column `hill_elev` is no longer `-SPILLHEIGHT`. It's set from
>    basin-physics data (placeholder: -2.64 m per Lee 2023 or -3.33 m per
>    our pipeline). The constraint is that lake hill_elev must be more
>    negative than the deepest FZ bin to keep chain monotonicity.
> 3. The flood zone is the dry continuation of the same basin as the lake
>    (not a separate buffer). FZ bin count expands to ≥50% of total
>    columns per PI direction.
> 4. Sections 5.2, 5.4, 5.5, 6.7.x, 7.1, and PI items #3 / #5 / #7 below
>    that reference the spillheight convention should be read as historical.
>    They are retained for context but no longer describe the working plan.
>
> **Current working plan:** `phases/E.5-bin-redesign.md` and the
> Phase E.5 entry in `STATUS.md`. Phase G (`phases/G-ctsm-lake-representation.md`)
> is effectively folded into Phase E.5.

---

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

**PI direction (2026-04-25):** Set `hill_distance(lake) ≈ stream width` (a few
meters). The exact value is mathematically inert under the current
configuration (routing off, `tdepth=0` → empty-stream guard fires → lake-to-
stream gradient clamped to zero). Picking ~5m vs 15m vs 1m gives identical
model behavior. Choose stream width as a defensible small nonzero value.

The earlier `hill_distance(lowest_land) / 2` recommendation gave ~15m and is
still defensible, but stream width is what the PI prefers and the choice is
inconsequential.

**Why ANY small nonzero value works:**

1. **Monotonicity guaranteed by construction:** Any value smaller than
   `hill_distance(lowest_land)` (~30m) keeps the col-to-col denominator
   positive. 5m, 15m, anything in this range — all fine.

2. **Land-to-lake gradient denominator:** With lake DTND = 5m and lowest land
   DTND = 30m, denominator = 25m (vs 15m at the earlier recommendation).
   Both are physically reasonable.

3. **Lake-to-stream gradient = 0:** With routing off and `tdepth(g) = 0`,
   the gradient is `(-0.4 - zwt) / X` regardless of X. This is negative,
   clamped to 0. **The absolute value of the lake's distance is irrelevant
   for this path.**

4. **No second-order effects:** If `soil_profile_set_lowland_upland` is used
   (topology-based), soil depth is unaffected. Met downscaling uses `hill_elev`,
   not distance. PFT assignment uses topology. Weights use area, not distance.

**Concrete value:** look up the production NetCDF's stream width
(`hillslope_stream_width` or computed from network) and use that. Likely
~5m at OSBS.

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
| `hill_distance` | ~stream width (~5m) | PI direction (2026-04-25); inert under current config (Section 4.4) |
| `hill_elev` | **-6.0 m** | Locked 2026-05-04. Set deeper than the deepest land bin's mean (-5.13 m, bin 1 of 24-bin scheme covering -6.35 to -4.0 m raw HAND) by ~0.87 m to ensure chain monotonicity. Conceptually a chain-bookkeeping value, not a physical lake-bottom elevation. Empirical context: mean NWI lake-surface raw HAND in the production domain is -2.53 m; mean basin-bottom-to-rim spill depth is 2.64-3.33 m (Lee 2023 / our pipeline). Supersedes earlier `-SPILLHEIGHT` convention (now retired with SPILLHEIGHT=0 per Phase E.5 PI reframe 2026-04-30). See Section 5.2.1 below for derivation. |
| `hill_area` | Sum(water_mask * pixel_area) | ~10.68 km² |
| `hill_width` | 1/2 NWI total perimeter | PI direction (2026-04-25); see Section 5.3 |
| `hill_slope` | 0.0 | PI direction (2026-04-25); "lake bottom" framing — water surface is horizontal. See Section 5.5. |
| `hill_aspect` | 0.0 (or `hill_aspect(col2)`) | Aggregate lake has no preferred direction; either value is inconsequential |
| `hill_bedrock_depth` | 0.0 | Inert under Uniform soil profile (Section 5.4) |

### 5.2.1 hill_elev derivation (locked 2026-05-04)

The lake column's `hill_elev` is set to **-6.0 m** as a chain-bookkeeping
value rather than a physical lake-bottom elevation. The derivation
walks through three constraints and lands at -6.0 m by elimination
rather than direct calculation.

**Constraint 1 — Chain monotonicity.** CTSM hillslope routing assumes
water flows monotonically from high to low through the column chain
(`downhill_column_index`). The lake is at chain index 1 (audit
Section 5.1) — water flows FROM the chain INTO the lake. So the lake
must have the *most negative* `hill_elev` in the chain. The 24-bin
scheme's deepest land bin (Phase E.5, bin 1: deep tail sentinel
covering -6.35 to -4.0 m raw HAND) has mean = **-5.13 m**. Therefore
lake `hill_elev` < -5.13 m is required.

**Constraint 2 — Empirical lake geometry doesn't reach -5 m.** The
mean raw HAND of NWI water pixels in the production domain is **-2.53
m** (10.68 M pixels, computed via the full pipeline filter). Lee 2023
reports OSBS mean spill depth = 2.64 m ± 0.95 m (n=14 field-surveyed);
our pipeline measures 3.33 m mean for non-NWI basins ≥ 1 ha (n=107).
None of these reach -5 m. The deepest 5% of NWI water raw HAND values
sit at Q05 = -6.58 m, but that's a tail of LIDAR water-surface
observations, not a representative bottom.

**Constraint 3 — Don't reorder the chain or fold the deep tail
sentinel.** Both options were considered and rejected by PI direction
(2026-05-04). Lake stays at column index 1; the 24-bin scheme stays
intact; no fold-into-lake.

**Resolution.** With (1) requiring < -5.13 m and (2) saying physical
lake geometry doesn't reach that depth and (3) blocking other
restructuring, the lake `hill_elev` becomes a *chain-ordering value*
that doesn't map to any single physical lake's bottom. Pick a value
deeper than the deepest land bin with margin: **-6.0 m** (PI's
suggested starting value). Margin = 0.87 m below the deepest land
bin's mean.

**What -6.0 m does NOT represent:**

- It is *not* the mean lake-bottom elevation in OSBS (that would be
  closer to -2.6 to -3.3 m by Lee / our pipeline).
- It is *not* derived from the spill-depth + lake-surface arithmetic
  (`-2.53 - 2.64 = -5.17 m` was considered but conflated rim-to-bottom
  with surface-to-bottom and landed in the chain monotonicity floor by
  coincidence rather than derivation).
- It is *not* the deepest pixel in NWI water (-11.6 m) or the Q05 of
  NWI water raw HAND (-6.58 m) — those characterize tails, not
  representatives.

**What -6.0 m IS:**

- A chain-bookkeeping reference that satisfies CTSM's monotonicity
  requirement.
- A value that allows water from the deepest land bin (mean -5.13 m,
  containing pixels in unmapped deep depressions) to flow downhill
  into the lake column at a positive Darcy gradient.
- A starting point. May be tuned after model output review.

**Tuning context.** If the lake column behavior in CTSM output suggests
the value is wrong (e.g., unrealistic ponding heights to reach
spillover, water table dynamics not matching real OSBS lakes,
lake-to-stream lateral flow misbehaving), we revisit. The chain
monotonicity constraint pins the value at < -5.13 m as a hard floor;
above that floor the choice is empirical/iterative.

**Spillheight context.** The earlier framing where `hill_elev =
-SPILLHEIGHT` (and SourceMod shifted it further negative at runtime)
is fully retired per the Phase E.5 reframe (2026-04-30). SPILLHEIGHT
is set to 0 in the SourceMod (effectively disabling the universal
shift). The NetCDF lake `hill_elev = -6.0 m` is the runtime value
directly — no SourceMod shift applied.

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

**PI direction (2026-04-25):** Use **1/2 the total NWI perimeter** as the
lake's width. Heuristic — captures the effective lateral exchange surface
(perimeter accounts for both inflow and outflow sides; half captures the
downslope-facing interface).

**Inert under current configuration.** With `use_hillslope_routing = .false.`
and `tdepth(g) = 0`, the lake-to-stream gradient is clamped to zero
(Section 7.3), so the lake's hill_width does NOT affect outflow. Inflow
FROM the adjacent land column TO the lake (line 2389) uses the **uphill
column's** width, not the lake's. So lake hill_width is essentially
inconsequential for our config. This makes the choice low-stakes; any
defensible value works.

**Practical computation:**

1. Load NWI shapefile, dissolve overlapping polygons (handles nested-polygon
   case where lakes have inner open-water rings within outer wetland
   polygons)
2. Compute total exterior-ring length across dissolved polygons
3. `hill_width(lake) = total_perimeter / 2`

**Computed value (2026-04-25):** Using a boundary-pixel approximation on
the NWI raster mask (rather than the polygon shapefile), the total perimeter
is approximately **102,810 m** (102,810 boundary pixels × 1 m per pixel).
Half of that gives **`hill_width(lake) ≈ 51,405 m`**. This is consistent
with the expected scale: 103 NWI features over 10.68 km² with subcircular
geometry would give roughly ~117 km of total perimeter for compact circles;
our 103 km is in the right range for slightly-irregular wetland polygons.

The boundary-pixel approximation uses scipy's `binary_erosion`: boundary
pixels are mask pixels whose erosion removes them (i.e., they have at least
one non-mask 4-neighbor). Each boundary pixel contributes ~1 m of perimeter
at our 1m resolution. This slightly underestimates the true polygon
perimeter (which would account for diagonal edges) but is within ~5-10% of
the polygon-shapefile-derived value and avoids the shapely dependency.

**Sanity check via P:A ratio.** Compute perimeter:area ratio per dissolved
polygon and per the aggregate. Compare to circle reference
(`P:A = 2/√(A·π)`). For OSBS sandhill wetlands (typically subcircular
sinkhole-like shapes), P:A should be moderate (within ~2-3× circle
reference). Extreme P:A (very high) would indicate dendritic shapes that
might motivate revisiting; OSBS lakes are unlikely to be in this regime.

**Note for future:** if we ever switch routing on, lake hill_width becomes
consequential and this choice should be revisited.

### 5.4 hill_bedrock_depth and the "Always Submerged" Constraint

**PI direction (2026-04-25):** Don't worry about lakes drying out cyclically.
Real OSBS lakes do dry out periodically over decadal/centennial cycles —
forcing permanent submersion would be wrong physically. Run with reasonable
parameters and watch model output. If vegetation patterns are clearly wrong
(e.g., pine trees colonizing the lake permanently), revisit. Otherwise,
occasional drying is correct hydrology.

This resolves PI item #5: **don't enforce permanent submersion
parametrically.** Use `hill_bedrock_depth = 0` consistent with land columns
and let CTSM physics produce realistic intermittency.

The earlier analysis below describes the three mechanisms that interact for
saturation. The takeaway: spillheight handles surface ponding, infiltration
handles soil recharge, and lateral drainage during dry periods is a feature
not a bug.

---

The PI's earlier framing (now superseded): require permanent submersion.
Three mechanisms interact:

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

### 5.5 hill_slope = 0 (lake-bottom framing)

**PI direction (2026-04-25):** Set `hill_slope(lake) = 0`. The lake column
represents a "lake bottom" — a permanently-ponded soil column with a
horizontal water surface above it. The water surface IS horizontal, so the
column's effective surface (the air-facing interface) has slope = 0.

This supersedes the earlier bathymetric-slope recommendation (~0.015 from
Lee et al. 2023 spill depth + bowl geometry). Both interpretations are
defensible:

- **Bathymetric** (~0.015): represents the inclination of the soil surface
  beneath the water. The "underwater terrain" framing.
- **Lake-bottom / water-surface** (0): represents the inclination of the
  air-facing surface, which is horizontal water.

The PI's lake-bottom framing aligns better with how `hill_slope` is used in
CTSM:

| Usage | Effect of slope = 0 |
|-------|---------------------|
| Surface outflow rate `k_wet = 1e-4 * max(slope, min_hill_slope)` | At minimum floor (1e-3), giving ~8.6 mm/day. For a lake the relevant outflow is spillheight-driven, not slope-driven, so this is fine. |
| Microtopography `micro_sigma` | Small sigma → low subgrid variability → high h2osfc fractional coverage. Correct for a lake. |
| Insolation angle | Horizontal water surface receives solar at horizontal incidence. slope=0 is more physically accurate than 0.015. |
| Kinematic Darcy (not active in osbs4) | Would use slope directly as gradient. Not relevant. |

**Recommended value:** `hill_slope(lake) = 0`. The min_hill_slope floor in
SurfaceWaterMod handles the surface-outflow edge case automatically.

**Note on Lee et al. 2023 and SPILLHEIGHT tuning.** Lee's mean OSBS spill
depth of 2.64 m (n = 14) and our pipeline's 3.33 m (n = 107 non-NWI basins
≥ 1 ha; see Section 6.7.3) both characterize basin-bottom-to-rim depth.
For SPILLHEIGHT tuning, the relevant quantity depends on what the lake
column is meant to represent: if it starts empty, basin-bottom-to-rim
(~2.6–3.3 m) is the right scale; if it starts at typical OSBS NWI water
levels, storage-above-water (our ~4.7 m for NWI-overlapping basins ≥ 1 ha)
is the right scale. Either interpretation suggests SPILLHEIGHT in the
1–3 m range — at least 5× the current 0.2 m default. Tuning is deferred
to model output per PI direction (item #7); these references support that
discussion when it happens.

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

### 6.6 Open Questions (historical, mostly resolved by Section 6.7 analysis)

- **Do unmapped wetland basins actually exist in the OSBS production domain?**
  YES — confirmed by the 2026-04-24 diagnostic. ~10.5M land pixels are inside
  depressions outside the NWI mask (~12% of the production domain). NWI is
  not comprehensive at 1m resolution.
- **Do we need to separately analyze water-adjacent pixels?** Less critical
  than initially thought; the fill-depth decomposition cleanly separates them.
- **Computational cost of diagnostics.** Confirmed cheap. Production-domain
  diagnostic ran in ~3 minutes after the pipeline data was saved.

### 6.7 The HAND Binning Fix (Pending PI Approval)

**Discovery:** After running the diagnostic and analyzing the 2026-04-24
production output (Section 6.2-6.5 + summary.json), we identified that the
pipeline's `compute_hand` produces geometrically incorrect HAND values for
pixels inside filled basins. This is more fundamental than the
"contamination" framing in Section 6.1 — it's a structural mismatch between
Swenson's HAND convention and our high-resolution low-relief application.

#### 6.7.1 The geometric issue

`compute_hand` uses the inflated (post-conditioning) DEM. After
`fill_depressions`, every pixel inside a closed basin has been raised to that
basin's spill elevation. Consequently, every pixel in a single basin — from
the floor to the rim — is assigned the SAME pipeline HAND value (= basin
spill elevation − stream elevation).

Concrete example: basin floor at raw elevation 26.3m, spill at 27.2m, stream
at 27.0m.

| Pixel | Raw elevation | Inflated elevation | Pipeline HAND | Bin (current scheme) | True HAND |
|-------|---------------|-------------------|---------------|----------------------|-----------|
| Basin floor | 26.30m | 27.20m | 0.20m | bin 3 | **−0.70m** |
| Mid-basin | 26.80m | 27.20m | 0.20m | bin 3 | **−0.20m** |
| Near-rim | 27.15m | 27.20m | 0.20m | bin 3 | +0.15m |

All three pixels get binned identically despite spanning ~85cm of real
elevation. Their physical relationship to the stream — submerged, just
below, or just above — is lost.

#### 6.7.2 Mathematical formulation

For any pixel:
```
pipeline_HAND = inflated_DEM[pixel] − inflated_DEM[stream]
              = (raw_DEM[pixel] + depression_fill_depth) − raw_DEM[stream]
              ≈ raw_HAND + depression_fill_depth
```

The approximation is exact for stream pixels with `depression_fill = 0`
(most of them). About 88K stream pixels have small pool fills, introducing
sub-cm error.

Inverting:
```
raw_HAND = pipeline_HAND − depression_fill_depth
```

This recovers the true elevation of each pixel above its stream termination,
using arrays we already compute in the pipeline.

#### 6.7.3 Diagnostic findings that exposed this

From the 2026-04-24 production diagnostic run (90 tiles, 90M pixels):

| Population | Count | Fraction |
|-----------|-------|----------|
| Total land pixels | 76.6M | 85% of domain |
| Land pixels inside filled basins (`depression_fill > 0`) | 19.6M | 25.6% of land |
| Bin 1 land pixels (current pipeline HAND ≤ 0.1m) | 16.7M | 18.6% of land |
| Bin 1 hot (`depression_fill > 0`) | 15.6M | 93.1% of bin 1 |
| Bins 2-15 hot | ~3.9M | varies 4-19% per bin |

Hot pixel `depression_fill_depth` distribution: median 0.82m, mean 1.56m,
q95 5.87m. For most hot pixels in low-HAND bins, the raw HAND is **negative**
— they sit physically below stream level.

**Empirical sanity check against Lee et al. 2023.** Per-basin spill-depth
statistics (max `depression_fill_depth` within each connected component of
`depression_fill > 0`) computed from the 2026-04-24 diagnostic arrays:

| Population | n | Mean spill depth | Median |
|------------|---|------------------|--------|
| Non-NWI basins ≥ 1,000 m² | 521 | 1.35 m | 0.68 m |
| **Non-NWI basins ≥ 1 ha** | **107** | **3.33 m** | **1.88 m** |
| Non-NWI basins ≥ 10 ha | 10 | 8.06 m | 8.59 m |
| NWI-overlapping basins ≥ 1 ha | 32 | 4.70 m | 5.03 m |

Non-NWI basins are entirely-dry depressions where the NEON LIDAR captured
the actual bed, so `max(depression_fill)` measures basin-bottom-to-rim
depth. NWI-overlapping basins contain water at flight time, so their
`depression_fill` measures storage capacity above the LIDAR-observed water
surface — a different quantity, not directly comparable to bottom-to-rim
spill depth.

Lee et al. 2023 reports a mean OSBS spill depth of **2.64 m (± 0.95 m SD,
n = 14)**, computed via McLaughlin et al. 2019's LIDAR methodology: ArcMap
Fill against field-surveyed PVC well bottoms in the deepest part of each
wetland, with dry-season LIDAR flights and a 5 m local-minima resample to
strip vegetation artifacts in dome centers. That measurement is
basin-bottom-to-rim, with the bottom datum *field-measured* (not
LIDAR-derived).

Our 3.33 m for non-NWI basins ≥ 1 ha is the methodologically equivalent
quantity (LIDAR captures dry bed, fill operation finds spill elevation),
and it sits within ~25% of Lee's 2.64 m. The 1.35 m / 1.88 m figures for
the broader ≥ 1,000 m² bracket and the 3.33 m / 1.88 m for the ≥ 1 ha
bracket together bracket Lee's reported value across plausible
size-distribution choices. The agreement validates that our pipeline's
basin-fill geometry is physically reasonable; it does not validate using
that fill output as a HAND value, which is the mis-placement issue Section
6.7 fixes.

**Vintage caveat (2026-04-30).** Our 3.33 m measurement is computed from
NEON DP3.30024.001 OSBS 2023-05 (late dry season — favorable for capturing
dry beds). Lee 2023's 2.64 m vintage is ambiguous — the paper cites NCALM
generically but doesn't specify which dataset was used for the OSBS subset.
If Lee used the 2010 NCALM Optech Gemini (a peak-wet-season flight), their
measurement may underestimate true depths in inundated wetlands; if Lee
used the 2018 USGS Florida Peninsular Putnam dataset (dry-conditions
design), the comparison is direct. See `docs/data-acquisition-dates.md`
for full provenance and the action item to resolve via Cohen.

**What was wrong with the earlier framing.** A prior revision (2026-04-25)
claimed "NWI-interior mean fill = 2.97m matches Lee's 2.64m." That
comparison was between two different physical quantities — NWI-interior
fill measures storage above current water surface (the LIDAR-observed
ceiling at flight time was the water itself), while Lee's 2.64 m measures
basin-bottom-to-rim. The numerical proximity was coincidental. The
non-NWI ≥ 1 ha comparison (3.33 m vs. 2.64 m) is the correct one because
both reference the dry bed.

#### 6.7.4 Why this is resolution-dependent

At MERIT 90m, basins are rare and shallow:
- A typical depression spans 1-2 cells; fill depth ~ a few meters
- Most pixels are real terrain with no fill applied
- Raw vs. inflated HAND differ negligibly (Swenson audit 2026-02-20:
  height correlation delta 0.002, slope 0.014, area fraction 0.007 — all
  within rounding)

At OSBS 1m, basins are common and large:
- A typical wetland spans hundreds to thousands of pixels
- Mean fill depth 1.56m; q95 5.87m
- 25.6% of land has depression_fill > 0
- Raw vs. inflated HAND differ by meters for ~17% of the domain

Swenson's "use inflated DEM" choice (rh:1685) was made under conditions
where it didn't matter. We're applying it where it does.

#### 6.7.5 Cross-bin implications

The fix doesn't just affect bin 1. Every depression-filled pixel — in any
bin — gets re-binned by raw HAND. Whether each pixel "stays" or "moves" to
a lower bin depends on whether its `depression_fill_depth` is smaller than
its current bin's width:

| Bin range (current) | Hot pixels | Bin width | Likely fate |
|---------------------|-----------|-----------|-------------|
| Bin 1 (0-10cm fixed) | 15.6M | 10cm | nearly all move to flood zone (median fill 82cm >> 10cm) |
| Bins 2-5 (10cm fixed each) | ~790K combined | 10cm each | nearly all move to flood zone |
| Bins 6-8 (log, 21-43cm wide) | ~960K combined | 21-43cm | most move; some stay if fill < bin width |
| Bins 9-15 (log, 0.6-5m wide) | ~1.7M combined | 0.6-5m | mixed; smaller fraction needs to move |

Total ~3.9M hot pixels in bins 2-15, in addition to 15.6M in bin 1. The fix
cleans the entire bin structure, not just the bottom.

#### 6.7.6 Proposed fix

Use the raw DEM for HAND values; keep the inflated DEM for routing.

| Operation | DEM source | Reason |
|-----------|-----------|--------|
| D8 flow direction | Inflated | Required for valid downhill paths |
| Flow accumulation | (derived from above) | Used for stream network identification |
| Drainage termination per pixel | (derived from above) | Used for HAND/DTND tracing |
| **HAND value (binning + hill_elev)** | **Raw** | Physically correct elevation above drainage |
| DTND (lateral path length) | (from inflated routing) | Already correct; not affected |
| Slope, aspect | NEON DP3.30025.001 | Independent products |

The two operations — routing (which paths exist) and elevation differencing
(how much vertical drop) — are logically separable. The pipeline currently
ties them by using the same DEM for both. The fix decouples them.

#### 6.7.7 Implementation

Single-line change in `run_pipeline.py`, after `compute_hand`:
```python
grid.compute_hand("fdir", "inflated", wide_channel_mask, ...)
hand_inflated = np.array(grid.hand)
depression_fill = np.array(grid.flooded_orig) - np.array(grid.pit_filled)
hand_raw = hand_inflated - depression_fill
# use hand_raw downstream for binning and write to NetCDF
```

`flooded_orig` is currently saved only in diagnostic mode. Either keep it
in memory always (without saving) or always save it. Trivial either way.

The `hand_raw` array is then used wherever the current code uses
`grid.hand` for binning and `hill_elev` computation.

#### 6.7.8 New bin structure (proposed)

The current 1×16 hybrid scheme (5 fixed 10cm + 10 log + 1 sentinel) was
designed assuming pipeline HAND. With raw HAND, the distribution shifts
substantially:
- Negative raw HAND becomes a sizable population (~15-19M pixels)
- Current bin 1 shrinks to ~1.5M pixels — the **True B1** that should have
  been there all along, finally separated from depression-fill contamination
- Higher bins are mostly unchanged

Provisional new structure (concrete bin edges TBD from raw HAND distribution
analysis; numeric labels avoid semantic ambiguity from the earlier
DEEP/MOD/SHALLOW labels which became awkward when discussing OSBS-specific
geometry):

```
Lake column          (NWI permanent water,     hill_elev = -SPILLHEIGHT)

Flood-zone bins      (raw HAND < 0, dry land below stream level):
  Setup choices ranging from 1 (single FZ bin) up to ~8 (equal-count
  quantile bins, FZ1=deepest through FZ8=shallowest). Each bin has its own
  hill_elev (negative), DTND, area, etc. PI input on the right granularity
  is the next step.

True B1              (raw HAND in (0, 0.1m]):
  - was bin 1 in the existing scheme; now shrunk to ~1.5M genuinely
    near-stream-or-near-shoreline pixels
  - represents what bin 1 should have been all along

Higher bins          (raw HAND > 0.1m):
  - approximately current bins 2-16
  - small minority of hot pixels move to flood zone after fix
```

This gives the lake column a defensible narrow scope (only NWI water),
populates the flood zone with unmapped depression pixels distributed by
their physical depth, and preserves upland bin structure.

#### 6.7.9 Methodological positioning

The fix is a **departure from Swenson's literal code** but **consistent with
Swenson's intent**.

Swenson's pipeline includes a `fflood` mechanism (rh:678-697) that excludes
lake pixels from the lowest 2m HAND bin. This is the same kind of correction
we're proposing, applied to a different population:

| Aspect | Swenson | Our proposal |
|--------|---------|-------------|
| Signal | Binary lake mask (slope ≈ 0) | Continuous `depression_fill_depth` |
| Action | Set HAND = −1 (excluded from binning) | Use raw_HAND directly (move pixel to its physical bin) |
| Population addressed | Lakes (open water) | Lakes + non-lake basin interiors |
| Resolution scale | Appropriate at MERIT 90m | Appropriate at 1m |

Swenson's mechanism handles the dominant problematic population at his
resolution. We're using a richer signal (continuous fill depth instead of
binary mask) to capture the additional populations that emerge at 1m.

The fix also brings us closer to the original physical concept of HAND
("height above drainage") rather than its computational approximation
("inflated elevation difference").

#### 6.7.10 Defensibility argument

- We're computing HAND on the raw DEM, which is what HAND nominally means
- Routing remains on the inflated DEM (required by D8)
- The two operations have always been logically separable; we're correcting
  an arbitrary coupling
- The 25.6% of land affected at OSBS is a resolution-driven phenomenon
  Swenson didn't encounter; the fix addresses it without inventing new
  methodology
- The fix is documented and motivated by direct quantitative evidence from
  the 2026-04-24 diagnostic
- We extend Swenson's existing `fflood` exclusion mechanism rather than
  introducing a new concept

#### 6.7.11 Implementation impact

- Single pipeline change (one subtraction line + bin-edge re-derivation)
- Pipeline rerun required (~25 min on production domain)
- Diagnostic script remains valid; numbers change but interpretation logic
  is the same
- NetCDF schema unchanged; only values shift
- Lake column scope tightens to NWI water only (reverses earlier "expand
  lake column to absorb depression pixels" suggestion)

#### 6.7.12 Pending decisions before implementation

1. **PI approval of the architectural shift** to flood-zone bins
2. **Concrete bin edges for the negative-HAND range.** Depends on the
   raw HAND distribution; analysis forthcoming (~10s of seconds on saved
   diagnostic arrays)
3. **Total bin count.** Current is 1×16; proposed is 1×N where N = (5 flood
   zone) + (current bins 1-16 effectively cleaned and possibly compressed) ≈
   18-20
4. **Lake column scope.** Stays NWI-only under this proposal (reverses the
   "expand to all submerged" framing from earlier)
5. **Whether to compute and persist `flooded_orig` in production runs**
   (vs. only diagnostic mode). Trivial either way; choose based on whether
   we want the full conditioning chain reproducible from saved files.

### 6.8 Sequencing: Diagnostic → Fix → Bins → Lake

(Renumbered from 6.5; original sequencing was written before the Section 6.7
findings.)

Updated order of operations:

1. **Diagnostic** — DONE 2026-04-24. Production pipeline run with
   `SAVE_DIAGNOSTICS=1`; standalone script generated 6 plots + summary JSON.
2. **Interpret results** — DONE 2026-04-25. Sections 6.7.3 and 6.7.5 record
   the findings that exposed the geometric issue.
3. **Compute raw HAND distribution from saved arrays** (~10 sec). Determine
   concrete bin edges for the negative-HAND range. Pending.
4. **PI sign-off** on the architectural shift (flood-zone bins + raw HAND
   binning). Pending.
5. **Implement the fix** in `run_pipeline.py` (single subtraction line +
   bin-edge constants update). Trivial diff.
6. **Rerun the pipeline** with corrected HAND values. ~25 min on production
   domain.
7. **Verify new pipeline output** by re-running the diagnostic script —
   bin populations should match expected post-fix distribution.
8. **Append lake column** at col 1 with parameters per Section 5.2 (PI
   direction). The lake column scope stays NWI-only; flood-zone pixels are
   handled by the bin restructure rather than absorbed into the lake.
9. **Phase F validation run** with the corrected NetCDF.

If the binning scheme is later modified, the lake column parameters per
Section 5.2 are mostly insensitive (`hill_distance ≈ stream width` is inert,
`hill_width = 1/2 NWI perimeter` is independent of bins). So the lake
column can be appended after binning is finalized without rework.

### 6.9 Pixel Categories and Binning Filter (raw-HAND scheme)

Reference table for what each pixel class looks like under the
wide-mask HAND scheme, and the corresponding pipeline filter recipe
for raw-HAND binning.

#### 6.9.1 Pixel categories

In `run_pipeline.py`, `compute_hand` is called with `wide_channel_mask`
(natural streams + NWI water pixels), so all wide-channel pixels get
`hand = 0` by construction. The categories below classify pixels by
their `water_mask`, `channel_mask`, and conditioning history.

| Category | water_mask | channel_mask | hand | dep_fill | raw_hand | In a hillslope bin? |
|----------|------------|--------------|------|----------|----------|---------------------|
| NWI lake interior | 1 | 0 | 0 | typically > 0 | < 0 (large negative) | No — lake column handles |
| Real stream (not in filled basin) | 0 | 1 | 0 | 0 | 0 | No — channel |
| Real stream inside a filled depression | 0 | 1 | 0 | > 0 | < 0 | No — still a channel |
| Upland | 0 | 0 | > 0 | 0 | > 0 (= hand) | Yes — upland bin |
| **Basin floor / TAI** (filled depression interior, dry continuation of lake basin) | 0 | 0 | small ≥ 0 | > 0 | spans negative through near-zero into slightly positive depending on local fill depth | Yes — TAI/FZ bin |

`raw_hand = hand - dep_fill`, where `dep_fill = flooded_orig - pit_filled`.

#### 6.9.2 TAI is emergent, not a fixed boundary

The Terrestrial-Aquatic Interface (TAI) is a continuous dynamic zone
that straddles `raw_hand = 0`. Bins on both sides of zero capture it:

- Deeply negative raw HAND — wet most of the time (basin floor near
  lake).
- Near zero (either side) — water table oscillates seasonally.
- Slightly positive (~0 to ~2 m raw HAND) — dry most of the time but
  inundates seasonally; CTSM's water table can rise into this band.

The TAI is therefore an *emergent property* of the model's water-table
dynamics, not a topographic threshold we define a priori. The
column-chain transition at `raw_hand = 0` is just a binning convention.

**Practical implication for bin design:** the immediate upland band
(0 to ~2 m raw HAND) is as TAI-relevant as the FZ side. Bin density
there should reflect that — not be sacrificed to give resolution to
the high ridge, where the model rarely interacts with the water table.
The Phase E.5 candidate bin schemes (3c, 3d, 3e, 3f) all concentrate
edges in this band; preserving that density across iteration matters.

#### 6.9.3 Filter recipe for raw-HAND binning

The existing pipeline binning filter `(hand > 0) & np.isfinite(hand)`
implicitly excludes both stream channel pixels and lake pixels because
both have `hand = 0` in the wide-mask scheme. When the binning input
switches from `hand` to `raw_hand` (Phase E.5), this shortcut breaks —
streams in filled depressions have `raw_hand < 0` and would be
misclassified as FZ.

**Use explicit channel and water masks instead:**

```python
valid = (
    (water_mask == 0)            # exclude NWI lakes (handled by lake column)
    & (channel_mask == 0)        # exclude all stream channel pixels regardless of fill
    & np.isfinite(raw_hand)      # exclude NaN
    & post_dtnd_tail_mask        # existing tail_index DTND outlier filter
    & (raw_hand >= q01)          # Phase E.5 lower outlier trim
    & (raw_hand <= q99)          # Phase E.5 upper outlier trim
)
```

`channel_mask` (just natural streams, not lakes) is computed at
`run_pipeline.py:881`. `wide_channel_mask` is the same plus NWI lakes;
either works here since `water_mask` already filters lakes.

**Why `channel_mask`, not `raw_hand != 0`:** the raw-HAND test misses
real stream pixels that pass through filled depressions. Those have
`hand = 0` and `dep_fill > 0`, so `raw_hand < 0`, and would slip past
the heuristic and end up in FZ bins. The explicit channel-mask filter
catches them regardless of conditioning history.

**Q01/Q99 cutoffs:** see Phase E.5 doc 2026-05-02 log entry for the
formalized defense. Computed dynamically from the post-DTND-tail-removal
pixel set (recompute each run; do not hardcode).

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

### 7.5 NWI Mask Holes from Nested-Polygon Rasterization

**Discovery (2026-04-29):** The NWI water mask
(`data/mosaics/production/water_mask.tif`) has holes inside larger lake
polygons. Pixels in these holes have `water_mask == 0` despite being
physically inside the lake, surrounded on all sides by `water_mask == 1`
pixels.

**Root cause:** NWI shapefiles often contain nested polygon rings — for
example, an outer "wetland" polygon with inner "open water" polygons inside
it. When the shapefile was rasterized to produce `water_mask.tif`, the
inner rings appear to have been treated as **negative space** (holes) rather
than as additional water area. Pixels inside the inner rings end up flagged
non-water.

**Quantitative scope (R4C10/C11 region as a sample):**
- 4 distinct hole regions inside a single lake
- 18,564 hole pixels total in this one lake
- Most (16,282) have NaN HAND (pysheds couldn't trace a flow path)
- Small minority have finite HAND: 244 pixels in (0, 0.1m], 2,038 with HAND > 0.1m
- All have `depression_fill_depth ≈ 1.6m` (consistent with surrounding lake basin)
- None are in `wide_channel_mask`

Estimated domain-wide impact: probably 50-200K hole pixels across all major
lakes. Less than 0.3% of the production domain.

**Pipeline data impact: small.** The NaN-HAND hole pixels (~88% of holes)
are excluded by the standard `np.isfinite(hand) & (hand > 0)` filter that
runs on all hillslope statistics. Only the small fraction with finite
positive HAND (~2,200 in the sample lake) could leak into hillslope binning
— and these would land in the lowest HAND bins where they'd be a tiny
contamination relative to the millions of legitimate pixels there.

**Visualization impact: noticeable.** The hole pixels appear:
- BRIGHT RED in the spatial contamination heat-map plots — heat layer renders
  their depression_fill values (~1.6m) and the NWI overlay doesn't cover them
  (water_mask == 0)
- WHITE in the categorical map — most have NaN HAND, falling outside every
  layer's mask

The production `hand_map.png` happens to look mostly clean because pixels
with NaN HAND render as transparent in viridis, masking the issue
inadvertently.

**Fix path: re-rasterize the NWI shapefile.** Properly handle nested
polygon rings (treat inner rings as additional water rather than holes).
This regenerates `water_mask.tif`. After that, a pipeline rerun produces
correct binning and clean visualizations automatically. The
`scripts/osbs/generate_water_mask.py` script (or equivalent) is the
candidate to update.

**Cosmetic-only workaround (not applied):** `scipy.ndimage.binary_fill_holes`
on the in-memory water_mask before plotting would close the visual holes
without modifying the underlying data file. Skipped per current direction —
the proper fix should land before any production pipeline rerun.

**Action item:** when we next rerun the pipeline (e.g., after PI sign-off
on the binning approach), regenerate `water_mask.tif` first with corrected
nested-polygon handling. Document the regeneration step in the run log.

---

## 8. Items Requiring PI Clarification

### Active (need PI input)

_(none currently — all items resolved as of 2026-04-25)_

Open question for follow-up:
- **HAND binning methodology change.** Whether to switch to raw-DEM HAND
  for binning (resolves the 17%-of-domain mis-placement of basin-interior
  pixels). Possible re-architecting into flood-zone bins with finer TAI
  resolution. Pending PI decision; covered in separate planning doc.

### Resolved during audit

| # | Question | Resolution |
|---|----------|------------|
| 1 | Lake-at-col-1 placement (7.1, 5.1) | **Design decision (2026-04-23):** lake at col 1, land at cols 2-17. Preserves current 16-column `wetlandisfull` behavior without Fortran changes. |
| 2 | `soil_profile_method` for osbs4 | **Uniform** (lnd_in:261). See Section 7.2. No distance/bedrock dependence on soil depth. |
| 3 | SPILLHEIGHT = 0.2m confirmed | Verified current value is 0.2m (HillslopeHydrologyMod.F90:55). The NetCDF convention `hill_elev = -SPILLHEIGHT` is a design choice; the current value produces `hill_elev = -0.2m` (runtime -0.4m). PI direction (2026-04-25): keep 0.2m default; revisit only if model output suggests issues. Item #7 (spill depth tuning) deferred to model output. |
| 4 | `tdepth(g) = 0` for OSBS? | **Yes, guaranteed.** `flds_r2l_stream_channel_depths = .false.` in nuopc.runconfig:111 → MOSART does not export Sr_tdepth → CTSM resets tdepth_grc to 0 every timestep (lnd_import_export.F90:624). Empty-stream guard fires; lake-to-stream gradient clamped to zero. See Section 7.3. |
| 5 | "Always submerged" definition (5.4) | **Don't enforce parametrically.** PI direction (2026-04-25): real lakes dry cyclically (decadal/centennial). Forcing permanent submersion would be wrong. Run with reasonable parameters; if vegetation patterns are clearly wrong (e.g., pine trees colonizing the lake permanently) revisit. |
| 6 | `hill_bedrock_depth` for lake (5.4) | **Moot.** Field is ignored under Uniform soil profile. Set to 0, consistent with all land columns. Saturation enforcement, if needed, must come from elsewhere (infiltration likely handles it). |
| 7 | Spill depth tuning (5.5) | **Deferred to model output.** PI direction (2026-04-25): keep current SPILLHEIGHT = 0.2m default. Don't tune to Lee 2023's 2.64m parametrically; let CTSM model output guide whether tuning is needed. Empirical references retained for the future tuning conversation: Lee 2023 (2.64 m, n=14, field-surveyed bottoms) and our own pipeline (3.33 m, n=107 non-NWI basins ≥ 1 ha; 4.70 m for NWI-overlapping basins ≥ 1 ha — different physical quantity, see Section 6.7.3). Both lines of evidence put the empirically supported range at 1–3 m, vs. the current 0.2 m default. |
| Lake hill_distance | DTND ≈ stream width (~5m) | PI direction (2026-04-25). Mathematically inert under current config (Section 4.4). Earlier "col2/2" recommendation (~15m) was also defensible; PI prefers stream width as the simpler choice. |
| Lake hill_slope | 0 | PI direction (2026-04-25). "Lake bottom" framing — water surface is horizontal. Earlier bathymetric value (0.015) reverted. See Section 5.5. |
| Lake hill_width | 1/2 NWI total perimeter | PI direction (2026-04-25). Inert under current config. Sanity-check via P:A ratio. See Section 5.3. |

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
  Source of OSBS mean spill depth (2.64 m, n=14). Methodology for OSBS
  wetlands cites McLaughlin et al. 2019 directly (CFS stage-breakpoint
  method failed because OSBS wetlands never reach spill).
- McLaughlin, D. L., Diamond, J. S., Quintero, C., Heffernan, J., & Cohen, M.
  J. (2019). Wetland Connectivity Thresholds and Flow Dynamics from Stage
  Measurements. *Water Resources Research*, 55(7), 6018–6032.
  https://doi.org/10.1029/2018WR024652
  LIDAR h_crit methodology used at OSBS by Lee 2023: dry-season flight
  timing, 5-m resample with local-minima filter on dense-vegetation areas
  to strip canopy artifacts, ArcMap Fill against field-surveyed PVC well
  bottoms (3-cm wells installed ~1 m below ground in deepest part of each
  wetland). h_crit = (fill elevation) − (well bottom elevation).
- USDA NRCS (May 2024). Storage and Release of Water in Coastal Plain
  Wetlandscapes. Conservation Effects Assessment Project (CEAP) Conservation
  Insight. (Summary of Lee et al. 2023 for practitioners.)
  `~/CEAP-Wetlands-Conservation-Insight-WetlandscapeConnectivity-May2024.pdf`
- Lane, C. R., & D'Amico, E. (2010). Calculating the ecosystem service of water
  storage in isolated wetlands using LiDAR in North Central Florida, USA.
  *Wetlands*, 30(5), 967-977. https://doi.org/10.1007/s13157-010-0085-z
  (LIDAR spill-depth method cited by Lee et al.)
