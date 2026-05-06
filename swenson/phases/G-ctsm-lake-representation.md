# Phase G: Submerged Lake Column in Hillslope File

Status: **Folded into Phase E.5** (reframed 2026-04-30)
Depends on: —
Blocks: —

**2026-05-04 update — lake column hill_elev locked at -6.0 m.** The
`hill_elev` table entry below has been updated; remaining narrative
text in this doc still uses the older `-SPILLHEIGHT` / `-0.2 m`
illustrative numbers in places (e.g., the "Why this works" section
and "Integration risks" sections) — those are kept as historical
context for the CTSM behavior analysis but are NOT the current values.
The canonical lake column reference is
`docs/lake-column-ctsm-audit.md` Sections 5.1-5.5 + 5.2.1.

**2026-04-30 update — Phase G effectively retired.** The PI meeting on
2026-04-30 retired the spillheight SourceMod and shifted the lake column
representation onto an empirical, data-derived `hill_elev` (locked
2026-05-04 at -6.0 m; see Section 5.2.1 of audit doc). The "submerged
lake column with spillheight SourceMod" framing of this phase no longer
applies. The lake column is now built as part of the bin redesign in
**Phase E.5** (`phases/E.5-bin-redesign.md`).

The content below is retained for historical reference only. See
`STATUS.md` Phase E.5 entry for the current plan.

**2026-04-25 update:** Lake column scope refined to NWI water only. Earlier
discussions about absorbing all depression-filled pixels into the lake
column have been superseded by Phase E.5 (HAND binning fix), which
introduces flood-zone bins to handle unmapped depression pixels separately.
Lake column = permanent open water; flood-zone bins = land that floods
seasonally. See `docs/lake-column-ctsm-audit.md` Section 6.7.

## Problem

CTSM's `istdlak` (lake) and `istsoil` (hillslope) land units are structurally independent — they combine at the gridcell level via area weighting but have no mechanism for lateral water exchange. The project's scientific goal (TAI dynamics — water table rise near lakes, suppressed aerobic decomposition, increased CH4) requires lake and hillslope to be coupled hydrologically.

The original Phase G plan (weir overflow, repurposing the stream infrastructure with ~70 lines of Fortran) was deemed more complexity than OSBS needs. Per PI consultation with collaborating scientists (2026-04-09), lake bathymetry detail and stream-as-lake repurposing are unnecessary for the OSBS goals — knowing the fractional lake area and treating it as permanently submerged is sufficient.

## Approach

Phase G has **two stages** with different deliverables and dependencies:

| Stage | Goal | Status | Notes |
|---|---|---|---|
| **1. Lake column construction + CTSM ingestion** | Add a submerged column to the hillslope NetCDF; verify CTSM reads it correctly and column weights are sensible | **Complete** (Phase E.5 + 2026-05-05 per-rep rescale) | Lake at chain index 1, `hill_elev = -6 m`, per-rep area/width via `nhill_implicit ≈ 533`. Validated by `osbs5.swenson.spinup` 100-yr run alongside Phase F. |
| **2. Routing-on validation** | Turn on `use_hillslope_routing = .true.`, verify lateral flow produces TAI behavior (water table rise near lake, suppressed aerobic decomposition, CH4 increase) | **Deferred** | Requires (a) switch from single-point mode to explicit `LND_DOMAIN_MESH` so `grc%area` is real, (b) hydraulic-conductivity sanity-check between columns. See 2026-05-05 log entry. |

**Phase G Stage 1 ran in parallel with Phase F**, not sequentially
after it. The 2026-04-25 PI direction folded the lake column into the
pipeline output (single submerged column, not a separate landunit),
dissolving the original "F provides lake-less baseline → G adds lake
column" ordering. The pipeline's hillslope NetCDF always includes the
lake column. The osbs5 validation case proves both: (a) the file
behaves sensibly over a long spinup (Phase F), and (b) CTSM ingests
the lake-included structure correctly with the right column weights
(Phase G Stage 1).

The rest of this section describes the unified mechanism (lake column +
PI's SourceMod). The **TAI dynamics narrative** ("water flows from
upland to lake, water table rises, decomposition suppressed, CH4 emits")
emerges only when **Stage 2** is enabled. Under Stage 1 alone (current
osbs5 config), columns are hydrologically isolated and any patterns in
gridcell aggregates come from independent per-column water balances
(precipitation distributed by area weight), NOT from lateral flow.

Append one extra column per gridcell to the hillslope NetCDF. The column represents the aggregate of all NWI-masked lake area as a single "submerged" hillslope column with negative `hill_elev`. This pipeline change works **in tandem** with the PI's existing spillheight SourceMod (see below) — the two mechanisms are complementary, not competing.

### Integration with the PI's spillheight SourceMod

The PI's SourceMod (in `osbs4-6/SourceMods/src.clm/`) modifies `InitHillslope` in `HillslopeHydrologyMod.F90` to subtract a spillheight scalar (currently 0.2m, hardcoded) from every column's `hill_elev` after reading the NetCDF. The SourceMod also adds custom runoff-suppression logic in `SurfaceWaterMod.F90` that prevents surface runoff from any column with `hill_elev + h2osfc < 0` (ponding instead of draining).

Combined with our added lake column, the runtime elevation profile becomes:

| Column type | Pipeline `hill_elev` | After SourceMod subtraction | Regime |
|---|---|---|---|
| Added lake column | `-SPILLHEIGHT` (e.g., -0.2m) | `-2*SPILLHEIGHT` (e.g., -0.4m) | Permanently submerged, deepest |
| Lowest HAND bins | Small positive (e.g., 0.03m) | Slightly negative (e.g., -0.17m) | Submerged by default, can dry in drought |
| Middle HAND bins | Meters positive | Still positive, reduced | Upland, occasionally wet |
| Ridge bins | Large positive | Large positive, slightly reduced | Dry upland |

This produces a **stratified submergence system**: the lake column anchors the "always wet" regime (NWI-identified open water), the spillheight subtraction enables the "usually wet, sometimes dry" TAI transition zone, and upper columns behave as normal upland. Each mechanism does a distinct job — they do not double-count or conflict.

### Lake column fields (resolved 2026-04-25 via PI direction)

| Field | Value | Source |
|---|---|---|
| `column_index` | 1 | Design decision (2026-04-23): place lake at col 1, shift land columns up. Preserves current 16-column `wetlandisfull` behavior in the PI's SourceMod by keeping the `cold == ispval` column processed first in the SurfaceWaterMod loop. |
| `downhill_column_index` | -9999 | Terminal column. Drains to stream. |
| `hillslope_index` | 1 | Same hillslope as all others (single-aspect setup). |
| `hill_elev` | **-6.0 m** | **Locked 2026-05-04** (supersedes the earlier `-SPILLHEIGHT` framing). PI suggestion. Chain-bookkeeping value, set 0.87 m below the deepest land bin mean (-5.13 m, bin 1 of the 24-bin scheme). SPILLHEIGHT=0 in the SourceMod (no runtime shift). The value does NOT represent a physical lake bottom — empirical lake geometry (NWI mean -2.53 m; Lee/pipeline spill 2.64-3.33 m) doesn't reach the chain monotonicity floor. Tuning deferred to model output. See `docs/lake-column-ctsm-audit.md` Section 5.2.1 for full derivation. |
| `hill_distance` | **0.5 × Bin 1's distance** (computed dynamically; locked 2026-05-04) | The lake-to-stream subsurface gradient is clamped under current config (routing off, `tdepth=0`), but the col-col Darcy gradient between Bin 1 and the Lake (audit Section 1.1, Path A) is NOT clamped — its denominator `d(Bin1) - d(Lake)` must stay positive. With raw-HAND binning, Bin 1's trap-fit DTND is small (~3 m on production), so the audit's earlier static "~5 m" / "stream width" framing inverted the gradient sign. Dynamic computation guarantees the constraint by construction. See audit Sections 4.4 and 1.1. |
| `hill_area` | `sum(water_mask) * pixel_area` ≈ 10.68 km² | Total NWI lake area from the water mask. Production domain: 103 features, 10.7M pixels. Defines fractional area of lake column within gridcell. |
| `hill_width` | 1/2 NWI total perimeter | PI direction (2026-04-25). Heuristic: half the perimeter captures the effective lateral exchange surface. Inert under current config (routing off, lake-to-stream gradient clamped). Sanity-check via P:A ratio. |
| `hill_slope` | 0 | PI direction (2026-04-25). "Lake bottom" framing — water surface is horizontal. Earlier bathymetric value (0.015 from Lee 2023 bowl geometry) reverted. |
| `hill_aspect` | 0 | Inconsequential for flat lake column. |
| `hill_bedrock_depth` | 0 | Matches hillslope convention. Inert under Uniform soil profile method (verified osbs4 lnd_in:261). |

NetCDF structure change: `nmaxhillcol = N_BINS + 1` where N_BINS depends on the Phase E.5 outcome. Under the proposed flood-zone-bin scheme: ~16-22 columns total (1 lake + ~5 flood zone + ~10-16 upland-direction bins).

Authoritative reference: `docs/lake-column-ctsm-audit.md` Sections 5.1-5.5 — full rationale for each parameter value with PI direction notes.

## Why this works

> **Prerequisite:** the lateral-flow pathway described below requires
> `use_hillslope_routing = .true.`. Our current osbs5 config (and
> osbs2/osbs4-6) all run with **routing OFF**. Under that configuration
> the columns exist as independent 1D soil columns — no col-to-col
> water exchange, no TAI emergence via lateral flow. The lake column
> only receives water from direct precipitation on its area share.
> See the 2026-05-05 log entry for the full breakdown of why routing
> is off (single-point mode + `grc%area = spval`) and what would be
> required to enable it. This section describes the eventual
> **Stage 2** behavior; **Stage 1** (lake column construction in the
> hillslope file) is what the rest of this doc and the Phase E.5 work
> have delivered.

**Negative `hill_elev` is permitted.** Confirmed in `ColumnType.F90:76` — no guards, sign checks, or absolute value operations on `hill_elev` anywhere in the source. The head gradient equation in `SoilHydrologyMod.F90` (line 2261) works with negative values because it computes differences:

```fortran
head_gradient = (col%hill_elev(c) - zwt(c)) - min((stream_water_depth - stream_channel_depth), 0._r8)
head_gradient = head_gradient / col%hill_distance(c)
```

A column at `hill_elev = -0.2m` (say, lake surface 0.2m below hillslope reference) with a water table near its surface produces a gradient that favors flow *from* upland columns *into* the lake column. Water drains from hillslope to lake naturally through the existing lateral flow pathway.

**TAI response is automatic.** CTSM's carbon-water coupling chain runs without modification:

```
zwt (water table depth)
  |-> soilpsi -> w_scalar -> decomposition rate
  |-> O2 transport -> o2stress_unsat -> o_scalar -> decomposition rate
  |-> finundated -> CH4 production, transport, oxidation
```

Columns adjacent to the lake (lowest-HAND hillslope bins) see water tables rising via the lateral flow, which saturates the soil, suppresses aerobic decomposition (`o_scalar` drops), and increases CH4 production (`finundated` rises). No biogeochemistry changes needed.

**Standing water is physically correct.** Per PI guidance (2026-04-09), the small hydraulic gradients between low-HAND columns that produce "stuck" or "standing" water states are the correct representation of wetland terrain, not a bug. This relaxes the previous concern about adjacent-bin height differences being too small to drive lateral flow.

## Integration risks (things to watch)

The lake column + SourceMod stratification is conceptually clean, but several details need careful handling. Flagging them here so they are not overlooked during implementation.

### SPILLHEIGHT value coordination

The pipeline's `SPILLHEIGHT` constant **must match** the value hardcoded in the PI's SourceMod (currently 0.2m in `HillslopeHydrologyMod.F90` line 55). If they diverge:

- If pipeline uses a larger value: lake column is deeper than the "natural floor" created by the spillheight subtraction on the lowest upland bins — probably fine.
- If pipeline uses a smaller value: lake column could end up *shallower* than the lowest upland bin after subtraction (e.g., `-0.1m` lake but lowest upland at `-0.17m`). This would put the "lake" above the wetland transition zone — physically wrong.

**Mitigation:** Keep `SPILLHEIGHT` in the pipeline as a named constant with a comment pointing to the SourceMod source. Any change in the SourceMod must prompt a corresponding pipeline update and file regeneration.

### SurfaceWaterMod ponding logic applies to the lake column

The PI's `SurfaceWaterMod.F90` modification triggers on `col%hill_elev(c) + h2osfc(c)*1.e-3_r8 < 0`:

```fortran
if (col%hill_elev(c)+h2osfc(c)*1.e-3_r8 < 0._r8) then
   qflx_h2osfc_surf(c) = 0._r8   ! no runoff, build pond
```

Our lake column will be at `hill_elev = -2*SPILLHEIGHT` at runtime — it will always trigger this branch and suppress surface runoff. That is the intended behavior for a permanent lake. Verify the logic does not produce unintended side effects (e.g., unbounded water accumulation if inputs exceed outputs). The `h2osfc_thresh` check in the adjacent `else if` branch handles the spillover threshold when water rises above the reference plane.

### SoilHydrologyMod modifications not yet fully investigated

The PI's SourceMod includes changes to `SoilHydrologyMod.F90` (147 KB file). The integration analysis did not fully trace what these changes do — they may affect head gradient calculations, lateral flow between columns, or the losing-stream exchange. The lake column's `hill_distance` and `hill_width` parameters feed into these calculations. Review these SourceMod changes before committing to specific default values for the lake column's geometric fields.

### Topology of `downhill_column_index`

CTSM's lateral flow routing uses `downhill_column_index` to determine which column each column drains to. The lake column must be positioned in the drainage chain so that water flows *into* it (not out of it). Three scenarios:

1. **Lake as terminal sink** (likely correct): lowest HAND bin → lake column → -9999 (stream sentinel). Water drains downhill through the HAND bins, accumulates in the lake.
2. **Lake as sibling**: each HAND bin → -9999, lake → -9999. No flow path *through* the lake; lake receives water only via the lateral head gradient from adjacent columns.
3. **Lake bypasses stream**: lake → -9999 directly, hillslope → -9999 directly. Same effect as sibling for Phase G purposes.

The PI's SourceMod does not modify `downhill_column_index` logic, so it uses whatever topology the NetCDF specifies. Choose the topology that produces physically sensible drainage; verify by inspecting water mass balance during the validation run.

### Lake column `hill_width`, `hill_slope`, `hill_aspect`

These fields are read by CTSM but have no physical meaning for a flat lake. The PI's `SurfaceWaterMod.F90` uses `col%hill_slope(c)` in the surface runoff calculation (`k_wet = 1e-4 * max(col%hill_slope(c), min_hill_slope)`). Setting `hill_slope = 0` is fine as long as `min_hill_slope` provides a floor; otherwise the lake column might produce zero runoff through that code path (probably what we want anyway for a lake, but verify the behavior is as intended).

Conservative defaults:
- `hill_slope = 0` (CTSM's `min_hill_slope` floor will prevent zero-divide issues)
- `hill_aspect = 0` (insolation correction is negligible at this flat angle; value doesn't matter for a submerged column)
- `hill_width = lake_area / lake_dtnd` (geometric closure — if CTSM computes any area quantities from width × distance, this keeps them consistent)

### Column count and `pct_hillslope` normalization

With 17 columns (16 HAND + 1 lake), `pct_hillslope` must sum to 100%. Options:
- **Area-weighted**: the lake column's percentage = lake area / total gridcell area. HAND bins' percentages scale to fill the rest.
- **Equal**: each column gets 100/17 % (does not reflect actual terrain coverage).

Area-weighted is correct. Verify that the NWI lake area (from the water mask) is consistent with the surface dataset's `PCT_LAKE` (which osbs2 has = 0 — meaning the hillslope landunit should claim 100% of the gridcell, and within the hillslope the lake column gets the NWI fraction).

## Why not weir overflow (historical)

The original Phase G plan (documented in `docs/water-masking-and-lake-representation.md`) proposed replacing Manning's equation in `HillslopeStreamOutflow` with a weir equation and repurposing the `stream_channel_*` fields as lake geometry. After PI consultation with collaborators:

- **Lake bathymetry detail is not strictly necessary.** The submerged-column approach captures the first-order effect (lake area is permanently saturated and draws water from upland) without per-lake depth curves or pour-point analysis.
- **Fortran modifications add scope and maintenance burden** for a research project that should focus on OSBS first.
- **The PI's existing spillheight SourceMod** already provides the CTSM-side tuning knob needed for this configuration.
- **Generalization (full lake representation, variable geometry, dynamic inundation)** is better pursued as a separate research project rather than bolted onto OSBS parameter generation.

The `water-masking-and-lake-representation.md` document retains value as reference — the CTSM source investigation (stream water cycle, lateral flow, carbon-water coupling) is still accurate and useful when reasoning about behavior. The modification options A-E in that document are no longer planned implementations.

## Pipeline tasks

- [ ] Add `SPILLHEIGHT` constant to `run_pipeline.py`. Initial value 0.2m to match PI's SourceMod hardcoded default. Document the coupling in a comment pointing to `HillslopeHydrologyMod.F90` line 55 — if the PI changes the SourceMod value, pipeline must be updated in lockstep.
- [ ] Add lake column computation step (between Step 5 hillslope params and Step 6 NetCDF write):
  - Compute `lake_dtnd = float(np.mean(dtnd[water_mask > 0]))`
  - Compute `lake_area = float(np.sum(water_mask)) * PIXEL_SIZE**2`
  - `hill_elev = -SPILLHEIGHT` (pre-SourceMod value)
  - `hill_width = lake_area / lake_dtnd` (geometric closure)
  - `hill_slope = 0`, `hill_aspect = 0`
  - `hill_bedrock_depth = 0`
  - Append to `params["elements"]`
- [ ] Update NetCDF writer to handle `nmaxhillcol = N_HAND_BINS + 1`
- [ ] Update `pct_hillslope` with area-weighted fractions (lake column gets `lake_area / total_gridcell_area`; HAND bins fill the rest proportionally)
- [ ] Update `nhillcolumns` metadata to reflect the extra column
- [ ] Set `downhill_column_index` so the lake is the terminal sink (lowest HAND bin → lake column → -9999 stream sentinel). Verify via water mass balance in the validation run.
- [ ] Set `hillslope_index` for the lake column. Default: reuse `1` (same hillslope as the HAND bins, just a submerged member). Flag if the PI's SourceMod expects a separate index.
- [ ] Expand to 16 HAND bins with focus on 0-50cm zone (PI confirmed). Revert to Q1/Q99 percentile endpoints — the "standing water is a feature" framing means small gradients and near-zero-height bins are acceptable. See `docs/hillslope-binning-rationale.md` for background. `nmaxhillcol` becomes 17 (16 bins + 1 lake column).
- [ ] Review PI's `SoilHydrologyMod.F90` modifications before finalizing lake column `hill_distance` and `hill_width`. The SourceMod analysis flagged 147 KB of changes not yet fully traced.
- [ ] Production run, verify NetCDF structure with `ncdump -h`
- [ ] CTSM test branch from osbs2 with modified hillslope file + PI's spillheight SourceMod. Include PI's `SurfaceWaterMod.F90` and `SoilHydrologyMod.F90` SourceMods (not just `HillslopeHydrologyMod.F90`) — all three are needed for the intended behavior.

## Open questions (TBD — need PI clarification)

1. **`SPILLHEIGHT` value.** The PI's existing SourceMod defines this. Need the exact number to hardcode (or configure) in the pipeline.

2. **`downhill_column_index` topology.** Three options:
   - **Lake in-chain:** lowest hillslope bin → lake → -9999 (stream sentinel). Water routes hillslope → lake → stream.
   - **Lake as sibling:** lowest bin → -9999, lake → -9999. No flow path through the lake.
   - **Lake at-bottom (no stream exit):** lowest bin → lake, lake → -9999. Lake is the terminal node below all hillslope columns.
   
   The PI's SourceMod likely assumes a specific topology.

3. **`hillslope_index` for the lake column.** Reuse `1` (treat as part of the single hillslope) or assign `2` (separate aspect). If separate, does `nhillslope` become `2`?

4. **Lake column `slope`, `aspect`, `width`.** Physically meaningless but read by CTSM. Likely defaults:
   - `slope = 0` (no lateral flow driven by slope gradient on the lake column itself)
   - `aspect = 0` (irrelevant for a flat lake surface)
   - `width = lake_area / lake_dtnd` (geometric closure of the trapezoidal interpretation)
   
   The PI's SourceMod may set these or require specific conventions.

5. **HAND binning strategy — DECIDED.** Expand to 16 bins with Q1/Q99 log-spaced endpoints. Current A2 (8 bins, 0.1m floor + Q95) will be replaced. The "standing water is a feature" framing (PI, 2026-04-09) means small gradients and near-zero-height bins in the 0-50cm zone are physically correct for wetland terrain. See `docs/hillslope-binning-rationale.md` for testing infrastructure (comparison script + cached arrays).

## Deliverable

Pipeline-generated hillslope NetCDF with 17 columns (16 HAND bins + 1 submerged lake column), consumed by CTSM with the PI's spillheight SourceMods (`HillslopeHydrologyMod.F90`, `SurfaceWaterMod.F90`, `SoilHydrologyMod.F90`) active. Comparison simulation against Phase F baseline showing the effect of the submerged lake column plus spillheight offset on water table dynamics, soil moisture, and CH4 production in the near-lake wetland transition zone.

## References

- **PI's case (reference implementation):** `/blue/gerber/sgerber/CTSM/cases/osbs4-6/` — user_nl_clm, xml config, SourceMods
- **PI's SourceMods:** `/blue/gerber/sgerber/CTSM/cases/osbs4-6/SourceMods/src.clm/` — 5 modified files (HillslopeHydrologyMod, SurfaceWaterMod, SoilHydrologyMod, InfiltrationExcessRunoffMod, SaturatedExcessRunoffMod)
- **PI's CTSM base:** `/blue/gerber/sgerber/CTSM/ctsm5.3.059` (our fork is 5.3.085; hillslope code identical between versions)
- **PI's hillslope file:** `/blue/gerber/sgerber/CTSM/subset_input/wetland.nc` (16-column custom file, provenance unknown)
- `docs/water-masking-and-lake-representation.md` — historical; CTSM source investigation still valid, modification options A-E superseded
- `docs/hillslope-binning-rationale.md` — binning strategy
- `$TOOLS/docs/SPILLHEIGHT_IMPLEMENTATION.md` — PI's existing spillheight SourceMod writeup (may need updating after the osbs4-6 investigation)
- `phases/F-validate-deploy.md` — baseline for Phase G comparison

## Log

### 2026-05-05 — Gridcell area in single-point mode and the per-rep rescale

Surfaced while drafting the Phase E.5 per-representative-hillslope rescale
(`E.5-bin-redesign.md` 2026-05-05). The 2026-05-04 production NetCDF had a
unit-convention mismatch: lake `hill_area` in total NWI water (11.082 km²),
land `hill_area` in Swenson trap-fit per-rep (sum 0.148 km²). 822x
asymmetry. Initial framing: "this breaks `nhill_per_landunit` and CTSM's
stream channel logic." Truer story below.

**What CTSM actually does for osbs2 single-point cases.** osbs2 (and
osbs4) cases run in CTSM's single-point / single-column mode. Trigger:
`PTS_LAT` and `PTS_LON` set in `env_run.xml`, `LND_DOMAIN_FILE = UNSET`,
`mesh_lnd = UNSET` in `nuopc.runconfig`. The case-side path:

`lnd_set_decomp_and_domain.F90:309-345` (`lnd_set_decomp_and_domain_for_single_column`):

```fortran
ldomain%lonc(1) = scol_lon
ldomain%latc(1) = scol_lat
ldomain%area(1) = spval        ! special value, ~1e36
ldomain%mask(1) = scol_mask
ldomain%frac(1) = scol_frac
```

`grc%area(g)` is set from `ldomain%area(gdc)` at `initGridCellsMod.F90:179`
— so `grc%area = spval` for our cases. Confirmed in v4 model output:

```
$ ncdump -v area osbs2.branch.v4.clm2.h0a.1110-12.nc
area = 9.999999616903162e+35 ;   # = spval
```

There is no 0.5° × 0.5° (2,684 km²) gridcell, no 0.9° × 1.25°
(12,654 km²) gridcell, no 90 km² gridcell. CTSM in this mode operates
on **fractional column weights** with no absolute physical area at the
gridcell level. The 0.5° × 0.5° `domain.lnd.360x720_cruncep_OSBS_*.nc`
file in `$CLM_USRDAT_DIR/datmdata/` is for CRUNCEP atmospheric forcing
subsetting only; it never propagates into `grc%area`.

**Where `grc%area` would matter (and doesn't, currently).**

1. `HillslopeHydrologyMod.F90:486` —
   `nhill_per_landunit = grc%area * 1e6 * wtgcell * pct_hillslope * 0.01 / hillslope_area`.
   Computes "how many copies of the rep hillslope tile the gridcell."
   Gated behind `if (use_hillslope_routing)` at line 475. **osbs2 has
   `use_hillslope_routing = .false.`. This block never executes.**
2. `HillslopeHydrologyMod.F90:502` — stream channel length =
   `Σ hill_width × 0.5 × nhill_per_landunit`. Same gating. Never runs.
3. `HillslopeHydrologyMod.F90:1117-1121` — column physical area
   conversions in some lateral-flow routines. Same routing gate.

What CTSM **does** use even with routing off (line 520):

```fortran
col%wtlunit(c) = (col%hill_area(c) / hillslope_area(nh)) * pct_hillslope * 0.01
```

Pure ratio of `hill_area` values within the landunit. Independent of
absolute `hill_area` magnitude, independent of `grc%area`. Then
`wtgcell = wtlunit × pct_landunit / 100`. All physics that matters for
Phase F (column-level state, per-area fluxes, gridcell aggregates
formed by area-weighted column sums) runs off these fractional weights.

**Implication for the lake column rescale.** The Phase E.5 rescale
(lake from 11.082 km² → 0.0208 km² per-rep) still matters, but the
real reason is column weight, not stamping math:

| Lake `hill_area` | Lake `wtlunit` | Gridcell aggregate signal |
|---|---|---|
| 11.082 km² (raw NWI total) | 11.082 / 11.230 = **98.7%** | ~99% lake-driven |
| 0.0208 km² (per-rep) | 0.0208 / 0.169 = **12.3%** | matches NWI water fraction in 90 km² OSBS domain |

The rescale gets the lake-to-land ratio right at the value we measured
from the LIDAR domain (12.3% NWI water by area). Without the rescale,
the lake column dominates every column-weighted quantity — soil
moisture, GPP, NEE, latent heat. The lake-to-land fraction in the
hillslope file IS what CTSM uses as the gridcell-level lake-vs-land
mix. So the rescale is essential for Phase F validity.

**Implication for Phase G (this phase).** Two scenarios depending on
how routing is enabled:

1. **Phase G with `use_hillslope_routing = .false.`** (current osbs2
   config, just adding the lake column without enabling routing). Same
   regime as Phase F. Column weights drive the comparison; `grc%area`
   irrelevant. The rescale guarantees the right lake/land mix. No
   gridcell-area concern.

2. **Phase G with `use_hillslope_routing = .true.`**. Lateral
   col-col flow turns on. `nhill_per_landunit` is computed at line
   486-487 using `grc%area`. With `grc%area = spval`,
   `nhill_per_landunit ≈ 1e36` — nonsense; stream channel length
   blows up. CTSM probably runs to completion (the multiplier may
   only feed informational outputs in non-pathological code paths)
   but produces nonsense for any quantity scaled by it. To enable
   routing meaningfully, we'd need to provide a real `grc%area` via
   a domain mesh file (LND_DOMAIN_MESH set explicitly). At that
   point the Swenson-style stamping kicks in: our 0.169 km² rep
   hillslope tiles N times to fill whatever gridcell area the mesh
   provides. N becomes "number of OSBS-like representative
   hillslopes assumed in the gridcell" — a modeling choice, not a
   pipeline output.

**Practical decision for now.** Do the per-rep rescale (Phase E.5).
For Phase F and Phase G as currently scoped (routing off), this is
sufficient. If/when routing is ever enabled, revisit gridcell-mesh
configuration as a separate (not pipeline-side) task.

**Reference paths for future debugging.**
- `ctsm5.3/src/cpl/share_esmf/lnd_set_decomp_and_domain.F90:309-345`:
  `lnd_set_decomp_and_domain_for_single_column` — sets `area = spval`.
- `ctsm5.3/src/main/initGridCellsMod.F90:179`: `grc%area = ldomain%area`.
- `ctsm5.3/src/biogeophys/HillslopeHydrologyMod.F90:475-505`: routing
  block where `grc%area` is used; gated by `use_hillslope_routing`.
- `cases/osbs2.branch.v4/CaseDocs/nuopc.runconfig`: confirm
  `mesh_lnd = UNSET`, `mesh_atm = UNSET`.
- v4 history file `area = 9.999...e+35`: smoking gun for spval at
  runtime.
