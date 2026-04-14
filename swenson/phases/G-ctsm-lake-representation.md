# Phase G: Submerged Lake Column in Hillslope File

Status: Not started
Depends on: Phase F
Blocks: None

## Problem

CTSM's `istdlak` (lake) and `istsoil` (hillslope) land units are structurally independent — they combine at the gridcell level via area weighting but have no mechanism for lateral water exchange. The project's scientific goal (TAI dynamics — water table rise near lakes, suppressed aerobic decomposition, increased CH4) requires lake and hillslope to be coupled hydrologically.

The original Phase G plan (weir overflow, repurposing the stream infrastructure with ~70 lines of Fortran) was deemed more complexity than OSBS needs. Per PI consultation with collaborating scientists (2026-04-09), lake bathymetry detail and stream-as-lake repurposing are unnecessary for the OSBS goals — knowing the fractional lake area and treating it as permanently submerged is sufficient.

## Approach

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

### Lake column fields

| Field | Value | Source |
|---|---|---|
| `hill_elev` | `-SPILLHEIGHT` (in NetCDF) | Pre-SourceMod value. The PI's SourceMod subtracts `SPILLHEIGHT` again at runtime, so the final in-memory value is `-2*SPILLHEIGHT`. With the default spillheight = 0.2m, the lake column ends up at -0.4m — permanently and deeply submerged, deeper than any upland column that might flip negative from the spillheight subtraction alone. |
| `hill_distance` | `mean(dtnd[water_mask])` | Pixel-wise mean of DTND across all NWI lake pixels. Equal-area pixels (1m²) make this equivalent to area-weighted per-lake averaging. Represents the average drainage distance for the aggregate "lake" column. |
| `hill_area` | `sum(water_mask) * pixel_area` | Total NWI lake area from the water mask (production domain: 103 features, 10.7M pixels, ~10.68 km²). This defines the fractional area of the lake column within the gridcell. |
| `hill_width` | TBD | See open questions |
| `hill_slope` | TBD (likely 0) | See open questions |
| `hill_aspect` | TBD (likely 0) | See open questions |
| `hill_bedrock_depth` | 0 | Matches hillslope convention |
| `column_index` | TBD (likely `N_HAND_BINS + 1`) | Sequential |
| `downhill_column_index` | TBD (3 topology options) | See open questions |
| `hillslope_index` | TBD (reuse `1` or new `2`) | See open questions |

NetCDF structure change: `nmaxhillcol = N_HAND_BINS + 1` (16 hillslope bins + 1 lake = 17 columns).

## Why this works

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
