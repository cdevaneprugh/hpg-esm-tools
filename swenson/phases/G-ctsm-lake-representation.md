# Phase G: Submerged Lake Column in Hillslope File

Status: Not started
Depends on: Phase F
Blocks: None

## Problem

CTSM's `istdlak` (lake) and `istsoil` (hillslope) land units are structurally independent — they combine at the gridcell level via area weighting but have no mechanism for lateral water exchange. The project's scientific goal (TAI dynamics — water table rise near lakes, suppressed aerobic decomposition, increased CH4) requires lake and hillslope to be coupled hydrologically.

The original Phase G plan (weir overflow, repurposing the stream infrastructure with ~70 lines of Fortran) was deemed more complexity than OSBS needs. Per PI consultation with collaborating scientists (2026-04-09), lake bathymetry detail and stream-as-lake repurposing are unnecessary for the OSBS goals — knowing the fractional lake area and treating it as permanently submerged is sufficient.

## Approach

Append one extra column per gridcell to the hillslope NetCDF. The column represents the aggregate of all NWI-masked lake area as a single "submerged" hillslope column with negative `hill_elev`. CTSM's existing lateral flow machinery handles the water exchange naturally — no Fortran modifications from our fork are planned.

### Lake column fields

| Field | Value | Source |
|---|---|---|
| `hill_elev` | `-SPILLHEIGHT` | Negative of the PI's spillheight SourceMod scalar. Example: if spillheight = 0.2m, then `hill_elev = -0.2m`. The lake column sits below the hillslope reference plane. |
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

## Why not weir overflow (historical)

The original Phase G plan (documented in `docs/water-masking-and-lake-representation.md`) proposed replacing Manning's equation in `HillslopeStreamOutflow` with a weir equation and repurposing the `stream_channel_*` fields as lake geometry. After PI consultation with collaborators:

- **Lake bathymetry detail is not strictly necessary.** The submerged-column approach captures the first-order effect (lake area is permanently saturated and draws water from upland) without per-lake depth curves or pour-point analysis.
- **Fortran modifications add scope and maintenance burden** for a research project that should focus on OSBS first.
- **The PI's existing spillheight SourceMod** already provides the CTSM-side tuning knob needed for this configuration.
- **Generalization (full lake representation, variable geometry, dynamic inundation)** is better pursued as a separate research project rather than bolted onto OSBS parameter generation.

The `water-masking-and-lake-representation.md` document retains value as reference — the CTSM source investigation (stream water cycle, lateral flow, carbon-water coupling) is still accurate and useful when reasoning about behavior. The modification options A-E in that document are no longer planned implementations.

## Pipeline tasks

- [ ] Add `SPILLHEIGHT` constant to `run_pipeline.py` (value from PI)
- [ ] Add lake column computation step (between Step 5 hillslope params and Step 6 NetCDF write):
  - Compute `lake_dtnd = float(np.mean(dtnd[water_mask > 0]))`
  - Compute `lake_area = float(np.sum(water_mask)) * PIXEL_SIZE**2`
  - Construct lake column dict with known + TBD fields
  - Append to `params["elements"]`
- [ ] Update NetCDF writer to handle `nmaxhillcol = N_HAND_BINS + 1`
- [ ] Update `pct_hillslope` and `nhillcolumns` metadata to reflect the extra column
- [ ] Expand to 16 HAND bins with focus on 0-50cm zone (PI confirmed). Revert to Q1/Q99 percentile endpoints — the "standing water is a feature" framing means small gradients and near-zero-height bins are acceptable. See `docs/hillslope-binning-rationale.md` for background. `nmaxhillcol` becomes 17 (16 bins + 1 lake column).
- [ ] Production run, verify NetCDF structure with `ncdump -h`
- [ ] CTSM test branch from osbs2 with modified hillslope file + PI's spillheight SourceMod

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

Pipeline-generated hillslope NetCDF with 17 columns (16 HAND bins + 1 submerged lake column), consumed by CTSM with the PI's spillheight SourceMod active. Comparison simulation against Phase F baseline showing the effect of the submerged lake column on water table dynamics, soil moisture, and CH4 production in near-lake hillslope columns.

## References

- `docs/water-masking-and-lake-representation.md` — historical; CTSM source investigation still valid, modification options A-E superseded
- `docs/hillslope-binning-rationale.md` — binning strategy (A2 current, may be revisited)
- `$TOOLS/docs/SPILLHEIGHT_IMPLEMENTATION.md` — PI's existing spillheight SourceMod
- `phases/F-validate-deploy.md` — baseline for Phase G comparison

## Log
