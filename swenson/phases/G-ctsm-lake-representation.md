# Phase G: CTSM Lake-in-Hillslope Representation

Status: Not started
Depends on: Phase F
Blocks: None

## Problem

CTSM's lake and hillslope land units are completely separate — there is no water exchange mechanism between them. Setting `PCT_LAKE > 0` creates an independent lake with its own hydrology, but no TAI coupling with the hillslope. The project's scientific goal (TAI dynamics — water table rise near lakes, suppressed aerobic decomposition, increased CH4) requires lateral coupling between lake storage and hillslope soil columns.

The existing stream infrastructure (`stream_water_volume`, Manning discharge, bidirectional hillslope-stream exchange) provides a natural mechanism to represent lake storage if the outflow equation is modified.

## Approach

**Option B from `docs/water-masking-and-lake-representation.md`:** Replace Manning's continuous discharge with a weir overflow equation. ~70 lines of Fortran, localized to one subroutine plus initialization.

```
Q = C * L * max(0, h - h_spill)^(3/2)
```

**Zero discharge below pour point.** Water accumulates in the stream bucket. Water table rises in the lowest column via losing-stream exchange. Soil becomes anaerobic. CH4 production increases. When water level exceeds spill height, discharge begins — the TAI overflow event. Carbon response comes automatically from existing w_scalar, o_scalar, and finundated pathways — no biogeochemistry changes needed.

**Stream fields repurposed as lake geometry:**

| NetCDF field | Standard meaning | Phase G meaning |
|---|---|---|
| `stream_channel_width` | Bankfull width | Effective lake width (volume-to-depth) |
| `stream_channel_depth` | Bankfull depth | Reference depth for exchange zero-point |
| `stream_channel_slope` | Channel slope | Unused (weir replaces Manning) |
| `stream_spill_height` (new) | — | Pour point elevation above lake surface |
| `stream_spill_width` (new) | — | Spillway width |

Note: `stream_channel_length` is computed internally by CTSM from hillslope geometry — not a free parameter.

## Tasks

- [ ] PI decision on approach (weir overflow recommended — see `docs/water-masking-and-lake-representation.md` for full option analysis)
- [ ] Implement weir overflow in CTSM fork:
  - [ ] Replace Manning block in `HillslopeStreamOutflow` (`HillslopeHydrologyMod.F90` lines 999-1041) with weir equation (~40 lines)
  - [ ] Add `stream_spill_height` and `stream_spill_width` to `LandunitType.F90` (~6 lines)
  - [ ] Read new variables from NetCDF in `InitHillslope` (~20 lines)
  - [ ] Optionally add `streamflow_method` namelist selector (Manning vs weir)
- [ ] Optionally add `h2osfc_thresh` override on lowest column (Option E, ~5 lines in `SoilHydrologyInitTimeConstMod.F90`)
- [ ] Update pipeline to compute lake parameters from DEM + NWI:
  - [ ] `stream_spill_height`: elevation difference between lake surface and pour point rim
  - [ ] `stream_spill_width`: effective spillway width from DEM gradient at pour point
  - [ ] `stream_channel_width`/`depth`: lake-like values from NWI area and bathymetry estimates
- [ ] Set `PCT_LAKE = 0`, `use_hillslope_routing = .true.`
- [ ] Run comparison: Phase F baseline (separate lake, routing off) vs weir overflow (lake-in-hillslope, routing on)

## Deliverable

CTSM fork with TAI-capable hillslope hydrology. Comparison showing impact of lake-in-hillslope representation on water table dynamics, carbon fluxes, and CH4 production.

## References

- `docs/water-masking-and-lake-representation.md` — full CTSM source investigation, 5 options analyzed, Fortran code sketches
- `docs/synthetic_lake_bottoms.md` — lake depth estimation methods (Cael power law, Hollister topography model, GLOBathy)
- `$TOOLS/docs/SPILLHEIGHT_IMPLEMENTATION.md` — PI's existing spillheight SourceMod experiment

## Log

