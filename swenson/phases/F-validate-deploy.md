# Phase F: Validate and Deploy

Status: Not started
Depends on: Phase E
Blocks: Phase G

## Problem

The final hillslope file needs validation at two levels before use in production runs: (1) physical plausibility of the parameters themselves, and (2) correct behavior when ingested by CTSM. The osbs2 baseline case (860+ year spinup with Swenson's global hillslope data) provides the comparison target.

## Key Context

**osbs2 runs with `use_hillslope_routing = .false.` and `PCT_LAKE = 0`** (no lake land unit, no stream routing). Phase F validation matches this configuration exactly — the only variable changed is the hillslope file itself. Stream params are irrelevant (never read with routing off).

`PCT_LAKE = 0` means there is no separate lake land unit in osbs2. The hillslope occupies 100% of the vegetated landunit. This is already the configuration Phase G needs (lake-within-hillslope via modified stream bucket). Phase F establishes the baseline that Phase G compares against.

## Tasks

- [ ] Compare custom hillslope file to Swenson reference (`hillslopes_osbs_c240416.nc`) — parameter-by-parameter
- [ ] Physical plausibility checks:
  - Elevation distribution matches known OSBS topography
  - Aspect distribution consistent with terrain
  - Stream network aligns with known hydrology
  - HAND values reasonable for low-relief wetlandscape (meters, not hundreds)
- [ ] Create CTSM test branch from osbs2 at year 861, `use_hillslope_routing = .false.`, `PCT_LAKE = 0`
- [ ] Run short simulation (1-5 years) with custom hillslope file as the only change
- [ ] Compare outputs to baseline:
  - Water table depth (ZWT)
  - Soil moisture profiles
  - Carbon fluxes (GPP, NEE)
  - Energy balance (latent/sensible heat)
  - Column-level differences (h1 stream)

## Deliverable

Validated hillslope file ready for production runs. Comparison document showing custom vs. global hillslope parameter differences and their impact on simulated fluxes.

## Log

