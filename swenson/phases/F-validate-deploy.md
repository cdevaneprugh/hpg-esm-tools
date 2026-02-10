# Phase F: Validate and Deploy

Status: Not started
Depends on: Phase D, Phase E
Blocks: None

## Problem

The final hillslope file needs validation at two levels before use in production runs: (1) physical plausibility of the parameters themselves, and (2) correct behavior when ingested by CTSM. The osbs2 baseline case (860+ year spinup with Swenson's global hillslope data) provides the comparison target.

## Tasks

- [ ] Compare custom hillslope file to Swenson reference (`hillslopes_osbs_c240416.nc`) — parameter-by-parameter
- [ ] Physical plausibility checks:
  - Elevation distribution matches known OSBS topography
  - Aspect distribution consistent with terrain
  - Stream network aligns with known hydrology
  - HAND values reasonable for low-relief wetlandscape (meters, not hundreds)
- [ ] Create CTSM test branch from osbs2 at year 861
- [ ] Run short simulation (1-5 years) with custom hillslope file
- [ ] Compare outputs to baseline:
  - Water table depth (ZWT)
  - Soil moisture profiles
  - Carbon fluxes (GPP, NEE)
  - Energy balance (latent/sensible heat)
  - Column-level differences (h1 stream)

## Deliverable

Validated hillslope file ready for production runs. Comparison document showing custom vs. global hillslope parameter differences and their impact on simulated fluxes.

## Log

