# Phase E: Complete the Parameter Set

Status: In progress
Depends on: None (can start independently)
Blocks: Phase F

## Problem

Remaining parameters need refinement or PI decisions (STATUS.md #5, #6):

**Stream channel parameters (#6) — partially resolved:**
Stream slope now computed from actual network. Depth and width use interim power-law scaling from drainage area (`depth = 0.001 * A^0.4`, `width = 0.001 * A^0.6`). Phase E will research OSBS-specific empirical relationships.

**Bedrock depth (#7) — resolved:** Now uses zeros, matching Swenson reference (commit 11f465e).

**DEM conditioning (#5) — open:** At 1m resolution, filling pits/depressions erases real features (sinkholes, wetland depressions, karst dissolution). This is a science question for the PI.

## Tasks

- [ ] Compute stream slope from actual stream network elevation profile
- [ ] Research stream depth/width — regional empirical relationships or MERIT Hydro
- [ ] Research bedrock depth — check CTSM behavior with different values, identify data source
- [x] Hillslope structure decision: 1 aspect x 8 log-spaced HAND bins (see `docs/hillslope-binning-rationale.md`)
- [ ] PI consultation on remaining open questions:
  - DEM conditioning approach (fill all vs. preserve real closed basins)
  - Final study boundary (interior tiles default, any adjustments?)
  - Stream channel parameter methodology
  - NEON slope/aspect products (DP3.30025.001) as validation baseline

## Deliverable

Complete set of physically motivated parameters with documented sources. PI decisions recorded on the open science questions.

## Log

### 2026-03-19: Hillslope structure decision

Switched pipeline from 4 aspects x 4 equal-area bins (16 columns) to 1 aspect x 8 log-spaced
HAND bins (8 columns). Log spacing concentrates resolution near the stream where TAI dynamics
dominate. 8 bins chosen over 16 to stay above DEM conditioning noise floor (~cm scale) while
giving 4-5 bins in the 0-2m TAI zone.

Tier 1 (R6C10) test run completed. All 8 bins populated, heights and distances monotonically
increasing. Total area matches 4x4 (0.373 vs 0.374 km^2). Comparison plot generated at
`output/plots/4x4_vs_1x8_r6c10.png`.

Full justification: `docs/hillslope-binning-rationale.md`.

