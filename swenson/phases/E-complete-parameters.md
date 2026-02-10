# Phase E: Complete the Parameter Set

Status: Not started
Depends on: None (can start independently)
Blocks: Phase F

## Problem

Three parameters are placeholders rather than physically motivated values (STATUS.md #5, #6, #7):

**Stream channel parameters (#6):**

| Parameter | Pipeline | Swenson reference |
|-----------|----------|-------------------|
| Stream depth | 0.3 m | 0.269 m |
| Stream width | 5.0 m | 4.414 m |
| Stream slope | heuristic | 0.00233 |

The guesses are in the right ballpark but have no methodology. Stream slope could be computed from DEM elevation drops along the stream network. Depth and width could come from regional empirical relationships or MERIT Hydro.

**Bedrock depth (#7):** Pipeline uses `1e6` (effectively infinite). Swenson reference has all zeros. Neither is physically meaningful. Need to determine what CTSM does with this parameter.

**DEM conditioning (#5):** At 1m resolution, filling pits/depressions erases real features (sinkholes, wetland depressions, karst dissolution). This is a science question for the PI.

## Tasks

- [ ] Compute stream slope from actual stream network elevation profile
- [ ] Research stream depth/width — regional empirical relationships or MERIT Hydro
- [ ] Research bedrock depth — check CTSM behavior with different values, identify data source
- [ ] PI consultation on open questions:
  - DEM conditioning approach (fill all vs. preserve real closed basins)
  - Hillslope structure (4 aspects x 4 elevation bins vs. 1 aspect x 8 bins)
  - Final study boundary (interior tiles default, any adjustments?)
  - Stream channel parameter methodology
  - NEON slope/aspect products (DP3.30025.001) as validation baseline

## Deliverable

Complete set of physically motivated parameters with documented sources. PI decisions recorded on the open science questions.

## Log

