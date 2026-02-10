# Phase B: Resolve Flow Routing Resolution

Status: Not started
Depends on: None
Blocks: Phase D

## Problem

The pipeline subsamples 1m LIDAR to 4m before flow routing (STATUS.md #2), discarding 93.75% of the data before the core hydrology computation. The entire justification for using 1m LIDAR is to capture fine-scale drainage in a low-relief wetlandscape — subsampling to 4m undermines that purpose.

The bottleneck is pysheds' `resolve_flats()`, which has poor scaling on large flat regions. One test at full resolution OOM'd at 64GB. Full resolution at 256GB was never tested. Neither was 2x subsampling.

**File:** `scripts/osbs/run_pipeline.py` line 1354

**Test plan:** `audit/240210-validation_and_initial_implementation/flow-routing-resolution.md`

## Tasks

- [ ] Test full-res (1m) on interior mosaic at 256GB — does `resolve_flats` complete?
- [ ] Test 2x subsampling (2m) at 128GB as middle ground
- [ ] If neither works: evaluate WhiteboxTools pre-conditioning as alternative
- [ ] Run full pipeline at 1m, 2m, 4m on the same region, compare hillslope parameters
- [ ] Document results with memory/time usage and parameter comparison

## Deliverable

Determined processing resolution with scientific justification. Either: (a) full 1m works and we use it, (b) 2m is the practical limit with quantified parameter impact vs 1m, or (c) alternative DEM conditioning tool identified and tested.

## Log

