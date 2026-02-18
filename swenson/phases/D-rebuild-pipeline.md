# Phase D: Rebuild Pipeline with Fixes

Status: Not started
Depends on: Phase A, Phase B, Phase C
Blocks: Phase F

## Problem

The current pipeline (STATUS.md #9) has ~400 lines of duplicated code between MERIT validation and OSBS scripts, and integrates several now-known-incorrect approaches (EDT-based DTND, np.gradient slope/aspect, hardcoded Lc fallback, 4x subsampling). Rather than patching individual issues, rebuild with the fixes from Phases A-C and extract shared code into a proper module.

**Target architecture:**

| Layer | Responsibility | Location |
|-------|---------------|----------|
| pysheds (fork) | Flow routing, HAND, DTND, slope/aspect | `$PYSHEDS_FORK/pysheds/pgrid.py` |
| Hillslope analysis module | Binning, trapezoidal fit, width, 6-param computation | `scripts/hillslope_params.py` (new) |
| Pipeline scripts | Orchestration, I/O, plotting | `scripts/osbs/run_pipeline.py`, `scripts/merit_validation/stage3_*.py` |

## Tasks

- [ ] Replace EDT-based DTND with pysheds hydrological DTND (from Phase A)
- [ ] Replace np.gradient slope/aspect with pgrid Horn 1981 (from Phase A) — sign bug already fixed as interim measure; this replaces the method entirely
- [ ] Set processing resolution (from Phase B)
- [ ] Set Lc and accumulation threshold (from Phase C)
- [ ] Extract shared hillslope analysis module (quadratic solver, trapezoidal fitting, HAND binning, width computation)
- [ ] Move `dirmap` to pysheds fork as a constant
- [ ] Rerun pipeline on interior mosaic
- [ ] Verify output NetCDF structure still matches Swenson reference format

## Deliverable

Pipeline that produces scientifically defensible hillslope parameters. Shared module eliminates code duplication. Both MERIT validation and OSBS pipeline import from the same source.

## Log

### 2026-02-17: pgrid cleanup moved to Phase A

pgrid.py cleanup (uphill loop extraction, CRS distance helper) was originally scoped here as opportunistic work. Moved to Phase A to finish fork work cleanly before pipeline rebuild. See Phase A for full context, decision record, and "what not to touch" guidelines.

