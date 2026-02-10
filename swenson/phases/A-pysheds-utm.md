# Phase A: Fix pysheds for UTM

Status: Not started
Depends on: None
Blocks: Phase D

## Problem

Both the DTND problem (STATUS.md #1) and the slope/aspect problem (#4) stem from the same root cause: pysheds assumes geographic coordinates throughout. The pysheds fork also has 33 deprecation warnings (#10) that obscure real issues.

**DTND:** `compute_hand()` computes the correct hydrologically-linked distance internally, but uses haversine math that produces garbage on UTM data. The pipeline works around this with `scipy.ndimage.distance_transform_edt`, which has correct math (Euclidean) but wrong concept (nearest stream pixel regardless of drainage basin). Neither is correct.

| Approach | Concept | Math |
|----------|---------|------|
| pysheds DTND | Correct (flow-path-linked) | Wrong (haversine on UTM) |
| Pipeline EDT | Wrong (nearest regardless of drainage) | Correct (Euclidean on UTM) |

**Slope/aspect:** Stage 8 of MERIT validation discovered `np.gradient`-based aspect had a Y-axis sign inversion swapping North/South. The fix (pgrid's `slope_aspect()` with Horn 1981 stencil) was applied to MERIT validation but not the OSBS pipeline, because `slope_aspect()` also assumes geographic coordinates.

**Fix location:** `$PYSHEDS_FORK/pysheds/pgrid.py`
- `compute_hand()` lines 1928-1942 (haversine distance)
- `slope_aspect()` / `_gradient_horn_1981()` (haversine spacing)

## Tasks

- [ ] Fix 33 deprecation warnings in `pgrid.py` (clean working state before making changes)
- [ ] Modify `compute_hand()` to detect UTM CRS and use Euclidean distance for DTND
- [ ] Modify `slope_aspect()` / `_gradient_horn_1981()` to use uniform pixel spacing for UTM
- [ ] Test both changes against MERIT validation (geographic CRS — should reproduce existing results)
- [ ] Test on OSBS 4x4km smoke test region (UTM, known results to compare)

## Deliverable

pysheds fork that correctly handles both geographic and UTM CRS for HAND/DTND computation and slope/aspect calculation. MERIT validation results unchanged. OSBS smoke test produces correct hydrological DTND and Horn 1981 slope/aspect.

## Log

