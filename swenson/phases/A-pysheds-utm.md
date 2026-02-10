# Phase A: Fix pysheds for UTM

Status: In progress
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

- [x] Fix deprecation warnings in `pgrid.py` (clean working state before making changes)
- [ ] Modify `compute_hand()` to detect UTM CRS and use Euclidean distance for DTND
- [ ] Modify `slope_aspect()` / `_gradient_horn_1981()` to use uniform pixel spacing for UTM
- [ ] Test both changes against MERIT validation (geographic CRS — should reproduce existing results)
- [ ] Test on OSBS 4x4km smoke test region (UTM, known results to compare)

## Deliverable

pysheds fork that correctly handles both geographic and UTM CRS for HAND/DTND computation and slope/aspect calculation. MERIT validation results unchanged. OSBS smoke test produces correct hydrological DTND and Horn 1981 slope/aspect.

## Log

### 2026-02-10: Task 0 — Fix deprecation warnings

Fixed all deprecation warnings in `pgrid.py`. Five changes total:

| Line | Change | Reason |
|------|--------|--------|
| 9 | `from distutils.version` → `from looseversion` | `distutils` removed in Python 3.12 |
| 1271 | `np.in1d` → `np.isin` | `np.in1d` deprecated in NumPy |
| 3050 | `np.in1d` → `np.isin` | same |
| 3063 | `np.in1d` → `np.isin` | same |
| 3876 | `Series._append()` → `pd.concat()` | `_append()` removed in pandas 2.0 |

The `_append` fix was critical — it crashed `resolve_flats()` and caused 11 of 15 tests to fail.

Previous sessions had already fixed 27 warnings (12x `np.warnings` → `warnings`, 13x `np.bool` → `bool`, 2x `np.float` → `np.float64`). STATUS.md's "33 deprecation warnings" count was outdated — the actual remaining count was 36 (3 from `np.in1d` x 11 tests + 1 from `distutils` + 1 from `_append` crash + 1 from `looseversion`).

Result: `pytest tests/test_hillslope.py -v` — 15/15 passed, 0 warnings. Strict mode (`-W error::DeprecationWarning`) also passes.

`ruff check` shows 30 pre-existing lint issues (bare excepts, unused variables, etc.) from Swenson's original code — out of scope for this task.
