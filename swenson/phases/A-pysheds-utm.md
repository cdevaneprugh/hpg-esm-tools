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

**Slope/aspect:** Stage 8 of MERIT validation discovered `np.gradient`-based aspect had a Y-axis sign inversion swapping North/South. The sign bug has been fixed directly in `run_pipeline.py` (`-dzdy` → `dzdy`, see log below), but the pipeline still uses `np.gradient` rather than pgrid's Horn 1981 stencil. The full replacement requires pgrid's `slope_aspect()` to work with UTM coordinates — currently it assumes geographic (haversine spacing).

**Fix location:** `$PYSHEDS_FORK/pysheds/pgrid.py`
- `compute_hand()` lines 1928-1942 (haversine distance)
- `slope_aspect()` / `_gradient_horn_1981()` (haversine spacing)

## Tasks

- [x] Fix deprecation warnings in `pgrid.py` (clean working state before making changes)
- [x] Create synthetic V-valley DEM for testing (1000x1000, UTM CRS, analytically known outputs)
- [ ] Modify `compute_hand()` to detect UTM CRS and use Euclidean distance for DTND
- [ ] Modify `slope_aspect()` / `_gradient_horn_1981()` to use uniform pixel spacing for UTM
- [ ] Validate against synthetic DEM (slope, aspect, HAND, DTND must match analytical expectations)
- [ ] Test both changes against MERIT validation (geographic CRS — should reproduce existing results)
- [ ] Test on OSBS 4x4km smoke test region (UTM, known results to compare)

## Deliverable

pysheds fork that correctly handles both geographic and UTM CRS for HAND/DTND computation and slope/aspect calculation. Validated three ways:
1. **Synthetic V-valley DEM** — analytically known slope, aspect, HAND, DTND in UTM (catches CRS math bugs directly)
2. **MERIT validation** — geographic CRS regression test (existing stages 1-9 reproduce >0.95 correlations)
3. **OSBS smoke test** — real 1m LIDAR in UTM (confirms fix works on actual data)

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

### 2026-02-10: Confirmed pipeline aspect bug (during synthetic DEM planning)

While analyzing the GeoTIFF affine transform convention for the synthetic DEM design, confirmed that the OSBS pipeline's aspect calculation at `run_pipeline.py:1527` has the same N/S swap that stage 8 of MERIT validation discovered.

The bug is in the sign of `dzdy`: `np.gradient` axis 0 gives `d(elev)/d(row) = d(elev)/d(south) = -d(elev)/d(north)`. The pipeline's `arctan2(-dzdx, -dzdy)` double-negates the north component, producing the uphill direction instead of downhill. Correct expression: `arctan2(-dzdx, dzdy)`.

This does not affect slope (sign-independent), HAND, DTND, or flow routing. It only affects aspect and aspect-based binning.

Phase A will fix this in pgrid. Phase D will replace the pipeline's `np.gradient` with the corrected pgrid method. The synthetic V-valley DEM catches this bug: expected east-side aspect is ~268° (west-facing); the buggy formula produces ~272° (3.8° error). The error is small because the cross-slope (0.03) dominates the downstream slope (0.001), so inverting the north component barely changes the angle. The CRS bugs (haversine on UTM) produce much larger errors and are the primary test target.

Updated STATUS.md problem #4 from "not validated" to "confirmed bug" with the full affine sign analysis.

### 2026-02-11: Task 1 — Create synthetic V-valley DEM

Created a synthetic V-valley DEM for validating pysheds UTM CRS handling. Generator script and documentation live in `$PYSHEDS_FORK/data/synthetic_valley/`.

**Files:**
- `generate_synthetic_dem.py` — generates 1000x1000 UTM GeoTIFF with analytically known outputs
- `README.md` — full documentation of elevation formula, D8 routing behavior, HAND/DTND derivations, known limitations
- `synthetic_valley_utm.tif` — generated output (not committed, regenerable)

**Geometry:** Two planar hillslopes (cross_slope=0.03 m/m) meeting at a central N-S channel (downstream_slope=0.001 m/m). Every pixel has a closed-form solution for slope, aspect, HAND, and DTND.

**What it catches:**
| Bug | Detection |
|-----|-----------|
| Haversine DTND on UTM | DTND should be 500m at ridge — haversine on meters gives garbage |
| Haversine gradient spacing on UTM | Slope/aspect wrong from haversine spacing |
| N/S aspect swap | 3.82 deg systematic offset (small because cross_slope >> downstream_slope) |

**What it can't catch:** EDT vs flow-path DTND (V-valley gives same answer for both — would need multi-basin DEM).

Committed and pushed to `uf-development` on the pysheds fork.

### 2026-02-10: Fix aspect sign bug in run_pipeline.py

Applied the aspect sign fix directly to `run_pipeline.py:1527`: changed `arctan2(-dzdx, -dzdy)` to `arctan2(-dzdx, dzdy)`. Rewrote the misleading comments at lines 1523-1526 to document the sign convention clearly. Updated STATUS.md problem #4 from "confirmed bug" to "FIXED" with expanded analysis of the root cause, downstream corruption chain, and history.

This is an interim fix — Phase D will replace `np.gradient` entirely with pgrid's `slope_aspect()` once Phase A makes it UTM-aware. The synthetic V-valley DEM (Phase A task 2) will formally verify the fix.
