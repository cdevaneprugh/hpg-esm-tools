# pgrid.py Refactoring Audit

Date: 2026-02-18
File: `$PYSHEDS_FORK/pysheds/pgrid.py` (4384 lines)
Branch: `feature/utm-crs-support`

## Purpose

Map pgrid.py into load-bearing infrastructure (don't touch), refactoring candidates (should fix), and trade-off decisions (discuss first). This document is the basis for Phase A cleanup work.

Related audit docs in this directory:
- `pysheds-test-audit.md` — test suite quality review (2026-02-17)
- `pysheds-utm-walkthrough.md` — line-by-line CRS change details

---

## 1. File Architecture

pgrid.py is a single `Grid` class containing ~96 methods. It is Swenson's pure-numpy reimplementation of upstream pysheds' sgrid.py, extended with hillslope analysis methods and (as of Phase A) UTM CRS support.

### Method categories

| Category | Count | Lines (approx) | Origin |
|----------|-------|-----------------|--------|
| Upstream-shared (I/O, flow routing, DEM conditioning) | ~50 | ~2400 | Swenson port of sgrid |
| Swenson additions (hillslope, channel, slope/aspect) | ~10 | ~1000 | Swenson original |
| Phase A additions (CRS detection, UTM branches) | ~3 | ~150 | Our work |
| Properties, utilities, internal helpers | ~33 | ~800 | Mixed |

### Dispatch mechanism

`grid.py` selects sgrid (numba) or pgrid (numpy) at import time:
```python
if _HAS_NUMBA:
    from pysheds.sgrid import sGrid as Grid
else:
    from pysheds.pgrid import Grid as Grid  # <-- our 1-line addition
```

pgrid.py is the ONLY file added to the fork. sgrid.py is completely untouched.

---

## 2. Load-Bearing Infrastructure (DO NOT CHANGE)

### 2a. Upstream-shared methods (~50 methods)

These reimplement sgrid.py's functionality in pure numpy. They have **diverged APIs** from sgrid (out_name/inplace pattern vs Raster returns) — they're parallel implementations, not copies.

**Why untouchable:**
- Working correctly for all existing use cases
- No analytical test coverage (geographic DEM tests are smoke-level only)
- Any regression would be caught only by the MERIT validation pipeline, which is slow and coarse
- Changing an upstream method risks breaking flow routing / DEM conditioning in ways that would be hard to diagnose

**Categories:**

| Group | Methods | Lines |
|-------|---------|-------|
| File I/O | `read_raster`, `read_ascii`, `from_raster`, `from_ascii`, `from_array`, `to_raster`, `to_ascii` | 228-400, 2934-3083 |
| Flow direction | `flowdir`, `_d8_flowdir`, `_dinf_flowdir`, `angle_to_d8`, `facet_flow` | 684-1116 |
| Catchment | `catchment`, `_d8_catchment`, `_dinf_catchment`, `detect_cycles` | 890-1116, 3624-3683 |
| Accumulation | `accumulation`, `_d8_accumulation`, `_dinf_accumulation` + cycle helpers | 1196-1552 |
| Flow distance | `flow_distance`, `_d8_flow_distance`, `_dinf_flow_distance` | 1553-1761 |
| DEM conditioning | `fill_pits`, `fill_depressions`, `resolve_flats`, `detect_pits`, `detect_flats`, `detect_depressions`, `_get_nondraining_flats` | 3536-4204 |
| Grid manipulation | `view`, `resize`, `clip_to`, `nearest_cell`, `set_bbox`, `set_indices`, `grid_indices` | 367-661 |
| Cell properties | `cell_distances`, `cell_slopes`, `cell_dh`, `cell_area` | 2340-2595 |
| Geometry | `polygonize`, `rasterize`, `snap_to_mask` | 4281-4384 |
| Internal | `_input_handler`, `_output_handler`, `_check_nodata_in`, `_generate_grid_props`, `_flatten_fdir`, `_unflatten_fdir`, `_construct_matching`, `_convert_grid_indices_crs`, `_dy_dx`, `_inside_indices`, `_select_surround`, `_select_surround_ravel`, `_set_dirmap` | Various |
| DEM conditioning internals | `_grad_from_higher`, `_grad_towards_lower`, `_get_high_edge_cells`, `_get_low_edge_cells`, `_drainage_gradient`, `_d8_diff` | 3814-3943 |

### 2b. `_pop_rim` / `_replace_rim` pattern

12+ paired calls with try/finally throughout the codebase. Temporarily extracts grid border pixels, replaces with nodata, then restores — prevents edge effects during spatial operations.

**Why untouchable:** Pervasive. Changing the pattern would require modifying every method that uses it. The try/finally structure is correct (guarantees restoration). It's verbose but safe.

Used by: `compute_hand` (dinf & d8), `compute_hillslope`, `_d8_catchment`, `_dinf_catchment`, `accumulation` (both), `flow_distance` (both).

### 2c. Bare `except:` clauses in upstream methods (23 instances)

Lines: 112, 450, 471, 629, 797, 864, 1025, 1109, 1349, 1485, 1679, 1756, 2504, 2525, 2601, 2607, 3136, 3195, 3677, 3794, 3802, 3810, 4153.

All follow `except: raise` pattern — functionally passthrough but suppress traceback context. From Swenson's original code.

**Why untouchable:** Same rationale as upstream methods. These are in methods we don't modify and don't have targeted tests for. Fixing them is trivially easy but increases the diff for no practical benefit and risks introducing typos in sensitive code paths.

### 2d. File splitting

Considered and rejected (decision record 2026-02-17). The sgrid/_sgrid split exists because numba requires standalone functions. pgrid is pure numpy — no technical driver for splitting. Two 2200-line files aren't meaningfully better than one 4400-line file.

---

## 3. Refactoring Candidates (SHOULD CHANGE)

These are in Swenson methods we actively develop, have analytical test coverage, and present clear improvement opportunities.

### 3a. Extract `_propagate_uphill()` helper

**Location:** `compute_hillslope()` lines 2173-2237

Three identical 20-line loops:
- Right bank search (2173-2192)
- Left bank search (2195-2214)
- Headwater search (2217-2237)

Each follows the same pattern:
```python
for _ in range(fdir.size):
    selection = self._select_surround_ravel(source, fdir.shape)
    selection[selection > (fdir.size-1)] = fdir.size-1
    selection[selection < 0] = 0
    ix = (fdir.flat[selection] == r_dirmap) & (bank.flat[selection] < 0)
    child = selection[ix]
    if not child.size:
        break
    bank.flat[child] = 1
    source = child
```

Swenson's own TODO at line 2229: "Not optimized (a lot of copying here)".

**Proposed extraction:**
```python
def _propagate_uphill(self, fdir, source, bank, r_dirmap):
    """Propagate classification uphill from source pixels via D8 flow directions."""
    for _ in range(fdir.size):
        selection = self._select_surround_ravel(source, fdir.shape)
        selection[selection > (fdir.size - 1)] = fdir.size - 1
        selection[selection < 0] = 0
        ix = (fdir.flat[selection] == r_dirmap) & (bank.flat[selection] < 0)
        child = selection[ix]
        if not child.size:
            break
        bank.flat[child] = 1
        source = child
    return source
```

Each call site becomes one line. ~40 lines saved. Behavior identical.

**Risk:** Low. `compute_hillslope` has 8 tests (TestHillslopeClassification in test_utm.py + tests in test_hillslope.py) that validate all 4 classification types. Any regression would be caught.

**Effort:** Small (extract function, replace 3 call sites, run tests).

### 3b. Module-level constants

**Problem:** `re = 6.371e6` and `dtr = np.pi/180` are redefined locally 7 times across 4 methods:

| Constant | Lines |
|----------|-------|
| `re = 6.371e6` | 1992, 3293, 4245 |
| `dtr = np.pi/180` | 1991, 2039, 3292, 4246 |

**Fix:** Define once at module level (after imports, before class):
```python
_EARTH_RADIUS_M = 6.371e6
_DEG_TO_RAD = np.pi / 180
```

Then replace all 7 local definitions with the module constants.

**Risk:** Trivial. Pure constant substitution, no logic change.

**Effort:** Trivial (define 2 constants, replace 7 lines).

### 3c. Bare `except:` in Swenson methods (4 instances)

Lines 1919, 2057, 2260, 2336 — all in methods we actively develop and test:
- `compute_hand()` dinf branch (1919)
- `compute_hand()` d8 branch (2057)
- `compute_hillslope()` (2260)
- `slope_aspect()` (2336)

All follow `except: raise` — the try/except/raise pattern is a no-op. The `raise` re-raises the original exception, but a bare `except:` catches `KeyboardInterrupt` and `SystemExit` too.

**Fix options:**
1. Remove the try/except entirely (it does nothing)
2. Change to `except Exception: raise` (excludes KeyboardInterrupt/SystemExit)
3. Change to `except Exception as e: raise` (preserves traceback explicitly)

These are all wrapped around `_replace_rim` calls in `finally` blocks, so the try/except structure is part of the _pop_rim pattern (section 2b). The bare `except` is the last handler before `finally`. Looking more carefully:

```python
try:
    ... main logic ...
    self._output_handler(...)
except:
    raise
finally:
    self._replace_rim(...)
```

The `except: raise` is redundant with the `finally` — exceptions propagate naturally. But removing the `except` block entirely is the cleanest fix since `finally` always runs regardless.

**Risk:** Low. Functionally identical. Tests cover all 4 methods.

**Effort:** Trivial (remove 2 lines x 4 sites = 8 lines removed).

---

## 4. Trade-Off Decisions (DISCUSS FIRST)

These have valid arguments for and against. Document the analysis; decide together.

### 4a. CRS distance helper extraction

**Original scope (Phase A task):** "Extract CRS distance helper to consolidate 4 duplicated haversine/Euclidean branches."

**What the exploration found:** The 4 branch sites compute fundamentally different things:

| Site | Method | Lines | Computation | Array shape |
|------|--------|-------|-------------|-------------|
| 1 | `compute_hand()` DTND | 1985-2011 | Point-to-drainage distance | Full grid via hndx |
| 2 | `compute_hand()` AZND | 2035-2052 | Point-to-drainage bearing | Full grid via hndx |
| 3 | `river_network_length_and_slope()` | 3289-3301 | Reach segment length | 1D profile arrays |
| 4 | `_gradient_horn_1981()` | 4241-4261 | Pixel spacing for stencil | 8-neighbor arrays, takes abs() |

**What a helper could cover:**
- Sites 1 and 3 both compute haversine/Euclidean distance between two points. A `_point_distance(dx, dy, y1, y2)` helper could serve both — but they operate on different array shapes (full grid vs 1D profile segments).
- Site 2 computes bearing (arctan2), not distance. Different formula entirely.
- Site 4 computes spacing magnitude (abs of coordinate diffs). The haversine branch includes a cos(lat) correction that doesn't appear in the others.

**Options:**

| Option | What changes | Lines saved | Risk |
|--------|-------------|-------------|------|
| A. Constants only | Promote `re`/`dtr` to module level (3b above). Leave 4 branches as-is. | 0 (but 7 fewer redefinitions) | None |
| B. Distance helper | Extract `_haversine_distance()` for sites 1 and 3. Constants + helper. | ~10 | Low — but two call sites with different shapes may need different signatures |
| C. Full CRS helper | Extract `_crs_distance()` and `_crs_bearing()`. Constants + two helpers. | ~20 | Medium — abstracts away context-specific comments that currently explain each site's semantics |

**Current recommendation:** Option A (constants only). Each branch site has detailed comments explaining its specific context — abstracting them into a helper would lose that documentation value. The duplication is structural (each site genuinely does something different) rather than copy-paste (identical code that drifted apart).

### 4b. Rename `_2d_geographic_coordinates()` → `_2d_crs_coordinates()` ✅

**Problem:** The name implies geographic CRS only, but after Phase A it returns coordinates in whatever CRS the grid uses (lon/lat for geographic, easting/northing for UTM). The Phase A docstring update clarifies this, but the name is still misleading.

**Options:**
- Rename to `_2d_crs_coordinates()` — touches 3 call sites + the definition (4 changes total)
- Leave the name, rely on the docstring
- Rename to `_2d_pixel_coordinates()` (describes what it does: maps pixel indices to CRS coordinates)

**Risk:** Low (pure rename, no logic). But it's a name that appears in the existing audit docs and Phase A log entries — renaming creates a documentation disconnect.

**Recommendation:** Rename to `_2d_crs_coordinates()` during the refactoring. Update references in audit docs.

### 4c. Haversine formula deduplication

**Problem:** The haversine formula appears twice:
- `compute_hand()` DTND (lines 1993-1996) — operates on full grid arrays
- `river_network_length_and_slope()` (lines 3294-3297) — operates on 1D profile segments

Both compute the same thing: `re * 2 * arctan2(sqrt(a), sqrt(1-a))` where `a = sin²(dy/2) + cos(y1)*cos(y2)*sin²(dx/2)`.

**Trade-off:**
- **For extraction:** DRY principle. If we ever fix the formula (unlikely — haversine is well-known), we'd fix it in two places.
- **Against extraction:** The two sites use different variable names, array shapes, and context. A helper would need `(dx, dy, y1, y2)` parameters — essentially the same code just wrapped in a function call. Marginal reduction in readability for marginal DRY compliance.

**Recommendation:** Skip unless combined with 4a option B. The formula is stable (haversine hasn't changed since Euler) and appears only in geographic CRS branches that we don't actively modify.

### 4d. `slope_aspect()` input mutation

**Problem:** Line 2315 modifies the input DEM in-place:
```python
dem.flat[dem_mask] = dem.max() + 1
```

This sets nodata pixels to `max+1` before computing the Horn 1981 gradient, preventing false gradients at nodata boundaries. It's intentional — the pipeline computes slope from the original DEM (not the pysheds-conditioned one) precisely to avoid this.

**Trade-off:**
- **For fixing:** Side effects are surprising. A caller might not expect their DEM array to be modified.
- **Against fixing:** Operating on a copy allocates a full DEM-sized array. For the OSBS interior mosaic (90M pixels at float64 = 720MB), this is non-trivial. The mutation is documented in the method's workflow and is standard practice in pysheds (the upstream methods also mutate grids via `_pop_rim`).

**Recommendation:** Skip. The mutation is intentional, documented, and consistent with pysheds' mutable-grid design. The risk of introducing a subtle bug (e.g., copying only the view, not the underlying data) outweighs the cleanliness gain.

---

## 5. Test Coverage Map

### Well-covered (analytical tests with closed-form solutions)

| Method | Tests | Files | Quality |
|--------|-------|-------|---------|
| `compute_hand()` — HAND | 7 | test_utm, test_split_valley, test_depression_basin | Analytical values, cross-validated with DTND |
| `compute_hand()` — DTND | 6 | test_utm, test_split_valley | Analytical, catches EDT bug, divergence zone |
| `compute_hand()` — AZND | 3 | test_utm | Cardinal directions (~90/270/180 deg) |
| `slope_aspect()` | 3 | test_utm | Magnitude and direction, analytical |
| `_gradient_horn_1981()` | (indirect via slope_aspect) | test_utm | CRS branching validated |
| `compute_hillslope()` | 6 | test_utm, test_hillslope | All 4 types present, bank separation |
| `fill_depressions()` | 4 | test_depression_basin | Spill elevation, non-lowering |
| `resolve_flats()` | 2 | test_depression_basin | No interior flats remain |
| `flowdir()` | 5 | test_utm, test_split_valley, test_depression_basin | Valid directions, topology |
| `_crs_is_geographic()` | 2 | test_utm | Geographic vs projected detection |

### Weak coverage — RESOLVED (2026-02-18, commit `ac71d9a`)

All 3 methods now have analytical tests with closed-form validation:

| Method | Tests | Files | Resolution |
|--------|-------|-------|------------|
| `river_network_length_and_slope()` | 3 | test_utm | Validates segment length, slope, reach count against V-valley geometry |
| `extract_profiles()` | 3 | test_utm | Profile length matches D8 distance, validates profile values |
| `create_channel_mask()` | 3 | test_utm | Center column classification, non-channel exclusion, ID uniqueness |

### No coverage (upstream methods — leave alone)

| Method | Notes |
|--------|-------|
| `cell_area()` | Would be needed for pipeline hillslope binning validation |
| `cell_distances()`, `cell_dh()`, `cell_slopes()` | Upstream cell property methods |
| `catchment()` | Catchment delineation to pour point |
| `to_raster()`, `to_ascii()` | Output serialization |
| `polygonize()`, `rasterize()`, `snap_to_mask()` | Geometry operations |
| `view()`, `resize()`, `clip_to()` | Grid manipulation |

### Test infrastructure notes

- **82 total executable tests** across 4 files (test_utm: 22, test_split_valley: 16, test_depression_basin: 17, test_hillslope: 16, test_grid: all skipped)
  - Note: agent counts varied (22 vs 41 for test_utm) due to counting individual assertions vs test methods. The 22 count reflects `pytest -v` output.
- **3 synthetic DEMs** with closed-form analytical solutions (V-valley, split valley, depression basin)
- **1 real geographic DEM** (`data/dem.tif`) used by test_hillslope and conftest fixtures
- **conftest.py** provides shared fixtures (dem, fdir, grid, paths, d) for geographic DEM tests
- **No pytest.ini or pyproject.toml** — uses default pytest discovery

---

## 6. Recommendations Summary

| # | Item | Risk | Effort | Value | Recommendation |
|---|------|------|--------|-------|----------------|
| 3a | Extract `_propagate_uphill()` | Low | Small | High | **DO** |
| 3b | Module-level constants (`re`, `dtr`) | Trivial | Trivial | Medium | **DO** |
| 3c | Remove bare `except: raise` in Swenson methods | Trivial | Trivial | Low | **DO** |
| 4a | CRS distance helper | Medium | Medium | Low | **SKIP** (constants suffice) |
| 4b | Rename `_2d_geographic_coordinates` → `_2d_crs_coordinates` | Low | Small | Low | **DONE** |
| 4c | Haversine deduplication | Low | Small | Low | **SKIP** |
| 4d | Fix `slope_aspect` mutation | Medium | Small | Low | **SKIP** |

### Proposed refactoring order

1. Module-level constants (3b) — smallest change, touches all 4 CRS sites
2. Extract `_propagate_uphill` (3a) — largest structural improvement
3. Remove bare excepts (3c) — cleanup
4. Rename `_2d_geographic_coordinates` → `_2d_crs_coordinates` (4b) — ✅ done

Run full test suite after each step. Total estimated diff: ~60 lines changed, ~40 lines removed.

---

## 7. What Ruff Says

`ruff check` on pgrid.py shows 30 pre-existing lint issues from Swenson's original code (bare excepts, unused variables, star imports). All are in upstream methods. Out of scope for this refactoring — consistent with section 2's "don't touch upstream" principle.

The 4 bare excepts in Swenson methods (section 3c) would be caught by ruff's `E722` (bare except). Fixing them aligns with our linting standards without touching upstream code.
