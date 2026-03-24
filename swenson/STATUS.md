# State of the Union: Swenson Hillslope Implementation for OSBS

Date: 2026-03-23

## Executive Summary

We are implementing Swenson & Lawrence (2025) representative hillslope methods to generate custom hillslope parameters for OSBS using 1m NEON LIDAR data. The goal is to replace the current global 90m MERIT-derived hillslope file (`hillslopes_osbs_c240416.nc`) with site-specific parameters that capture the fine-scale drainage structure of this low-relief wetlandscape.

**The methodology is validated and the pipeline produces scientifically defensible output.** MERIT validation achieved >0.95 correlation with Swenson's published data on 5 of 6 parameters. Phases A-D are complete: pysheds handles UTM CRS, flow routing runs at full 1m resolution, Lc is established (356m for the production domain), and the pipeline has been rebuilt with all known fixes and verified by equation-by-equation audit. Output is CTSM-compatible NetCDF matching the Swenson reference structure.

**Remaining work is in Phase E (parameter completion).** Hillslope structure is 1 aspect x 4 equal-area HAND bins (Swenson's method, interim while water masking is developed). NEON slope/aspect products adopted. Stream channel parameters use interim power-law scaling. DEM conditioning and study boundary are open questions for the PI. Phase F (CTSM validation) follows.

---

## What Is Done Well

### Methodology is validated against published data

The MERIT validation (stages 1-9) demonstrated that our pysheds fork and pipeline correctly reproduce Swenson's results:

| Parameter | Correlation with published |
|-----------|---------------------------|
| Height (HAND) | 0.9979 |
| Distance (DTND) | 0.9992 |
| Slope | 0.9839 |
| Aspect | 1.0000 (circular) |
| Width | 0.9919 |
| Area fraction | 0.9244 |

This gives confidence that the *approach* is correct — we understand Swenson's methodology and can implement it. (Correlations updated 2026-02-20 after full pipeline audit, resolve_flats fallback fix, DTND tail removal, and 6 prior fixes including n_hillslopes bug, DEM conditioning chain, and catchment-level aspect averaging.)

### Bugs found and fixed during validation

- **Width calculation** (stage 6): Fixed from 0.09 to 0.96 correlation. Root cause was using raw pixel areas instead of fitted trapezoidal areas.
- **Polynomial fit weighting** (merit_regression.py, area_fraction_diagnostics.py): Our `lstsq`-based trapezoidal fit applied w^2 weighting where Swenson's `_fit_polynomial` applies w^1. Fixed to match Swenson's normal equations. Area fraction improved +0.006 (0.8157→0.8215), width changed from 0.9604 to 0.9410. See `scripts/merit_validation/area_fraction_research.md` for full analysis.
- **North/South aspect swap** (stage 8): Fixed a Y-axis sign inversion in the gradient calculation that systematically swapped N and S aspects. Area fraction correlation improved from 0.64 to 0.82. The fix (switching to pgrid's `slope_aspect()`) only applied to the MERIT validation scripts — the OSBS pipeline could not use pgrid because its haversine math fails on UTM data. The same bug persisted in `run_pipeline.py` until an interim sign fix was applied (see problem #4). Full resolution requires Phase A (UTM-aware pgrid) and Phase D (replace `np.gradient` with Horn 1981 stencil).
- **Gridcell alignment** (stage 7): Fixed region extraction to match the published 0.9x1.25 degree gridcell exactly.
- **Mandatory 2m HAND bin constraint** (stage 7): Corrected our optional implementation to match Swenson's mandatory constraint per the paper.

### Pipeline engineering is solid

- **Connected-component extraction** correctly isolates the largest contiguous data region, avoiding flow fragmentation across mosaic gaps.
- **Nodata edge trimming** solves the pysheds edge handling problem (all-nodata edges cause max_accumulation=1).
- **NetCDF structure** matches the Swenson reference file exactly — verified via `ncdump`.
- **Circular mean** for aspect correctly handles 0/360 wraparound.
- **Slope computed from original DEM**, not the pysheds-conditioned DEM (prevents false gradients at nodata boundaries from pysheds' high-value fill).
- **Width fix** properly implemented in the OSBS pipeline using fitted trapezoidal areas with the quadratic solver.

### Data infrastructure is in place

- 233 NEON DTM tiles downloaded (1m, EPSG:32617, 19x17 km)
- 231 NEON slope/aspect tiles (DP3.30025.001) — all 90 production tiles have matching data
- Full mosaic stitched (`OSBS_full.tif`, 17000x19000 pixels)
- **Production domain:** R4-R12, C5-C14 (90 tiles, 9x10 km, 0 nodata pixels out of 90M). This is the largest contiguous rectangle of fully valid tiles. Validated in Phase C and used for all tier 3 runs.
- Tile nodata coverage documented (`data/neon/tile_coverage.md`)
- Tile reference system documented (R#C# format, KML for Google Earth)
- osbs2 baseline case identified for future comparison (860+ year spinup)

### Shared modules extracted (Phase D)

- **`hillslope_params.py`**: Binning, trapezoidal fit, width computation — used by both MERIT validation and OSBS pipeline (resolves former code duplication)
- **`spatial_scale.py`**: FFT-based Lc computation, dual-CRS (geographic + UTM)
- **`dem_processing.py`**: Basin detection, open water identification

### NEON slope/aspect products adopted (Phase E)

PI approved using NEON DP3.30025.001 slope/aspect products directly, replacing pgrid Horn 1981 computation on the raw DTM. Scientific rationale:

1. **Reduced TIN interpolation noise on flat terrain.** NEON applies a 3x3 pre-filter before Horn 1981. OSBS is low-relief — flat areas dominate, and raw 1m gradients are noisiest there. The +0.008 m/m slope bias (pgrid steeper) is consistent with pgrid amplifying noise that NEON smooths.
2. **No border artifacts.** NEON computes with 20m tile-edge buffer — all pixels have valid slope/aspect. pgrid zeros the outermost pixel ring (Horn 1981 3x3 stencil).
3. **Authoritative provenance.** Published, versioned data product with ATBD (NEON.DOC.003791vB). Citable source vs local computation.
4. **Better aspect on flat terrain.** Pre-smoothing gives physically meaningful aspect values where raw gradients are noise-dominated. Matters for catchment-mean aspect used by CTSM insolation correction (shr_orb_cosinc).
5. **Consistency with DTM product.** Both from same LIDAR collection and processing pipeline.

Comparison across 90 production tiles (commit 418880c): slope Pearson r=0.91, aspect circular r=0.84. See `output/osbs/slope_aspect_comparison/`.

---

## Problems, In Order of Importance

### 1. DTND uses the wrong algorithm — RESOLVED

**Status: RESOLVED in Phase A.** Pipeline now uses `grid.compute_hand()` (line 1073) which is UTM-aware. The EDT workaround has been removed.

**Impact:** Corrupted the distance parameter for all hillslope columns.

**File:** `scripts/osbs/run_pipeline.py` (formerly lines 1480-1483, now resolved)

The pipeline computes DTND using `scipy.ndimage.distance_transform_edt` — Euclidean distance to the *geographically nearest* stream pixel. Swenson's method computes distance to the *hydrologically nearest* stream pixel (the one each pixel actually drains to via D8 routing). These differ whenever a pixel is geographically close to a stream on the opposite side of a divide.

The workaround exists because pysheds' `compute_hand()` computes the correct hydrologically-linked distance internally, but uses haversine math that assumes geographic coordinates. Our UTM data produces garbage distances through haversine.

Neither DTND is currently correct:

| Approach | Concept | Math |
|----------|---------|------|
| pysheds DTND | Correct (flow-path-linked) | Wrong (haversine on UTM) |
| Pipeline EDT | Wrong (nearest regardless of drainage) | Correct (Euclidean on UTM) |

**Fix path:** Modify `compute_hand()` in `$PYSHEDS_FORK/pysheds/pgrid.py` (lines 1928-1942) to detect UTM CRS and use Euclidean distance instead of haversine. Alternatively, expose the `hndx` array (which maps each pixel to its drainage stream pixel — already computed at line 1919) and compute UTM distance externally.

**Downstream effect:** DTND feeds directly into the hillslope distance parameter and the trapezoidal width fit (via A_sum(d) curve). Wrong DTND means wrong widths and wrong distances for all columns.

### 2. Flow routing resolution (4x subsampling) — RESOLVED

**Impact:** Discards 93.75% of the 1m LIDAR data before the core hydrology computation.

**File:** `scripts/osbs/run_pipeline.py` line 1354

**Status: RESOLVED.** Full 1m resolution works at 64GB (peak 29.2 GB, 5.9 min for 90M pixels). Resolution comparison across 1m/2m/4m on two domains confirms 1m is the correct choice: height and distance correlations >0.999 across resolutions (resolution-insensitive), slope systematically underestimated at coarser resolutions, and computational cost is not a barrier (17 min / 58 GB for the full domain). No subsampling will be used. See `phases/B-flow-resolution.md` for full results.

### 3. Characteristic length scale (Lc) — RESOLVED

**Impact:** Everything downstream depends on Lc — accumulation threshold, stream network density, HAND, DTND, all 6 parameters.

**File:** `scripts/osbs/run_pipeline.py` lines 1275-1287

**Status: RESOLVED.** Lc = 300m (range 285-356m), A_thresh = 45,000 m². Confirmed by restricted-wavelength FFT, PI acceptance, and physical validation. See `phases/C-characteristic-length.md` for full analysis.

**Phase C established that the Laplacian spectrum has two real features at 1m resolution:** a micro-topographic peak at ~8m (k² amplification artifact) and a drainage-scale peak at ~285-356m (visible when short wavelengths are excluded). See `output/osbs/phase_c/` for plots.

**Lc comparison across all datasets:**

| Dataset | Lc source | Lc value | A_thresh |
|---------|-----------|----------|----------|
| MERIT (90m) | FFT peak | 763m | 275,400 m² |
| OSBS (full, 4x sub) | Forced minimum | 100m | 5,000 m² |
| OSBS (interior, 4x sub) | FFT peak (4m res) | 166m | 13,778 m² |
| OSBS (interior, 1m, full range) | FFT peak (artifact) | 8.1m | 33 m² |
| **OSBS (interior, 1m, cutoff>=20m)** | **FFT peak** | **356m** | **63,368 m²** |
| **OSBS (interior, 1m, cutoff>=100m)** | **FFT peak** | **285m** | **40,612 m²** |

**Production Lc by domain size (1m, min_wavelength=20m):**

| Tier | Domain | Lc | A_thresh |
|------|--------|----|----------|
| 1 (R6C10) | 1 tile, 1M px | 541m | 146,156 |
| 2 (R6-R10, C7-C11) | 25 tiles, 25M px | 479m | 114,631 |
| **3 (R4-R12, C5-C14)** | **90 tiles, 90M px** | **356m** | **63,362** |

Lc varies with domain size because larger domains include more drainage structure, shifting the FFT peak. The production value (tier 3) is 356m.

**Physical validation (2026-02-17, interpretation closed 2026-02-23):** Run on 5x5 tile block (R6-R10, C7-C11, 25M pixels at 1m). Both checks pass:
- **Check 1 (DTND vs Lc): PASS.** P95 DTND/Lc = 1.17 ("similar magnitude" per paper). The original max(DTND)/Lc = 3.1 appeared to fail, but `max()` is not comparable between 90m MERIT (~12K pixels, implicitly smoothed) and 1m OSBS (25M pixels, raw). At 90m, each pixel averages a 90x90m area, blunting ridge extremes; at 1m, individual ridgeline pixels are preserved. The 2000x sample size difference shifts the extreme value rightward. P95 is the resolution-fair comparison.
- **Check 2 (mean catchment / Lc²): PASS.** Ratio = 0.876, close to Swenson's 0.94 calibration.

Results: `output/osbs/smoke_tests/lc_physical_validation/`.

### 4. Slope/aspect: N/S aspect swap in OSBS pipeline — RESOLVED

**Status: RESOLVED in Phase A/D, superseded by Phase E.** Pipeline now loads NEON DP3.30025.001 slope/aspect products directly (lines 607-619), bypassing pgrid slope/aspect computation entirely. The `np.gradient` workaround, the interim sign fix, and the pgrid `slope_aspect()` call have all been removed.

**Impact:** Corrupted aspect for all pixels, which corrupted aspect-based binning and area fractions for all hillslope columns.

**File:** `scripts/osbs/run_pipeline.py` (formerly lines 1514-1528, now resolved)

#### How the bug happened

The OSBS pipeline cannot use pgrid's `slope_aspect()` because `_gradient_horn_1981()` internally computes haversine distances, which produce garbage on UTM data. So the pipeline uses `np.gradient(dem, pixel_size)` as a workaround and applies the same `arctan2(-dzdx, -dzdy)` formula from pgrid.

The problem: pgrid and `np.gradient` return `dzdy` with **opposite sign conventions**, and the formula was copied without adjusting for this.

**pgrid's `_gradient_horn_1981()`** references neighbors by compass direction (N, NE, E, ...) and normalizes spacings with `abs()`:
```python
dy = re * np.abs(dtr * dlat)  # always positive regardless of data ordering
```
Combined with the explicit neighbor ordering, this always returns `dzdy = d(elev)/d(north)` — positive means elevation increases northward.

**`np.gradient`** computes finite differences along array axes, following array indexing order. In a standard GeoTIFF (row 0 = north, row index increases southward):
- `dzdy = d(elev)/d(row) = d(elev)/d(south) = -d(elev)/d(north)` — **opposite sign** from pgrid

The pipeline's original comment showed the reasoning that led to the error:
```python
# Note: numpy y-axis increases downward (row index), x-axis increases rightward (col index)
# For geographic North (up on map), we need -dzdy for the y component
```
The author correctly recognized that dzdy follows rows (southward) and that `-dzdy` converts to d(elev)/d(north). But the aspect formula needs `north_downhill` = **-**d(elev)/d(north) = dzdy (no negation). The extra negation in the formula produces `north_uphill` instead — one level of negation too many.

#### What the bug does

The bug reflects aspect across the east-west axis (the 90°-270° line):

| Correct | Buggy | Error |
|---------|-------|-------|
| 0° (N) | 180° (S) | 180° |
| 45° (NE) | 135° (SE) | 90° |
| 90° (E) | 90° (E) | 0° |
| 135° (SE) | 45° (NE) | 90° |
| 180° (S) | 0° (N) | 180° |
| 270° (W) | 270° (W) | 0° |

Pure east/west aspects are unaffected. Everything else is corrupted, with maximum error (180°) for pure north/south aspects. At OSBS with nearly uniform aspect distribution, this systematically swaps N-bin and S-bin pixel assignments.

#### What this does NOT affect

Slope magnitude (`sqrt(dzdx² + dzdy²)` — sign-independent), HAND (elevation differences), DTND (distances, always positive), flow routing (compares neighbor elevations directly).

#### What this DOES affect — downstream corruption chain

Everything downstream of line 1527 operated on wrong aspect values:

| Step | Lines | Corruption |
|------|-------|-----------|
| Aspect binning | 1587 | Pixels assigned to wrong N/E/S/W hillslope |
| Area fractions | 1619-1657 | Wrong pixel counts per aspect bin |
| Trapezoidal width fit | 1624-1626 | Fit on wrong pixel populations |
| Circular mean aspect | 1681 | Mean computed from wrong pixel sets |
| All 6 params × 16 elements | 1659-1710 | height, distance, area, slope, aspect, width |
| NetCDF output | 934-1182 | All hillslope variables in output file |

The `circular_mean_aspect()` function (line 745) and `get_aspect_mask()` function (line 576) are themselves correct — they just received wrong input. No changes to those functions were needed.

#### Where the bug did NOT occur

- **pgrid's `slope_aspect()`** (`pgrid.py:2223`): Uses the same `arctan2(-dzdx, -dzdy)` formula, but correct because pgrid's `_gradient_horn_1981()` returns `dzdy = d/d(north)`.
- **MERIT validation** (`stage3_hillslope_params.py:800`): Uses pgrid's `slope_aspect()` directly — correct.
- **`calc_gradient_utm()`** (`run_pipeline.py:275`): Only used for FFT Laplacian computation. The Laplacian is a sum of second derivatives where signs cancel — not affected.

#### History

Stage 8 of MERIT validation discovered the N/S swap and fixed it by switching to pgrid's `slope_aspect()`. That fix could not be applied to the OSBS pipeline because pgrid uses haversine math that fails on UTM data.

**NEON validation data:** NEON provides precalculated slope/aspect rasters (DP3.30025.001) which could serve as an additional validation baseline — see PI question #5.

### 5. DEM conditioning may erase real geomorphic features

**Impact:** At 1m resolution, pits and depressions include real features: sinkholes, wetland depressions, karst dissolution. Filling them forces a continuous drainage network but destroys information about closed basins that are central to OSBS hydrology.

**File:** `scripts/osbs/run_pipeline.py` lines 932-933, 963

**Status:** This is a science question for the PI, not a bug. Standard D8 flow routing requires a depression-free DEM. Alternative approaches (like RICHDEM's priority-flood with depression retention) exist but add complexity.

### 6. Stream channel parameters use interim scaling — PARTIALLY RESOLVED

**Impact:** Directly affects CTSM lateral subsurface flow and stream-groundwater exchange.

**File:** `scripts/osbs/run_pipeline.py` lines 1413-1432

**Status: Improved but still interim.** No longer hardcoded guesses. Stream slope is now computed from the actual stream network elevation profile (line 1415). Depth and width use power-law scaling from total drainage area (lines 1431-1432: `depth = 0.001 * A^0.4`, `width = 0.001 * A^0.6`), following Swenson's approach (`rh:1104-1114`). Phase E will research OSBS-specific empirical relationships.

| Parameter | Old (hardcoded) | Current (interim) | Swenson reference |
|-----------|-----------------|-------------------|-------------------|
| Stream depth | 0.3 m | power-law from drainage area | 0.269 m |
| Stream width | 5.0 m | power-law from drainage area | 4.414 m |
| Stream slope | heuristic | computed from stream network | 0.00233 |

### 7. Bedrock depth is a placeholder — RESOLVED

**Status: RESOLVED (commit 11f465e).** Pipeline now uses zeros (line 553: `bedrock_depth = np.zeros()`), matching Swenson's reference file. CTSM ignores this field when using default soil depth parameters.

**File:** `scripts/osbs/run_pipeline.py` line 553

Pipeline previously used `1e6` (effectively infinite). Swenson reference has all zeros. Updated to match reference.

### 8. FFT parameters untested for OSBS — RESOLVED

**Status:** Resolved by Phase C sensitivity sweep. **Lc is insensitive to all tested parameters.**

**File:** `scripts/osbs/run_pipeline.py` lines 80-85, 489-506

Phase C ran the full test matrix (tests A-E, 20 configurations) at full 1m resolution on the interior mosaic. Lc ranged 8.0-9.8 m across all tests, with the lognormal model selected unanimously. No parameter needs calibration — the spectral peak is strong enough (psharp 9-12) that preprocessing choices don't affect the result.

| Test | Variable | Values tested | Lc range (m) |
|------|----------|---------------|---------------|
| A | blend_edges | 4, 25, 50, 100, 200 | 8.1 - 9.6 |
| B | zero_edges | 5, 20, 50, 100 | 8.1 - 9.8 |
| C | NLAMBDA | 20, 30, 50, 75 | 8.0 - 8.9 |
| D | MAX_HILLSLOPE_LENGTH | 500, 1000, 2000, 10000 | 8.1 (stable) |
| E | detrend | True, False | 8.1 (stable) |

Region size test (F) was deferred as unnecessary given stability. Full results in `phases/C-characteristic-length.md`.

The remaining question is not parameter sensitivity but *interpretation* of the 8m peak — see problem #3.

### 9. ~400 lines of duplicated code — RESOLVED

**Status: RESOLVED in Phase D (commit a51b0e3, 624f05d).** Shared modules extracted. Both pipelines now import from the same source.

**Files:** Formerly `run_pipeline.py` lines 275-568 (spatial scale), lines 666-742 (hillslope computation)

Both the spatial scale analysis functions and the hillslope computation functions were duplicated between the merit_validation scripts and the OSBS pipeline. Any fix had to be applied twice.

**Implemented architecture:**

| Layer | Responsibility | Location |
|-------|---------------|----------|
| pysheds (fork) | Flow routing, HAND, DTND, slope/aspect | `$PYSHEDS_FORK/pysheds/pgrid.py` |
| `hillslope_params.py` | Binning, trapezoidal fit, width, 6-param computation | `scripts/hillslope_params.py` |
| `spatial_scale.py` | FFT-based Lc computation (dual-CRS) | `scripts/spatial_scale.py` |
| `dem_processing.py` | Basin detection, open water identification | `scripts/dem_processing.py` |
| Pipeline scripts | Orchestration, I/O, plotting | `scripts/osbs/run_pipeline.py`, `scripts/merit_validation/merit_regression.py` |

`dirmap` (D8 flow direction mapping) has been moved to the pysheds fork as a constant.

### 10. pysheds fork: deprecation warnings and test suite — RESOLVED

**Status: RESOLVED in Phase A.** All 33 deprecation warnings fixed. Test suite expanded with 28 synthetic DEM tests and mutation testing (30 mutations, 100% effective score). `test_hillslope.py` and `test_grid.py` both pass.

**File:** `$PYSHEDS_FORK/pysheds/pgrid.py`

**Previous state:** 33 deprecation warnings in `pgrid.py` from Swenson's older code created noise that obscured real issues during development. `test_grid.py` had pre-existing API mismatch failures.

**Resolution:** Addressed during Phase A alongside UTM CRS changes. Bare `except: raise` patterns removed, deprecated variable names updated, scipy import quirk resolved.

---

## Work Flow: What Needs to Happen and In What Order

**Phase tracking files:** `phases/` — one file per phase with tasks, results, and decisions.

### Phase A: Fix pysheds for UTM — Complete

**Status: Complete.** Both the DTND problem (#1) and the slope/aspect problem (#4) stemmed from the same root cause: pysheds assumed geographic coordinates. Phase A made pysheds UTM-aware, resolving both.

**Completed:**
1. Fixed 33 deprecation warnings in `pgrid.py`
2. Modified `compute_hand()` to detect UTM CRS and use Euclidean distance for DTND
3. Modified `slope_aspect()` / `_gradient_horn_1981()` to use uniform pixel spacing for UTM
4. Validated against MERIT regression (PASS — all 6 parameters within tolerance)
5. Validated on R6C10 UTM smoke test (14/14 checks pass)
6. Added 28 synthetic DEM tests, mutation testing (30 mutations, 100% effective score)

**Deliverable:** pysheds fork that correctly handles both CRS types. See `phases/A-pysheds-utm.md`.

### Phase B: Resolve flow routing resolution — Complete (1m, no subsampling)

**Status:** Complete. Full 1m resolution at 64GB. See `phases/B-flow-resolution.md`.

**Completed:**
1. Scalability test: 90M pixels at 1m completes in 5.9 min, peak 29.2 GB (64GB allocation)
2. Resolution comparison: 1m/2m/4m on 5x5 block (25M px) and full contiguous region (90M px)
3. Height and distance >0.999 correlated across all resolutions (resolution-insensitive)
4. Slope systematically underestimated at coarser resolutions (~50% lower at 4m in lowest HAND bin)
5. No parameter improves with subsampling; computational cost is negligible (17 min / 58 GB at 1m)

**Deliverable:** Use 1m resolution, no subsampling. The 4x subsampling was a premature optimization.

### Phase C: Establish trustworthy Lc — Complete (Lc = 356m production)

**Status:** Complete. Lc confirmed by spectral analysis, PI acceptance, and physical validation. Production value (tier 3, 90 tiles): 356m, A_thresh = 63,362. See `phases/C-characteristic-length.md`.

**Completed:**
1. Full-resolution (1m) FFT on interior mosaic — Laplacian peak at 8.1m (artifact)
2. Sensitivity sweep (tests A-E) — Lc insensitive to FFT parameters
3. Code audit — implementation verified against Swenson's original
4. Raw DEM spectrum test — confirms 8.1m is k² artifact, raw spectrum is red noise
5. Single-tile FFT (R6C10) — confirms spectral features are intrinsic, not mosaic artifacts
6. Restricted wavelength sweep — **200-500m hump IS a real peak at cutoff >= 20m: Lc = 285-356m**
7. Tile coverage documented (`data/neon/tile_coverage.md`)
8. PI decision: ~300m accepted as working value
9. Physical validation on 5x5 tile block (R6-R10, C7-C11, 25M pixels at 1m) — Check 1 PASS (P95 DTND/Lc = 1.17), Check 2 PASS (mean catchment/Lc² = 0.876). The original max(DTND)/Lc = 3.1 "failure" was a resolution mismatch: `max()` is not comparable between 90m MERIT (~12K pixels) and 1m OSBS (25M pixels) due to extreme value statistics and implicit smoothing at 90m. P95 is the fair comparison.

**Deliverable:** Lc = 300m with scientific justification. A_thresh = 45,000 m².

### Phase D: Rebuild pipeline with fixes — Complete

**Status: Complete.** Pipeline rebuilt with all Phase A/B/C fixes and verified.

**Completed:**
1. Replaced EDT-based DTND with pysheds hydrological DTND (Phase A)
2. Replaced np.gradient slope/aspect with pgrid Horn 1981 (Phase A)
3. Set processing resolution to 1m, no subsampling (Phase B)
4. Lc computed dynamically via FFT with min_wavelength=20m cutoff (Phase C)
5. Extracted shared modules: `hillslope_params.py`, `spatial_scale.py`, `dem_processing.py`
6. Created tiered SLURM wrappers (tier 1/2/3)
7. Comprehensive equation-by-equation audit — all 7 key equations verified correct
8. Tier 3 production run successful (21.8 min, 90 tiles, NetCDF structure verified)

**Deliverable:** Pipeline that produces scientifically defensible hillslope parameters. See `phases/D-rebuild-pipeline.md`.

### Phase E: Complete the parameter set — In Progress

**Status: In progress.** Hillslope structure decision made. Stream and remaining parameters pending.

**Completed:**
- [x] Hillslope structure: 1 aspect x 4 equal-area HAND bins (interim, 2026-03-24). Log-spaced bins deferred until water masking addresses lake pixel contamination. See `docs/hillslope-binning-rationale.md`.
- [x] NEON slope/aspect comparison (2026-03-23): slope r=0.91, aspect circ_r=0.84. Decision pending.

**Remaining:**
- [ ] Research stream depth/width — OSBS-specific empirical relationships (current: interim power-law scaling)
- [ ] PI consultation: DEM conditioning approach, final study boundary, NEON slope/aspect adoption

**Deliverable:** Complete set of physically motivated parameters.

### Phase F: Validate and deploy

**Tasks:**
1. Compare custom hillslope file to Swenson reference (`hillslopes_osbs_c240416.nc`)
2. Physical plausibility checks (elevation, aspect distribution, stream network vs known hydrology)
3. Create CTSM test branch from osbs2 at year 861
4. Run short simulation (1-5 years) with custom hillslope file
5. Compare outputs to baseline (water table, soil moisture, carbon fluxes)

**Deliverable:** Validated hillslope file ready for production runs.

### Dependency Diagram

```
Phase A (fix pysheds UTM) ─────── COMPLETE ─┐
                                             ├──> Phase D (rebuild pipeline) ── COMPLETE
Phase B (resolve resolution) ── COMPLETE ───┤
                                             │
Phase C (establish Lc) ──────── COMPLETE ───┘

Phase E (complete params) ── IN PROGRESS ──┐
                                            ├──> Phase F (validate & deploy)
Phase D (pipeline ready) ── COMPLETE ──────┘
```

Phases A-D complete. Phase E in progress. Phase F blocked by E.

---

## Open Questions for PI

These require scientific judgment, not engineering work:

1. **DEM conditioning:** Should we fill all depressions, or is there an approach that preserves real closed basins (sinkholes, wetland depressions)? Standard D8 requires depression-free DEM, but filling at 1m erases features that matter for OSBS hydrology.

2. **Hillslope structure — RESOLVED (1x8 log-spaced, 2026-03-19).** Switched from 4 aspects x 4 equal-area bins (16 columns) to 1 aspect x 8 log-spaced HAND bins (8 columns). Log spacing concentrates bins near the stream where TAI dynamics dominate. See `docs/hillslope-binning-rationale.md` for full justification. CTSM reads `nhillslope` and `nmaxhillcol` dynamically — no Fortran changes needed. MERIT regression test intentionally retains 4x4 structure for validation against Swenson's published data.

3. **Study boundary:** Interior tiles (150 tiles, ~150 km^2) are the current default. Any areas to specifically include or exclude?

4. **Stream channel parameters:** Should we compute these from the DEM/stream network, or use values from MERIT Hydro or regional empirical relationships?

5. **NEON slope/aspect products — RESOLVED (use NEON, 2026-03-23).** PI approved using NEON DP3.30025.001 products directly. Comparison across all 90 production tiles (commit 418880c): slope Pearson r=0.91, aspect circular r=0.84, slope bias +0.008 m/m (pgrid steeper — NEON's 3x3 pre-filter reduces TIN interpolation noise on flat terrain). Results in `output/osbs/slope_aspect_comparison/`. Implemented in Phase E (commit 73c09fe).

6. **Lc at 1m resolution — RESOLVED (Lc = 356m production, 2026-02-23).** The restricted-wavelength FFT finds a drainage-scale peak when micro-topographic wavelengths (< 20m) are excluded. Physical validation confirms: P95 DTND/Lc = 1.17 (PASS), mean catchment/Lc² = 0.876 (PASS). Lc is computed dynamically per run and varies with domain size: tier 1 (1 tile) = 541m, tier 2 (25 tiles) = 479m, tier 3 (90 tiles, production) = 356m, A_thresh = 63,362. See `phases/C-characteristic-length.md` for full analysis.

   **Important: Lc resolution and flow routing resolution are independent.** Lc is used for exactly one thing — setting A_thresh = 0.5 * Lc², which controls stream network density. Once A_thresh is determined, it can be applied at any routing resolution. The fact that Lc requires filtering out sub-20m wavelengths does **not** imply the DEM should be subsampled to ~100m for flow routing. The restricted-wavelength approach computes the full 1m FFT and filters during peak fitting — no information destruction, no aliasing. Flow routing, HAND, and DTND still benefit from 1m resolution: wetland depressions (10-50m across), stream channels (1-5m wide), and sub-meter elevation differences that drive TAI dynamics all require fine resolution to resolve.

   **What the 20m transition means physically:** The Laplacian spectrum separates cleanly at ~20m wavelength — below this is micro-topographic noise (tree-throw mounds, animal burrows, shallow rills), above is organized drainage structure. This is the scale boundary Swenson's method implicitly assumes when using 90m MERIT data (where everything below ~180m is already averaged away). At 1m, the boundary must be made explicit.

   **Uncertainty in the Lc range:** Gaussian fit at cutoff=100m gives 285m; lognormal at cutoff=20m gives 356m. This 25% range in Lc translates to ~56% range in A_thresh (40,000 vs 63,000 m²) since A_thresh scales with Lc². Whether this matters for final hillslope parameters is an empirical question — testable by running the rebuilt pipeline at both endpoints.

---

## Where We Are on the Original Arc

From `progress-tracking.md`, mapped to current phase structure:

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Setup | Complete | pysheds fork, environment, Swenson code review |
| 2. Port Swenson functions | Complete | pgrid.py copied, tests passing |
| 3. Validate against published | Complete | 5/6 params >0.95 correlation |
| 4. Apply to OSBS (Phases A-D) | **Complete** | pysheds UTM-aware, 1m resolution, Lc=356m, pipeline rebuilt and audited |
| 5. Generate final dataset (Phase E) | **In progress** | 1x4 equal-area bins (interim), NEON slope/aspect adopted, stream params interim |
| 6. CTSM validation (Phase F) | Pending | Depends on Phase E completion |

The pipeline produces scientifically defensible output. Remaining work is parameter completion (Phase E) and CTSM validation (Phase F).

---

## Reference Documents

| Document | Location | Content |
|----------|----------|---------|
| MERIT validation audit | `audit/claude-audit.md` | Personal audit notes (reduced; key context absorbed into this document) |
| OSBS pipeline audit | `audit/osbs_pipeline_audit.md` | Issue catalog with line numbers and fix options |
| Flow routing resolution | `audit/flow-routing-resolution.md` | Testing plan for subsampling problem |
| Progress tracking | `progress-tracking.md` | Full implementation history |
| Swenson paper summary | `docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` | Methodology reference |
| Phase C Lc analysis | `phases/C-characteristic-length.md` | Lc results, sensitivity, interpretation |
| Phase C job log | `logs/phase_c_lc_24705742.log` | Full output from Lc analysis run |
| Phase C plots | `output/osbs/phase_c/` | Baseline spectrum and sensitivity sweep plots |
| Lc physical validation | `output/osbs/smoke_tests/lc_physical_validation/` | Check 1 & 2 results, plots, JSON |
