# State of the Union: Swenson Hillslope Implementation for OSBS

Date: 2026-02-17

## Executive Summary

We are implementing Swenson & Lawrence (2025) representative hillslope methods to generate custom hillslope parameters for OSBS using 1m NEON LIDAR data. The goal is to replace the current global 90m MERIT-derived hillslope file (`hillslopes_osbs_c240416.nc`) with site-specific parameters that capture the fine-scale drainage structure of this low-relief wetlandscape.

**The basic methodology is proven.** MERIT validation (stages 1-9) achieved >0.95 correlation with Swenson's published data on 5 of 6 hillslope parameters. The pipeline runs, produces CTSM-compatible NetCDF output, and the structure is correct.

**The current output is not scientifically trustworthy.** The audit identified 4 issues that directly corrupt the 6 hillslope parameters the pipeline produces. Until these are resolved, the output should not be used for CTSM simulations.

---

## What Is Done Well

### Methodology is validated against published data

The MERIT validation (stages 1-9) demonstrated that our pysheds fork and pipeline correctly reproduce Swenson's results:

| Parameter | Correlation with published |
|-----------|---------------------------|
| Height (HAND) | 0.9999 |
| Distance (DTND) | 0.9982 |
| Slope | 0.9966 |
| Aspect | 0.9999 (circular) |
| Width | 0.9597 |
| Area fraction | 0.8200 |

This gives confidence that the *approach* is correct — we understand Swenson's methodology and can implement it.

### Bugs found and fixed during validation

- **Width calculation** (stage 6): Fixed from 0.09 to 0.96 correlation. Root cause was using raw pixel areas instead of fitted trapezoidal areas.
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
- Full mosaic stitched (`OSBS_full.tif`, 17000x19000 pixels)
- Interior tile selection defined (150 tiles, excludes edge/urban areas)
- Interior mosaic created (`OSBS_interior.tif`) — note: 37.5% nodata from tile gaps
- Contiguous sub-region identified: R4-R12, C5-C14 (90 tiles, 9x10 km, 0 nodata pixels out of 90M) — used as clean baseline for Phase C spectral analysis
- Tile nodata coverage documented (`data/neon/tile_coverage.md`)
- Tile reference system documented (R#C# format, KML for Google Earth)
- osbs2 baseline case identified for future comparison (860+ year spinup)

---

## Problems, In Order of Importance

### 1. DTND uses the wrong algorithm

**Impact:** Corrupts the distance parameter for all 16 hillslope columns.

**File:** `scripts/osbs/run_pipeline.py` lines 1480-1483

The pipeline computes DTND using `scipy.ndimage.distance_transform_edt` — Euclidean distance to the *geographically nearest* stream pixel. Swenson's method computes distance to the *hydrologically nearest* stream pixel (the one each pixel actually drains to via D8 routing). These differ whenever a pixel is geographically close to a stream on the opposite side of a divide.

The workaround exists because pysheds' `compute_hand()` computes the correct hydrologically-linked distance internally, but uses haversine math that assumes geographic coordinates. Our UTM data produces garbage distances through haversine.

Neither DTND is currently correct:

| Approach | Concept | Math |
|----------|---------|------|
| pysheds DTND | Correct (flow-path-linked) | Wrong (haversine on UTM) |
| Pipeline EDT | Wrong (nearest regardless of drainage) | Correct (Euclidean on UTM) |

**Fix path:** Modify `compute_hand()` in `$PYSHEDS_FORK/pysheds/pgrid.py` (lines 1928-1942) to detect UTM CRS and use Euclidean distance instead of haversine. Alternatively, expose the `hndx` array (which maps each pixel to its drainage stream pixel — already computed at line 1919) and compute UTM distance externally.

**Downstream effect:** DTND feeds directly into the hillslope distance parameter and the trapezoidal width fit (via A_sum(d) curve). Wrong DTND means wrong widths and wrong distances for all columns.

### 2. Flow routing resolution (4x subsampling)

**Impact:** Discards 93.75% of the 1m LIDAR data before the core hydrology computation.

**File:** `scripts/osbs/run_pipeline.py` line 1354

The entire justification for using 1m LIDAR is to capture fine-scale drainage in a low-relief wetlandscape. Subsampling to 4m undermines that purpose — small channels, wetland margins, and subtle elevation differences that drive TAI dynamics become invisible.

The bottleneck is pysheds' `resolve_flats()`, which has poor scaling on large flat regions. One test at full resolution OOM'd at 64GB. Full resolution at 256GB was **never tested**. Neither was 2x subsampling.

**Fix path:** Documented in `audit/flow-routing-resolution.md`. In order of simplicity:
1. Test full resolution at 256GB (maybe it just works)
2. Test 2x subsampling at 128GB (preserves 4x more data than current)
3. Use WhiteboxTools for DEM conditioning, feed result to pysheds
4. Tile-based processing with overlap
5. Optimize resolve_flats in pysheds

**Downstream effect:** Resolution affects the stream network, HAND, DTND, and through them all 6 hillslope parameters. Also affects Lc if the FFT is coupled to the same subsampled grid (it shouldn't be — see #3).

### 3. Characteristic length scale (Lc): ~300m accepted as working value, physical validation pending Phase A

**Impact:** Everything downstream depends on Lc — accumulation threshold, stream network density, HAND, DTND, all 6 parameters.

**File:** `scripts/osbs/run_pipeline.py` lines 1275-1287

**Phase C established that the Laplacian spectrum has two real features at 1m resolution:** a micro-topographic peak at ~8m (k² amplification artifact) and a drainage-scale peak at ~285-356m (visible when short wavelengths are excluded). See `phases/C-characteristic-length.md` for full results and `output/osbs/phase_c/` for plots.

**Lc comparison across all datasets:**

| Dataset | Lc source | Lc value | A_thresh |
|---------|-----------|----------|----------|
| MERIT (90m) | FFT peak | 763m | 275,400 m² |
| OSBS (full, 4x sub) | Forced minimum | 100m | 5,000 m² |
| OSBS (interior, 4x sub) | FFT peak (4m res) | 166m | 13,778 m² |
| OSBS (interior, 1m, full range) | FFT peak (artifact) | 8.1m | 33 m² |
| **OSBS (interior, 1m, cutoff>=20m)** | **FFT peak** | **356m** | **63,368 m²** |
| **OSBS (interior, 1m, cutoff>=100m)** | **FFT peak** | **285m** | **40,612 m²** |

**Why the full-range method fails at 1m:** The raw elevation spectrum is red noise — no single scale stands out. The Laplacian's k² weighting amplifies micro-topography (~8m) by ~1400x relative to 300m, creating an artificial peak. At 90m, micro-topography is averaged away and the method works as designed.

**Restricted wavelength sweep resolves this.** Excluding wavelengths below 20m causes a sharp transition: the peak jumps from 11.7m to 356m (lognormal, psharp=3.95). This is stable through cutoff=100m (Lc=285m, gaussian, psharp=4.21). Above 180m cutoff, too few bins remain for peak fitting.

**All Phase C follow-up tests complete:**
1. ~~Raw DEM spectrum~~ — Confirms 8m is k² artifact. Raw spectrum has no peak.
2. ~~Single-tile FFT~~ — Tile R6C10 reproduces same spectral structure. Mosaic stitching is not creating artifacts.
3. ~~Restricted wavelength range~~ — **200-500m hump IS a real peak when micro-topography excluded. Lc = 285-356m.**

**PI accepts ~300m as working Lc (2026-02-11).** The spectral analysis is complete. Lc ~300m (range 285-356m) is the working value, with A_thresh ~45,000-63,000 m² (same order of magnitude as MERIT).

**Physical validation (2026-02-17):** Run on 5x5 tile block (R6-R10, C7-C11, 25M pixels at 1m). Check 2 (mean catchment area / Lc^2 = 0.876) **passes** — close to Swenson's calibration of 0.94. Check 1 (max DTND / Lc = 3.1) **fails** — but driven by a single 931m outlier pixel on a large ridge; P99/Lc = 1.6 and P95/Lc = 1.2 are within range. The max(DTND) is the same pixel across all three Lc values tested, suggesting one anomalously long catchment rather than a systematic Lc mismatch. Revisiting after refactoring — may warrant using P99 instead of max, or running on a larger domain. Results: `output/osbs/smoke_tests/lc_physical_validation/`.

### 4. Slope/aspect: N/S aspect swap in OSBS pipeline — FIXED

**Impact:** Corrupted aspect for all pixels, which corrupted aspect-based binning and area fractions for all 16 hillslope columns.

**File:** `scripts/osbs/run_pipeline.py` lines 1514-1528

**Status: FIXED.** The sign bug (`-dzdy` → `dzdy`) has been applied to `run_pipeline.py:1527`. Phase D will replace `np.gradient` entirely with pgrid's `slope_aspect()` once Phase A makes it UTM-aware.

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

**File:** `scripts/osbs/run_pipeline.py` lines 1436-1440

**Status:** This is a science question for the PI, not a bug. Standard D8 flow routing requires a depression-free DEM. Alternative approaches (like RICHDEM's priority-flood with depression retention) exist but add complexity.

### 6. Stream channel parameters are hardcoded guesses

**Impact:** Directly affects CTSM lateral subsurface flow and stream-groundwater exchange.

**File:** `scripts/osbs/run_pipeline.py` lines 1013-1022

| Parameter | Pipeline | Swenson reference |
|-----------|----------|-------------------|
| Stream depth | 0.3 m | 0.269 m |
| Stream width | 5.0 m | 4.414 m |
| Stream slope | heuristic | 0.00233 |

The guesses are in the right ballpark, but there is no methodology. Stream slope could be computed from actual DEM elevation drops along the stream network. Depth and width could come from regional empirical relationships or MERIT Hydro.

### 7. Bedrock depth is a placeholder

**File:** `scripts/osbs/run_pipeline.py` line 1025

Pipeline uses `1e6` (effectively infinite). Swenson reference has all zeros. Neither is physically meaningful. Need to determine what CTSM does with this parameter.

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

### 9. ~400 lines of duplicated code

**Files:** `run_pipeline.py` lines 275-568 (spatial scale), lines 666-742 (hillslope computation)

Both the spatial scale analysis functions and the hillslope computation functions are duplicated between the merit_validation scripts and the OSBS pipeline. Any fix must be applied twice.

**Target architecture:**

| Layer | Responsibility | Location |
|-------|---------------|----------|
| pysheds (fork) | Flow routing, HAND, DTND, slope/aspect | `$PYSHEDS_FORK/pysheds/pgrid.py` |
| Hillslope analysis module | Binning, trapezoidal fit, width, 6-param computation | Missing — should be `scripts/hillslope_params.py` |
| Pipeline scripts | Orchestration, I/O, plotting | `scripts/osbs/run_pipeline.py`, `scripts/merit_validation/stage3_*.py` |

The middle layer is what's missing. The quadratic solver, trapezoidal fitting, HAND binning logic, and width computation should be extracted into a shared module that both the MERIT validation and OSBS pipeline import.

**Note:** `dirmap` (D8 flow direction mapping) is hardcoded in pipeline scripts — arguably should be a fork constant in pysheds.

### 10. pysheds fork: deprecation warnings and test suite

**Impact:** 33 deprecation warnings in `pgrid.py` from Swenson's older code create noise that obscures real issues during development.

**File:** `$PYSHEDS_FORK/pysheds/pgrid.py`

**Test suite status:**
- `test_hillslope.py`: Passes (with 33 deprecation warnings). Hangs on first run due to scipy import quirk.
- `test_grid.py`: Pre-existing API mismatch failures from when Swenson methods were added. Functions tested here are prerequisites to test_hillslope. Notes/skips were added to provide context.

**Fix path:** Address during Phase A while already modifying the fork. Natural to clean up warnings alongside the UTM CRS changes.

---

## Work Flow: What Needs to Happen and In What Order

**Phase tracking files:** `phases/` — one file per phase with tasks, results, and decisions.

### Phase A: Fix pysheds for UTM (blocks everything)

Both the DTND problem (#1) and the slope/aspect problem (#4) stem from the same root cause: pysheds assumes geographic coordinates. Fixing pysheds once resolves both.

**Tasks:**
0. Fix 33 deprecation warnings in `pgrid.py` (clean working state before making changes)
1. Modify `pgrid.py:compute_hand()` to detect UTM CRS and use Euclidean distance for DTND
2. Modify `pgrid.py:slope_aspect()` / `_gradient_horn_1981()` to use uniform pixel spacing for UTM
3. Test both changes against the MERIT validation (should reproduce existing results since MERIT is geographic)
4. Test on the OSBS 4x4km smoke test region (UTM, known results to compare)

**Deliverable:** pysheds fork that correctly handles both CRS types.

### Phase B: Resolve flow routing resolution (blocks trustworthy output)

Run in parallel with Phase A (independent work).

**Tasks:**
1. Test full-res (256GB) on interior mosaic — characterize whether resolve_flats completes
2. Test 2x subsampling (128GB) as middle ground
3. If neither works: evaluate WhiteboxTools pre-conditioning
4. Comparison: run full pipeline at 1m, 2m, 4m on the same region, compare hillslope parameters

**Deliverable:** Determined processing resolution with scientific justification.

### Phase C: Establish trustworthy Lc — ~300m accepted as working value, physical validation mixed

**Status:** Spectral analysis complete. Physical validation run (2026-02-17): Check 2 (mean catchment / Lc^2) passes cleanly at 0.88; Check 1 (max DTND / Lc) fails at 3.1 due to a single outlier ridge pixel (P99/Lc = 1.6 passes). Revisiting after refactoring. See `phases/C-characteristic-length.md`.

**Completed:**
1. Full-resolution (1m) FFT on interior mosaic — Laplacian peak at 8.1m (artifact)
2. Sensitivity sweep (tests A-E) — Lc insensitive to FFT parameters
3. Code audit — implementation verified against Swenson's original
4. Raw DEM spectrum test — confirms 8.1m is k² artifact, raw spectrum is red noise
5. Single-tile FFT (R6C10) — confirms spectral features are intrinsic, not mosaic artifacts
6. Restricted wavelength sweep — **200-500m hump IS a real peak at cutoff >= 20m: Lc = 285-356m**
7. Tile coverage documented (`data/neon/tile_coverage.md`)
8. PI decision: ~300m accepted as working value
9. Physical validation run on 5x5 tile block (R6-R10, C7-C11, 25M pixels at 1m) — Check 2 PASS (0.876), Check 1 FAIL (3.105, outlier-driven). Results: `output/osbs/smoke_tests/lc_physical_validation/`

**Remaining:**
10. Revisit Check 1 interpretation after refactoring — max vs P99, domain size effects, PI discussion

**Deliverable:** Lc value with scientific justification. Working value established; physical validation partially confirms.

### Phase D: Rebuild pipeline with fixes (depends on A, B, C)

**Tasks:**
1. Replace EDT-based DTND with pysheds hydrological DTND (from Phase A)
2. Replace np.gradient slope/aspect with pgrid Horn 1981 (from Phase A)
3. Set processing resolution (from Phase B)
4. Set Lc and accumulation threshold (from Phase C)
5. Extract shared hillslope analysis module (resolve code duplication)
6. Rerun pipeline on interior mosaic

**Deliverable:** Pipeline that produces scientifically defensible hillslope parameters.

### Phase E: Complete the parameter set (can overlap with D)

**Tasks:**
1. Compute stream slope from actual stream network elevation profile
2. Research stream depth/width — regional empirical relationships or MERIT Hydro
3. Research bedrock depth — check CTSM behavior, identify data source
4. PI consultation: DEM conditioning approach, single vs 4-aspect hillslopes, final study boundary

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
Phase A (fix pysheds UTM) ─────────────────┐
                                            ├──> Phase D (rebuild pipeline) ──> Phase F (validate)
Phase B (resolve resolution) ──────────────┤                                        ↑
                                            │                                        │
Phase C (establish Lc) ────────────────────┘                                        │
                                                                                     │
Phase E (complete params) ──────────────────────────────────────────────────────────┘
```

Phases A, B, and C can proceed in parallel. Phase D requires all three. Phase E can start independently and merge at Phase F.

---

## Open Questions for PI

These require scientific judgment, not engineering work:

1. **DEM conditioning:** Should we fill all depressions, or is there an approach that preserves real closed basins (sinkholes, wetland depressions)? Standard D8 requires depression-free DEM, but filling at 1m erases features that matter for OSBS hydrology.

2. **Hillslope structure:** 4 aspects x 4 elevation bins (Swenson default) vs 1 aspect x 8 elevation bins? OSBS is nearly flat — aspects are nearly uniformly distributed and slopes are 0.01-0.06 m/m, so aspect-dependent insolation has negligible physical impact. More elevation bins may better capture TAI dynamics. But 4x4 enables direct comparison to the osbs2 baseline.

   **Technical detail:** CTSM is fully flexible — it reads `nhillslope` and `nmaxhillcol` from the input file (`surfrdMod.F90` lines 1082-1096), no hardcoded 4x4 requirement. A 1x8 configuration would skip aspect binning, put all non-stream pixels in one pool, bin by HAND into 8 bands, and fit one trapezoidal model with 4x more data points. The aspect risk: circular mean of a uniform distribution gives an arbitrary number, but insolation correction scales with `slope * cos(aspect - solar_azimuth)` — negligible at OSBS slopes (0.01-0.06 m/m). Practical option: generate both 4x4 and 1x8 configurations, let comparison speak for itself.

3. **Study boundary:** Interior tiles (150 tiles, ~150 km^2) are the current default. Any areas to specifically include or exclude?

4. **Stream channel parameters:** Should we compute these from the DEM/stream network, or use values from MERIT Hydro or regional empirical relationships?

5. **NEON slope/aspect products:** NEON DP3.30025.001 provides precalculated slope/aspect rasters. Worth using as a validation baseline for our gradient calculation, especially for flat terrain where the noise-to-signal ratio is high. Requires grid alignment verification. Decision: use as a validation check only, or replace our calculation entirely?

6. **Lc at 1m resolution — ~300m accepted as working value (2026-02-11).** The restricted-wavelength FFT finds a drainage-scale peak at Lc = 285-356m when micro-topographic wavelengths (< 20m) are excluded. This gives A_thresh ~45,000-63,000 m², the same order of magnitude as MERIT. PI accepts ~300m as working value; final judgement reserved until physical validation (Lc vs max DTND, Lc² vs mean catchment area) after Phase A. See `phases/C-characteristic-length.md` for full analysis.

   **Important: Lc resolution and flow routing resolution are independent.** Lc is used for exactly one thing — setting A_thresh = 0.5 * Lc², which controls stream network density. Once A_thresh is determined, it can be applied at any routing resolution. The fact that Lc requires filtering out sub-20m wavelengths does **not** imply the DEM should be subsampled to ~100m for flow routing. The restricted-wavelength approach computes the full 1m FFT and filters during peak fitting — no information destruction, no aliasing. Flow routing, HAND, and DTND still benefit from 1m resolution: wetland depressions (10-50m across), stream channels (1-5m wide), and sub-meter elevation differences that drive TAI dynamics all require fine resolution to resolve.

   **What the 20m transition means physically:** The Laplacian spectrum separates cleanly at ~20m wavelength — below this is micro-topographic noise (tree-throw mounds, animal burrows, shallow rills), above is organized drainage structure. This is the scale boundary Swenson's method implicitly assumes when using 90m MERIT data (where everything below ~180m is already averaged away). At 1m, the boundary must be made explicit.

   **The restricted-wavelength approach is analytically defensible.** Swenson's method assumes the Laplacian peak corresponds to drainage-scale periodicity. At 90m that's true by construction. At 1m, explicitly filtering sub-drainage wavelengths before peak fitting achieves the same thing. This is more principled than physical subsampling (which introduces aliasing) or smoothing (which blurs features unevenly).

   **Uncertainty in the Lc range:** Gaussian fit at cutoff=100m gives 285m; lognormal at cutoff=20m gives 356m. This 25% range in Lc translates to ~56% range in A_thresh (40,000 vs 63,000 m²) since A_thresh scales with Lc². Whether this matters for final hillslope parameters is an empirical question — testable by running the rebuilt pipeline at both endpoints.

   **Predictions from Lc ~300m (tested 2026-02-17 on R6-R10, C7-C11):**
   - Max ridge-to-channel distance (DTND) ~300m — **Actual: max = 931m (3.1x), P99 = 477m (1.6x), P95 = 350m (1.2x).** Max fails due to single outlier ridge pixel; bulk distribution is consistent.
   - Mean catchment area ~90,000 m² — **Actual: 78,882 m² (0.88x Lc²). PASS.** Close to Swenson's calibration of 0.94.
   - A_thresh ~45,000 m² — stream pixels where accumulated drainage area exceeds this

   **Alternative to FFT:** Set Lc empirically from aerial imagery or field knowledge of drainage spacing. The spectral result provides a starting point, but isn't the only option.

---

## Where We Are on the Original Arc

From `progress-tracking.md`:

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Setup | Complete | pysheds fork, environment, Swenson code review |
| 2. Port Swenson functions | Complete | pgrid.py copied, tests passing |
| 3. Validate against published | Complete | 5/6 params >0.95 correlation |
| 4. Apply to OSBS | **In progress — blocked** | Pipeline runs but output has known issues |
| 5. Generate final dataset | Pending | Depends on Phase 4 fixes |
| 6. CTSM validation | Pending | Depends on Phase 5 |

The audit revealed that Phase 4 was declared "substantially complete" prematurely. The pipeline runs and produces output, but 4 scientific issues mean the output values are not yet trustworthy. The path forward is clear: fix pysheds for UTM, resolve the resolution question, get a real Lc, and rebuild.

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
