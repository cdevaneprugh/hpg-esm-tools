# State of the Union: Swenson Hillslope Implementation for OSBS

Date: 2026-02-10

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
- **North/South aspect swap** (stage 8): Fixed a Y-axis sign inversion in the gradient calculation that systematically swapped N and S aspects. Area fraction correlation improved from 0.64 to 0.82.
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
- Interior mosaic created (`OSBS_interior.tif`)
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

### 3. No trustworthy characteristic length scale (Lc)

**Impact:** Everything downstream depends on Lc — accumulation threshold, stream network density, HAND, DTND, all 6 parameters.

**File:** `scripts/osbs/run_pipeline.py` lines 1275-1287

The FFT ran on the 4x-subsampled grid. For the full mosaic, it failed to find a real peak and fell back to the hardcoded minimum of 100m. For the interior mosaic, it found 166m at 4m resolution — but subsampling to 4m means the FFT can only detect features at wavelengths >= 8m, and we don't know whether the actual peak is at a shorter wavelength.

The full-resolution FFT on the OSBS interior has **never been run**. numpy handles arrays of this size trivially — this is an easy win.

**Lc comparison across datasets:**

| | MERIT | OSBS (full, 4x sub) | OSBS (interior, 4x sub) |
|---|---|---|---|
| Lc source | FFT peak | Forced minimum | FFT peak (4m res) |
| Lc value | 763m | 100m | 166m |
| Threshold | 34 cells | 312 cells | 864 cells |
| Stream coverage | 2.17% | 2.32% | 1.44% |

Stage 9 threshold sensitivity testing on MERIT found low sensitivity in the 20-50 cell range (correlation 0.80-0.83) but rapid degradation above 100 cells. Conclusion: threshold isn't the cause of remaining discrepancy — Lc is the controlling variable. An incorrect Lc propagates through to all 6 parameters.

**Fix path:** Decouple FFT from flow routing. Run FFT at full 1m resolution on a representative interior region. Use the resulting Lc (scaled to subsampled pixel size) for flow routing threshold.

Also test FFT parameter sensitivity — see problem #8 for the full test matrix.

### 4. Slope/aspect calculation not validated

**Impact:** Potentially corrupts slope and aspect parameters for all 16 columns; also corrupts aspect-based binning (which affects area fractions).

**File:** `scripts/osbs/run_pipeline.py` lines 1514-1528

Stage 8 of the MERIT validation discovered that `np.gradient`-based aspect had a Y-axis sign inversion that swapped North/South. The fix (use pgrid's `slope_aspect()` with Horn 1981 stencil) was applied to `stage3_hillslope_params.py` but **not** to the OSBS pipeline.

The OSBS pipeline uses `np.gradient` with `arctan2(-dzdx, -dzdy)`, which *may* compensate for the sign issue, but this was **never validated** against the stage 8 findings.

The reason pgrid's method wasn't used directly: it assumes geographic coordinates (haversine spacing). Same CRS problem as DTND.

**Fix path:** Either:
- Adapt pgrid's `slope_aspect()` / `_gradient_horn_1981()` for UTM (replace haversine spacing with uniform pixel_size)
- Validate the existing np.gradient approach against a known-correct result

This is the easiest of the 4 critical issues to resolve.

**NEON validation data:** NEON provides precalculated slope/aspect rasters (DP3.30025.001) which could serve as a validation baseline. Key tradeoffs: NEON likely has better flat-area handling (where near-zero gradients make aspect noisy), but adds an external dependency and grid alignment must be verified. However, bin-averaged means over thousands of pixels may wash out per-pixel differences anyway. Worth a comparison test — see PI question #5.

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

### 8. FFT parameters untested for OSBS

**Impact:** Lc determination is sensitive to these parameters, and all were copied from Swenson's 90m defaults without validation at 1m.

**File:** `scripts/osbs/run_pipeline.py` lines 80-85, 489-506

**Why parameters matter at 1m resolution:**

- **MAX_HILLSLOPE_LENGTH (10km):** Creates a search range 50x wider than needed for OSBS. Interior tile results show ridge-to-channel distances ~100m — the algorithm searches for peaks across a huge spectral range when the actual feature is at the low end. Reducing to 1-2km would focus the search. Test 500m, 1km, 2km, 10km.
- **NLAMBDA (30):** At 90m, the resolvable wavelength range spans ~1.5 orders of magnitude. At 1m (or 4m subsampled), it spans 3+ orders of magnitude. Same 30 bins over a wider log-wavelength range = coarser peak resolution. Increasing to 50-75 might help, but more bins also means a noisier amplitude curve.
- **blend_edges (50px):** At 1m, 50 pixels = 50m of geographic edge smoothing. Swenson's 4 pixels at 90m = 360m. Current setting provides 7x less smoothing than Swenson's intent. Should test 50, 100, 200.
- **zero_edges (5px):** At 1m, 5 pixels = 5m buffer. Swenson's 5 pixels at 90m = 450m. The Horn 1981 gradient stencil needs 3 pixels minimum, and OSBS has irregular nodata boundaries from tile coverage. Needs at least 20-50 pixels at 1m.
- **detrend_elevation:** Fine as-is. OSBS has a general NW-to-SE slope that would bias FFT toward large wavelengths if not removed.
- **land_threshold / min_land_elevation:** Irrelevant for OSBS — everything is well above sea level.

**Test matrix** (one parameter at a time on a representative interior region at full 1m resolution):

| Test | Variable | Values |
|------|----------|--------|
| A | blend_edges window | 4, 25, 50, 100, 200 |
| B | zero_edges margin | 5, 20, 50, 100 |
| C | NLAMBDA | 20, 30, 50, 75 |
| D | MAX_HILLSLOPE_LENGTH | 500m, 1km, 2km, 10km |
| E | detrend_elevation | True, False |
| F | Region size | 2000x2000, 5000x5000, 8000x8000 |

**Interpretation:** If Lc is stable across all parameters, they don't matter and we move on. If it's sensitive, we know which need calibration. Each FFT run takes seconds.

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

### Phase C: Establish trustworthy Lc (depends on B for resolution decision, but FFT itself is independent)

**Tasks:**
1. Run full-resolution FFT on interior mosaic (decoupled from flow routing)
2. Run FFT parameter sensitivity tests using the test matrix from problem #8 (blend_edges, zero_edges, NLAMBDA, MAX_HILLSLOPE_LENGTH, detrend_elevation, region size — one variable at a time). If Lc is stable across all parameters, they don't matter and we move on. If it's sensitive, we know which need calibration.
3. Determine whether Lc is stable or sensitive to parameters
4. Set final Lc value with justification

**Deliverable:** Lc value with error bounds or sensitivity analysis.

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
