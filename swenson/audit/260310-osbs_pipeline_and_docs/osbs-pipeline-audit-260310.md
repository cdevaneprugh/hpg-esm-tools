# OSBS Pipeline Full Audit — 2026-03-10

Full audit of the OSBS hillslope pipeline (`scripts/osbs/run_pipeline.py`, 1539 lines), cross-referenced against the Swenson & Lawrence (2025) paper (all 19 pages), the MERIT validation pipeline (source of truth), CTSM source code, and Phase D tier test results. Subsumes and extends the tier test audit of 2026-02-26.

## Executive Summary

The pipeline is a well-engineered implementation of Swenson & Lawrence (2025) that correctly follows the paper's methodology with appropriate adaptations for UTM CRS and 1m NEON LIDAR data. The shared modules (`hillslope_params.py`, `spatial_scale.py`, `dem_processing.py`) cleanly separate concerns and enable code reuse between the OSBS pipeline and the MERIT validation regression test.

**The pipeline is "in spirit" with Swenson's methodology.** All 7 key equations (Eq 6-12, 14, 16-17) are implemented correctly. The processing order matches Figure 9's flowchart. Deviations from Swenson are intentional, documented, and justified by CRS/resolution differences.

**15 issues identified.** 2 significant (could affect production results), 6 moderate (correctness edge cases or science questions), 7 minor (code hygiene). The tier test audit's 4 concerns are incorporated as issues T1-T4.

---

## Tier Test Results (Phase D)

| Tier | Domain | Tiles | Pixels | Runtime | Lc | HAND range |
|------|--------|-------|--------|---------|-----|-----------|
| 1 | R6C10 (single tile) | 1 | 1M | 9s | 332m | 0 - 12.7m |
| 2 | R6C7-R10C11 (5x5 block) | 25 | 25M | 2.8 min | 285m | 0 - 21.1m |
| 3 | R4C5-R12C14 (max contiguous) | 90 | 90M | 21.8 min | 356m | 0 - 25.1m |

Tier 3 is the maximum contiguous rectangle of tiles with zero nodata pixels.

---

## 1. Pipeline Order vs Paper Methodology (Figure 9)

| Paper Step | Paper Reference | Pipeline Location | Match? |
|---|---|---|---|
| 1. Identify spatial scale (FFT on Laplacian) | Sec 2.3, Eq 1-5 | Step 2 (lines 812-841) | YES |
| 2. Set A_thresh = 0.5 * Lc^2 | Sec 2.4, Eq 6 | Line 834 | YES |
| 3. DEM conditioning | Sec 2.2 (MERIT-HYDRO ref) | Step 3d (lines 947-988) | YES |
| 4. Flow direction + accumulation | Sec 2.4 | Step 3e (lines 991-1025) | YES |
| 5. Channel mask + catchment delineation | Sec 2.4 | Step 3f (lines 1028-1074) | YES |
| 6. Compute HAND/DTND | Sec 2.5.2 | Step 4 (lines 1082-1096) | YES |
| 7. Hillslope classification (L/R bank, headwater) | Sec 2.4, 2.6.1 | Step 4 (lines 1108-1118) | YES |
| 8. Catchment-level aspect averaging | Sec 2.6.1 | Line 1116 | YES |
| 9. Aspect binning (4 cardinal) | Sec 2.6.1, Table 1 | Lines 1248-1265 | YES |
| 10. HAND binning (4 equal-area, 2m constraint) | Sec 2.6.1 | Lines 1214-1219 | YES |
| 11. Trapezoidal width fit (A_sum vs d) | Sec 2.6.2, Eq 9-16 | Lines 1278-1286 | YES |
| 12. 6 parameter computation per element | Sec 2.5, Eq 17 | Lines 1326-1399 | YES |

**No missing or reordered steps.** The OSBS pipeline adds three pre-processing steps not in Swenson (basin detection, connected-component extraction, nodata edge trimming) that are appropriate for mosaicked LIDAR data with gaps.

---

## 2. Mathematical Accuracy — Equation-by-Equation Verification

### Eq 6: A_thresh = k * Lc^2 (k = 1/2)
- **Pipeline:** `accum_threshold = int(0.5 * lc_px**2)` (line 834)
- **Correct.** Uses pixel-unit Lc, so A_thresh is in cells (pixels). At 1m, cells = m^2.

### Eq 7: Slope = sqrt((dz/dx)^2 + (dz/dy)^2)
- **Pipeline:** `grid.slope_aspect("dem")` (line 954) using pgrid's Horn 1981 stencil
- **Correct.** Phase A made Horn 1981 UTM-aware. Slope computed on ORIGINAL DEM, not conditioned DEM — preventing false gradients from fill artifacts.

### Eq 8: Aspect = -arctan(dz/dx / dz/dy)
- **Pipeline:** `grid.slope_aspect("dem")` (line 954)
- **Correct** after Phase A fix. The N/S swap (STATUS #4) is fully resolved.

### Eq 9: w(d) = w_base + 2*alpha*d
- **Pipeline:** `width = trap_width + 2 * trap_slope * le` (line 1370)
- **Correct.**

### Eq 10-11: w(d) = -dA_sum/dd, A_sum = sum of A(d') for d' > d
- **Pipeline:** A_cumsum built in `fit_trapezoidal_width` (hillslope_params.py:200-203)
  - `A_cumsum[k] = np.sum(area[mask])` where `mask = dtnd >= dtnd_bins[k]`
- **Correct.** Cumulative area is decreasing function of distance. The polynomial derivative gives width.

### Eq 12, 14: A_sum(d) = a2*d^2 + a1*d + a0; alpha = -a2, w_base = -a1
- **Pipeline:** `trap_slope = -coeffs[2]`, `trap_width = -coeffs[1]`, `trap_area = coeffs[0]` (hillslope_params.py:233-235)
- **Correct.**

### Eq 16: A(l) = alpha*l^2 + w_base*l
- **Pipeline:** `quadratic([trap_slope, trap_width, -da_width])` (line 1369)
- **Correct.** Solving trap_slope*d^2 + trap_width*d - accumulated_area = 0.

### Eq 17: Distance formula — A_hill * (sum F_i<n + F_n/2) = alpha*D_n^2 + w_base*D_n
- **Pipeline:** `da_dist = sum(fitted_areas[:h_idx+1]) - fitted_areas[h_idx] / 2` (line 1378)
- Expanding: sum(F_0..F_n) - F_n/2 = sum(F_0..F_{n-1}) + F_n/2
- Since fitted_areas[i] = trap_area * area_frac[i] = A_hill * F_i
- This gives: A_hill * (sum_{i<n} F_i + F_n/2) — **matches Eq 17 exactly.**

### Weighted least squares (Swenson _fit_polynomial)
- **Pipeline:** `hillslope_params.py:222-227` — W = diag(A_cumsum), w^1 weighting
- **Correct.** Matches Swenson's normal equations after the w^2 -> w^1 fix documented in Phase C.

### Circular mean aspect
- **Pipeline:** `circular_mean_aspect` (hillslope_params.py:296-307)
- Standard sin/cos decomposition with 0-360 normalization. **Correct.**

### All mathematics verified. No errors found.

---

## 3. Code Consistency: OSBS vs MERIT Pipelines

Both pipelines import from the same shared modules (`hillslope_params.py`, `spatial_scale.py`). The core computation logic is identical. Intentional differences:

| Feature | MERIT (merit_regression.py) | OSBS (run_pipeline.py) | Justified? |
|---|---|---|---|
| CRS | Geographic (EPSG:4326) | UTM (EPSG:32617) | Yes -- data source |
| FFT call | `elon/elat` parameters | `pixel_size` parameter | Yes -- CRS modes |
| Slope/aspect | pgrid `slope_aspect()` (haversine) | pgrid `slope_aspect()` (Euclidean) | Yes -- Phase A |
| DTND | pgrid `compute_hand()` (haversine) | pgrid `compute_hand()` (Euclidean) | Yes -- Phase A |
| min_dtnd in trap fit | `res_m` (~90m) | `PIXEL_SIZE` (1m) | Yes -- pixel-sized |
| Pixel areas | Haversine spherical element | Uniform pixel_size^2 | Yes -- UTM |
| Valid mask | `np.isfinite(hand_flat)` only | `np.isfinite(hand_flat) & valid_mask_flat` | Yes -- nodata gaps |
| Basin/water detection | Inline copy | Imported from `dem_processing.py` | See issue #7 |
| MemoryError handling | Fatal (sys.exit) | Warning (branches=None) | Defensive -- OSBS is larger |
| Basin masking (flat terrain) | Fatal (`raise ValueError`, line 586) | Warning + continue (line 1186) | Defensive -- allows pipeline to complete |
| DTND n_bins in fit | 10 (default) | 10 (default) | Paper uses 50 -- see issue #8 |

**No unintentional divergences found.** The MERIT regression remains frozen as the validated source of truth.

---

## 4. Issues Found

### SIGNIFICANT (could affect production results)

#### Issue #1: FFT on nodata-contaminated mosaics

**Files:** `run_pipeline.py:812-841`, `spatial_scale.py:688-697`

The FFT runs on the full mosaic (step 2) BEFORE connected-component extraction (step 3a). For mosaics with nodata gaps (like the "interior" production run with 37.5% nodata per STATUS.md), the UTM path fills nodata with mean elevation (spatial_scale.py:694-695). This creates sharp transitions from real topography to flat mean-elevation fills, introducing spectral artifacts that could bias the Lc estimate.

- **Current tier tests are unaffected** -- all three use contiguous tile selections with 0% nodata.
- **Production run IS affected** -- the INTERIOR_TILE_RANGES define a non-rectangular selection that will have tile gaps filled with nodata, which then get mean-filled before FFT.
- **The TILE_SELECTION_MODE="interior" + INTERIOR_TILE_RANGES path has never been tested with the rebuilt pipeline.**

**Recommendation:** Use the contiguous tier 3 rectangle (R4C5-R12C14) as the production domain. This avoids nodata entirely. If the non-rectangular interior is needed later, restructure to run FFT on the connected-component-extracted region, or validate the nodata-filled FFT result against the contiguous-tile baseline.

#### Issue #2: INTERIOR_TILE_RANGES doesn't match documented interior

**File:** `run_pipeline.py:104-111`

```python
INTERIOR_TILE_RANGES = [
    "R4C10-R4C12",    # 3 tiles
    "R5C9-R5C12",     # 4 tiles
    "R6C9-R6C14",     # 6 tiles
    "R7C7-R7C14",     # 8 tiles
    "R8C6-R8C14",     # 9 tiles
    "R9C6-R9C14",     # 9 tiles
]                      # = 39 tiles
```

This is 39 tiles covering R4-R9, much smaller than:
- The 150-tile "interior" referenced in STATUS.md
- The 90-tile tier 3 test region (R4C5-R12C14)
- Rows R10-R12 are entirely absent

The `run_pipeline.sh` production wrapper sets `TILE_SELECTION_MODE=interior`, which uses these ranges. **The production mosaic would be ~39 tiles, not the 90 tiles validated in tier 3.** This is a different domain with different boundary effects, and its FFT Lc could differ from the tier test results.

**Recommendation:** Change `run_pipeline.sh` to use `TILE_RANGES="R4C5-R12C14"` instead of `TILE_SELECTION_MODE=interior`. This matches the validated tier 3 domain exactly. The INTERIOR_TILE_RANGES constant can remain in the code for reference but should not be the production default.

---

### MODERATE (correctness edge cases and science questions)

#### Issue #3: Flood filter is a no-op at 1m AND logically broken for binary masks

**File:** `run_pipeline.py:1146-1158`

Two problems:
1. **Currently dead code:** `identify_open_water()` returns zero detections at 1m resolution (both detection methods fail -- documented in `docs/synthetic_lake_bottoms.md`). So `n_flooded = 0` and the entire block is skipped.

2. **Logic flaw for when it IS active:** The filter iterates `ft` from 0 to 20 in 50 steps (`np.linspace(0, 20, 50)`, step ≈ 0.408), checking `basin_mask_flat > ft`. But `basin_mask` is binary (0 or 1 from `erode_dilate_mask`). For ft < 1.0, ALL basin pixels pass (`1 > ft` is true). For ft >= 1.0, ZERO basin pixels pass (`1 > 1.0` is false). The loop tries 3 values of ft below 1.0 (0, 0.41, 0.82), all producing identical results, then 47 values at or above 1.0, all producing zero results. The iteration over `ft` does nothing useful -- it's either "mark all basin pixels with HAND < 2.0" or "mark nothing."

**What actually happens:** On the first iteration (ft=0), `basin_mask_flat > 0` selects all basin pixels (since basin_mask is binary 0/1). The check becomes: "if <95% of basin pixels have HAND < 2.0, mark those pixels as invalid and break." All subsequent iterations with ft < 1 produce identical results. At ft >= 1, no pixels are selected (`1 > 1.22` is false), so zero pixels are marked and the loop breaks immediately. **The entire loop reduces to a single boolean check on the first iteration.** For non-binary basin masks (e.g., continuous confidence scores), the iteration would meaningfully threshold -- but `erode_dilate_mask` always produces binary output.

The same code exists in `merit_regression.py:544-555` where water detection DOES find water bodies at 90m resolution.

**Impact:** None currently (dead code). Will matter when synthetic lake bottoms enables water detection.

**Recommendation:** When implementing synthetic lake bottoms, replace this filter with simple boolean logic: `if n_flooded > 0: mark basin pixels with HAND < 2.0 as invalid, unless that would remove >95% of basin pixels`.

#### Issue #4: resolve_flats fallback uses unconditioned DEM

**File:** `run_pipeline.py:978-988`

If `resolve_flats` fails (catches `ValueError`), the pipeline falls back to the original DEM (pre-fill_pits, pre-fill_depressions). At 1m resolution on flat terrain like OSBS, the original DEM has millions of small pits from LIDAR noise. D8 routing on this would produce a severely fragmented stream network.

The merit_regression.py has the same fallback with a detailed rationale comment (lines 405-408: "Reverting all conditioning is safer than routing on an uninflated flooded DEM where flats have arbitrary flow direction"). The OSBS pipeline (line 981) omits this rationale -- just says "falling back to original DEM" without explaining why.

**Mitigating factor:** resolve_flats has never been observed to fail on OSBS data (no failures documented in any phase file or test log).

**Recommendation:** Add monitoring -- log a WARNING with the resolve_flats failure details and add a metadata flag in the output JSON. Consider falling back to the FLOODED DEM (post-fill_depressions) rather than the raw DEM, since that at least has connected drainage.

#### Issue #5: Stream parameters use wrong input variable

**File:** `run_pipeline.py:1414, 1434-1435`

```python
total_area_m2 = sum(elem["area"] for elem in params["elements"])
stream_depth = 0.001 * total_area_m2**0.4
stream_width = 0.001 * total_area_m2**0.6
```

`total_area_m2` is the sum of 16 fitted trapezoidal element areas (~235,000 m^2 for tier 3). This is the **per-hillslope representative area**, not the drainage area. Stream depth/width power laws (e.g., Leopold & Maddock) scale with contributing drainage area of the stream reach, which for the full OSBS domain would be ~90 km^2 = 90,000,000 m^2 -- three orders of magnitude larger.

Using per-hillslope area produces stream_depth ≈ 0.14m and stream_width ≈ 1.7m (`0.001 × 235187^0.4` and `0.001 × 235187^0.6`), which are unrealistically small for any natural channel. For comparison, with the full domain drainage area (~90 km^2 = 9×10^7 m^2), depth ≈ 1.6m and width ≈ 40m -- physically reasonable for OSBS headwater channels.

**This is labeled INTERIM and deferred to Phase E**, which is appropriate. But the power law is not just badly calibrated -- it's fed the wrong variable.

**Recommendation:** For Phase E, use either: (a) the total domain area or mean catchment area as the scaling variable, (b) empirical stream data for OSBS-region channels, or (c) MERIT Hydro parameters. Document clearly which area metric is used and why.

#### Issue #6: Lake pixels contaminate hillslope statistics

Since both water detection methods fail at 1m resolution -- `identify_basins()` at step 3b (pre-conditioning basin detection, `run_pipeline.py:931`) finds zero basins because no single elevation value exceeds the 25% histogram frequency threshold with 1m vertical precision, and `identify_open_water()` at step 3d (`run_pipeline.py:960`) finds zero water pixels because LIDAR returns on water produce noisy slopes well above the 1e-4 threshold:
- Lake/pond pixels pass through DEM conditioning as regular terrain
- `fill_depressions` fills them, `resolve_flats` inflates them
- Their HAND values are ~0 (filled surface near stream elevation)
- Their slope values are noisy (LIDAR vertical noise on water returns)
- They concentrate in the **lowest HAND bin**, contaminating the most hydrologically important hillslope element

At OSBS, lakes and wetland ponds cover a significant fraction of the landscape. Their inclusion in the lowest HAND bin (which already has the near-zero Q25 problem from issue T1) compounds the issue: **bin 1 contains both the hydraulically dead riparian strip AND misclassified water body pixels.**

**Recommendation:** Blocked on synthetic lake bottoms implementation. For interim output, document which HAND bins are likely contaminated and flag the lowest bin as unreliable.

#### Issue T1: HAND Bin 1 Near-Zero (Q25 = 0.00027m) -- REAL PROBLEM

*From tier test audit, 2026-02-26. Traced through Swenson's code, CTSM source, and tier test output.*

The tier 3 HAND bins are `[0, 0.00027, 1.61, 5.29, 25.1]`. Bin 1 holds 25% of all valid pixels at heights of ~0.08mm above the stream. This is **not a bug** -- the algorithm correctly excludes stream pixels (HAND=0) before computing quartiles, matching Swenson's `SpecifyHandBounds` (`terrain_utils.py:353`). The problem is that at 1m resolution on flat terrain, Q25 of non-zero HAND is genuinely sub-millimeter.

**Why it's a problem.** CTSM computes lateral flow via head gradient (`SoilHydrologyMod.F90:2260`):
```fortran
head_gradient = (col%hill_elev(c)-zwt(c)) / col%hill_distance(c)
```
For bin 1: `0.000077m / 44.4m = 1.7e-6`. Combined with typical transmissivity (~1e-3 m^2/s) and width (~200m), lateral flow is ~3.4e-7 m^3/s -- negligible. **25% of the gridcell area contributes zero lateral flow.** The riparian zone becomes hydraulically dead despite being the most hydrologically active zone.

**Why it doesn't happen at 90m.** Each pixel averages a 90x90m area, smoothing sub-meter elevation differences. Q25 falls naturally around 0.15-0.30m (Swenson reference bin 1 height = 0.178m). Swenson's algorithm assumes Q25 is on the order of decimeters -- no minimum bin width is needed because 90m resolution guarantees it.

**Swenson's code has no fix for this.** `SpecifyHandBounds` (`terrain_utils.py:364-408`) has a high-relief branch (Q25 > 2m) that widens bin 1, but **no low-relief branch** for when Q25 is vanishingly small. The algorithm hits the "common case" branch and produces the near-zero boundary.

**Resolution options** (science question for PI):
1. **Enforce a minimum bin 1 width** (e.g., Q25 >= 0.5m). Simple but arbitrary.
2. **Use 1x8 configuration** with octile binning -- 8 bins gives finer resolution in the 0-2m range without needing a minimum. However, octile Q12.5 would be even closer to zero, so this does NOT fix the fundamental problem.
3. **Aggregate HAND to ~90m equivalent** before binning -- preserves methodological continuity but discards 1m information.
4. **Accept it** -- if the science question is about wetland extent (area fraction below some HAND threshold), not lateral flow magnitude, the near-zero bin may be acceptable.

**Key files:**
- Swenson binning: `Representative_Hillslopes/terrain_utils.py:299-412`
- Our binning: `scripts/hillslope_params.py:39-140` (line 76: `hand > 0` filter)
- Pipeline call: `scripts/osbs/run_pipeline.py:1216-1218`
- CTSM head gradient: `ctsm5.3/src/biogeophys/SoilHydrologyMod.F90:2260`

#### Issue T4: Domain Boundary Effects -- MODERATE RISK

*From tier test audit, 2026-02-26.*

The tier 3 domain (9x10 km) has hard rectangular edges that cut through real drainage basins. pysheds handles this by saving/restoring "rim" pixels (`pgrid.py:1909-1910,1934-1937`) -- flow paths that reach the boundary terminate, producing incomplete HAND/DTND values.

**Affected vs unaffected parameters:**

| Parameter | Boundary-sensitive | Why |
|---|---|---|
| Slope | No | Local DEM gradient, boundary-independent |
| Aspect | No | Local DEM gradient, boundary-independent |
| HAND | Low risk | Vertical elevation difference; flow path truncation mainly shortens path but doesn't change height above stream much |
| DTND | Higher risk | Flow-path distance; truncated paths produce artificially short DTND values for boundary pixels whose real drainage exits the domain |
| Width | Moderate | Fitted from A_sum(d) curve which depends on DTND distribution |
| Area fraction | Moderate | Boundary pixels with invalid DTND may be filtered out, changing aspect distributions |

**Scale of the boundary zone.** With max(DTND) = 1989m, pixels within ~2km of the boundary could have truncated flow paths. Perimeter = 38 km, buffer area = 38 x 2 = 76 km^2 out of 90 km^2 total. However, most interior pixels drain to interior streams well before reaching the boundary. At 0.23% stream cells, the mean inter-stream distance is ~Lc/2 = 178m. Most pixels are within one Lc of a stream.

**Swenson's approach.** Swenson processes entire 0.9x1.25 degree gridcells (~100 km^2) with no explicit boundary handling. At 90m resolution, the domain is large relative to Lc (763m) and boundary effects are negligible. For our domain with Lc=356m, the domain/Lc ratio is 25-28, comparable to Swenson's ~130. Boundary effects should be proportionally small, but not as negligible as for global data.

**Mitigating factors already in the pipeline:**
1. DTND tail removal clips extreme DTND values (2.8% of pixels), which likely includes many boundary-affected pixels.
2. Connected component extraction ensures the processing region is contiguous valid data.
3. Nodata edge trimming removes purely-nodata rows/columns.

**Recommendation:** Likely modest for tier 3. Could be tested empirically by running on the interior of the interior (crop 2km from each edge) and comparing parameters.

---

### NOT A PROBLEM (investigated, resolved)

#### Issue T2: DTND Tail Removal (2.8%) -- NOT A PROBLEM

*From tier test audit, 2026-02-26.*

Tier 3 removed 2,482,585 pixels (2.8%). The algorithm (`hillslope_params.py:432-480`) fits an exponential to the DTND distribution and clips pixels beyond the 5% PDF threshold. This is **identical to Swenson's `TailIndex`** (`terrain_utils.py:286-296`) -- same exponential fit, same 5% threshold (`hval=0.05`).

MERIT validation removed 33,199 pixels (1.96%). The OSBS 2.8% is only 1.4x higher, explained by 1m resolution preserving more extreme ridge-top pixels, low-relief terrain amplifying relative noise in DTND, and larger sample (90M vs 1.7M) pushing extreme values further from the bulk. No changes needed.

#### Issue T3: AREA NetCDF Field (0.24 km^2) -- MISLEADING BUT HARMLESS

*From tier test audit, 2026-02-26.*

The scalar `AREA` field in our NetCDF is 0.24 km^2 because it sums the 16 trapezoidal-fitted element areas (`run_pipeline.py:1414`). Swenson's reference file has `AREA = 12,654 km^2` (the full 0.9x1.25 degree gridcell), but this was **added post-hoc via ncks** -- Swenson's `representative_hillslope.py` does not write an `AREA` variable at all.

**CTSM does not read `AREA`.** `surfrd_hillslope` (`surfrdMod.F90:1048-1155`) reads `nhillslope`, `nmaxhillcol`, and `nhillcolumns`. `HillslopeHydrologyMod.F90:328` reads `hillslope_area` (the 16-element array). **Neither reads the scalar `AREA` field.** CTSM uses `grc%area(g)` from the surface dataset for gridcell area.

**hillslope_area values (what CTSM actually uses):**

| | Our tier 3 (per element) | Swenson ref (per element) |
|---|---|---|
| Range | 13,361 - 16,474 m^2 | 63,095 - 105,473 m^2 |
| Sum (16 elements) | 235,187 m^2 | 1,187,739 m^2 |
| Mean | 14,699 m^2 | 74,234 m^2 |

Our element areas are ~5x smaller than Swenson's. Physically correct -- our denser stream network (Lc=356m vs 763m) produces smaller catchments and smaller representative hillslopes. CTSM uses these areas for:

1. **Column weights** (`HillslopeHydrologyMod.F90:520`): `wtlunit(c) = (hill_area(c)/hillslope_area(nh)) * (pct_hillslope(nh)*0.01)` -- ratios, so absolute scale cancels.
2. **Stream network scaling** (`HillslopeHydrologyMod.F90:486`): `nhill_per_landunit = grc%area * wtgcell * pct_hillslope / hillslope_area` -- smaller hillslope_area means more copies tiled across the gridcell, giving a denser stream network. Correct.
3. **Lateral flow flux** (`SoilHydrologyMod.F90:2377`): `qflx_latflow = volume / hill_area` -- smaller area gives larger specific flux per unit area. Correct.

No changes needed.

---

### MINOR (code hygiene, edge cases, documentation)

#### Issue #7: merit_regression.py has frozen copies of basin/water detection code

**Files:** `merit_regression.py:203-307` vs `dem_processing.py:17-122`

The merit regression has inline copies of `_four_point_laplacian`, `_expand_mask_buffer`, `erode_dilate_mask`, `identify_basins`, and `identify_open_water`. The comment in `dem_processing.py:7` explains this is intentional: "merit_regression.py stays frozen as a regression test."

Correct design. But any future fix to these functions must be applied to BOTH files, with re-validation of MERIT correlations.

**Recommendation:** Add a comment in merit_regression.py at line 203 noting this duplication is intentional, referencing dem_processing.py as the "evolving" copy.

#### Issue #8: Trapezoidal fit uses 10 DTND bins vs paper's 50

**Files:** `hillslope_params.py:149` (default n_bins=10), paper Figure 8 caption ("n=50")

The paper's Figure 8 uses n=50 bins for the A_sum(d) curve. The pipeline uses 10. The MERIT validation achieves >0.99 correlation with 10 bins, so this is sufficient. But it's an undocumented divergence from the paper.

**Recommendation:** Add a comment: "Paper uses n=50 bins for illustration (Fig 8), but 10 bins suffice for the polynomial fit. MERIT validation confirms >0.99 correlation with 10 bins."

#### Issue #9: run_pipeline.sh missing `set -euo pipefail`

**File:** `scripts/osbs/run_pipeline.sh`

The main SLURM wrapper doesn't use `set -euo pipefail`, while all three tier wrappers do. If `conda activate` fails silently or `module load` errors occur, the pipeline could run with wrong Python or missing packages.

**Recommendation:** Add `set -euo pipefail` after the `#SBATCH` directives.

#### Issue #10: A_thresh safety valve is aggressive

**File:** `run_pipeline.py:1021-1024`

```python
if max_acc < accum_threshold:
    accum_threshold = int(max_acc / 100)
```

Dividing by 100 is arbitrary and could produce pathologically dense stream networks for small domains. Mitigating factor: only fires when max_acc < A_thresh, which is unlikely for any domain >= 1 tile.

**Recommendation:** Add a comment noting the heuristic. Consider `max_acc / 10` or `max_acc / 4`.

#### Issue #11: `grid.inflated_dem = grid.inflated` is fragile

**Files:** `run_pipeline.py:1055`, `merit_regression.py:478`

This alias exists because `river_network_length_and_slope` accesses `self.inflated_dem` by hardcoded attribute name. If the pysheds fork method is renamed or the internal attribute changes, this breaks silently.

**Recommendation:** Add a comment explaining WHY this alias exists. Consider fixing the attribute name in the fork.

#### Issue #12: Missing equation references in comments

**File:** `run_pipeline.py`

Several key computations lack references to the paper equations they implement:
- Line 834: `accum_threshold = int(0.5 * lc_px**2)` -- Add "Swenson Eq 6".
- Line 1378: `da_dist = sum(fitted_areas[:h_idx+1]) - fitted_areas[h_idx] / 2` -- Add "Swenson Eq 17".
- Line 1171: `smallest_dtnd = 1.0` -- Add: "Fixed 1.0m minimum, same as MERIT regression (`merit_regression.py:567`). Separate from `min_dtnd` in `fit_trapezoidal_width` (which IS resolution-dependent: MERIT uses `res_m` ~90m at line 651, OSBS uses `PIXEL_SIZE` 1m at line 1282)."
- Lines 1434-1435: Stream power law should note `total_area_m2` is per-hillslope area, not drainage area.

#### Issue #13: Flood filter threshold loop intent unclear

**File:** `run_pipeline.py:1148-1158`

Even though the flood filter is dead code (issue #3), it needs a comment explaining the intent of the threshold loop and noting that it operates on a binary mask (making the loop a no-op).

---

## 5. Future-Proofing: 1x8 Configuration

If the PI approves 1 aspect x 8 elevation bins:

### What changes in the pipeline

| Component | Current (4x4) | 1x8 Change | Difficulty |
|---|---|---|---|
| `N_ASPECT_BINS` | 4 | 1 | Trivial |
| `N_HAND_BINS` | 4 | 8 | Trivial |
| `ASPECT_BINS` | 4 cardinal tuples | Single `[(0, 360)]` | Trivial |
| `compute_hand_bins` | Quartile (4 bins) | Octile (8 bins) | Moderate -- needs percentile logic update |
| `pct_hillslope` | 4 values summing to 100 | Single `[100]` | Trivial |
| `nhillslope` dim | 4 | 1 | Trivial |
| `nmaxhillcol` dim | 16 | 8 | Trivial |
| Trapezoidal fit | 4 fits (1 per aspect) | 1 fit (all pixels) | Better -- 4x more data |
| `catchment_mean_aspect` | Used for binning | Still needed for circular mean per element | Keep |
| Aspect in NetCDF | Per-element circular mean | Single circular mean for all 8 elements | OK |
| `write_hillslope_netcdf` | `n_columns=16, n_aspects=4, n_bins=4` hardcoded (lines 528-530) | Must be parameterized | Moderate -- also affects `downhill_column_index` logic (lines 554-558) |
| Area fraction validation | 0.82 correlation (weakest) | **Eliminated** -- no aspect partitioning | Improvement |

### The critical issue for 1x8

`compute_hand_bins` (hillslope_params.py:39-140) is hardcoded for 4 bins:
- Line 133: `quartiles = [0.25, 0.5, 0.75, 1.0]`
- Line 88-127: High-relief branch uses `0.33` and `0.66` (thirds of above-bin1 pixels)

For 8 bins, this needs octile percentiles: `[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]`. The high-relief branch would need 7ths instead of thirds. The 2m HAND constraint still applies but with 12.5% of pixels (instead of 25%), making Q12.5 even closer to zero.

**Recommendation:** Parameterize `compute_hand_bins` to accept `n_bins` argument. The percentile logic generalizes naturally. The high-relief branch can split above-bin1 pixels into `n_bins - 1` equal groups.

### Impact on existing concerns

- **HAND bin 1 near-zero (T1):** NOT fixed. Q12.5 would be even closer to zero than Q25. The near-zero lowest bin is a fundamental consequence of 1m resolution on flat terrain, not of how many aspects we use.
- **DTND tail removal (T2):** Unchanged -- already not a problem.
- **AREA field (T3):** Still misleading but still harmless. CTSM doesn't read the scalar AREA field regardless.
- **Domain boundary effects (T4):** Unchanged -- driven by domain geometry vs Lc, not by aspect partitioning.

The main benefit of 1x8 is eliminating the weakest parameter (area fraction correlation = 0.82 in MERIT validation) by bypassing aspect partitioning entirely.

---

## 6. Future-Proofing: Synthetic Lake Bottoms

When implemented, synthetic lake bottoms will affect:

| Pipeline Step | Impact | Notes |
|---|---|---|
| Basin detection (3b) | **Needs new detection method** | Current histogram+slope methods fail at 1m |
| DEM conditioning (3d) | Lake pixels get synthetic bottoms BEFORE fill_pits | Two-DEM strategy per docs |
| Flow routing (3e) | Better -- flow paths through lakes instead of across flat surfaces | Main benefit |
| HAND (Step 4) | Lake-interior HAND becomes meaningful | Currently ~0 (flooded surface) |
| Hillslope params (Step 5) | Cleaner lowest HAND bin | Lake pixels no longer contaminate bin 1 |
| Flood filter (5a) | **Needs fix** -- currently broken for binary masks (issue #3) | Will matter when detection works |

### What needs to change before implementing

1. **New water detection method** -- Neither histogram-based (`identify_basins`) nor slope-based (`identify_open_water`) works at 1m. Options from `docs/synthetic_lake_bottoms.md`: NEON open water product, spectral data fusion, or a multi-resolution approach (compute slope at coarser resolution before thresholding).

2. **Synthetic bottom generator** -- Not yet implemented. The `dem_processing.py` module is the right place. The distance-from-shore bowl approach (Hollister 2011) seems most appropriate for OSBS.

3. **Flood filter fix** -- Replace the threshold loop with simple boolean logic (issue #3).

4. **Separation of flow-routing DEM from physical-characteristics DEM** -- Already designed in `docs/synthetic_lake_bottoms.md` as the "two-DEM strategy." The pipeline already partially does this (slope computed from original DEM, flow routing from conditioned DEM), but synthetic bottoms need to be applied ONLY to the flow-routing DEM.

### Impact on existing concerns

- **HAND bin 1 (T1):** Lake bottoms add depression structure but don't change the fundamental HAND distribution. The near-zero Q25 comes from the vast flat areas between streams, not from water bodies.
- **DTND tail removal (T2):** May slightly change the tail distribution if lakes redirect flow paths, but the 5% threshold algorithm is self-calibrating.
- **AREA field (T3):** Unaffected.
- **Domain boundary effects (T4):** Unaffected.

Water detection itself (`identify_basins()` and `identify_open_water()`) currently produces zero detections at 1m resolution across all three tier tests. The histogram-based threshold (>25%) and slope threshold (<1e-4) were calibrated for 90m data. See `docs/synthetic_lake_bottoms.md` for details.

---

## 7. SLURM Wrapper Review

| Wrapper | Resources | Shell Safety | PYTHONPATH | Notes |
|---|---|---|---|---|
| `run_pipeline.sh` | 4hr, 64GB | **Missing `set -euo pipefail`** | Missing | See issue #9 |
| `run_pipeline_tier1.sh` | 30min, 8GB | `set -euo pipefail` | Exported | Clean |
| `run_pipeline_tier2.sh` | 1hr, 32GB | `set -euo pipefail` | Exported | Clean |
| `run_pipeline_tier3.sh` | 4hr, 64GB | `set -euo pipefail` | Exported | Clean |

`run_pipeline.sh` doesn't export PYTHONPATH with the pysheds fork. It sets `PYSHEDS_FORK` but doesn't add it to PYTHONPATH. The Python script handles this internally (line 49-50: `sys.path.insert(0, pysheds_fork)`), so not a functional issue -- just inconsistent with the tier wrappers.

Tier resource allocations are well-calibrated based on Phase B results (peak 29.2 GB for 90M pixels at 1m). The tier 3 allocation of 64GB provides ~2x headroom.

---

## 8. Comment Quality Assessment

**Strengths:**
- Swenson reference annotations: `"Swenson rh:699-700"`, `"Swenson tu:307, 353"`, `"Swenson rh:1725-1751"`. Cross-referencing is trivial.
- Phase decision annotations: `"Phase A: pgrid slope_aspect() and compute_hand() are UTM-aware"`, etc.
- Divergence comments explain WHY: e.g., the `min_dtnd = PIXEL_SIZE` comment at line 1277.
- Module docstring (lines 1-28) provides good orientation.

**Gaps:** See issues #12 and #13.

---

## 9. Summary of All Issues

| # | Severity | Issue | Action | Blocks? |
|---|---|---|---|---|
| 1 | **Significant** | FFT on nodata-contaminated mosaics | Restrict production to contiguous tiles | Production run |
| 2 | **Significant** | INTERIOR_TILE_RANGES (39 tiles) != tested domain (90 tiles) | Use TILE_RANGES in production | Production run |
| 3 | Moderate | Flood filter is dead code AND logically broken | Fix when synthetic lake bottoms lands | No |
| 4 | Moderate | resolve_flats fallback to unconditioned DEM | Add monitoring; consider flooded-DEM fallback | No |
| 5 | Moderate | Stream params use per-hillslope area, not drainage area | Fix in Phase E | Phase E |
| 6 | Moderate | Lake pixels contaminate lowest HAND bin | Blocked on synthetic lake bottoms | Phase E |
| T1 | Moderate | HAND bin 1 near-zero (Q25 = 0.00027m) | PI decision on binning approach | Science question |
| T4 | Moderate | Domain boundary effects | Likely mitigated; empirical test possible | No |
| 7 | Minor | merit_regression.py has frozen basin/water code | Document intentional duplication | No |
| 8 | Minor | 10 DTND bins vs paper's 50 | Add explanatory comment | No |
| 9 | Minor | run_pipeline.sh missing `set -euo pipefail` | Add it | No |
| 10 | Minor | A_thresh safety valve /100 is arbitrary | Add comment | No |
| 11 | Minor | `grid.inflated_dem` alias is fragile | Document; consider fork fix | No |
| 12 | Minor | Missing equation references in comments | Add references | No |
| 13 | Minor | Flood filter threshold loop intent unclear | Add comment | No |

Tier test issues T2 (DTND tail removal) and T3 (AREA field) are resolved -- not problems.

---

## 10. Verification Plan

After addressing issues #1-2, run the full tier test suite:

1. **Tier 1 (R6C10):** Smoke test -- confirm pipeline runs, output structure matches reference NetCDF
2. **Tier 2 (5x5):** Medium scale -- verify Lc, HAND distribution, all 16 elements populated
3. **Tier 3 (R4C5-R12C14):** Full scale -- compare to existing tier 3 results from Phase D
4. **MERIT regression:** Run after any changes to shared modules -- must still PASS

For production readiness:
- Verify NetCDF structure with `ncdump -h` against `hillslopes_osbs_c240416.nc` (Swenson reference)
- Validate `pct_hillslope` sums to 100
- Check all 16 elements have non-zero height, distance, width, area
- Verify aspect values span all 4 quadrants
- Confirm `hillslope_area` values are physically reasonable (~10,000-20,000 m^2 for Lc=300m)
