# State of the Union: Swenson Hillslope Implementation for OSBS

Date: 2026-05-06

## Executive Summary

We are implementing Swenson & Lawrence (2025) representative hillslope methods to generate custom hillslope parameters for OSBS using 1m NEON LIDAR data. The goal is to replace the current global 90m MERIT-derived hillslope file (`hillslopes_osbs_c240416.nc`) with site-specific parameters that capture the fine-scale drainage structure of this low-relief wetlandscape.

**The methodology is validated and the pipeline produces scientifically defensible output.** MERIT validation achieved >0.95 correlation with Swenson's published data on 5 of 6 parameters. Phases A-D are complete: pysheds handles UTM CRS, flow routing runs at full 1m resolution, Lc is established (356m for the production domain), and the pipeline has been rebuilt with all known fixes and verified by equation-by-equation audit. Output is CTSM-compatible NetCDF matching the Swenson reference structure.

**Phase E (parameter completion) is complete.** Pipeline produces 1 aspect × 24 HAND bins + 1 lake column (25 columns total). Bin scheme is TAI-focused (12 FZ + 12 upland, 0.25 m floor in TAI core, smooth 2× width progression). Q01/Q99 outlier trim on raw HAND. NWI water masking via dual-mask approach. NEON slope/aspect products adopted directly.

**Phase E.5 + E.6 + Phase G Stage 1 are complete.** The submerged lake column at chain index 1 with `hill_elev = -6 m` is built into the pipeline and CTSM ingestion is validated. `spillheight = 0.0` retires the elevation-offset SourceMod mechanism per the 2026-04-30 PI reframe. Per-rep rescale (2026-05-05) gives the lake column a sensible 12.3% landunit weight (matching NWI water fraction) instead of a runaway 98.7%.

**Phase F is in progress.** `osbs5.swenson.spinup` (fresh accelerated startup, NOT branched from osbs2) ran 100 years 2026-05-06; 8 spinup-analysis plots show sensible early-spinup behavior of the new hillslope file. Next milestone: extend to ~600 yr accelerated, verify TOTECOSYSC drift < 3% over last 50 yr.

**Phase G Stage 2 (routing-on validation) is deferred.** Requires switching from single-point mode to explicit `LND_DOMAIN_MESH` configuration (single-point `grc%area = spval` breaks `nhill_per_landunit`). Out of scope until lateral-flow physics is the priority.

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

Pipeline uses NEON DP3.30025.001 products directly instead of computing slope/aspect from the raw DTM. NEON's 3x3 pre-filter reduces TIN interpolation noise on OSBS's flat terrain and eliminates border artifacts. Comparison across 90 production tiles (commit 418880c): slope Pearson r=0.91, aspect circular r=0.84. See `output/osbs/slope_aspect_comparison/`.

### NWI water masking implemented (Phase E)

NWI data (HU8_03080103 watershed): 103 open water features covering 10.7M pixels (11.9% of production domain). The pipeline uses a dual-mask approach — the natural stream network drives catchment delineation while a "wide mask" (natural streams + lake pixels) defines HAND reference surfaces. Land pixels adjacent to lakes get HAND measured from the lake edge; lake pixels themselves get HAND=0 and are excluded from all binning and tail-fitting statistics.

Boundary forcing (Swenson's original approach) was tried first and broke catchment delineation at 1m resolution by fragmenting ~800 catchments into ~222K. The dual-mask approach preserves the natural catchment count (~508 on the production domain). Lc is insensitive to water masking (raw DEM gives Lc=356m; both masking approaches destroy the spectral peak). See `phases/E-complete-parameters.md` and `output/osbs/water_masking_comparison/`.

---

## Problems — All Resolved

The ten problems identified in the initial audit have all been resolved. See the referenced phase docs for detailed analysis, bug writeups, and resolution history.

| # | Problem | Resolved in | Reference |
|---|---------|-------------|-----------|
| 1 | DTND used Euclidean distance (wrong algorithm) | Phase A | `phases/A-pysheds-utm.md` |
| 2 | Flow routing subsampled to 4m (discarded 94% of data) | Phase B | `phases/B-flow-resolution.md` |
| 3 | Characteristic length scale Lc undefined at 1m | Phase C | `phases/C-characteristic-length.md` |
| 4 | N/S aspect swap from sign convention mismatch | Phase A + E | `docs/ns-aspect-bug.md` |
| 5 | DEM conditioning erases microtopography at 1m | Phase E (accepted) | `phases/E-complete-parameters.md` |
| 6 | Stream channel parameters use interim scaling | Phase E (accepted) | `phases/E-complete-parameters.md` |
| 7 | Bedrock depth was placeholder (1e6) | commit 11f465e | `phases/E-complete-parameters.md` |
| 8 | FFT parameters untested for OSBS | Phase C | `phases/C-characteristic-length.md` |
| 9 | ~400 lines of duplicated code between pipelines | Phase D | `phases/D-rebuild-pipeline.md` |
| 10 | pysheds fork: 33 deprecation warnings + broken tests | Phase A | `phases/A-pysheds-utm.md` |

**Production outcome:** The pipeline produces a CTSM-compatible NetCDF at `output/osbs/<DATE>_production/` with 16 HAND bins (1 aspect × 16, hybrid: 5 fixed 10cm + 10 log-spaced to Q99 + 1 sentinel), NWI water masking applied (dual-mask approach), NEON slope/aspect products, Lc = 356m on the 90-tile production domain. The 2026-04-09 NetCDF used the prior 1×8 A2 scheme; the 2026-04-14 run supersedes it with the 16-bin hybrid.

---

## Work Flow: What Needs to Happen and In What Order

**Phase tracking files:** `phases/` — one file per phase with tasks, results, and decisions.

### Phase A: Fix pysheds for UTM — Complete

Both the DTND problem (#1) and the slope/aspect problem (#4) stemmed from pysheds assuming geographic coordinates. Phase A made pysheds UTM-aware, fixed 33 deprecation warnings in `pgrid.py`, and expanded the test suite (28 synthetic DEM tests, mutation testing with 100% effective score). Validated against MERIT regression (all 6 parameters within tolerance) and R6C10 UTM smoke test (14/14 checks). See `phases/A-pysheds-utm.md`.

### Phase B: Resolve flow routing resolution — Complete

Full 1m resolution at 64GB confirmed correct; 4x subsampling was a premature optimization. Resolution comparison on 5x5 block and full contiguous region: height/distance correlations >0.999 across 1m/2m/4m (resolution-insensitive), slope systematically underestimated at coarser resolutions (~50% lower at 4m in the lowest HAND bin). See `phases/B-flow-resolution.md`.

### Phase C: Establish trustworthy Lc — Complete (Lc = 356m, A_thresh = 63,362)

Restricted-wavelength FFT (`min_wavelength = 20m`) identifies the drainage-scale peak and filters the k² micro-topographic artifact at ~8m. Lc varies with domain size: tier 1 (1 tile) = 541m, tier 2 (25 tiles) = 479m, tier 3 (90 tiles, production) = 356m. Physical validation passes on the 5x5 tile block: P95 DTND/Lc = 1.17, mean catchment / Lc² = 0.876. Sensitivity sweep confirmed Lc is insensitive to FFT preprocessing parameters. See `phases/C-characteristic-length.md`.

### Phase D: Rebuild pipeline with fixes — Complete

Pipeline rebuilt with all Phase A/B/C fixes: pysheds hydrological DTND, pgrid Horn 1981 slope/aspect, full 1m resolution, dynamic Lc via FFT. Shared modules extracted (`hillslope_params.py`, `spatial_scale.py`, `dem_processing.py`) to eliminate ~400 lines of duplicated code between the MERIT validation and OSBS pipelines. Equation-by-equation audit passed; tier 3 production run verified. See `phases/D-rebuild-pipeline.md`.

### Phase E: Complete the parameter set — Complete (2026-04-09)

All parameters finalized. 1 aspect × 16 HAND bins, hybrid fixed+log (5 × 10cm fixed in the 0-0.5m TAI zone + 10 log-spaced to Q99 + 1 sentinel; moved 2026-04-14 from the prior 8-bin A2 scheme per PI request for tighter TAI resolution). NEON DP3.30025.001 slope/aspect products adopted directly. NWI water masking via dual-mask approach. Stream parameters left as interim power-law values (osbs2 uses `use_hillslope_routing = .false.`, params never read). All PI questions resolved. See `phases/E-complete-parameters.md` and `docs/hillslope-binning-rationale.md`.

### Phase E.5: Bin redesign and spillheight removal (Reframed 2026-04-30)

**Status: In design.** Outlier strategy locked (2026-05-02). Bin scheme exploration complete; final scheme awaits PI selection. See `phases/E.5-bin-redesign.md`.

**Outlier cutoffs (locked 2026-05-02):** Q01 = -6.34 m (FZ) and Q99 = +17.46 m (upland). True discard, ~2% of land pixels removed. Cutoffs match classical sigma clipping on the raw HAND distribution (mean − 2σ = −6.10 m; mean + 3σ = +17.65 m); the asymmetric σ multipliers reflect the right-skewness of the distribution. Lee 2023 OSBS spill depths (mean 2.64 m + 2·SD = 4.54 m, n = 14) used as a one-way physical sanity check — our deep cutoff is 1.8 m deeper than any wetland in their field survey. See phase doc 2026-05-02 log entry for the full defense framing.

**PI meeting reframe (2026-04-30).** The PI meeting dissolved several constraints we had been working against and pivoted the approach:

1. **The flood zone is the dry continuation of the same basin as the lake**, not a separate buffer. NWI lake pixels and surrounding non-NWI flood-zone pixels were both raised by `fill_depressions` to the same spill elevation. Physically and hydrologically continuous.

2. **The PI's spillheight SourceMod is being retired.** Its purpose was to fake low-lying wetland and flood-zone columns by lowering all `hill_elev` values uniformly. Raw HAND now provides those columns from real data. SPILLHEIGHT will be set to 0 (effectively disabling it); SourceMods stay in place for now.

3. **Lake column `hill_elev` becomes data-derived**, not the arbitrary `-SPILLHEIGHT` convention. **Locked 2026-05-04: -6.0 m** as a chain-bookkeeping value, set 0.87 m below the deepest land bin mean (-5.13 m, bin 1 of 24-bin scheme) to satisfy chain monotonicity. Empirical context: NWI lake-surface mean -2.53 m; Lee 2023 / pipeline spill depths 2.64-3.33 m. The -6.0 m value doesn't represent any single physical lake; it's the chain reference. Tuning deferred to model output. See audit doc Section 5.2.1.

4. **The flood zone gets ≥50% of the columns** (possibly ⅔ per PI). FZ areas should increase going uphill, mirroring upland-bin behavior. Bin design is iterative — outlier removal first, then a baseline log-spaced 16-bin scheme, then permutations as needed.

**Working notes:**
- Bin design is **not** a "final" decision on first pass. Expect iteration.
- Observation-date research (NEON LIDAR, NWI, Lee LIDAR) is parallel, not blocking.
- Phase G is folded into this phase (the "submerged lake column with spillheight SourceMod" framing no longer applies).

**Reference:** `phases/E.5-bin-redesign.md` for the full task list, log, and iteration record. Audit doc Section 5.x and 6.7 sections about spillheight are flagged as superseded. `docs/data-acquisition-dates.md` documents the NEON DTM, NWI mask, and Lee 2023 LIDAR vintages (NEON 2023-05, NWI 2017 1m TC; Lee LIDAR ambiguous, awaiting confirmation from Cohen).

### Phase E.6: NWI water mask regeneration (Complete 2026-04-30)

**Status: Complete.** Fix applied; mask regenerated. Awaits next pipeline rerun (Phase E.5 / Phase F) to consume.

**Discovery (2026-04-29):** Inspection of the categorical and contamination spatial maps revealed that `data/mosaics/production/water_mask.tif` had **holes inside larger lake polygons**. Pixels in these holes had `water_mask = 0` despite sitting fully inside the NWI lake outline, surrounded on all sides by `water_mask = 1`.

**Root cause:** The NWI shapefile contains nested polygon rings (e.g., outer "wetland" polygon with inner "open water" rings). The original rasterization treated inner rings as **negative space** rather than additional water area.

**Scope (measured 2026-04-30):** 400,219 hole pixels in 7 connected components (4× higher than the original 50–200K estimate). Largest single hole: 377,916 px (~0.38 km²).

**Fix:** `scripts/osbs/generate_water_mask.py` patched with a `scipy.ndimage.binary_fill_holes` post-process after rasterization. Flood-fills any topologically enclosed background pixel; safe at OSBS because the production domain has no real islands inside lakes. Regenerated mask: 11,082,394 water pixels (12.3% of domain), up from 10,682,175 (11.87%). Post-fix diagnostic confirms zero remaining hole pixels.

**Diagnostic:** `scripts/osbs/diagnose_water_mask.py` (new). Plots `output/osbs/water_mask_diagnostic/before.png` and `after.png` — simple blue-on-white renderings of the mask with hole stats. Pre-fix backup retained at `data/mosaics/production/water_mask_pre_holefill.tif`.

**Reference:** `docs/lake-column-ctsm-audit.md` Section 7.5 — original quantitative scope and root-cause analysis.

### Phase F: Validate and deploy

**Status: In progress** (osbs5.swenson.spinup, 100-yr accelerated AD
spinup complete 2026-05-06; longer spinup to convergence pending after
HiPerGator maintenance window).

**Phase F runs in parallel with Phase G Stage 1.** The 2026-04-25 PI
direction folded the lake column into the pipeline output (single
submerged column at chain index 1, not a separate landunit). There is
no "lake-less" version of the hillslope file to validate as a
standalone baseline — the file produced by the pipeline always
includes the lake column. The osbs5 case validates BOTH the long
spinup behavior of the file (Phase F) AND that CTSM correctly ingests
the lake-included structure with sensible column weights (Phase G
Stage 1). The two phases share a single validation case but track
distinct deliverables:

| Phase | Deliverable | osbs5's role |
|---|---|---|
| **F** | Long convergent spinup with the new hillslope file; verify drift criterion | Currently 100 yr (early ramp-up); needs ~600 yr accelerated to converge |
| **G Stage 1** | Lake column construction + CTSM ingestion verified | Confirmed: 25 columns, lake `wtlunit` ≈ 12.3%, monotone chain, no NetCDF read errors |

**Phase F approach.** Built `osbs5.swenson.spinup` as a **fresh
startup** (per user direction, not branched from osbs2) with our
2026-05-05 hillslope file. This isolates the hillslope file's effect
from sgerber's 600-year initial-condition history. Configuration
matches osbs4-6 except: hillslope_file repointed, `spillheight = 0.0`
(SourceMods retained but inert per Phase E.5 reframe), accelerated
spinup. `use_hillslope_routing = .false.` (matches osbs4-6).

**Key context:** osbs2/osbs4-6 run with `use_hillslope_routing =
.false.` and `PCT_LAKE = 0`. Phase F matches this exactly. Under this
config, columns are **hydrologically isolated 1D soil columns** —
multi-column structure provides aspect-dependent radiation,
elevation downscaling, independent per-column water balance, but NO
lateral water exchange. TAI dynamics in the strict sense (lateral
flow → upland-saturation-driven lake filling) are dormant; that's
**Phase G Stage 2** territory. See `phases/F-validate-deploy.md`
"What 'routing off' means for interpretation" for the full framing.

**Done:**
- ✅ Custom hillslope file built and validated (Phase E.5)
- ✅ Physical plausibility checks (elevation range, stream network, HAND values)
- ✅ Apples-to-apples comparison vs Swenson reference (parameter table in `STATUS.md`)
- ✅ `osbs5.swenson.spinup` case created, built, and run (100 yr)
- ✅ 8 spinup-analysis plots generated (5 gridcell + 3 column-level)
- ✅ New plotting tools: `scripts/hillslope.analysis/plot_col_timeseries.py`

**Pending:**
- Extend spinup to ~600 yr accelerated (matches sgerber's osbs4 length)
- Optional post-AD continuation (~200 yr, `CLM_ACCELERATED_SPINUP = off`)
- Re-generate plots over longer span; verify `TOTECOSYSC` drift < 3% over last 50 yr
- Optional: parallel case with Swenson reference hillslope under same osbs5 setup for direct attribution

**Deliverable:** Validated hillslope file ready for production runs +
convergent spinup baseline. See `phases/F-validate-deploy.md` 2026-05-06
log entry for the full milestone summary.

### Phase G: Submerged lake column in hillslope file

**Status: Stage 1 complete (in parallel with Phase F); Stage 2 deferred.**

The original phase ordering treated G as sequential after F (F
establishes a "lake-less" baseline; G adds the lake column on top).
That ordering dissolved when the 2026-04-25 PI direction folded the
lake column into the pipeline output as a single submerged column
(not a separate landunit). The pipeline's hillslope NetCDF always
includes the lake column now — there is no lake-less version to use
as F's baseline. Phase F and Phase G Stage 1 share the osbs5
validation case, with each tracking a distinct deliverable.

| Stage | Goal | Status |
|---|---|---|
| **1. Lake column construction + CTSM ingestion** | Add submerged column to the hillslope NetCDF; verify CTSM ingests it correctly and column weights are sensible | **Complete** (Phase E.5 + 2026-05-05 per-rep rescale; validated in osbs5 100-yr run alongside Phase F) |
| **2. Routing-on validation** | Enable `use_hillslope_routing = .true.`; verify lateral flow produces TAI behavior (water table rise near lake → decomposition suppression → CH4) | **Deferred** — requires (a) explicit `LND_DOMAIN_MESH` (single-point `grc%area = spval` breaks `nhill_per_landunit`), (b) hydraulic-conductivity sanity check |

**Direction changed 2026-04-09** after PI consultation — weir overflow plan abandoned. **Scope refined 2026-04-25** — lake column stays narrow (NWI water only); unmapped depression pixels go into flood-zone bins from Phase E.5.

Add one column to the hillslope NetCDF representing the aggregate of all NWI-mapped lake area as a single submerged column with negative `hill_elev`. CTSM's existing lateral flow machinery draws water from adjacent upland columns into the lake column automatically **(under Stage 2 — routing on)**. TAI response (rising water table, suppressed aerobic decomposition, CH4) emerges via existing `w_scalar`, `o_scalar`, `finundated` pathways. No CTSM Fortran modifications from our fork planned — the PI's existing spillheight SourceMod handles the model-side behavior (currently inert: `spillheight = 0.0`).

**Lake column parameters (resolved 2026-04-25 via PI direction):**

| Field | Value | Source |
|---|---|---|
| `column_index` | 1 | Design decision (avoid wetlandisfull behavior change) |
| `downhill_column_index` | -9999 | Terminal |
| `hill_elev` | **-6.0 m** | Locked 2026-05-04. Chain-bookkeeping value below deepest land bin mean (-5.13 m). SPILLHEIGHT=0, no SourceMod shift. See audit doc Section 5.2.1 for full derivation. |
| `hill_distance` | 0.5 × deepest land bin's distance (computed dynamically) | Keeps col-col Darcy denominator positive (audit Section 1.1, Path A). With raw-HAND binning, Bin 1's DTND is small (~3 m on production); a static ~5 m would invert the gradient sign. |
| `hill_area` | Sum(water_mask × pixel_area) ≈ 10.68 km² | NWI mask |
| `hill_width` | 1/2 NWI total perimeter | PI direction; inert under current config |
| `hill_slope` | 0 | PI direction; "lake bottom" framing |
| `hill_aspect` | 0 | Inconsequential |
| `hill_bedrock_depth` | 0 | Inert under Uniform soil profile |

**Stage 1 tasks (complete):**
1. ✅ Phase E.5 (HAND binning fix) — 24-bin TAI-focused scheme, raw-HAND binning, Q01/Q99 trim
2. ✅ Compute lake perimeter from NWI shapefile (boundary-pixel approximation; ½ perimeter rescaled to per-rep)
3. ✅ Pipeline: add lake column at chain index 1, shift land columns up
4. ✅ NetCDF writer for `nmaxhillcol = N_BINS + 1` = 25
5. ✅ Production runs (2026-05-04, 2026-05-05) verify NetCDF structure
6. ✅ Per-rep rescale fixes lake column wtlunit from 98.7% → 12.3% (matches NWI water fraction)
7. ✅ CTSM ingestion verified via `osbs5.swenson.spinup` 100-yr run

**Stage 2 tasks (deferred — routing-on validation):**
1. Switch from single-point mode (`PTS_LAT/PTS_LON`) to explicit `LND_DOMAIN_MESH` so `grc%area` is a real number (not `spval`)
2. Set `use_hillslope_routing = .true.` in user_nl_clm
3. Sanity-check col-col Darcy fluxes (hydraulic conductivity, gradient signs)
4. Run a short test simulation; verify lake column water table responds to upland saturation (not just direct precipitation)
5. Long-form validation: lateral coupling drives TAI emergence (CH4 increase, finundated rise) in low-HAND columns
6. SPILLHEIGHT tuning if needed (currently 0.0 per Phase E.5 reframe; Lee 2023 measured 2.64m as alternative — revisit if model output suggests issues)

**Deliverable (Stage 1):** Hillslope NetCDF with submerged lake column + flood-zone bins (per Phase E.5), consumed by CTSM with PI's spillheight SourceMod (rendered inert via `spillheight = 0.0`). Validated in `osbs5.swenson.spinup` 100-yr run (column weights correct, per-column water/carbon behavior sensible).

**Deliverable (Stage 2):** Lateral-flow validation showing lake water table coupled to upland saturation, TAI emergence (water table rise → decomposition suppression → CH4 production) in low-HAND columns.

**Reference:** `docs/lake-column-ctsm-audit.md` — full audit including Sections 5 (lake column parameters) and 6.7 (HAND binning fix). `phases/G-ctsm-lake-representation.md` — phase plan and open questions.

### Dependency Diagram

```
Phase A (fix pysheds UTM) ─────── COMPLETE ─┐
                                             ├──> Phase D (rebuild pipeline) ── COMPLETE
Phase B (resolve resolution) ── COMPLETE ───┤
                                             │
Phase C (establish Lc) ──────── COMPLETE ───┘

Phase E (params) ──────────────── COMPLETE ─┐
Phase E.5 (bin redesign +                    │
            spillheight=0 +                  ├──> osbs5.swenson.spinup ── 100 yr DONE
            lake column) ───────  COMPLETE ──┤      validates BOTH
Phase E.6 (NWI mask hole-fill) ── COMPLETE ──┘      ↓               ↓
                                                Phase F          Phase G Stage 1
                                                (long spinup     (lake col + CTSM
                                                 to convergence)  ingestion)
                                                IN PROGRESS       COMPLETE
                                                (extend to 600 yr)

                                                Phase G Stage 2 (routing on)
                                                ── DEFERRED (needs mesh + tuning)

Phase E (complete params) ── COMPLETE ─────┐
                                            ├──> Phase F (validate & deploy) ──> Phase G (lake column)
Phase D (pipeline ready) ── COMPLETE ──────┘
```

Phases A-E complete. Phase F blocked by E (ready to start). Phase G blocked by F.

---

## PI Decisions (all resolved)

All questions raised during Phase C-E have been decided. See phase docs for the reasoning chains.

| # | Question | Resolved | Decision |
|---|----------|----------|----------|
| 1 | DEM conditioning (fill vs. preserve basins) | 2026-03-30 | Standard fill for D8. Pipeline characterizes macro-scale watershed, not microtopography. |
| 2 | Hillslope structure (bins × aspects) | 2026-04-14 | 1 aspect × 16 HAND bins, hybrid fixed+log (5 × 10cm + 10 log Q99 + sentinel). Previous 1×8 A2 scheme superseded. See `docs/hillslope-binning-rationale.md`. |
| 3 | Study boundary | 2026-03-30 | 90-tile production domain (R4-R12, C5-C14, 0 nodata). Largest contiguous rectangle; nodata at edges breaks pysheds flow routing. |
| 4 | Stream channel parameters | 2026-03-30 | Leave interim values. osbs2 uses routing off; Phase G will append a submerged lake column. |
| 5 | NEON slope/aspect vs. pgrid Horn 1981 | 2026-03-23 | Use NEON DP3.30025.001 directly. Slope r=0.91, aspect circ_r=0.84. |
| 6 | Lc interpretation at 1m resolution | 2026-02-23 | Lc = 356m with `min_wavelength = 20m` cutoff. Sub-20m features are resolve_flats artifacts; drainage-scale peak is stable across sensitivity sweeps. |

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
| Water masking comparison | `output/osbs/water_masking_comparison/` | Lc comparison: raw vs mean-fill vs zero-Laplacian (3 methods, spectral plots, JSON) |
| Lc comparison script | `scripts/osbs/compare_lc_water_masking.py` | FFT Lc comparison with and without lake masking |
| Water masking & lake representation | `docs/water-masking-and-lake-representation.md` | Historical: CTSM source investigation (still valid) + superseded weir overflow design (Phase G direction changed 2026-04-09) |
