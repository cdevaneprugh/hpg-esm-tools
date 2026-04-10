# State of the Union: Swenson Hillslope Implementation for OSBS

Date: 2026-03-30

## Executive Summary

We are implementing Swenson & Lawrence (2025) representative hillslope methods to generate custom hillslope parameters for OSBS using 1m NEON LIDAR data. The goal is to replace the current global 90m MERIT-derived hillslope file (`hillslopes_osbs_c240416.nc`) with site-specific parameters that capture the fine-scale drainage structure of this low-relief wetlandscape.

**The methodology is validated and the pipeline produces scientifically defensible output.** MERIT validation achieved >0.95 correlation with Swenson's published data on 5 of 6 parameters. Phases A-D are complete: pysheds handles UTM CRS, flow routing runs at full 1m resolution, Lc is established (356m for the production domain), and the pipeline has been rebuilt with all known fixes and verified by equation-by-equation audit. Output is CTSM-compatible NetCDF matching the Swenson reference structure.

**Phase E (parameter completion) is complete.** Hillslope structure is 1 aspect x 8 HAND bins (log-spaced with 0.1m noise floor, Q95 upper endpoint). NWI water masking is implemented (dual-mask approach: natural streams for catchments, wide mask for HAND). NEON slope/aspect products adopted. All PI questions resolved (2026-03-30). Phase F (CTSM validation) and Phase G (submerged lake column in hillslope file, no CTSM Fortran changes planned) follow.

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

**Production outcome:** The pipeline produces a CTSM-compatible NetCDF at `output/osbs/2026-04-09_production/hillslopes_osbs_production_c260409.nc` with 8 HAND bins (1 aspect x 8, log-spaced with 0.1m noise floor + Q95 upper endpoint), NWI water masking applied (dual-mask approach), NEON slope/aspect products, Lc = 356m on the 90-tile production domain.

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

All parameters finalized. 1 aspect x 8 HAND bins, log-spaced with 0.1m noise floor + Q95 upper endpoint (Strategy A2, tested against 5 alternatives). NEON DP3.30025.001 slope/aspect products adopted directly. NWI water masking via dual-mask approach. Stream parameters left as interim power-law values (osbs2 uses `use_hillslope_routing = .false.`, params never read). All PI questions resolved. See `phases/E-complete-parameters.md` and `docs/hillslope-binning-rationale.md`.

### Phase F: Validate and deploy

**Status: Not started.** Blocked by Phase E (log-spaced bin re-evaluation).

**Key context:** osbs2 runs with `use_hillslope_routing = .false.` and `PCT_LAKE = 0` (no lake land unit). Phase F validation matches this configuration — routing off, stream params irrelevant, no lake land unit. The only variable changed is the hillslope file. This establishes a clean baseline comparison before Phase G enables routing with lake-modified parameters.

**Tasks:**
1. Compare custom hillslope file to Swenson reference (`hillslopes_osbs_c240416.nc`)
2. Physical plausibility checks (elevation, aspect distribution, stream network vs known hydrology)
3. Create CTSM test branch from osbs2 at year 861, `use_hillslope_routing = .false.`, `PCT_LAKE = 0`
4. Run short simulation (1-5 years) with custom hillslope file as the only change
5. Compare outputs to baseline (water table, soil moisture, carbon fluxes)

**Deliverable:** Validated hillslope file ready for production runs.

### Phase G: Submerged lake column in hillslope file

**Status: Not started.** Blocked by Phase F (need validated baseline for comparison). **Direction changed 2026-04-09** after PI consultation with collaborators — previous weir overflow plan abandoned.

Add one extra column to the hillslope NetCDF representing the aggregate of all NWI-masked lake area as a single submerged column with negative `hill_elev`. CTSM's existing lateral flow machinery draws water from adjacent upland columns into the lake column automatically. TAI response (rising water table, suppressed aerobic decomposition, CH4) emerges via existing `w_scalar`, `o_scalar`, `finundated` pathways. No CTSM Fortran modifications from our fork planned — the PI's existing spillheight SourceMod handles the model-side behavior.

**Lake column parameters:**

| Field | Value |
|---|---|
| `hill_elev` | `-SPILLHEIGHT` (from PI's SourceMod scalar) |
| `hill_distance` | `mean(dtnd[water_mask])` |
| `hill_area` | Total NWI lake area (sum of water mask × pixel area) |
| `hill_width`, `hill_slope`, `hill_aspect` | TBD (defaults or PI convention) |

**Tasks:**
1. Pipeline: add lake column computation step after Step 5, append to NetCDF output
2. Update NetCDF writer for `nmaxhillcol = N_HAND_BINS + 1`
3. Revisit HAND binning strategy (relaxed criteria under "standing water is a feature" framing)
4. PI clarifications: SPILLHEIGHT value, `downhill_column_index` topology, `hillslope_index` convention, slope/aspect/width defaults
5. Production run, verify NetCDF structure
6. CTSM test branch from osbs2 with modified hillslope file + PI's spillheight SourceMod, compare vs Phase F baseline

**Deliverable:** Hillslope NetCDF with submerged lake column, consumed by CTSM with the PI's spillheight SourceMod. Comparison showing effect on water table dynamics and CH4 production in near-lake columns.

**Reference:** `phases/G-ctsm-lake-representation.md` — full plan, open questions, rationale. `docs/water-masking-and-lake-representation.md` — historical CTSM source investigation (still valid), superseded weir overflow design.

### Dependency Diagram

```
Phase A (fix pysheds UTM) ─────── COMPLETE ─┐
                                             ├──> Phase D (rebuild pipeline) ── COMPLETE
Phase B (resolve resolution) ── COMPLETE ───┤
                                             │
Phase C (establish Lc) ──────── COMPLETE ───┘

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
| 2 | Hillslope structure (bins × aspects) | 2026-04-09 | 1 aspect × 8 HAND bins, Strategy A2 (0.1m floor + Q95 log). See `docs/hillslope-binning-rationale.md`. |
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
