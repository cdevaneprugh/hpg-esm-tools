# Documentation Update Plan: Swenson Implementation Pages

Date: 2026-03-17

## Background

The GitHub Pages documentation at `hpg-esm-docs` has 5 pages under the "Swenson Implementation" section. These pages were written during the initial audit (~Feb 2026), **before Phases A-D were completed**. They describe a pipeline with 4 blocking issues, 4x subsampling, EDT-based DTND, and a constrained Lc of 100m. Every page is now materially wrong about the current state of the system.

Since those pages were written:

- **Phase A (pysheds UTM):** Added CRS detection to pysheds fork. DTND, slope/aspect, AZND, and river network slope all have dual code paths (geographic: haversine, projected: Euclidean). Validated with synthetic V-valley DEM (23 analytical tests), MERIT geographic regression (6/6 params match baseline), and R6C10 UTM smoke test (14/14 checks pass). Mutation testing: 30 mutations, 100% effective score. 82 total tests, 0 failures.

- **Phase B (resolution):** Full 1m resolution works at 64GB (29.2 GB peak, 5.9 min for 90M pixels). Resolution comparison across 1m/2m/4m confirmed height/distance >0.999 correlation (resolution-insensitive), slope systematically underestimated at coarser resolution. No parameter improves with subsampling.

- **Phase C (Lc):** Characteristic length scale determined as ~300m (range 285-356m) via restricted-wavelength FFT. The naive 1m Laplacian FFT peak at 8.1m is a k-squared amplification artifact — the raw DEM spectrum is red noise with no peaked feature. Excluding wavelengths below 20m reveals the drainage-scale peak. Physical validation passes: P95 DTND/Lc = 1.17, mean catchment/Lc-squared = 0.876.

- **Phase D (pipeline rebuild):** Replaced EDT DTND with pysheds hydrological DTND. Replaced np.gradient slope/aspect with pgrid Horn 1981. Set 1m resolution, Lc=356m, A_thresh=63,362. Extracted shared `hillslope_params.py` module (481 lines). Created tiered SLURM wrappers (tier 1/2/3). Three audits completed: equation audit (7 equations verified), full pipeline audit (15 issues found, all resolved), line-by-line divergence audit (114 divergences cataloged, all justified).

- **MERIT validation improved:** Post-audit fixes (catchment-level aspect averaging, n_hillslopes indexing, basin mask correction, tail removal) improved area fraction from 0.82 to 0.92. Consolidated from 9 stage scripts to single `merit_regression.py`.

The pipeline now produces scientifically defensible hillslope parameters for the tier 3 production domain (R4C5-R12C14, 90 tiles, 90M pixels at 1m).

## Goals

1. **Primary:** Explain the capabilities, limitations, and use of the current codebase
2. **Secondary:** Serve as a record of the work done
3. **Tone:** Methods section of a scientific paper — these docs will serve as the basis for the actual paper's methods section

### What to Include

- Scientific decisions and their justifications (Lc, resolution, CRS handling, restricted-wavelength FFT)
- What changed from Swenson and why (CRS, resolution, Lc filtering)
- Validation evidence (MERIT regression, physical checks, synthetic tests)
- Current capabilities and limitations
- Key technical decisions with enough detail to reproduce

### What to Exclude

- Debugging narratives (4-attempt mosaic story, individual bug fix chronologies)
- Phase-by-phase chronology (organize by topic, not by when things happened)
- Code refactoring details (extracting modules, linting)
- SLURM configurations and runtime stats (except where they demonstrate feasibility)
- Individual code changes ("changed line X to Y")

### Grey Area (mention briefly, don't narrate)

- N/S aspect bug: "found and fixed during validation"
- Width calculation: "uses fitted trapezoidal areas, not raw pixel counts"
- The fact that issues existed and were resolved systematically (credibility)
- One-sentence context for decisions that emerged from problems (e.g., "Full 1m resolution is used. Earlier work subsampled to 4m; Phase B testing showed this was unnecessary and degraded slope accuracy.")

## Execution Plan

Work in two batches. Batch 1: the three pages that are updates to existing content. Review with user. Batch 2: the two pages that are major rewrites.

### Batch 1: Updates to Existing Content

#### Page 1: Overview (index.md)

**File:** `$DOCS/docs/swenson/index.md` (currently 81 lines)
**Change type:** Rewrite

Current page describes a blocked pipeline with 4 known issues and pending phases. Needs to reflect: Phases A-D complete, pipeline produces defensible output, Phases E-F remaining.

**Structure:**

1. Goal (keep — the motivation paragraph is good)
2. Key Differences from Swenson — update table:
   - Remove "4x subsampling for flow routing" → "Full 1m resolution"
   - Remove "Euclidean (UTM meters)" DTND → "Hydrological (pysheds, UTM-aware)"
   - Add "Lc determination" row: "FFT natural peak" vs "Restricted-wavelength FFT (min 20m)"
3. Guiding Principle (keep as-is)
4. Implementation Arc — rewrite to reflect completed phases:
   - Phase 1 (Port): Complete, link to pysheds page
   - Phase 2 (Validate): Complete, link to MERIT validation page
   - Phase 3 (Adapt): Phases A-C complete, link to methodology page
   - Phase 4 (Execute): Phase D complete, link to methodology page
   - Phase 5 (Parameters): Phase E pending
   - Phase 6 (Deploy): Phase F pending
5. **NEW: Development History** — timeline table showing phases, dates, key decisions, linking to internal phase files for anyone who wants the full story:

   | Date | Phase | Decision | Internal Reference |
   |------|-------|----------|--------------------|
   | 2025-12 | — | Ported pysheds, created test suite | phases/A-pysheds-utm.md |
   | 2026-01 | — | 9-stage MERIT validation (5/6 params >0.95) | merit-validation.md |
   | 2026-02 | A | UTM CRS support in pysheds fork | phases/A-pysheds-utm.md |
   | 2026-02 | B | Full 1m resolution (no subsampling needed) | phases/B-flow-resolution.md |
   | 2026-02 | C | Lc = 300m via restricted-wavelength FFT | phases/C-characteristic-length.md |
   | 2026-02 | — | MERIT post-audit: area fraction 0.82 → 0.92 | audit/250223-*/ |
   | 2026-03 | D | Pipeline rebuild, equation audit, divergence audit | phases/D-rebuild-pipeline.md |

6. Current Status — production pipeline works, Phase E/F remaining, what's left
7. Tools and Repositories (keep, minor updates)
8. Cross-References (keep, update links)

#### Page 2: pysheds Fork (pysheds-porting.md → keep filename)

**File:** `$DOCS/docs/swenson/pysheds-porting.md` (currently 130 lines)
**Change type:** Expand with Phase A content

Current page covers: fork setup, what Swenson added, NumPy 2.0 fixes, test suite (14 passed).

**Add after existing content:**

1. **UTM CRS Support (Phase A)** section:
   - Problem: haversine math fails on UTM coordinates
   - Solution: `_crs_is_geographic()` detection, dual code paths
   - What was modified: `compute_hand()` (DTND), `_gradient_horn_1981()` (slope/aspect), AZND, river network slope
   - Validation approach (three-level): synthetic V-valley, MERIT regression, R6C10 smoke test

2. **Additional fixes** subsection:
   - Deprecation fixes: distutils, np.in1d, pd._append (5 fixes)
   - Code improvements: `_propagate_uphill()` extraction, module-level constants, bare except removal
   - Variable renaming for CRS-neutral naming

3. **Updated test suite** section — replace the old "14 passed" with:
   - 82 total tests across 4 files
   - 23 UTM-specific analytical tests (synthetic V-valley with known solutions)
   - Mutation testing: 30 mutations, 100% effective score
   - 0 failures, 0 warnings

4. **Update API Notes** — note that `slope_aspect()` now handles both CRS types transparently

#### Page 3: MERIT Validation (merit-validation.md)

**File:** `$DOCS/docs/swenson/merit-validation.md` (currently 244 lines)
**Change type:** Update with post-audit improvements

Keep the detailed 9-stage walkthrough (user explicitly requested this). Add new sections at the end.

1. **Update Final Results table** — current page shows:

   | Parameter | Correlation |
   |-----------|-------------|
   | Height | 0.9999 |
   | Distance | 0.9982 |
   | Slope | 0.9966 |
   | Aspect | 0.9999 |
   | Width | 0.9597 |
   | Area | 0.8200 |

   Replace with current expected values from `merit_regression.py`:

   | Parameter | Correlation |
   |-----------|-------------|
   | Height | 0.9979 |
   | Distance | 0.9992 |
   | Slope | 0.9839 |
   | Aspect | 1.0000 |
   | Width | 0.9919 |
   | Area | 0.9244 |

   Note: these are the expected baseline values. The actual regression run from 2026-02-24 shows Height=0.9981, Distance=0.9991, Slope=0.9866, Aspect=1.0000, Width=0.9916, Area=0.9224 — all within tolerance.

2. **NEW section: Post-Audit Improvements (February 2026)** — after stage 9, add:

   Explain that a comprehensive pipeline audit identified additional alignment opportunities. Show the area fraction progression table:

   | Fix | Area fraction | Delta | Mechanism |
   |-----|--------------|-------|-----------|
   | Stage 8 baseline | 0.82 | — | N/S aspect fix (pgrid slope_aspect) |
   | Basin mask correction | 0.83 | +0.01 | Binary identify_open_water mask, not DEM difference |
   | Catchment-level aspect averaging | 0.90 | +0.08 | Per-catchment circular mean before binning |
   | n_hillslopes indexing fix | 0.91 | +0.01 | Extract drainage_id to gridcell before indexing |
   | DTND tail removal | 0.92 | +0.01 | Exponential fit to remove long-DTND outliers |

   Key finding: catchment-level aspect averaging was the single largest improvement (+0.08). Swenson replaces per-pixel aspects with catchment-side circular means before binning, preventing pixels near aspect boundaries from landing in wrong bins.

3. **NEW section: Consolidated Regression Test** — explain that the 9 stage scripts were consolidated into `merit_regression.py` (single-file regression) that runs as a SLURM job and serves as an automated regression test for the pysheds fork. Pass criteria: all 6 parameter correlations within 0.01 of expected, Lc within 5% of 763m.

4. **Update Area Discrepancy Analysis** — the current analysis attributes the gap to region/gridcell mismatch and bin computation. Update to reflect the 15-hypothesis investigation (area_fraction_research.md): dominant remaining source is stream network delineation differences between pysheds versions. The 0.92 correlation is the ceiling for this validation approach.

5. **Update Summary of Bugs** table — add the post-audit fixes (basin mask, catchment aspect averaging, n_hillslopes, polynomial weighting).

### Batch 2: Major Rewrites

#### Page 4: Methodology (replaces osbs-implementation.md)

**File:** `$DOCS/docs/swenson/osbs-implementation.md` (currently 297 lines)
**Change type:** Complete rewrite
**Rename in nav:** "OSBS Implementation" → "Methodology" (update mkdocs.yml)

Current page is a debugging narrative. Replace with a methods description organized by topic.

**Structure:**

1. **Study Domain and Data**
   - NEON LIDAR properties (233 tiles, 1m, EPSG:32617, 2023-05 collection)
   - Production domain: R4C5-R12C14 (90 tiles, 9x10 km, 0% nodata) — largest contiguous rectangle
   - Tile reference system (R#C# format)
   - DEM properties: 23-69m elevation, ~46m total relief
   - Keep the tile grid image (tile_grid_reference.png)

2. **Characteristic Length Scale (Lc)**
   - Purpose: controls accumulation threshold, stream network density, all downstream parameters
   - The k-squared artifact problem: at 1m, Laplacian FFT peak appears at 8m. This is NOT the drainage scale — the raw DEM spectrum is red noise with no natural peak. The Laplacian operator's k-squared weighting amplifies short-wavelength features.
   - Solution: restricted-wavelength FFT excluding wavelengths below 20m. The 20m cutoff separates micro-topography (tree-throw mounds, animal burrows, shallow rills) from organized drainage structure. This mirrors what 90m MERIT data does implicitly by averaging.
   - Result: Lc = 300m (range 285-356m), A_thresh = 45,000 m-squared
   - Physical validation: P95 DTND/Lc = 1.17 (Swenson calibration: "similar magnitude"), mean catchment/Lc-squared = 0.876 (Swenson calibration: 0.94). Both checks pass.
   - Sensitivity: Lc is insensitive to FFT parameters (blend_edges, zero_edges, NLAMBDA, MAX_HILLSLOPE_LENGTH, detrend). 20 configurations tested, Lc stable across all.
   - Reference output: `output/osbs/phase_c/`

3. **Stream Network Delineation**
   - Accumulation threshold from Lc: A_thresh = 0.5 * Lc-squared = 63,362 pixels (at 1m)
   - Connected-component extraction isolates largest contiguous data region
   - Edge trimming required for irregular NEON tile coverage (pysheds silently fails when all boundary cells are nodata)
   - DEM conditioning: fill_pits → fill_depressions → resolve_flats → flow direction → accumulation
   - Stream coverage: 0.23% of domain (207,832 cells)
   - Note: DEM conditioning fills all depressions, which erases real features (sinkholes, wetland depressions) — this is a known limitation, standard for D8 routing

4. **Hillslope Parameter Computation**
   - HAND and DTND from pysheds compute_hand() (hydrological DTND, not geographic EDT)
   - Slope and aspect from pgrid Horn 1981 stencil (computed from original DEM, not pysheds-conditioned)
   - Catchment-level aspect averaging: per-pixel aspects replaced with catchment-side circular mean before binning
   - Aspect binning: 4 bins (N: 315-45, E: 45-135, S: 135-225, W: 225-315)
   - HAND binning: 4 elevation bins with mandatory 2m upper bound on lowest bin
   - Trapezoidal width fitting: cumulative area A_sum(d), fit w(d) = w_base + 2*alpha*d
   - Width from quadratic solver on fitted trapezoidal areas (not raw pixel counts)
   - DTND tail removal: exponential fit to DTND distribution, applied before basin masking

5. **Processing Resolution**
   - Full 1m resolution, no subsampling
   - One-sentence context: "Earlier work subsampled to 4m after an OOM failure. Phase B testing demonstrated that the OOM occurred only on nodata-contaminated mosaics, and that 90M pixels at 1m complete in 6 minutes at 29 GB peak memory."
   - Resolution comparison results: height/distance >0.999 correlation across 1m/2m/4m (resolution-insensitive). Slope systematically underestimated at coarser resolution. No parameter improves with subsampling.

6. **Adaptations from Swenson**

   Table with 5 categories:

   | Category | What | Why |
   |----------|------|-----|
   | **Kept identical** | Trapezoidal width model, HAND binning algorithm, quadratic solver, 2m lowest-bin constraint, circular mean aspect, DEM conditioning chain | Core methodology validated against published data |
   | **CRS adaptation** | Euclidean DTND (vs haversine), Horn 1981 with uniform pixel spacing (vs haversine spacing), planar AZND (vs spherical bearing) | UTM coordinates require Euclidean math; haversine on meters produces garbage |
   | **Resolution adaptation** | Restricted-wavelength FFT (min 20m), blend_edges=50px (vs 4px), full 1m processing | 1m data contains micro-topographic noise invisible at 90m; larger edge windows needed for UTM pixel sizes |
   | **Improved during validation** | Catchment-level aspect averaging, basin mask as binary (not DEM difference), n_hillslopes indexing | Alignment with Swenson's methodology discovered through systematic audit |
   | **Known limitations** | Stream depth/width (placeholders), bedrock depth (set to 0), depression filling (erases real features) | Phase E — requires field data, PI decisions, or empirical relationships |

7. **Output Format**
   - NetCDF structure (dimensions, variables) — keep from current page but update
   - Production output: `hillslopes_osbs_tier3_contiguous_c260317.nc`
   - 16 columns (4 aspects x 4 HAND bins), verified against Swenson reference structure

8. **Current Limitations and Remaining Work**
   - Stream channel parameters are placeholders (depth: 0.14m, width: 1.7m, slope: 0.005 — interim power law)
   - Bedrock depth set to 0 (no-op under CTSM's Uniform soil profile method used by osbs2)
   - DEM conditioning erases real closed basins
   - Hillslope structure may change: 4 aspects x 4 bins (current) vs 1 aspect x 8 bins (under consideration)
   - No CTSM simulation comparison yet (Phase F)

#### Page 5: Dataset Comparison (dataset-comparison.md)

**File:** `$DOCS/docs/swenson/dataset-comparison.md` (currently 140 lines)
**Change type:** Major update with current production data

Current page compares Swenson global data against two pre-fix datasets (Interior 150-tile, Trimmed 39-tile). Both pre-fix datasets used EDT DTND, 4x subsampling, constrained Lc=100m, and had the N/S aspect bug. Replace with current production data.

**Structure:**

1. **Datasets** — update table:

   | Dataset | Source DEM | Resolution | Coverage | Lc | A_thresh |
   |---------|-----------|------------|----------|-----|---------|
   | Swenson (Global) | MERIT | ~90m | Full OSBS gridcell | 763m | 275,400 m-sq |
   | OSBS (Current) | NEON LIDAR | 1m | R4C5-R12C14 (90 tiles, 90 km-sq) | 356m | 63,362 m-sq |

2. **Parameter Comparison** — use current production data (from hillslope_params.json):

   Values averaged across all 4 aspects:

   | Parameter | Swenson (Global) | OSBS 1m (Current) |
   |-----------|------------------|-------------------|
   | Elevation range (m) | 0.17 - 8.14 | 0.0001 - 9.3 |
   | Distance range (m) | 67 - 541 | 34 - 367 |
   | Mean slope (m/m) | 0.011 | 0.052 |
   | Mean width (m) | 539 | 153 |
   | Total area (km-sq) | 1.19 | ~0.24 per aspect |
   | Aspect distribution | ~25% each | N: 25.3%, E: 25.2%, S: 25.8%, W: 23.6% |

   Source for Swenson numbers: `hillslopes_osbs_c240416.nc` at `/blue/gerber/sgerber/CTSM/subset_input/`

3. **Key Observations** — update analysis:
   - Distance scales differ by ~2-10x (vs old 5-25x claim) — closer now that DTND is hydrological, not EDT
   - Slopes ~5x higher (vs old 3.5x) — full 1m resolution captures more local variability
   - Width scaling follows distance proportionally
   - Elevation ranges comparable (~8-9m) — expected for same geographic area
   - Aspect distribution now symmetric (~25% each) — N/S aspect bug fix resolved the E/W bias from old runs
   - HAND bin 1 near-zero: lowest bin has effectively zero height (Q25 = 0.00027m) due to equal-area binning on flat terrain — this is correct behavior, not a bug

4. **Implications for CTSM Simulations** — keep and update
5. **Remove "Known issues" warning** — the current data is from the fixed pipeline

6. **Images** — the existing comparison images (elevation/width profiles, column areas) are from pre-fix data. Note: new images should be generated from current production output. The old pre-fix images (interior_*, trimmed_*) should be removed or moved to an archive section. The swenson_* images can stay.

   Images to generate:
   - `current_elevation_width.png` — elevation/width profiles from tier 3 production data
   - `current_col_areas.png` — column area distribution from tier 3 production data

   These can be generated using existing scripts in `scripts/hillslope.analysis/` or `scripts/visualization/`.

### Navigation Update

Update `$DOCS/mkdocs.yml` nav:

```yaml
- Swenson Implementation:
    - Overview: swenson/index.md
    - pysheds Fork: swenson/pysheds-porting.md
    - MERIT Validation: swenson/merit-validation.md
    - Methodology: swenson/osbs-implementation.md
    - Dataset Comparison: swenson/dataset-comparison.md
```

Change display name "OSBS Implementation" → "Methodology" in the nav. Keep the filename `osbs-implementation.md` to avoid breaking any external links.

### Cross-Reference Fix

The `research/hillslope.md` page (line 24) says "17 columns" including a stream column. This is incorrect — the stream channel is a landunit-level boundary condition, not a column. The correct number is 16 columns. Fix this while updating the Swenson pages.

## Key Data Sources for Writing

### Current production output
- Parameters: `output/osbs/2026-03-17_tier3_contiguous/hillslope_params.json`
- Summary: `output/osbs/2026-03-17_tier3_contiguous/tier3_contiguous_summary.txt`
- NetCDF: `output/osbs/2026-03-17_tier3_contiguous/hillslopes_osbs_tier3_contiguous_c260317.nc`

### MERIT regression baseline
- Summary: `scripts/merit_validation/output/summary.txt`
- Expected correlations: Height 0.9979, Distance 0.9992, Slope 0.9839, Aspect 1.0000, Width 0.9919, Area 0.9244

### Swenson reference data
- Global hillslope file: `/blue/gerber/sgerber/CTSM/subset_input/hillslopes_osbs_c240416.nc`

### Internal documentation (for context, not for copying)
- Phase files: `phases/A-pysheds-utm.md` through `phases/F-validate-deploy.md`
- STATUS.md: Living project status
- Audit docs:
  - `audit/250223-pysheds_and_merit_pipeline_audit/` — Feb 2026 pysheds/MERIT audit
  - `docs/osbs-pipeline-audit-260310.md` — March 10 equation audit
  - `docs/osbs-pipeline-divergence-audit-260316.md` — March 16 line-by-line divergence audit
- Area fraction research: `audit/250223-*/area_fraction_research.md`

## Verification

After writing all pages:

1. Build the docs locally: `cd $DOCS && mkdocs serve` (or just `mkdocs build` to check for errors)
2. Verify all internal links resolve
3. Check that no "blocked" or "known issues affect" warnings remain on pages where they no longer apply
4. Verify the MERIT validation final numbers match `scripts/merit_validation/output/summary.txt`
5. Verify production data numbers match `output/osbs/2026-03-17_tier3_contiguous/hillslope_params.json`
