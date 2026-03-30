# Phase E: Complete the Parameter Set

Status: In progress
Depends on: None (can start independently)
Blocks: Phase F

## Problem

Remaining parameters need refinement or PI decisions (STATUS.md #5, #6):

**Stream channel parameters (#6) — resolved (2026-03-30):**
Leave interim power-law values as-is. osbs2 runs with `use_hillslope_routing = .false.` — stream params are never read by CTSM. Phase G will repurpose stream fields as lake geometry for weir overflow.

**Bedrock depth (#7) — resolved:** Now uses zeros, matching Swenson reference (commit 11f465e).

**DEM conditioning (#5) — resolved (2026-03-30):** Proceed with standard depression filling for D8. Pipeline characterizes macro-scale watershed structure, not microtopography. Limitations understood and accepted.

## Tasks

- [x] Hillslope structure decision: 1 aspect x 8 bins (see `docs/hillslope-binning-rationale.md`). Interim: equal-area bins until water masking is implemented. Log spacing deferred — lowest bins contaminated by unmasked lake pixels (Q1 HAND ~ 4e-6 m).
- [x] Stream slope from actual network (implemented in run_pipeline.py line 1415)
- [x] Bedrock depth: zeros matching Swenson reference (commit 11f465e)
- [x] NEON slope/aspect decision: use NEON DP3.30025.001 products directly (PI approved 2026-03-23). Comparison: slope r=0.91, aspect circ_r=0.84. See `output/osbs/slope_aspect_comparison/`.
- [x] Implement NEON slope/aspect ingestion in pipeline (commit 73c09fe):
  - [x] Create one-time slope/aspect mosaics for production domain (`data/mosaics/production/`)
  - [x] Add mosaic loading to pipeline (Step 1, lines 607-619), convert slope degrees to m/m
  - [x] Remove `grid.slope_aspect("dem")` call — replaced with NEON mosaic loading
  - [x] Verify `identify_open_water(slope)` works with NEON slope (still 0 detections at 1m — unchanged)
  - [x] Smoke test (R6C10): slope ~0.008 m/m lower (NEON smoother), height/distance/width/area identical
  - [x] Production run: 23.4 min, all parameters correct, slope differences consistent with pre-smoothing
- [x] NWI water mask: download, filter, rasterize, visualize (2026-03-24)
- [x] Integrate water mask into pipeline (2026-03-27, commit 8c727ca). Dual-mask approach: natural streams for catchment delineation, wide mask (streams + NWI lakes) for HAND. Water pixels excluded from HAND binning and DTND tail fitting.
- [ ] Re-evaluate 1x8 log-spaced HAND bins with water masking in place (confirmed target, 2026-03-30)
- [x] Stream depth/width: leave interim power-law as-is (2026-03-30). osbs2 uses `use_hillslope_routing = .false.` — params never read. Phase G repurposes as lake geometry.
- [x] PI consultation — all questions resolved (2026-03-30):
  - DEM conditioning: standard fill for D8, pipeline characterizes macro-scale watershed
  - Study boundary: production domain (90 tiles, 0 nodata) is final
  - Stream params: leave as-is, Phase G repurposes
  - Hillslope structure: 1x8 log-spaced confirmed as target

## Deliverable

Complete set of physically motivated parameters with documented sources. PI decisions recorded on the open science questions.

## Log

### 2026-03-19: Hillslope structure decision

Switched pipeline from 4 aspects x 4 equal-area bins (16 columns) to 1 aspect x 8 log-spaced
HAND bins (8 columns). Log spacing concentrates resolution near the stream where TAI dynamics
dominate. 8 bins chosen over 16 to stay above DEM conditioning noise floor (~cm scale) while
giving 4-5 bins in the 0-2m TAI zone.

Tier 1 (R6C10) test run completed. All 8 bins populated, heights and distances monotonically
increasing. Total area matches 4x4 (0.373 vs 0.374 km^2). Comparison plot generated at
`output/plots/4x4_vs_1x8_r6c10.png`.

Full justification: `docs/hillslope-binning-rationale.md`.

### 2026-03-23: NEON slope/aspect adopted

Replaced pgrid Horn 1981 slope/aspect computation with NEON DP3.30025.001 products.
`stitch_mosaic.py` creates production mosaics in `data/mosaics/production/` (DTM, slope,
aspect for R4C5-R12C14). Pipeline loads all three in Step 1, converts NEON slope from
degrees to m/m via `tan(deg2rad())`.

Verified on smoke test (R6C10) and production (90 tiles, 23.4 min). Slope values are
~0.008 m/m lower (NEON's 3x3 pre-filter reduces noise). Height, distance, width, area
are bit-for-bit identical to pgrid run — only slope/aspect changed. Scientific rationale
documented in STATUS.md.

Also refactored pipeline: removed mosaic creation bloat (-242 lines), dead code cleanup
(-33 lines), renamed tier3 to production. Pipeline is now 1261 lines (down from 1536).

### 2026-03-24: Switch to 1x4 equal-area bins (interim)

Switched from 1x8 log-spaced bins to 1x4 equal-area bins (Swenson's `compute_hand_bins()`).
Log-spaced bins are deferred until water masking addresses lake pixel contamination in the
lowest HAND bins (Q1 ~ 4e-6 m from resolve_flats micro-gradients on flat water surfaces).

The 4 equal-area bins provide a direct comparison baseline against the existing 4x4 tier 3
runs. HAND boundaries should be identical; height/distance/width/area should match the
aspect-averaged 4x4 values. Slope/aspect will differ (NEON vs pgrid source).

`compute_hand_bins_log()` retained in hillslope_params.py for future use.

### 2026-03-24: NWI water mask created

Downloaded NWI data for Lower St. Johns Watershed (HU8_03080103, 57,213 features).
Filtered to open water: Lacustrine (`L*`) and Palustrine Unconsolidated Bottom (`PUB*`)
— 103 features in production domain. Excludes forested/emergent/shrub wetlands per PI
preference ("just the actual open water").

Rasterized onto the production DTM grid (`data/mosaics/production/water_mask.tif`):
10.7M water pixels (11.9% of domain), 1068 ha. Verified alignment with DEM hillshade
and Google Earth (KML export). Fixed two coordinate issues found during development:
1. Perimeter KML grid origin off by one row (TILE_GRID_ORIGIN_NORTHING 3292000 -> 3293000)
2. NWI KML clip must happen in UTM, not WGS84 (UTM rectangle != WGS84 rectangle, up to 86m
   boundary mismatch at NW corner)

New scripts:
- `scripts/osbs/generate_water_mask.py` — one-time rasterization
- `scripts/osbs/overlay_nwi_water.py` — hillshade + mask overlay visualization
- `scripts/visualization/export_nwi_water_kml.py` — KML export for Google Earth

Next: integrate mask into pipeline, then re-evaluate log-spaced bins.

### 2026-03-25: Water mask integration — boundary forcing (BROKEN)

First integration attempt. Replaced `identify_open_water()` (slope-based, 0 detections at 1m)
with NWI raster mask. Used Swenson's boundary-forcing approach: inject lake boundaries into the
stream network so lakes drain to their edges. This BROKE catchment delineation — fragmented from
~800 to 222K catchments at 1m resolution. Widths collapsed to ~1m, area to ~86 m²/element.
Runtime bloated from 20 to 44 min.

**Why it failed:** At 1m, lake boundaries are thousands of pixels long. Injecting them into the
stream network creates thousands of "reaches" that fragment every surrounding catchment into
single-pixel-wide strips.

Run stats: 220,575 hillslopes, 222,046 reaches, max_acc 3.0M, HAND bins [0, 0.26, 2.31, 5.94,
25.1], widths ~1m, areas ~86 m²/element, stream depth=0.010/width=0.0/slope=0.01343, 44.0 min.
See `logs/production_27948388.log`.

### 2026-03-26: Lc water masking comparison

Tested whether lake pixels contaminate the FFT-derived Lc (`compare_lc_water_masking.py`).
Three methods compared:

| Method | Lc (m) | Model | psharp |
|--------|--------|-------|--------|
| Raw DEM (current) | 356 | lognormal | 3.95 |
| Mean-fill lakes | 20 | linear | 0 |
| Zero Laplacian at lakes | 20,000 | linear | 0 |

**Conclusion:** Raw DEM is the correct FFT input. Lake surfaces contribute near-zero Laplacian
(flat water = zero second derivative) and do not contaminate the drainage-scale spectral peak.
Both masking approaches destroy the peak (psharp drops to 0) by introducing artificial spectral
energy at lake boundaries. Results in `output/osbs/water_masking_comparison/`.

### 2026-03-27: Dual-mask fix (commit 8c727ca)

Replaced boundary forcing with dual-mask strategy:

1. **Natural stream mask** for catchment delineation — preserves natural catchment structure
   (~508 hillslopes vs 222K with boundary forcing)
2. **Wide mask** (natural streams + lake pixels) for HAND computation — land near lakes gets
   HAND relative to lake surface; lake pixels get HAND=0
3. **Water pixel exclusion** from HAND binning and DTND tail fitting

Production run verified: HAND bins clean (lowest boundary 0.27m vs 0.00027m without masking),
widths 339-569m, areas ~37,700 m²/element, 16.4 min. See `logs/production_28119563.log`.

Run stats: 508 hillslopes, 552 reaches, max_acc 3.0M, HAND bins [0, 0.27, 2.30, 5.94, 25.2],
widths 569/504/429/339m, areas ~37,700 m²/element, stream depth=0.118/width=1.3/slope=0.00691,
16.4 min.

### 2026-03-30: PI decisions — all open questions resolved

Four remaining PI questions resolved in one session:

1. **DEM conditioning:** Standard fill for D8. Pipeline characterizes macro-scale watershed — not
   designed for microtopography. Limitations accepted.
2. **Study boundary:** 90-tile production domain (R4-R12, C5-C14) is final. Nodata pixels break
   pysheds flow routing (edges need valid data for flow exit). Expanding not worth the complexity.
3. **Stream channel parameters:** Leave interim power-law as-is. Key finding: osbs2 runs with
   `use_hillslope_routing = .false.` — stream params are never read by CTSM. Phase F validation
   matches this (routing off). Phase G will repurpose stream fields as lake geometry for weir
   overflow. No OSBS-specific stream research needed.
4. **Hillslope structure:** 1x8 log-spaced HAND bins confirmed as target. Ready to re-evaluate
   now that water masking provides clean HAND values.

Only remaining Phase E work: log-spaced bin re-evaluation.
