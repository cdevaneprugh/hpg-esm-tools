# 260512 Cleanup Archive

Cleanup pass on 2026-05-12. Scripts moved here are one-off / done — they
ran to completion for a specific purpose and the resulting decision or
artifact is captured elsewhere in the project (phase docs, output dirs,
or current pipeline state). Kept for git history and reference; not
expected to be re-run from here.

## Contents

### `dem_processing.py` — Archived dead code

Extracted from `merit_regression.py` during Phase D for future use in
the OSBS pipeline (basin detection, open water identification). Never
actually invoked by either production script. `identify_open_water`
was superseded by the NWI water mask (Phase E). Zero importers across
the entire `swenson/` tree at archive time. Retained for git history
and possible future reference.

### `osbs/` — One-off scripts from `scripts/osbs/`

- **`stitch_mosaic.py`** — Built `data/mosaics/production/{dtm,slope,aspect}.tif`
  from 90 NEON tiles. Re-runnable if the NEON tile set changes.
- **`generate_water_mask.py`** — Built `data/mosaics/production/water_mask.tif`
  from NWI shapefile (with hole-fill post-fix from Phase E.6). Re-runnable
  if the NWI source changes.
- **`extract_subset.py`** — Ad-hoc OSBS DTM subset extraction with heatmap
  verification. Development utility.
- **`compare_lc_water_masking.{py,sh}`** — Phase E.5 FFT Lc comparison:
  raw DEM vs mean-fill vs zero-Laplacian masking strategies. Decision
  locked in `phases/E-complete-parameters.md`.
- **`compare_slope_aspect.{py,sh}`** — Phase E.5 NEON DP3.30025.001 vs
  pgrid Horn 1981 comparison across 90 production tiles. Decision: use
  NEON directly.

### `phase_b/` — Phase B scalability and resolution tests

Phase B is Complete. Test scripts retained for the validation record.

- **`test_scalability.py` / `.sh`** — pysheds `resolve_flats()` memory
  scaling tests (64GB / 128GB / 256GB on full 90M-pixel interior).
- **`test_resolution_comparison.py` / `.sh`** — Hillslope params at
  1m / 2m / 4m resolutions. Result: 1m is correct.

### `visualization/` — One-off KML reference artifacts

- **`export_kml.py`** — OSBS 1km×1km tile grid KML for Google Earth.
  Static reference; output at `output/google-earth/osbs_tile_grid.kml`.
- **`export_perimeter_kml.py`** — Production-domain perimeter overlay
  KML.

### `smoke_tests/` — Phase A UTM validation smoke test

Single-tile (R6C10) smoke test for the pysheds fork's UTM CRS code path.
Served its purpose during Phase A development (28 synthetic tests +
this real-tile smoke check confirmed the UTM code paths work). With
Phase A locked and the MERIT regression test as the ongoing validation
mechanism, this script is no longer routinely run.

- **`run_r6c10_utm.py`** — Single-tile UTM pipeline test
- **`run_r6c10_utm.sh`** — SLURM wrapper

Note: the original script imports `merit_validation.stage3_hillslope_params`,
which referred to the now-archived `audit/merit_validation_stages/`
location. Path is broken; reviving would require fixing the import to
the current `hillslope_params.py` (in `scripts/osbs/` or
`scripts/merit_validation/`).

## What's not archived here

Still in `scripts/`:
- `scripts/osbs/run_pipeline*.{py,sh}` — production pipeline (current)
- `scripts/osbs/{spatial_scale,hillslope_params}.py` — shared modules
  moved here on 2026-05-12 (was at `scripts/` root; see STATUS.md)
- `scripts/diagnostics/*` — diagnostic scripts moved out of `scripts/osbs/`
  but still useful for re-running on updated inputs
- `scripts/visualization/export_nwi_water_kml.py` — current diagnostic
- `scripts/merit_validation/*` — pysheds regression test (active);
  `spatial_scale.py` and `hillslope_params.py` now also live here as
  frozen copies

## Note on broken imports in archived scripts

After the 2026-05-12 shared-module de-coupling (`spatial_scale.py` and
`hillslope_params.py` moved from `scripts/` root into each pipeline's
own directory), any archived script that imports these modules via
the old parent-path pattern will fail with `ModuleNotFoundError`.
Affected archives include `compare_lc_water_masking.py` in `osbs/`
within this batch, plus older items like
`audit/merit_validation_stages/stage*.py`. These are reference-only;
not expected to run. If anyone needs to revive one, fix the import
path manually (the module names are unchanged, only the location is).
