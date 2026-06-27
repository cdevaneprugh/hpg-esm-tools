# 260627 Cleanup Archive

Cleanup pass on 2026-06-27 driven by a full-repo audit (results captured in
the chat session that produced this batch). Items moved here are stale,
broken, redundant, or one-off — they are no longer expected to be read or
re-run from their previous locations. Pattern follows
`audit/260512-cleanup/README.md`.

## Contents

### `diagnostics/` — Stale or broken diagnostic scripts from `scripts/diagnostics/`

- **`compare_hillslope_configs.py`** — Generic per-config profile comparison.
  Reads `meta['n_hand_bins']` (line 60); the current pipeline (Phase E.5+)
  writes `meta['n_land_bins']`. Crashes with `KeyError` on any production
  JSON dated 2026-05-05 or later. Use cases overlapped with the now-also-
  archived `plot_hillslope_comparison.py`.
- **`plot_hillslope_comparison.py`** — OSBS-vs-Swenson MERIT plotter,
  hard-coded for the obsolete 1×8 bin scheme. `DEFAULT_INPUT` points to
  `output/osbs/2026-04-09_production/` (a Phase E artifact, not the current
  Phase E.5 production at `2026-05-05_production/`). Reviving would require
  refactoring for the 24-bin TAI scheme + lake column.

### `osbs/` — One-off scripts

- **`run_pipeline.sh`** — Older sibling of `run_pipeline_production.sh`.
  Identical pipeline invocation; only differences were a banner format and
  an unused `TILE_RANGES=R4C5-R12C14` env var (the pipeline never read it).
  Consolidated to `run_pipeline_production.sh` as the single canonical
  production SLURM wrapper.
- **`generate_plots.py`** / **`generate_categorical_map.py`** /
  **`generate_hypothetical_setups.py`** — One-off Phase E.5 design-decision
  diagnostics from 2026-04-25, reading saved arrays at
  `output/osbs/2026-04-24_diagnostic/diagnostics/`. The locked bin scheme
  these scripts informed is described in `phases/E.5-bin-redesign.md`. The
  PNG outputs remain at `output/osbs/HAND-diagnostic-2026-04-25/` for
  reference. `generate_hypothetical_setups.py` had a stale import
  (`sys.path.insert(0, str(REPO / "scripts"))` predates the 2026-05-12
  module move into `scripts/osbs/`), so it would no longer run without
  edits anyway.

### `docs/` — Superseded documentation

- **`hillslope-binning-rationale.md`** — Documented the 1×16 hybrid bin
  scheme used briefly in April 2026. Superseded 2026-05-04 by the 24-bin
  TAI-focused scheme. Current rationale lives in
  `phases/E.5-bin-redesign.md` (working scheme + LIDAR error budget).
- **`ns-aspect-bug.md`** — Standalone technical explainer for the N/S
  aspect sign-convention bug. The bug was fixed in 2026-02 (interim
  correction + Phase A pgrid UTM CRS support). The doc's "Phase D will
  replace `np.gradient` entirely" prediction never landed
  (`spatial_scale.py` still uses `np.gradient` and is unlikely to change),
  but the core history is preserved in `phases/A-pysheds-utm.md` and in
  the pysheds fork's commit history. Reference-only at this point.
- **`water-masking-and-lake-representation.md`** — Mixture of (a) the
  abandoned weir-overflow Phase G design (replaced 2026-04-09 by
  submerged lake column) and (b) CTSM stream/PCT_LAKE/MOSART source
  notes. The NWI dual-mask implementation it described is in current
  use; canonical descriptions are now in `run_pipeline.py`'s docstrings
  and inline comments (`Step 1`, `Step 3c`, `Step 4`) plus
  `docs/lake-column-ctsm-audit.md` Section 6.9. The CTSM source notes
  are useful background for Phase H Track A but no longer load-bearing
  for any active code.
- **`synthetic_lake_bottoms.md`** — Brainstorming notes from 2026-02-20,
  flagged "(not implemented)" in its own header. Replaced operationally
  by the NWI dual-mask + lake column construction described in
  `lake-column-ctsm-audit.md` and built in the pipeline. Inactive
  references to the pre-2026-05-12 numbered STATUS.md problems make it
  hard to follow today.

## What's NOT in this batch

Still in `docs/` (active, current, or load-bearing):

- `lake-column-ctsm-audit.md` — Canonical lake-column parameter audit.
  Revision banner trimmed 2026-06-27 (full history in git log).
- `neon-data-products.md` — Active NEON DP catalog; updated 2026-05-20.
- `pysheds-utm-walkthrough.md` — Phase A historical walkthrough; kept
  for context.
- `data-acquisition-dates.md` — Active reference with a live open
  follow-up (Lee 2023 LIDAR vintage). Line-number references inside it
  refreshed 2026-06-27.

Still in `scripts/diagnostics/`:

- `diagnose_water_mask.py` — Phase E.6 hole-detection diagnostic; works
  against current `data/mosaics/production/water_mask.tif`.
- `overlay_nwi_water.py` — DTM hillshade + NWI overlay; runs on current
  production mosaics.

Still in `scripts/osbs/`:

- `run_pipeline.py`, `run_pipeline_production.sh`, `run_pipeline_smoke.sh`,
  `spatial_scale.py`, `hillslope_params.py` — production pipeline.
- `make_osbs_scrip.py` — Phase H Track A SCRIP-file generator (mesh-mode
  workaround for CTSM Issue #1432).

## Cleanup actions also performed this batch

Not file-moves but logged here for the audit trail:

- Deleted `swenson/.pytest_cache/` and `swenson/.ruff_cache/` (gitignored
  cache cruft).
- Relocated loose `output/osbs/nwi_water_overlay.png` into a dated subdir
  `output/osbs/2026-03-24_nwi_overlay/`.
- Created `data/HU8_03080103_Watershed/README.md` (provenance for the
  previously-undocumented 115 MB NWI shapefile directory).
- Docstring fixes in `scripts/osbs/run_pipeline.py` (removed dead
  reference to the archived `dem_processing.py`) and
  `scripts/osbs/spatial_scale.py` (replaced stale `STATUS.md #4` anchors
  with phase-doc pointers).
- Trimmed the 9-line dated revision banner at the top of
  `docs/lake-column-ctsm-audit.md` to a one-line git-log pointer
  (preserving the 2026-04-30 PI-redesign callout, which is still
  load-bearing for downstream sections).
- Refreshed stale `run_pipeline.py:NNN` line references in
  `docs/data-acquisition-dates.md`.
- Updated `CLAUDE.md` for: MERIT runtime (~3-4 min → ~10-20 min, 48 GB
  RAM); listing of `make_osbs_scrip.py` under `scripts/osbs/`; "frozen
  copies" wording correction; addition of `data/HU8_03080103_Watershed/`
  to the directory tree; sync with this batch's archive moves.

## Note on broken imports in archived scripts

After the 2026-05-12 shared-module de-coupling (`spatial_scale.py` and
`hillslope_params.py` moved from `scripts/` root into each pipeline's
own directory), any archived script that imports these modules via the
old parent-path pattern will fail with `ModuleNotFoundError`. Affected
items in this batch include
`output/osbs/HAND-diagnostic-2026-04-25/generate_hypothetical_setups.py`
(now at `osbs/generate_hypothetical_setups.py:45-47`). These are
reference-only; not expected to run. If anyone needs to revive one, fix
the import path manually — the module names are unchanged, only the
location is.
