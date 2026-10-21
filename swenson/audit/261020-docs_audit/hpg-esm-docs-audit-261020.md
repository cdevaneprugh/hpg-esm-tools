# hpg-esm-docs audit — Research, Swenson Implementation, Reference pages

**Date:** 2026-10-20
**Scope:** `hpg-esm-docs/docs/research/*.md` (3 pages), `docs/swenson/*.md` (8 pages), `docs/reference/*.md` (2 pages). CTSM-porting / installation / running pages deliberately not audited.
**Mode:** read-only fact-finding. No files in any repo were modified. This report lives in the session scratchpad; move it to `swenson/audit/261020-docs_audit/` if you want it in the record.

## 0. Method and sources of truth

Order of authority used when a doc claim and a record disagreed:

1. Code and data files: `swenson/scripts/osbs/run_pipeline.py`, `hillslope_params.py`, `spatial_scale.py`, `merit_regression.py`; the production NetCDF `output/osbs/2026-05-05_production/hillslopes_osbs_production_c260505.nc` (+ `hillslope_params.json`, `production_summary.txt`); the pysheds fork (`uf-development` @ `ed72724`); the NCAR-NEON fork (`uf-osbs` @ `07786dd`); the CTSM fork (`uf-ctsm5.3.085` @ `5b930dc8d`) and its ccs_config submodule; the operative case `$CASES/osbs.swenson.spinup` (`user_nl_clm`, `CaseDocs/lnd_in`, `SourceMods/src.clm/*.F90`, `CaseStatus`); NEON forcing files on disk; `output/2026-05-19_osbs.swenson.spinup_timeseries/h0a_annual.nc`; Swenson's reference code at `$BLUE/Representative_Hillslopes/`.
2. Git history of all five repos (198 commits in hpg-esm-tools; 19 fork commits in pysheds; 2 in NCAR-NEON; 11 in CTSM; 4 in ccs_config).
3. Phase records `swenson/phases/*.md`, `swenson/docs/*.md`, the archived audits under `swenson/audit/`, and `STATUS.md` — used for the story and for numbers that only exist in logs (e.g. the January 2026 MERIT stage results, Phase B memory numbers).
4. External: GitHub (PRs #3700/#3701, Issue #1432) fetched; the DiscussCESM thread could not be fetched (TLS error).

Several figures were recomputed directly from data rather than taken from any record (DTM relief, NEON annual precipitation, `drift_50yr`, production bin statistics, pytest collection counts, mutation totals). Those are listed in Appendix A.

**Severity legend**

| Tag | Meaning |
|---|---|
| **MAJOR** | Wrong about the production method or result; would misstate a methods section if copied. |
| **MODERATE** | Misleading, stale relative to the code, or certainty overstated. |
| **MINOR** | Rounding, wording, small omission. |
| **NARRATIVE** | Not wrong, but the development story has a gap worth filling for a paper. |
| **UNVERIFIED** | Could not be checked against anything in the repos; needs you or the PI. |
| OK | Checked against the source and correct. Listed only where it is useful to know it was checked. |

Line numbers below are from `cat -n` of the files as of hpg-esm-docs commit `e025306`.

---

## 1. Summary

The documentation is in good shape overall. Of the 13 pages, five are essentially clean (`merit-validation.md`, `neon-forcing.md`, `lateral-flow-and-routing.md`, `dataset-comparison.md`, `reference/modifications.md`), and the numeric production values (Lc, A_thresh, Q01/Q99, lake column, per-rep rescale, chain geometry, stream parameters, regression correlations, forcing-regression metrics) all match the files on disk. The remediation batches of 2026-08 and 2026-10-20 clearly did their job on the numbers.

The problems that remain are mostly **descriptions of method that were written for an earlier pipeline and never updated when the pipeline changed**, plus a few places where test-run numbers are presented as production numbers. Counts: 4 MAJOR, 10 MODERATE, ~20 MINOR, 4 NARRATIVE, 8 UNVERIFIED.

**The four MAJOR items (fix these before anything is reused in a methods section):**

1. **`osbs-implementation.md:151-153` — slope and aspect are described as computed with pgrid's Horn (1981) stencil on the original DEM.** The production pipeline has used NEON's DP3.30025.001 slope/aspect products directly since 2026-03-23 (PI decision; locked in STATUS.md). The page never mentions NEON slope/aspect at all.
2. **`osbs-implementation.md:107-109` and `index.md:24` — "connected-component extraction and edge trimming" is described as a current preprocessing step.** That code existed from 2026-01-26 to 2026-03-23 and was deleted as dead code. The production pipeline refuses any mosaic with nodata and relies on the pre-selected 0%-nodata domain.
3. **`hand-binning-and-lake-column.md:149-157` — the PI SourceMods are described as "retained but inert" under `spillheight = 0.0`.** Only the uniform elevation shift is disabled. Three mechanisms remain active in the operative case (lake/flood-zone ponding with overflow above `hill_elev = 0`, inter-column surface-water chaining, and global suppression of saturated-excess surface runoff), and the lake-column stability verdict depends on one of them. The same shorthand appears in `STATUS.md`'s locked-decisions table.
4. **`pysheds-porting.md:93-108` — the test-suite table is wrong on every row** (file names, counts, a non-existent `test_dem.py`, two real test files missing). Current fork: 116 tests across 5 files.

**MODERATE items worth a pass:** Lc plausibility checks quoted at the 300 m test value rather than the 356 m production value (`osbs-implementation.md:86-89`); the FFT "sensitivity sweep" was run on the 8 m artifact peak, not the 356 m production peak (`:93`); "~29 GB peak" is the partial-pipeline figure, the full pipeline peaked at 58 GB (`:189`, `index.md:42`); "DTND tail removal applied per-catchment-side" is not what the code does (`:175`); `tdepth_grc = 0` stated as fact where the phase record says unverified (`lateral-flow-and-routing.md:141`); the subset-data tree in `neon-sites.md:115-128` does not match what is on disk; the SourceMod inventory ("two SourceMods", "hard-coded constant") in `hand-binning-and-lake-column.md:130-135`; `reference/modifications.md` omits the fork's `spillheight` namelist addition.

**NARRATIVE gaps:** Phase E (NEON slope/aspect adoption, NWI mask, four intermediate binning schemes, 2026-03-30 PI decisions) is absent from the phase story in `index.md`; the lake-column design history (weir-overflow idea → submerged column → NWI-only → spillheight retirement → −6 m lock → the 5 m `hill_distance` bug → per-rep rescale) is compressed to a parameter table; the active SourceMods are not part of the Phase F description anywhere.

---

## 2. The development story as reconstructed from the repos

For orientation when revising the narrative pages. Dates are commit dates unless noted.

| Period | What happened | Evidence |
|---|---|---|
| 2026-01-09 → 01-14 | hpg-esm-tools created; CTSM fork work (mksurfdata fixes, HiPerGator config, `spillheight` namelist 2026-01-12); paper summaries added 2026-01-14. | hpg-esm-tools `660f665`…`4c58391`; CTSM `3506a69c0`, `5315b9fcc`, `4a0319fee` |
| 2026-01-21 → 01-23 | `swenson/` created; Swenson's `pgrid.py` copied into our pysheds fork (`7b20e8e`, 2026-01-22, 4,220 lines); MERIT stages 1–9 (stage 5–6 unit/width fixes 01-22, stage 7–9 01-23). No repo evidence of work before 2026-01-09. | hpg-esm-tools `e67845d`, `7fc8076`, `fdfceff`, `5f56504`, `5234fd6`, `d74284a`; `progress-tracking.md` |
| 2026-01-25 → 01-27 | First OSBS pipeline: full mosaic, connected-component extraction, 4× subsampling, EDT-based DTND, `np.gradient` slope/aspect. | `8963fc3`, `7ca1430`; `progress-tracking.md:1555-1570` |
| 2026-02-10 → 02-23 | Phase scheme introduced (`794fd14`). **A**: UTM CRS in pgrid, synthetic DEMs, test audit + mutation testing (fork commits `556e11a`…`ed72724`). **C**: full-res FFT → 8.1 m artifact → raw-spectrum test → restricted-wavelength sweep → Lc 356 m at 20 m cutoff (02-10); PI accepts ~300 m (02-11); physical checks at Lc = 300 on the 5×5 block pass (02-17/02-23). **B**: 1 m at 64 GB feasible (02-23). Post-audit MERIT fixes raise area fraction 0.82 → 0.92 (02-19/02-20). | `0721e51`, `6ed92b1`, `b42b112`, `41bcac8`, `2bf637e`; `phases/A,B,C` |
| 2026-02-24 → 03-16 | **D**: shared modules extracted, UTM-aware `spatial_scale.py`, tiered wrappers, production domain R4C5–R12C14 run (Lc 356.0 m, 21.8 min); two audits (03-10, 03-16). | `a51b0e3`, `624b5f3`, `bd92edf`, `fbb5795`, `25e24f8` |
| 2026-03-19 → 04-14 | **E**: 1×8 log bins (03-19); NEON slope/aspect adopted (03-23, `73c09fe`); dead code incl. component extraction removed (03-23, `4ef4196`); 1×4 equal-area interim (03-24); NWI mask (03-24) → broken boundary forcing (03-25) → dual mask (03-27, `8c727ca`); PI decisions on conditioning, domain, stream params (03-30); 1×8 log fails on the noise floor (03-30); A2 8-bin (04-09); 1×16 hybrid (04-14/15). | `phases/E-complete-parameters.md` |
| 2026-04-09 → 04-30 | **G**: weir-overflow idea abandoned (04-09), submerged lake column (04-10), lake audit (04-21…24), NWI-only scope (04-25), PI meeting retires spillheight SourceMod and reframes flood zone (04-30). | `phases/G`, `docs/lake-column-ctsm-audit.md` |
| 2026-05-01 → 05-05 | **E.5/E.6**: outlier diagnostic (05-01), Q01/Q99 true discard locked (05-02), 24-bin scheme + 0.25 m floor + lake `hill_elev` −6 m locked (05-04), dynamic lake `hill_distance` after a 5 m static value inverted the Darcy denominator (05-04), NWI hole-fill (05-04), per-rep rescale (05-05) → production NetCDF `c260505`. | `phases/E.5`, `3e7dcdb`, `38f0a6e` |
| 2026-05-05 → 05-19 | **F**: osbs5 100-yr → `osbs.swenson.spinup` 600-yr AD (05-11 → 05-14); **H** Track A mesh workaround + routing smoke (05-11/12); routing-gate audit (05-19); Phase F closeout (05-19). | `CaseStatus`; `phases/F,H` |
| 2026-07-15 → 08-17 | **I**: NEON forcing — v4 fetch, smoke, token fix, NCAR-NEON fork, I4 reproduce-v4, I5 full record + 2017 splice, ingestion smoke + `dtlimit=-1`, AD convergence, recipe. | `phases/I`; NCAR-NEON `2d30ebb`, `07786dd` |
| 2026-08-19 → 10-20 | Bridge-zone resolved by PI (fix unknown); H B/C retired; record-hygiene and docs remediation. | `STATUS.md` change log |

Where the docs' story departs from this record is called out per page below.

---

## 3. Page-by-page findings

### 3.1 `research/overview.md` (111 lines)

- **L9 — OK.** "DOE-BER … (2023-2026)" matches `hpg-esm-tools/docs/papers/DOE_Grant_Summary.md:9-11`.
- **L94 — MINOR — which forcing product the 1385 mm/yr comes from.** Recomputed from the NetCDF files (`PRECTmms × 1800 s`, summed per calendar year): the **custom** Phase I set gives 2018–2024 mean **1385 mm** (955–1810); the **pre-built v4** set gives **1407 mm** (1102–1759). The sentence says "NEON tower forcing", which is ambiguous between the two. Suggest "the custom NEON forcing (Phase I) averages ~1385 mm/yr over 2018–2024".
- **L95 — OK.** 11.08 km² / 90 km² = 12.3%.
- **L97 — OK.** Fan "position 3" wording matches `Fan_2019_Hillslope_Hydrology_ESM_Summary.md:171`.
- **L103–105 — OK.** Matches `STATUS.md`, `CaseStatus` (case.run success 2026-05-14 23:30), `lnd_in`.

### 3.2 `research/hillslope.md` (234 lines)

- **L36 — MINOR — slope maximum.** "median ~0.05 m/m, max ~0.06 m/m". Production land-bin slopes (`hillslope_params.json`): median 0.048, **max 0.067** (columns 2 and 3 are 0.0673 and 0.0664). Say "max ≈0.07".
- **L50 — OK.** Distance definition matches `run_pipeline.py:1476-1484`.
- **L153–169 — OK.** Matches `src/biogeophys/HydrologyDrainageMod.F90:132-160` and `WaterFluxType.F90:506-534`.
- **L177–192 — OK.** Listing matches `scripts/hillslope.analysis/` (only `README.md` and `docs/` omitted, which is fine).
- **L200–207 — OK.** Four streams match `user_nl_clm` (`hist_type1d_pertape = ' ', 'COLS', ' ', 'COLS'`; h2 and h3 share the vertically-resolved list).
- **L218 — OK.** `QLATFLOWOUT` registered under `use_hillslope` (`WaterFluxType.F90:506-509`).

### 3.3 `research/neon-sites.md` (164 lines)

- **L39 — OK.** `LONGXY = 278.006569` → 81.99°W.
- **L48 — MINOR.** Same as overview L94: "~1385 mm/yr … (range 955-1810)" are the custom-set numbers; v4 gives 1407 (1102–1759). Name the product.
- **L98 — MINOR — TBOT product.** "Meteorological | DP1.00002/3". The forcing generator reads TBOT from **DP1.00003.001** only (`flow.api.clm.R` `listDpNum`); DP1.00002.001 is not used. `neon-forcing.md:29` has it right.
- **L109 — OK.** `tools/site_and_regional/` contains `run_tower`, no `run_neon`.
- **L115–128 — MODERATE — subset-data tree does not match disk.** `/blue/gerber/earth_models/shared.subset.data/` contains `osbs-cfg.16pfts.PtVeg.no-dompft.1901-1921/`, `osbs-cfg.78pfts.hist.no-dompft.1901-1921/`, `subset_data_test/` — there is no `OSBS/` directory. Inside an `osbs-cfg.*` dir: `datmdata/` holds the `clmforc.cruncep.V7.c2016.0.5d.{Prec,Solr,TPQWL}.OSBS.YYYY-MM.nc` files directly (no `atm_forcing.datm.CRUNCEP.*/` subdirectory), plus `surfdata_OSBS_hist_1850_78pfts_c250925.nc` and `user_mods/{shell_commands,user_nl_clm,user_nl_datm_streams}`. Note also that the operative case points `CLM_USRDAT_DIR` at `/blue/gerber/sgerber/CTSM/subset_input` (surfdata `c251002`), not at `shared.subset.data`. Either label the tree "illustrative" or correct it.
- **L132–134 — OK.** Matches Phase I record and the `run_tower` memory note.
- **L146 — OK.**
- **L156–157 — OK.** `xmlquery`: `RUN_TYPE=startup`, `CLM_ACCELERATED_SPINUP=on`, `RESUBMIT=0`, `STOP_N=100`; `lnd_in:79-80`; post-AD `CaseStatus` last `st_archive success 2026-05-21 18:52`.
- **L158 — UNVERIFIED — osbs2 "860+ yr".** The osbs2 case directory recorded in `swenson/CLAUDE.md` (`/blue/gerber/sgerber/earth_model_output/cime_output_root/osbs2/`) does not exist or is not readable from this account; the 860-yr figure could not be checked. (`swenson/CLAUDE.md`'s path is stale too.)
- **L160 — MINOR.** `$CASES` also holds `osbs2.branch.v4`, `osbs4.branch.v1`, `osbs4.branch.v2` (the E.5 record refers to `osbs4.branch.v2/SourceMods`). Only `osbs2.branch.spillheight/user_nl_clm` sets `spillheight=0.5`. Consider listing all retired cases or saying "and later branch cases".

### 3.4 `swenson/index.md` (93 lines)

- **L21–26 — mostly OK.** `uf-development` branch confirmed; "Restricted-wavelength FFT (min 20m cutoff)" matches `run_pipeline.py:85`.
- **L24 — MAJOR (stale) — "Edge handling | Connected-component extraction + edge trimming".** This code (`scipy.ndimage.label` → largest component → trim all-nodata rows/cols) was added 2026-01-26 (`8963fc3`) and removed 2026-03-23 (`4ef4196`, "remove mosaic bloat and dead code"). No `ndimage.label` call exists anywhere in `scripts/` or the archived cleanup dirs today. The production pipeline instead **requires** a contiguous 0%-nodata mosaic (`run_pipeline.py:956-968` exits on any nodata pixel) and the 90-tile domain was selected for that property (`data/neon/tile_coverage.md:41-45`; PI decision 2026-03-30, `phases/E:163-165`). Replace with "Edge handling | Not needed: domain pre-selected as the largest 0%-nodata rectangle (R4C5–R12C14); pipeline asserts no nodata".
- **L38 — UNVERIFIED — "December 2025 to January 2026".** Earliest repo evidence of hillslope work: paper summaries 2026-01-14 (`4c58391`), `swenson/` directory 2026-01-21 (`e67845d`), pgrid port 2026-01-22 (pysheds `7b20e8e`). `progress-tracking.md` contains no 2025 dates. If the December work was uncommitted exploration, say so; otherwise "January 2026".
- **L40 — OK.** Stage 8: height 0.9999, distance 0.9982, slope 0.9966, aspect 0.9999, width 0.9597 > 0.95; area 0.82 (`progress-tracking.md:698-707`); regression baseline per `merit_regression.py:72-79`.
- **L42 — MINOR / MODERATE.** (a) "82 unit tests, 0 failures" was true on 2026-02-18 (`phases/A:317`); the fork at `ed72724` now collects **116 tests in 5 files** (`pytest --collect-only`). Date the claim or update it. (b) **"29 GB peak for 90 M pixels" is the wrong figure for the pipeline.** 29.2 GB was the Phase B *scalability* test, which ran only DEM conditioning + flow routing + HAND (`phases/B:87-99`). The *full* pipeline on the full 90-tile domain in the Phase B resolution comparison peaked at **58 GB** (`phases/B:126, 147`), and that is why Phase D set the wrapper to 64 GB (`phases/D:60`). Say "58 GB peak (Phase B full-pipeline run); 64 GB allocation".
- **L44 — NARRATIVE — Phase E is missing.** The story jumps from "Phases A–D" to "Phases E.5 + E.6". Phase E (2026-03-19 → 04-15) contains decisions a methods section needs: NEON DP3.30025.001 slope/aspect replacing pgrid (PI, 03-23); NWI open-water mask and the dual-mask strategy (03-24 → 03-27, including the failed boundary-forcing attempt); the 2026-03-30 PI decisions (standard depression filling, domain final, stream params left as-is); and four intermediate bin schemes (1×8 log, 1×4 equal-area interim, A2 8-bin, 1×16 hybrid) before E.5's 24-bin scheme. "E.5 replaced Swenson's 4-bin equal-area scheme" is true only in the sense that 4-bin was the starting point in January.
- **L46 — OK.** `drift_50yr = −0.48%` recomputed from `h0a_annual.nc`: mean(yr 501–550) = 4211.0, mean(551–600) = 4190.7 gC/m² → (b−a)/b = **−0.485%**. (The Phase F record quotes the magnitude, 0.48%; the docs' signed value with definition is the better statement.) Lake 5.78 m / 2.5 m per `phases/F:189-193`.
- **L46 — NARRATIVE.** The "lake column stable" verdict and its "6 m" ceiling come from the PI's `SurfaceWaterMod.F90` SourceMod (Mechanism A, `phases/H:721, 727-741`), not from stock CTSM. Nothing on this page or on `osbs-implementation.md` says the operative case runs with PI SourceMods. See §3.7.
- **L50 — OK.** `DATM_MODE = CLMCRUNCEPv7`.
- **L58, 62–64, 81–85 — OK.**

### 3.5 `swenson/pysheds-porting.md` (186 lines)

- **L27 — OK.** Swenson's bundled `pgrid.py` is 4,209 lines (`$BLUE/Representative_Hillslopes/pysheds/pysheds/pgrid.py`); upstream `695d8b2` (2021-12-06, "Refactor grid methods") and `e3c511d` (2025-08-14, "Remove pgrid to fix linting errors") both exist in the fork's history.
- **L44 — MINOR.** `_2d_geographic_coordinates()` is listed as a Swenson supporting method; it is now `_2d_crs_coordinates()` (`pgrid.py:1769`). The rename is stated at L185, but the L44 table should carry a "(since renamed)" note.
- **L66 — MINOR (wording).** "copy the entire `pgrid.py` from Swenson's fork (commit `7b20e8e`, 2026-01-22)" reads as if `7b20e8e` is a commit in Swenson's repository. It is the commit in *our* fork that added the copy (`cdevaneprugh/pysheds` `uf-development`), and it added 4,220 lines — already slightly modified from Swenson's 4,209. "4,365 lines" today: OK.
- **L74 — OK.** `pysheds/grid.py` diff: `raise ImportError` → `from pysheds.pgrid import Grid`.
- **L80–87 — OK.** 12 / 13 / 2 per `phases/A:106`; `_signed_mintype` at `pgrid.py:50`.
- **L93–100 — MAJOR (stale) — test table.** Current fork (`ed72724`), by `pytest --collect-only` and by counting `def test_`:

  | File | Docs | Actual |
  |---|---|---|
  | `test_hillslope.py` | 15 | **11** |
  | `test_utm.py` | 22 | **38** |
  | `test_grid.py` | ~40 | **33** (upstream file, 18 lines changed to skip incompatible tests) |
  | `test_dem.py` | ~5 | **does not exist** |
  | `test_split_valley.py` | — | **16** (missing from table) |
  | `test_depression_basin.py` | — | **18** (missing from table) |
  | total | 82 | **116** |

  The "82 passed" block (L107) was the state on 2026-02-18 (`phases/A:317, 383`); the test_hillslope count fell when 7 weak tests were deleted (`868472a`), and test_utm grew with the analytical tests (`ac71d9a`, `e8b6d49`). Either date the block or replace with the current numbers. Note: all tests still collect; I did not run the suite.
- **L102, L158 — MINOR — V-valley DEM size.** The generator defaults to **1000×1000 at 1 m** (`data/synthetic_valley/generate_synthetic_dem.py:35-37`); the unit tests build a **200×200 grid at 5 m** in memory (`tests/test_utm.py:37-39`; `phases/A:183`). "1000x1000, 5m pixel spacing" conflates the two. "22 tests" at L158 → 38.
- **L110 — OK.** `.test-mutations.json`: 30 mutants, 29 killed, 1 survived (M30 `rbank > 0 → >= 0`, equivalent), 8 CRS-branch, score 96.7; M4 killed only after `test_dtnd_finite_where_hand_valid` was added; 8 target methods.
- **L162 — OK.** 14/14 (`phases/A:248-252`).
- **L170–186 — OK.** Deprecation fixes per `phases/A:96-102`; helpers and constants present (`pgrid.py:46-47, 2049`).

### 3.6 `swenson/merit-validation.md` (298 lines)

This page checks out. Every stage number was traced to the January record (`audit/240210-.../progress-tracking.md`) or the archived stage outputs, and the final/regression numbers to `merit_regression.py` and `scripts/merit_validation/output/`.

- **L9 — OK** (`merit_regression.py:50-54`, `826-848`).
- **L19–29 — OK** (`stage1_summary.txt`: 2.17 %, HAND −54.05 … 619.06 m, DTND 0 … 22,931 m, 200.7 s).
- **L39–45 — OK** (original 500/1000/2000 regions → 766/760/758 m, `progress-tracking.md:314-316`; current 500/1000/3000 + full → 766/760/754/2679, median 763.0, `results.json`).
- **L53–60 — OK** (`progress-tracking.md:366-377`).
- **L70–77 — OK** (`:419-425`). **L87–88 — OK** (`:468`). **L109–116 — OK** (`:508`).
- **L126–153 — OK** (`:529`, `:650`; `merit_regression.py:50-54, 88`).
- **L161–183 — OK** (`:676-677`, `:698-707`).
- **L193–201 — OK** (`stage9_summary.txt`).
- **L209–218 — OK** (`EXPECTED` dict; 2026-02-24 run PASS, all deltas ≤ 0.0027).
- **L227 — OK** ("15 hypothesis tests across 7 runs", `area_fraction_research.md:492-493`). MINOR aside: that research doc concluded 0.82 was "an inherent limit"; the later post-audit work disproved it, and the page's "no test established it as a hard ceiling" is the correct current statement.
- **L254–276 — OK** (`full_pipeline_audit.md:899-901, 1214, 1243, 1391-1392, 1822`).
- **L293–296 — OK** (`LC_TOLERANCE_PCT = 5`, `TOLERANCE = 0.01`, `merit_regression.sh --mem=48gb`).

### 3.7 `swenson/osbs-implementation.md` (304 lines)

- **L19–23 — OK** (`data/neon/README.md`).
- **L27 — MINOR — "233 tiles, 323 km², 58.5% valid data".** 323 km² is the 17×19 bounding canvas (323 tile positions), not the area of the 233 existing tiles (233 km²); 58.5 % = the 189,024,929-pixel largest connected component of that 323 M-pixel canvas (`progress-tracking.md:1564, 1757`). Suggest "the full 17×19 tile canvas (323 km²; 233 tiles exist; the largest contiguous valid region covered 58.5 % of the canvas)".
- **L29 — OK.** Recomputed from `data/mosaics/production/dtm.tif`: min 23.53, max 55.50, p1 25.04, p99 51.14 m; 0 nodata.
- **L36–37 — OK.**
- **L48, 52, 54 — OK** (`phases/C:178-188, 126`).
- **L60–71 — MINOR.** Table row "None (full) | 8.1" — on the *contiguous production mosaic* the full-spectrum Lc was **7.7 m** (`phases/C:244`); 8.1 m was the earlier nodata-contaminated interior mosaic (`:66`). The other rows match `phases/C:253-261`.
- **L75 — OK** (355.98 m; 63,362).
- **L86–89 — MODERATE — plausibility numbers are from the Lc = 300 m test, not production.** P95 DTND/Lc = 1.17, mean catchment/Lc² = 0.876 and max/Lc = 3.1 were computed with **Lc = 300 m on the 5×5 block R6–R10/C7–C11** (`phases/C:313-345`). At **Lc = 356 m** the same test gave P95/Lc = **0.98**, mean/Lc² = **0.877**, max/Lc = **2.6** (`phases/C:323-327`). Both pass, but the page attaches the 300 m numbers to the 356 m result. Quote the 356 m row, or state that the checks were run at the ~300 m working value on the 5×5 block before Lc was finalized. Also L89: "25M pixels" is the 5×5 block, the production domain is 90 M.
- **L93 — MODERATE — the sensitivity sweep was on the wrong peak.** The 20-configuration sweep (variation 1.7 m, "psharp 9–12") was run on 2026-02-10 against the **full-spectrum 8.1 m micro-topographic peak** (`phases/C:58-87`: "Lc = 8.1 m, rock-solid stable"; psharp 11.56/8.95), before the restricted-wavelength method existed. No parameter sweep of the **356 m** peak is recorded; its sharpness is psharp 3.95/3.67 (`phases/C:256`). As written, the paragraph implies the production Lc was sweep-tested. Reword to what was actually shown: the artifact peak is parameter-insensitive; the drainage-scale peak is stable across the 20–100 m cutoffs and both peak models (285–356 m).
- **L101 — MINOR.** 34 cells × 8,100 m² = 275,400 m² ≈ **0.28 km²**, not "roughly 0.25" (same at `dataset-comparison.md:14`).
- **L105 — MINOR omission.** The conditioning chain omits the water-mask step: NWI water pixels are lowered by **0.1 m in the depression-filled DEM** before `resolve_flats` so D8 flow passes through them (`run_pipeline.py:1016-1025`; `docs/data-acquisition-dates.md:144-146`). That is part of the method and should be in the list.
- **L107–109 — MAJOR (stale) — connected-component extraction / edge trimming.** See `index.md` L24. The `scipy.ndimage.label` + trim steps are not in the production pipeline (removed 2026-03-23, `4ef4196`). The explanation of *why* nodata boundaries break D8 accumulation (`max_accumulation = 1`, `progress-tracking.md:1519-1579`) is still the correct motivation for the 0 %-nodata domain and for the `sys.exit` guard at `run_pipeline.py:961-968`; rewrite the section as "domain selection", not "preprocessing steps".
- **L115–117, 124, 126 — OK** (production summary; `phases/E:90-93`; 11,082,394 − 10,682,175 = 400,219 px).
- **L138, 142–143, 145, 147 — OK** (`run_pipeline.py:1104-1139`; `ncdump`).
- **L151–153 — MAJOR — slope and aspect source.** The page says slope/aspect are "computed from the original DEM … using pgrid's Horn (1981) 8-neighbor stencil" with the UTM-aware code path. **The production pipeline does not compute slope or aspect at all.** Since 2026-03-23 (`73c09fe`; PI decision recorded in `phases/E:23, 61-71` and `STATUS.md` locked decisions) it loads NEON's **DP3.30025.001** slope and aspect mosaics and converts slope from degrees (`run_pipeline.py:864-876`: `slope = np.tan(np.deg2rad(neon_slope_deg))`; `data/neon/README.md:6-7, 22-24`: NEON applies Horn 1981 to a 3×3-averaged DTM with a 20 m tile buffer). The comparison that motivated the switch: pgrid vs NEON slope r = 0.91 (per-tile 0.78–0.98), aspect circular r = 0.84, NEON slopes ~0.008 m/m lower (`output/osbs/slope_aspect_comparison/results.json`; `phases/E:68-70`). The "nodata → max_elevation + 1 false gradients" rationale (L151) is moot on a 0-nodata domain. The Horn-stencil description is correct for the **MERIT regression** (`merit_regression.py:377`) and for the fork, not for OSBS production. The "CRS adaptation" bullet at **L216** and the `pysheds-porting.md` content are fine as fork descriptions, but this page's Slope/aspect section and the "Adaptations" list should say the production file uses the NEON product. Catchment-level aspect averaging (L155–157) *is* still applied to the NEON aspect (`run_pipeline.py:1175-1182`) — that part is correct.
- **L157 — OK** (`terrain_utils.py:236`; +0.0763).
- **L161 — MINOR / UNVERIFIED.** "all bins below 0.06 m/m" — bins 1–2 are 0.0673 / 0.0664. "at most ~6 % of incident solar" — no calculation is recorded anywhere; the relative cos-incidence correction for a 3.4° slope depends on solar elevation and exceeds 6 % at low sun. Treat as an order-of-magnitude statement or cite a calculation.
- **L165, 167–169 — OK** (`phases/E:151`; `hillslope_params.py:145-221`).
- **L175 — MODERATE — "Applied per-catchment-side" is incorrect.** `tail_index()` is applied **once to the whole land-pixel population** (`run_pipeline.py:1236-1244`; same in `merit_regression.py:591-599`; Swenson's `TailIndex`, `terrain_utils.py:286`, is likewise global). Per-catchment-side grouping is used for **aspect averaging**, not tail removal. Delete the sentence.
- **L179–183 — OK** (`hillslope_params.py:296-316`; `run_pipeline.py:1464-1484`; Swenson `representative_hillslope.py:847-859` "median distances"). "did not occur for any of the 24 production bins" is plausible (the 2026-05-04 diagnostic reports `fallback_used: false` for the trapezoid fit) but the per-bin quadratic fallback is not logged; low-risk.
- **L189 — MODERATE.** "~29 GB — comfortably under the 64 GB ceiling": the 58 GB full-pipeline peak is the relevant number (see `index.md` L42). "803 s" OK.
- **L191 — OK** (`phases/B:83`).
- **L193–199 — MINOR.** "Area and aspect: 0.64-0.99 correlations" understates: 5×5 block r(1 m, 4 m) was **0.35** for aspect and **0.59** for area; full domain 0.66–0.9999 (`phases/B:132-138`). The conclusion (1 m retained) is unaffected.
- **L207–211, 225, 232–248 — OK** (`run_pipeline.py:929-930`; `spatial_scale.py:736`; `phases/E.5`).
- **L252–256, 266–292 — OK** (`ncdump -h`; `phases/D:98`; `lnd_in:266`).
- **L298 — OK.** depth = 0.001·(9×10⁷)^0.4 = 1.519 m, width = 0.001·(9×10⁷)^0.6 = 59.2 m (`run_pipeline.py:1640-1651`); Swenson applies the same coefficients to `np.sum(area)` of the hillslope elements (`representative_hillslope.py:1105-1109`); reference file 0.2691 / 4.414 / 0.002327.
- **L300, 302 — OK.**
- **L304 — NARRATIVE.** The Phase F description should say that the operative case carries the PI's five-file `SourceMods/src.clm` set, of which three mechanisms are active (§3.8, L149–157), since they change the surface-water physics the results are read against.

### 3.8 `swenson/hand-binning-and-lake-column.md` (167 lines)

- **L9, 22–35, 39–48, 52 — OK** (`phases/E:151`; `phases/E.5:531-566, 373-516, 606-639`).
- **L58–63 — OK, one caveat (MINOR).** The 33.5× / 34.8× density ratios and "negligible singleton mass" were computed on the **pre-E.6** data (Q01 = −6.34, Q99 = +17.46; `output/osbs/2026-05-01_outlier_strategy/summary.json`); they were not recomputed after the hole-fill moved Q99 to 17.02. "~1.5 M (~2 %)" OK: 75,967,350 − 74,448,002 = 1,519,348.
- **L65, 75–76 — OK.**
- **L82 — UNVERIFIED — "NWI v3.2 OSBS subset".** No "v3.2" appears anywhere in hpg-esm-tools. The recorded provenance is the USFWS NWI **HUC8 03080103 (Lower St. John) watershed extract**, database version `202409`, extract October 2024, 2017 true-colour imagery (`data/HU8_03080103_Watershed/README.md`; `docs/data-acquisition-dates.md:40-72`). Replace "v3.2" with that, or tell me where v3.2 comes from. "~400,000 pixels" OK.
- **L88–105 — OK** (`ncdump`; `run_pipeline.py:1532-1612`; `docs/lake-column-ctsm-audit.md:407-443, 445-517, 519-575`; `phases/E.5:1269-1299` for the 5 m static-distance bug). L92's `wetlandisfull` rationale matches `lake-column-ctsm-audit.md:423-426`.
- **L109, 116–122 — OK** (`nhill_implicit = 533.706`; lake fraction 0.12314; `phases/E.5:1363-1396`; `phases/G:367-381`).
- **L130–135 — MODERATE — SourceMod inventory.** Checked by diffing `$CASES/osbs.swenson.spinup/SourceMods/src.clm/*.F90` against the fork's `src/`:
  - There are **five** modified files, not two: `HillslopeHydrologyMod.F90`, `SurfaceWaterMod.F90`, `SaturatedExcessRunoffMod.F90` (`qflx_sat_excess_surf(c) = 0` for all columns, dated 2025-11-26 in the PI's comment), `InfiltrationExcessRunoffMod.F90` and `SoilHydrologyMod.F90` (comment-only, no-ops).
  - "`SPILLHEIGHT` constant (originally 0.2 m, hard-coded)": in the SourceMod it is a module variable with default `0.2_r8` **that is read from the `hillslope_properties_inparm` namelist** (added to the namelist list and `shr_mpi_bcast`), and the fork adds the matching `spillheight` entry to `bld/namelist_files/namelist_definition_ctsm.xml` (commit `4a0319fee`, 2026-01-12). That is the only reason `spillheight = 0.0` in `user_nl_clm` is legal. The "originally … hard-coded" phrasing and the "rather than editing the SourceMod constant" phrasing at L149 should be reconciled with this.
  - The `HillslopeHydrologyMod.F90` SourceMod contains a second change (terminal-column-only surface contribution to stream water inside `HillslopeUpdateStreamWater`; routing-gated, hence dormant) that the page does not mention.
- **L133 — MINOR.** The `SurfaceWaterMod` description covers the pond-build/overflow logic (SourceMod lines 516–520) but not the inter-column surface-water chain it also adds (lines 547–561; Phase H "Mechanism D").
- **L149–157 — MAJOR (misleading) — "keeps the SourceMod files retained but inert".** Only the uniform `hill_elev − spillheight` shift becomes a numerical no-op. `phases/H-lateral-flow.md:704-741` (§7.5, revised 2026-05-19 after a review of all five files) records that with `use_hillslope=.true.`, `use_hillslope_routing=.false.`, `spillheight=0`:
  - **A** (`SurfaceWaterMod.F90:516-520`) pond-build for `hill_elev < 0`, instantaneous overflow once `hill_elev + h2osfc > 0` — **active**; this is the lake column's 6 m release valve ("the 6 m threshold is the lake's `hill_elev` value, not the spillheight");
  - **D** (`SurfaceWaterMod.F90:547-561`) inter-column surface-water chaining — **active**;
  - **E** (`SaturatedExcessRunoffMod.F90:294-295`) zero saturated-excess surface runoff for **all** columns — **active globally**;
  - **B** terminal-column stream contribution — dormant (routing off); **C** elevation shift — no-op.

  The Phase F lake verdict ("max 5.78 m at year 107, just under the 6 m Mechanism-A overflow threshold", `phases/F:189-193`) depends on A. For a methods section the operative physics is "CTSM 5.3.085 + three PI surface-water modifications", not stock CTSM. `STATUS.md`'s locked-decisions row "SPILLHEIGHT | 0.0 (namelist override; SourceMod inert)" carries the same shorthand (out of scope here, but you may want to fix it in the same pass). Suggest replacing L149–157 with a short "What remains active" table copied from Phase H §7.5, and cross-linking from `osbs-implementation.md` L304 and `lateral-flow-and-routing.md`.
- **L141–142, 163–167 — OK.**

### 3.9 `swenson/lateral-flow-and-routing.md` (191 lines)

Line references were checked against the fork source; all the quoted CTSM lines resolve as stated.

- **L16 — OK.** Tag `ctsm5.3.085` is an annotated tag whose commit is `fde1fc622` (tag object `44aa588a5`); `git diff ctsm5.3.085 dc3aa5ddc -- src/` shows only added `CLAUDE.md` files.
- **L24–37 — OK** (`HydrologyDrainageMod.F90:132-160`).
- **L39 — OK** (`namelist_defaults_ctsm.xml:446-447`; `SoilWaterMovementMod.F90:221-236`; `lnd_in:204`).
- **L43–60 — OK** (`SoilHydrologyMod.F90:2087, 2261-2263, 2280, 2358; 1703, 1818-1820, 1838`).
- **L64–79 — OK per `phases/H:1069-1089`** (the 2433–2509 block was not re-read line by line here).
- **L85–100 — OK.** A grep finds exactly 12 `if (use_hillslope_routing)` sites in `src/`. MINOR: `HillslopeStreamOutflow` begins at `:939` and `HillslopeUpdateStreamWater` at `:1053`; the quoted `:977+` / `:1078+` point inside the bodies. Harmless, but "939+" / "1053+" is cleaner.
- **L104–118 — OK** (`:2265-2272`).
- **L124–128 — OK per `phases/H:1096-1108` and `phases/F:231-234`** (not recomputed from h1 output).
- **L141 — MODERATE — certainty overstated.** "`tdepth_grc` (which under the current DATM + MOSART setup **defaults to 0**, in effect treating the chain-bottom boundary as an empty stream channel)" is presented as fact. The audit it is drawn from lists this as **unverified**: "Whether `tdepth_grc` is actually populated by MOSART in our case … likely 0" (`phases/H:1176-1182`; also `:1168` "likely 0"; `phases/F:81` "likely 0"). Soften to "assumed to be 0 (default fallback at `lnd_import_export.F90:623`); not verified against MOSART output".
- **L156–162 — OK.** `src/cpl/share_esmf/lnd_set_decomp_and_domain.F90:342`; `initGridCellsMod.F90:179`. Consider adding the `share_esmf` path, since the file is not under `cpl/nuopc`.
- **L167–171 — OK** (`:486-487`, `:501-502`).
- **L173 — OK.** GitHub: Issue #1432 "Area is not set for NUOPC single point cases", created Jul 20 2021, still open.
- **L179 — OK** (`python/ctsm/mesh_maker.py:214-216`).
- **L181 — OK** (`phases/H:1628-1633`).
- **L190 — UNVERIFIED.** The DiscussCESM thread could not be fetched (TLS error from this host); attribution and date not checked.

### 3.10 `swenson/dataset-comparison.md` (122 lines)

- **L11–12, 16 — OK** (reference file `AREA = 12,654.5 km²`, `nhillslope = 4`, `nmaxhillcol = 16`).
- **L14 — MINOR.** "≈0.25 km²" → ≈0.28 km² (275,400 m²).
- **L26–33, 39–44 — OK** (reference values 0.2691 m / 4.414 m / 0.002327 m/m; `representative_hillslope.py:1105-1109`).
- **L54–60 — OK** (Swenson HAND 0.174–8.144, distance 67.3–541.4; OSBS −5.126–12.460, 2.59–352.6).
- **L68 — MINOR (wording).** "~1.5–2×" compares only the upper ends (541 vs 353 m). The lower ends differ ~26× (67 vs 2.6 m) because the OSBS chain starts inside filled depressions next to the wide mask. The retired "5–25×" figure is documented in `audit/260310-.../docs-update-plan-260317.md:282` ("~2-10x (vs old 5-25x claim)"). Suggest stating both ends.
- **L70, 72–74 — OK.**
- **L76 — MINOR.** "slopes universally below 0.06 m/m" — two bins at 0.066–0.067; "<6 %" see §3.7 L161.
- **L93 — OK** (zero crossing between col 13 at 50.4 m and col 14 at 62.0 m; widths 555.6 → 268.5 m).
- **L110 — OK** (lake 12.31 %; FZ 0.73–2.90 %; upland 3.60–7.99 %).
- **L116–122 — OK.**

### 3.11 `swenson/neon-forcing.md` (175 lines)

This page is accurate; every edit-table row was matched to the fork diff and every metric to the output files.

- **L15–16 — OK** (84 and 101 `.nc` files on disk).
- **L27–36 — OK** (`docs/neon-data-products.md:26-33, 40`; `flow.api.clm.R` `listDpNum`).
- **L46 — OK** (NCAR-NEON `LICENSE` is AGPL-3.0).
- **L48–50 — OK** (merge-base `43a0cf4`; `+77/−31`).
- **L58–71 — OK.** All twelve rows correspond to hunks in `git diff main..uf-osbs -- TowerTools_ForcingData/flow.api.clm.R`.
- **L82–87 — OK** (`renv.lock`: R 4.0.5, 195 packages; `setup/environment.yml`; `install_source_pkgs.R:14-17, 26, 36-40`; `Makevars.conda`).
- **L100–105 — OK per `phases/I:1179-1191`** (1063 zips, 11 GB, 10 products).
- **L109–117 — OK** (`splice_2017_precip.py` docstring; `phases/I:1152-1171`). MINOR aside: the QC flag tally also shows PRECTmms 100 % gap-filled in 2018-01 and 91 % in 2019-12 (`scripts/neon_forcing/output/qc_summary.txt:28-29`), unrelated to the 2017 outage; worth one sentence so a reader does not assume 2017 was the only precipitation gap.
- **L123–134 — OK.**
- **L142 — OK** (`results.json`: PRECTmms both-measured rms 3.96e-7, maxabs 3.06e-5, r = 0.99999955; WIND rms 4.55e-6, maxabs 2.6e-4, r ≈ 1; +2.9 %).
- **L144 — OK** (`qc_summary.txt` PASS, NaN = 0).
- **L146, 148 — OK** (`phases/I:1093-1097, 992-999`).
- **L151 — OK** (`components/cdeps/streams/dshr_strdata_mod.F90:1047-1050`: the `'(a,i8)'` write with a character second argument).
- **L159–167 — OK.**

### 3.12 `reference/modifications.md` (222 lines)

- **L19–42 — OK.** Diff confirmed. PR #3700 merged **2026-02-18** (GitHub). **UNVERIFIED:** "first released in `ctsm5.4.021`" — the PR page names no release tag.
- **L44–68 — OK.** Diff confirmed at lines 271/295/328. PR #3701 merged **2026-03-12**; `ctsm5.4.025` confirmed via the linked merge PR #3813.
- **L70–87 — OK** for the diff (line 170). The statement that `ccs_config/machines/cmake_macros/gnu.cmake` already supplies the flags for CIME builds was not checked.
- **L89–97, 134–151 — OK** (diffs).
- **L157–167, 173–177 — OK** (`config_machines.xml:82-88, 110-112`).
- **OMISSION — MODERATE.** The fork also adds a `spillheight` namelist definition (`bld/namelist_files/namelist_definition_ctsm.xml`, group `hillslope_properties_inparm`; commit `4a0319fee` "Add spillheight parameter for hillslope hydrology research", 2026-01-12). This is the research-relevant fork change and the reason the `spillheight = 0.0` override in `user_nl_clm` works with the PI's SourceMods. It belongs in the CTSM list and the contribution table ("Not upstreamed — research-specific"). The fork's 10 `CLAUDE.md` files and `claude-todo.md` are documentation only and can stay unlisted.
- **L209–217 — OK.**

### 3.13 `reference/resources.md` (88 lines)

External links only; nothing to verify against code. "32 chapters" for the Technical Note was not checked. No issues found.

---

## 4. Cross-cutting issues

1. **Stale pipeline descriptions (MAJOR).** Two sections of `osbs-implementation.md` (slope/aspect; connected-component extraction) and one table row in `index.md` describe the January–March 2026 pipeline. The 2026-03-23 commits (`73c09fe`, `4ef4196`) changed both. Anything in the docs that says "pgrid `slope_aspect()`" or "`scipy.ndimage.label`" in the OSBS context is pre-Phase-E.
2. **Test-value vs production-value mixing (MODERATE).** Lc plausibility ratios (Lc = 300, 5×5 block), the FFT sensitivity sweep (8 m artifact peak), and the memory figure (29 GB partial pipeline) are all real numbers from the record attached to the wrong object. Each has a correct production counterpart in the same phase file.
3. **The operative case is not stock CTSM (MAJOR for a methods section).** Five SourceMod files, three active mechanisms, plus a fork namelist addition. The docs describe only the retired spillheight shift and call the rest inert. The authoritative description is `phases/H-lateral-flow.md` §7.5.
4. **Numbers that drifted with the test suite (MODERATE).** The pysheds page's test table and "82 passed" are February snapshots.
5. **Narrative compression (NARRATIVE).** For a paper, the two best-told stories (the Lc journey on `osbs-implementation.md:46-75`; the MERIT stage-by-stage page) are the model. Phase E's binning iterations, the NWI boundary-forcing failure → dual-mask fix, and the lake-column design history (`phases/E.5:1176-1193`, `phases/G` log) are the pieces still missing.

---

## 5. Items I could not resolve — need you or the PI

| # | Page:line | Question |
|---|---|---|
| 1 | `index.md:38` | Did hillslope work start in December 2025? No repo evidence before 2026-01-09; `swenson/` dates from 2026-01-21. |
| 2 | `neon-sites.md:158` | osbs2 "860+ yr spinup": the recorded osbs2 path is not accessible; where does the 860 figure come from? |
| 3 | `hand-binning-and-lake-column.md:82` | "NWI v3.2" — source of the version string? Repo provenance says HUC8 03080103 extract, DB version 202409. |
| 4 | `lateral-flow-and-routing.md:141` | Has `tdepth_grc = 0` ever been confirmed from MOSART/coupler output? Phase H §8.7 says no. |
| 5 | `osbs-implementation.md:161`, `dataset-comparison.md:76` | The "~6 % insolation correction" — is there a calculation behind it? |
| 6 | `modifications.md:42` | Release tag `ctsm5.4.021` for PR #3700 — confirm against the ChangeLog (PR page has no tag). |
| 7 | `lateral-flow-and-routing.md:190` | DiscussCESM thread attribution (poster, date) — not fetchable from HiPerGator. |
| 8 | `hand-binning-and-lake-column.md:149-157` | Policy: do you want the PI's active SourceMod mechanisms described in the public docs? They are the PI's work; the docs currently describe only part of them. |

---

## 6. Things checked and found correct (so they need not be re-checked)

- All production NetCDF numbers quoted anywhere in the docs: Lc 355.98 m, A_thresh 63,362, 20 m cutoff, Q01 −6.3459 / Q99 17.0201, pixels 75,967,350 → 74,448,002, 24 bin edges, lake `hill_elev` −6.0, `hill_distance` 1.296 (= 0.5 × 2.592), lake area 11,082,394 m² → 20,765 m²/rep, perimeter 96,540 → width 48,270 → 90.4 m/rep, `nhill_implicit` 533.7, lake weight 12.3 %, chain 1 → 25, `downhill_column_index` −9999/1…24, stream depth 1.519 / width 59.2 / slope 0.006896, `AREA = 90`, `LATIXY/LONGXY`, bedrock 0, 20 variables.
- MERIT: every stage number, the post-audit progression, the regression baseline and its 2026-02-24 PASS, Lc 763 m, 34 cells, gridcell bounds, 0.07 % area match, 42 % overlap, N/S swap counts.
- Lateral-flow page: all CTSM source line references (dispatch, Darcy gradient, routing-gated sites, spval chain, `mesh_maker` abort), Issue #1432 date/state, `grc%area = 90.006 km²`.
- NEON forcing: file counts, product table, all 12 fork edits, environment pins, splice details, regression and QC metrics, CDEPS bug line, Lombardozzi citation.
- Reference/modifications: all five CTSM diffs, both PR merge dates, `ctsm5.4.025`, ccs_config module versions and PIO env vars.
- `drift_50yr = −0.48 %` with the stated definition (recomputed: −0.485 %).
- Production DTM relief, NWI hole-fill count, 103 NWI features, Phase B/C/E.5/F/G/H/I numbers cited on the pages.

---

## Appendix A — values recomputed from data during this audit

| Quantity | Value | Source |
|---|---|---|
| Production DTM | min 23.53, max 55.50, p1 25.04, p99 51.14 m, 0 nodata | `data/mosaics/production/dtm.tif` |
| NWI mask pixels | 11,082,394 | `data/mosaics/production/water_mask.tif` |
| v4 annual precip 2018–2024 (mm) | 1759, 1102, 1145, 1397, 1450, 1551, 1442; mean 1407 | `data/datm/neon_OSBS/v4/OSBS/*.nc` |
| custom annual precip 2017–2025 (mm) | 1767, 1810, 955, 1121, 1343, 1465, 1598, 1400, 392 (half-year); 2018–2024 mean 1385 | `data/datm/neon_OSBS/custom/OSBS/atm/*.nc` |
| TOTECOSYSC means | yr 501–550: 4211.0; yr 551–600: 4190.7 gC/m²; drift (b−a)/b = −0.485 % | `output/2026-05-19_osbs.swenson.spinup_timeseries/h0a_annual.nc` |
| Production land-bin slopes | median 0.048, max 0.0673 (cols 2–3: 0.0673, 0.0664) | `hillslope_params.json` |
| Per-column area fractions | lake 12.31 %; FZ 0.73–2.90 %; upland 3.60–7.99 % | `hillslope_params.json` |
| pysheds tests collected | 116 (utm 38, grid 33, depression_basin 18, split_valley 16, hillslope 11) | `pytest --collect-only` in `$PYSHEDS_FORK` |
| Mutation totals | 30 / 29 killed / 1 survived (M30) / 96.7 % | `$PYSHEDS_FORK/.test-mutations.json` |
| `use_hillslope_routing` gate sites | 12 `if` sites in `src/` | `grep -rn` on the fork |
| CTSM tag | `ctsm5.3.085` → commit `fde1fc622` | `git rev-parse 'ctsm5.3.085^{commit}'` |
| Component-extraction code lifetime | added `8963fc3` (2026-01-26), removed `4ef4196` (2026-03-23) | `git log -S"ndimage.label("` |

## Appendix B — files read in full or in the cited ranges

hpg-esm-docs: all 13 in-scope pages, `mkdocs.yml`, git log. hpg-esm-tools: `swenson/CLAUDE.md`, `STATUS.md`, `phases/{A,B,C,D,E,E.5,F,G(part),H(part),I(part)}`, `docs/{data-acquisition-dates, lake-column-ctsm-audit (§4-5), neon-data-products (part)}`, `scripts/osbs/{run_pipeline,hillslope_params,spatial_scale}.py`, `scripts/merit_validation/{merit_regression.py,README.md,output/*}`, `scripts/neon_forcing/{splice_2017_precip.py (header), setup/*, output/*}`, `data/{neon/README.md, neon/tile_coverage.md (part), datm/neon_OSBS/README.md, HU8_03080103_Watershed/README.md}`, `output/osbs/2026-05-05_production/*`, `output/osbs/2026-05-01_outlier_strategy/summary.json`, `output/osbs/2026-05-04_bin_schemes/summary.json`, `output/osbs/slope_aspect_comparison/results.json`, archived MERIT stage summaries, `area_fraction_research.md` and `full_pipeline_audit.md` (grep), `progress-tracking.md` (grep), `docs/papers/{DOE_Grant_Summary,Fan_2019…}.md`. pysheds fork: git log/diff, `pgrid.py` (grep), `tests/*` (collect), `.test-mutations.json`, `data/synthetic_valley/generate_synthetic_dem.py` (grep). NCAR-NEON fork: git log, full `flow.api.clm.R` diff, `renv.lock`, `LICENSE`. CTSM fork: git log/diff vs tag, `src/biogeophys/{HydrologyDrainageMod,SoilHydrologyMod,HillslopeHydrologyMod,WaterFluxType,BalanceCheckMod,SoilWaterMovementMod}.F90`, `src/main/{lnd2atmMod,controlMod,initGridCellsMod}.F90`, `src/cpl/{nuopc/lnd_import_export,share_esmf/lnd_set_decomp_and_domain}.F90`, `python/ctsm/mesh_maker.py`, `bld/namelist_files/namelist_defaults_ctsm.xml`, `components/cdeps/streams/dshr_strdata_mod.F90`, ccs_config `machines/hipergator/*`. Cases: `osbs.swenson.spinup` (`user_nl_clm`, `lnd_in`, `SourceMods` diffs, `CaseStatus`, `xmlquery`), `osbs.swenson.post-ad` and `osbs.swenson.neon.spinup` (`CaseStatus` tails), `osbs2.branch.*` (`user_nl_clm`). Swenson reference code: `representative_hillslope.py` (`:840-862`, `:1095-1120`), `terrain_utils.py` / `geospatial_utils.py` (function index), reference NetCDF `hillslopes_osbs_c240416.nc` (full dump).

---

## Appendix C — Re-verification of the four MAJOR findings (2026-10-20, second pass)

Each item was re-checked from scratch with evidence independent of the first pass. All four stand.

**1. Slope/aspect source (`osbs-implementation.md:149-153`).** The pipeline at the commit that produced the production file (`3e7dcdb`, 2026-05-05 16:42, same day as the NetCDF `creation_date`) reads `neon_slope_deg = src.read(1)` at line 870 and contains no `slope_aspect`, `_gradient_horn`, or `np.gradient` call; today's file is the same. The slope/aspect mosaics it reads (`data/mosaics/production/{slope,aspect}.tif`) are dated 2026-03-23, the day of commit `73c09fe` "Use NEON slope/aspect products in pipeline (Phase E)". The page describes pgrid's Horn stencil on the original DEM — the January–March pipeline.

**2. Connected-component extraction (`osbs-implementation.md:107-109`, `index.md:24`).** Commit `4ef4196` (2026-03-23) removed `from scipy.ndimage import label` and the "Finding largest connected component" block. The production-commit pipeline exits with "Pipeline requires a contiguous mosaic" on any nodata pixel (line 969); the current file imports only `binary_erosion`/`binary_dilation` from scipy.

**3. SourceMods not inert (`hand-binning-and-lake-column.md:149-157`).** Three independent lines:
- *Build:* `$EXEROOT/lnd.bldlog.260511-141737.gz` lines 182 and 312 show `mpif90` compiling `SourceMods/src.clm/SaturatedExcessRunoffMod.F90` and `SourceMods/src.clm/SurfaceWaterMod.F90`, with the SourceMods directory first on the include path; build 2026-05-11 14:17, SourceMod files written 14:16.
- *Code:* `spillheight` appears only in `HillslopeHydrologyMod.F90` (the `hill_elev − spillheight` shift, line 363). The pond-build/overflow rule (`SurfaceWaterMod.F90:516-520`), the inter-column surface chain (`:547-561`) and `qflx_sat_excess_surf(c) = 0` (`SaturatedExcessRunoffMod.F90:295`) key on `hill_elev`, not `spillheight`.
- *Output:* `output/2026-05-19_osbs.swenson.spinup_timeseries/h1a_annual.nc` — lake column H2OSFC max 5,783.8 mm (year 107), 2,531 mm at year 600; flood-zone columns 1–3 max 1,894 / 724 / 595 mm. Stock `SurfaceWaterMod.F90:486-497` caps `qflx_h2osfc_surf` at `(h2osfc − h2osfc_thresh)/dtime` and drains everything above the centimetre-scale threshold, so metre-scale standing water requires the SourceMod rule.

**4. Test table (`pysheds-porting.md:93-108`).** `uf-development` @ `ed72724`: `test_utm.py` 38, `test_hillslope.py` 11, `test_split_valley.py` 16, `test_depression_basin.py` 18, `test_grid.py` 33 (whole module `pytest.mark.skip`, API-incompatible). `test_dem.py` never existed in the fork's history. Per-commit counts show the docs' 15 / 22 figures correspond to ~2026-02-12 → 02-17; by `e8b6d49` (2026-02-18) `test_utm.py` already had 38 and `test_hillslope.py` 11. Full run on 2026-10-20 (`ctsm` env, `PYTHONPATH=$PYSHEDS_FORK pytest tests/ -q`): **83 passed, 33 skipped, 12 warnings in 36.5 s**. Suggested replacement for the results block: that line, dated.
