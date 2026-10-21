# hpg-esm-docs — Second-Pass Review: Prose, Phrasing, and Story

**Date:** 2026-10-20
**Scope:** The 13 pages under *Research*, *Swenson Implementation*, and *Reference* in `hpg-esm-docs` at commit `ef1fc44` (after the four factual-fix commits of the first audit).
**Purpose:** Grade the prose and the narrative as the future basis of a paper's methods section. Every number or line reference relied on below was looked up this session in the scripts, source, production outputs, phase records, case directories, or git history — not taken from the first audit or from memory.
**Mode:** Read-only. No file in any repository was modified. A strict MkDocs build was run into the scratchpad only.

---

## 1. Verdict

**Factual health: good.** After the first audit's fixes, every quantitative claim I re-derived from the production NetCDF, `production_summary.txt`, `hillslope_params.py`, `merit_regression.py` and its `results.json`, the CTSM fork source, the pysheds and NCAR-NEON git histories, the v4-regression summary, the slope/aspect comparison JSON, and the DTM mosaic itself matches the text (ledger in §5). The strict build passes and all cross-page anchors resolve.

**Readability: uneven — B overall.** The technical core (`osbs-implementation.md`, `hand-binning-and-lake-column.md`, `lateral-flow-and-routing.md`, `pysheds-porting.md`) is clear, concrete, and already close to methods-section register. The connective tissue (`swenson/index.md`, the two Research status pages, the Design-history bullets) reads as compressed, date-stamped change logs rather than narrative. Four story-level problems need attention before any of this is lifted into a paper; they are in §2.

**Story: mostly told, with one false start and one glossed-over pivot.** The documentation tells the pysheds-port → MERIT-verification → OSBS-pipeline → bin-redesign → lake-column → spinup arc coherently. What it does not tell accurately is *where the project began* (December 2025 was CTSM case work with the global file and the spillheight SourceMods, not the pgrid port) and *how Lc went from "~300 m working value" to "356 m"* (the pipeline computes it; Phase C never "established 356").

| Dimension | Grade | One-line reason |
|---|---|---|
| Factual accuracy (after audit 1) | A− | Everything re-derived this pass matches; two residual wording issues (§2.3, §2.4) |
| Prose clarity, technical pages | B+ | Concrete, well-sequenced; a few 100-word sentences and in-page duplicates |
| Prose clarity, overview/status pages | C+ | Date-stamped log paragraphs; 31 ISO dates in a 1,600-word page |
| Narrative continuity | B− | Origin misattributed; Lc pivot glossed; "PASS" label on a scripted check that returned MARGINAL/FAIL |
| Methods-section readiness | C+ | Right content, wrong register: PI-attribution, dates, team-paths, and duplicated passages must be stripped and consolidated |

---

## 2. Story-level findings (ordered by importance)

### 2.1 The origin story is misattributed — `swenson/index.md:38, :40`

**What the docs say.** "Early work (December 2025 to January 2026) focused on porting Swenson's `pgrid.py` to the pysheds fork and reproducing his published MERIT results" and "**Pre-phase (2025-12 to 2026-01) — Methodology verification.** Ported Swenson's `pgrid.py` …".

**What the record shows.**

| Evidence | Date |
|---|---|
| PI's first hillslope case `/blue/gerber/sgerber/CTSM/cases/osbs2_1` (dir mtime) | 2025-11-05 |
| Our first branch case `$CASES/osbs2.branch.v2` (`CaseStatus` line 1) | 2025-12-03 |
| `$CASES/osbs2.branch.spillheight` created; its 5 SourceMods written | 2025-12-15 |
| CTSM fork: mksurfdata fixes, HiPerGator config, `spillheight` namelist entry (`4a0319fee`) | 2026-01-12 |
| hpg-esm-tools: paper summaries added (`4c58391`) | 2026-01-14 |
| `Representative_Hillslopes` cloned (`.git/HEAD` mtime) | 2026-01-21 |
| `swenson/` directory created in hpg-esm-tools (`e67845d`, "Phase 2") | 2026-01-21 |
| pysheds fork: `pgrid.py` imported (`7b20e8e`) | 2026-01-22 |
| First dated entry in `progress-tracking.md` (lines 40/168 read "2025-01-22", a typo for 2026) | 2026-01-22 |

No commit, case, or record places the pgrid port in December 2025. December 2025 was the *CTSM-side* start: running the PI's global-file hillslope configuration in branch cases and testing the spillheight runtime offset. That thread is documented well on `hand-binning-and-lake-column.md:139-143` ("November 2025 to April 2026" — supportable via the PI's `osbs2_1` case) but is absent from the timeline on `index.md`, so the two pages tell different first chapters.

**Why it matters for the paper.** The methods story has two independent tracks that converged: (a) a runtime approach to wetland columns (spillheight, Dec 2025 → Apr 2026) and (b) a data-side approach (custom hillslope file, Jan 2026 → May 2026). Track (b) *replaced* track (a) on 2026-04-30. The index currently narrates only (b) and mis-dates its start, so a reader cannot see that the lake column and raw-HAND binning were the answer to a problem first attacked at run time.

**Suggested fix (wording only).** Replace the first sentence of the Phases section with something like: "Work began in December 2025 with CTSM branch cases on the PI's global-file hillslope configuration and a runtime elevation offset (`spillheight`) for wetland columns. The data-side track — porting Swenson's `pgrid.py` and reproducing his MERIT results — started in January 2026 and, by April, superseded the runtime offset (see HAND Binning and Lake Column)." Relabel the paragraph "Pre-phase (2026-01) — Methodology verification."

### 2.2 The Lc narrative skips the 300 → 356 m pivot — `index.md:42`, `osbs-implementation.md:76-95`

**What the docs say.** "C established **Lc = 356 m** via a restricted-wavelength FFT (20 m cutoff)". STATUS's locked-decision table dates "Lc = 356 m" to 2026-02-11.

**What the record shows.** `phases/C-characteristic-length.md`:

- `:290` "2026-02-11: PI accepts ~300m as working Lc"
- `:294` "Accept Lc ~300m (range 285-356m) as the working value for pipeline development. Reserve final judgement until Phase A delivers correct DTND…"
- `:313` physical validation run at 300 m; `:388` "**Lc = 300m is confirmed.**"; `:392` "**Final Lc: ~300m (range 285-356m).** … A_thresh = 0.5 × 300² = 45,000 m²"

Meanwhile `scripts/osbs/run_pipeline.py:85` hard-codes only `MIN_WAVELENGTH = 20`; Lc itself is a run-time FFT output (`:938 lc_m = fft_result["spatialScale_m"]`). On the production mosaic the 20 m-cutoff lognormal fit returns 355.98 m (the same value already in Phase C's cutoff table at `:256` on 2026-02-10), and that is what `production_summary.txt` records.

So the *locked methodological decision* is the 20 m cutoff + lognormal peak; **356 m is a pipeline output**, and **~300 m was the Phase C working value** used for the physical-validation checks and early Phase D runs. The docs collapse this into "C established 356 m", which is not what Phase C concluded.

**Why it matters.** A methods section must say whether Lc was prescribed or computed. The honest, and stronger, statement is: "Lc is computed by the pipeline from the production mosaic (355.98 m at the 20 m cutoff); the 285–356 m spread across cutoffs and spectral models is the uncertainty." `osbs-implementation.md` already half-says this ("Lc = 356 m (355.98 m; lognormal fit at 20 m cutoff)" and "(1.17 at the 300 m working value)"), but `index.md` and the "Result" heading present 356 as a fixed finding of Phase C.

**Suggested fix.** On `index.md:42`: "C established the restricted-wavelength FFT method (20 m cutoff), with ~300 m (range 285–356 m) as the working value; the production pipeline computes Lc at run time and returns 356 m on the final domain." On `osbs-implementation.md`, add one sentence under *Result* saying Lc is computed, not prescribed.

### 2.3 Two plausibility checks are labelled **PASS** that the script itself did not pass — `osbs-implementation.md:85-91`

The JSON the page cites (`output/osbs/smoke_tests/lc_physical_validation/validation_results.json`) records:

| Lc | `check1_verdict` (max DTND / Lc, pass range 0.5–2.0) | max/Lc | p95/Lc | `check2_verdict` | mean area / Lc² |
|---|---|---|---|---|---|
| 285 | FAIL | 3.27 | 1.17 | PASS | 0.851 |
| 300 (primary) | FAIL | 3.10 | 1.17 | PASS | 0.876 |
| 356 | **MARGINAL** | 2.62 | 1.09 | PASS | 0.877 |
| overall `result` | **FAIL** | | | | |

The page says "**P95 DTND / Lc = 1.09.** … **PASS.**" and "**Mean catchment area / Lc² = 0.877.** … **PASS.**", then explains in a following paragraph why `max()` is not resolution-fair and P95 is. The numbers are all correct (I re-read them from the JSON), and the P95 argument is reasonable, but the bold PASS labels present a post-hoc reinterpretation as the scripted outcome. Phase C's own log is more candid: `:347` "Closing analysis — Check 1 passes, resolution context explains the apparent failure".

**Suggested fix.** Keep the numbers; change the framing to: "The script's criterion (max DTND / Lc within 0.5–2.0) returned MARGINAL at 356 m and FAIL at 285/300 m. Because `max()` is not comparable across resolutions (next paragraph), the 2026-02-23 closing analysis adopted P95 as the comparison statistic, giving 1.09." Readers of a methods section will trust a disclosed reinterpretation; they will not trust a PASS they later discover was MARGINAL in the artifact.

### 2.4 "`src/` is identical" overstates slightly — `lateral-flow-and-routing.md:16, :191`

`git diff --stat ctsm5.3.085 dc3aa5ddc -- src/` shows four added files: `src/CLAUDE.md`, `src/biogeochem/CLAUDE.md`, `src/biogeophys/CLAUDE.md`, `src/main/CLAUDE.md`. Restricting to `*.F90` the diff is empty. The tag's commit is `fde1fc622` as stated (annotated tag object is `44aa588a5`). Wording fix: "the Fortran under `src/` is identical".

### 2.5 Phase F results are narrated four times

`drift_50yr` / `O_SCALAR` verdicts appear in full on `swenson/index.md` (twice: Phases and Current Status), `osbs-implementation.md:308`, `lateral-flow-and-routing.md:143`, and in prose on `research/overview.md:103`. The spillheight retirement is told on three pages, `binary_fill_holes` on three, `use_hillslope_routing = .false.` on five, the 2026-08-19 PI resolution on four. None of the copies contradict each other now, but every future edit has to land in four places, and a methods-section author will have to pick one. (The user has deferred consolidation to the planned reorganization; recorded here so it is not lost.)

### 2.6 The register is "project log", not "methods" — quantified

| Page | words | ISO dates | "PI" | we/our/us | bold spans |
|---|---|---|---|---|---|
| swenson/index.md | 1,614 | **31** | **10** | 4 | 17 |
| hand-binning-and-lake-column.md | 3,053 | 17 | **9** | 1 | 21 |
| lateral-flow-and-routing.md | 1,759 | 10 | 1 | 2 | 6 |
| osbs-implementation.md | 4,654 | 7 | 2 | 1 | 21 |
| neon-forcing.md | 2,514 | 4 | 1 | 1 | **60** |
| reference/modifications.md | 1,347 | 5 | 2 | 6 | 40 |
| merit-validation.md | 2,579 | 0 | 0 | 2 | 29 |
| research/neon-sites.md | 1,128 | 4 | 3 | 1 | 34 |

Three register problems a methods editor will flag immediately:

1. **Decisions attributed to "the PI" by date** ("PI direction (2026-04-25)", "PI-approved 2026-03-23", "the PI has the routing/drainage situation handled"). A paper states the decision and its rationale; who approved it and when belongs in the phase record. The docs already carry the rationale in most of these spots, so the fix is deletion, not research.
2. **Dates inside sentences.** `index.md`'s Phase E paragraph is ~190 words with eight dates. It reads as a log entry. A two-column table (decision → rationale) or a timeline figure would carry the same information.
3. **Bold as emphasis, not structure.** `neon-forcing.md` bolds 60 spans; in several sentences three fragments are bold, so none stand out (e.g. `:134`, `:142`, `:148`).

### 2.7 Duplicated passages *within* a page

- `osbs-implementation.md:132-134` ("Known limitation: depression filling") and `:306` (bold "DEM conditioning erases real closed basins…") say the same thing with the same examples (sinkholes, wetland depressions, karst dissolution features; "different hydrological framework"). Keep the longer `:306` version, which also explains the raw-HAND recovery.
- `osbs-implementation.md:168-171` (HAND binning) and `:237-243` (OSBS-specific refinements) both restate the Q25 = 0.00027/0.0003 m argument; `hand-binning-and-lake-column.md` restates it a third and fourth time (`:9`, `:141`).
- `hand-binning-and-lake-column.md` "Water masking strategy" (`:69-82`) duplicates `osbs-implementation.md:123-130` almost verbatim, including the hole-fill paragraph.

### 2.8 Unit-spacing inconsistency inside single pages

`osbs-implementation.md` writes "1m" 27 times and "1 m" 15 times; `dataset-comparison.md` 9 vs 11; `index.md` 7 vs 4. Pick one (SI style is "1 m") and apply it page-wide. Same for "90m/90 m", "4m/4 m", "64GB/64 GB".

---

## 3. Page-by-page grades and notes

Grades: **A** publication-ready prose; **B** clear, needs polish; **C** accurate but needs restructuring; **D** problematic. Two grades per page: prose / story.

### research/overview.md — B− / B
Serviceable landing page. Issues: "R ~1450 mm/yr" (`:94`) — the symbol R is never defined (precipitation); "position 3 in Fan 2019 Figure 6a" (`:97`) needs a five-word gloss (the page does gloss it on `neon-sites.md:67`, not here); *Current Status* (`:101-107`) duplicates `index.md` and will go stale first. The Fan 2019 bullet list (`:17-20`) is good and reusable in an introduction.

### research/hillslope.md — B / B+
Good primer; the *Column Organization* paragraph (`:24-38`) is the clearest statement anywhere of why the column count is a data decision, not a code decision. Issues: "The paper emphasizes…" (`:76`) — ambiguous antecedent (Fan 2019 is meant; Swenson was last named); *Analysis Scripts* and *History Output* are operational, not methods, and should stay on the site but out of the paper. The two-toggle explanation (`:151-169`) is a tight summary of the lateral-flow page and could be the paper's one-paragraph statement of the configuration.

### research/neon-sites.md — B− / B
Mixes site science (good: *Why OSBS*, hydrologic setting) with team file trees (`:117-132`). The precipitation row (`:48`) is now accurate but is a 40-word table cell; move the custom/v4 numbers to the forcing page and leave "~1450 mm/yr (DOE proposal)" here. *Reference Cases* (`:158-164`) is well done and is the right place for the retired-case paragraph. "860+ yr" for `osbs2` (`:162`) remains unverifiable from our side (no `osbs2` output directory is visible under the PI's `cime_output_root/` or `archive/`).

### swenson/index.md — C+ / B−
Right content, wrong form. The *Phases* section is eight dense paragraphs of dated clauses; the Phase E paragraph in particular (`:44`) packs four interim bin schemes, two mask designs, a failed boundary-forcing experiment, and three PI decisions into one paragraph. Findings §2.1 and §2.2 live here. The *Key Differences* table (`:17-26`) and *Guiding Principle* (`:30-32`) are excellent and paper-ready. Recommendation: convert *Phases* to a table (phase, dates, problem, outcome, record link) and move the narrative to the pages that own it.

### swenson/pysheds-porting.md — B+ / A−
Concrete and verifiable (every count checked: 4,209 / 4,220 / 4,365 lines; 12/13/2 NumPy replacements; 11/38/16/18/33 tests; 30/29/1 mutants from `.test-mutations.json`; upstream `695d8b2` 2021-12-06 and `e3c511d` 2025-08-14). Minor register: "produce garbage distances" (`:143`), "catch CRS math errors directly" — fine for docs, soften for a paper. The *Verification* triad (synthetic → MERIT regression → R6C10 smoke) is the model for how the other pages should present evidence.

### swenson/merit-validation.md — B / C+ (as methods)
As provenance it is exemplary: nine stages, each with goal → result → bug → fix, every number traceable to `progress-tracking.md` or the stage scripts (Stage 8 values confirmed in `run_merit_regression.sh:30-31`). As a methods basis it is inverted: a reader reaches the final method only at `:205` after 200 lines of debugging history. Recommendation: move *Final Results* + *Consolidated Regression Test* to the top as "What the verification establishes", and demote Stages 1–9 to a "How the baseline was reached" appendix. Also: `:227` cites `area_fraction_research.md` by bare filename; it exists at `audit/250223-pysheds_and_merit_pipeline_audit/merit_validation/` — give the path or a link.

### swenson/osbs-implementation.md — B+ / B+
The core methods page and the best-structured one: Domain → Lc → Delineation → Parameters → Resolution → Adaptations → Output → Limitations mirrors a methods section. Strengths: the k² artifact explanation (`:52-56`) is the clearest piece of scientific prose on the site; the conditioned-vs-raw HAND split (`:118-125`) is exactly what a reviewer will ask. Issues: §2.3 (PASS labels); §2.7 (in-page duplicates); the *Sensitivity* paragraph (`:95`) is one 120-word sentence chain — split it; the *Domain contiguity requirement* (`:109-113`) now tells the story correctly but is long for a limitation that no longer applies — compress to three sentences and link the phase record; unit spacing (§2.8).

### swenson/hand-binning-and-lake-column.md — B+ / A−
This page tells the project's central scientific story — why 24 bins, why raw HAND, why a lake column, why spillheight died — and tells it well. The *Why 0.25 m* derivation (`:37-48`) and *Why we accept non-monotonic bin areas* (`:50-52`) are paper-ready once the PI/date attributions are stripped. Issues: "PI direction" ×9; *Design history* (`:112-121`) is a dated bullet log (correct and useful on the site; not paper prose); the lake-column table (`:90-101`) has rationale cells of 40+ words — move the long rationales to the two bold paragraphs that follow; the Motivation/SPILLHEIGHT sections repeat the Q25 argument.

### swenson/lateral-flow-and-routing.md — A− / A
Best-written page. Claim → source lines → empirical confirmation → implication, with every cited line verified this pass (`HydrologyDrainageMod.F90:132-160`; `SoilHydrologyMod.F90:1817-1820, :1838, :2261-2263, :2265-2272, :2280, :2358, :2362-2367, :2386-2388, :2434`; `WaterFluxType.F90:506-509`; `namelist_defaults_ctsm.xml:447`; `SoilWaterMovementMod.F90:229`). The twelve-site table has twelve rows. Only §2.4 (wording) and a mild over-use of "Phase H Track A/B/C" labels, which mean nothing to an outside reader.

### swenson/dataset-comparison.md — B / B
Clear tables; the *Key Observations* are the right five points. Issues: Observation 1 (`:68`) says "much closer than previous (pre-fix) runs showed (5–25× difference)" without naming the fix (EDT → hydrological DTND, Phase A/D) — add four words; the sentence is also 80 words. Unit spacing (§2.8). The two figure captions are good.

### swenson/neon-forcing.md — B / B+
Thorough and honest (the CO₂-flux absence, the wipe-guard warning, the "inferred, not confirmed" hedge at `:76`). Issues: bold density (60 spans); the edits table (`:58-71`) is a reproducibility record, not prose — keep, but the paragraph before it already summarizes the four properties (a–d), so the paper needs only that paragraph; three "Team reference — HiPerGator" admonitions interrupt the method with paths. The *Verification* section (`:138-148`) is well-sequenced (fidelity → QC → ingestion → spinup) and paper-ready.

### reference/modifications.md — B / n/a
Reference material; not methods except item 6 (spillheight), which is now accurate and consistent with the hand-binning page. The *mpi-serial Incompatibility* essay (`:99-132`) is useful but belongs on an installation page.

### reference/resources.md — A / n/a
A link list. Nothing to grade.

---

## 4. What a methods section would take from these pages (and what it would drop)

| Methods subsection | Source page(s) | Keep | Drop or move to SI |
|---|---|---|---|
| Site and data | neon-sites (site table), osbs-implementation §Study Domain | DTM/slope products, domain selection rationale, relief statistics | tile grid mechanics, file trees |
| Software | pysheds-porting §UTM, §Verification | CRS adaptation table; three-level verification | NumPy fix counts, deprecation table |
| Verification against published data | merit-validation §Final Results, §Consolidated Regression | baseline table; what the 0.92 area-fraction gap means | Stages 1–9 narrative |
| Characteristic length scale | osbs-implementation §Lc | k² artifact; cutoff sweep table; P95 reinterpretation *disclosed* (§2.3); Lc computed at run time (§2.2) | parameter sweep history |
| Delineation and HAND | osbs-implementation §Stream Network, §HAND and DTND | dual-mask design; conditioned vs raw HAND | contiguity-requirement history |
| Binning and lake column | hand-binning (all but Design history) | 24-bin rationale, 0.25 m floor, outlier discard, lake-column table, per-rep rescale | PI/date attributions; design history |
| Model configuration | lateral-flow-and-routing, hillslope §Key Namelist Parameters | two-toggle distinction; tdepth_grc = 0 by configuration | Issue #1432 workaround (SI) |
| Forcing | neon-forcing §Source data, §Procedure, §Verification | product table; offline conversion (a–d); 2017 remediation; verification triad | edits table; environment build; HiPerGator admonitions |

---

## 5. Verification ledger (this session)

All lookups below were performed this session; "✓" means the doc text matches the source.

**Production outputs** (`output/osbs/2026-05-05_production/`)
- `production_summary.txt`: Lc 356 m (355.98 px→m), A_thresh 63,362, stream cells 136,237 (0.15%), network length 168,763 m, slope 0.006896; conditioned HAND 0–25.2 m, median 2.3 m; Q01 −6.3459, Q99 17.0201; pixels 75,967,350 → 74,448,002 (discard 1,519,348 = 2.0%); 24 bin edges; nhill_implicit 533.7; lake 20,765 m² per rep, 11,082,394 px raw, width 90.4 m per rep, half-perimeter 48,270 m; 803 s ✓ (`osbs-implementation.md:93-95, :120-121, :195`; `hand-binning:58, :63, :75, :97-98, :129-131`)
- NetCDF `c260505`: land HAND −5.126 … +12.460 m; distance 2.592 … 352.6 m; lake 1.296 m; width 555.6 → 268.5 m; land slope median 0.048, max 0.067; lake 12.31 % of per-rep area; flood-zone bins 0.73–2.90 %; upland 3.60–7.99 %; land area Σ 0.148 km²; stream depth 1.519 / width 59.23 / slope 0.006896; bedrock 0; aspect in radians ✓ (`dataset-comparison:55-60, :93, :110`; `hillslope:36`; `hand-binning:95-96, :121, :149`)
- `hillslope_params.json` metadata: perimeter 96,540 m; lake width 48,270 m ✓
- Swenson `hillslopes_osbs_c240416.nc`: HAND 0.174–8.144; distance 67.3–541.4; depth 0.269; width 4.414; slope 0.002327 ✓ (`dataset-comparison:40-42, :56-57`)
- `data/mosaics/production/dtm.tif`: 0 nodata; 23.53–55.50 m; p1/p99 25.04/51.14 ✓ (`osbs-implementation:31`)
- `output/osbs/slope_aspect_comparison/results.json`: 90 tiles; slope r 0.909 (0.780–0.980), bias 0.0084; aspect circular r 0.844; mean angular diff 21.7° ✓ (`osbs-implementation:131`; `index:44`). *Sign convention of the bias (which product is lower) not confirmed from a script — the comparison script is not in the repo.*
- `output/osbs/smoke_tests/lc_physical_validation/validation_results.json`: at 356 m p95/Lc 1.093, mean/Lc² 0.877, max/Lc 2.616; script verdicts MARGINAL / FAIL / overall FAIL with primary 300 → §2.3

**Pipeline code**
- `scripts/osbs/hillslope_params.py:184-205` bin edges match the width table ✓ (`hand-binning:24-33`)
- `scripts/osbs/run_pipeline.py:85, :928, :938` — only `MIN_WAVELENGTH = 20` is fixed; Lc is a run-time FFT output → §2.2
- `scripts/merit_validation/merit_regression.py:57` regions 500/1000/3000 + full; `:829` positivity + finite filter for height/distance/slope/width; circular aspect; `results.json` Lc 766/760/754/2,679, median 763, all six parameters PASS ✓ (`merit-validation:9, :45, :293-294`)
- `data/merit/n30w095_dem.tif` present; `merit_regression.py:44` reads it ✓

**CTSM fork** (`/blue/gerber/cdevaneprugh/ctsm5.3`, HEAD `5b930dc8d`)
- Tag `ctsm5.3.085` → commit `fde1fc622` ✓; `dc3aa5ddc` exists (2026-01-21); `src/` differs only by four `CLAUDE.md` files → §2.4
- `HydrologyDrainageMod.F90:132-160`, `SoilHydrologyMod.F90:1817-1820, :1838, :2261-2263, :2265-2272, :2280, :2358, :2362-2367, :2386-2388, :2434`, `WaterFluxType.F90:506-509` (QLATFLOWOUT under `use_hillslope`), `namelist_defaults_ctsm.xml:447` (lbc = 2), `SoilWaterMovementMod.F90:229` ✓
- `4a0319fee` 2026-01-12 spillheight namelist entry ✓; `gen_mksurfdata_build:170` flags ✓; `mksurfdata.F90:271, :295, :328` ✓; `config_machines.xml` gcc/14.2.0, openmpi/5.0.7, esmf/8.8.1 ✓; `tools/site_and_regional/` has `run_tower`, no `run_neon` ✓; `cime-config.md#pio-shared-library` anchor exists ✓

**pysheds fork** (`uf-development`)
- `7b20e8e` 2026-01-22, +4,220 lines; `pgrid.py` now 4,365; Swenson's copy 4,209; upstream `695d8b2` 2021-12-06 / `e3c511d` 2025-08-14 ✓
- `np.warnings` 12, `np.bool` 13, `np.float` 2 in Swenson's `pgrid.py` ✓
- `tests/` five files; `data/synthetic_valley/` three generators + three DEMs ✓
- `79db91f` 2026-02-17 and `.test-mutations.json`: 30 mutations, 29 killed, 1 equivalent, 96.7 % raw / 100 % effective ✓; Phase A `:383` 82 tests on 2026-02-18 ✓; R6C10 14/14 checks ✓

**NCAR-NEON fork and Phase I**
- `git diff --stat 43a0cf4..07786dd`: +77/−31 in `flow.api.clm.R` ✓; `renv.lock` 195 packages, R 4.0.5 ✓
- `download_raw.R`: 10 products (DP1.00006.001 explicitly dropped in a comment) ✓
- `scripts/neon_forcing/output/summary.txt`: both-measured RMS PRECTmms 3.96e-7, WIND 4.55e-6; five variables RMS 0; precip 1,810 vs 1,759 mm (+2.9 %) ✓
- `phases/I-neon-forcing.md:1159-1167`: r = 0.96, ~2 %, 35 months (2018–2021), ~1330–1350 mm/yr, 163 mm on 2017-09-10 ✓; 1,063 zips ✓; `docs/neon-data-products.md:26-32` product start dates ✓

**Phase and audit records**
- Phase B: 189 M px / 37.5 % nodata OOM; 58 GB full-pipeline peak; 29.2 GB routing-only ✓
- Phase C: 620× / (300/8)² ≈ 1400× ✓; 20-configuration sweep, 1.7 m max deviation ✓; psharp 11.56/8.95 and 3.95/3.67/3.37/3.13 ✓; cutoff table ✓; conclusion "~300 m working value" → §2.2
- Phase E: Q25 = 0.00027 (audit 260310 `:223-227`) ✓; 103 NWI features ✓; ~800 → 222 K catchments ✓; 22 % noise-floor pixels ✓; slope/aspect PI approval 2026-03-23 ✓
- Phase E.5: density 0.62 / 13.9 / 0.92 km²/m ✓
- Phase F: lake max 5.78 m at yr 107, drained to 2.5 m ✓
- Phase H: QDRAI −4.32e-5/1.47e-5 vs −4.15e-5/1.43e-5; QRUNOFF −1.156e-4/1.992e-4; ZWT 0.011 → 8.6 m ✓
- `docs/lake-column-ctsm-audit.md:438`: NWI mean −2.53 m; Lee 2.64 / ours 3.33 m ✓
- `progress-tracking.md`: 58.5 % largest component ✓; Stage 1 3.4 min ✓; Stage 8 values in `run_merit_regression.sh:30-31` ✓
- `docs/papers/DOE_Grant_Summary.md:11`: 2023–2026 ✓
- Cases: `osbs2.branch.v2` 2025-12-03; `osbs2.branch.spillheight` 2025-12-15 (SourceMods same day); `osbs4.branch.v1/v2` 2026-04-14/15; `osbs5.swenson.spinup` 2026-05-05; `osbs.swenson.spinup` 2026-05-11; PI `osbs2_1` 2025-11-05 → §2.1

**Build**: `mkdocs build --strict` into the scratchpad succeeds; every in-page and cross-page anchor in the 13 pages resolves.

**Still unverifiable from our side** (unchanged from audit 1, plus one): `osbs2` "860+ yr" (`neon-sites:162`); NEON ≤ 0.15 m RMSE and ATBD number (external); DiscussCESM attribution (`lateral-flow:190`); `ctsm5.4.021` / `5.4.025` release tags (`modifications:42, :68`); slope-bias sign (`osbs-implementation:131`).

---

## 6. Recommended order of work

1. **Fix the two story points that are wrong, not just unpolished:** §2.1 (origin) and §2.2 (Lc) — three sentences each on `index.md`, one on `osbs-implementation.md`.
2. **Reframe the plausibility checks** (§2.3) — one paragraph.
3. **One-word wording fix** (§2.4).
4. When the planned reorganization happens: consolidate the four Phase F narrations and the three water-mask/spillheight tellings (§2.5, §2.7); strip PI/date attributions from the pages that will feed the paper (§2.6); fix unit spacing page-wide (§2.8); invert `merit-validation.md` (final method first, stages as appendix).
5. Tools-side (not this repo): Phase C's summary still reads "Final Lc ~300 m / A_thresh 45,000"; a one-line note there that the pipeline's production value is 356 m would close the loop the docs now paper over.
