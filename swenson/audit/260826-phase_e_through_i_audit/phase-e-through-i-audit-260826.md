# Swenson Hillslope — Completeness & Correctness Audit (Phase E.5 → I)

**Date:** 2026-08-26
**Auditor:** Claude Code (read-only)
**Scope:** all work in `$SWENSON` since the last substantive audit (2026-03-10, mid-Phase-D)
**Repo state at audit:** `hpg-esm-tools` `main` @ `81a77f5`, in sync with origin, working tree clean
**Method:** 6 subagents (3 scoping + 3 deep-verification) + direct hard-evidence commands + one
independent numeric recompute. Nothing was modified; this report file is the only artifact created.

---

## 1. Executive summary

**The recorded state is trustworthy.** Across everything new since 2026-03-10 — Phase E.5/E.6
(24-bin + lake-column pipeline rewrite), F (validate/deploy), G (lake representation), H (routing),
and the large self-contained Phase I (NEON atmospheric forcing) — **every "we did X" claim that was
checkable against software was found to be backed by software that actually does X**, frequently
*more* correctly or completely than the prose states. There was **no instance of "claimed-done but
actually-undone"** — the specific failure mode this audit was commissioned to catch.

- **No correctness bugs** were found in any script, the pipeline, the fork edits, or the produced data.
- The headline Phase I scientific claim (**AD spinup CONVERGED**) was **independently recomputed** from
  the archived annual history and **corroborated** — drift < 0.25 % on all carbon pools (details §3.1).
- The Phase I engineering deliverable is real and reproducible: 101-file custom forcing dataset,
  QC-clean, ingestion-smoke-passed, AD-converged, with an I7 recipe that is **byte-for-byte consistent
  with the actual running case's namelist and XML**.
- All 8 cited CTSM source locations behind the load-bearing "lateral flow runs under `use_hillslope`"
  claim were **confirmed** in the committed fork.

**What the audit *did* surface** is a short list of **documentation-precision drift** (a few stale
numbers/labels in STATUS.md / CLAUDE.md, all explained by later phases post-dating the values) plus
**two minor hygiene items** (an uncommitted `ccs_config` submodule edit; a two-comment drift between
the two `spatial_scale.py` copies), and the **expected public-docs lag** at `hpg-esm-docs` — which the
user has already flagged as acceptable to resolve later. None of these is a correctness or completeness
defect in the code or the delivered data.

**Overall grade: green.** The software substantiates the documentation. Phase I is genuinely
engineering-complete; the one open item (I8, PI adoption) is correctly PI-gated. Recommended follow-ups
are ~2 one-line local doc corrections plus a public-docs refresh when convenient (§7).

---

## 2. Scope & baseline

The last real audit was **2026-03-10** (`audit/260310-osbs_pipeline_and_docs/`), which verified the OSBS
pipeline's equations and code against Swenson (2025) while Phase D was still closing. **91 commits**
landed between then and this audit, spanning:

| Area | What changed since 2026-03-10 |
|---|---|
| Phase E / E.5 / E.6 | NEON DP3.30025 slope/aspect adopted; NWI dual-mask water masking; **24-bin TAI scheme + lake column** (a substantial `run_pipeline.py` rewrite the 2026-03-10 audit predates); NWI hole-fill |
| Phase F / G | Routing-off AD-spinup validation; submerged lake column; 2026-08-19 reconciliation (bridge-zone resolved by PI, file un-frozen) |
| Phase H | Track A mesh-mode workaround (CTSM #1432); routing-gate source audit; Tracks B/C retired 2026-08-19 |
| **Phase I** | **NEON atmospheric forcing — the largest new body of work (~30 commits): pipeline fork, full dataset, ingestion smoke, AD convergence, config recipe** |

**What was checked:** the three forks (`ncar-neon`, `pysheds_fork`, `ctsm5.3`), all Phase I scripts and
the offline generator fork diff, the OSBS pipeline source, the produced forcing dataset (101 + 84 + 1063
raw), the production hillslope NetCDF (values), the archived AD-spinup output, the CTSM source-trace
citations, the operative case namelists, and the local (`$SWENSON`) vs public (`hpg-esm-docs`) docs.

**What was *not* done (honesty note):** the operative CTSM cases live outside the repo under `$CASES` /
the archive; this audit verified their **archived output and namelists** but did not re-run any model.
The convergence result in §3.1 is recomputed from archived annual history, not from a fresh run.

---

## 3. Phase-by-phase verification (claim → evidence → verdict)

### 3.1 Phase I — NEON atmospheric forcing  *(deepest scrutiny; largest new work)*

**Dataset artifacts** — all confirmed on disk:

| Claim | Evidence | Verdict |
|---|---|---|
| Custom forcing = 101 monthly NetCDFs, 2017-02 → 2025-06 | `data/datm/neon_OSBS/custom/OSBS/atm/`: 101 `OSBS_atm_*.nc`, first `2017-02`, last `2025-06`, 16 MB | CONFIRMED |
| 2017 precip splice landed (secondary gauge, fqc=5, backup) | `pre_splice_backup/` holds exactly the 6 gap months; `2017-09` = 1440/1440 fqc==5, `2017-07` = 206 fqc==5 steps (matches the documented outage geometry to the step); no −9999/NaN | CONFIRMED (empirical) |
| Pre-built v4 = 84 files, 2018-01 → 2024-12 | `data/datm/neon_OSBS/v4/OSBS/`: 84 `.nc`, 13 MB | CONFIRMED |
| Raw archive = ~11 GB, 1063 zips | `/blue/gerber/earth_models/neon/raw/OSBS`: 11 GB, 1063 zips | CONFIRMED |

**Offline generator fork** (`/blue/gerber/cdevaneprugh/ncar-neon`, branch `uf-osbs`):

| Claim | Evidence | Verdict |
|---|---|---|
| HEAD `07786dd`, parent `2d30ebb`, both touch `flow.api.clm.R` | `git log`; 2 commits ahead of merge-base `43a0cf4` | CONFIRMED |
| All offline edits present (offline gate, DirDnld repoint, edits C/S/T/P2, DP1.00006→DP1.00045) | `git diff 43a0cf4 uf-osbs` — every claimed hunk verified individually | CONFIRMED |
| **Commits "unpushed"** (STATUS.md:279) | `uf-osbs` == `origin/uf-osbs`, **0 ahead / 0 behind → PUSHED** | **stale doc** — see §4/F5 |

**Script logic** — verified by reading the code, not just confirming file existence:

| Script | Claim | Verdict |
|---|---|---|
| `neon_forcing_qc.py` | structural (ntime==days×48, fqc present, NaN==0, monotonic) + sanity bounds + gap-fill tallies + PASS/FAIL exit | CONFIRMED (all checks real & correct) |
| `splice_2017_precip.py` | UTC-aligned, writes only PRECTmms + fqc, backs up first, dry-run default | CONFIRMED (cannot corrupt other vars) |
| `neon_v4_regression.py` | fqc-partitioned RMS/corr/bias, both-measured threshold, exit 0/1 | CONFIRMED (partition is genuine) |
| output verdicts | I4 `RESULT: PASS` (both-measured RMS≈0 all 7 vars; precip +2.9%); I5 QC `RESULT: PASS` (101 files, 0 NaN) | CONFIRMED (quoted from `output/*.txt`/`*.json`) |

**Ingestion smoke → production requirement:** the `dtlimit=-1` fix for the CDEPS cycle-wrap crash
(`dshr_strdata_mod.F90:1050`) is real and **present in the live case** on both NEON streams
(`osbs.swenson.neon.spinup` streams file, lines 147 & 253). CONFIRMED.

**AD-spinup convergence — independently recomputed** from the 196 archived `h0a` annual files
(`archive/osbs.swenson.neon.spinup/lnd/hist/`; yr-180 handoff restart `r.2197-02-01` present):

| Pool | Recomputed 20-yr block drift (last vs 80 yr earlier) | Last-50yr trend | Limit-cycle amplitude | STATUS.md logged | Verdict |
|---|---|---|---|---|---|
| TOTECOSYSC | **0.09 %** | 0.05 %/decade | 12.8 % p-p (±6.4 %) | 0.15 % | CONVERGED ✓ |
| TOTSOMC | **0.23 %** | 0.04 %/decade | 8.0 % p-p (±4.0 %) | 0.48 % | CONVERGED ✓ |
| TOTVEGC | **0.03 %** | 0.06 %/decade | 23.0 % p-p (±11.5 %) | 0.51 % | CONVERGED ✓ |

The verdict is **robust** — my recompute is even *smaller* than the logged figures (block-boundary
choice differs), and both agree emphatically: all pools flat, drift well under 1 %, the documented
"~90-yr ±6 % AD limit cycle" matches TOTECOSYSC's 12.8 % peak-to-trough. **Precision note:** the
archive contains **195 complete sim-years (2017→2211)** + a partial 2212 stub, so STATUS.md's
"~180 continuous yr" *understates* the actual run (180 is the handoff-restart year, not the run length).

**I7 recipe** (`docs/neon-forcing-case-recipe.md`): every named setting (compset `I1PtClm60Bgc`,
`PTS_LAT/LON` 29.689282/278.006569, `RUN_STARTDATE=2017-02-01`, `DATM_YR_*`=2017/2025/2017,
`dtlimit=-1` on both streams, `taxmode=cycle`, `MPILIB=openmpi`) is **byte-for-byte consistent with the
running case's env XML and streams file**. CONFIRMED.

**Phase I verdict: fully substantiated.** The scripts implement the documented logic (not shells); the
data exists at the claimed shape; the recorded PASS verdicts match the output files; the convergence
claim survives independent recompute; the recipe matches the live case.

### 3.2 Phase E.5 / E.6 — pipeline rewrite  *(changed heavily since the 2026-03-10 audit)*

All seven locked decisions were traced into `scripts/osbs/run_pipeline.py` and cross-checked against
the produced NetCDF (`output/osbs/2026-05-05_production/hillslopes_osbs_production_c260505.nc`):

| Locked decision | Code (run_pipeline.py / hillslope_params.py) | NetCDF | Verdict |
|---|---|---|---|
| 24 bins = 12 FZ + 12 upland, 0.25 m floor | `N_LAND_BINS=24`; edge list crosses 0.0 at index 12; min spacing 0.25 m | 12 land cols elev<0, 12 ≥0 | CONFIRMED |
| Q01/Q99 raw-HAND true-discard | `:1265-1267` percentile mask on raw HAND | — | CONFIRMED |
| Lake at chain index 1, 25 cols | lake element prepended `:1615`; `column_index[i]=i+1` | `nmaxhillcol=25`, col-1 = lake | CONFIRMED |
| 6 lake params (elev −6, dist 0.5×bin-1, area Σmask, width 0.5×perim, slope/aspect 0) | `:1531-1611` | col-1: elev −6, slope 0, aspect 0, largest area | CONFIRMED |
| Per-rep rescale (nhill_implicit≈533, wtlunit≈12.3%) | `:1567-1577` | 533.706; 12.31 % | CONFIRMED |
| Dual-mask + NEON DP3.30025 slope/aspect | wide mask `:1108` → `compute_hand`; NEON slope `np.tan(deg2rad)` `:871` | — | CONFIRMED |
| Core Swenson eqs (A_thresh=0.5·Lc², w=−∂A/∂d) | `:940`; `fit_trapezoidal_width` | `accumulation_threshold=63362` | CONFIRMED |

One intentional, self-documented deviation from the MERIT/Swenson form (the `mean(hand)≤0` bin-skip
guard is disabled for raw-HAND binning) is correct by design, not a bug.

**Two stale values found in STATUS.md's scientific-decisions table** (code correct; the table records
pre-E.6 snapshots — see §4/F1).

### 3.3 Phase F / G / H — verify what's verifiable  *(operative cases live outside the repo)*

| Check | Evidence | Verdict |
|---|---|---|
| Routing-gate source trace (8 CTSM citations: lateral flow ungated by routing; stream-side gated) | All 8 confirmed in `ctsm5.3` @ `uf-ctsm5.3.085` (one 1-line drift, 2086→2087) | CONFIRMED |
| Phase H Track A artifacts | `make_osbs_scrip.py` + `output/mesh/osbs_{scrip,mesh}_90km2_c260512.nc`; A4 `grc%area=90.006 km²` recorded | CONFIRMED |
| MERIT regression still validates pysheds | `output/summary.txt` `RESULT: PASS`, all 6 params + Lc 763.0, at current pysheds HEAD `ed72724` | CONFIRMED |
| Case hydrology SourceMods (the "6-file set") | `$CASES/osbs.swenson.spinup/SourceMods/src.clm/` = 5 `.F90` + README | CONFIRMED |
| F/G/H headers vs STATUS.md table | F "Complete routing-off / bridge-zone resolved / un-frozen", G "Complete Stage-1", H "Track A complete, B/C retired" — all agree | CONFIRMED (consistent) |

---

## 4. Software-vs-doc mismatches (findings, severity-ranked)

All findings below are **documentation drift or hygiene** — in every case the **software is correct**
and, where relevant, *more* current than the doc. Suggested fixes are **not executed** (read-only audit).

**F1 — MEDIUM — two stale values in STATUS.md's "Scientific decisions (locked)" table.**
The production NetCDF (post-E.6) no longer matches two locked values that were set pre-E.6:
- `STATUS.md:49` — **Q99 = +17.46 m**; production computed **+17.02 m** (`q99_cutoff_m=17.0201`). (Q01 matches.)
- `STATUS.md:54` — lake `hill_area ≈ 10.68 km²`; production **11.08 km²** (`lake_area_total_m2=11,082,394`).
  The ~0.40 km² / ~402K-pixel gap **is exactly the E.6 `binary_fill_holes` fix** (STATUS.md:137,
  "400K hole pixels fixed"). The cutoffs/areas are computed dynamically each run, so the code is right;
  the *locked-decision table* (meant to be authoritative) records the pre-E.6 snapshot.
  → *Suggest:* update both to the production values, or annotate "(pre-E.6 snapshot; production = …)".

**F2 — LOW-MED — CTSM fork not clean (uncommitted submodule edit).**
`ctsm5.3` superproject is clean, but the `ccs_config` submodule (branch `uf-hipergator`) has one
uncommitted change: `machines/hipergator/config_batch.xml` (M). It is a machine batch-config edit, not
a CTSM source change, and does not affect any Phase-verified code. → *Suggest:* commit or revert it so
the fork state is reproducible; decide whether the batch tweak belongs in the fork.

**F3 — LOW — the two `spatial_scale.py` copies are no longer byte-identical.**
`md5`: osbs `0526dcf3…` vs merit_validation `e23e9449…`. The **code is identical**; the diff is exactly
two comment hunks (osbs lines 76-77 & 745-746 point at `audit/260627-cleanup/docs/ns-aspect-bug.md`
after the 2026-06-27 ref-refresh; the merit copy still says `STATUS.md #4`, itself now a defunct
reference). `hillslope_params.py` remains identical (`9975cf20…`). This is exactly the drift
`CLAUDE.md`'s own "no automated sync" caveat warns about, but `CLAUDE.md:174` still asserts a
"co-located **byte-identical** sibling." → *Suggest:* re-sync the two comments, or soften CLAUDE.md to
"functionally identical (comments may drift)".

**F4 — LOW / cosmetic — "~180 continuous sim-yr" understates the AD run.**
The archive holds 195 complete annual records (2017→2211); "~180" is the handoff-restart year, not the
run length. → *Suggest:* say "~195 sim-yr; clean yr-180 restart handed off."

**F5 — COSMETIC — two "unpushed" change-log notes are now superseded.**
`STATUS.md:276` ("Scripts + docs unpushed") and `:279` ("Commits (unpushed): fork … hpg-esm-tools …")
were true when written; everything is now pushed (fork 0/0; `main` synced). This is normal dated-log
behavior, not an error. → *Suggest (optional):* append "(since pushed)" if it risks confusing a scanner.

**F6 — COSMETIC — two label nits, no behavior impact.**
`neon_forcing_qc.py` docstring says "8 physical vars" but `PHYS_VARS` has 7 (checks 7, correct); the
fork's partition-resilience edit is labeled "UF:" in-code rather than the doc's "P2". → *Suggest:* fix
the docstring count if touched; ignore otherwise.

---

## 5. Public-docs (`hpg-esm-docs`) gap catalog  *(note-only — user resolves later)*

The public MkDocs site is a **~2026-07-09/10 snapshot, ~6 weeks behind STATUS.md (2026-08-19)**. It is
internally consistent and remains **accurate on the hillslope-pipeline mechanics** (24-bin/25-column
scheme, lake column, Lc=356, the routing-gate audit). The gaps are the entire Phase I arc and the
2026-08-19 F/H reconciliation. The user has already accepted docs-lag as OK; catalogued here for the
later batch, severity relative to a public reader.

| # | Sev | Location | Issue |
|---|---|---|---|
| G1 | HIGH | (whole site) | **No Phase I / NEON-forcing page exists** — the dataset, ingestion smoke, AD convergence, and the I7 recipe have zero public coverage |
| G2 | HIGH | `research/neon-sites.md:131-133` | **Statement is now INVERTED** — "NEON-native atmospheric forcing is a future direction, not the current path — the pre-built `run_tower` NEON forcing was tested and found insufficient (per PI, a manual NCAR-NEON pipeline is required)". Phase I *built* that pipeline; it is engineering-complete. Single most misleading public line. |
| G3 | MED | `swenson/index.md`, `research/overview.md` | "production hillslope file is **frozen**" — un-frozen 2026-07-15 |
| G4 | MED | `swenson/index.md`, `research/overview.md` | **bridge-zone** framed as an open PI-investigation question — resolved by the PI 2026-08-19 |
| G5 | MED | `swenson/index.md`, `lateral-flow-and-routing.md` | **Phase H** framed as "contingent / may not be pursued" — Tracks B/C retired 2026-08-19 |
| G6 | LOW | all `swenson/` + `research/` pages | freshness anchor: last committed 2026-07-09/10; phase-status labels are a May snapshot |

**Resolved (NOT a gap):** the `DP3.30024.001` (LIDAR DEM) vs `DP3.30025.001` (slope/aspect) codes in
the public docs were cross-checked against `data/neon/README.md` — they are **distinct NEON products**
and every public reference cites the correct one. No action.

---

## 6. Code completeness / correctness notes  *(light — refactoring out of scope)*

- **No correctness bugs found** in any audited script, the pipeline, the fork edits, or the produced data.
- Path handling is robust — the Python tools self-locate `$SWENSON` rather than hardcoding. The one
  exception, `smoke_compare_v4.py`, hardcodes ephemeral case-archive paths, but its docstring flags
  them and both archives still exist. Not a defect.
- No stray `TODO`/`FIXME`, no dead code of concern in the Phase I toolset.
- The one intentional pipeline deviation from the MERIT/Swenson reference form (§3.2) is documented and
  correct for the raw-HAND design.

---

## 7. Suggestions (prioritized; none executed)

1. **Now — two one-line local doc fixes (F1).** Update or annotate `STATUS.md:49` (Q99 → +17.02 m) and
   `STATUS.md:54` (lake area → 11.08 km²) as post-E.6 production values. These live in the authoritative
   locked-decisions table, so they carry the most weight.
2. **Now — soften or re-sync (F3).** Either re-sync the two `spatial_scale.py` comments or change
   `CLAUDE.md:174` "byte-identical" → "functionally identical." Also drop the merit copy's defunct
   "STATUS.md #4" reference.
3. **Soon — fork hygiene (F2).** Resolve the `ccs_config` submodule's uncommitted `config_batch.xml`.
4. **When you tackle public docs (your later batch).** Add a Phase I page mirroring
   `neon-forcing-case-recipe.md`, then flip G2 first (it asserts the opposite of reality), then
   G3/G4/G5 status updates.
5. **Optional/cosmetic.** F4 (~195 yr), F5 ("since pushed" notes), F6 (docstring count).

---

## 8. Overall assessment

The Swenson project's documentation is a **reliable record of what the software actually does.** This
audit checked the full un-audited span (Phase E.5 → I) against the code, the forks, the produced data,
the archived model output, and the CTSM source — and found the recorded engineering **substantiated in
every checkable claim**, with the produced artifacts often *more* correct/current than the prose. The
one failure mode the audit targeted — "documented as done but not actually built" — **did not occur**.

Phase I is genuinely **engineering-complete and reproducible**: the custom forcing dataset exists at the
claimed shape, is QC-clean, drives CTSM end-to-end, and produced an independently-verified converged AD
spinup, with a recipe that matches the live case exactly. The sole open project item, **I8 (PI
adoption)**, is correctly PI-gated and not our engineering work.

The follow-up list is short and low-risk: **two one-line numeric corrections** to the STATUS locked-
decisions table, a **comment re-sync / claim-softening**, one **submodule commit**, and — when
convenient — a **public-docs refresh** (a Phase I page plus five stale F/H/forcing statements). None
blocks moving on to the next part of the project.

---

*Appendix — reproducibility.* Every finding above cites a `file:line`, an `md5`/`git`/`ncdump` output,
or a recomputed number. The AD-drift recompute script used in §3.1 read only the archived `h0a` files
and wrote nothing to the project; its figures are re-derivable from
`archive/osbs.swenson.neon.spinup/lnd/hist/*.h0a.*.nc` (TOTECOSYSC/TOTSOMC/TOTVEGC, 20-yr end-anchored
block means, last block vs 80 yr earlier).
