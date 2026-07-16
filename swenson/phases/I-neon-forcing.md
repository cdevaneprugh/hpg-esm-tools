# Phase I: NEON Atmospheric Forcing

Status: **Not started** — adopt pre-built NCAR-NEON tower forcing (2018–2024, v4)
for the OSBS case to replace CRUNCEPv7; custom NEON→DATM pipeline documented as a
PI-gated contingency, not active work.
Depends on: — (independent of the hillslope track A–H)
Blocks: — (input-quality upgrade; does not gate routing on/off decisions)

## Problem

The operative OSBS case (`osbs.swenson.spinup`) is driven by **CRUNCEPv7** — a
coarse 0.5° global reanalysis, bilinearly interpolated from the global grid to
the OSBS point. Two consequences:

- It does not reflect site meteorology; the OSBS flux-tower record is not used
  at all.
- The current DATM stream config reads only `TBOT/WIND/QBOT/PSRF` from the TPQW
  stream and lets DATM **derive** downward longwave internally — measured `FLDS`
  is discarded.

NEON operates a flux tower at OSBS (site OSBS, domain D03) with a continuous
instrument record. The goal of this phase is to drive the OSBS case with
NEON-derived DATM atmospheric forcing — real site meteorology (including
measured longwave) instead of interpolated global reanalysis.

## Key context (2026-07-15)

Research this session (NEON API + on-disk verification) reshaped the effort. The
central finding: **gap-filled, CTSM-ready NEON forcing for OSBS already exists
through 2024-12** — the "process the data" work is done upstream by NEON.

- **The "2018–2021 only" premise was wrong.** That was a namelist cap
  (`NEONVERSION=v2`, `DATM_YR_END=2021`) in the CTSM NEON usermods, not a data
  limit. NCAR-NEON publishes processed forcing in four versions (server
  `listing.csv`):

  | Version | Coverage | Files |
  |---|---|---|
  | v1 | 2018-01 → 2021-09 | 45 |
  | v2 | 2018-01 → 2022-04 | 52 |
  | v3 | 2018-01 → 2024-06 | 78 |
  | **v4** | **2018-01 → 2024-12** | **84** |

  The **v3 set (78 files) is already on disk** from a prior run at
  `/blue/gerber/cdevaneprugh/cases/run_tower.OSBS.250925-094358/OSBS.transient/run/inputdata/atm/cdeps/v3/OSBS/`.

- **File spec our forcing must match** (verified via `ncdump -h` of a real
  `OSBS_atm_2018-01.nc`): one combined NetCDF **per month**,
  `OSBS_atm_YYYY-MM.nc`, single-point `(time, lat=1, lon=1)`, `time` in
  days-since-month-start on a **`gregorian`** calendar, half-hourly (48/day),
  `_FillValue=-9999`, variables all `double(time,lat,lon)`:

  | Var | Units | Meaning |
  |---|---|---|
  | `FSDS` | W/m² | incident shortwave |
  | `FLDS` | W/m² | incident longwave (**measured**) |
  | `PRECTmms` | mm/s | precipitation |
  | `TBOT` | K | air temperature |
  | `RH` | % | **relative humidity** (not CRUNCEP's `QBOT` specific humidity) |
  | `WIND` | m/s | wind speed |
  | `PSRF` | Pa | surface pressure |
  | `ZBOT` | m | observation height |

  Plus informational `<VAR>_fqc` QC flags (`method_gap-fill`: 0=none,
  1=regression, 2–4=ReddyProc methods A/B/C). CTSM reads only the 8 physical
  vars. Files carry `created_with = "flow.api.clm.R"` — the external NCAR-NEON R
  pipeline (ReddyProc gap-fill) — confirming generation is upstream, not in-tree.

- **Integration is a config change, not new machinery.** The current case
  **cycles** a fixed 20-year CRUNCEP block (1901–1920,
  `DATM_YR_START/END/ALIGN = 1901/1920/1`, `taxmode=cycle`) ~30× over the 600-yr
  spinup — no blending. Cycling the ~7-yr NEON block is structurally identical.
  The canonical NEON path uses compset `I1PtClm60Bgc` / `IHist1PtClm60Bgc`,
  `DATM_MODE=1PT`, and usermods `cime_config/usermods_dirs/clm/NEON/OSBS/`;
  `buildnml` reads `listing.csv` and downloads monthly files from
  `storage.neonscience.org/neon-ncar/NEON/atm/cdeps/`. **Caveat:** the fork's
  `buildnml` `_get_neon_data_availability` only searches versions v3/v2/v1 —
  reaching v4 (2024-12) needs a one-line patch or an explicit `NEONVERSION=v4`.

- **The prior "run_tower is insufficient" verdict is attributable to the v2
  cap.** No specific gap that pre-built v4 cannot fill has been identified.
  Confirming this with the PI (task I8) gates any custom-pipeline work.

## Approach

Two tracks, sequenced pre-built-first (user decision 2026-07-15):

- **Track 1 (primary) — adopt/validate pre-built v4.** Verify the research
  record, obtain the v4 files, wire them into a test case that keeps our custom
  hillslope surface data, and validate against the CRUNCEP baseline. Nearly free
  (data exists, CTSM downloads it) and de-risking.
- **Track 2 (contingency, PI-gated) — custom pipeline.** The from-scratch
  NEON-API → ReddyProc → NetCDF pipeline. **Dormant.** Fires only if a PI
  conversation (I8) identifies a concrete gap v4 can't fill (pre-2018 years,
  time-varying CO₂, gap-fill quality, or post-2024 recency).

## Tasks

### Track 1 — pre-built forcing (primary)

- [ ] **I1. Formal claims verification.** Work the claims-to-verify checklist
  (below) against the NEON API (`.../products/<DP>/sites/OSBS/RELEASE-2026`) and
  on-disk evidence. Record outcomes. Correct the `docs/neon-data-products.md`
  weighing-gauge claim (the API shows DP1.00044.001 "Precipitation – weighing
  gauge" present 2016-09→, contradicting "NOT installed at OSBS").
- [ ] **I2. Obtain v4 files.** Reuse the on-disk v3 set for a first test; fetch
  the v4 remainder (2024-07 … 2024-12) via the `listing.csv` URLs. Decide
  storage: per-case run dir vs a curated `data/datm/neon_OSBS/`.
- [ ] **I3. Resolve the `buildnml` v4 gap.** Patch
  `_get_neon_data_availability`'s version list to include v4, or set
  `NEONVERSION=v4` explicitly; verify resolution to 2024-12.
- [ ] **I4. Wire NEON forcing into a test case that keeps our custom hillslope
  surfdata + `hillslope_file`.** NEON usermods stage their own NEON surface
  dataset — the integration point is NEON DATM streams + our CLM
  surface/hillslope inputs. Ensure RH is read and measured FLDS is used.
- [ ] **I5. Caveat handling.** Inspect/flag the **2018 TBOT anomaly (NCAR-NEON
  Issue #34)** before use; verify `ZBOT` matches the OSBS instrument height.
- [ ] **I6. Test.** Build + run a short (≈5-yr) NEON-forced case; confirm from
  `datm.log`/`lnd.log` that `OSBS_atm_*.nc` are the active streams, FLDS is
  measured (not derived), RH is ingested; sanity-compare TBOT/PRECT/FLDS against
  the CRUNCEP baseline (reuse `case.analyzer` / `/case-check`).
- [ ] **I7. PI decision — cycle vs blend** for the 600-yr spinup: cycle the
  ~7-yr NEON block, or blend (long reanalysis for AD/post-AD spinup, NEON for the
  final transient/evaluation run, as successive cases). Document the choice.

### Track 2 — custom pipeline (contingency, PI-gated — do NOT start without PI go-ahead)

- [ ] **I8. PI conversation.** Does pre-built v4 suffice, or is there a concrete
  gap (pre-2018 years / CO₂ / gap-fill quality / post-2024 recency)? Record the
  verdict — this gates I9/I10.
- [ ] **I9.** *(only if I8 finds a gap)* Install NCAR-NEON per the three paths in
  `docs/neon-data-products.md` (HiPerGator R module + `renv` recommended).
- [ ] **I10.** *(only if needed)* Produce custom `OSBS_atm_YYYY-MM.nc` matching
  the I2 / Key-context spec (8 vars, units, gregorian half-hourly, `-9999` fill).

### Claims-to-verify checklist (for I1)

Per-product OSBS availability (verify each via NEON API, RELEASE-2026):

- [ ] 7 Tier-1 DPs — TBOT `DP1.00002.001`, radiation `DP1.00023.001`, pressure
  `DP1.00004.001`, wind `DP1.00001.001` (all ~2014–), RH `DP1.00098.001`
  (2015-06–), precip-tipping `DP1.00045.001` (2016-08–).
- [ ] **`DP1.00044.001` weighing-gauge precip** — doc says "NOT installed";
  **API shows present 2016-09→**. Correct the doc.
- [ ] CO₂ — `DP4.00200.001` bundle (2017-02–); standalone `DP1.00034.001` /
  `DP1.00099.001` "FUTURE" status.
- [ ] 2018 TBOT anomaly / NCAR-NEON Issue #34 exists and is a 2018 air-temp
  issue.
- [ ] run_tower "insufficient" verdict — now attributable to the v2/2021 cap
  (documented in Key context).
- [ ] Alt sources — PLUMBER2 excludes OSBS; AmeriFlux US-xSB 2019–2024 (confirm
  US-xSB = OSBS).
- [ ] RELEASE-2026 consistency across atmospheric products ("no release
  divergence").
- [ ] NCAR-NEON install claims (`renv.lock`, three HiPerGator paths) — spot-check
  when/if Track 2 activates.
- [ ] Lee-2023 OSBS LIDAR vintage — still awaiting Cohen; non-blocking, tracked
  in `docs/data-acquisition-dates.md`.

## Deliverable

A validated NEON-forced OSBS test case (pre-built v4) plus a documented
forcing-swap path for the production spinup. The custom pipeline is recorded here
as a contingency, ready to activate only on a PI-confirmed gap.

## References

- `docs/neon-data-products.md` — DP catalog, pipeline install paths, caveats.
- `docs/data-acquisition-dates.md` — provenance/vintage (LIDAR, NWI, Lee 2023).
- On-disk v3 forcing:
  `/blue/gerber/cdevaneprugh/cases/run_tower.OSBS.250925-094358/OSBS.transient/run/inputdata/atm/cdeps/v3/OSBS/`.
- CTSM tooling: `ctsm5.3/tools/site_and_regional/{run_tower,run_neon}`,
  `.../listing.csv`; DATM
  `components/cdeps/datm/cime_config/{buildnml,stream_definition_datm.xml}`; NEON
  usermods `cime_config/usermods_dirs/clm/NEON/OSBS/`.
- Current-case config:
  `$CASES/osbs.swenson.spinup/{env_run.xml,CaseDocs/datm.streams.xml}`.
- NCAR-NEON project: https://github.com/NEONScience/NCAR-NEON — Wieder et al.
  2023, *Geosci. Model Dev.* 16, 5979–6000, DOI 10.5194/gmd-16-5979-2023.
- `STATUS.md` — project status; Phase I registered under roadmap track 7.

## Log

### 2026-07-15 — Phase created from research findings

Founding entry. Phase I split out to cover NEON atmospheric forcing (a
site-input-quality track, independent of the hillslope pipeline A–H and
orthogonal to the routing on/off question).

Three Explore agents plus direct NEON-API and on-disk verification established:

1. **Pre-built NEON forcing reaches 2024-12** (v4, 84 monthly files; server
   `listing.csv`). The v3 set (78 files, 2018 → 2024-06) is already on disk from
   a prior `run_tower` run. The long-standing "2018–2021 only" belief — and the
   "run_tower insufficient" verdict built on it — was a namelist cap
   (`NEONVERSION=v2`), not a data limit.
2. **The forcing is already gap-filled and CTSM-ready**
   (`created_with = flow.api.clm.R`, ReddyProc). Format verified by `ncdump`:
   combined monthly `OSBS_atm_YYYY-MM.nc`, single-point, gregorian half-hourly, 8
   vars incl. measured FLDS, RH-based humidity.
3. **Integration is a config change** — the current case cycles a fixed 20-yr
   CRUNCEP block; cycling the NEON years is identical. The fork's `buildnml` caps
   at v3 (needs a patch or `NEONVERSION=v4` for 2024-12).

**Scope decision (user):** pre-built-first. Adopt/validate v4 (Track 1); the
custom NEON→DATM pipeline (Track 2) is a PI-gated contingency, dormant until a PI
conversation (I8) identifies a gap v4 can't fill — none identified so far.
Recorded in STATUS.md (current-state row I, roadmap track 7, change log
2026-07-15).
