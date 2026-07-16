# Phase I: NEON Atmospheric Forcing

Status: **Not started** — adopt pre-built NCAR-NEON tower forcing (2018–2024, v4)
for the OSBS case to replace CRUNCEPv7; custom NEON→DATM pipeline documented as a
PI-gated contingency, not active work. Scoping research complete 2026-07-15
(pipeline confirmed real and runnable; coverage gap quantified at ~3 yr) — no
task begun.
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
  gap-fill quality, or post-2024 recency). **CO₂ is NOT among these** — it is
  architecturally separate from met forcing and needs no pipeline; see Research
  notes §6.

## Coverage options — the I8 decision

Pre-built v4 is **not** the full NEON record. The raw tower data spans a longer
period than NCAR-NEON chose to process; the pre-built set is trimmed at both
ends.

| Layer | Coverage | Notes |
|---|---|---|
| Raw NEON tower data (NEON API) | **2016-08 → 2026-06** (~10 yr, all 7 core vars) | Temp/rad/pressure/wind reach back to 2014-08; RH 2015-06; precip 2016-08. Gapped per-sensor CSVs — not CTSM-usable as-is |
| Pre-built NCAR-NEON **v4** | **2018-01 → 2024-12** (7 yr) | Gap-filled + CTSM-ready. Largest set that exists — verified live 2026-07-15, **no v5**; server-wide only v1–v4 |

**The gap is ~3 years (35 months):** 2017 at the front, 2025 + 2026-01…06 at the
back.

| Option | Coverage | Work |
|---|---|---|
| **1. Use v4 as-is** | 7 yr (2018–2024) | None — CTSM downloads it |
| **2. v4 + custom-extend back end** | ~8.5 yr (2018 → 2026-06) | Build pipeline, run 2025–2026 only. **Mixes gap-fill provenance** (NEON's fills + ours in one record) |
| **3. Full custom** | ~10 yr (2016-08 → 2026-06) | Build pipeline, run whole record. Uniform provenance |

Weighing it:

- **For the 600-yr spinup** the block is cycled regardless (7 yr ≈ 85 loops,
  10 yr ≈ 60 loops). 7 vs 10 yr is a marginal difference in the
  interannual-variability sample. Low value.
- **For a transient / evaluation run** against real observations, extra years and
  recency (2025–2026) matter considerably more. Moderate value.
- **Provenance mixing** is the catch with Option 2 — a caveat carried forever. If
  the pipeline gets built at all, Option 3 is usually cleaner than patching edges.
- The **2018 TBOT anomaly** (Issue #34) sits *inside* v4's window. If real, v4 is
  effectively "6 clean years + 1 flagged," narrowing the gap further. Verify in I1.
- **CO₂ is not part of this trade-off.** It is absent from *both* the pre-built
  NEON files and our current CRUNCEP files by design — a separate DATM stream,
  not a met variable. It cannot motivate Option 2 or 3. See Research notes §6.

## Research notes (2026-07-15): custom-pipeline feasibility

Verdict: **configure-and-run, not reimplement — and not plug-and-play.** The cost
is dominated by standing up a 2021-era R stack, not by writing pipeline code.

### 1. The generator is public and is the exact tool that made v4

`TowerTools_ForcingData/flow.api.clm.R` in
https://github.com/NEONScience/NCAR-NEON (1,480 lines, author David Durden) is
**the script named in the v4 files' `created_with` attribute**, by the same
author as their `created_by`. Running it means running NEON's production
generator, not a reimplementation. Repo README: *"To generate monthly forcing
files from tower met data — modify and run `/TowerTools_ForcingData/flow.api.clm.R`.
This code does the following: Collates NEON data from API, Gap-filling with
ReddyProc, Packages output in NCAR CLM netcdf format, Makes simple plots to
highlight missing data fields."* Repo also carries `flow/flow.dnld.neon.ncar.R`
(download), `gapFilling/`, and `utilities/flow.renv.init.rstr.R` (renv bootstrap).

### 2. Configuration surface is small — and the knobs are exactly ours

Top-of-script parameters (~lines 88–115):

| Parameter | Default | For us |
|---|---|---|
| `Site` | `"TOOL"` | `"OSBS"` |
| `dateBgn` | `"2018-01-01"` | **= v4's start**; → `2016-08-01` for the full record |
| `dateEnd` | `"2024-12-31"` | **= v4's end**; → `2026-06-30` |
| `MethOut` | `c("local","gcs")[2]` | → `[1]`. **Defaults to uploading to NEON's GCS bucket** |
| `DirDnld` | `c("/home/ddurden/eddy/tmp/CLM", tempdir())[1]` | → `[2]` or our path |
| `lowmem` / `maxmonths` | `FALSE` / `2` | memory throttles — likely needed for a 10-yr run |
| `Pack` / `TimeAgr` | `"basic"` / `30` | 30-min averaging; leave |

There is also an **env-var override path** (if `METHPARAFLOW` is set, the script
reads `SITE`, `DATEBGN`, `DATEEND`, `DIROUT`, `LOWMEM`), so it can be driven from
a SLURM wrapper with **no source edits**. Prefer this over editing the script.

### 3. Environment — dedicated conda env, with renv for the package tree

**Decision (user, 2026-07-15): a dedicated conda environment**, not the project
`ctsm` env — this is an R 4.0.5-era stack and would pollute the Python dev env.

Verified 2026-07-15:

| Fact | Value |
|---|---|
| `renv.lock` pins | **195 packages** (`docs/neon-data-products.md` says "~50–100" — **wrong**), R **4.0.5** |
| Pinned repos | 4 × Posit Package Manager CRAN snapshots (2022-02-28 … 2023-10-22) — **all live, HTTP 200** |
| `eddy4R.base` 0.2.24 / `eddy4R.qaqc` 0.2.14 | GitHub source — **not on CRAN, not on conda** |
| `REddyProc` 1.3.2 | CRAN/Repository |
| `rhdf5` 2.34.0 | **Bioconductor** |
| HiPerGator lmod | **`R/4.0` exists** — matches the 4.0.5 pin |
| Container fallback | `quay.io/battelleecology/rstudio:4.0.5` — **still pullable** (537 MB, 2021-07) |

Recommended shape: conda supplies the **R interpreter + system libraries**
(hdf5/netcdf — where conda earns its keep on HPC); **`renv::restore()` supplies
the 195 pinned R packages** from the live Posit snapshots. Do *not* try to
resolve the tree by hand from conda-forge: `eddy4R` is GitHub-only and the lock
pins exact versions.

Known friction: `docs/neon-data-products.md` already warns that renv inside conda
can be finicky around system libs (`ncdf4`, anything spatial) — that warning is
the main risk here. Bioconductor (`rhdf5`) and GitHub (`eddy4R`) restores are the
classic renv failure points. If the restore fights us, the Apptainer/container
path gets the exact 2021 environment and is a **stronger fallback than the "last
resort" framing** in that doc implies.

### 4. It is research code — expect rough edges

No tests, no CLI, no releases; 6 stars; last push 2025-12-18. Observed: a literal
`# CHANGE ME FOR DESIRED CONFIGURATION`; hardcoded `/home/ddurden/` paths
including a GCS credential file; `ver <- paste0("v3/", ...)` disagreeing with the
`v4` upload path; commented-out dependency-hell scars
(`# remove.packages("neonUtilities")`, `# detach(rlang)`); and a mangled comment
`#WhOSBSich NEON site are we grabbing data from` — "OSBS" spliced into "Which" by
a careless find/replace, itself evidence someone previously ran this for OSBS.

**Important:** the script's own dependency block uses ad-hoc `install.packages()`
and calls `library(eddy4R.base)` **without installing it** — i.e. the script's
install path and the `renv.lock` path disagree. Reconcile before running.

### 5. Validation oracle — reproduce v4 before extending

The script's **default date range (2018-01-01 → 2024-12-31) is exactly v4's
range**, and we already hold v4's output (v3 set on disk; v4 fetchable). So the
first run should change **only** `Site="OSBS"` and `MethOut="local"`, then diff
the result against the real v4 files. Reproducing them proves the pipeline
end-to-end against a known-correct reference. **Only then** extend `dateBgn` /
`dateEnd`. This turns "will this work?" into a controlled experiment with a known
answer — a rare luxury. Do not skip it.

### 6. CO₂ is architecturally separate — not a forcing-file variable

**Verified 2026-07-15 against CTSM/CDEPS source and real files.** A natural
assumption is that CO₂, being atmospheric, belongs in the `*_atm_*.nc` forcing
file. It does not — in CTSM, **CO₂ never travels with the meteorology**, for any
forcing dataset.

Evidence:

- The NEON file has **no CO₂** (8 vars only; grep for co2/soil/veg → 0 matches).
- **Our current CRUNCEP file has no CO₂ either** (`ncdump` → 0 matches). So this
  is not a NEON shortcoming; it is how CTSM forcing works generally.
- CO₂ has its **own stream family**, `co2tseries.*`, in
  `components/cdeps/datm/cime_config/stream_definition_datm.xml`, gated by the
  `DATM_CO2_TSERIES` xml variable (valid values: `none`, `20tr`, `omip.*`,
  `SSP1-1.9` … `SSP5-8.5`). Our case: `DATM_CO2_TSERIES = none`.
- That stream points at an entirely different file:
  `$DIN_LOC_ROOT/atm/datm7/CO2/fco2_datm_global_simyr_1750-2014_CMIP6_c180929.nc`,
  mapping `CO2 → Sa_co2diag`, with `stream_meshfile = none` and
  `mapalgo = none` — i.e. **a single global scalar per time step, no spatial
  dimension.**

**Why the separation is correct.** CO₂ is a well-mixed gas: weather varies over
metres and minutes, background CO₂ is ~uniform globally. Embedding one global
number into per-site half-hourly weather files would duplicate it across every
site and timestep. It is factored onto its own axis of variation by design.

**Why tower CO₂ must NOT be used as forcing.** NEON *does* measure CO₂ at OSBS
(in `DP4.00200.001`, which the pipeline already downloads). But canopy-height
tower CO₂ is background **plus the ecosystem's own influence**: respired CO₂
pools under a stable nocturnal boundary layer (readings above background), and
photosynthesis draws it down by afternoon (below background). That diurnal swing
*is the ecosystem signal*. Forcing CLM with it would be circular — the model's
vegetation responding to a concentration that vegetation itself already
depleted/enriched, double-counting a flux the model is computing. **Tower CO₂ is
a validation target for our fluxes, not an input to them.** This is why
NCAR-NEON pulls the EC bundle for T/RH/wind QC and gap-filling but deliberately
omits CO₂ from `OSBS_atm_*.nc`.

**Consequence — CO₂ is a one-line lever, orthogonal to this phase.** If the PI
wants time-varying CO₂ instead of the constant `CCSM_CO2_PPMV = 284.7`, the
answer is *not* a custom NEON pipeline:

```sh
./xmlchange DATM_CO2_TSERIES=20tr    # data already ships with CTSM
```

This composes cleanly with the PI's likely design (spin up at preindustrial
CO₂ + `TSERIES=none`, then run an experiment off that restart with NEON weather
and, if desired, a CO₂ time series — independent knobs). **Caveat:** the shipped
historical file ends at **2014**, before NEON's 2018–2024 window, so a
present-day run needs an SSP variant for the overlap — verify what is on disk
under `$DIN_LOC_ROOT/atm/datm7/CO2/` when it matters. Note also that the NEON
usermods set `CCSM_CO2_PPMV=408.83` (present-day) — see the I4 config-vs-weather
caution.

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

- [ ] **I8. PI conversation — now a quantified question.** Supersedes the earlier
  "no specific gap identified" framing. Pre-built v4 gives **7 yr** (2018–2024,
  one year carrying a known TBOT anomaly, nothing after 2024); the raw record
  gives **~10 yr**. **The gap is ~3 years.** Put the three options from
  "Coverage options" above to the PI and ask: is the extra ~3 yr — particularly
  2025–2026 recency — worth building the pipeline? **CO₂ is NOT a driver here
  and should be struck from the question** — it is architecturally separate from
  met forcing (a `co2tseries` stream / `DATM_CO2_TSERIES` xml lever, not a
  forcing-file variable), so wanting time-varying CO₂ is a one-line `xmlchange`,
  not a reason to build a pipeline. Tower CO₂ is a validation target, not an
  input. See Research notes §6. Record the verdict — this gates I9/I10.
- [ ] **I9.** *(only if I8 finds a gap)* Stand up the environment in a
  **dedicated conda env** (NOT the project `ctsm` env — R 4.0.5-era stack):
  conda for the R interpreter + system libs (hdf5/netcdf), `renv::restore()` for
  the 195 pinned packages from the live Posit snapshots, `eddy4R` from GitHub.
  Apptainer from `quay.io/battelleecology/rstudio:4.0.5` is the fallback.
  Reconcile the script's ad-hoc `install.packages()` block against `renv.lock`
  first. See Research notes §3.
- [ ] **I10.** *(only if needed)* **Reproduce v4 before extending.** Run
  `flow.api.clm.R` changing only `Site="OSBS"` and `MethOut="local"` (its default
  dates already equal v4's range), then diff against the real v4 files. Once it
  reproduces, extend `dateBgn`/`dateEnd` to `2016-08-01`/`2026-06-30` — prefer
  the `METHPARAFLOW` env-var path over source edits — and produce the full-record
  `OSBS_atm_YYYY-MM.nc` set matching the Key-context spec (8 vars, units,
  gregorian half-hourly, `-9999` fill). See Research notes §2 and §5.

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
- [ ] NCAR-NEON install claims — **partially verified 2026-07-15** (Research
  notes §3): repo, `flow.api.clm.R`, `renv.lock`, and Dockerfile all confirmed
  real. Corrections still to apply to `docs/neon-data-products.md`: it pins
  **195** packages, not "~50–100"; the "renv::restore() is the path" claim
  conflicts with the script's own ad-hoc `install.packages()` block; Apptainer is
  undersold as "last resort"; and the install path is now a **dedicated conda
  env** per user decision, not the lmod R module.
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
- NCAR-NEON repo internals (inspected 2026-07-15):
  `TowerTools_ForcingData/flow.api.clm.R` (the generator that produced v4),
  `flow/flow.dnld.neon.ncar.R`, `gapFilling/`, `utilities/flow.renv.init.rstr.R`,
  `renv.lock` (195 pkgs, R 4.0.5, Posit snapshot repos), `Dockerfile`
  (`quay.io/battelleecology/rstudio:4.0.5`).
- Live forcing inventory: `https://storage.neonscience.org/neon-ncar/listing.csv`
  (301-redirects to `storage.googleapis.com` — fetch with `curl -L`).
- `STATUS.md` — project status; Phase I registered under roadmap track 7.

## Log

### 2026-07-15 — CO₂ removed as a custom-pipeline driver

Third pass, from the question "isn't CO₂ a standard atmospheric variable that
would be in the atm files?" Answer verified against CTSM/CDEPS source and real
files: **no — CO₂ never travels with the meteorology in CTSM, for any forcing
dataset.** Our current CRUNCEP files contain no CO₂ either; it is delivered by a
separate `co2tseries.*` stream (`DATM_CO2_TSERIES` xml lever) reading a global
scalar file with `meshfile=none`. Full evidence in Research notes §6, added this
pass.

Two consequences:

1. **CO₂ is struck as a Track 2 driver.** Earlier framing (Approach, Coverage
   options, I8) listed "time-varying CO₂" among the gaps that might justify
   building the custom pipeline. That was **wrong** — wanting time-varying CO₂ is
   `./xmlchange DATM_CO2_TSERIES=20tr`, fully orthogonal to NEON forcing. All
   three sites corrected. This simplifies the I8 conversation.
2. **Tower CO₂ is a validation target, not an input.** Canopy CO₂ carries the
   ecosystem's own respiration/photosynthesis signal; forcing CLM with it would
   double-count a flux the model computes. This is why NCAR-NEON downloads the
   EC bundle for QC/gap-filling but deliberately omits CO₂ from the forcing
   files.

Noted for later: the shipped historical CO₂ file ends at 2014, before NEON's
2018–2024 window — a present-day run needs an SSP variant.

Still no Phase I task started.

### 2026-07-15 — Custom-pipeline feasibility + coverage gap quantified

Second pass the same day, prompted by two questions: is v4 the largest pre-built
set, and what would building the pipeline actually entail?

**Coverage.** Verified against the *live* `listing.csv` (the on-disk copy was
cached 2025-09-25, so re-checking mattered): v4 (2018-01 → 2024-12, 84 files) is
the newest and largest pre-built set — **no v5 exists**; server-wide only v1–v4.
Cross-referenced against the raw NEON API record (all 7 core vars span 2016-08 →
2026-06), the pre-built set is trimmed at both ends and **the gap is ~3 years**
(2017 at the front; 2025 + 2026-01…06 at the back). Added a "Coverage options"
section with the three-option table as the framing for the I8 PI conversation.
I8's earlier "no specific gap identified" framing is **superseded** — the gap is
now quantified.

**Feasibility.** The custom pipeline is **configure-and-run, not reimplement.**
`flow.api.clm.R` is public and is the exact script named in v4's `created_with`
attribute (same author as `created_by`). Its config surface is ~6 parameters
(site / dates / output mode), with a `METHPARAFLOW` env-var override path that
needs no source edits — and its default dates already equal v4's range, giving a
free validation oracle (reproduce v4, then extend). The environment is the real
cost: 195 pinned packages on an R 4.0.5-era stack — but all four pinned Posit
snapshot repos are live, HiPerGator carries `R/4.0`, and the 2021 container is
still pullable. Full detail in Research notes (2026-07-15).

**Environment decision (user):** a **dedicated conda env**, not the project
`ctsm` env — conda for the R interpreter + system libs, `renv::restore()` for the
pinned package tree. This supersedes the lmod-R-module recommendation in
`docs/neon-data-products.md`, which also undercounts the package tree (195, not
"~50–100") and undersells the container path.

Still no Phase I task started.

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
