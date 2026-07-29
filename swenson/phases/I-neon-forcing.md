# Phase I: NEON Atmospheric Forcing

Status: **In progress — I1–I2.5 done; single linear plan (reworked 2026-07-15).**
Fetch pre-built v4 → smoke-test the CTSM integration with it → build our own
NEON→DATM pipeline and validate against v4 → produce the full 2016–2025 dataset (released) →
full CTSM integration (PI-gated tail). I1 (verification), I2 (v4 fetched), and
I2.5 (integration smoke test — **PASSED**) complete; I3 (pipeline) next.
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
through 2024-12** — NEON's pipeline does the "process the data" work upstream for
that window. (The plan nonetheless runs that same pipeline ourselves for the
fuller 2016–2025 record; v4 is the temporary start + validation reference — see
Approach.)

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
  `storage.neonscience.org/neon-ncar/NEON/atm/cdeps/`. **We bypass that
  auto-download entirely:** we fetch v4 ourselves (I2) and point the stream at our
  files via a `datafiles` override (§9, I6). So the fork's `buildnml` version cap
  (`_get_neon_data_availability` searches only v3/v2/v1, missing v4) never applies.

- **The prior "run_tower is insufficient" verdict is attributable to the v2
  cap.** Pre-built v4 (2018–2024) is the temporary starting forcing; the plan is
  to build our own pipeline for the fuller 2016–2025 record and validate it
  against v4 (I3–I5). The remaining PI decision is narrower — whether to *adopt*
  the custom dataset for the production respin (I8).

## Approach

**Single linear track (reworked 2026-07-15 — no pre-built-vs-custom split).** v4
and any custom dataset wire into CTSM identically (a `user_nl_datm_streams`
`datafiles` override, §9), so the only difference is which files sit in our
directory. The plan:

1. **Fetch pre-built v4** to our directory (I2) — the temporary starting forcing
   and the validation reference.
2. **Smoke-test the CTSM integration** with v4 (I2.5) — an early dry run of the
   Approach-B wiring, before the pipeline build, to catch integration blockers
   cheaply.
3. **Stand up our own pipeline** (I3).
4. **Validate against v4** — run the pipeline for 2018–2024 and compare to v4
   bit-for-bit / within tolerance (I4). Go/no-go gate.
5. **Produce the full dataset** — 2016-08 → 2025-06 (I5; released data only).
6. **Integrate into CTSM (full)** — build a case on the forcing, keeping our
   hillslope surfdata (I6–I8). Downstream tail: needs the dataset first, carries
   the PI knob decisions, and pairs with the eventual spinup respin.

CO₂ drives none of this — it is architecturally separate from met forcing (§6).

## Coverage — v4 vs. the full custom record

Pre-built v4 is **not** the full NEON record. The raw tower data spans a longer
period than NCAR-NEON chose to process; the pre-built set is trimmed at both
ends.

**Decision (2026-07-15): build the full custom record, released data only.** v4 is
the temporary starting forcing and the validation reference; our pipeline produces
the full **2016-08 → 2025-06** set (Option 3, capped at the RELEASE-2026 cut — PI
decision: no provisional 2025-07 → 2026-06 tail). The options below are the
recorded rationale.

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
- **Provenance mixing** is the catch with Option 2 — a caveat carried forever;
  Option 3 (chosen) keeps the whole record uniform rather than patching edges.
- The **2018 TBOT anomaly** (Issue #34) is **retired** — verified in I1 as a
  v1-era unit artifact fixed by reprocessing; on-disk v3 2018 TBOT is sane
  (269–307 K). All 7 pre-built years are clean.
- **CO₂ is not part of this trade-off.** It is absent from *both* the pre-built
  NEON files and our current CRUNCEP files by design — a separate DATM stream,
  not a met variable. It cannot motivate Option 2 or 3. See Research notes §6.

## Research notes (2026-07-15): custom-pipeline feasibility

Verdict: **configure-and-run, not reimplement — and not plug-and-play.** The cost
is dominated by standing up a 2021-era R stack, not by writing pipeline code —
**but the gating unknown sits upstream of the R stack:** whether HiPerGator can
reach NEON's raw-data API at all. The `/data/` download endpoint currently 403s
from this host (§7). No R environment fixes a blocked download — resolve §7
before investing in §3.

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
| `dateEnd` | `"2024-12-31"` | **= v4's end**; → `2025-06-30` (released cut) |
| `MethOut` | `c("local","gcs")[2]` | → `[1]`. **Defaults to uploading to NEON's GCS bucket** |
| `DirDnld` | `c("/home/ddurden/eddy/tmp/CLM", tempdir())[1]` | → `[2]` or our path |
| `lowmem` / `maxmonths` | `FALSE` / `2` | memory throttles — likely needed for a 10-yr run |
| `Pack` / `TimeAgr` | `"basic"` / `30` | 30-min averaging; leave |

There is also an **env-var override path** (if `METHPARAFLOW` is set, the script
reads `SITE`, `DATEBGN`, `DATEEND`, `DIROUT`, `LOWMEM`), so it can be driven from
a SLURM wrapper with **no source edits**. Prefer this over editing the script.

### 3. Environment — dedicated conda env, with renv for the package tree

**Decision (user, 2026-07-15): a dedicated environment**, not the project `ctsm`
env — this is an R 4.0.5-era stack and would pollute the Python dev env. **Venue
(I3):** processing on a personal machine via NEON's Docker image is now the
recommended path — it sidesteps both this env build and the `/data/` block (§7);
the HiPerGator conda + `renv` build below is the alternative.

Verified 2026-07-15:

| Fact | Value |
|---|---|
| `renv.lock` pins | **195 packages** (R **4.0.5**; `docs/neon-data-products.md` corrected from "~50–100" this session) |
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
usermods set `CCSM_CO2_PPMV=408.83` (present-day) — see the I6 config-vs-weather
caution.

### 7. Raw-data access from HiPerGator — the gating unknown (verify first)

**Found 2026-07-15; not yet resolved.** The pipeline pulls raw tower data with
`neonUtilities` (`zipsByProduct` / `loadByProduct`), which calls NEON's REST
`/data/{DP}/{site}/{month}` endpoint. **That endpoint returns HTTP 403 "Access
Denied" from HiPerGator** (login node). Diagnosis:

- **Not rate-limiting:** the 403 carries `x-ratelimit-remaining: 200` (full
  quota) — rejected at the edge before the limiter.
- **Not a host/connectivity problem:** `/products/` and `/sites/` return 200 from
  the same host, and the storage bucket downloads fine (that is how the pre-built
  v4 files are fetched).
- **Scope:** the whole `/data/` *download* family is blocked (`/data/` and
  `/data/query` both 403); metadata endpoints are open. Signature is **IP-based
  access control on datacenter/HPC ranges**, not obviously a token gate — but a
  token is untested.

Why it matters: the entire pipeline exists to pull raw NEON data, and the
standard route is blocked from where we would run it. **This gates the pipeline (I3)** — no
environment work matters if the inputs can't be fetched. Resolution options,
cheapest first: (1) register a free NEON API token and pass it to `neonUtilities`
(test whether it lifts the 403); (2) run the pull from a non-blocked network, or
download raw data via the NEON portal (web / Globus / S3) and feed the pipeline
pre-downloaded files locally; (3) ask UF RC / NEON whether HiPerGator's range is
blocked. **Do this before building the R env.**

### 8. NEON is not a drop-in for CRUNCEP — how to structure the swap

**Verified 2026-07-15 against `components/cdeps/datm/` source and real namelists
(I1).** NEON tower forcing cannot be swapped into the CRUNCEP streams as a file
substitution. It differs in four ways — but **all four are handled by CDEPS,
provided the case uses the NEON DATM machinery** (`DATM%1PT` +
`CLM_USRDAT_NAME=NEON` + `NEONSITE`); none are handled by repointing the CRUNCEP
streams at NEON files.

| Axis | CRUNCEP (now) | NEON |
|---|---|---|
| DATM_MODE | `CLMCRUNCEPv7` | `1PT` (both run internal datamode `CLMNCEP`) |
| Streams | 3 gridded (Solar/Precip/TPQW), 0.5° mesh, bilinear | 2 single-point (`NEON.$SITE` + `NEON.NEON_PRECIP.$SITE`), `meshfile=none` |
| Humidity | `QBOT → Sa_shum` (specific) | `RH → Sa_rh` — **CDEPS converts to shum internally**, `datm_datamode_clmncep_mod.F90:436-461` |
| Longwave | computed by DATM | measured `FLDS → Faxa_lwdn` (automatic) |
| Calendar | `noleap` | `gregorian` file, run under `noleap` for control/spinup (set by NEON usermods per compset) |

**Recommended structure — Approach B (use the NEON DATM mode, re-assert our
surface data).** Build with compset `I1PtClm60Bgc` (control/spinup) or
`IHist1PtClm60Bgc` (transient) + the NEON usermods so CDEPS constructs the
supported NEON streams, then override CLM's surface data in `user_nl_clm`:

```
fsurdat        = '<our OSBS surfdata>'
hillslope_file = '<our production hillslope nc>'
use_hillslope  = .true.
```

`fsurdat`/`hillslope_file`/`use_hillslope` are **CLM** namelist vars, independent
of DATM — fully compatible with NEON forcing. (Approach A — hacking the CRUNCEP
streams via `user_nl_datm_streams` — is rejected: it can only modify existing
streams, cannot add the NEON stream or flip `datamode`.)

**Swap the weather, not the experiment.** The NEON usermods bundle scientific
changes beyond the weather. To keep the 1850 experiment, do NOT inherit them —
keep `CCSM_CO2_PPMV=284.7` (NEON sets 408.83), `DATM_PRESAERO/NDEP/O3=clim_1850`
(NEON sets SSP3-7.0), `DATM_CO2_TSERIES=none`, `CLM_NML_USE_CASE=1850_control`.

**Caveats.** (a) The `1Pt` compsets use `SROF` (stub river), not the current
case's `MOSART`. (b) `buildnml` version auto-discovery caps at `v3` — set
`NEONVERSION` explicitly for v4 — moot for us: the `datafiles` override (§9)
bypasses auto-discovery, so the cap never applies. (c) NEON weather only exists
~2018+, so the swap means *cycling* the 2018–2024 NEON years under fixed 1850
boundary conditions — not a like-for-like 1901–1920 replacement.

Source: `components/cdeps/datm/cime_config/{config_component.xml,
namelist_definition_datm.xml, stream_definition_datm.xml, buildnml}`,
`datm_datamode_clmncep_mod.F90`, `cime_config/usermods_dirs/clm/NEON/`,
`config_compsets.xml`.

### 9. Producer/consumer contract — what CDEPS converts vs. what any file must provide

**The DATM machinery is format-driven, not source-driven.** CDEPS is the
*consumer*; NEON's cloud files and a custom `flow.api.clm.R` run are both
*producers* of the same format. The consumer does not care which made the file —
so a **custom "fuller" dataset (pre-2018, post-2024, or the full 2016→2025 record)
feeds the exact same `1PT` machinery as prepackaged v4.** This is what makes
building our own pipeline worthwhile, and why I4 validates by *reproducing v4*
first.

**What the consumer converts for you** (runtime, `CLMNCEP` datamode) — supply a
compact set, it derives the rest:
- `RH` (%) → specific humidity (`datm_datamode_clmncep_mod.F90:436-461`)
- `FLDS` used if present, else longwave computed
- total `PRECTmms` → rain/snow split; total `FSDS` → direct/diffuse + vis/near-IR
  bands (standard CLMNCEP derivations)
- time interpolation to the model step; calendar reconciliation (gregorian file
  under a noleap spinup)

**What "correct format" demands** (the consumer trusts these **blindly**):
1. **Names** — `FSDS, FLDS, PRECTmms, TBOT, RH, WIND, PSRF, ZBOT`.
2. **Units** — K, Pa, mm/s, W/m², m/s, %. **Not converted, not validated.** Wrong
   units are ingested silently — this *is* Issue #34 (Celsius read as Kelvin →
   562 K; the file was structurally perfect).
3. **Structure** — single-point `(time, lat=1, lon=1)` + `LONGXY/LATIXY`,
   gregorian half-hourly.
4. **No gaps** — every timestep needs a real value; a `-9999` fed as forcing
   wrecks the run. Gap-filling is the *producer's* job (ReddyProc, upstream); the
   consumer assumes it is done.

**The two things the consumer will NOT do — fix units, fill gaps — both fall on
the producer.** Our producer is `flow.api.clm.R` (NEON's own generator): it emits
correct names/units/structure *and* gap-fills, so "correct format" comes out by
construction. The risk appears only if someone hand-rolls the NetCDF instead of
running the generator — hence the pipeline routes through the real
`flow.api.clm.R`, not a reimplementation, and I4 proves it by reproducing v4.

**Wiring a fuller-than-NEON record.** Stock NEON file *discovery* is driven by
NEON's `listing.csv`, so it only sees NEON-published months. To use a record that
extends beyond that, override the active `NEON.$SITE` stream's `datafiles` in
`user_nl_datm_streams` to point at our custom files (allowed — modifying an
*existing active* stream's datafiles is fine; the §8 "can't use
`user_nl_datm_streams`" caveat was only about *adding* a stream or flipping
`datamode`), and set `DATM_YR_START/END` to span the fuller range.

## Tasks

**Single linear track (reworked 2026-07-15).** No pre-built-vs-custom split: v4
and any custom dataset wire into CTSM identically (a `user_nl_datm_streams`
`datafiles` override, §9), so the only difference is which files sit in our
directory. Spine (I1–I5) produces and validates the forcing dataset — with an
early integration smoke test (I2.5) that dry-runs the CTSM wiring on v4 before the
pipeline build; the tail (I6–I8) does the full integration (downstream / PI-gated).

- [x] **I1. Formal claims verification. DONE 2026-07-15.** Claims re-verified via
  three adversarial agents (NEON API / CTSM source / external web) + on-disk
  checks; `docs/neon-data-products.md` corrected (see the checklist below and the
  Log entry). Refuted: wind/radiation start 2014-08 (not 2013), tipping-bucket
  2016-08 (not 2014), weighing gauge IS installed, CO₂ bundle 2017-02, TBOT source
  is 00003 not 00002. Drop-in question answered in Research notes §8. 2018 TBOT
  verified sane in pre-built (269–307 K).
- [x] **I2. Fetch the full pre-built v4 set. DONE 2026-07-15.** 84 files
  (2018-01 → 2024-12, **12.08 MB**) fetched from the NEON storage bucket to
  `data/datm/neon_OSBS/v4/OSBS/` (`*.nc` gitignored; provenance README at
  `data/datm/neon_OSBS/README.md`). Integrity 84/84 valid; **v3 sanity check
  PASS** — v4 is a full reprocessing (all overlap months differ from the on-disk
  v3) but only at reprocessing scale (RMS Δ: TBOT 0.17 K, PSRF 9 Pa, FSDS
  0.07 W/m²); new months physical, no fills. See Log. The on-disk run_tower v3 set
  is superseded. This is the temporary starting forcing + the reference for I4.
- [x] **I2.5. Integration smoke test — v4 dry run. DONE 2026-07-15 — PASSED.**
  Purpose (met): prove the Approach-B wiring on known-good v4 forcing before
  building the pipeline, isolating wiring bugs from data bugs. Case
  `osbs.swenson.neon-v4-smoke` (`I1PtClm60Bgc` + NEON usermods; our
  `fsurdat`/`hillslope_file`/`use_hillslope=.true.`; `datafiles` override → the I2
  v4 files; present-day knobs; cold-start, 2 yr) **completed cleanly**: **26
  hillslope columns** active, ran with our v4 files (168 runtime stream refs, 0
  auto-download), forcing ingested with **measured FLDS** (354.9 W/m²) and
  **converted RH** (RH2M 84.6%) — empirically confirming §8/§9. All three untested
  combos — `1PT`+hillslope, `SROF`+hillslope, `datafiles` override — work. **Four
  integration issues found, each carrying to I6** (see Log). See §8, §9.
- [ ] **I3. Stand up the pipeline — HiPerGator venue, raw-data access, R
  environment.** **Venue decided (2026-07-29): build and run on HiPerGator** (for
  reproducibility, PI access, and output next to CTSM); the raw download is the one
  necessary off-HPG step — NEON IP-blocks `/data/` from HPG (§7), so pull raw zips
  on a non-blocked machine (`zipsByProduct`, released-only) and Globus them in.
  Environment = **conda-first hybrid**: conda-forge + bioconda supply 185/195
  packages (r-base, the HDF5/NetCDF stack, `eddy4R.qaqc` deps, `devtools`/`remotes`,
  and conda-forge compilers — *not* lmod gcc, for ABI match with conda's R); the
  remaining ~10 leaves install from source (`REddyProc` + `solartime`/`bigleaf`;
  `eddy4R.base`/`eddy4R.qaqc` via `install_github ref=898a72d`; `eddy4R.base` deps
  `DataCombine`/`EMD`/`robfilter`; standalone `metScanR`/`prism`) plus the local
  `NEON.gf`. Point the offline script at the transferred cache via the `DirDnld`
  seam + a `doDnld` flag (EC bundle already split; met products swap
  `loadByProduct` → `stackByTable`; `MethOut="local"`). **Sub-decision resolved (2026-07-29): conda-current versions + tolerance**
  (accepting a newer `r-base` than 4.0.5 — the 4.0.5 + 2023-pin combo is likely
  unsolvable on conda). Run 1 is validated against v4 by the **fqc-partitioned
  comparison** (I4), not bit-for-bit; `renv::restore()` pinned is the fallback
  *only if* that comparison fails to clear the reference band. Full plan: `docs/neon-forcing-pipeline-hipergator.md`. See
  Research notes §3, §7.
- [ ] **I4. Reproduce-v4 validation — go/no-go gate (fqc-partitioned comparison).**
  Run the pipeline for **2018–2024** (its default range = v4's), then compare the
  output against the v4 files from I2. Same generator + range → identical
  grids/vars/units → clean element-wise diff (no regrid/time-align). **Partition
  each timestep by its `<VAR>_fqc` flag:**
  - **Measured (fqc=0):** same raw data + deterministic conversions → must match v4
    to ~machine precision *regardless of library versions*. Proves the pipeline
    logic (units/structure/time/conversions) is faithful; a units bug surfaces here
    (cf. Issue #34).
  - **Gap-filled (fqc>0):** the only place library-version drift lives → the
    *tolerance* applies here, and only here.
  **Reference band** (acceptable filled-point drift) = the I2 v4-vs-v3 sanity check:
  RMS Δ TBOT 0.17 K, PSRF 9 Pa, FSDS 0.07 W/m², RH 0.21 %, WIND 0.025 m/s. Filled
  drift ≤ that band, measured points far below. **Tools:** CPRNC
  (`CTSM_CPRNC_Deterministic_Analysis.md`), NCO `ncdiff`, or xarray. **Deliver as a
  committed regression script** — the forcing analog of `merit_regression.py`
  (a `neon_v4_regression` with pass/fail, re-runnable after any env change or record
  extension). **Pass** = measured near-exact + gap-filled within the band → pipeline
  trusted; **fail** → fall back to the renv-pinned build. See Research notes §5, §9.
- [ ] **I5. Produce the full dataset.** Once I4 passes, run the pipeline over the
  full record (via the `METHPARAFLOW` env-var path, no source edits) into the
  curated dir. **Usable start is 2016-08** — all 7 core variables are real there;
  do **not** placeholder pre-2016 precip/RH (they can't be invented; a reanalysis
  blend for pre-2016 would be a separate PI decision). **End date = 2025-06 — PI
  decision: released data only.** The RELEASE-2026 cut ends 2025-06 (frozen,
  citable, EC reprocessed); the PI does **not** want the provisional
  2025-07 → 2026-06 tail (a moving target NEON revises without notice, not citable,
  EC not pre-release-reprocessed). Target record: **2016-08 → 2025-06**. See §9.

### Integration tail — downstream (needs the dataset first; PI-gated; pairs with the respin)

- [ ] **I6. Integrate into a CTSM case (Approach B), keeping our hillslope
  surfdata.** Per Research notes §8, a **compset change**, not a stream-swap in the
  existing spinup case: build `I1PtClm60Bgc` + NEON usermods; re-assert `fsurdat` +
  `hillslope_file` + `use_hillslope=.true.` in `user_nl_clm`; **point the NEON
  stream at our curated files via a `user_nl_datm_streams` `datafiles` override**
  (this is where the old buildnml-version task folded in — no auto-discovery, no v4
  cap). RH→shum conversion and measured FLDS are automatic. **Swap the weather, not
  the experiment** — override the NEON usermods' present-day knobs back to 1850
  (`CCSM_CO2_PPMV=284.7`, `DATM_PRESAERO/NDEP/O3=clim_1850`, `DATM_CO2_TSERIES=none`,
  `CLM_NML_USE_CASE=1850_control`) unless the PI wants present-day. Note `1Pt`
  compsets use `SROF`, not `MOSART`. Also verify `ZBOT` matches the OSBS instrument
  height. See §8, §9.
- [ ] **I7. Run + validate the case.** Build + run a short (≈5-yr) NEON-forced
  case; confirm from `datm.log`/`lnd.log` that `OSBS_atm_*.nc` are the active
  streams, FLDS is measured (not derived), RH is ingested; sanity-compare
  TBOT/PRECT/FLDS against the CRUNCEP baseline (reuse `case.analyzer` /
  `/case-check`).
- [ ] **I8. PI decisions.** (a) **Cycle vs blend** for the 600-yr spinup — cycle
  the NEON block, or blend (long reanalysis for AD/post-AD spinup, NEON for the
  final transient/evaluation run, as successive cases). (b) **Adoption** — whether
  to drive the production respin with the full custom dataset. (c) **End date —
  RESOLVED (PI): released data only**, so the dataset ends **2025-06**; the
  provisional 2025-07 → 2026-06 tail is excluded. Note: the production hillslope
  file is **no longer frozen** — the PI is proceeding with the existing file via
  soil-value adjustments (some concerns remain; left in the PI's wheelhouse), so
  adoption is not freeze-blocked.

### Claims-to-verify checklist (for I1)

Per-product OSBS availability (verify each via NEON API, RELEASE-2026):

- [x] 7 Tier-1 DPs — verified. **All start 2014-08** (not 2013): wind
  `DP1.00001.001` and radiation `DP1.00023.001` were wrongly "2013–"; tipping
  `DP1.00045.001` is 2016-08 (was "2014–"); pressure `DP1.00004.001` 2014-08; RH
  `DP1.00098.001` 2015-06. TBOT: pipeline uses `DP1.00003.001` (triple), not
  `DP1.00002.001` (single) — doc reconciled.
- [x] **`DP1.00044.001` weighing-gauge precip** — REFUTED "NOT installed"; present
  2016-09→, RELEASE-2026, the pipeline's primary precip. Doc corrected.
- [x] Pipeline aux inputs — also pulls `DP1.00024.001` (PAR) + `DP1.00014.001`
  (direct/diffuse SW); added to doc.
- [x] CO₂ — `DP4.00200.001` bundle **2017-02–** (was "2016–"); standalone
  `DP1.00034.001`/`DP1.00099.001` confirmed FUTURE (null siteCodes).
- [x] 2018 TBOT / Issue #34 — real ("TBOT ... unrealistic", 562 K) but a v1-era
  C→K unit artifact **fixed by reprocessing (2021)**; on-disk v3 2018 TBOT sane
  (269–307 K). Not a gap-fill concern for pre-built.
- [x] run_tower "insufficient" — confirmed = the v2/2021 namelist cap.
- [x] Alt sources — PLUMBER2 excludes OSBS (confirmed, Ukkola 2022 ESSD 14, 449);
  US-xSB = OSBS confirmed (BASE 2018–2024, FLUXNET 2019–2024 — doc split).
- [x] RELEASE-2026 — confirmed consistent; released cut ends **2025-06**, last 12
  mo (2025-07→2026-06) PROVISIONAL. Noted in doc.
- [x] NCAR-NEON install — repo/`flow.api.clm.R`/`renv.lock`/Dockerfile real;
  **195** packages (doc corrected from "~50–100"). renv-vs-script mismatch,
  Apptainer framing, and dedicated-conda-env decision recorded in Research §3.
- [ ] Lee-2023 OSBS LIDAR vintage — still awaiting Cohen; non-blocking, tracked
  in `docs/data-acquisition-dates.md`.

## Deliverable

A trusted NEON forcing dataset for OSBS — our own pipeline output, validated
against pre-built v4 (I4) and extended to the full 2016–2025 record (I5) — plus a
NEON-forced CTSM case that keeps our hillslope surfdata and demonstrably runs
(I6–I7). The production respin / adoption decision is the PI-gated tail (I8).

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

### 2026-07-29 — Fidelity fork resolved: conda-current + tolerance; I4 = fqc-partitioned v4 comparison

PI accepted the tolerance path — **conda-current package versions, not renv
bit-for-bit** — conditioned on a solid way to compare our output to the pre-built
v4. Resolved this session:

- **Decision:** build with conda-current versions (accept a newer `r-base` than
  4.0.5, since 4.0.5 + the 2023 pins is likely an unsolvable conda solve).
  `renv::restore()` pinned stays as the fallback *only if* the v4 comparison fails.
- **I4 method — fqc-partitioned comparison.** Run 1 and v4 share the
  generator/range, so it's a clean element-wise diff. Split each timestep by
  `<VAR>_fqc`: measured (fqc=0) must match to ~machine precision (validates the
  pipeline logic independent of library versions); gap-filled (fqc>0) carries the
  tolerance (the only place library drift lives). More diagnostic than bit-for-bit —
  separates "is the code correct?" from "how much does the gap-fill drift?"
- **Reference band** = the I2 v4-vs-v3 RMS Δ (TBOT 0.17 K, PSRF 9 Pa, FSDS
  0.07 W/m², RH 0.21 %, WIND 0.025 m/s).
- **Deliverable:** a committed `neon_v4_regression` script — the forcing analog of
  `merit_regression.py` — reporting the fqc-partitioned per-variable stats with a
  pass/fail, re-runnable after any env change.

I3 sub-decision marked resolved; I4 rewritten with the method; pipeline doc
(Stage 5 + Friction) updated to match. Doc only; no build/script yet.

### 2026-07-29 — I3 venue decision: build on HiPerGator (conda-hybrid); guide repurposed

Reversed the 2026-07-26 "process on a personal machine" lean: **the pipeline will
be built and run on HiPerGator** (PI direction — reproducibility, PI access, output
next to CTSM). Only the raw NEON download stays off-HPG (the `/data/` IP block is
unchanged), transferred in via Globus; the environment build and all processing run
on HPG.

Environment strategy settled as a **conda-first hybrid**, grounded in a verified
dependency sweep (renv.lock parsed programmatically + eddy4R/REddyProc DESCRIPTIONs
+ anaconda.org availability checks):
- **195 packages** in the lock (189 CRAN-mirror + 4 Bioconductor + 2 GitHub), R
  4.0.5. **Conda covers 185** (conda-forge `r-*` + bioconda `rhdf5` family);
  **10 must come from source** — `REddyProc` (the gap-fill engine is *not* on any
  conda channel) + `solartime`/`bigleaf`; `eddy4R.base` + `DataCombine`/`EMD`/
  `robfilter`; `eddy4R.qaqc`; standalone `metScanR`/`prism` — plus the local
  `NEON.gf`.
- **Compilers: conda-forge, not lmod gcc** — the source builds load into conda's R
  and must be ABI-matched; lmod gcc 14.2 vs conda's older `libstdc++` risks GLIBCXX
  failures. (Opposite of CTSM, where lmod gcc is correct.)
- **NetCDF is portable** — the pipeline env is fully decoupled from CTSM's netcdf; no
  cross-env library matching (unlike CTSM's linked executable).
- **Open fork:** conda-current versions (likely, since `r-base=4.0.5` + 2023 pins is
  probably unsolvable → newer R + drift → I4 tolerance-pass) vs `renv::restore()`
  pinned (bit-for-bit, more friction). To be decided before building.

Repurposed the local-download guide and **renamed
`docs/neon-forcing-download-guide.md` → `docs/neon-forcing-pipeline-hipergator.md`**
(raw-download-off-HPG + conda env on HPG + the `DirDnld`/`doDnld` script edits + the
two runs + friction/open risks). I3 rewritten to the HiPerGator venue. No
`environment.yml` yet (still planning). Doc only.

### 2026-07-26 — I3 prep: local-download handoff guide written (superseded 2026-07-29)

Decided the pipeline venue: **process on a personal machine** (the `/data/` 403 is
a HiPerGator-IP block — confirmed still active, rejected at NEON's Google edge with
full rate-limit quota; a residential/campus IP is not blocked). Checked HiPerGator's
docs (docs.rc.ufl.edu): outbound is open (wget/curl documented, no proxy/firewall
on standard nodes), there is no proxy to route around a *remote* provider's block,
and their endorsed pattern for exactly this case is "download off-cluster, bring it
in via Globus" — so the transfer-in route is the sanctioned one.

Wrote `docs/neon-forcing-download-guide.md` (renamed 2026-07-29 →
`neon-forcing-pipeline-hipergator.md`): a self-contained brief for a Claude
instance on a laptop to run NEON's `flow.api.clm.R` and produce the OSBS forcing.
Covers why-local, the deliverable/format, the exact NEON products pulled (with the
verified DP table), the `flow.api.clm.R` parameter surface (Site/dates/`MethOut=local`
/dirs/lowmem + the `METHPARAFLOW` env path), released-only enforcement (dateEnd
cap 2025-06-30 + `release="RELEASE-2026"`), the two runs (validation 2018→2024 =
84 files, then full 2016-08→2025-06 = 107 files), sanity checks, and what to hand
back (~30 MB: two NetCDF dirs + plots + provenance). The user will run it locally
and upload via Globus/SFTP; the authoritative Run-1-vs-v4 comparison (I4) stays on
HiPerGator. Doc only.

### 2026-07-15 — PI decisions: released data only + hillslope file un-frozen

Two PI directions folded in:
- **Dataset end date RESOLVED — released data only.** The custom record targets
  **2016-08 → 2025-06** (the RELEASE-2026 cut); the provisional 2025-07 → 2026-06
  tail is **excluded** (the PI does not want a moving, non-citable base). Updated
  I5, I8(c), the Coverage decision, the Approach/Status/Deliverable end dates, and
  the `flow.api.clm.R` `dateEnd` (→ 2025-06-30).
- **Production hillslope file is no longer frozen.** The PI can work with the
  existing file via soil-value adjustments — generally working, some concerns
  remain, left in the PI's wheelhouse. So the I8 adoption decision is no longer
  freeze-blocked (I8 note updated). Not tracking the soil-value work here — it's
  the PI's.

Doc only.

### 2026-07-15 — I2.5 smoke test PASSED (first CTSM run of Phase I)

Built and ran a cold-start `1PT` + hillslope case (`osbs.swenson.neon-v4-smoke`)
on our pre-built v4 forcing via the `datafiles` override. **Result: PASS** — the
2-yr run completed (`case.run success`, ~17 min), **26 hillslope columns** active,
and the runtime `datm.streams.xml` used our v4 files exclusively (168 refs, 0
auto-download). Forcing empirically confirmed from h0a: TBOT 289.7 K, RAIN
5.2e-5 mm/s, FSDS 118 W/m², **measured FLDS 354.9 W/m²** (not DATM-derived),
**RH2M 84.6%** (CDEPS converted RH→shum) — §8/§9 proven end to end. All three
untested combos (`1PT`+hillslope, `SROF`+hillslope, `datafiles` override) work.

**Four integration issues found — each a blocker cleared for I6:**
1. **MPI library.** NEON usermods force `MPILIB=mpi-serial`, but HiPerGator's
   ESMF/PIO are OpenMPI builds → the final link fails on OpenMPI symbols. **Force
   `MPILIB=openmpi`, set before the first build** (a mid-stream switch left stale
   state). Working recipe (from `osbs.swenson.spinup` + run_tower): openmpi,
   `NTASKS=NTHRDS=1`, `GMAKE_J=1`, `PIO_TYPENAME=netcdf`, DEBUG FALSE, nuopc.
2. **SourceMods.** The operative `user_nl_clm` sets `spillheight=0.0`, valid only
   with the operative case's **6-file hydrology SourceMod set**
   (`HillslopeHydrologyMod`, `InfiltrationExcessRunoffMod`,
   `SaturatedExcessRunoffMod`, `SoilHydrologyMod`, `SurfaceWaterMod`). A fresh
   case without them ENDRUNs at `read_hillslope_properties_namelist`. **I6 must
   carry those SourceMods** (or drop `spillheight`). The smoke test ran stock — OK
   for wiring, but I6 needs them for the production hydrology science.
3. **Coordinate mismatch.** Our OSBS surfdata is at (29.689282, 278.006569); the
   NEON usermods point the domain at NEON's official OSBS (29.68819, 278.00655),
   ~120 m off → `surfrdMod` domain-vs-surfdata ENDRUN. **Set `PTS_LAT/PTS_LON` to
   the surfdata coords** (or regenerate surfdata at NEON coords).
4. **Walltime.** Cold-start hillslope BGC ran ~1 sim-yr / ~8 min single-point; the
   2-yr run took 17 min. Budget walltime accordingly for longer I6 runs.

Case retained at `$CASES/osbs.swenson.neon-v4-smoke` for reference. Next: I3
(stand up the pipeline).

### 2026-07-15 — I5/I8: released-vs-provisional reproducibility trade-off

Folded NEON's data-release model into the dataset end-date choice (verified
against NEON's `data-revisions-releases` docs). Provisional data (2025-07 →
2026-06) is auto-QC'd but a **moving target** — revised without notice, not
citable, and its eddy-covariance bundle (which the pipeline pulls for QC +
gap-fill) skips the pre-release reprocessing that released data gets; RELEASE-2026
(to 2025-06) is frozen + citable. So the end date is a reproducibility choice:
2025-06 (frozen/reproducible) vs 2026-06 (provisional, ~1 yr more recency). Added
detail to **I5** and a decision item (c) to **I8**. Doc only.

### 2026-07-15 — I2 done: fetched pre-built v4 forcing (first operational step)

Fetched the full NCAR-NEON **v4** OSBS forcing — 84 files (2018-01 → 2024-12),
**12.08 MB** — to `data/datm/neon_OSBS/v4/OSBS/` (`*.nc` gitignored; provenance
README at `data/datm/neon_OSBS/README.md`). First operational Phase I step;
everything prior was documentation.

- **Integrity:** 84/84 open cleanly with the 8-var spec; byte total 12.08 MB
  (exact match to the GCS-reported size).
- **v4 is a full reprocessing, not v3 + 6 months.** All 78 months overlapping the
  older on-disk v3 differ (0/78 byte-identical; every file regenerated 2025-11-09).
  Differences are **reprocessing-scale**: RMS Δ over the overlap is TBOT 0.17 K,
  PSRF 9 Pa, FSDS 0.07 W/m², FLDS 0.09 W/m², precip 3×10⁻⁴ mm/s, RH 0.21 %, WIND
  0.025 m/s, ZBOT 0. Larger max-Δ (TBOT 14.6 K, RH 29 %) are isolated gap-filled
  intervals refilled differently — bulk unchanged. **Sanity PASS.**
- New months (2024-07 → 2024-12) physically plausible; no `-9999` in physical vars.
- No formal release notes in the bucket; v4 (ends 2024-12) is entirely within the
  RELEASE-2026 released window (to 2025-06) — released, not provisional.
- Pointers updated: `docs/neon-data-products.md` storage-location question
  resolved; `CLAUDE.md` (Key Resources + data tree) and `STATUS.md` (References +
  change log) point at the new location.

Next: I2.5 (integration smoke test) — not started.

### 2026-07-15 — Added early integration smoke test (I2.5)

Per user, inserted a small smoke-test step after the v4 download. Rationale
(verified this session): the run_tower v3 case ran successfully, but in the
*vanilla* NEON config (`use_hillslope=.false.`, default NEON surfdata) — so it
proves the NEON forcing path works but **not** our Approach-B integration (`1PT` +
our hillslope surfdata + `datafiles` override + 1850 knobs), which has never been
run. I2.5 dry-runs that wiring on known-good v4 forcing before the pipeline build,
isolating integration bugs from data bugs and surfacing any `1PT`+hillslope /
`SROF`+hillslope blocker cheaply. Cold-start + short → safe against the production
freeze. Sub-numbered (not a renumber) to avoid re-churning the just-reworked list.
Approach list, spine description, and status header updated. Doc only.

### 2026-07-15 — Task list reworked to a single linear track

Dropped the Track 1 / Track 2 split (user decision — option A). The two-track
framing was an artifact of treating "pre-built vs. custom" as different
mechanisms; §9 established they wire in identically (a `datafiles` override), so
there is one linear plan: fetch v4 → stand up the pipeline → validate against v4
→ produce the full dataset → integrate into CTSM (downstream/PI-gated tail).

Old → new task mapping (earlier Log entries keep their original numbers as a
dated record):
- I1 → **I1** (unchanged; done)
- I2 (fetch v4) → **I2**
- I3 (buildnml v4 patch) → **dropped** — the `datafiles` override (§9) bypasses
  auto-discovery, so no patch is needed; folded into I6 as a wiring step
- I8b (`/data/` access) + I9 (R env) → **I3** (stand up the pipeline — venue +
  access + env; recommends process-on-laptop via Docker)
- I10 reproduce-half → **I4** (reproduce-v4 validation gate)
- I10 extend-half → **I5** (produce the full 2016–2026 dataset)
- I4 (build case) + I5 (ZBOT) → **I6** (integrate into a CTSM case)
- I6 (test) → **I7** (run + validate the case)
- I7 (cycle vs blend) + I8 (adoption) → **I8** (PI decisions)

I8's old "is it worth building the pipeline?" go/no-go is resolved — we build it;
the surviving PI decision is adoption for the respin. Status header, Approach,
Coverage heading, and the §6–§9 cross-references were updated to the new numbers.
Doc only; nothing operational.

### 2026-07-15 — §9 added: producer/consumer contract (format vs. source)

Follow-on to the §8 drop-in analysis, clarifying a recurring question: does
"CDEPS handles it" lock us to the prepackaged data? No — the DATM machinery is
**format-driven, not source-driven**, so a custom "fuller" dataset feeds the same
`1PT` machinery as prepackaged v4. §9 records what CDEPS converts at runtime
(RH→shum, longwave fallback, precip rain/snow + solar band splits, time interp,
calendar) vs. what "correct format" demands (names, units — trusted blindly,
structure, no gaps), the two burdens the consumer will NOT do (units, gaps — both
the producer's job, hence routing Track 2 through the real `flow.api.clm.R`), and
how to wire a fuller-than-NEON record (`user_nl_datm_streams` datafiles override).
Issue #34 is the units cautionary tale; ReddyProc is the gap-fill. I10
cross-referenced. Doc-only; nothing operational.

### 2026-07-15 — I1 complete: claims re-verified, drop-in analysis, doc corrected

Executed I1 (first Phase I task). Research via three adversarial Explore agents
(NEON API / CTSM source / external web) + direct on-disk checks; corrections
applied to `docs/neon-data-products.md`, `STATUS.md`, and this file.

- **Claims re-verified, several REFUTED.** Wind + radiation start 2014-08 (doc
  said "2013–"); tipping-bucket 2016-08 (said "2014–"); **weighing gauge
  DP1.00044.001 IS installed** (2016-09→, said "NOT installed"); CO₂ bundle
  2017-02 (said "2016–"); TBOT source is `DP1.00003.001` (triple), not the doc's
  `DP1.00002.001` (single); pipeline also pulls PAR + direct/diffuse SW. Standalone
  CO₂ confirmed FUTURE. RELEASE-2026 confirmed, with a provisional tail after
  2025-06.
- **Issue #34 (2018 TBOT) is a non-issue for pre-built.** A v1-era C→K unit
  artifact (562 K) fixed by upstream reprocessing in 2021; **on-disk v3 2018 TBOT
  verified sane (269–307 K).** The "6 clean + 1 flagged" coverage caveat is
  retired — all 7 pre-built years are clean.
- **External claims confirmed:** PLUMBER2 excludes OSBS (Ukkola 2022, ESSD 14,
  449); AmeriFlux US-xSB = OSBS (BASE 2018–2024, FLUXNET 2019–2024).
- **Drop-in verdict (new Research note §8): NOT a drop-in.** NEON is a different
  DATM_MODE / stream structure / humidity var / calendar than CRUNCEP, but CDEPS
  handles all four via the NEON `1PT` machinery (RH→shum converted internally,
  `datm_datamode_clmncep_mod.F90:436-461`; measured FLDS automatic). Recommended
  structure = Approach B: build the `I1PtClm60Bgc` compset + NEON usermods, then
  re-assert our `fsurdat`/`hillslope_file`/`use_hillslope` in `user_nl_clm` (CLM
  vars, independent of DATM). "Swap the weather, not the experiment" — keep the
  1850 CO₂/chemistry knobs. I4 reframed accordingly.
- **CO₂** re-confirmed separate (§6): constant `CCSM_CO2_PPMV` or `co2tseries.*`
  stream, never in the met file.

I1 checked off; checklist ticked. Still no data fetched / case built / CTSM source
patched.

### 2026-07-15 — Pre-PI-meeting scoping: v4 fetch proven, pipeline access blocker found

Fourth pass, prepping for the PI meeting: what does fetching v4 and standing up
the pipeline actually entail, and how does each slot into the task list?

- **v4 fetch proven.** Pulled the live `listing.csv` and test-downloaded the
  2024-12 file (153 KB, valid 8-var forcing file). Fetching the full 84-file v4
  set is a trivial `curl -L` loop, ~13 MB. **I2 rewritten**: fetch the full set
  to a curated `data/datm/neon_OSBS/v4/OSBS/`, and **ignore the on-disk
  run_tower v3 set** (superseded per user).
- **Pipeline access blocker found (Research notes §7, new).** The raw-data
  `/data/` endpoint `neonUtilities` depends on **403s from HiPerGator** (full
  rate-limit quota, metadata endpoints fine → IP-based edge block, not
  throttling). Gates the whole custom pipeline and was absent from the doc.
  Added as **task I8b** (Track-2 step 0) and flagged in the feasibility verdict —
  resolve before building the R env.
- **Track 2 gating reframed.** Was "do NOT start without PI go-ahead." Split into
  *exploratory* (verify access, build env, reproduce v4 — may run now, ahead of
  the PI meeting) vs. *adoption* (commit to a custom full-record product — stays
  PI-gated on I8). I9/I10 relabeled; I10 split into reproduce (exploratory) and
  extend (gated) halves.
- **I2↔I10 coupling made explicit:** v4 on disk (I2) is the reference oracle for
  the reproduce-v4 step (I10), so Task A precedes Task B's validation.

Mapping recorded: "fetch v4" = I2; "get the pipeline working" = I8b + I9 + I10
(reproduce half). Still no Phase I task started.

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
