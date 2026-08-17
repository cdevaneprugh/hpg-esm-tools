# NEON Custom Forcing — CTSM Case Recipe (Phase I / I7)

The complete, minimal set of case settings **unique to driving a CTSM case with the
custom OSBS NEON atmospheric forcing.** Apply these to swap the custom NEON weather into
any CTSM case. Science / experiment knobs (CO₂ year, aerosol / N-dep / ozone scenario,
present-day vs 1850, MOSART vs SROF, run length) are **independent of this recipe** — set
them however the experiment requires.

**Custom dataset:** 101 monthly NetCDFs, **2017-02 → 2025-06**, QC-clean, CTSM-ready
(strict superset of the pre-built NCAR-NEON v4).
`swenson/data/datm/neon_OSBS/custom/OSBS/atm/OSBS_atm_YYYY-MM.nc`

**Validated:** ingested and drove CTSM end-to-end over the full record (ingestion smoke,
2026-08-15); cold-start AD spun up and converged on the cycled record (2026-08-17).

---

## Easiest path — clone the working case

The working case already has every setting below wired in. Clone it, then change only the
science knobs:

```bash
cd $CIME_SCRIPTS
./create_clone --case $CASES/<new-name> --clone $CASES/osbs.swenson.neon.spinup --keepexe
```

The rest of this doc is the **from-scratch** recipe (what the clone already contains), so the
settings are explicit and portable to a fresh case.

---

## From scratch — the forcing settings

### 1. Compset & resolution (`create_newcase`)

```bash
./create_newcase --case $CASES/<name> \
    --compset I1PtClm60Bgc \
    --res CLM_USRDAT \
    --run-unsupported
```

- `I1PtClm60Bgc` longname = `2000_DATM%1PT_CLM60%BGC_SICE_SOCN_SROF_SGLC_SWAV_SESP`.
  The **`DATM%1PT`** piece is the single-point NEON DATM machinery (vs `DATM%CRUv7` for
  CRUNCEP) — it defines the two NEON streams and does the RH→specific-humidity conversion
  internally. Sets `DATM_MODE=1PT`.
- `2000_` (present-day CO₂/chemistry) and `SROF` (stub river) are **science knobs** baked into
  this alias — change the compset if you want 1850 or MOSART.
- `--res CLM_USRDAT` — user-supplied point domain (goes with the OSBS surfdata).

### 2. Point location — must match the OSBS surfdata

```bash
./xmlchange PTS_LAT=29.689282,PTS_LON=278.006569
```

- These are the **surfdata grid-cell coordinates** (`PTS_LON` is 0–360, i.e. −81.993°E).
- Caveat: the physical NEON tower is ~120 m away; use the *surfdata* coordinates, not the
  tower's, or CTSM and the forcing land in different cells.

### 3. Cycling window + start date

```bash
./xmlchange RUN_STARTDATE=2017-02-01
./xmlchange DATM_YR_START=2017,DATM_YR_END=2025,DATM_YR_ALIGN=2017
```

- Start **2017-02-01** — the record's first month. A conventional `0001-01-01` / January
  start would demand the absent Jan-2017.
- `CALENDAR=NO_LEAP` is the compset default and is correct for these files — leave it.

### 4. Point the DATM streams at the custom files — `user_nl_datm_streams`

There are **two** NEON weather streams; override the file list on both:

```
NEON.OSBS:datafiles = <comma-separated list of all 101 OSBS_atm_*.nc>
NEON.OSBS:taxmode = cycle

NEON.NEON_PRECIP.OSBS:datafiles = <same 101-file list>
NEON.NEON_PRECIP.OSBS:taxmode = cycle
```

- `NEON.OSBS` carries the met variables; `NEON.NEON_PRECIP.OSBS` carries precipitation.
- The `datafiles` value is a comma-separated list of all 101 files (2017-02 … 2025-06) in the
  custom dir, with `\` line-continuations. It is long — **copy the two blocks verbatim from the
  working case** `$CASES/osbs.swenson.neon.spinup/user_nl_datm_streams`, or regenerate the list
  from the directory. Overriding `datafiles` is what makes the case use *our* files instead of
  the pre-built v4 the NEON compset would otherwise stage.

### 5. THE critical one — `dtlimit = -1` on BOTH NEON streams

```
NEON.OSBS:dtlimit = -1
NEON.NEON_PRECIP.OSBS:dtlimit = -1
```

- **Why:** cycling a finite window (`taxmode=cycle`) wraps past the last record and trips
  CDEPS's default `dtlimit=1.5`, hard-crashing the model at `dshr_strdata_mod.F90:1050`.
  `-1` is CDEPS's own escape hatch (`override_annual_cycle`) for streams that don't cycle on
  January boundaries. Namelist-only — **no rebuild.**
- Without this, a cycled run crashes **at every cycle boundary** (hit in the ingestion smoke at
  the final timestep; would hit a spinup at every wrap). Required whenever this forcing is
  cycled.

### 6. HiPerGator build note

```bash
./xmlchange MPILIB=openmpi
```

- HPG build requirement (not forcing-specific, but the case won't build without it here).

---

## Operational notes for long cycled spinups

Not settings, but hard-won during Phase I — relevant if you run a multi-decade cycled spinup:

- **Generous walltime + single-segment runs.** `gerber` QOS allows a 31-day walltime.
  Set `JOB_WALLCLOCK_TIME` well above the expected runtime and use `RESUBMIT=0` /
  `STOP_N=<full length>` so there is no chunk-to-chunk handoff.
- **If a job dies mid-segment, run `./case.st_archive` BEFORE resubmitting.** `CONTINUE_RUN`
  resumes from the last *short-term-archived* restart, not the newest restart file in the run
  dir — an un-archived timed-out chunk silently rewinds the resume.
- **Partial-year wrap.** The record is 2017-02 → 2025-06, so cycling wraps **Jun→Feb** — a
  benign ~8-month seasonal jolt each cycle (`dtlimit=-1` carries it). If you want a
  clean-calendar cycle instead, set `DATM_YR_START=2018,DATM_YR_END=2024` (complete years
  only — this is exactly the pre-built v4 span, and discards the 2017 spliced precip + 2025
  tail that are the custom set's value-add).

---

## Optional — forcing-assessment history output

Not needed to *run* the forcing; used to *evaluate* it (forcing mirror + surface fluxes on a
gridcell tape). The 4-tape config (h0 forcing/flux + h1/h2/h3 hillslope) is in the working
case's `user_nl_clm` (`hist_fincl1..4`, `hist_nhtfrq/mfilt/dov2xy/type1d_pertape`). Copy that
block if you want NEON-vs-CRUNCEP / model-vs-tower comparison output; otherwise the CTSM
default h0 is fine.

---

## NOT part of this recipe (separate concerns)

- **OSBS hillslope inputs** (in `user_nl_clm`): `fsurdat`, `hillslope_file`, `use_hillslope=.true.`,
  `spillheight=0.0`, and the 5 hydrology SourceMods. These make it the OSBS *hillslope* case
  (Phase E/F) — independent of the weather.
- **Science-input streams** `presaero` / `presndep` / `preso3` (SSP3-7.0, `dtlimit=30`) —
  aerosol / N-deposition / ozone scenario. A science choice, not weather forcing.
- **Experiment knobs** — CO₂ year (2000 vs 1850), SROF vs MOSART, run type/length. The PI's
  to set.
