# NEON OSBS Forcing — Local Download & Processing Guide

**Audience:** a Claude instance running on a personal machine (laptop/workstation)
with an ordinary residential/campus internet connection.
**Goal:** produce CTSM/CLM atmospheric-forcing NetCDF files for the NEON **OSBS**
site by running NEON's own generator, and hand the (small) output back for upload
to HiPerGator.

This document is self-contained. You do not need the wider project repo — every
specific you need (site, products, date ranges, output format, gotchas) is below.

---

## 1. Why this runs on a laptop and not on the HPC

The processing pipeline (`flow.api.clm.R`, below) pulls raw NEON tower data from
NEON's REST download API (`data.neonscience.org/api/v0/data/...`). **That endpoint
returns HTTP 403 from HiPerGator** — NEON blocks the HPC's IP range at its cloud
edge (the rejection carries a full rate-limit quota and happens before auth, so a
token does not fix it; NEON's metadata endpoints and storage bucket work fine, but
the raw-data `/data/` family is blocked). A residential/campus IP is **not**
blocked, so the download + processing must happen off-HPC. The output is tiny
(~15 MB per dataset), so we generate it here and SFTP/Globus it up.

**The deliverable is the processed NetCDF, not the raw NEON data.** You will
download GB of raw NEON data as an intermediate, but only the processed monthly
forcing files go back to HiPerGator.

---

## 2. What to produce (the deliverable)

NEON's generator writes **one combined NetCDF per month**, named
`OSBS_atm_YYYY-MM.nc`. Each file must match this spec exactly (it will, because
we run NEON's own tool):

- single point: dims `(time, lat=1, lon=1)`, plus `LONGXY`/`LATIXY`
- `time`: days-since-month-start, **`gregorian`** calendar, **half-hourly** (48/day)
- `_FillValue = -9999`; all physical vars `double(time,lat,lon)`
- eight physical variables CTSM consumes:

  | Var | Units | Meaning |
  |---|---|---|
  | `FSDS` | W/m² | incident shortwave |
  | `FLDS` | W/m² | incident longwave (**measured**) |
  | `PRECTmms` | mm/s | precipitation |
  | `TBOT` | K | air temperature |
  | `RH` | % | relative humidity |
  | `WIND` | m/s | wind speed |
  | `PSRF` | Pa | surface pressure |
  | `ZBOT` | m | observation height |

  (Plus informational `<VAR>_fqc` gap-fill flags — fine to keep; CTSM ignores them.)

**Produce two datasets, in two separate output directories:**

| # | Purpose | Date range | Expected files |
|---|---|---|---|
| **Run 1** | Validation ("reproduce v4") | **2018-01-01 → 2024-12-31** | **84** |
| **Run 2** | Full deliverable | **2016-08-01 → 2025-06-30** | **107** |

Do **Run 1 first** (see §6). Run 1's range is exactly the range of NEON's
pre-built "v4" product, which we already hold on HiPerGator — reproducing it is a
known-answer check that the pipeline works before we trust the full record.
(107 = 5 months of 2016 + 8×12 + 6 months of 2025.)

---

## 3. The tool

**Repo:** https://github.com/NEONScience/NCAR-NEON (Wieder et al. 2023, *GMD* 16,
5979–6000). This is the pipeline NEON/NCAR uses to make the pre-built CLM forcing;
it is the exact tool named in the pre-built files' `created_with` attribute — you
are running the production generator, not a reimplementation.

**Entry point:** `TowerTools_ForcingData/flow.api.clm.R` (~1,480 lines). It:
collates NEON data from the API → gap-fills with `ReddyProc` → writes the
CLM-format monthly NetCDF above → makes simple missing-data plots.

Supporting files in the repo: `flow/flow.dnld.neon.ncar.R` (download),
`gapFilling/`, `utilities/flow.renv.init.rstr.R` (renv bootstrap), `renv.lock`
(pins the R package tree), and `Dockerfile` / `Dockerfile_rminimal`.

---

## 4. Environment (recommended: Docker)

The pipeline is a **2021-era R 4.0.5 stack** with 195 pinned packages, including
GitHub-only (`eddy4R.base`, `eddy4R.qaqc`) and Bioconductor (`rhdf5`) sources —
painful to assemble by hand. **Use the repo's Docker path**; it bakes the exact
environment and sidesteps the dependency reconciliation below.

- The repo ships `Dockerfile` (full) and `Dockerfile_rminimal` (lighter). Read
  both and the README; build the one that matches your machine (the full one is
  most faithful).
- Base image is `quay.io/battelleecology/rstudio:4.0.5` (R 4.0.5 + eddy4R,
  ~537 MB, still pullable). If you'd rather not build: pull that base, clone the
  repo into it, `R -e 'renv::restore()'`, then run the script.
- Bind-mount two host directories into the container: a **raw-download cache**
  (GB-scale, §5) and an **output dir** (the small NetCDF deliverable).

**Known dependency trap (why Docker):** the script's own header uses ad-hoc
`install.packages()` *and* calls `library(eddy4R.base)` **without installing it** —
its install path and `renv.lock` disagree. The Docker image already has the tree,
so this doesn't bite you. If you build a native env instead, you must reconcile
this (restore from `renv.lock`, don't trust the script's inline installs).

---

## 5. Parameters to set

`flow.api.clm.R` has a small config block near the top (~lines 88–115). It also
honors an **environment-variable override**: if `METHPARAFLOW` is set, the script
reads `SITE`, `DATEBGN`, `DATEEND`, `DIROUT`, `LOWMEM` from the env — prefer this
over editing the script where possible.

| Parameter | Default | Set to | Notes |
|---|---|---|---|
| `Site` / `SITE` | `"TOOL"` | **`"OSBS"`** | pulls OSBS site metadata (lat/lon, tower/instrument height → `ZBOT`) from the script's site table. OSBS is known to the script. |
| `dateBgn` / `DATEBGN` | `2018-01-01` | per run (§2) | |
| `dateEnd` / `DATEEND` | `2024-12-31` | per run (§2) | **Run 2 caps at `2025-06-30`** — this is the primary guard for "released data only" (see below). |
| `MethOut` | `gcs` (index `[2]`) | **`"local"`** (index `[1]`) | **Critical.** The default uploads results to *NEON's* GCS bucket using a hardcoded `/home/ddurden/` credential file you don't have. May not be covered by the env-var path → **edit the script line** to force local. |
| `DirDnld` | hardcoded `/home/ddurden/...` or `tempdir()` | a real persistent path | raw-download cache. Use a persistent dir (not `tempdir()`) so Run 2 reuses Run 1's cache instead of re-downloading. |
| `DirOut` / `DIROUT` | — | a per-run output dir | e.g. `.../out/validation_2018-2024` and `.../out/full_2016-2025`. |
| `lowmem` / `LOWMEM`, `maxmonths` | `FALSE`, `2` | raise throttling if needed | a multi-year run can exhaust RAM; set `lowmem=TRUE` / small `maxmonths` if it OOMs. |
| `Pack`, `TimeAgr` | `"basic"`, `30` | leave | 30-min averaging is what we want. |

### Released data only (PI decision)

The dataset must use **RELEASE-2026 (released) data only**, not provisional. Two
guards, use both:

1. **Date cap (primary):** ending Run 2 at **2025-06-30** excludes the provisional
   window (provisional is 2025-07 → 2026-06). Everything ≤ 2025-06 is released.
2. **Explicit release (belt-and-suspenders):** in the download call
   (`flow.dnld.neon.ncar.R` / the `neonUtilities::loadByProduct` /
   `zipsByProduct` invocation), request released data — set `release="RELEASE-2026"`
   and/or `include.provisional=FALSE`. Check the exact argument names in the pinned
   `neonUtilities` version; if unavailable, the date cap alone still suffices.

---

## 6. Execution order

1. **Set up** the Docker environment (§4). Clone the repo; read the top of
   `flow.api.clm.R` to confirm the parameter names/line numbers (research code —
   verify, don't assume).
2. **Run 1 — validation (do this first).** Set `Site="OSBS"`, `MethOut="local"`,
   a persistent `DirDnld`, `DirOut=.../validation_2018-2024`, and **leave the
   default dates** (2018-01-01 → 2024-12-31). Run it. This should download the raw
   products (§7), gap-fill, and write **84** `OSBS_atm_YYYY-MM.nc` files.
   - `neonUtilities` reports the total download size before pulling — the raw pull
     is **GB-scale** (dominated by the eddy-covariance bundle). Confirm you have
     the disk + bandwidth.
3. **Sanity-check Run 1** (§8). If it looks physical, the pipeline works.
4. **Run 2 — full record.** Same settings but `dateBgn=2016-08-01`,
   `dateEnd=2025-06-30`, `DirOut=.../full_2016-2025`. Reuses the cached raw data
   for overlapping months; downloads only the extra 2016-08…2017-12 and
   2025-01…06 months. Writes **107** files.
5. **Sanity-check Run 2** (§8).

(We run the authoritative bit-for-bit comparison of Run 1 vs the pre-built v4
files back on HiPerGator — you don't need v4 locally. Your job is to produce Run 1
and Run 2 and confirm they're internally sane.)

---

## 7. What the pipeline downloads (for reference / verification)

You don't have to fetch these by hand — the script pulls them — but this is the
product list so you can confirm the download grabbed the right things. NEON site
**OSBS**, domain D03. Availability verified against the live NEON API 2026-07-15.

**Core forcing inputs**

| DP Number | Product | Feeds | OSBS start |
|---|---|---|---|
| DP1.00003.001 | Triple Aspirated Air Temperature (tower top) | `TBOT` | 2014-08 |
| DP1.00002.001 | Single Aspirated Air Temperature | `TBOT` (alt) | 2014-08 |
| DP1.00004.001 | Barometric Pressure | `PSRF` | 2014-08 |
| DP1.00001.001 | 2D Wind Speed & Direction | `WIND` | 2014-08 |
| DP1.00098.001 | Relative Humidity | `RH` | 2015-06 |
| DP1.00023.001 | Shortwave & Longwave Radiation (net radiometer) | `FSDS`, `FLDS` | 2014-08 |
| DP1.00044.001 | Precipitation — Weighing Gauge (primary) | `PRECTmms` | 2016-09 |
| DP1.00045.001 | Precipitation — Tipping Bucket (secondary) | `PRECTmms` | 2016-08 |

**Gap-fill / partitioning inputs** (not written to the forcing file)

| DP Number | Product |
|---|---|
| DP1.00024.001 | Photosynthetically Active Radiation (PAR) |
| DP1.00014.001 | Direct & Diffuse Shortwave |

**QC / CO₂ bundle** (the big download; **CO₂ is NOT written to the forcing file** —
it's a validation target only, delivered to CTSM by a separate stream)

| DP Number | Product |
|---|---|
| DP4.00200.001 | Bundled Eddy Covariance (embeds CO₂; carries redundant T/RH/wind for QC) | 2017-02 |

**Why the start is 2016-08:** precipitation is the last core sensor to come online
(tipping bucket 2016-08). Temperature/radiation/pressure/wind reach 2014-08 and RH
2015-06, but you can't make continuous forcing before precip exists. Do **not**
try to invent pre-2016 precip/RH; a pre-2016 reanalysis blend would be a separate
decision, out of scope here.

---

## 8. Sanity checks (run on both output sets)

For each output directory:

- **File count:** Run 1 → 84 files; Run 2 → 107 files. Contiguous months, no gaps
  in the sequence.
- **Each file opens** and has the 8 physical vars (§2), dims `(time, lat=1, lon=1)`,
  gregorian calendar, 48 timesteps/day.
- **No `-9999` in the 8 physical variables.** A `_FillValue` left in a physical var
  means a gap that didn't fill — it will wreck the CTSM run. Gap-filling is the
  pipeline's job; if you see `-9999` in `TBOT`/`FSDS`/etc., something went wrong.
- **Physical ranges:** `TBOT` ~250–320 K, `PSRF` ~95,000–103,000 Pa, `RH` 0–100 %,
  `FSDS` ≥ 0, `WIND` ≥ 0, `PRECTmms` ≥ 0, `FLDS` ~150–500 W/m².
- **2018 `TBOT` specifically:** confirm it's ~269–307 K. (An old v1-era bug wrote a
  ~562 K Celsius→Kelvin artifact; it was fixed upstream by reprocessing, but verify
  it's gone in your output.)
- Skim the script's **missing-data plots** for months with heavy gap-fill; note any.

---

## 9. What to hand back

Package for upload to HiPerGator (all small):

- **`validation_2018-2024/`** — the 84 Run 1 files.
- **`full_2016-2025/`** — the 107 Run 2 files.
- The script's **missing-data plots** and any run log.
- A short **provenance note** (`PROVENANCE.md`) recording, for reproducibility:
  - NCAR-NEON repo **commit hash**; Docker image/tag used
  - `neonUtilities` version, `ReddyProc` version, `eddy4R` versions
  - `release` argument used (RELEASE-2026) and confirmation provisional was excluded
  - date the runs were done; any months flagged as heavily gap-filled; any 2018
    `TBOT` note

You do **not** need to upload the multi-GB raw NEON cache — keep it locally in case
a re-run is needed, but only the NetCDF + plots + provenance go up.

---

## 10. Known gotchas (this is research code)

- **`MethOut` default = GCS upload.** Left unchanged, the script tries to upload to
  NEON's cloud bucket with a credential file you don't have. Force `"local"`.
- **Hardcoded `/home/ddurden/` paths** appear throughout (download dir, credential
  file). Repoint `DirDnld`/`DirOut` to your paths.
- **Install path vs `renv.lock` disagree** (§4) — use Docker to dodge it.
- A literal `# CHANGE ME FOR DESIRED CONFIGURATION` marker and a mangled
  `#WhOSBSich NEON site...` comment exist in the script — the latter is a leftover
  from a prior OSBS run (reassuring: OSBS has been run through this before).
- **Memory:** a ~10-year run can OOM; raise `lowmem`/lower `maxmonths` if so.
- The `/data/` **403 does not apply here** — that's a HiPerGator-IP problem; on a
  normal network the download just works. If you *do* see a 403, you're likely on a
  blocked network (VPN back to campus, cloud VM); switch networks.

---

## 11. One-line summary for the impatient

Run NEON's `flow.api.clm.R` (Docker, `quay.io/battelleecology/rstudio:4.0.5`) for
site **OSBS**, `MethOut="local"`, released data only: first the default range
**2018-01→2024-12** (84 files, a known-answer check), then the full
**2016-08→2025-06** (107 files). Sanity-check both, hand back the two NetCDF
directories + plots + a provenance note. Total upload ≈ 30 MB.
