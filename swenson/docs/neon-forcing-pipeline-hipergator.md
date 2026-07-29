# NEON OSBS Forcing — Building the Pipeline on HiPerGator

**Status: planning-stage.** The architecture and the fidelity approach
(conda-current versions + a *tolerance* check against v4, validated by the
fqc-partitioned comparison in Stage 5) are settled; the exact `r-base`/package
versions fall out of the conda solve. No `environment.yml` is committed yet — that
is the next step.

**Goal:** produce CTSM/CLM DATM atmospheric forcing for NEON **OSBS** by running
NEON's own generator (`flow.api.clm.R`) **on HiPerGator** — so the pipeline lives
where the PI and future work can re-run it, and the output lands where CTSM uses
it. Everything runs on HiPerGator except the one step that cannot: the raw NEON
data pull.

---

## 1. The one unavoidable off-HPG step

The pipeline's raw-data download uses `neonUtilities`, which calls NEON's REST
`/data/` endpoint. **That endpoint returns HTTP 403 from HiPerGator** — NEON blocks
the HPC's IP range at its cloud edge (rejected before auth, so an API token does
not help; NEON's metadata endpoints and storage bucket work, only `/data/` is
blocked). So the **raw download must happen on a non-blocked machine** (a laptop /
campus workstation) and be transferred in. Everything downstream — the whole
environment and all processing — runs on HiPerGator.

(The only way to make the download itself run on HPG would be for NEON / UF RC to
allowlist HiPerGator's egress IP — not pursued yet, and not required for this
plan.)

---

## 2. Architecture at a glance

| Where | Does what | Why there |
|---|---|---|
| **Off-HPG** (laptop) | Download raw NEON zips (`zipsByProduct`) → Globus to HPG | `/data/` API is IP-blocked from HPG |
| **HiPerGator** | Build conda env → run the modified pipeline offline → `OSBS_atm_*.nc` | reproducibility, PI access, output next to CTSM |

The seam between the two is a single directory (`DirDnld`): the laptop *fills* it
(download); HiPerGator *reads* it (offline processing).

---

## 3. What we're producing (unchanged from the original plan)

NEON's generator writes **one combined NetCDF per month**, `OSBS_atm_YYYY-MM.nc`,
single point `(time, lat=1, lon=1)`, `time` in days-since-month-start on a
**`gregorian`** calendar, **half-hourly** (48/day), `_FillValue=-9999`, eight
physical variables CTSM consumes:

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

(Plus informational `<VAR>_fqc` gap-fill flags; CTSM ignores them.)

**Two runs, in separate output dirs:**

| # | Purpose | Date range | Files |
|---|---|---|---|
| **Run 1** | Validation ("reproduce v4") | 2018-01-01 → 2024-12-31 | 84 |
| **Run 2** | Full deliverable | 2016-08-01 → 2025-06-30 | 107 |

Run 1's range is exactly NEON's pre-built **v4** range, which we already hold on
HiPerGator (`data/datm/neon_OSBS/v4/OSBS/`) — so Run 1 is a known-answer check
(see Stage 5). Released data only (PI decision): the 2025-06-30 cap excludes the
provisional 2025-07 → 2026-06 tail.

---

## 4. Stage 1 — Raw data: download off-HPG, transfer in

The laptop side is now **small** — it only downloads, it does not run the
pipeline. It needs a minimal R + the single `neonUtilities` package (or NEON's
Docker image used only for the download), nothing else.

**What the pipeline pulls** (NEON site OSBS, domain D03; availability verified
against the live NEON API 2026-07-15):

| DP Number | Product | Feeds | OSBS start |
|---|---|---|---|
| DP1.00003.001 | Triple Aspirated Air Temperature | `TBOT` | 2014-08 |
| DP1.00002.001 | Single Aspirated Air Temperature | `TBOT` (alt) | 2014-08 |
| DP1.00004.001 | Barometric Pressure | `PSRF` | 2014-08 |
| DP1.00001.001 | 2D Wind Speed & Direction | `WIND` | 2014-08 |
| DP1.00098.001 | Relative Humidity | `RH` | 2015-06 |
| DP1.00023.001 | Shortwave & Longwave Radiation (net radiometer) | `FSDS`, `FLDS` | 2014-08 |
| DP1.00044.001 | Precipitation — Weighing Gauge (primary) | `PRECTmms` | 2016-09 |
| DP1.00045.001 | Precipitation — Tipping Bucket (secondary) | `PRECTmms` | 2016-08 |
| DP1.00024.001 | PAR | gap-fill input | 2014-08 |
| DP1.00014.001 | Direct & Diffuse Shortwave | gap-fill input | 2014-08 |
| DP4.00200.001 | Bundled Eddy Covariance | QC + CO₂ (validation only) | 2017-02 |

`DP4.00200.001` is the GB-scale one and dominates the transfer. **Usable start is
2016-08** (precip is the last core sensor online); do not invent pre-2016 data.

**How:**
1. On the non-blocked machine, `zipsByProduct(dpID=…, site="OSBS", savepath=DirDnld,
   startdate, enddate, release="RELEASE-2026", include.provisional=FALSE)` for each
   product. This writes zips into `DirDnld/filesToStack<dpID>/` — the layout the
   HPG-side stacking step expects. Use `zipsByProduct` (download-to-disk), **not**
   `loadByProduct` (in-memory), so you get transferable files.
2. `include.provisional=FALSE` + the 2025-06-30 end date are the two guards for
   released-only.
3. **Globus** `DirDnld` up to HiPerGator (`docs.rc.ufl.edu/data_transfer/globus_transfer`
   — the sanctioned path for GB-scale external data).

Land it at a curated HPG path, e.g. `swenson/data/neon/met/DirDnld/` (gitignored).

---

## 5. Stage 2 — The pipeline environment on HiPerGator (conda-first hybrid)

Three shelves. **Conda builds the first two; you hand-place the third.**

- **Shelf 1 — R + the plumbing.** `r-base` and the HDF5/NetCDF system libraries.
  Conda provides these, and the system libs (libhdf5, libnetcdf) come
  *automatically* as dependencies of `r-ncdf4` / `bioconductor-rhdf5` — no manual
  apt-style install. This is conda's payoff on HPC.
- **Shelf 2 — standard R packages (185 of 195).** conda-forge for the CRAN
  packages (`r-*`), bioconda for the `rhdf5` family — including `devtools`/`remotes`
  (the tools that install shelf 3) and the entire `eddy4R.qaqc` dependency subtree.
- **Shelf 3 — the ~10 leaves conda can't supply, + NEON.gf.** Installed from
  source *after* conda sets up shelves 1–2:
  - **GitHub:** `eddy4R.base`, `eddy4R.qaqc`
    (`remotes::install_github("NEONScience/eddy4R/pack/…", ref="898a72d")` to match
    the lockfile SHA).
  - **CRAN-source:** `REddyProc` (the gap-fill engine — not on any conda channel)
    and its deps `solartime`, `bigleaf`; `eddy4R.base`'s deps `DataCombine`, `EMD`,
    `robfilter`; standalone `metScanR`, `prism`. Most are pulled automatically by R
    when you install their parent.
  - **Local:** `NEON.gf` (lives inside the pipeline repo —
    `devtools::install("gapFilling/pack/NEON.gf")`; needs only `robustbase`, which
    conda has).

**Package facts (verified this session):** the lockfile has **195 packages**
(189 CRAN-mirror + 4 Bioconductor + 2 GitHub), R **4.0.5**. Conda covers **185**;
**10 must come from source** (listed above), plus the local `NEON.gf`.

**lmod's role is minimal:** `module load conda`. This is lighter than our CTSM
workflow because R packages arrive as pre-built conda binaries, whereas CTSM is
compiled from source against lmod libraries.

**Compilers — use conda's, not lmod's.** The shelf-3 packages compile from source
(`REddyProc` is Rcpp/C++; `EMD`, `robfilter` have compiled code), so the env needs
a toolchain. Use conda-forge's `gxx_linux-64` / `gfortran_linux-64`, **not** the
lmod `gcc`. Reason: the compiled `.so` files load into conda's R and must be
ABI-compatible with it; HiPerGator's modern lmod gcc (14.2) against conda's older
`libstdc++` risks `GLIBCXX` load-time failures. Conda's compilers are the same gcc
family, version-matched to the env by construction. (This is the opposite of CTSM,
where lmod gcc is correct because everything in that link is lmod-provided.)

**NetCDF is portable — the pipeline env is fully decoupled from CTSM's.** A `.nc`
file is a self-describing interchange format, not bound to the library that wrote
it. The conda-written forcing is read fine by CTSM's lmod-linked netcdf — exactly
as the v4 files (written by yet another netcdf build) are read today. So the
pipeline's netcdf need **not** match CTSM's; conda only has to be internally
consistent (it is, automatically). This is what makes the conda-compiler choice
above safe: nothing here has to line up with the model's libraries.

**Decided (2026-07-29): conda-current versions + tolerance.** The package layer
comes from conda's current versions (not renv-pinned), accepting a newer `r-base`
than 4.0.5 — so Run 1 is validated against v4 by a *tolerance* comparison, not
bit-for-bit. This is acceptable because the fqc-partitioned comparison (Stage 5)
isolates pipeline correctness (measured timesteps → near-exact) from library drift
(gap-filled timesteps → tolerance). renv-pinned stays the fallback only if that
comparison fails. The exact `r-base`/package versions fall out of the conda solve.

---

## 6. Stage 3 — Point the pipeline at the pre-staged data (offline)

The stock script auto-downloads; we make it read the transferred `DirDnld`
instead. The edits are localized (a few call sites + one flag), not a rewrite.

- **`DirDnld` is the seam.** Gate the download calls behind a `doDnld` flag —
  `TRUE` on the laptop (download mode), `FALSE` on HiPerGator (offline).
- **EC bundle (`DP4.00200.001`) is already split.** The script does
  `zipsByProduct(... savepath=DirDnld)` then, separately,
  `stackEddy(filepath=DirDnld/filesToStack00200/)`. `stackEddy` reads local files
  and hits no API — so just skip/guard the one `zipsByProduct` line; `stackEddy`
  runs as-is. Nearly free.
- **Met products (`DP1.*`) use combined `loadByProduct`** (download+stack, always
  contacts the API). Swap each to `stackByTable()` pointed at the pre-downloaded
  zips — because `loadByProduct` ≡ `zipsByProduct` + `stackByTable`, and only
  `stackByTable` is offline. Small mechanical edit (mind the return-shape
  difference: `loadByProduct` returns a named list; `stackByTable` writes/loads
  stacked tables).
- **`MethOut="local"`** — the script's default uploads results to *NEON's* GCS
  bucket with a credential file we don't have. Force local. (Likely a one-line
  script edit; may not be covered by the env-var override.)

Exact line edits against the current script are a follow-up (deliberately not
pinned here while the plan is still forming).

---

## 7. Stage 4 — Run the pipeline (the two runs)

`flow.api.clm.R` has a config block near the top (~lines 88–115) and honors a
`METHPARAFLOW` env-var override (`SITE`, `DATEBGN`, `DATEEND`, `DIROUT`, `LOWMEM`)
— prefer the env path over editing the script.

| Parameter | Set to | Note |
|---|---|---|
| `Site` / `SITE` | `"OSBS"` | pulls OSBS site metadata (lat/lon, height → `ZBOT`) |
| `dateBgn` / `dateEnd` | per run (Stage 3 table) | Run 2 caps `dateEnd=2025-06-30` (released-only) |
| `MethOut` | `"local"` | not GCS |
| `DirDnld` | the transferred cache | offline input |
| `DirOut` / `DIROUT` | per-run dir | `.../validation_2018-2024`, `.../full_2016-2025` |
| `lowmem` / `maxmonths` | raise if needed | a multi-year run can OOM |

**Order:** Run 1 (validation) first — it's the fastest path to "does the pipeline
work," and produces the v4-comparison set. Then Run 2 (full record). Run these as
SLURM batch jobs (single node; size memory to `lowmem`/`maxmonths`), not on a
login node.

---

## 8. Stage 5 — Verify (on HiPerGator)

Because processing is on HPG, the v4 reference is already local — the validation
loop closes without any upload.

- **Sanity (both runs):** file counts (84 / 107, contiguous months); each file
  opens with the 8 vars, gregorian, 48/day; **no `-9999` in physical vars**
  (a leftover fill = an unfilled gap that will wreck the CTSM run); physical
  ranges (`TBOT` ~250–320 K, `PSRF` ~95–103 kPa, `RH` 0–100 %, `FLDS` ~150–500 W/m²);
  2018 `TBOT` sane (~269–307 K, not the old 562 K unit artifact).
- **Reproduce-v4 (the go/no-go gate) — fqc-partitioned comparison.** Diff Run 1
  against the v4 files already at `data/datm/neon_OSBS/v4/OSBS/`. Our files and v4
  come from the same generator over the same range, so grids/vars/units are
  identical — a clean element-wise diff (no regrid/time-align). **Partition every
  timestep by its `<VAR>_fqc` flag:**
    - **Measured (fqc=0):** same raw data + deterministic conversions → must match
      v4 to ~machine precision *regardless of library versions*. Proves the pipeline
      logic (units, structure, time, conversions) is faithful; a units bug surfaces
      here (cf. Issue #34).
    - **Gap-filled (fqc>0):** the only place library-version drift lives → the
      *tolerance* applies here, and only here.
  More diagnostic than bit-for-bit: it separates "is the code correct?" from "how
  much does the gap-fill drift?" **Reference band** for acceptable filled-point
  drift = the I2 v4-vs-v3 sanity check (RMS Δ TBOT 0.17 K, PSRF 9 Pa, FSDS
  0.07 W/m², RH 0.21 %, WIND 0.025 m/s); measured points should sit far below it.
  **Tools:** CPRNC (`CTSM_CPRNC_Deterministic_Analysis.md`), NCO `ncdiff`, or xarray.
  **Build it as a committed regression script** — the forcing analog of
  `merit_regression.py`: a `neon_v4_regression` that reports the fqc-partitioned
  per-variable stats with a pass/fail, re-runnable after any env change or record
  extension. Pass → pipeline trusted; fail → fall back to the renv-pinned build.
- **Output stays on HiPerGator** at the curated forcing dir and feeds CTSM directly
  (the `user_nl_datm_streams` `datafiles` override, per Phase I §9 / I6). No
  hand-back / upload step — that's the point of building here.

---

## 9. Friction points / open risks

1. **Version fidelity — DECIDED (2026-07-29): conda-current + tolerance.** conda
   serves current package versions, not the 2023 RSPM pins, so Run 1 is validated
   against v4 by *tolerance*, not bit-for-bit. Accepted, because the fqc-partitioned
   comparison (Stage 5) isolates pipeline correctness (measured timesteps →
   near-exact) from library drift (gap-filled timesteps → tolerance).
   `renv::restore()` against the RSPM snapshots is the fallback *only if* that
   comparison fails to clear the reference band.
2. **`r-base` newer than 4.0.5 accepted (consequence of #1).** conda-forge R
   packages are built against a specific R ABI, and the 2023-era versions target
   `r-base` **4.2–4.4**, so `r-base=4.0.5` + the pins is likely an unsolvable conda
   solve; the env lands on newer R + drifted packages. **Verify eddy4R.base/qaqc
   build against the resulting R** (they need only R≥3.4, so this is expected fine).
3. **conda-for-everything — DECIDED (chose (a)).** conda supplies all 185 at current
   versions; the conda-foundation + `renv::restore()` pinned tree (option (b): max
   fidelity, renv-in-conda friction, long source compiles) is retained only as the
   fallback if #1's comparison fails, or if the PI later needs a citable,
   exactly-reproducible build.
4. **Compiler ABI.** Shelf-3 source builds need a toolchain ABI-matched to conda's
   R — use conda-forge compilers, not lmod gcc (§5).
5. **Channel mixing.** `rhdf5` forces bioconda alongside conda-forge; strict
   channel priority is a classic source of unsolvable / subtly-broken R envs.
6. **eddy4R version drift.** The lock pins SHA `898a72d`; current `master` has
   diverged (adds an `R6P` import to `eddy4R.base`). Install with `ref="898a72d"`
   to match, or take current and adjust.
7. **Loose ends.** `mlegp` is in the script's install loop but not in the lockfile
   and not needed by pinned `REddyProc 1.3.2` — likely unused; confirm before
   treating as required. `metScanR` / `prism` are in the lock but standalone —
   verify the OSBS path actually calls them.
8. **The script is research code.** Hardcoded `/home/ddurden/` paths, a
   `# CHANGE ME` marker, and an install block that disagrees with `renv.lock`
   (it `library(eddy4R.base)` without installing it — expected from the Docker base
   image, which we're replacing with the conda env, so we must install it
   ourselves).

---

## 10. What changed from the original local-only plan

The first version of this doc ran the **whole** pipeline (download + process) on a
laptop and shipped the finished NetCDF up. This version keeps only the **raw
download** off-HPG (forced by the `/data/` block) and moves the **environment and
all processing onto HiPerGator** — so the pipeline is reproducible where the PI and
future work live, and the output lands next to CTSM with no upload step. The cost
is standing up the conda-hybrid R environment on HPG (Stage 2), which the
laptop-Docker path avoided.

---

## References

- Phase record + task list: `swenson/phases/I-neon-forcing.md` (I3–I5; Research
  notes §3 pipeline, §7 the `/data/` block, §8 drop-in analysis, §9 producer/
  consumer contract).
- Verified DP catalog: `swenson/docs/neon-data-products.md`.
- Pre-built v4 (Run 1 reference): `swenson/data/datm/neon_OSBS/v4/OSBS/`.
- NCAR-NEON pipeline: https://github.com/NEONScience/NCAR-NEON
  (`TowerTools_ForcingData/flow.api.clm.R`, `renv.lock`, `gapFilling/pack/NEON.gf`,
  Wieder et al. 2023, *GMD* 16, 5979–6000).
- eddy4R: https://github.com/NEONScience/eddy4R (`pack/eddy4R.base`,
  `pack/eddy4R.qaqc`).
- HiPerGator: conda envs — `docs.rc.ufl.edu/software/conda_creation`; bulk transfer
  — `docs.rc.ufl.edu/data_transfer/globus_transfer`.
