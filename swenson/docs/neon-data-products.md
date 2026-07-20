# NEON Data Products for OSBS — Catalog

Notes on NEON Data Products relevant to our OSBS CTSM single-point work.
Atmospheric forcing is the active concern (replacing CRUNCEPv7 in
`osbs.swenson.spinup`). Vegetation/soil products are secondary, listed
here for later reference.

Existing NEON data in pipeline: OSBS 2023-05 collection, **RELEASE-2026**
(see `data/neon/README.md`). Atmospheric products below are also
RELEASE-2026 at OSBS — no release divergence. **The released cut ends 2025-06;
the most recent 12 months (2025-07 → 2026-06) are PROVISIONAL**, not part of
RELEASE-2026.

OSBS start months below were **verified against the live NEON API on 2026-07-15**
(`https://data.neonscience.org/api/v0/products/<DPNUM>`). All OSBS tower met
products commission 2014-08; nothing predates it.

---

## Primary — atmospheric forcing

### Tier 1: Core DATM forcing (7 CTSM-required variables)

| DP Number | Product | DATM var | Resolution | OSBS start |
|---|---|---|---|---|
| DP1.00003.001 | Triple Aspirated Air Temperature (tower top) | TBOT | 1-min, 30-min | 2014-08 |
| DP1.00002.001 | Single Aspirated Air Temperature (all levels) | TBOT (alt) | 1-min, 30-min | 2014-08 |
| DP1.00044.001 | Precipitation — Weighing Gauge (primary) | PRECTmms | 1-min, 30-min | 2016-09 |
| DP1.00045.001 | Precipitation — Tipping Bucket (secondary) | PRECTmms | 1-min, 30-min | 2016-08 |
| DP1.00023.001 | Shortwave & Longwave Radiation (net radiometer) | FSDS, FLDS | 1-min, 30-min | 2014-08 |
| DP1.00004.001 | Barometric Pressure | PSRF | 1-min, 30-min | 2014-08 |
| DP1.00001.001 | 2D Wind Speed & Direction | WIND | 2-min, 30-min | 2014-08 |
| DP1.00098.001 | Relative Humidity | RH → Sa_rh | 1-min, 30-min | 2015-06 |

TBOT source: the NCAR-NEON pipeline uses the **triple-aspirated** product
(DP1.00003.001, NEON's tower-top primary reference); single-aspirated
(DP1.00002.001) is an alternative available at all tower levels. Humidity: the
forcing file carries **RH (%)**; CDEPS converts RH → specific humidity internally
(`components/cdeps/datm/datm_datamode_clmncep_mod.F90`), so no `QBOT` is stored.
The pipeline additionally ingests **DP1.00024.001 (PAR)** and **DP1.00014.001
(direct/diffuse shortwave)** as gap-fill/partitioning inputs — these are not CTSM
output variables.

Caveats:
- **DP1.00044.001 (weighing-gauge precip) IS installed at OSBS** (2016-09→,
  RELEASE-2026) and is the pipeline's primary precip source. (Corrected
  2026-07-15 — the live NEON API and the pipeline both use it; the earlier "NOT
  installed" claim was wrong.)
- OSBS 2018 TBOT: NCAR-NEON Issue #34 reported an unphysical ~562 K value in the
  **v1** files — a Celsius→Kelvin unit artifact, fixed upstream by reprocessing
  (2021). Verified sane in the on-disk v3 set (2018 TBOT 269–307 K), so it is
  **not** a concern for pre-built v3/v4; only relevant if pulling raw/v1-era data.

### Tier 2: CO2

| DP Number | Product | Why | OSBS start |
|---|---|---|---|
| DP4.00200.001 | Bundled Eddy Covariance | Embeds CO2 mole fraction (storage + turbulent) at multiple tower heights; carries redundant T/RH/WIND for QC | 2017-02 |

Standalone CO2 DPs (DP1.00034.001, DP1.00099.001) are still FUTURE
status (no OSBS data). Use the bundle.

**CO2 is not a forcing-file variable.** CTSM delivers CO2 via a separate
mechanism — the constant `CCSM_CO2_PPMV` or the `co2tseries.*` DATM stream
(`DATM_CO2_TSERIES`), never inside the met file. Tower CO2 (in the bundle) is a
validation target, not a forcing input. See `phases/I-neon-forcing.md` §6.

### Pre-built / alternative forcing sources surveyed

| Source | OSBS coverage | Format | Status |
|---|---|---|---|
| NCAR-NEON pre-built forcing | **v4: 2018-01 → 2024-12** (84 files) | CTSM-DATM NetCDF, gap-filled on NEON's cloud | **The path forward.** Server has v1–v4; v3 (2018→2024-06) already on disk. Exceeding v4 requires a custom pipeline run (raw record reaches 2016-08 → 2025-06 released). |
| CTSM `run_tower OSBS` | 2018–2021 as shipped | fetches the NCAR-NEON files above | The "2018–2021" is a **namelist year-cap** (`NEONVERSION=v2`, `DATM_YR_END=2021`), NOT a data limit — the same mechanism reaches v4/2024-12. The earlier "tested insufficient" verdict traced to this cap. |
| AmeriFlux US-xSB (= OSBS) | BASE 2018–2024; FLUXNET 2019–2024 | AmeriFlux BASE / FLUXNET-1F | Optional cross-check; different gap-fill provenance |
| PLUMBER2 | — | — | **Does NOT include OSBS** (confirmed — 170 sites from FLUXNET2015 + La Thuile + OzFlux, all pre-NEON; Ukkola et al. 2022, ESSD 14, 449) |

---

## Secondary — vegetation, soil, AOP, validation

Set aside per PI priority. Listed for later reference if/when PFT
customization or validation work becomes active.

### Vegetation structure / PFT customization

| DP Number | Product | Use |
|---|---|---|
| DP1.10098.001 | Vegetation structure (tree DBH, height, species, stems) | PFT identity, canopy height |
| DP1.10058.001 | Plant presence and percent cover | Groundcover/understory splits |
| DP1.10026.001 | Plant foliar physical & chemical properties | Leaf traits (mass/area, N, P) |
| DP1.10055.001 | Plant phenology observations | Ground-truth phenology |
| DP1.10033.001 | Litterfall + fine woody debris production/chemistry | Carbon flux validation |

Years: roughly 2013–present (varies per protocol).

### Belowground / soil

| DP Number | Product | Use |
|---|---|---|
| DP1.10047.001 | Soil phys/chem properties — megapit | Deep soil column setup (one-time, ~2014) |
| DP1.10086.001 | Soil phys/chem properties — periodic | Shallow texture/C/N |
| DP1.10066.001 | Root biomass & chemistry — periodic | Below-ground C |
| DP1.10067.001 | Root biomass & chemistry — megapit | Deep root profiles (one-time, ~2014) |
| DP1.10078.001 | Soil microbe biomass | BGC pool validation |
| DP1.10100.001 | Soil stable isotopes | Process-level validation |

Years: periodic products 2014–present; megapit products one-time only.

### AOP remote sensing (1m airborne, same flight series as our DTM)

| DP Number | Product | Use |
|---|---|---|
| DP3.30015.001 | Canopy Height Model (CHM) | Spatial canopy height |
| DP3.30016.001 | Digital Surface Model (DSM) | Top-of-canopy elevation |
| DP3.30012.001 | LAI (tile-mosaicked) | Spatially-resolved LAI |
| DP3.30014.001 | fPAR | Light absorption |
| DP3.30011.001 | Albedo (spectrometer-derived) | Validate CTSM albedo |
| DP3.30018.001 | Canopy nitrogen | Foliar N spatial map |
| DP3.30026.001 | Vegetation indices (NDVI etc.) | Greenness |
| DP3.30019.001 | Canopy water indices | Equivalent water thickness |
| DP3.30006.001 | Spectrometer L3 reflectance | Hyperspectral input |
| DP3.30010.001 | High-res RGB orthomosaic | Visual reference |

OSBS AOP flight years: 2014, 2017, 2018, 2019, 2021, 2023. (Verify; OSBS
is annual cadence in NEON D03.) Our LIDAR DTM/slope/aspect already in
pipeline is from the 2023-05 flight.

### Validation tower products (model output evaluation)

| DP Number | Product | Use |
|---|---|---|
| DP1.00094.001 | Soil water content (multi-depth) | Validate CTSM `H2OSOI` |
| DP1.00041.001 | Soil temperature (multi-depth) | Validate CTSM `TSOI` |
| DP1.00040.001 | Soil heat flux | Energy balance check |
| DP1.00095.001 | Subcanopy soil CO2 (multi-depth) | CH4/CO2 efflux validation |

Years: 2014–present (varies by sensor commissioning).

---

## NCAR-NEON pipeline — stitching into continuous CTSM forcing

Repo: https://github.com/NEONScience/NCAR-NEON
Paper: Wieder et al. 2023, *Geosci. Model Dev.* 16, 5979–6000.
DOI: https://doi.org/10.5194/gmd-16-5979-2023

What it does:
- Pulls raw NEON tower data via the NEON API
- Gap-fills using `ReddyProc` (R package; marginal distribution sampling, Reichstein 2005)
- Writes CTSM-formatted NetCDF — monthly files, single-point grid, DATM stream conventions

Workflow (broad strokes):
1. Clone repo to `$BLUE/ncar-neon` (or similar)
2. Install R + `ReddyProc` + dependencies (conda-friendly per repo instructions)
3. Configure site metadata — lat/lon, canopy height, measurement height. OSBS is pre-configured.
4. Run pull scripts — fetches DP1.* CSVs from NEON API for target year range
5. Run gap-fill scripts — ReddyProc fills missing intervals
6. Run packaging scripts — writes NetCDF with TBOT, PRECTmms, FSDS, FLDS, PSRF, WIND, QBOT, etc.

Caveats:
- 2018 TBOT (Issue #34) is anomalous — flag manually before gap-filling
- Long gaps (multi-week) gap-fill but quality degrades — flag downstream
- Tower reference height (`ZBOT`) hard-coded per site; verify matches actual OSBS instrument height

### Pipeline installation on HiPerGator

No compile step — NCAR-NEON is interpreted R with an `renv.lock`-pinned
dependency tree (**195 packages**, R 4.0.5, including Bioconductor (`rhdf5`) and
GitHub-only (`eddy4R`) sources — not CRAN-only). Three install paths, in order of
recommendation:

**1. HiPerGator R module + `renv` (recommended)**

```bash
module spider R                     # find available R versions
module load R/<version>             # pick closest to NCAR-NEON's renv.lock R version
cd $BLUE
git clone https://github.com/NEONScience/NCAR-NEON
cd NCAR-NEON
R -e 'install.packages("renv"); renv::restore()'
```

`renv::restore()` is the long step — 20–60 min on first run while CRAN
packages with C/Fortran dependencies compile. Fewest moving parts on
HiPerGator. If module R version diverges from `renv.lock`, renv warns;
override only if necessary.

**2. Conda `r-base` + `renv` (fallback)**

```bash
conda create -n ncar-neon -c conda-forge r-base r-essentials
conda activate ncar-neon
cd $BLUE && git clone https://github.com/NEONScience/NCAR-NEON
cd NCAR-NEON
R -e 'install.packages("renv"); renv::restore()'
```

Isolated env, but renv inside conda can be finicky around system libs
(`ncdf4`, anything spatial). Use only if Path 1 hits dependency issues.

**3. Apptainer from upstream Dockerfile (last resort)**

HiPerGator doesn't run Docker but Apptainer can build from a Dockerfile.
Most reproducible — bit-identical to upstream — but most setup effort
(non-root build, writable overlays, bind mounts for `$BLUE`). Only
worth it if exact reproducibility against the Wieder 2023 GMD paper is
required.

**Not viable:** there is no conda/PyPI/CRAN package for NCAR-NEON
itself. `ReddyProc` (the gap-filling library underneath) is on conda as
`r-reddyproc`, but bypassing NCAR-NEON means reimplementing the
NEON-API → ReddyProc → CTSM-NetCDF glue.

---

## Local data layout (raw + processed)

Recommended directory structure for downloaded NEON met data and
processed CTSM-DATM output. Mirrors the existing `data/neon/{dtm,slope,aspect}`
pattern (raw products by type) and adds a sibling `data/datm/` for
processed model-input format.

```
data/
├── neon/
│   ├── dtm/                       # existing
│   ├── slope/                     # existing
│   ├── aspect/                    # existing
│   ├── met/                       # NEW — raw NEON met downloads (gitignored)
│   │   ├── DP1.00001.001/         # one dir per DP, NEON API naming
│   │   ├── DP1.00002.001/
│   │   ├── DP1.00004.001/
│   │   ├── DP1.00023.001/
│   │   ├── DP1.00045.001/
│   │   ├── DP1.00098.001/
│   │   └── DP4.00200.001/
│   └── README.md                  # update to catalog the new section
├── datm/                          # NEW — CTSM-DATM-ready output (gitignored)
│   └── neon_OSBS/
│       └── clmforc.neon.OSBS.{Prec,Solr,TPQWL}.YYYY-MM.nc
└── .gitignore                     # extend to ignore neon/met/, datm/
```

### Filename convention

Mirror CRUNCEPv7's 3-stream split exactly so the DATM namelist swap is a
drop-in path change:

| New (NEON-derived) | Existing (CRUNCEPv7) | Variables |
|---|---|---|
| `clmforc.neon.OSBS.Prec.YYYY-MM.nc` | `clmforc.cruncep.V7.c2016.0.5d.Prec.OSBS.YYYY-MM.nc` | PRECTmms |
| `clmforc.neon.OSBS.Solr.YYYY-MM.nc` | `clmforc.cruncep.V7.c2016.0.5d.Solr.OSBS.YYYY-MM.nc` | FSDS |
| `clmforc.neon.OSBS.TPQWL.YYYY-MM.nc` | `clmforc.cruncep.V7.c2016.0.5d.TPQWL.OSBS.YYYY-MM.nc` | TBOT, PSRF, WIND, QBOT, FLDS |

Existing CRUNCEPv7 forcing lives at `/blue/gerber/sgerber/CTSM/subset_input/datmdata/`.

### Raw data preservation

Keep two layers under each `data/neon/met/DP*/`:
- **Pristine zips** from NEON API — the source of truth (downloads timestamps drift; preserve as-received)
- **Extracted/concatenated CSVs** — intermediate, regeneratable from zips; safe to wipe

### Gitignore additions

Extend `data/.gitignore` from:

```
*.tif
*.nc
```

to:

```
*.tif
*.nc
neon/met/
datm/
```

Directory-level ignores cover us against unexpected file formats from the NEON API (zip, CSV, JSON metadata, etc.).

### Per-directory READMEs

Provenance traceability — at minimum:
- `data/neon/met/README.md`: DPs downloaded, year range, NEON API call snippet used, RELEASE-2026 confirmation per product
- `data/datm/neon_OSBS/README.md`: NCAR-NEON pipeline commit hash, ReddyProc version, years with anomalies flagged (e.g. Issue #34 OSBS 2018 TBOT), conversion script path

Without these, six months from now nobody can tell whether the NetCDFs were generated with the buggy 2018 TBOT or with it patched.

### Open question — final storage location

Two options for the processed DATM output:
- `swenson/data/datm/` — co-located with this project, recommended initially
- `$BLUE/datm_neon/` (or `/blue/gerber/sgerber/CTSM/subset_input/datmdata_neon/`) — sibling to other shared data, available to any future OSBS case

Start swenson-local; promote to a shared location once a second case needs it. Easier to move data once the access pattern is known than to predict it.
