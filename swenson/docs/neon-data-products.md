# NEON Data Products for OSBS — Catalog

Notes on NEON Data Products relevant to our OSBS CTSM single-point work.
Atmospheric forcing is the active concern (replacing CRUNCEPv7 in
`osbs.swenson.spinup`). Vegetation/soil products are secondary, listed
here for later reference.

Existing NEON data in pipeline: OSBS 2023-05 collection, **RELEASE-2026**
(see `data/neon/README.md`). Atmospheric products below are also
RELEASE-2026 at OSBS — no release divergence.

Years listed are approximate site availability based on NEON sensor
commissioning. Verify per-product via the NEON API:
`https://data.neonscience.org/api/v0/products/<DPNUM>/sites/OSBS/RELEASE-2026`.

---

## Primary — atmospheric forcing

### Tier 1: Core DATM forcing (7 CTSM-required variables)

| DP Number | Product | DATM var | Resolution | OSBS years |
|---|---|---|---|---|
| DP1.00002.001 | Single Aspirated Air Temperature | TBOT | 1-min, 30-min | 2014– |
| DP1.00003.001 | Triple Aspirated Air Temperature (tower top, redundant) | TBOT QC | 1-min, 30-min | 2014– |
| DP1.00045.001 | Precipitation — Tipping Bucket (secondary) | PRECTmms | 1-min, 30-min | 2014– |
| DP1.00023.001 | Shortwave & Longwave Radiation (NR01) | FSDS, FLDS | 1-min, 30-min | 2013– |
| DP1.00004.001 | Barometric Pressure | PSRF | 1-min, 30-min | 2014– |
| DP1.00001.001 | 2D Wind Speed & Direction | WIND | 2-min, 30-min | 2013– |
| DP1.00098.001 | Relative Humidity | RH → QBOT/SHUM | 1-min, 30-min | 2015– |

Caveats:
- DP1.00044.001 (primary weighing-gauge precip) is NOT installed at OSBS.
- OSBS 2018 TBOT has a documented anomaly (NCAR-NEON Issue #34); inspect/flag before gap-filling.

### Tier 2: CO2

| DP Number | Product | Why | OSBS years |
|---|---|---|---|
| DP4.00200.001 | Bundled Eddy Covariance | Embeds CO2 mole fraction (storage + turbulent) at multiple tower heights; carries redundant T/RH/WIND for QC | 2016– |

Standalone CO2 DPs (DP1.00034.001, DP1.00099.001) are still FUTURE
status. Use the bundle.

### Pre-built forcing — what's available out of the box

| Source | OSBS years | Format | Effort |
|---|---|---|---|
| CTSM `run_tower OSBS` | **2018–2021** | CTSM-DATM NetCDF (shipped in `$BLUE/ctsm5.3/tools/site_and_regional/`) | One command |
| AmeriFlux US-xSB | 2019–2024 | FLUXNET BASE | Column rename + NetCDF conversion |
| NCAR-NEON pipeline (raw) | 2014–2024 (in principle) | NetCDF after running pipeline | Setup + per-year processing |
| PLUMBER2 | **NOT INCLUDED** (confirmed via Ukkola 2022 site list) | — | — |

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

Shortest-path alternative: CTSM `run_tower OSBS` uses the **2018–2021
pre-built** NEON forcing already shipped in the fork at
`$BLUE/ctsm5.3/tools/site_and_regional/`. Run that first to validate the
end-to-end workflow before investing in extending the record with the
full pipeline.
