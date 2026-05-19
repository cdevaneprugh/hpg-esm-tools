# Phase H: Enable Lateral Subsurface Flow

Status: Track A complete. **Tracks B and C on hold; there is a good
chance routing-on is not pursued at all** now that the 2026-05-19
routing-gate audit (Section 8) showed inter-column lateral flow
already runs under `use_hillslope=.true.` The remaining tasks
(A5–A6, B1–B4, C1–C4) are all contingent on a decision to run
routing-on production, which is itself contingent on what Phase F
shows.
Depends on: Phase F (long spinup provides the evidence to decide
whether routing-on is worth pursuing)
Blocks: nothing within current project scope

## Problem

The DOE project's central scientific question — terrestrial-aquatic
interface (TAI) dynamics in coastal-plain wetlandscapes — requires the
following chain of physics to operate:

```
upland saturation
  → lateral subsurface flow between hillslope columns
  → water table rise in low-HAND columns near the lake
  → soil becomes anoxic (o_scalar drops)
  → aerobic decomposition suppressed
  → CH4 production rises (finundated rises)
```

CTSM has two hillslope namelist switches. **The 2026-05-19 routing-gate
audit (Section 8) revised this table — the earlier framing that
routing toggles inter-column lateral flow was wrong.**

| Switch | What it actually toggles | Current state |
|---|---|---|
| `use_hillslope` | Multi-column subgrid structure with per-column geometry, slope, aspect, AND **inter-column lateral subsurface flow** via Darcy gradient at the water table and perched water table (`SoilHydrologyMod.F90:1703-1987` (`PerchedLateralFlow`) and `:2087-2623` (`SubsurfaceLateralFlow`), dispatched unconditionally from `HydrologyDrainageMod.F90:139-148`). | **ON** |
| `use_hillslope_routing` | Stream-side state inside CTSM: channel geometry init (`HillslopeHydrologyMod.F90:378-507`), `stream_water_volume` ledger and Manning streamflow (`HillslopeStreamOutflow` + `HillslopeUpdateStreamWater`, called only at `HydrologyDrainageMod.F90:150-158`), swap of the terminal-column boundary depth from MOSART's `tdepth_grc` to CTSM-internal stream state (`SoilHydrologyMod.F90:1822, 2265`), losing-stream outflow cap (`:2362`), `VOLUMETRIC_STREAMFLOW` hist registration (`WaterFluxType.F90:525`), and lnd→rof streamflow export (`lnd2atmMod.F90:343`). **Does NOT gate whether inter-column lateral flow runs.** | **OFF** |

So the TAI-physics chain has two stages:

- **Inter-column lateral flow is already active under
  `use_hillslope=.true.`** in every `osbs.*` case on this machine.
  Phase F column-level water-table and soil-moisture differences are
  already partly the product of lateral redistribution, not solely
  independent per-column water balance.
- **Stream-coupling boundary condition at the bottom of the chain**
  is what Phase H turns on. Under routing-off the terminal column
  sees `tdepth_grc` from the coupler (defaults to 0 if MOSART doesn't
  send `Sr_tdepth`; see `lnd_import_export.F90:619-624`). Under
  routing-on it sees an internal stream whose volume is built up
  from column drainage and drained via Manning's equation.

This reframes the 2026-05-12 A4 smoke test result: gridcell aggregates
bit-identical and column-level offsets tiny at year 5 are expected
when both cases run lateral flow identically; the offsets isolate the
terminal-column boundary condition, not "lateral flow on vs off."

All ~22 osbs cases on this machine (sgerber's + ours) have run
routing-off. The Phase H deliverable is narrower than originally
framed: validate that adding internal stream state produces a
physically reasonable BC at the lake/terminal column interface, and
quantify whether it materially changes the TAI signal already present
under routing-off.

## Technical blocker: the spval bug

Switching `use_hillslope_routing = .true.` exposes a CTSM bug that
silently produces garbage in NUOPC single-point mode.

### Source code chain (verified 2026-05-11 against CTSM 5.3.085)

**Origin: single-column initialization** (`src/cpl/share_esmf/lnd_set_decomp_and_domain.F90:342`):

```fortran
ldomain%area(1) = spval        ! spval ≈ 1e36, an uninitialized sentinel
```

This code path triggers when CTSM is configured via `PTS_LAT`/`PTS_LON`
(single-column NUOPC mode), which is how every osbs case on this
machine is set up.

**Propagation to grc** (`src/main/initGridCellsMod.F90:179`):

```fortran
grc%area(gdc) = ldomain%area(gdc)
```

So `grc%area(g) = spval` for the duration of the run. Confirmed
empirically — h0a output for our cases shows `area = 9.999...e+35`.

**Routing-gate block** (`src/biogeophys/HillslopeHydrologyMod.F90:475`):

```fortran
if (use_hillslope_routing) then
```

This gate protects all uses of `grc%area` that would otherwise blow up
under spval. While the gate is closed (routing off), nothing
downstream is corrupted.

**The poisoned multiplication** (`src/biogeophys/HillslopeHydrologyMod.F90:486-487`):

```fortran
nhill_per_landunit(nh) = grc%area(g)*1.e6_r8*lun%wtgcell(l) &
     *pct_hillslope(l,nh)*0.01/hillslope_area(nh)
```

With `grc%area = spval`, this gives `nhill_per_landunit ≈ 1e36` —
physically nonsense. `nhill_per_landunit` is meant to be "how many
copies of the representative hillslope tile fit in this gridcell."

**Downstream cascade** (`src/biogeophys/HillslopeHydrologyMod.F90:501-502`):

```fortran
lun%stream_channel_length(l) = lun%stream_channel_length(l) &
     + col%hill_width(c) * 0.5_r8 * nhill_per_landunit(col%hillslope_ndx(c))
```

`stream_channel_length` inherits the 1e36 magnitude.

**Secondary exposure, indirectly protected**
(`src/biogeophys/HillslopeHydrologyMod.F90:1117-1121` in
`HillslopeUpdateStreamWater`):

Three more `grc%area` multiplications, gated by an `active_stream`
flag rather than `use_hillslope_routing` directly. `active_stream`
requires `stream_channel_length > 0`, which is itself set inside the
routing gate. So under routing-off, `stream_channel_length` stays 0,
`active_stream` stays false, and the secondary multiplications never
execute. **Fragile coupling worth flagging:** if any future code path
sets stream-channel properties from outside the routing gate, the
spval would propagate.

**No defensive guards.** A search across all of `src/` found zero
validation of `grc%area` against `spval` — no asserts, no error
checks, no fallback. CTSM trusts that `grc%area` is always real, which
held under the old MCT coupler but not under NUOPC single-point.

## External context

**CTSM Issue #1432** — "Area is not set for NUOPC single point cases"
(https://github.com/ESCOMP/CTSM/issues/1432). Open since
2021-07-20, assigned to @jedwards4b, zero comments, no PR. From
@ekluzek's original report:

> "No significant scientific errors unless area is required (such as
> in hillslope modeling)."

The CTSM core team has known about this exact problem for nearly five
years. The bug is not in any release note, tech note, or user guide.
The community has not encountered it in practice because Swenson's
hillslope users run gridded (the published mode), and single-point
users have run with routing off.

**Swenson 2025 paper** (Swenson & Lawrence 2025, JAMES,
https://doi.org/10.1029/2024MS004410). The published hillslope dataset
ships at **0.9°×1.25° global only**. Routing-on validation was at
gridded resolution. No documented single-point routing-on case exists.

**PR #1715** — original landing of the seven hillslope namelist
options (merged Feb 2024 by @swensosc). Validation described as
"twenty year simulations to check for water and energy balance
errors" — gridded only.

**DiscussCESM thread, Johanna Teresa, Feb 2025**
(https://bb.cgd.ucar.edu/cesm/threads/point-scale-simulation-with-ctsm5-3.11125/).
The only public thread on this combination. Teresa quotes Swenson's
direct recommendation:

> "In the meantime I will also try and generate a mesh for the single
> point location following a suggestion from Sean Swenson."

The mesh-generation suggestion came from Swenson personally and was
specifically tied to her intent to use hillslope hydrology. The thread
resolves on other build issues; mesh-for-hillslope is mentioned but
never fully demonstrated publicly.

## Approach

Mesh-mode migration for OSBS. **Strict OSBS scope** — we are not
pursuing an upstream CTSM PR. The bit-for-bit regression burden, 2–4
month review cycle, and uncertainty about the design choice the
CTSM team will accept are not justified for the OSBS deliverable.
Future contribution remains an option after the science is done.

The mechanics: replace the `PTS_LAT`/`PTS_LON` single-column shortcut
with an explicit single-cell ESMF mesh (`LND_DOMAIN_MESH`).  CTSM's
mode-decision logic then routes through `lnd_set_decomp_and_domain_from_readmesh`
instead of `lnd_set_decomp_and_domain_for_single_column`, and
`ldomain%area` is populated from the mesh via
`ESMF_FieldRegridGetArea()` — a real number, not spval.

This is what Swenson recommends to single-point users who want
hillslope work. The tooling (`tools/site_and_regional/mesh_maker`)
exists in the CTSM tree.

Three sub-tracks, run roughly sequentially:

A. **Software** — Pure engineering. No PI input needed.
B. **Scientific decisions** — Need PI consultation before software
   work is meaningful. The gridcell-area choice is central.
C. **Validation** — Short test runs, then long-spinup decision.

## Research notes (2026-05-12): input data, NEON, gridcell area, mesh tooling

Deep research pass on Phase H prerequisites. Five strands: existing
inputs we use, CTSM's built-in tower-run pathway, NEON OSBS data
products, scale reconciliation across our hillslope file / NEON site
extent / global inputs, and the mesh-creation tool.

### 1. Existing input data inventory

Our case (`osbs.swenson.spinup`) mirrors sgerber's setup:

| Component | Source | Implied scale | Notes |
|---|---|---|---|
| COMPSET | `1850_DATM%CRUv7_CLM60%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP` | — | PTS mode (`ATM_DOMAIN_MESH=UNSET`) |
| Surface dataset | `surfdata_OSBS_hist_1850_78pfts_c251002.nc` (52 KB) | f09 cell ≈ 12,654 km² | Subset from `fv0.9x1.25_141008_polemod_ESMFmesh.nc` via `subset_data`. Raw inputs: PFT 0.25°, soil 5 min, peat 0.5°, urban 0.5°, etc. — finer than f09 but aggregated to the f09 cell at OSBS. **No AREA variable.** |
| DATM forcing | CRUNCEPv7 0.5° monthly (Prec/Solr/TPQW), 1901–1921 only (~756 files) | 0.5° cell ≈ 2,700 km² | sgerber-extracted subset; the 600-yr spinup cycles these 21 years repeatedly |
| Hillslope file | `hillslopes_osbs_production_c260505.nc` (this work) | **AREA = 90 km²** | Our 24-bin TAI + lake column; LATIXY/LONGXY at OSBS, 25 nmaxhillcol, 1 nhillslope |
| Fire pop. density | `clmforc.Li_2017_HYDEv3.2_CMIP6_hdm_0.5x0.5_AVHRR_*` | 0.5° | Overridden to match sgerber's spinup vintage |
| `use_init_interp` | `.false.` | — | Cold start from `1850_control` (current case is fresh startup) |

sgerber's `subset_input/` also contains an alternative hillslope
file (`hillslopes_osbs_c240416.nc`, 4.4 KB, **AREA = 12,654 km²**)
— that is the nearest-neighbor extract from Swenson's published
0.9°×1.25° global hillslope dataset at `lsmlat=127, lsmlon=222`.
This is what every other osbs# case on the machine uses; **only
`osbs.swenson.spinup` uses our 90-km² file.** The 12,654 km² figure
literally comes from the global gridcell area at OSBS latitude.

### 2. CTSM's built-in NEON OSBS configuration

Lives at `cime_config/usermods_dirs/clm/NEON/OSBS/` (entry point:
`tools/site_and_regional/run_tower`). Key facts:

- Coordinates: `PTS_LAT=29.68819`, `PTS_LON=278.00655` (matches our
  case exactly).
- PFT: 1 (needleleaf evergreen temperate — longleaf pine).
- Years: 2018–2021 transient (`2018-PD_transient` use case).
- Forcing: NEON tower atmospheric data via `NEONVERSION=v2`,
  `DATM_PRESAERO=SSP3-7.0`, `DATM_PRESNDEP=SSP3-7.0`.
- Compset variants: HIST (transient with NEON forcing) or non-HIST
  (`2018_control`, CALENDAR=NO_LEAP).
- **Not a drop-in for our work** — NEON tower data only goes back to
  2018, so this template is for *transient* runs, not 1850 AD spinup.
  But it's the canonical setup for any future site-specific transient
  run we may build on top of the AD spinup.

### 3. NEON OSBS data products — practical priorities for CTSM

OSBS is NEON Domain 03 (Southeast USA), AmeriFlux ID **US-xSB**, IGBP
class evergreen needleleaf (longleaf pine, ~3–4 yr burn rotation).
NEON terrestrial sampling boundary is **36.81 km²** (total property
38.5 km²); flux tower is 35 m tall; tower footprint integrates over
roughly 10⁴–10⁶ m² (~3 km² order-of-magnitude). Distributed soil
plots (5 of them) sit inside the tower airshed.

Worthwhile NEON products, in CTSM-relevance order:

| Tier | Product | NEON ID | Cadence / coverage | CTSM use |
|---|---|---|---|---|
| 1 | Soil megapit | DP1.00096 (= DP1.00097) | 1 pedon, 4 horizons (A, Bw1–3), 200 cm + 354 cm auger; texture, BD, OC, N, pH | Direct fsurdat soil profile override. Single point — limiting for spatial variation. |
| 1 | Soil temp profiles | DP1.00041 | 5 plots × multiple depths, 1-min | Validation of CTSM TSOI |
| 1 | Soil water content | DP1.00094 | 5 plots × multiple depths, 1-min, ongoing | Validation of H2OSOI |
| 1 | Soil CO₂ profiles | DP1.00095 | 5 plots | Validation of soil respiration |
| 1 | Soil chemistry (distributed) | DP1.10086 | 5-yr cadence, 5 plots | Soil OC/N init + validation |
| 1 | Root biomass + chemistry | DP1.10067 | 5-yr cadence | Root C/N validation |
| 1 | Litterfall + chemistry | DP1.10031 / DP1.10033 | Annual | NPP allocation, litter pool init |
| 1 | Soil microbial biomass | DP1.10104 | Annual | Decomposition validation |
| 1 | Soil N transformations | DP1.10080 | Annual | N-cycle validation |
| 2 | Eddy covariance bundle | DP4.00200 | 30-min, 2018–present | LH/SH/NEE validation; full DATM tower-forcing source |
| 2 | Triple-aspirated air T | DP1.00003 | Tower, 1-min | DATM TBOT |
| 2 | 2-D wind | DP1.00001 | Tower, 1-min | DATM WIND |
| 2 | Precipitation | DP1.00006 | Tower + secondary gauge | DATM PRECTmms |
| 2 | Shortwave + PAR | DP1.00014, DP1.00024 | Tower, 1-min | DATM FSDS |
| 2 | Net + LW radiation | DP1.00023 | Tower | DATM FLDS |
| 2 | Relative humidity | DP1.00098 | Tower profile | DATM QBOT |
| 2 | Barometric pressure | DP1.00004 | Tower | DATM PSRF |
| 3 | Vegetation structure | DP1.10098 | Distributed + tower plots, ~5-yr | DBH, height, biomass for PFT validation |
| 3 | Ecosystem structure (CHM) | DP3.30015 | 1 m raster, full AOP, ~annual | Canopy height for PFT mask |
| 3 | LAI from spectra | DP3.30012 | 1 m raster (AOP), ~annual | LAI override, PFT mapping |
| 3 | Hyperspectral reflectance | DP3.30006 | 1 m raster (AOP), ~annual | PFT classification at 1 m |
| 3 | LIDAR DTM/DSM | DP3.30024, DP3.30025 | 1 m | **Already used** for hillslope geometry |
| — | Aquatic / stream products | DP1.20020, DP1.20264 | — | **Not collected at OSBS** (aquatic site is FLNT, Flint River, GA) |

**Practical priorities for production-quality OSBS inputs** (orthogonal
to the gridcell-size question — these are independent upgrades to
quality of inputs, regardless of routing-on/off):

1. **Replace soil profile in fsurdat with NEON megapit (DP1.00096)** —
   one-shot win; current fsurdat uses 5×5 min WISE-aggregated values at
   f09 scale.
2. **NEON tower DATM** for any post-2018 transient runs on top of the
   AD spinup (DP4.00200 + DP1.00001/00003/00004/00006/00014/00023/
   00024/00098). Cannot replace CRUNCEPv7 for 1850 spinup itself.
3. **AOP hyperspectral + LAI** to refine PFT distribution at 1 m
   resolution within our 90 km² LIDAR domain. CTSM is currently using
   the global PFT distribution at f09 — likely a single dominant PFT.
4. **NEON soil moisture / temperature / CO₂ distributed plots** for
   column-level validation of H2OSOI / TSOI / SR. These 5 plots sit in
   the tower airshed (~3 km² subset of our 90 km² domain) — useful for
   sniff-test validation, not gridded calibration.
5. **Eddy covariance fluxes** for LH/SH/NEE validation post-spinup.

The 1850 AD spinup is unaffected by NEON data — no NEON product covers
pre-2018. NEON integration becomes valuable for the *transient* run
that follows spinup, or for fsurdat overrides (megapit, AOP PFT) that
are time-invariant.

### 4. Scale analysis and gridcell-area decision

The B1 options in their current form (90 km² / 0.17 km² / 2,700 km² /
12,654 km²) are not all equally valid. **Our pipeline's
`nhill_implicit` calibration constrains the answer.**

Pipeline-side bookkeeping (`scripts/osbs/run_pipeline.py:1559–1591`):

```
nhill_implicit = total_land_area_m2 / sum_land_bin_areas_m2
              ≈ 533    [LIDAR domain land area / rep hillslope land area]
```

CTSM-side bookkeeping (`HillslopeHydrologyMod.F90:486`):

```
nhill_per_landunit = grc%area × 1e6 × wtgcell × pct_hillslope / hillslope_area
```

For data and code to agree, `nhill_per_landunit ≈ nhill_implicit
≈ 533`. With `wtgcell ≈ 1.0` (single-landunit setup) and
`pct_hillslope = 100`, this requires:

```
grc%area [km²] = nhill_implicit × hillslope_area [km²] / wtgcell / 0.01
              ≈ 533 × 0.169 ≈ 90 km²
```

So **the gridcell area must be 90 km² for the existing hillslope file
to be self-consistent with its statistics**. The hillslope NetCDF's
`AREA` field literally records `90` for this reason. There are two
distinct ways to deviate from that, depending on whether we also
change the underlying statistics:

**(a) Rescale-only — change `nhill_implicit` without touching bin
geometry.** Keep the same 24 bins, same lake column, same per-rep
land areas. Change only the `nhill_implicit` multiplier and the
recorded `AREA`. Physical claim: "the rep hillslope's statistical
mix is representative of the new (smaller or larger) area." Cost:
~1 hour (regenerate one NetCDF). Defensible only when the existing
extent and the target extent have similar landscape statistics
(lake fraction, slope distribution, PFT mix). For OSBS, the 90 km²
LIDAR rectangle and the 36.81 km² NEON polygon are both within
the same Florida sandhill-and-depression landscape and likely close
enough.

**(b) Full re-derivation — re-extract LIDAR within the new boundary
and re-run the pipeline.** New bin areas, new lake fraction, new
slope/aspect distributions. Physical claim: "the rep hillslope's
statistics are derived from exactly the new extent." Cost depends on
boundary shape (rectangular tile-subset ≈ half day; arbitrary
polygon ≈ 1–2 days).

| Alternative gridcell area | Cheap rescale (a)? | Full re-derive (b) cost |
|---|---|---|
| 0.17 km² (single rep) | n/a — sets `nhill_implicit = 1` and loses tiling framing | n/a |
| 36.81 km² (NEON terrestrial sampling boundary) | **~1 hour, low risk** if Florida sandhill is uniform enough at OSBS | ~1–2 days (polygon clip + pipeline rerun) |
| 2,700 km² (Swenson 0.5° convention) | Rescale `nhill_implicit` 30× without changing geometry — physically dubious; rep statistics no longer represent the gridcell extent | n/a — can't acquire 2,700 km² of 1m LIDAR |
| 12,654 km² (Swenson 0.9°×1.25° paper) | 140× rescale; same problem | n/a |

Scale comparison across our data and external references:

| Reference | Area | Relation to our 90 km² gridcell |
|---|---|---|
| Single representative hillslope | ~0.17 km² | 533 tile-copies fill the gridcell |
| Lake area (NWI total in LIDAR domain) | ~10.68 km² | ~11.8% of gridcell area |
| NEON tower flux footprint | ~3 km² | Subset of gridcell; eddy-covariance representative volume |
| NEON terrestrial sampling boundary | 36.81 km² | ~41% of gridcell; NEON's "site" extent |
| NEON total property | 38.5 km² | ~43% of gridcell |
| **Our LIDAR production domain** | **90 km²** | **Self** — what the hillslope pipeline was built on |
| sgerber's hillslope subset | 12,654 km² | 140× larger; Swenson global cell at OSBS lat |
| Swenson published resolution | 12,654 km² | Same — what the paper validates against |

**Three options worth presenting to the PI:**

1. **Status quo: 90 km², no rework.** Self-consistent with existing
   hillslope NetCDF and pipeline bookkeeping. Defensible physical
   referent (the actual LIDAR-covered rectangle). Cost: zero.
2. **Rescale-only to NEON: 36.81 km², `nhill_implicit ≈ 218`,
   1-hour rework.** Reuses the existing rep hillslope and bin
   statistics; only the gridcell-area multiplier changes. Framing:
   "OSBS at the NEON-site scale, using hillslope statistics from a
   90 km² rectangular LIDAR mosaic that overlaps and contains the
   NEON polygon." Defensible if the broader 90 km² rectangle is
   statistically similar to the smaller NEON polygon — likely true
   at OSBS because the landscape is uniform.
3. **Full re-derive to NEON: 36.81 km² with NEON-polygon statistics,
   ~1–2 days rework.** Re-extract LIDAR within the NEON polygon,
   re-run pipeline. Statistically faithful to the NEON extent.
   Costs the polygon-clip engineering (the pipeline currently
   assumes rectangular tile grids).

The 2,700 km² and 12,654 km² options stay on the list for
completeness but are recommended against — see the table above.

### 5. Mesh creation: tool, command, gotchas

**What the mesh actually does** (orientation for anyone new to CTSM
single-point work): the mesh is pure geometry — "I am one cell, at
lat/lon (X, Y), with area A km²." No physics, no data. It is the
**scale anchor** that tells CTSM how big the gridcell is so the model
can translate the surface dataset's percentages and intensive
properties into absolute quantities.

The surface dataset (`fsurdat`) is mostly ratios (`PCT_LAKE`,
`PCT_NATVEG`, `PCT_NAT_PFT`, `PCT_SAND`, …) and intensive properties
(`ORGANIC` in kg/m³, soil color, etc.). These are dimensionless or
per-unit-area / per-unit-volume — they don't care how big the gridcell
is. At runtime, CTSM multiplies them by `grc%area` (from the mesh) to
produce absolute landunit areas, column areas, soil volumes, etc.

The hillslope file is the mix of intensive (`hillslope_distance`,
`hillslope_slope`, `hillslope_elevation`) and absolute (per-bin
`hillslope_area` in m²). The model multiplies `grc%area` by hillslope
weights and divides by the rep's total `hillslope_area` to get
`nhill_per_landunit` — the multiplier for how many copies of the
rep hillslope tile the gridcell.

Consequence: **a single surface dataset works at any gridcell size**.
If our `fsurdat` was extracted from the f09 cell at OSBS lat
(~12,654 km²), the ratios encoded in it are the 12,654 km² Florida
averages. CTSM will happily apply those ratios to a 90 km² or
36.81 km² mesh; whether the smaller subset actually has those same
ratios is a science question (and the reason we override the
hillslope geometry with site-specific data, and may eventually want
NEON megapit soil / AOP PFT overrides — see Section 3).

This is also why **the rescale-only option (c′)** in Section 4 works
in principle: the rep hillslope's bin areas and lake fraction are
themselves ratios-derived statistics. Saying "this rep tiles 218
times to fill 36.81 km²" instead of "533 times to fill 90 km²" is
internally consistent — it just relies on the assumption that the
two extents have similar landscape statistics. At OSBS they do, to
first order.

**Tool mechanics — actual workflow (verified 2026-05-12).** CTSM's
`mesh_maker.py` aborts on 1×1 input (`mesh_maker.py:214`: "No need
to create a mesh file for a single point grid"). This is the
misleading message Johanna Teresa hit in Feb 2025. We instead use
the canonical CDEPS workflow: construct a SCRIP-format NetCDF, run
`ESMF_Scrip2Unstruct` to convert it to the ESMF unstructured mesh
CTSM consumes. This is the same path CTSM's own test suite uses
(`python/ctsm/test/test_sys_mesh_modifier.py:100`) and is
documented in `components/cdeps/doc/source/extending.rst`.

ESMF 8.9.1 binaries are shipped inside the `ctsm` conda env — no
`module load` needed:

```
/blue/gerber/cdevaneprugh/.conda/envs/ctsm/bin/ESMF_Scrip2Unstruct
/blue/gerber/cdevaneprugh/.conda/envs/ctsm/bin/ESMF_Regrid
/blue/gerber/cdevaneprugh/.conda/envs/ctsm/bin/ESMF_RegridWeightGen
```

**SCRIP file schema** (verified, used in production):

| Variable | Dims | dtype | units | Value (90 km² at OSBS) |
|---|---|---|---|---|
| `grid_dims` | `(grid_rank,)` | int32 | — | `[1, 1]` |
| `grid_center_lat` | `(grid_size,)` | float64 | `degrees` | `[29.689282]` |
| `grid_center_lon` | `(grid_size,)` | float64 | `degrees` | `[278.006569]` |
| `grid_corner_lat` | `(grid_size, grid_corners)` | float64 | `degrees` | `[[29.647, 29.647, 29.732, 29.732]]` |
| `grid_corner_lon` | `(grid_size, grid_corners)` | float64 | `degrees` | `[[277.957, 278.056, 278.056, 277.957]]` |
| `grid_imask` | `(grid_size,)` | int32 | — | `[1]` |
| `grid_area` | `(grid_size,)` | float64 | `steradian` | `[2.2173e-6]` |

Corner ordering: CCW from SW (positions `[SW, SE, NE, NW]`),
matching `tools/mkmapgrids/mkscripgrid.ncl:135-150`. Required
dimensions: `grid_size=1`, `grid_corners=4`, `grid_rank=2`.

**Production commands** (executed 2026-05-12, verified):

```bash
# Step 1: Build SCRIP NetCDF (defaults to our production hillslope file)
python scripts/osbs/make_osbs_scrip.py --verbose

# Step 2: Convert to ESMF unstructured mesh
ESMF_Scrip2Unstruct \
    output/mesh/osbs_scrip_90km2_c260512.nc \
    output/mesh/osbs_mesh_90km2_c260512.nc \
    0
```

The trailing `0` = non-dual (cell-based) mesh. Default output
format is ESMF.

**xmlchange recipe** for migrating a case from PTS to mesh mode:

```bash
./xmlchange PTS_LAT=-999.99
./xmlchange PTS_LON=-999.99
./xmlchange LND_DOMAIN_MESH=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/mesh/osbs_mesh_90km2_c260512.nc
./xmlchange ATM_DOMAIN_MESH=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/mesh/osbs_mesh_90km2_c260512.nc
./xmlchange MASK_MESH=/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/mesh/osbs_mesh_90km2_c260512.nc
```

**Output ESMF mesh schema** (canonical, from `ESMF_Scrip2Unstruct`):

```
Dimensions:
  origGridRank = 2
  nodeCount = 4
  coordDim = 2
  elementCount = 1
  maxNodePElement = 4

Variables:
  origGridDims(origGridRank)              int32   = [1, 1]
  nodeCoords(nodeCount, coordDim)         float64 = [[lon, lat], ...]
  elementConn(elementCount, maxNodePElement) float64 = [[1, 2, 3, 4]]
  numElementConn(elementCount)            int32   = [4]
  centerCoords(elementCount, coordDim)    float64 = [[278.006569, 29.689282]]
  elementArea(elementCount)               float64 = [2.2173e-6]      (steradian)
  elementMask(elementCount)               int32   = [1]

Global attrs:
  gridType = "unstructured mesh"
  version = "0.9"
```

Note `centerCoords` is `(lon, lat)` not `(lat, lon)`. Schema is
identical to `fv0.9x1.25_141008_ESMFmesh.nc` plus one benign extra
variable (`origGridDims`, an ESMF 8.9 addition).

**Gotchas remaining:**
- The DiscussCESM thread is still the only public attempt; nobody
  has confirmed a working routing-on case. We are still the first
  documented attempt at the full workflow (mesh + case + routing
  on).
- `mesh_maker.py:214` still aborts on 1×1. If anyone tries the
  obvious `mesh_maker` path, they will hit this. Worth flagging in
  any contribution we make to community docs.

### 6. Mesh-mode community precedent — none

Issue #1432 is **open since 2021-07-20**, never resolved. mvertens'
position ("for a single point area does not make sense and is
arbitrary") was contested by swensosc ("gridcell area is used in the
hillslope model"). jedwards4b was tagged in late 2021 to provide a
reproducer; nothing happened. Only event since: a label change in
March 2025 (samsrabin). No PR linked.

Johanna Teresa's Feb 2025 DiscussCESM thread is the only public record
of a user attempting Swenson's mesh-mode workaround. She never posted
back confirming success. Her resolution is on Automake/ESMF/mpi-serial
build issues, not on the mesh.

Swenson's mesh recommendation in that thread is paraphrased by Teresa,
not quoted directly. No archived recipe, no example commands, no
documented working case. **For our purposes, the mesh-mode pathway is
folklore validated by source-code reading, not validated by a working
community example.**

This affects Phase H planning in two ways:
- We must allocate engineering time for inevitable surprises (the
  area-units mismatch above is already one).
- A6 (new task): Document the working configuration, mesh creation
  command, and any unexpected build/runtime issues. The OSBS work
  becomes the canonical reference for single-point routing-on cases —
  worth recording cleanly even if we don't pursue an upstream PR.

### 7. Stream dimensions and lake-column interface under routing-on

This section is the result of a focused review of the stream-channel
parameters in the hillslope NetCDF and how they interact with the
lake column under `use_hillslope_routing = .true.`. Verified by
reading the SourceMods in `$CASES/osbs.swenson.spinup/SourceMods/
src.clm/` against the upstream CTSM source.

#### 7.1 The stream values in our file, and where they come from

The production hillslope NetCDF (2026-05-05) carries three
gridcell-level stream-channel parameters:

| Variable | Value | Provenance |
|---|---|---|
| `hillslope_stream_depth` | 1.52 m | Swenson power law: `0.001 × A^0.4` |
| `hillslope_stream_width` | 59.23 m | Swenson power law: `0.001 × A^0.6` |
| `hillslope_stream_slope` | 0.0069 m/m | Network mean from LIDAR-derived flow accumulation |

The depth and width formulas live in `scripts/osbs/run_pipeline.py:
1645-1651` with this explicit comment:

```python
# Stream depth/width: Swenson power laws (rh:1104-1114). Kept as-is
# per PI decision (#4, 2026-03-30): osbs2 runs with
# use_hillslope_routing=.false., so these values are never read by
# CTSM under the current spinup configuration.
stream_depth = 0.001 * total_area_m2**0.4
stream_width = 0.001 * total_area_m2**0.6
```

With `A = 90 km² = 9e7 m²`:
- depth = 0.001 × (9e7)^0.4 = **1.52 m** ✓
- width = 0.001 × (9e7)^0.6 = **59.23 m** ✓

The stream slope comes from real flow-network statistics computed
during the pipeline, so it's data-grounded for our LIDAR domain.
The depth and width are formulas applied unmodified.

#### 7.2 Why Swenson's power laws give these values

Swenson & Lawrence (2025) calibrate these power laws against the
**global MERIT 90 m DEM dataset**. In that calibration:

- The dataset's gridcells are 0.9° × 1.25° (~12,654 km² at OSBS
  latitude). At that scale, a 90 km² subgrid hillslope catchment
  is typical for a real tributary in mountain or upland landscapes
  (Appalachians, Cascades, Rocky Mountains, etc.).
- For such a 90 km² tributary basin, a stream channel of order
  ~1 m deep × ~60 m wide IS approximately correct. Think Allegheny
  tributary, Yakima fork, similar.
- The formulas are calibrated against the *kind* of landscape that
  dominates MERIT-resolution gridcells: structured uplands with
  clear channel networks, where stream geometry scales with
  drainage area roughly as predicted by hydraulic geometry theory
  (e.g., Leopold-Maddock 1953).

So the values aren't arbitrary — they're statistically calibrated
against a global ensemble of real river systems at MERIT scale.

#### 7.3 Why those assumptions don't fit OSBS

OSBS is in the **Florida coastal plain**, a landscape Swenson's
power-law calibration doesn't represent well:

| Aspect | Swenson's calibration domain | OSBS reality |
|---|---|---|
| Topographic relief | Hundreds of meters | <20 m across 90 km² (LIDAR-confirmed) |
| Channel concentration | Well-defined networks | Diffuse wetland flow, few discrete channels |
| Real "stream" features | River tributaries (m wide, m deep) | Drainage swales 1–5 m wide, <1 m deep |
| Lake connectivity | Streams connect lakes to networks | Most OSBS lakes are isolated (Long Pond, Ross Lake) — no inflow/outflow stream |
| Dominant flow mechanism | Channelized surface flow + subsurface | Groundwater + diffuse wetland sheet flow |
| Stream density typical | 1–5 km/km² | 0.1–0.5 km/km² in Florida coastal plain |

The 59 m channel width is the most striking mismatch. **OSBS doesn't
have a 59 m wide stream anywhere in our 90 km² LIDAR domain.** The
largest channel in our flow network is the outlet of a wetland
depression, probably <5 m wide. The power law has applied global
scaling to a landscape that doesn't follow the scaling.

The depth (1.52 m) is on the high side but defensible — real OSBS
swales can be ~0.5–1 m deep. The slope (0.0069 = 0.69%) is from
data, so it's the actual mean slope of the steepest accumulation
paths in our LIDAR (could be biased high by sharp drops into
wetland depressions).

#### 7.4 What CTSM does with these under routing-on

The stream parameters enter `HillslopeHydrologyMod.F90` in two main
places:

**A. Manning equation flow velocity** (line ~975 in our SourceMod
fork):

```fortran
flow_velocity = (hydraulic_radius)**manning_exponent &
              * sqrt(stream_channel_slope) / manning_roughness
volumetric_streamflow = cross_sectional_area * flow_velocity
```

Where `cross_sectional_area = stream_water_depth × stream_width` and
`hydraulic_radius = area / wetted_perimeter`. For our params:

- Bankfull area = 1.52 × 59.23 = **90.0 m²** per unit length
- Bankfull hydraulic radius ≈ 1.34 m (after dividing by wetted
  perimeter 1.52×2 + 59.23 ≈ 62.3 m)
- Bankfull velocity ≈ 1.34^0.667 × √0.0069 / 0.03 ≈ **3.36 m/s**
- Bankfull discharge per unit length × stream length ≈ 90 × 3.36 ≈
  **302 m³/s**

That's a large stream discharge potential. **The stream has so much
capacity it will almost never be near bankfull.** Under typical
flow conditions (inches of water depth, not meters), velocity is
much lower and flow doesn't fill the channel.

**B. Stream channel length** (line ~501 in HillslopeHydrologyMod):

```fortran
stream_channel_length += hill_width(c) × 0.5 × nhill_per_landunit
```

Summed across columns, our setup gives:
- Σ hill_width × 0.5 × 533.7 ≈ **3,046 km of stream channel** in
  90 km² = 33.8 km/km² stream density

That's enormous (~10× typical real river-network density). The
implied stream cross-section × length = 90 m² × 3,046 km =
**2.74 × 10⁸ m³ of bankfull storage**. The stream can absorb 274
million cubic meters of upland drainage before it overflows.

#### 7.5 What the SourceMods do that's still active

> **Revised 2026-05-19 after the routing-gate audit.** Section 8 below
> documents the full review of all 5 SourceMod files. Two changes vs.
> the earlier four-mechanism table: (B) is **inactive** under
> routing-off (the surrounding subroutine is routing-gated), and a
> fifth mechanism (zeroing of saturated-excess surface runoff in
> `SaturatedExcessRunoffMod.F90:295`) was missed in earlier analysis
> and is active.

The SourceMods in `$CASES/osbs.swenson.spinup/SourceMods/src.clm/`
contain five distinct modifications. The 2026-04-30 PI decision to
set `spillheight = 0.0` disabled exactly ONE of them; routing-off
gates out a second; the other three are active.

| Mechanism | File:line (SourceMod copy) | Gating | Status with our config (`use_hillslope=.true.`, `use_hillslope_routing=.false.`, `spillheight=0`) |
|---|---|---|---|
| A. Pond-build for `hill_elev<0`; instantaneous overflow when water surface > 0 | `SurfaceWaterMod.F90:516-520` | `col%is_hillslope_column(c) .and. col%active(c)` only | **Active** — lake column release valve at the 6 m threshold |
| B. Terminal-column-only contribution to streamwater | `HillslopeHydrologyMod.F90:1120-1126` | Inside `HillslopeUpdateStreamWater`, which is only called when `use_hillslope_routing=.true.` (`HydrologyDrainageMod.F90:150-158`) | **Inactive** under routing-off — earlier "Active" claim was wrong |
| C. Global `hill_elev` shift by `−spillheight` | `HillslopeHydrologyMod.F90:363` | Runs every init; effect proportional to spillheight | **Numerical no-op** — `spillheight=0` makes it identity, but the line still executes |
| D. Inter-column surface chain bookkeeping (subtract `downstream_vol` from each column's `qflx_h2osfc_surf`) | `SurfaceWaterMod.F90:547-561` | `col%is_hillslope_column(c) .and. col%active(c)` only | **Active** — surface-water routing happens for any hillslope chain |
| E. Zero saturated-excess surface runoff for all columns | `SaturatedExcessRunoffMod.F90:294-295` | None — applies in the `do fc = 1, num_hydrologyc` loop | **Active globally** (not just hillslope columns) — "rolls down the hill until it finds a pond" framing |

The release-valve we get for the lake column is Mechanism A:

```fortran
if (col%hill_elev(c) + h2osfc(c)*1e-3 < 0)  then
   qflx_h2osfc_surf(c) = 0._r8       ! build pond
else if (col%hill_elev(c) < 0 .and. col%hill_elev(c) + h2osfc(c)*1e-3 > 0) then
   qflx_h2osfc_surf(c) = ...         ! release as overflow
endif
```

For a column with `hill_elev = -6 m`: water accumulates as surface
pond until `h2osfc > 6000 mm` (= 6 m of water above the lake floor),
at which point excess starts running off. **The 6 m threshold is the
lake's hill_elev value, not the spillheight.** Spillheight=0 didn't
remove this mechanism.

Mechanism B will become load-bearing **only under routing-on**: it
gates `qflx_surf_vol` so that only the terminal column (our lake,
since it's at chain index 1 with `cold == spval`) contributes surface
runoff to the stream's water-volume ledger. Upland columns can't dump
directly into stream state — they have to flow down the chain via
Mechanism D. Under our current routing-off configuration the
surrounding subroutine (`HillslopeUpdateStreamWater`) is not called,
so this code path stays dormant.

#### 7.6 The lake-column interface under routing-on: full picture

Combining everything:

**Geometry:**
- Stream channel reference at hill_elev = 0
- Stream water surface ranges 0 to +1.52 m (bankfull)
- Lake column floor at hill_elev = −6 m
- Lake water surface ranges −6 m (dry) to 0 (overflow threshold)

**Inflows to the lake column (routing-on):**
- Direct precipitation (~1300 mm/yr at OSBS)
- Lateral subsurface flow from bin 2 (lowest land bin, ~0.25 m elev),
  driven by Darcy gradient Δh ≈ 6.25 m / hill_distance → strong
  downhill driver
- Lateral surface flow from bin 2 via chain-routed `qflx_h2osfc_surf`
  (Mechanism D)

**Outflows from the lake column:**
- Open-water evaporation (CTSM models this; ~1500 mm/yr potential ET
  in north Florida — the dominant real-world drainage mechanism for
  isolated lakes)
- Subsurface drainage `qflx_drain` to stream: **near zero**, because
  Darcy gradient is unfavorable (lake water table at −6 m, stream
  water surface ≥ 0)
- Surface overflow via Mechanism A: **zero until h2osfc reaches 6 m
  above lake floor**, then excess QOVER

**Expected dynamics:**
1. Wet season: precipitation + lateral inflow exceed ET → lake
   H2OSFC rises
2. Dry season: ET exceeds inflow → lake H2OSFC falls
3. Multi-year wet anomaly: lake fills toward 6 m threshold; if it
   reaches threshold, overflow begins (rare)
4. Multi-year dry anomaly: lake can fall back toward 0 (dry lake bed)

This is broadly correct for real Florida lake behavior, except for
two scale issues:

**Issue 1 — Range is too wide.** Real OSBS lakes (Long Pond, Ross
Lake) fluctuate by 1–2 m seasonally, not 6 m. With our `hill_elev
= −6 m`, the model's "capacity" before overflow is several times
larger than real-world capacity. The lake will look like it never
overflows even in wet years.

**Issue 2 — Stream is too generous a sink.** Even when the lake
does overflow into the stream, the 59 m × 1.52 m stream channel
absorbs the discharge with negligible water-depth response. Stream
acts more like a "magic drain" than a real channel that would
back up and create floodplain dynamics.

The two issues compound: a too-deep lake threshold combined with
a too-generous stream sink means the model produces less inundation
and less stream-mediated TAI signal than reality.

#### 7.7 Tuning options

In order of cost / impact:

**Option 1: Stream cross-section reduction (pipeline, ~1 hour).**
Override the power-law-derived depth and width in the pipeline. For
OSBS coastal-plain reality:

```python
# In run_pipeline.py, replace the Swenson power laws with:
stream_depth = 0.5    # m  (OSBS swale-scale, see note on sourcing)
stream_width = 5.0    # m
# stream_slope = unchanged (data-derived)
```

Effect: stream fills more readily, stream surface rises, less
asymmetry between lake water surface (capped at 0) and stream
water surface. Stream might back up during wet events, creating
some surface-flow dynamics. Won't fix the 6 m lake capacity
directly.

**Important: those 0.5 m / 5 m numbers above are placeholders, not
measurements.** They're back-of-envelope estimates from general
Florida coastal-plain knowledge, not from our pipeline or any
OSBS-specific dataset. Before locking these for production, get a
defensible width from real data via one of:

1. **USGS NHD (National Hydrography Dataset) — recommended.**
   Pull the high-resolution NHD shapefile for Putnam County, FL,
   intersect with our 90 km² LIDAR domain (`data/mosaics/production/
   dtm.tif` bounds), and read channel widths from the
   `NHDFlowline` and `NHDArea` attributes. URL:
   https://www.usgs.gov/national-hydrography/national-hydrography-dataset
   Cost: ~2 hours. Produces published, citable values.
2. **Extract from our own pipeline (~half day).** The pysheds
   flow-accumulation raster and stream-network mask are already
   computed for the 2026-05-05 production run. Re-derive
   deterministically from the LIDAR mosaic and measure
   channel widths directly from contiguous channel-pixel
   counts. Gives an LIDAR-derived width that's self-consistent
   with our `stream_slope` provenance.
3. **Manual measurement via Google Earth + LIDAR DTM (~1 hour).**
   Identify visible drainage features in our DTM, measure widths.
   Less rigorous than NHD or pipeline extraction.

NHD is the strongest path: published, peer-recognized, and
specifically intended for hydrology applications. Pipeline
extraction (option 2) is the cleanest "from our own data"
alternative if NHD coverage is sparse in coastal-plain wetlands.

**Option 2: Lake floor elevation reduction (pipeline, ~1 hour).**
Change `lake_hill_elev` from −6 m to −1 or −2 m. The Phase E.5
documentation calls hill_elev=−6 a "chain bookkeeping value" —
needs to be lower than Bin 1 (~0.25 m) but exact value doesn't
matter for chain ordering. Reducing to −2 m would:

- Keep the chain bookkeeping intact (still below all land bins)
- Reduce overflow threshold from 6 m to 2 m of water accumulation
- Match real OSBS lake seasonal range
- Trigger more frequent overflow events into the stream

This is the higher-leverage change.

**Option 3: Both.** Tune both stream and lake. Compounding effects:
narrower stream + lower overflow threshold = more frequent lake-to-
stream surface flow events = realistic TAI dynamics emerge.

**Option 4: Revive `spillheight` (namelist, ~1 second).** Set
`spillheight > 0` to enable Mechanism C (global elevation shift).
Pushes all column elevations down. This is the "whole hillslope is
a wetland depression" framing. **Doesn't fit our lake-as-one-column
model** — we represent the wetland as a single column at the bottom
of the chain, not as a whole-hillslope feature. Recommend against.

**Option 5: Regional Darcy drain on the lake column (SourceMod).
PI's idea, raised 2026-05-19. Very vague; contingent on Phase F.**
The concern: in single-point mode there is no downstream cell to
receive water leaving the gridcell, so the lake column at the
bottom of the chain may accumulate lateral inflow faster than
`QDRAI` + ET removes it. PI floated the idea of adding an
engineered Darcy sink on the lake column referenced to a distant
sea-level boundary (drain "to the ocean").

Status: idea only. No design, no parameter values, no
implementation. Whether to pursue it depends entirely on what
Phase F's column-level output shows. If `H2OSFC` and `ZWT` at the
lake column reach stable equilibrium via existing budget terms,
this mechanism isn't needed. If they trend toward unbounded
accumulation, this is one of two responses (the other being
Option 2: lower `lake_hill_elev` to enable overflow via Mechanism
A). The advantage over Option 2 is that it works under both
routing-off and routing-on since it doesn't depend on the
stream-side machinery.

If pursued, the design will need to address: what reference head
(sea level vs. nearest local stream like St. Johns River, ~30 km
east), what gradient length, what K value, where in
`SoilHydrologyMod` to subtract the flux, how to expose parameters
via namelist, and how to gate the drain on `zwt_lake > h_ref` to
avoid pulling water from an already-dry column. None of this is
worked out yet.

#### 7.8 What we'll observe in the sniff test

With current parameters (`hill_elev_lake = −6 m`, stream as-is):

| Variable | Expected behavior with routing on |
|---|---|
| `H2OSFC[lake_column]` | Rises during wet season, falls during dry; equilibrium level depends on actual upland inflow vs ET balance |
| `H2OSFC[bin 2]` (FZ) | Rises seasonally but drained downhill; should be wettest land column |
| `H2OSFC[bin 25]` (ridge) | Modest, dries quickly |
| `QOVER[lake_column]` | Zero almost always (only triggers above 6 m) |
| `QDRAI[lake_column]` | Near zero (unfavorable Darcy gradient) |
| `QDRAI[bin 2]` | Positive, drains laterally to lake |
| `ZWT[lake_column]` | At surface (= 0) — lake is permanently saturated |
| `ZWT[bin 2]` | Shallow, rises after rain events |
| `ZWT[bin 25]` | Deep, slow response |
| `STREAM_WATER_VOLUME` | Small but non-zero (mostly fed by tiny lake-to-stream Darcy plus rare overflows) |
| `STREAM_WATER_DEPTH` | Centimeters at most (channel oversized) |
| `VOLUMETRIC_STREAMFLOW` | Small (low water depth → low Manning velocity) |

**Sniff-test diagnostics that flag tuning issues:**
- Lake `H2OSFC` trends toward 6 m without seasonal reversal →
  upland inflow >> ET, lake capacity too generous, recommend
  Option 2 tuning
- Lake `H2OSFC` saturates at 6 m → overflow active, system finding
  its own balance; check `QOVER[lake]` is non-zero
- Stream `WATER_DEPTH` < 1 cm year-round → stream oversized,
  recommend Option 1 tuning
- Bin 2 ZWT stays at surface (saturated) while higher bins drain →
  TAI structure emerging, model is working

#### 7.9 Why this matters for B2 (PI consultation)

B2 asks "is the Darcy gradient (Δh/L) physically meaningful?" This
section refines what's needed:

- The Darcy gradient from upland to lake is driven by `hill_elev`
  differences (Bin 2 at +0.25 m, Lake at −6 m → Δh = 6.25 m). With
  bin distances of 50–500 m, gradient is 0.01 to 0.1 m/m.
- For OSBS sandy soils with K_sat ≈ 10⁻⁵ to 10⁻⁴ m/s, Darcy flux
  q ≈ K × Δh / L is on the order of 10⁻⁶ to 10⁻⁵ m/s, or
  **86 to 864 mm/day per column-width**.
- That's a substantial lateral flow rate. The lake column will
  receive water FAST under routing on.

The B2 question now has a concrete framing for the PI: "is OSBS
upland → lake lateral flow really at the meters-per-day scale, or
should bin elevation/distance be retuned to give realistic
gradients?" Whatever the answer, it interacts with the stream and
lake column choices above.

#### 7.10 Summary for PI conversation

The whole stream-channel + lake-column setup was locked under
routing-off assumptions where these values didn't matter. With
routing on:

1. Stream params (depth 1.52 m, width 59.23 m) come from Swenson's
   global MERIT-calibrated power laws. They're appropriate for the
   landscapes that calibrated them; they're 5–10× too generous for
   OSBS Florida coastal plain.

2. The lake column at `hill_elev = −6 m` gets a 6 m overflow
   threshold via Mechanism A in our SourceMod. That's the actual
   release valve, and it's still active (spillheight=0 didn't
   disable it). 6 m is several times larger than real OSBS lake
   seasonal range.

3. Two tuning levers cost ~1 hour each in pipeline rework:
   - Stream cross-section (override power laws)
   - Lake floor elevation (change `lake_hill_elev` from −6 to −2)

4. Recommend running the sniff test first to see which tuning
   knobs the model actually needs (which is which of the issues
   above actually surfaces in the routing-on dynamics).

### 8. Routing-gate audit (2026-05-19) — what `use_hillslope_routing` actually controls

Comprehensive review of every `use_hillslope_routing` reference in
CTSM 5.3.085 source (`git rev dc3aa5ddc`), cross-checked against the
SourceMods in `$CASES/osbs.swenson.spinup/SourceMods/src.clm/` and
empirical h1a output from both `osbs.routing.test`/`control` (5-yr
A4 smoke test) and the operative `osbs.swenson.spinup` (currently
running). Conclusion: the project's working framing — that
`use_hillslope_routing` gates inter-column lateral flow — is wrong.
The corrected understanding feeds the Problem section table above
and the change-log entry in STATUS.md.

#### 8.1 The dispatch site

`HydrologyDrainageMod.F90:127-160`:

```fortran
if (use_aquifer_layer()) then
   call Drainage(...)                          ! aquifer-layer mode (not us)
else
   call PerchedLateralFlow(...)                ! ALWAYS, no routing gate
   call SubsurfaceLateralFlow(...)             ! ALWAYS, no routing gate
   if (use_hillslope_routing) then
      call HillslopeStreamOutflow(...)         ! routing-gated
      call HillslopeUpdateStreamWater(...)     ! routing-gated
   endif
endif
```

Our case takes the `else` branch (`lower_boundary_condition=2` →
`bc_zero_flux` → `use_aquifer_layer()=.false.`; see
`bld/namelist_files/namelist_defaults_ctsm.xml:448` and
`SoilWaterMovementMod.F90:229`). So `PerchedLateralFlow` and
`SubsurfaceLateralFlow` are called every hydrology step regardless
of routing.

#### 8.2 What the lateral-flow routines actually do

`SubsurfaceLateralFlow` (`SoilHydrologyMod.F90:2086+`) is the
canonical inter-column water-table flow routine. Loop at
`:2249-2402` processes every column; hillslope-column branch at
`:2254` is gated only by `if (col%is_hillslope_column(c) .and.
col%active(c))`.

Inside that branch, two paths:

- **Internal column (has a downhill neighbor, `col%cold(c) /= ispval`):**
  Darcy head gradient computed at `:2261-2263` between this column's
  surface elevation minus water-table depth and the downhill
  neighbor's. `qflx_latflow_out_vol = transmis × hill_width ×
  head_gradient` at `:2358`. The downhill column accumulates the
  inflow at `:2386-2388`:

  ```fortran
  qflx_latflow_in(col%cold(c)) = qflx_latflow_in(col%cold(c)) + &
       1.e3_r8*qflx_latflow_out_vol(c)/col%hill_area(col%cold(c))
  ```

  No routing gate anywhere in this path.

- **Terminal column (`col%cold(c) == ispval`):** Same Darcy machinery,
  but the "downhill neighbor" is the stream channel. Here routing
  matters — `:2265-2272` selects the stream-depth source:

  ```fortran
  if (use_hillslope_routing) then
     stream_water_depth = stream_water_volume(l) / &
                          lun%stream_channel_length(l) / &
                          lun%stream_channel_width(l)
     stream_channel_depth = lun%stream_channel_depth(l)
  else
     stream_water_depth = tdepth(g)               ! from MOSART via coupler
     stream_channel_depth = tdepth_bankfull(g)    ! from MOSART via coupler
  endif
  ```

  The terminal column still gets a Darcy gradient computed against
  some stream depth in both branches. The only difference is where
  that depth comes from.

`PerchedLateralFlow` (`:1703+`) is structurally identical for the
perched water table; the same Darcy gradient between adjacent
columns runs at `:1817-1820`, the same routing-gated stream-depth
swap at `:1822-1829`.

#### 8.3 Where the flow is applied to soil water

`SubsurfaceLateralFlow` lines `:2433-2509`:

```fortran
qflx_net_latflow(c) = qflx_latflow_out(c) - qflx_latflow_in(c)
...
drainage(c) = qflx_net_latflow(c)                 ! for hillslope columns
...
drainage_tot = - drainage(c) * dtime
if (drainage_tot > 0.) then                        ! rising water table
   ! water added to h2osoi_liq, zwt rises
else                                               ! deepening water table
   ! water removed from h2osoi_liq, zwt falls
endif
```

So if a column receives more lateral inflow than it sends out
(`qflx_latflow_in > qflx_latflow_out`), `drainage` is negative,
`drainage_tot` is positive, and water is **added** to its soil
column. The transfer is real and applied to `h2osoi_liq`.

#### 8.4 Empirical confirmation

Two independent observations confirm inter-column flow runs under
routing-off:

1. **Routing.test vs routing.control h1a (5-yr smoke test).** QDRAI
   has negative values at hillslope columns in both cases. Min/max:
   test = −4.32×10⁻⁵ / 1.47×10⁻⁵, control = −4.15×10⁻⁵ /
   1.43×10⁻⁵. Both cases show the lateral-inflow signature; the
   values differ only at ~10⁻⁶ scale (the boundary-condition delta).

2. **`osbs.swenson.spinup` h1a (600 yr, routing-off).** QRUNOFF
   column-level min/max = −1.156×10⁻⁴ / 1.992×10⁻⁴ mm/s. ZWT spans
   0.011 → 8.6 m across columns. If columns were "hydrologically
   isolated 1D soil columns" as the earlier docs claimed, we'd see
   no negative QRUNOFF and far less ZWT variation. The negative
   values are the unambiguous fingerprint of inter-column lateral
   redistribution.

#### 8.5 Enumerated routing-gated sites

For completeness, every `use_hillslope_routing` reference in
CTSM 5.3.085 source (excluding declaration/bcast/log plumbing):

| File:line | What it gates |
|---|---|
| `HillslopeHydrologyMod.F90:378-411` | Read `hillslope_stream_depth/width/slope` from NetCDF |
| `HillslopeHydrologyMod.F90:475-507` | Compute `nhill_per_landunit`, `stream_channel_length`, `stream_channel_number` |
| `HillslopeHydrologyMod.F90:1078+` (`HillslopeUpdateStreamWater`) — called only from `HydrologyDrainageMod.F90:150-158` | Advance `stream_water_volume`; includes the SourceMod terminal-column gate at `:1118-1126` |
| `HillslopeHydrologyMod.F90:977+` (`HillslopeStreamOutflow`) — called only from `HydrologyDrainageMod.F90:151-153` | Manning's-equation streamflow velocity |
| `SoilHydrologyMod.F90:1822-1829` | Perched-LF stream-depth source switch |
| `SoilHydrologyMod.F90:2265-2272` | Subsurface-LF stream-depth source switch |
| `SoilHydrologyMod.F90:2362-2367` | Losing-stream outflow cap |
| `WaterFluxType.F90:525-534` | Register `VOLUMETRIC_STREAMFLOW` history field |
| `WaterFluxType.F90:922-928` | Zero `volumetric_streamflow_lun` each step |
| `BalanceCheckMod.F90:274-280, 744-750` | Add `stream_water_volume` / `qflx_streamflow_grc` into gridcell water-balance ledger |
| `lnd2atmMod.F90:343+` | Sum streamflow over landunits for lnd→rof coupling |
| `lnd_import_export.F90:916-919` | Add stream component to subsurface-runoff field exported to coupler |

Every site is on the stream-channel side. None of them are inside
the column-to-column flow path of `PerchedLateralFlow` or
`SubsurfaceLateralFlow` proper.

#### 8.6 Implications for the project narrative

The corrected understanding does **not** invalidate any decision
made under the earlier framing — it reframes what those decisions
mean physically:

- **Phase F is delivering more TAI physics than its doc claimed.**
  The 600-yr spinup is running inter-column lateral flow. The
  column-level water-table and soil-moisture differentiation
  observed in the h0a/h1a plots is a combination of (a) independent
  per-column forcing and (b) actual lateral redistribution. The
  TAI mechanism is not "dormant" in Phase F.
- **Phase G Stage 1 (lake column construction) is correct.** The
  doc actually got this right in its detailed sections (line 92,
  125-138 of `phases/G-ctsm-lake-representation.md`): "water drains
  from hillslope to lake naturally through the existing lateral
  flow pathway" — that lateral flow is exactly the unconditional
  inter-column flow this audit identifies. Only the Stage-1 framing
  at lines 63-65 ("columns are hydrologically isolated") contradicts
  the doc's own physics description.
- **Phase H Track A is correct; Track B/C deliverable narrows.**
  The mesh-mode work still solves the spval bug. The routing-on
  test still validates that the stream-side machinery functions.
  But "lateral subsurface flow activation" is not what Phase H
  delivers — that activation happened automatically when
  `use_hillslope=.true.` was first turned on. Phase H delivers
  the stream-coupling BC and the internal stream-water ledger.
- **PI consultation B1-B4 still relevant**, with refined framing:
  - B2 (Darcy gradient reasonableness) — applies to inter-column
    gradients that are *already running*. Reviewing actual
    `osbs.swenson.spinup` `QDRAI`/ZWT trajectories may be more
    informative than purely theoretical reasoning.
  - B4 (stream geometry, lake overflow threshold) — applies only
    once routing is on; until then the relevant BC is `tdepth_grc`
    (likely 0 with our DATM+MOSART setup).
- **The 2026-05-12 A4 smoke test result has a cleaner physical
  interpretation now**: bit-identical gridcell aggregates and
  near-identical column states isolate the BC swap, not the
  presence/absence of lateral flow.

#### 8.7 What needs verification before the production routing-on run

- **Whether `tdepth_grc` is actually populated by MOSART in our
  case.** If it stays at 0 (default fallback at
  `lnd_import_export.F90:623`), then the routing-off terminal
  column effectively sees an empty stream — Darcy gradient
  dominated by `col%hill_elev - zwt` minus 0. If MOSART sends
  meaningful values, the BC difference between routing-on and
  routing-off is more subtle than the A4 test suggests.
- **Whether the column-level QDRAI/QRUNOFF time series in the
  600-yr spinup actually show TAI-like behavior** (saturated
  low-HAND columns near the lake, drier ridges, seasonal cycles
  of low-HAND saturation). The audit confirms the mechanism is
  present; whether it manifests at OSBS scales requires looking
  at the data.

## Contingency note (2026-05-19)

The task lists below (A5–A6, B1–B4, C1–C4) were drafted under the
pre-audit framing that routing-on was needed to activate
inter-column lateral flow. After the 2026-05-19 routing-gate
audit, that motivation is gone — lateral flow is already running
in Phase F. **There is a good chance we do not pursue routing-on
at all**, in which case none of the remaining tasks are necessary.
The decision is contingent on Phase F:

- If Phase F's column-level output shows acceptable TAI emergence
  and a stable lake column, routing-on is optional polish and the
  remaining tasks below can be retired.
- If Phase F shows the lake column accumulating water without
  bound, the response is either Option 5 below (regional Darcy
  drain SourceMod, works under both routing settings) or pursuing
  routing-on with lake-overflow tuning. The Darcy drain idea is
  currently very vague; details would only be worked out if it's
  actually needed.

Treat the task lists as a frozen record of what would be done
under the old framing, not as an active to-do list.

## Software problems — engineering only

Pure engineering / configuration tasks. Can be planned, prototyped,
and reported on without PI input.

- [x] **A1.** Identify the input format and command-line interface of
  `tools/site_and_regional/mesh_maker`. Document expected inputs (lat,
  lon, area variable names) and the produced ESMF mesh format. **Done
  2026-05-12** — see "Research notes" section above. Tool requires
  `--lat` and `--lon` (required), `--area` (optional, must be radians²
  if supplied), `--mask` (optional). Our hillslope NetCDF carries
  `LATIXY`/`LONGXY`/`AREA(km²)` — directly usable except for the
  units mismatch on `--area`. Concrete command and gotchas documented
  in research notes.

- [x] **A2.** Generate the production-target 90 km² mesh. **Done
  2026-05-12.** The CTSM `mesh_maker.py` aborts on 1×1 input, so
  switched to the canonical CDEPS workflow (SCRIP → ESMF mesh via
  `ESMF_Scrip2Unstruct`, available in the `ctsm` conda env as ESMF
  8.9.1). Wrote `scripts/osbs/make_osbs_scrip.py` (~200 lines) that
  defaults its center/area from the production hillslope NetCDF,
  constructs a single-cell SCRIP file, and self-verifies on write.
  Conversion via `ESMF_Scrip2Unstruct ... 0` produces a canonical
  unstructured mesh: nodeCount=4, elementCount=1, maxNodePElement=4,
  coordDim=2, with `elementArea = 2.2173e-6 sr (= 90.000 km²)`,
  `elementMask = 1`, `centerCoords = (278.006569, 29.689282)`.
  Schema compared against `fv0.9x1.25_141008_ESMFmesh.nc` —
  identical layout, identical dtypes except `numElementConn` is
  int32 (ours) vs int8 (reference). Plus one extra var
  `origGridDims` added by ESMF 8.9. Both are benign. Artifacts at
  `output/mesh/osbs_scrip_90km2_c260512.nc` and
  `output/mesh/osbs_mesh_90km2_c260512.nc`.

- [x] **A3.** Construct a paired test case + control case as siblings of
  `osbs.swenson.spinup`. **Done 2026-05-12.** Two cases:
  `$CASES/osbs.routing.test` (routing on) and
  `$CASES/osbs.routing.control` (routing off). Both: identical
  COMPSET (1850_DATM%CRUv7_CLM60%BGC), identical mesh
  (`osbs_mesh_90km2_c260512.nc`), identical SourceMods (5 .F90 files
  copied from `osbs.swenson.spinup`), identical fsurdat and
  `hillslope_file`, `CLM_ACCELERATED_SPINUP=off`, cold start. 5-year
  run, 6-hourly output, ~monthly files. **Only difference:**
  `use_hillslope_routing = .true.` vs `.false.`. Build ~5.5 min each.
  Run wallclock: test 42 min, control 52 min (both well under 12-hr
  budget).

- [x] **A4.** Inspect output. **Done 2026-05-12.** Headline results
  (see Log entry below for full analysis):
  - `grc%area` = **90.006 km²** in both cases — **NOT spval**.
    Phase H mesh-mode workaround works.
  - `VOLUMETRIC_STREAMFLOW` registered and populated only in test
    (mean ~8 m³/s); correctly absent from control.
  - Water budget closes (no balance-check errors).
  - 5-yr gridcell-aggregate variables (RAIN, ET, TWS, GPP, soil C
    pools) bit-identical between test and control to 4–7 sig figs.
  - `H2OSFC = 0` in every column at every timestep, both cases:
    cold-start + Florida ET dries the soils faster than rainfall
    can saturate. No standing surface water emerges in 5 years.
  - **Year-5 deep soil moisture (H2OSOI, layer 15 ≈ 3 m) shows
    the textbook TAI signature** at column level:
    `H2OSOI[lake, TEST] − H2OSOI[lake, CTRL] = +7.14e-4`
    (lake wetter under routing)
    `H2OSOI[bridge col 13, TEST] − [bridge, CTRL] = −9.61e-5`
    (drier — water drained downhill toward lake)
    high upland (cols 20–24): no signal (too far for 5 yr).
  - Stream-channel parameters (depth/width from Swenson power laws,
    Phase H Section 7) didn't manifest as problems in this short
    run. The "too generous" stream concern is real but invisible
    here because nothing accumulates enough to overflow.

- [ ] **A5.** Document the cookbook of xmlchange commands needed to
  migrate a PTS_MODE case to mesh-mode. Production-ready recipe
  exists in `phases/H-lateral-flow.md` Section 5 and was validated
  by the A3/A4 work above. Optional refinement: extract into a
  standalone shell script. Decision deferred until ready for the
  production spinup.

- [ ] **A6.** Record the working configuration as the canonical
  reference for single-point routing-on (since no community precedent
  exists — see research notes Section 6). **A3/A4 are now the
  canonical working example.** The user_nl_clm + xmlchange recipe
  for the two paired cases is documented in this phase doc Section 5,
  in the 2026-05-12 log entry below, and in the case directories
  themselves (`$CASES/osbs.routing.test`, `$CASES/osbs.routing.control`).
  Whether to extract into a tutorial-style README in `scripts/` or
  `docs/` is deferred until after the production spinup.

## Scientific decisions — PI consultation required

These determine what the model represents physically. Software work
above is unblocked by these but **a long routing-on spinup is not
meaningful without resolving them.**

- [ ] **B1. Gridcell area choice.** Updated with research findings
  (2026-05-12, revised 2026-05-13). The research notes Section 4
  above contains the full scale analysis and pipeline-consistency
  check.

  | Option | Area | Pipeline rework | Physical referent |
  |---|---|---|---|
  | (a) **LIDAR production domain** | **~90 km²** | **None** | The actual LIDAR-covered study area |
  | (c′) **NEON site, rescale-only** | ~36.81 km² | **~1 hour** (regenerate hillslope NetCDF with new `nhill_implicit ≈ 218` and `AREA = 36.81`) | NEON-site scale, with hillslope statistics inherited from the 90 km² LIDAR rectangle |
  | (c) **NEON site, full re-derive** | ~36.81 km² | ~1–2 days (polygon-clip LIDAR + pipeline rerun) | NEON-site scale, with hillslope statistics derived from exactly the NEON polygon |
  | (b) Single representative hillslope | ~0.17 km² | Sets `nhill_implicit = 1`; loses tiling framing | A single TAI unit |
  | (d) Swenson 0.5°×0.5° convention | ~2,700 km² | 30× rescale without geometry change — physically dubious | OSBS represents a regional cell |
  | (e) Swenson 0.9°×1.25° paper | ~12,654 km² | 140× rescale; same problem | OSBS represents a continental-scale cell |

  **My recommendation: present (a), (c′), and (c) to the PI.** All
  three are defensible:

  - **(a) — zero engineering cost.** Use what we already have. Physical
    referent: the LIDAR-covered rectangle. Publication framing:
    "OSBS at 90 km² LIDAR domain scale."
  - **(c′) — one hour of engineering cost.** Reuse existing rep
    hillslope statistics but rescale to the NEON-site area. Defensible
    because the broader 90 km² rectangle and the smaller NEON polygon
    are both within the same uniform Florida sandhill-and-depression
    landscape. Publication framing: "OSBS at NEON site scale, using
    hillslope statistics from a containing LIDAR mosaic."
  - **(c) — full statistical fidelity to the NEON polygon.** Costs the
    polygon-clip engineering (~1–2 days; the pipeline currently
    assumes rectangular tile grids). Publication framing: "OSBS at
    NEON site scale, derived from LIDAR within the NEON polygon."

  Options (b), (d), (e) recommended against. (b) loses the tiling
  framing which is the whole point of representative hillslopes;
  (d) and (e) require rescaling without statistical change to
  hillslope geometry, implicitly claiming our representative tile is
  faithful to a much larger region we don't have data for.

  **Strongly recommend against (d) and (e).** Both require rescaling
  `nhill_implicit` without changing the underlying hillslope geometry,
  which means each rep hillslope physically represents a hillslope
  unit that does not exist at scale in the surrounding 2,700–12,654
  km² Florida landscape. Defeats the point of building a 1m-LIDAR
  dataset.

  Sub-questions for the PI:
  - Is (a) acceptable, or should we re-run the pipeline for (c)?
  - For publication: how do we frame "gridcell = 90 km² LIDAR domain"
    given the Swenson paper uses 12,654 km² gridcells globally?
  - Does the choice need to match published Swenson methodology for
    intercomparison, or are we doing site-specific science where we
    can choose differently?

- [ ] **B2. Hydraulic conductivity sanity check.** Implicit lateral
  flow rates between bins depend on CTSM soil-property fields
  (`HKSAT`, `WATSAT`) and our column geometry (`hill_distance`,
  `hill_elev` differences). At our 24-bin TAI-focused scheme,
  neighboring-bin elevation differences in the 0–50 cm zone are small
  (~10 cm). Two science questions:
  - Is the resulting Darcy gradient (Δh/L) physically reasonable for
    OSBS sandy soils?
  - If the gradient is too small, do we revisit bin spacing for
    routing-on (Phase E.5 was designed under routing-off framing)?

  The PI should look at the soil-property values and judge whether
  the lateral flow rates emerging from the model match expectations
  for the site.

- [ ] **B3. Validation framing.** No published single-point routing-on
  case exists. How do we judge whether observed behavior is "correct"?
  Options to discuss:
  - Compare to a gridded routing-on run at coarse resolution (would
    require running a second case)
  - Qualitative TAI signal check (water table rises in low-HAND bins
    during wet periods, drains during dry periods)
  - Comparison to OSBS field observations (flux tower, well data,
    NEON measurements)
  - Comparison to Lee 2023 OSBS-specific observations

  Without a clear validation framing agreed up front, any output is
  hard to interpret or publish.

- [ ] **B4. Stream-channel geometry and lake column overflow
  threshold.** New (2026-05-12). Stream depth (1.52 m) and width
  (59.23 m) come from Swenson's MERIT-global power laws and are
  ~5–10× too generous for OSBS Florida coastal plain. The lake
  column overflow threshold is 6 m of water above lake floor (from
  `lake_hill_elev = −6 m`), several times wider than real OSBS
  lake seasonal range (~1–2 m). Section 7 of research notes above
  has the full analysis with numerical examples.

  Sub-questions for the PI:
  - Do we tune stream cross-section to match OSBS observed swale
    geometry before the production spinup? Pipeline rework ~1 hour
    once we have a target value.
  - **Where does the target stream width/depth come from?**
    Suggested sourcing in priority order (full discussion in
    research notes Section 7.7):
    1. **USGS NHD** (National Hydrography Dataset) high-resolution
       shapefile for Putnam County, FL — published, citable,
       ~2 hours of work.
    2. **Extract from our own pipeline** (~half day) — re-derive
       channel widths from the pysheds flow-network mask and
       LIDAR mosaic. Self-consistent with our data-derived
       `stream_slope`.
    3. Manual Google Earth + DTM inspection (~1 hour, lower
       confidence).

    The earlier-quoted "~5 m × ~0.5 m" is an order-of-magnitude
    placeholder, not a measurement. NHD is the strongest path to
    a defensible value before production.
  - Do we reduce `lake_hill_elev` from −6 m to ~−1 or −2 m to
    match real lake seasonal range and produce more frequent
    overflow events? Pipeline rework ~1 hour.
  - The decision interacts with B2 (Darcy gradients): if lateral
    inflow is small, lake stays in normal range and tuning is
    less critical. If inflow is large, tuning becomes essential.
  - **PI floated a vague idea (2026-05-19)** of adding a regional
    Darcy drain on the lake column via SourceMod to prevent
    unbounded water accumulation. No design exists yet. Whether
    to pursue this depends entirely on what Phase F shows — if
    QDRAI + ET closes the lake column budget on its own, this
    mechanism isn't needed. See Section 7.7 Option 5.

  Smoke test result (2026-05-12): the system stayed too dry for
  any of these levers to matter in 5 years. Production spinup
  will saturate soils and surface this question. Recommend
  getting a real NHD-derived stream width BEFORE starting the
  600-yr production spinup.

## Validation tasks (after A + B)

Once software (A) is in place and scientific decisions (B) are made:

- [ ] **C1.** Build the production routing-on case with the chosen
  gridcell area and a regenerated mesh of that area. Run for 10–20
  model years.

- [ ] **C2.** Inspect water-budget closure under routing-on. CTSM has
  internal balance checks; verify no warnings emerge that didn't
  appear under routing-off.

- [ ] **C3.** Compare column-level ZWT trajectories to routing-off
  Phase F baseline. Look for TAI signatures: water table rising in
  low-HAND bins during wet periods, finundated > 0 in those bins,
  o_scalar dropping in saturated columns.

- [ ] **C4.** Long-spinup decision. If short-run validation looks
  reasonable, configure a 600-yr routing-on case (either fresh start
  or extension from a Phase F routing-off restart with namelist
  switch). Otherwise iterate on B1/B2 with the PI.

## Software vs scientific summary

Explicit categorization so each task tracks the right authority:

| Decision | Type | Owner |
|---|---|---|
| Mesh generation workflow | Software | Engineering |
| xmlchange syntax for mesh-mode | Software | Engineering |
| Build + short test run | Software | Engineering |
| `grc%area` post-run validation | Software | Engineering |
| **Choosing the gridcell area value** | **Scientific** | **PI** |
| **`nhill_per_landunit` interpretation** | **Scientific** | **PI** |
| **Hydraulic conductivity reasonableness** | **Scientific** | **PI** |
| **Validation framing without precedent** | **Scientific** | **PI** |
| **Stream cross-section tuning (Swenson power laws vs OSBS coastal plain)** | **Scientific** | **PI** |
| **Lake overflow threshold (`lake_hill_elev` choice)** | **Scientific** | **PI** |
| Bin spacing revisit (if needed) | Mixed | PI + Engineering |
| Stream/lake retuning pipeline rework | Software | Engineering (once direction decided) |
| Long-spinup approval | Mixed | PI (timing) + Engineering (config) |

## Deliverable

A routing-on OSBS configuration, with mesh-mode case, validated for
short-run integrity and (subject to PI judgment) physically reasonable
TAI behavior. Either a 600-yr spinup with routing-on or a documented
decision not to pursue further with explicit scientific rationale.

## References

**Project context:**
- Phase G (lake column construction) — `phases/G-ctsm-lake-representation.md`
- Phase F (routing-off long spinup) — `phases/F-validate-deploy.md`
- Phase E.5 (bin design rationale) — `phases/E.5-bin-redesign.md`
- Lake column CTSM audit — `docs/lake-column-ctsm-audit.md`

**External:**
- CTSM Issue #1432 — https://github.com/ESCOMP/CTSM/issues/1432
- CTSM PR #1715 (original hillslope landing) — https://github.com/ESCOMP/CTSM/pull/1715
- DiscussCESM thread (Swenson's mesh recommendation) — https://bb.cgd.ucar.edu/cesm/threads/point-scale-simulation-with-ctsm5-3.11125/
- Swenson 2025 paper — https://doi.org/10.1029/2024MS004410
- mesh_maker documentation — https://escomp.github.io/ctsm-docs/versions/master/html/users_guide/using-mesh-maker/how-to-make-mesh.html

**CTSM source paths** (all under `$BLUE/ctsm5.3/`):
- `src/cpl/share_esmf/lnd_set_decomp_and_domain.F90` — single-column area assignment
- `src/main/initGridCellsMod.F90` — grc%area propagation
- `src/biogeophys/HillslopeHydrologyMod.F90` — InitHillslope, routing gate, nhill_per_landunit
- `src/biogeophys/SoilHydrologyMod.F90` — lateral Darcy flow routines
- `tools/site_and_regional/mesh_maker` — mesh creation tool
- `components/cmeps/cime_config/config_component.xml` — LND_DOMAIN_MESH variable definition

## Log

### 2026-05-19 — Phase H reframed as contingent; PI floated Darcy drain idea

Two updates after talking through the implications of the
routing-gate audit:

1. **Phase H is now contingent on Phase F.** Track A is complete;
   Tracks B/C may not be pursued at all. Inter-column lateral
   flow — the original scientific motivation for routing-on — is
   already running under `use_hillslope=.true.` in Phase F. The
   remaining value of routing-on is narrow (stream-coupling BC
   at chain bottom, internal `stream_water_volume` ledger,
   `VOLUMETRIC_STREAMFLOW` diagnostic). Whether that's worth the
   B1–B4 PI consultation cost + 600-yr respin depends on what
   Phase F shows. Status header and a new "Contingency note"
   section before the task lists capture this.

2. **PI floated the idea of a regional Darcy drain on the lake
   column** to prevent unbounded accumulation. The concern: in
   single-point mode there is no downstream cell to receive water
   leaving the gridcell. The idea is currently very vague — no
   design, no parameter values, no implementation. Whether to
   pursue it is contingent on Phase F's column-level evidence.
   Logged as Option 5 in Section 7.7 and as a sub-bullet under
   B4. Works under both routing-off and routing-on if needed.

### 2026-05-19 — Routing-gate source audit; reframe of project narrative

Comprehensive review of every `use_hillslope_routing` reference in
CTSM 5.3.085 source plus empirical check of h1a outputs from
`osbs.routing.test`/`control` and the operative
`osbs.swenson.spinup`. Result: `use_hillslope_routing` does NOT
gate inter-column lateral subsurface flow. That flow runs under
`use_hillslope=.true.` via `PerchedLateralFlow` and
`SubsurfaceLateralFlow`, both dispatched unconditionally from
`HydrologyDrainageMod.F90:139, 143`. The routing switch controls
stream-side state (channel geometry, internal `stream_water_volume`,
Manning streamflow, lnd→rof export) and a swap of the terminal-
column boundary depth from MOSART's `tdepth_grc` to internal
stream state.

Empirical confirmation: spinup case h1a shows column-level QRUNOFF
ranging −1.16×10⁻⁴ to +1.99×10⁻⁴ mm/s. Negative values at hillslope
columns are the unambiguous fingerprint of lateral inflow exceeding
outflow — only possible if inter-column flow is happening. ZWT
spans 0.011 to 8.6 m across columns, far more variation than would
emerge from isolated 1D columns under identical forcing.

Documents corrected:
- Problem section table (this doc, top) — revised what each switch toggles
- Section 7.5 table — Mechanism B "Active" → "Inactive under routing-off"; added Mechanism E (SaturatedExcessRunoffMod 295)
- A4 smoke-test log entry below — reinterpreted as boundary-condition delta, not "TAI emergence from lateral flow"
- Section 8 (new) — full source-code audit with file:line citations
- STATUS.md — cross-cutting concerns bullet rewritten + change-log entry
- `phases/F-validate-deploy.md` — Key Context callout (the "hydrologically isolated columns" claim was wrong)
- `phases/G-ctsm-lake-representation.md` — Stage 1 framing fix at lines 63-65

Implications for Phase H deliverable:
- Track A (mesh-mode workaround) still needed — solves the spval
  bug and is the necessary infrastructure for routing-on.
- Track B/C (PI decisions + routing-on validation) narrows in
  scope. Phase H validates the stream-side coupling, not the
  emergence of lateral flow itself.
- B2 (Darcy-gradient reasonableness) should be revisited with
  the 600-yr spinup's actual column-level QDRAI / ZWT
  trajectories rather than purely theoretical reasoning — the
  flow has been running all along; we can measure it.
- B4 (stream geometry, lake overflow threshold) is unaffected;
  still applies only when routing turns on.

Phase H status header updated to reflect the narrowed scope.

### 2026-05-12 — A3/A4 smoke test: routing-on infrastructure validated

Paired-case smoke test completed. Routing-on infrastructure works
end-to-end; cold-start dryness limits surface signal in 5 years but
the depth-profile TAI signature is emerging.

**Case setup** (executed 2026-05-12):

| Property | Test | Control |
|---|---|---|
| Case dir | `$CASES/osbs.routing.test` | `$CASES/osbs.routing.control` |
| COMPSET | 1850_DATM%CRUv7_CLM60%BGC… | (same) |
| Mesh | `output/mesh/osbs_mesh_90km2_c260512.nc` | (same) |
| SourceMods | 5 .F90 copied from `osbs.swenson.spinup/SourceMods/src.clm/` | (same) |
| fsurdat / hillslope_file | same as spinup case | (same) |
| `CLM_ACCELERATED_SPINUP` | off | off |
| Run | 5 yr cold start, 6-hourly output, ~monthly files | (same) |
| `use_hillslope_routing` | `.true.` | `.false.` |
| Build time | ~5.5 min | ~5.5 min |
| Run wallclock | 42 min | 52 min |

**Two namelist fixes caught during the run:**

1. `SOILLIQ_vr` is not a CTSM hist variable name — `SOILLIQ` itself
   is 2D on `levsoi`. Fixed: replaced in h3 list, removed from h1.
2. `VOLUMETRIC_STREAMFLOW` registration is gated by
   `use_hillslope_routing=.true.` in `WaterFluxType.F90:526`.
   Including it in the control case's `hist_fincl2` would crash
   the htapes check. Fixed: removed from control only; kept in test.
   `STREAM_WATER_VOLUME` and `STREAM_WATER_DEPTH` are gated only
   on `use_hillslope` and stay in both (zero in control).

**Validation results:**

| Check | Result |
|---|---|
| `grc%area` at gridcell | **90.006 km²** in BOTH cases (not spval) — mesh workaround verified |
| `VOLUMETRIC_STREAMFLOW` | Present only in test (~8 m³/s mean); correctly absent in control |
| `STREAM_WATER_VOLUME`, `_DEPTH` | Present in both (gated on `use_hillslope` only) |
| Water budget closure | No CTSM balance-check errors in either run |
| Gridcell aggregates (5-yr means) | Bit-identical to 4–7 sig figs: RAIN, TBOT, QVEGT, QSOIL, QFLX_EVAP_TOT, QRUNOFF, TWS, GPP, TOTECOSYSC, TOTSOMC, TOTVEGC, EFLX_LH_TOT, FSH |
| Column-level state, Year 1 | Bit-identical between test and control |
| Surface water `H2OSFC` | Zero everywhere, all timesteps, both cases — no surface pool emerges |
| `FH2OSFC` (inundated fraction) | Zero everywhere, both cases |

**TAI signature in deep soil moisture (Year-5 December, layer 15 ≈ 3 m):**

| Column | hill_elev (m) | Type | H2OSOI TEST − CTRL |
|---|---|---|---|
| 0 | −6.00 | lake | **+7.14×10⁻⁴** |
| 1 | −5.13 | deepest FZ | **+1.56×10⁻⁴** |
| 13 | +0.12 | bridge / TAI | **−9.61×10⁻⁵** |
| 20–24 | +4.5 to +12.5 | high upland | ~0 (no signal) |

Lake column got wetter under routing, bridge column drained
slightly, high upland unaffected at this timescale.

> **Reinterpreted 2026-05-19 after the routing-gate audit (Section 8).**
> Lateral inter-column flow runs in BOTH cases (`use_hillslope=.true.`
> is sufficient). What the test/control delta actually measures is
> the effect of the terminal-column boundary swap: routing-off uses
> `tdepth_grc` (likely 0 with our DATM+MOSART setup, where MOSART
> doesn't push `Sr_tdepth` back to CTSM); routing-on uses CTSM-internal
> stream state that builds up from column drainage. Lake column gets
> +7×10⁻⁴ wetter under routing because internal stream water at the
> chain bottom raises the effective stream depth seen by the
> lake-column Darcy gradient — i.e., less subsurface drainage out of
> the lake. Not "TAI emergence from lateral flow turning on"; rather,
> a stream-side BC effect on top of identical inter-column flow.
> Magnitude is small (~0.07% volumetric) because 5 years isn't long
> enough for the BC change to propagate deeply.

**Why the surface signal is invisible:**

Cold start from CTSM defaults → soil at field capacity (~0.30
vol/vol), surface dry. Florida potential ET (~1500 mm/yr) >
precipitation (~1300 mm/yr). With sandy OSBS soils, the column
drains faster than it can saturate. For surface water to pool,
soil must saturate first; that doesn't happen in 5 years from
cold start. Hence `H2OSFC = 0` everywhere in both cases.

This is exactly why the smoke test was designed as a smoke test:
infrastructure validation, not science emergence. **Both pass.**

**Stream-channel issue (Section 7) — no manifestation here.**
The "too generous 59 m × 1.52 m channel" concern is real but
invisible in this run because: (a) almost no water makes it to the
stream (everything ET'd away first), and (b) when water does flow,
the stream is so oversized that depth stays near zero. We'll only
see channel sizing matter once soils saturate during a real spinup.
Same for `lake_hill_elev = −6 m` overflow threshold — the lake
never fills enough to test it.

**Production estimate.** Test ran 42 min for 5 yr = ~7 yr/wallhr.
Control 52 min for 5 yr = ~5.8 yr/wallhr. At this rate, a 600-yr
production AD spinup with routing on would take ~85–100 wallclock
hours, very similar to `osbs.swenson.spinup`'s observed pace.

**Phase H Track A: complete.** A1, A2, A3, A4 all done. A5 (cookbook)
and A6 (canonical reference) folded into the doc itself and the
existing case directories. Ready to move to the scientific decisions
(B1–B4) and, once those are resolved, the production routing-on
spinup (Track C).

### 2026-05-12 — Stream + lake routing-on interface analysis

Documented the interaction between Swenson power-law stream
geometry, the lake column at `hill_elev = −6 m`, and the SourceMod
mechanisms when `use_hillslope_routing = .true.`. Full analysis in
Section 7 of research notes above. Headline findings:

1. **Stream depth (1.52 m) and width (59.23 m) are from Swenson's
   MERIT-global power laws** (`run_pipeline.py:1645-1651`):
   `depth = 0.001×A^0.4`, `width = 0.001×A^0.6`. The values were
   originally locked under routing-off (PI #4, 2026-03-30) where
   CTSM never read them. They're physically appropriate for the
   global river-system ensemble the power laws were calibrated
   against (Appalachians, Cascades — places where 90 km² = real
   tributary basin). They are **5–10× too generous for OSBS
   coastal plain reality** (~5 m wide × ~0.5 m deep real swales).

2. **The lake-column overflow threshold is 6 m above lake floor,
   not 0 (i.e. not "no release valve").** Earlier framing was
   wrong. The release valve is Mechanism A in our SurfaceWaterMod
   SourceMod (pond-builds for `hill_elev<0`, overflows when water
   surface exceeds 0). It's still active despite `spillheight=0`.
   The 6 m threshold comes from `lake_hill_elev = −6 m` directly,
   not from spillheight.

3. **Four mechanisms in the SourceMods** (verified by diffing
   `$CASES/osbs.swenson.spinup/SourceMods/src.clm/` against
   upstream): pond-building (A), terminal-column-only stream
   coupling (B), global elevation shift (C — disabled with
   spillheight=0), inter-column surface chain (D). Only Mechanism
   C was disabled.

4. **Two tuning levers, ~1 hour each:**
   - Override stream depth/width in pipeline → realistic OSBS
     cross-section
   - Reduce `lake_hill_elev` from −6 to ~−1 or −2 m → realistic
     overflow threshold

5. **Added B4 to Scientific decisions** for PI consultation. Both
   levers interact with B2 (Darcy gradient reasonableness) — the
   answer depends on actual lateral inflow magnitude, which the
   sniff test will measure directly.

6. Added two rows to the Software vs scientific summary: stream
   cross-section tuning and lake overflow threshold, both PI calls.

### 2026-05-12 — A2 done: 90 km² ESMF mesh built and validated

Phase H Track A milestone. Built the single-cell ESMF mesh for the
90 km² OSBS gridcell.

**Tooling decision.** CTSM's `mesh_maker.py` aborts on 1×1 input
(line 214). Discovered ESMF 8.9.1 binaries are shipped inside the
`ctsm` conda env (no `module load` needed): `ESMF_Scrip2Unstruct`,
`ESMF_Regrid`, `ESMF_RegridWeightGen`. Switched to the canonical
CDEPS workflow (SCRIP → ESMF mesh), which is also what CTSM's own
test suite uses (`test_sys_mesh_modifier.py:100`). Ported logic
from `tools/mkmapgrids/mkscripgrid.ncl` (Erik Kluzek, 2011, NCL)
into Python because we don't run NCL.

**SCRIP format pinned (verified, not guessed).** From inspecting
the ESMF parser behavior and the NCL reference:
- dimensions: `grid_size=1`, `grid_corners=4`, `grid_rank=2`
- coords in degrees (not radians); area in steradian
- corner ordering: CCW from SW, i.e. `[SW, SE, NE, NW]`

**Geometry computed.** For a square 90 km² cell at lat=29.689°:
side = √90 = 9.487 km → dLat ≈ 0.0853°, dLon ≈ 0.0982° (= side /
(111.195 km/° × cos(lat))). Area in steradians via the spherical-cap
formula `(sin(lat_N) − sin(lat_S)) × dLon_rad = 2.2173e-6 sr`,
matching `90 / R²` to four significant figures.

**Artifacts.** Both under `output/mesh/`:
- `osbs_scrip_90km2_c260512.nc` (16 KB) — SCRIP input
- `osbs_mesh_90km2_c260512.nc` (13 KB) — ESMF mesh output

**Script.** `scripts/osbs/make_osbs_scrip.py` reads center / area
defaults from the production hillslope NetCDF (so the common-case
invocation is just `python scripts/osbs/make_osbs_scrip.py
--verbose`). CLI overrides for lat / lon / area let the same script
build the (c′) NEON-boundary mesh later if needed (`--area 36.81`).

**Schema validation.** Side-by-side comparison against
`fv0.9x1.25_141008_ESMFmesh.nc`:
- Same six core variables, same dimension layout.
- One benign difference: `numElementConn` is int32 in ours vs int8
  in the reference. Both hold values 1–4. Worth flagging in case
  CTSM strict-types the read at runtime, but unlikely to matter.
- One ESMF 8.9 addition: `origGridDims` variable. Records the
  original SCRIP grid dims `[1, 1]`. Should be ignored by CTSM.

A1 → A2 → ready for A3 (case construction with this mesh).

### 2026-05-13 — Revisions to gridcell-area decision space

Two changes following PI pushback on the 2026-05-12 framing:

1. **Time estimate for re-running pipeline at NEON boundary was
   conservative.** Revised down from "~half-week" to a split: ~half
   day for rectangular tile-subset within the NEON polygon, ~1–2
   days for arbitrary-polygon clipping (which is the real
   engineering cost — the pipeline assumes rectangular tile grids).

2. **Added rescale-only option (c′)** to B1: keep existing rep
   hillslope statistics, change only `nhill_implicit` and `AREA` to
   target the NEON-site scale. ~1-hour rework. Defensible at OSBS
   because the 90 km² LIDAR rectangle and the 36.81 km² NEON polygon
   are both inside the same uniform sandhill-and-depression
   landscape. Now three options worth presenting to the PI:
   (a) 90 km² no rework, (c′) 36.81 km² rescale-only, (c) 36.81 km²
   full re-derive.

   Added a "What the mesh actually does" subsection at the top of
   Section 5 to explain why rescaling works in principle — the
   surface dataset is mostly ratios that don't care about gridcell
   size, and the mesh is purely the scale anchor.

### 2026-05-12 — Deep research pass on inputs, NEON, gridcell area, mesh tooling

Five-strand research before starting Phase H software work:

1. **Existing input inventory** — our case uses sgerber's setup: f09
   global surface-data subset, CRUNCEPv7 0.5° DATM (1901–1921 cycle),
   our 2026-05-05 hillslope NetCDF (AREA = 90 km²). All `MESH` xml
   variables UNSET (PTS mode, the spval-bug pathway).

2. **CTSM tower-run capability** — `cime_config/usermods_dirs/clm/
   NEON/OSBS` exists with `run_tower` entry point. NEON OSBS at
   29.68819°N, -81.99345°W (matches our PTS_LAT/LON exactly). Uses
   NEON tower forcing for 2018–2021 transient only — not a drop-in
   for 1850 AD spinup but the canonical template for *transient*
   site-specific runs.

3. **NEON data products** — survey landed five biogeochem products
   worth incorporating into production fsurdat (megapit DP1.00096,
   distributed soil chem/moisture/temp/CO₂), plus full DATM-forcing
   suite from tower meteorology for post-2018 transient runs. None
   help 1850 spinup. NEON terrestrial sampling boundary at OSBS is
   36.81 km² — relevant for the gridcell-size decision.

4. **Gridcell-area analysis** — pipeline's `nhill_implicit ≈ 533`
   pins gridcell area to ~90 km². Other options (NEON boundary,
   Swenson 0.5°, Swenson 0.9°×1.25°) require pipeline rework and most
   of them are physically dubious for site-specific work.
   Recommendation: option (a), 90 km². B1 updated accordingly.

5. **Mesh tooling and community precedent** — `mesh_maker` works
   from our hillslope NetCDF with a small caveat (km² vs radians²
   units; skip `--area` or convert). **Issue #1432 has zero working
   community precedent for single-point routing-on.** Johanna Teresa's
   Feb 2025 attempt is the only public attempt and she never posted
   back. We will be the first documented working case — added A6 to
   record this as canonical reference.

A1 marked Complete (research-only task; tool exploration done). A2
through A5 remain pending. A6 added.

### 2026-05-11 — Phase created from research findings

Phase split out from G Stage 2 after a research pass confirmed the
`grc%area = spval` issue is canonical **CTSM Issue #1432** (open since
2021-07-20). Mesh-mode workaround verified as Swenson's personal
recommendation (per Feb 2025 DiscussCESM thread). All source line
numbers in this doc verified against CTSM 5.3.085.

Two findings beyond what was previously documented:

1. **Secondary `grc%area` exposure** in `HillslopeUpdateStreamWater`
   (HillslopeHydrologyMod.F90:1117-1121). Gated by `active_stream`
   rather than `use_hillslope_routing` directly. Under routing-off,
   `stream_channel_length` stays 0 → `active_stream` stays false →
   no spval propagation. Indirect protection that is load-bearing but
   fragile.

2. **No defensive guards anywhere** in CTSM `src/` for `grc%area ==
   spval`. Zero asserts, zero error checks, zero fallback logic.

Phase G marked Complete (Stage 1 done; Stage 2 moved here as Phase H).
