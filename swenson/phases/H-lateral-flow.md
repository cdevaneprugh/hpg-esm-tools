# Phase H: Enable Lateral Subsurface Flow

Status: Pending (research + PI decisions before implementation)
Depends on: Phase F (long spinup provides routing-off baseline)
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

**Every link past "lateral subsurface flow" is dormant in our current
configuration.** CTSM has two hillslope namelist switches:

| Switch | Effect | Current state |
|---|---|---|
| `use_hillslope` | Multi-column subgrid structure: per-aspect / per-elevation columns with their own area, geometry, slope, aspect. Aspect-dependent radiation, elevation downscaling, independent per-column vertical water balance. | **ON** |
| `use_hillslope_routing` | Column-to-column lateral subsurface flow via Darcy gradient. This is the mechanism that physically couples upland → flood zone → lake. | **OFF** |

Under routing-off, columns are hydrologically isolated 1D soil columns
that share a gridcell but never exchange water. Differences between
columns at runtime come from independent per-column forcing,
aspect/elevation-dependent radiation, and independent vertical water
balances — *not* from lateral coupling.

All ~22 osbs cases on this machine (sgerber's + ours) have run
routing-off. The science question hasn't been asked at OSBS yet. Phase
H is the work to ask it.

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

- [ ] **A3.** Construct a new test case as a sibling of
  `osbs.swenson.spinup`. Override:
  - `PTS_LAT = -999.99` (unset)
  - `PTS_LON = -999.99` (unset)
  - `LND_DOMAIN_MESH = /path/to/test_mesh.nc`
  - `MASK_MESH = /path/to/test_mesh.nc`
  - `use_hillslope_routing = .true.` (in user_nl_clm)

  Build, run for 1–5 model years. Goal: verify the model completes
  without errors, not yet to do science.

- [ ] **A4.** Inspect output to verify:
  - `grc%area` in h0a is a real number (not 1e36)
  - `nhill_per_landunit` is computed and sensible (need to either
    add a hist variable or check via debug print in a SourceMod)
  - Stream channel length is a sensible magnitude
  - Water budget closes (CTSM internal balance checks pass)

- [ ] **A5.** Document the cookbook of xmlchange commands needed to
  migrate a PTS_MODE case to mesh-mode. This will be reused for the
  production routing-on run after the scientific decisions land.
  Skeleton recipe in research notes Section 5 — refine after A2–A4.

- [ ] **A6.** Record the working configuration as the canonical
  reference for single-point routing-on (since no community precedent
  exists — see research notes Section 6). Either in this phase doc,
  in a new `docs/` file, or as a tutorial-style README under
  `scripts/`. Decision deferred until after A4 validates the
  configuration actually works.

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
| Bin spacing revisit (if needed) | Mixed | PI + Engineering |
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
