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

## Software problems — engineering only

Pure engineering / configuration tasks. Can be planned, prototyped,
and reported on without PI input.

- [ ] **A1.** Identify the input format and command-line interface of
  `tools/site_and_regional/mesh_maker`. Document expected inputs (lat,
  lon, area variable names) and the produced ESMF mesh format. No PI
  input needed — pure tool exploration.

- [ ] **A2.** Generate a test mesh file for an arbitrary single cell
  (e.g., 1 km² centered on OSBS coords) to verify the workflow
  end-to-end before committing to the science-chosen area. Output:
  one NetCDF mesh file.

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

## Scientific decisions — PI consultation required

These determine what the model represents physically. Software work
above is unblocked by these but **a long routing-on spinup is not
meaningful without resolving them.**

- [ ] **B1. Gridcell area choice.** Not pre-decided. The central
  scientific question of this phase. Options to discuss with PI:

  | Option | Area | What it physically represents |
  |---|---|---|
  | (a) LIDAR production domain | ~90 km² | The actual OSBS study area |
  | (b) Single representative hillslope | ~0.17 km² | A single TAI unit, no tiling |
  | (c) Swenson 0.5°×0.5° convention | ~2,700 km² | OSBS represents a regional cell |
  | (d) Swenson 0.9°×1.25° (paper resolution) | ~12,654 km² | OSBS represents a continental-scale cell |
  | (e) Something else | — | — |

  Sub-questions for the PI:
  - What does `nhill_per_landunit` physically represent — number of
    OSBS-like hillslopes that tile the gridcell? Is "tiling" a
    meaningful concept at this site?
  - If a Swenson-convention area, are we implicitly claiming OSBS
    represents a much larger region? Defensible for what science
    question and what publication framing?
  - Does the choice need to match published Swenson methodology for
    intercomparison, or are we doing site-specific science where we
    can choose differently?

  This choice affects stream channel length, lateral flow rates, and
  what gridcell aggregates physically mean. Different choices likely
  produce qualitatively different TAI signal magnitudes.

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
