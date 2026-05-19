# Phase F: Validate and Deploy

Status: In progress (osbs.swenson.spinup, 600-yr accelerated AD spinup
in flight as of 2026-05-11; 4-stream hist config; osbs5 superseded after
ntapes restriction at restart prevented adding h2/h3 streams mid-spinup)
Depends on: Phase E (parameter pipeline), Phase E.5 (bin scheme + lake column)
Blocks: Phase H (routing-on validation)
Runs in parallel with: Phase G Stage 1 (lake column construction + CTSM ingestion, complete)

## Scope (refined 2026-05-06)

**Phase F is the long convergent spinup deliverable.** The originally-
scoped 1–5 yr branch comparison from osbs2 is no longer the primary
focus — the 2026-04-25 PI direction folded the lake column into the
pipeline output, dissolving the original "F provides lake-less
baseline; G adds lake column" sequencing. The pipeline's hillslope
NetCDF always includes the lake column now.

The osbs5 case validates both Phase F and Phase G Stage 1:

- **Phase F goal:** Run the case to BGC equilibrium (TOTECOSYSC drift
  < 3% over last 50 yr per CTSM convention). Currently 100 yr of an
  expected ~600 yr accelerated spinup. Plots inspect convergence of
  carbon pools, water budget, column-level differentiation under
  routing-off configuration.
- **Phase G Stage 1 goal** (tracked in `phases/G-ctsm-lake-representation.md`):
  Verify CTSM correctly ingests the 25-column hillslope file with
  lake at chain index 1, sensible column weights, no NetCDF read
  errors. **Already confirmed** by the 100-yr osbs5 run.

## Problem

The final hillslope file needs validation at two levels before use in production runs: (1) physical plausibility of the parameters themselves, and (2) correct behavior when ingested by CTSM. The osbs2 baseline case (860+ year spinup with Swenson's global hillslope data) provides the comparison target.

## Key Context

**osbs2 runs with `use_hillslope_routing = .false.` and `PCT_LAKE = 0`** (no lake land unit, no stream routing). Phase F validation matches this configuration exactly — the only variable changed is the hillslope file itself. Stream params are irrelevant (never read with routing off).

`PCT_LAKE = 0` means there is no separate lake land unit in osbs2. The hillslope occupies 100% of the vegetated landunit. This is already the configuration Phase G needs (lake-within-hillslope via modified stream bucket). Phase F establishes the baseline that Phase G compares against.

### What "routing off" means for interpretation

> **Corrected 2026-05-19 (see `phases/H-lateral-flow.md` Section 8 for
> the source-code audit).** The earlier text in this section asserted
> that under routing-off "columns are hydrologically isolated 1D soil
> columns that share a gridcell but never exchange water" — that
> claim was wrong. `use_hillslope_routing` does NOT gate inter-column
> lateral subsurface flow. The lateral-flow machinery
> (`PerchedLateralFlow`, `SubsurfaceLateralFlow` in
> `src/biogeophys/SoilHydrologyMod.F90`) runs whenever
> `use_hillslope=.true.`, dispatched unconditionally from
> `HydrologyDrainageMod.F90:139, 143`. The section below is the
> corrected reading; preserved for context, with the wrong
> sub-paragraphs annotated.

CTSM has two hillslope namelist switches:

- `use_hillslope = .true.` — multi-column structure with per-aspect /
  per-elevation columns AND **inter-column lateral subsurface flow**
  via Darcy gradient at the water table and perched water table.
  Aspect-dependent radiation, elevation-based downscaling, per-column
  water balance, and lateral water redistribution between adjacent
  columns. **ON in Phase F → lateral flow has been running.**
- `use_hillslope_routing = .true.` — stream-side state: CTSM-internal
  `stream_water_volume`, channel geometry, Manning streamflow, and
  swap of terminal-column boundary depth from MOSART's `tdepth_grc`
  to internal stream state. **OFF in Phase F → terminal column sees
  external `tdepth_grc` (likely 0 with our DATM+MOSART setup).**

Under routing-off, what's actually running is:

- Per-column water and energy balance (forcing, ET, infiltration,
  drainage)
- Aspect-dependent radiation and elevation downscaling
- **Inter-column lateral subsurface flow** — water moves from
  higher-elevation columns to lower-HAND columns via the Darcy
  gradient at column-to-column interfaces, applied as drainage to
  `h2osoi_liq` (see `SoilHydrologyMod.F90:2261-2263, 2386-2388,
  2433-2509`)
- Inter-column surface-water chain bookkeeping in
  `SurfaceWaterMod.F90:547-561` (SourceMod Mechanism D, also
  not routing-gated)

What does NOT happen under routing-off:

- Internal CTSM stream-channel state (no `stream_water_volume`
  ledger, no Manning streamflow, no lnd→rof streamflow export)
- The lake column's terminal Darcy gradient against an internal
  stream that builds up from chain drainage — instead the lake
  sees `tdepth_grc` from the coupler

So when interpreting Phase F plots: lake column wetness, FZ
ponding, and ZWT differentiation all reflect both per-column forcing
AND lateral water redistribution. The columns DO communicate
hydrologically; what they don't communicate with is a CTSM-internal
stream channel. Differences between Phase F and a future routing-on
case (Phase H) will isolate the BC at the bottom of the chain, not
the lateral-flow mechanism itself.

True TAI validation now means measuring whether the lateral flow
that *is already running* produces a physically-meaningful TAI
signal at OSBS — water-table rise in low-HAND bins, suppressed
o_scalar in saturated columns, CH4 emergence. The 600-yr spinup
provides the data to answer this without needing routing-on.
Routing-on (Phase H) refines the BC; it does not enable the
mechanism.

## Tasks

- [x] Compare custom hillslope file to Swenson reference (`hillslopes_osbs_c240416.nc`) — parameter-by-parameter (`output/osbs/2026-05-04_production/`, `STATUS.md` "Confirmation from Swenson's published OSBS gridcell" table)
- [x] Physical plausibility checks (Phase E.5 production summaries):
  - Elevation distribution matches known OSBS topography (range −6.35 → +17.02 m raw HAND)
  - Stream network aligns with known hydrology (`stream_network.png`)
  - HAND values reasonable for low-relief wetlandscape (meters, not hundreds)
- [x] Build CTSM case with custom hillslope file as a fresh startup (NOT branched per user direction): `osbs5.swenson.spinup` — `use_hillslope_routing = .false.`, `PCT_LAKE = 0`, SourceMods retained with `spillheight = 0.0`, accelerated AD spinup (`CLM_ACCELERATED_SPINUP = on`).
- [x] Run 100-year accelerated AD spinup (job 31936055, completed 2026-05-06)
- [x] Generate 8 spinup-analysis plots (5 gridcell, 3 column-level) at `$SWENSON/output/2026-05-06_osbs5_spinup_timeseries/`
- [ ] **Extend spinup to convergence** (post-maintenance): continue from year 101 → 601, target < 3% drift in `TOTECOSYSC` over last 50 years. Standard CTSM spinup criterion.
- [ ] **Optional apples-to-apples comparison:** parallel case with Swenson reference hillslope file (`hillslopes_osbs_c240416.nc`) under same osbs5 setup; side-by-side plots attribute differences specifically to hillslope file
- [ ] Compare to osbs2 / osbs4-6 outputs (qualitative, since they used different hillslope files entirely):
  - Water table depth (ZWT) seasonal cycle
  - Soil moisture profiles
  - Carbon fluxes (GPP, NEE)
  - Energy balance (latent/sensible heat)
  - Column-level differences (h1 stream)

## Deliverable

Validated hillslope file ready for production runs. Comparison document showing custom vs. global hillslope parameter differences and their impact on simulated fluxes. **Stage 1 deliverable** (column weights and per-column water/carbon behavior under independent 1D balances) is in hand from the 100-yr osbs5 run; convergent spinup completes the picture for routing-off use.

## Log

### 2026-05-19 — Reframe: lateral flow is active in Phase F (routing-off does NOT gate it)

Routing-gate source audit (full trace in `phases/H-lateral-flow.md`
Section 8) corrects this phase's "Key Context" interpretation. The
earlier framing claimed routing-off meant columns are
"hydrologically isolated 1D soil columns that share a gridcell but
never exchange water." That was wrong. `PerchedLateralFlow` and
`SubsurfaceLateralFlow` (CTSM `src/biogeophys/SoilHydrologyMod.F90`)
run unconditionally under `use_hillslope=.true.`, dispatched from
`HydrologyDrainageMod.F90:139, 143`. Inter-column Darcy flow,
qflx_latflow_in/out accumulation, and net-flow application to
h2osoi_liq all happen without any routing gate.

Empirical confirmation: spinup case h1a shows column-level QRUNOFF
ranging −1.16×10⁻⁴ to +1.99×10⁻⁴ mm/s; negative values are the
fingerprint of lateral inflow exceeding outflow at receiving columns
— only possible if inter-column flow is running.

What this means for Phase F's deliverable:
- The 600-yr spinup is producing **real TAI physics** — water
  redistributes laterally from upland columns to low-HAND
  columns. Analysis of the column-level ZWT/H2OSOI/QDRAI
  trajectories should reveal whether the OSBS Darcy gradients
  are large enough to produce visible TAI behavior (water-table
  rise in flood-zone columns, drier ridges, seasonal cycles of
  low-HAND saturation).
- The "Phase F is routing-off validation; Phase H delivers
  TAI" mental model was wrong. Phase F is already delivering
  TAI physics. Phase H adds the stream-side coupling.
- The Deliverable framing was over-modest: "column weights and
  per-column water/carbon behavior under independent 1D
  balances" — should read "and inter-column lateral
  redistribution under the stream-coupling BC from MOSART's
  tdepth_grc."

"Key Context" section above corrected inline; original text was
overwritten because the original framing was factually incorrect
rather than provisional. Source citations in
`phases/H-lateral-flow.md` Section 8 are the canonical reference.

### 2026-05-06 — 100-yr accelerated AD spinup complete; first analysis plots generated

Built `osbs5.swenson.spinup` as a fresh startup (NOT branched from
osbs2 — per user direction, "see how this spins up on its own").
Configuration:

- Compset, SourceMods, surfdata, DATM forcing all match `osbs4-6`
- `RUN_TYPE = startup`, `CLM_ACCELERATED_SPINUP = on`
- `STOP_N = 100`, `RESUBMIT = 0`, `JOB_WALLCLOCK_TIME = 15:00:00`
- `hillslope_file` = our 2026-05-05 production NetCDF (lake at chain
  index 1, per-rep rescaled, `spillheight = 0.0`)
- `use_hillslope = .true.`, `use_hillslope_routing = .false.` (matching
  osbs4-6)

SLURM job 31936055 ran 2026-05-05 18:11 → 2026-05-06 06:49 (12:37
elapsed). Throughput came in at ~7.9 yr/wall-hr, slightly slower than
sgerber's 9.5 yr/hr osbs4-6 average. Output: 60 h0a + 40 h1a monthly
files, 5 restart bundles. Archived cleanly via manual `case.st_archive`
on the login node (st_archive SLURM job 31936056 was reservation-blocked
by maintenance window; canceled and run directly).

**Spinup analysis plots (8 total, at
`$SWENSON/output/2026-05-06_osbs5_spinup_timeseries/`):**

Tier 1 (h0a, gridcell):
- `h0a_TOTECOSYSC.png` — 1400 → 4200 gC/m², still climbing
- `h0a_TOTSOMC.png` — slow pool ramping (nowhere near steady state)
- `h0a_TOTVEGC.png` — fast pool, quasi-steady within ~10 yr
- `h0a_GPP.png` — seasonal cycle established quickly
- `h0a_TWS.png` — water budget steady within ~5 yr

Tier 2 (h1a, column-level, grouped Lake/FZ/Upland):
- `h1a_ZWT.png` — lake at ZWT≈0 (saturated), FZ 0–1m, upland deeper.
  Reasonable column-level differentiation under independent 1D water
  balances.
- `h1a_H2OSFC.png` — lake holds 1–5m of standing water (dominates),
  FZ shows occasional ponding events, upland near zero
- `h1a_GPP.png` — upland highest, FZ slightly lower, lake similar to
  upland (vegetation above the −6 m datum)

Tooling produced/updated for this milestone:
- `scripts/hillslope.analysis/plot_col_timeseries.py` (new) — full-span
  column-level plot with Lake/FZ/Upland classification matching the
  Phase E.5 hillslope_params plot color scheme. Two-panel: group-
  weighted means + individual columns.
- `scripts/hillslope.analysis/plot_timeseries_full.py` (updated) —
  adaptive title and tick spacing, derived from `mcdate` diffs.

Both committed (ed062b8) and pushed.

**Caveats:**

- 100 yr is far from BGC equilibrium. Standard CTSM guidance: 600 yr
  accelerated → 200 yr post-AD. Plots show monotone ramp-up, not
  steady state.
- Routing OFF in this case. Column differentiation observed in
  plots is from BOTH independent per-column forcing AND inter-
  column lateral exchange (the 2026-05-19 routing-gate audit
  established that lateral flow runs under `use_hillslope=.true.`,
  not `use_hillslope_routing`; see `phases/H-lateral-flow.md`
  Section 8). The TAI mechanism is ACTIVE in Phase F output;
  what's missing is the stream-side coupling at the chain bottom.
- No baseline comparison case yet — would need a parallel osbs5-style
  case with Swenson's reference `hillslopes_osbs_c240416.nc` to
  attribute observed patterns specifically to our hillslope file.

**Next steps (post-maintenance window):**

1. Extend `osbs5.swenson.spinup` to year ~600 with
   `CLM_ACCELERATED_SPINUP = on` (matches sgerber's osbs4 length).
2. Optional: post-AD continuation (`CLM_ACCELERATED_SPINUP = off`,
   another 200 yr) for full equilibrium.
3. Re-generate the 8 plots over the longer span; verify TOTECOSYSC
   drift < 3% over last 50 yr.
4. Optional apples-to-apples comparison case (Swenson reference
   hillslope file, same setup).
