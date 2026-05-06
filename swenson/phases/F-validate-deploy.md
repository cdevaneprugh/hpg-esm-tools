# Phase F: Validate and Deploy

Status: In progress (osbs5.swenson.spinup, 100-yr accelerated AD spinup
complete 2026-05-06; 8 spinup-analysis plots generated; longer spinup
to convergence pending after HiPerGator maintenance window)
Depends on: Phase E (parameter pipeline), Phase E.5 (bin scheme + lake column)
Blocks: Phase G Stage 2 (routing-on validation)
Runs in parallel with: Phase G Stage 1 (lake column construction + CTSM ingestion)

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

CTSM has two hillslope namelist switches:

- `use_hillslope = .true.` — multi-column structure: per-aspect /
  per-elevation columns with their own area, geometry, slope, aspect.
  Aspect-dependent radiation, elevation-based atmospheric downscaling,
  per-column water balance. **ON in Phase F.**
- `use_hillslope_routing = .true.` — col-to-col lateral subsurface
  flow (Darcy gradient between adjacent columns). This is the
  mechanism that physically couples upland → flood-zone → lake.
  **OFF in Phase F (matching osbs2).**

With routing off, columns are **hydrologically isolated 1D soil
columns** that share a gridcell but never exchange water. Differences
between columns at runtime come from:

- Independent forcing (each column's own precipitation, transpiration,
  drainage)
- Aspect/elevation-dependent radiation and downscaling
- Independent vertical water balance per column

What does NOT happen with routing off:

- Runoff propagating from upland to lake/wetland
- Saturation-induced lateral inflow filling the lake column
- TAI dynamics in the strict sense (lateral wet/dry boundary expansion)

So when interpreting Phase F plots: lake column "stays wet" because
its 12% area share receives precipitation; FZ "ponds seasonally"
because of its own H2OSFC accumulation, not because runoff drained
in from upland. The columns differentiate but do not communicate.

True TAI validation (lateral coupling, water table rise driven by
upland saturation, CH4 emergence) is **Phase G Stage 2** —
`use_hillslope_routing = .true.` — and requires its own configuration
work (see Phase G doc, 2026-05-05 log entry).

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
- Routing OFF: column differentiation is from independent per-column
  forcing, not lateral exchange. TAI mechanism is dormant.
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
