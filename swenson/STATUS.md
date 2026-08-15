# STATUS — Swenson Hillslope for OSBS

**Updated:** 2026-08-15

## Project context

This work is part of the DOE-funded study of water and carbon dynamics in
coastal-plain wetlandscapes. The central scientific question is the role
of the **terrestrial-aquatic interface (TAI)** — the dynamic boundary
between wet and dry zones in low-relief landscapes — which drives a
disproportionate share of carbon exchange. Current Earth System Models
treat wetland extent as static and lack lateral water flow between
hillslope columns, so they cannot resolve TAI dynamics.

The OSBS test site (Ordway-Swisher Biological Station, north-central
Florida sandhills with wetland depressions) has 1m NEON LIDAR coverage
— ~90× finer resolution than the global 90m MERIT DEM that the
standard CTSM hillslope dataset is built from. Our goal is to apply
the Swenson & Lawrence (2025) representative hillslope methodology to
that 1m data to produce site-specific hillslope parameters that
capture the fine-scale drainage structure needed for TAI dynamics.

The output is a CTSM-compatible hillslope NetCDF replacing the global
`hillslopes_osbs_c240416.nc`. Validation runs with the new file should
show TAI emergence (water table rise near lake → suppressed aerobic
decomposition → CH4 production). Inter-column lateral subsurface flow
is already active under `use_hillslope=.true.` (see the 2026-05-19
routing-gate audit in Cross-cutting concerns) — the operative
`osbs.swenson.spinup` case is delivering this physics now. **We are
not assuming we will pursue a routing-on configuration for CTSM.**
That decision is contingent on what Phase F shows.

## Scientific decisions (locked)

Values that downstream work depends on. Each row links to the phase
where the decision was made and (where applicable) the date of PI
consultation.

| Decision | Value | Date locked | Reference |
|---|---|---|---|
| Characteristic length scale | Lc = 356 m (production domain) | 2026-02-11 | Phase C |
| Accumulation threshold | A_thresh = 63,362 m² (= 0.5 × Lc²) | 2026-02-11 | Phase C |
| FFT preprocessing | min_wavelength = 20m cutoff | 2026-02-11 | Phase C |
| Flow routing resolution | 1m (no subsampling) | Phase B | Phase B |
| Production domain | R4-R12, C5-C14 (90 tiles, 9×10 km, 0 nodata) | 2026-03-30 (PI) | Phase D |
| DEM conditioning | Standard fill for D8 | 2026-03-30 (PI) | Phase E |
| Slope/aspect source | NEON DP3.30025.001 directly | 2026-03-23 (PI) | Phase E |
| Water masking | Dual-mask (streams for delineation, wide mask for HAND) | 2026-03-27 | Phase E |
| Outlier cutoffs (raw HAND) | Q01 = -6.34 m, Q99 = +17.46 m (true discard) | 2026-05-02 | Phase E.5 |
| HAND binning | 24 bins TAI-focused (12 FZ + 12 upland, 0.25 m floor) | 2026-05-04 (PI) | Phase E.5 |
| Lake column placement | Chain index 1 (land columns shift to 2-25) | 2026-04-25 (PI) | Phase E.5, G |
| Lake hill_elev | -6.0 m (chain-bookkeeping value) | 2026-05-04 (PI) | Phase E.5, G |
| Lake hill_distance | 0.5 × Bin 1 distance (dynamic) | 2026-05-04 | Phase E.5, G |
| Lake hill_area | Σ(water_mask × pixel_area) ≈ 10.68 km² (rescaled per-rep) | Phase E.5 | Phase E.5 |
| Lake hill_width | 0.5 × NWI total perimeter | 2026-04-25 (PI) | Phase G |
| Lake hill_slope / hill_aspect | 0 / 0 | 2026-04-25 (PI) | Phase G |
| Per-rep rescaling | nhill_implicit ≈ 533; lake `wtlunit` 12.3% | 2026-05-05 | Phase E.5 |
| SPILLHEIGHT | 0.0 (namelist override; SourceMod inert) | 2026-04-30 (PI) | Phase E.5 |
| Routing config (Phase F) | use_hillslope_routing = .false. | inherited from osbs2 | Phase F |
| 2017 precip source (NEON forcing) | Secondary tipping bucket (DP1.00045) — primary weighing gauge down 2017 Jul–Dec; gauges agree r=0.96 / ~2% | 2026-08-14 | Phase I I5 |

## Open questions

### Phase H (routing-on) — contingent, may not be pursued

The 2026-05-19 routing-gate audit removed the original motivation
for routing-on. Inter-column lateral flow already runs under
`use_hillslope=.true.` (Phase F is delivering it). Routing-on's
remaining value is narrow: stream-coupling boundary condition at
the chain bottom, internal `stream_water_volume` ledger, and the
`VOLUMETRIC_STREAMFLOW` diagnostic. Whether that's worth the PI
consultation cost and 600-yr respin depends on what Phase F shows.

The four PI-consultation items previously listed here (B1
gridcell area, B2 Darcy gradient sanity, B3 validation framing,
B4 stream geometry + lake overflow) are frozen pending Phase F
evidence. Full task descriptions live in
`phases/H-lateral-flow.md` Section "Scientific decisions — PI
consultation required" but are flagged as a frozen record there,
not an active to-do list.

A separate, vague idea — the PI floated a regional Darcy drain
on the lake column via SourceMod to prevent unbounded
accumulation — is also contingent on Phase F. No design exists.
Logged as Section 7.7 Option 5 in `phases/H-lateral-flow.md`.

### Awaiting external clarification

- **Lee 2023 OSBS LIDAR vintage** — awaiting response from Cohen.
  Affects the framing of field-survey comparison in Phase E.5
  documentation. Non-blocking.

### Status of secondary tracks

- **Post-AD continuation (`osbs.swenson.post-ad`).** Initial attempt
  2026-05-19 hit a nitrogen-state error in
  `SoilBiogeochemNitrogenStateType.F90:874` ("Error in entering/exiting
  spinup - should occur only when nstep = 1"). Recovered by 2026-05-20;
  two 100-yr runs completed successfully (jobs 32805477 + 32890432,
  2026-05-20 → 2026-05-21). Currently idle. Not prioritized while PI
  investigates the AD-spinup TAI / bridge-zone questions.

### Open scientific questions surfaced by Phase F analysis (2026-05-19)

> **Update 2026-07-15: the production hillslope file is no longer frozen.** The PI
> can work with the existing file
> (`output/osbs/2026-05-05_production/hillslopes_osbs_production_c260505.nc`) via
> soil-value adjustments — generally working, some concerns remain, left in the
> PI's wheelhouse. The two questions below are still being investigated.

These are independent follow-ups from the routing-off results, not
Phase H prerequisites.

- **O_SCALAR (anoxia) is essentially 1.0 across the full 25-column ×
  600-year array.** The TAI carbon-side signature (suppressed aerobic
  decomposition in saturated columns) is not visible in this output.
  Headline issue for the project's central scientific question.
- **Bridge-zone anomaly** at chain indices 3-6 (HAND -3 to -1.5 m).
  These columns have the deepest water tables (1.3-1.4 m) of any
  lower-hillslope column, despite being closest to the lake. Caused
  by steep Darcy gradients (Δh/L ≈ 0.4-0.6 m/m) over short distances.
  Connects to B2 (hydraulic conductivity / bin spacing) — now de facto
  a Phase F follow-up rather than a Phase H prerequisite.

Plots backing these findings: `output/2026-05-19_osbs.swenson.spinup_timeseries/`
(8 PNGs + h0a/h1a annual NetCDFs). The narrative analysis report
produced during the 2026-05-19 close-out was local-only and is no
longer on disk; verdicts above and in the 2026-05-19 change-log entry
are the canonical record.

## Current state at a glance

| Phase | Topic | Status | One-line note |
|---|---|---|---|
| A | pysheds UTM CRS | Complete | 28 synthetic tests + MERIT regression locked |
| B | Flow routing resolution | Complete | 1m at 64GB verified |
| C | Characteristic length scale | Complete | Lc = 356 m |
| D | Pipeline rebuild | Complete | All Phase A/B/C fixes integrated; equation audit passed |
| E | Parameter set | Complete | 16-bin hybrid (superseded by E.5) |
| E.5 | Bin redesign + lake column | Complete | 24-bin TAI scheme + lake at chain index 1 |
| E.6 | NWI mask hole-fill | Complete | binary_fill_holes; 400K hole pixels fixed |
| F | Validate and deploy | **Complete (routing-off, AD only)** | 600-yr spinup analyzed 2026-05-19. Convergence PASS; TAI signal ABSENT; lake stable. PI investigating TAI / bridge-zone. Plots at `output/2026-05-19_osbs.swenson.spinup_timeseries/` |
| G | Submerged lake column | Complete | Stage 1 done; Stage 2 moved to Phase H |
| H | Stream-side coupling (routing-on) | **Track A complete; B/C on hold** | May not be pursued at all — original motivation collapsed when 2026-05-19 audit showed lateral flow already runs under `use_hillslope=.true.` |
| I | NEON atmospheric forcing | **Dataset COMPLETE (I1–I5) + ingestion smoke PASSED (2026-08-15); I6a calibration done (5.54 min/yr → 200+200 ≈ 1.5 days), I6b/c spinup + I7 recipe remain — engineering; I8 adoption + knobs are the PI's** | **Full custom forcing produced: 101 monthly NetCDFs, 2017-02 → 2025-06, QC-clean, CTSM-ready — a strict superset of pre-built v4** (v4's 2018–2024 span + 11 mo earlier + 6 later, RELEASE-2026 throughout). Reproduces v4 to machine precision (I4 PASS). Generated in **annual chunks on the burst QOS** (LOWMEM=TRUE broken; LOWMEM=FALSE ~3.5 GB/mo). Whole-record QC via new `neon_forcing_qc.py`. The one data gap — 2017 primary weighing-gauge precip outage — recovered from the secondary tipping bucket (validated r=0.96 / ~2% totals; `splice_2017_precip.py`, fqc=5, no source edit). Start 2017-02 (EC-anchored; earlier needs a declined source edit); end 2025-06 (released-only). Input-quality upgrade, independent of routing on/off. **2026-08-15: full-dataset CTSM ingestion smoke PASSED** (`osbs.swenson.neon-custom-smoke`) — custom stream drives CTSM end-to-end over the whole record; over the 2018-2019 v4 overlap, forcing-driven fields match v4 to ≤0.03% (correct ingestion), prognostic fields differ only by the expected spinup offset. Surfaced a production requirement: `dtlimit=-1` on both NEON streams (cycled spinup wraps the finite window → stock CDEPS bug without it). **2026-08-15 calibration: 5.54 min/yr; partial-year cycling (2017-02→2025-06) wraps Jun→Feb → a benign ~8-month seasonal jolt each cycle; the clean-boundary alternative is 2018–2024 = the v4 range (I8 sub-choice)** |

## Roadmap

```
1. Methodology validation        MERIT regression  ─ frozen (proven on published data)
2. Pipeline foundations          A, B, C, D        ─ Complete
3. Parameter set                 E, E.5, E.6       ─ Complete
4. Long spinup with lateral flow F + G Stage 1     ─ IN PROGRESS (lateral flow active under use_hillslope=.true.)
5. Stream-coupling (routing-on)  H                 ─ Track A done; Tracks B/C on hold; may not be pursued
6. Post-AD continuation          (optional)        ─ Future
7. Site-specific inputs (NEON)   I                 ─ IN PROGRESS (I1–I5 done + ingestion smoke PASSED 2026-08-15; I6a calibration done (200+200 ≈ 1.5 days), I6b/c spinup + I7 recipe remain — engineering; I8 adoption + knobs are the PI's; NEON soil/PFT future siblings)
```

Phases run sequentially within each track. F + G Stage 1 share the
osbs.swenson.spinup case as a single validation vehicle (originally
sequential per design; the 2026-04-25 PI direction folded the lake
column into the pipeline output, dissolving the F → G ordering).
Phase H Track A (mesh-mode workaround) is complete and ready if
needed, but Tracks B/C are on hold — the original scientific
motivation (activate lateral flow) collapsed when the 2026-05-19
audit showed lateral flow already runs under `use_hillslope=.true.`
**We are not assuming routing-on will be pursued.** Whether to do so
depends on what Phase F shows.

## What's running now

Nothing actively running. The Phase I **full custom forcing dataset is COMPLETE**
(2026-08-14): **101 monthly NetCDFs, 2017-02 → 2025-06**, at
`data/datm/neon_OSBS/custom/OSBS/atm/`, QC-clean and CTSM-ready (the 2017 primary-gauge
precip outage recovered from the secondary tipping bucket). Generated in annual chunks on
the burst QOS from the 11 GB raw archive at `/blue/gerber/earth_models/neon/raw/OSBS`. The
**full-dataset CTSM ingestion smoke PASSED 2026-08-15** (`osbs.swenson.neon-custom-smoke`):
the custom stream ingests and drives CTSM cleanly over the whole record and is v4-comparable
(`dtlimit=-1` required on both NEON streams — a cycled-forcing CDEPS requirement, not a data
issue). `osbs.swenson.spinup` 600-yr accelerated
AD spinup completed 2026-05-14 and was analyzed 2026-05-19 (plots at
`output/2026-05-19_osbs.swenson.spinup_timeseries/`). `osbs.swenson.post-ad`
hit an N-state crash on first attempt 2026-05-19, recovered by
2026-05-20, ran 200 yr successfully through 2026-05-21, and has been
idle since. PI is investigating the AD-spinup TAI absence and
bridge-zone anomaly; the production hillslope file is no longer frozen
(2026-07-15) — PI proceeding via soil-value adjustments.

## Methodology validation summary

MERIT regression test (`scripts/merit_validation/merit_regression.py`)
demonstrates the pysheds fork and pipeline math are correct against
Swenson's published data:

| Parameter | Correlation vs published |
|---|---|
| Height (HAND) | 0.9979 |
| Distance (DTND) | 0.9992 |
| Slope | 0.9839 |
| Aspect | 1.0000 (circular) |
| Width | 0.9919 |
| Area fraction | 0.9244 |

The regression test is the canonical "is the math still right" check
and is run after any pysheds fork modification. It validates the
geographic CRS code path; the OSBS pipeline exercises the same math
through the UTM code path.

## Cross-cutting concerns

- **CDEPS `dtlimit` crash when cycling a finite forcing window (NEON forcing).**
  A stock CDEPS bug (`dshr_strdata_mod.F90:1050` — a buggy `(a,i8)` error-print)
  hard-crashes the model when a cycling stream (`taxmode=cycle`) wraps past its
  last record and trips the default `dtlimit=1.5`. Hit by the 2026-08-15 ingestion
  smoke at the final timestep, and it **will** hit the cycled 600-yr spinup at
  every cycle boundary. Fix: **`dtlimit=-1` on both NEON streams** in
  `user_nl_datm_streams` (CDEPS's own `override_annual_cycle` escape hatch for
  streams not cycling on January boundaries; namelist-only, no rebuild). Required
  whenever the custom 2017-2025 forcing is adopted. See `phases/I-neon-forcing.md`
  `2026-08-15`.
- **CTSM Issue #1432: `grc%area = spval` in NUOPC single-point mode.**
  Open since 2021. Doesn't affect routing-off (Phase F is unaffected),
  but blocks routing-on because `nhill_per_landunit` ≈ 1e36.
  Mesh-mode workaround is the OSBS-side fix (Phase H); not pursuing
  upstream PR. See `phases/H-lateral-flow.md` for the full source
  trace and references.
- **`use_hillslope_routing` toggles the stream-side machinery, not
  the inter-column lateral flow.** Audited against CTSM 5.3.085 source
  on 2026-05-19. `PerchedLateralFlow` and `SubsurfaceLateralFlow`
  (`src/biogeophys/SoilHydrologyMod.F90:1703, :2086`) are dispatched
  from `HydrologyDrainageMod.F90:139,143` outside any routing gate,
  and the inter-column Darcy gradient computation
  (`SoilHydrologyMod.F90:2260-2263`) plus net-flow application
  (`:2434, :2449-2509`) run whenever `use_hillslope=.true.`. Routing-on
  adds: stream-channel geometry init
  (`HillslopeHydrologyMod.F90:378-507`), CTSM-internal stream-water
  state (`HillslopeStreamOutflow` + `HillslopeUpdateStreamWater`,
  called only at `HydrologyDrainageMod.F90:150-158`), a swap of the
  terminal-column boundary depth from `tdepth_grc` (MOSART) to
  internal `stream_water_volume / channel geometry`
  (`SoilHydrologyMod.F90:1822, 2265`), losing-stream outflow capping
  (`:2362`), `VOLUMETRIC_STREAMFLOW` history registration
  (`WaterFluxType.F90:525`), and lnd→rof streamflow export
  (`lnd2atmMod.F90:343`). Empirical confirmation: the spinup case
  shows negative QRUNOFF values at hillslope columns (signature of
  lateral inflow exceeding outflow) under routing-off. Phase F
  column-level differentiation is driven by both inter-column lateral
  flow AND per-column forcing, not forcing alone. Phase H adds the
  stream-coupling boundary condition at the chain bottom, not the
  lateral flow itself. See `phases/H-lateral-flow.md` Section 8
  for the full audit.

## References

| Doc | Path | Use |
|---|---|---|
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` | Methodology blueprint |
| Lake column CTSM audit | `docs/lake-column-ctsm-audit.md` | Canonical lake-column parameter values + CTSM source investigation |
| Phase docs | `phases/{A,B,C,D,E,E.5,F,G,H,I}-*.md` | Detailed records of each phase |
| Audit history | `audit/{240210,250223,260310,260512}-*/` | Historical audits + cleanup record |
| Production NetCDF | `output/osbs/2026-05-05_production/hillslopes_osbs_production_c260505.nc` | Current operative hillslope file |
| Operative case | `$CASES/osbs.swenson.spinup` | Current 600-yr accelerated AD spinup |
| NEON v4 forcing | `swenson/data/datm/neon_OSBS/v4/OSBS/` | Pre-built NCAR-NEON forcing (2018–2024, 84 files, 12 MB; Phase I I2) |
| Custom NEON forcing (Phase I I5) | `swenson/data/datm/neon_OSBS/custom/OSBS/atm/` | Full custom dataset — 101 monthly NetCDFs, 2017-02 → 2025-06, QC-clean, CTSM-ready (deliverable) |
| MERIT regression | `scripts/merit_validation/merit_regression.py` | Pysheds-fork validation test |

## Change log

- **2026-08-15** — Phase I **I6a calibration done; partial-year wrap discontinuity noted**. 10-yr AD calibration (`osbs.swenson.neon-custom-spinup`, cloned `--keepexe`, accelerated, lean annual output, cycled) completed clean at **5.54 min/sim-yr → 200+200 ≈ 36–37 hr compute (~1.5 days)**. Output sensible (TOTECOSYSC 1318→3195 and TOTVEGC building, TOTSOMC AD burn-down then stable, TWS filling, no NaN). **Discontinuity:** cycling the *partial-year* record (2017-02 → 2025-06) wraps **Jun→Feb** → an ~8-month seasonal jolt each cycle (~24 over a 200-yr spinup); benign, and `dtlimit=-1` held through the wrap. **Clean-cycle alternative = the v4 range:** the custom record's complete calendar years are exactly **2018–2024** (= v4's `2018-01→2024-12`, 84 mo), so a clean Dec→Jan wrap discards precisely the partial years (2017 spliced precip + 2025 H1) that are the custom set's value-add — over 2018–2024 the custom data is measured-bit-identical to v4 (I4), so a clean-boundary spinup is **effectively v4-equivalent**. New I8 sub-choice: clean cycle 2018–2024 vs full-record cycle 2017-2025 (calibration validated the full-record path). See `phases/I-neon-forcing.md` `2026-08-15 — I6a calibration`.
- **2026-08-15** — Phase I remaining work **re-scoped to engineering only** (PI direction): **I6** = mechanical AD→post-AD spinup (**200 + 200**, cycled) proving the custom dataset drives a full spinup; **I7** = a config recipe (every `xmlchange` + `user_nl_*` edit) for the PI. Experiment knobs (CO₂/aero/N-dep/use-case, MOSART-vs-SROF, present-day-vs-1850) + adoption (**I8**) are the PI's — we neither set nor reason about them. Cycle-vs-blend resolved: **PI chose cycle**. 200+200 (down from 600+200) is adequate — textbook CTSM-BGC is ~200+200 and OSBS converges faster than the arctic worst case. A **~1-hr calibration smoke** (first ~10 yr, lean monthly hist) sizes the 200+200 wall-clock before committing the multi-day resubmit chain. I6/I7/I8 rewritten. See `phases/I-neon-forcing.md` `2026-08-15 — Remaining work re-scoped`.
- **2026-08-15** — Phase I **full-dataset CTSM ingestion smoke PASSED** (the I6/I7 ingestion aspect). Wired the full 101-file custom stream into a cold-start `I1PtClm60Bgc` + hillslope case (`$CASES/osbs.swenson.neon-custom-smoke`, cloned from the I2.5 v4-smoke recipe — only the forcing + span changed) and ran the whole 2017-02 → 2025-06 record. First run crashed at the **final timestep** on a **stock CDEPS bug** (`dshr_strdata_mod.F90:1050` — a buggy error-print that fires when `taxmode=cycle` wraps past the last forcing record and trips the default `dtlimit=1.5`), **not a data problem**. Fixed with **`dtlimit=-1` on both NEON streams** (CDEPS's own `override_annual_cycle` escape hatch; namelist-only, no rebuild — verified `BUILD_COMPLETE` stayed TRUE); re-run **COMPLETED clean** (`case.run` + `st_archive` success; 101 h0a/h0i + 3070 h1a files). **v4 comparison over the 2018-2019 overlap**: forcing-driven fields essentially identical (TBOT −0.011%, FSDS −0.025%, FLDS −0.011%; precip −3.3% = the I4 gap-fill-window difference) → **correct ingestion**; prognostic fields differ by the **spinup offset only** (TWS +58%, H2OSOI +26%, ELAI +23%, GPP +6%) — custom pre-equilibrated from its 2017 wet-season head start (incl. Hurricane Irma), v4 still ramping from a dry cold-start (unmistakable in the H2OSOI panel); custom forcing is actually *drier* (−3.3%) yet the state is wetter, ruling out a wet-forcing bias. **Production implication: `dtlimit=-1` is required for the cycled spinup** (wraps at every cycle boundary → same crash without it). Ingestion validated end-to-end; **I6** (production compset integration — 1850 knobs + hydrology SourceMods) **and I8** (adoption) remain. New tool `scripts/neon_forcing/smoke_compare_v4.py`; plot `output/osbs/2026-08-15_neon-custom-smoke/` (gitignored). See `phases/I-neon-forcing.md` `2026-08-15`.
- **2026-08-14** — Phase I **I5 full-record dataset COMPLETE** — the custom NEON forcing is produced, QC-clean, and CTSM-ready: **101 monthly NetCDFs, 2017-02 → 2025-06** (`data/datm/neon_OSBS/custom/OSBS/atm/`), a strict superset of v4. Generated in **annual chunks on the burst QOS** (LOWMEM=TRUE is broken — crashes at `flow.api.clm.R:883`, upstream of the atm write; LOWMEM=FALSE ~3.5 GB/mo → 9 one-year runs at 64 GB each; two `gerber`-QOS attempts OOM'd). New whole-record QC (`scripts/neon_forcing/neon_forcing_qc.py`, the sniff-test analog of `neon_v4_regression.py`) caught the one defect: **2017 Jul–Dec precip missing** because NEON's **primary weighing gauge (DP1.00044) was physically down** (raw all-NA, finalQF=1 — likely why v4 starts at 2018). Recovered from the **secondary tipping bucket (DP1.00045)** via `scripts/neon_forcing/splice_2017_precip.py` (post-processing, **no source edit**; flags `PRECTmms_fqc=5`, backup first). Validated substitution: gauges agree **r=0.96 / ~2% on totals** (35 months); spliced values bit-match the raw at the same UTC timestamp; heaviest Sep-2017 rain lands on 2017-09-10 (**Hurricane Irma**), confirming alignment; only `PRECTmms`+`_fqc` changed. **Full record now passes QC (0 NaN).** New scientific decision locked (2017 precip source). Scripts + docs unpushed. Remaining: I6–I8 (CTSM ingestion smoke + PI adoption). See `phases/I-neon-forcing.md` `2026-08-14 — I5 full-record dataset COMPLETE`.
- **2026-08-14** — Phase I **I5 raw pull COMPLETE** (job `39397006`, 32.5 min, `--qos=gerber`). Full 2016-08 → 2025-06 archive at `/blue/gerber/earth_models/neon/raw/OSBS`: all 10 products at full span (**1063 zips, 0 short**; EC `00200` 101 mo from 2017-02, DP1 met 107 mo from 2016-08), zips integrity-checked. **Size 11 GB compressed — NOT 22.6 GB**: the manifest sums UNCOMPRESSED `/data/` files (~2× the zips); 11 GB is complete and matches the original estimate. Coverage: EC exists only from 2017-02, so 2016-08→2017-01 has met but no flux (a generation-time start-date decision). Archive ready for full-record forcing generation (`run_forcing.sh`, `LOWMEM=TRUE`). See `phases/I-neon-forcing.md` `2026-08-14 — I5`.
- **2026-08-14** — Phase I **I4 done — reproduce-v4 (2018) PASS**. Ran the offline pipeline over full 2018 and compared to pre-built v4 (12 months) via a new `scripts/neon_forcing/neon_v4_regression.py` (fqc-partitioned RMS/corr/bias + plots; the forcing analog of `merit_regression.py`). Headline: **both-measured RMS ≈ 0 for all 7 physical vars** — where both datasets carry a real measurement, our pipeline reproduces v4 to **machine precision** (measured data bit-identical); *all* divergence is in gap-filled timesteps (release + gap-fill-window differences), within/near the I2 reference band. Precip annual total +2.9%. Confirms pipeline fidelity to v4. Archive now holds all of 2018 (subset of the full pull); comparison outputs gitignored; script committed (hpg-esm-tools `20905da`). See `phases/I-neon-forcing.md` `2026-08-14`.
- **2026-08-12** — Phase I **I3 done — Option B pipeline implemented + smoke-validated**. The forked `flow.api.clm.R` (`uf-osbs`) runs fully offline against a shared raw archive; a Mar–Jun 2018 smoke produced 4 atm + 4 eval NetCDFs whose atm output **matches v4** (FLDS/FSDS/RH/PSRF identical; TBOT/WIND/PRECT within reprocessing scale). Archive `/blue/gerber/earth_models/neon/raw/OSBS`; output `swenson/data/datm/neon_OSBS/custom/OSBS/{atm,eval}/`. Testing forced **two edits beyond the plan** — **C** (copy zips to a per-session `tempdir()` because `stackByTable` *deletes* its inputs → would consume the shared archive) and **P2** (non-fatal REddyProc partition + NaN-fill CO₂ eval columns) — and surfaced the `FLDS_MDS`/`Rg`→`DP1.00023.001` duplicate, REddyProc's **≥90-day** flux minimum, an edit-T period-end boundary fix, and the headline finding: **OSBS EC CO₂ flux (NEE/FC) is absent at the NEON source** (all-NA, own QC flag all-bad — a genuine IRGA gap, *not* the script's mask; energy fluxes fine) → **no CO₂-flux validation is possible for OSBS 2018**. Full-pull size (`size_manifest.R`) = **22.6 GB** uncompressed (EC ≈58%; the zipped download is ~11 GB — see 2026-08-14 I5). Commits (unpushed): fork `2d30ebb`+`07786dd`, hpg-esm-tools `ba4d922`+`cac3be9`. See `phases/I-neon-forcing.md` `2026-08-12 (impl)`.
- **2026-08-01** — Phase I raw-data access **RESOLVED**: a free NEON API token (scope `rate:public`) lifts the `/data/` 403 from HiPerGator — verified from login11 (403 anonymous / 200 with token, 3×; `neonUtilities::zipsByProduct` downloaded a RELEASE-2026 month end-to-end in the `neon-forcing` env). The block was an anonymous-request gate on `/data/` from HPC IP ranges, not an IP ban / storage block / rate-limit. **Consequence:** the raw download **moves on-HPG** (compute node) — the off-HPG laptop + Globus path is retired, `docs/neon-raw-download-runbook.md` **removed**, `download_raw.R` repurposed for on-HPG use. A **compute-node test ladder** (real-tool smoke → exact-size manifest → EC probe → sustained/timing → full run) is defined before the full pull; the connectivity gate is unnecessary — compute nodes have outbound internet (confirmed 2026-08-01). See `phases/I-neon-forcing.md` Research §12.
- **2026-07-15** — PI decisions folded into Phase I: (1) **released data only** — the custom dataset targets 2016-08 → 2025-06 (RELEASE-2026 cut), excluding the provisional 2025-07 → 2026-06 tail (I5 / I8(c) resolved); (2) **production hillslope file no longer frozen** — PI proceeding with the existing file via soil-value adjustments (some concerns remain, in the PI's wheelhouse), so Phase I adoption (I8) is not freeze-blocked. See `phases/I-neon-forcing.md`.
- **2026-07-15** — Phase I task **I2.5 done — integration smoke test PASSED**. Built + ran a cold-start `1PT` + hillslope case (`$CASES/osbs.swenson.neon-v4-smoke`) on the v4 forcing via a `user_nl_datm_streams` `datafiles` override: 2-yr run completed, 26 hillslope columns active, forcing ingested (measured FLDS, CDEPS-converted RH) — §8/§9 confirmed end-to-end. Found 4 integration issues that carry to I6: force `MPILIB=openmpi`; the operative case's 6-file hydrology SourceMod set (needed for `spillheight`); surfdata-vs-NEON coordinate mismatch (~120 m → set `PTS_LAT/LON` to surfdata coords); walltime budget. See `phases/I-neon-forcing.md`.
- **2026-07-15** — Phase I task **I2 done** — fetched pre-built NCAR-NEON v4 forcing (84 files, 2018-01 → 2024-12, 12.08 MB) to `swenson/data/datm/neon_OSBS/v4/OSBS/` (`*.nc` gitignored; provenance README alongside). Integrity 84/84; v3 sanity check PASS (v4 is a full reprocessing, differences reprocessing-scale — RMS Δ TBOT 0.17 K, PSRF 9 Pa). First operational Phase I step. See `phases/I-neon-forcing.md`.
- **2026-07-15** — Phase I task list reworked to a single linear track (dropped the Track 1 / Track 2 split). One plan: fetch v4 → build + validate our own pipeline against it → produce the full 2016–2026 dataset → CTSM integration (downstream/PI-gated tail). v4 and custom data wire in identically (a `user_nl_datm_streams` `datafiles` override), so the two-track framing was moot. See `phases/I-neon-forcing.md`.
- **2026-07-15** — Phase I task I1 complete (claims verification + NEON↔CRUNCEP drop-in analysis). Re-verified all NEON product claims (live API + CTSM source + external web, 3 adversarial agents) and corrected `docs/neon-data-products.md`: wind/radiation start 2014-08 (not "2013–"), tipping-bucket 2016-08, **weighing gauge DP1.00044.001 IS installed** (was "not installed"), CO₂ bundle 2017-02, TBOT source is DP1.00003.001 (triple) not DP1.00002.001; RELEASE-2026 has a provisional tail after 2025-06. **NEON is NOT a drop-in for CRUNCEP** — differs in DATM_MODE/streams/humidity(RH↔QBOT)/calendar, but CDEPS handles all four via the `1PT` machinery (RH→shum converted internally); recommended structure is a compset change (`I1PtClm60Bgc`) + re-asserting our hillslope surfdata in `user_nl_clm`, keeping 1850 CO₂/chemistry knobs ("swap the weather, not the experiment"). 2018 TBOT anomaly (Issue #34) retired — verified sane in pre-built (269–307 K). Drop-in analysis added as phase Research note §8.
- **2026-07-15** — Phase I created (NEON atmospheric forcing). Pre-built NCAR-NEON forcing found available through 2024-12 (v4); the old "2018–2021" ceiling was a namelist cap (`NEONVERSION=v2`), not a data limit — the v3 set (2018 → 2024-06) is already on disk. Scoped pre-built-first (adopt/validate v4); custom NEON→DATM pipeline is a PI-gated contingency (no gap v4 can't fill has been identified). Registered as roadmap track 7. See `phases/I-neon-forcing.md`. Task list written but NOT started.
- **2026-06-27** — Doc reconciliation pass. Removed references to the never-persisted `output/2026-05-19_phase_F_analysis/REPORT.md` (closeout commit said it was gitignored; not on disk — verdicts inlined here and in `phases/F-validate-deploy.md` are the canonical record; plots backing them remain at `output/2026-05-19_osbs.swenson.spinup_timeseries/`). Corrected post-AD framing: initial 2026-05-19 N-state crash recovered by 2026-05-20, 200 yr ran successfully through 2026-05-21, idle since; reframed "Known blockers" → "Status of secondary tracks." Added PI-investigating callout under Open scientific questions plus an explicit production-hillslope-file freeze pending PI's findings.
- **2026-05-19** — Phase F routing-off track complete. 600-yr accelerated AD spinup converged cleanly (drift_50yr = 0.48%, well under 3%). Three verdicts: (1) **convergence PASS**, (2) **TAI signal ABSENT** — the expected "FZ wet, Upland dry, anoxia depression" pattern does not emerge; O_SCALAR is essentially 1.0 everywhere all years, (3) **lake column stable** — max 5.78 m at yr 107 (just under 6 m overflow), drained to 2.5 m by yr 600, no runaway. **Darcy drain (Phase H Option 5) NOT NEEDED**. Two open scientific questions for PI conversation: O_SCALAR not triggering despite saturated columns (TAI carbon signature missing); bridge-zone anomaly at chain indices 3-6 (HAND -3 to -1.5 m show deepest water tables — consequence of steep Darcy gradients over short distances). Phase F current-state row, Open questions section, and Phase F doc all updated. Apples-to-apples and osbs2/4-6 comparisons struck per user.
- **2026-05-19** — Phase H reframed as contingent. **We are not assuming a routing-on CTSM configuration will be pursued.** Track A (mesh-mode workaround) is complete and verified, but Tracks B/C are on hold and may never run — the original scientific motivation (activating inter-column lateral flow) collapsed when the routing-gate audit showed that flow is already active under `use_hillslope=.true.` Routing-on's remaining value is narrow (stream-coupling BC at chain bottom, internal stream-water ledger, `VOLUMETRIC_STREAMFLOW` diagnostic); whether that's worth the B1–B4 + C1–C4 cost depends on what Phase F shows. PI floated a vague idea of a regional Darcy drain on the lake column to address possible unbounded accumulation — no design exists; also contingent on Phase F. STATUS.md project context, Open questions, current-state table, roadmap, and Phase H doc all updated to reflect this framing.
- **2026-05-19** — Routing-gate source audit. CTSM source trace (`src/biogeophys/SoilHydrologyMod.F90`, `HillslopeHydrologyMod.F90`, `HydrologyDrainageMod.F90`) plus empirical check of the spinup case's h1a output corrects a load-bearing project-wide assumption: **column-to-column lateral subsurface flow runs under `use_hillslope=.true.`, not under `use_hillslope_routing=.true.`.** Routing toggles the stream-side state (channel geometry, internal `stream_water_volume`, Manning streamflow, lnd→rof export) and swaps the terminal-column boundary depth from MOSART's `tdepth_grc` to CTSM-internal stream state. Corrections applied to STATUS.md (this bullet + the cross-cutting concerns row), `phases/H-lateral-flow.md` Problem section + Section 7.5 table + new Section 8 + smoke-test reinterpretation, `phases/F-validate-deploy.md` Key Context corrective callout, `phases/G-ctsm-lake-representation.md` Stage-1 framing fix. Implication: Phase F is delivering more TAI physics than its doc claimed; Phase H's value is narrower (stream-side coupling, not the lateral-flow mechanism).
- **2026-05-12** — Phase H A3/A4 smoke test: paired test/control 5-yr cold-start cases built and run. **`grc%area = 90.006 km²` confirmed (not spval) — mesh-mode workaround verified.** Gridcell aggregates bit-identical between test and control; H2OSFC stays 0 everywhere (cold-start + Florida ET); but Year-5 deep-soil H2OSOI shows correct-signed TAI emergence (lake +7×10⁻⁴, bridge −1×10⁻⁴). Phase H Track A complete. [Note 2026-05-19: see routing audit above — the test-vs-control delta isolates the stream-coupling boundary condition, not "lateral flow on vs off."]
- **2026-05-12** — Phase H stream/lake routing-on interface analysis: Section 7 added (Swenson power-law stream params 5–10× too generous for OSBS; lake overflow threshold 6 m from `lake_hill_elev=−6m`; SourceMod Mechanism A is the actual release valve and stays active despite `spillheight=0`). B4 added to scientific decisions for PI consultation.
- **2026-05-13** — Phase H gridcell-area decision space updated: added rescale-only option (c′ ≈ 1 hr) and revised pipeline-rerun estimate (rectangular subset ≈ half-day; polygon-clip ≈ 1–2 days). Added mesh-mechanics primer to Phase H Section 5.
- **2026-05-12** — Phase H deep research pass: input data inventory, NEON product survey, scale analysis, mesh tooling, community precedent — 332 lines added to `phases/H-lateral-flow.md`.
- **2026-05-12** — STATUS.md restructured (30 KB → ~7 KB); CLAUDE.md updated to explicit index role; partially-superseded docs/* annotated.
- **2026-05-12** — `scripts/` hygiene cleanup + shared-module de-coupling (merit_validation and osbs each own a copy of `spatial_scale.py` + `hillslope_params.py`).
- **2026-05-11** — Phase H created (lateral subsurface flow); Phase G marked Complete (Stage 2 split to Phase H).
- **2026-05-11** — `osbs.swenson.spinup` (4-stream config) replaces `osbs5.swenson.spinup` as operative case after CTSM ntapes-mismatch prevented adding h2/h3 streams mid-spinup.
- **2026-05-06** — `osbs5.swenson.spinup` 100-yr AD spinup completed; 8 analysis plots generated.
- **2026-05-05** — Per-rep rescale lands; lake column `wtlunit` 98.7% → 12.3%.
- **2026-05-04** — Lake `hill_elev` locked at -6.0 m (PI suggestion).
- **2026-05-02** — Outlier cutoffs locked: Q01 = -6.34 m, Q99 = +17.46 m.
- **2026-04-30** — PI meeting: spillheight SourceMod retired; lake column becomes data-derived.
- **2026-04-25** — Lake column scope refined to NWI water only.
- **2026-04-14** — 16-bin hybrid HAND scheme adopted (superseded 2026-05-04 by 24-bin TAI scheme).
- **2026-04-09** — First production NetCDF generated; PI consultation on weir overflow plan → abandoned.
- **2026-03-30** — Production domain locked (R4-R12, C5-C14).
- **2026-03-23** — NEON slope/aspect adopted over pgrid Horn 1981.
- **2026-02-11** — Lc = 356 m locked.
