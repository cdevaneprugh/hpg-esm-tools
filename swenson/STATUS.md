# STATUS — Swenson Hillslope for OSBS

**Updated:** 2026-05-12

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
decomposition → CH4 production) once lateral subsurface flow is
activated.

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

## Open questions

### Awaiting PI consultation (Phase H)

These block meaningful routing-on validation. See
`phases/H-lateral-flow.md` Section 6 for full context.

- **B1. Gridcell area choice** for mesh-mode case. Three options worth
  presenting (per 2026-05-13 research notes in `phases/H-lateral-flow.md`):
  (a) 90 km² LIDAR domain — no rework; (c′) 36.81 km² NEON
  terrestrial-sampling boundary with `nhill_implicit` rescaled and
  hillslope statistics inherited from the 90 km² LIDAR mosaic —
  ~1 hour rework; (c) 36.81 km² NEON boundary with statistics
  re-derived from polygon-clipped LIDAR — ~1–2 days rework. Options
  ~2,700 km² and ~12,654 km² (Swenson conventions) recommended
  against. Central scientific decision; affects what gridcell
  aggregates physically represent and the magnitude of stream
  channel length and lateral flow rates.
- **B2. Hydraulic conductivity reasonableness.** Bin elevation
  differences in the 0–50cm TAI zone are small (~10 cm). Is the
  resulting Darcy gradient (Δh/L) physically meaningful for OSBS sandy
  soils?
- **B3. Validation framing.** No published single-point routing-on
  case exists. How do we judge whether observed behavior is "correct"
  without precedent?
- **B4. Stream-channel geometry and lake column overflow threshold.**
  Stream depth/width come from Swenson's MERIT-global power laws and
  are ~5–10× too generous for OSBS coastal-plain reality (~5 m × 0.5 m
  vs power-law 59 m × 1.52 m). Lake column overflow threshold is 6 m
  (from `lake_hill_elev = −6 m`), several times wider than real OSBS
  lake seasonal range (~1–2 m). Two pipeline tuning levers, ~1 hr each.
  Full analysis in `phases/H-lateral-flow.md` Section 7.

### Awaiting external clarification

- **Lee 2023 OSBS LIDAR vintage** — awaiting response from Cohen.
  Affects the framing of field-survey comparison in Phase E.5
  documentation. Non-blocking.

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
| F | Validate and deploy | **In progress** | osbs.swenson.spinup 600-yr accelerated AD spinup running |
| G | Submerged lake column | Complete | Stage 1 done; Stage 2 moved to Phase H |
| H | Lateral subsurface flow | **Track A complete** | Smoke test passed; awaiting PI decisions (B1–B4) before production spinup |

## Roadmap

```
1. Methodology validation        MERIT regression  ─ frozen (proven on published data)
2. Pipeline foundations          A, B, C, D        ─ Complete
3. Parameter set                 E, E.5, E.6       ─ Complete
4. Routing-off validation        F + G Stage 1     ─ IN PROGRESS (long spinup running)
5. Routing-on activation         H                 ─ Pending (PI decisions + mesh-mode)
6. Post-AD continuation          (optional)        ─ Future
```

Phases run sequentially within each track. F + G Stage 1 share the
osbs.swenson.spinup case as a single validation vehicle (originally
sequential per design; the 2026-04-25 PI direction folded the lake
column into the pipeline output, dissolving the F → G ordering).
Phase H decisions and engineering happen during F so the routing-on
case is ready when F completes.

## What's running now

`osbs.swenson.spinup`: SLURM job 32206376 (chunk 2 of 6) running yr
101 → yr 200 on c0700a-s2. ETA ~2026-05-12 21:50.

Full chain: 6 chunks × 100 yr each = 600 yr accelerated AD spinup,
expected to finish ~2026-05-15 evening at current throughput
(~6.4 yr/wall hr with 4 history streams).

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
| Phase docs | `phases/{A,B,C,D,E,E.5,F,G,H}-*.md` | Detailed records of each phase |
| Audit history | `audit/{240210,250223,260310,260512}-*/` | Historical audits + cleanup record |
| Production NetCDF | `output/osbs/2026-05-05_production/hillslopes_osbs_production_c260505.nc` | Current operative hillslope file |
| Operative case | `$CASES/osbs.swenson.spinup` | Current 600-yr accelerated AD spinup |
| MERIT regression | `scripts/merit_validation/merit_regression.py` | Pysheds-fork validation test |

## Change log

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
