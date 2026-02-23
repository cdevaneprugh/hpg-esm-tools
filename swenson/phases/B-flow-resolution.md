# Phase B: Resolve Flow Routing Resolution

Status: Complete
Depends on: None
Blocks: Phase D

## Problem

The pipeline subsamples 1m LIDAR to 4m before flow routing (STATUS.md #2), discarding 93.75% of the data before the core hydrology computation. The entire justification for using 1m LIDAR is to capture fine-scale drainage in a low-relief wetlandscape — subsampling to 4m undermines that purpose.

The bottleneck is pysheds' `resolve_flats()`, which has poor scaling on large flat regions. One test at full resolution OOM'd at 64GB. Full resolution at 256GB was never tested. Neither was 2x subsampling.

**File:** `scripts/osbs/run_pipeline.py` line 1354

**Test plan:** `audit/240210-validation_and_initial_implementation/flow-routing-resolution.md`

## Tasks

### Script 1: Scalability (`scripts/phase_b/test_scalability.py`)

Tests whether `resolve_flats` completes on the full contiguous interior region (R4-R12, C5-C14, 90M pixels at 1m) at various memory allocations. Logs timing and peak RSS at each pipeline step.

- [x] Run at 64GB — **PASS: completed in 5.9 min, peak 29.2 GB** (2026-02-23)
- [~] Run at 128GB — unnecessary, 64GB sufficient
- [~] Run at 256GB (gerber-b QOS) — unnecessary
- [~] If 256GB fails: evaluate WhiteboxTools pre-conditioning — N/A

### Script 2: Resolution comparison (`scripts/phase_b/test_resolution_comparison.py`)

Runs full pipeline at 1m, 2m, 4m on the 5x5 tile block (R6-R10, C7-C11, 25M pixels). Computes all 6 hillslope parameters for all 16 columns at each resolution.

- [x] Run resolution comparison (64GB, 1hr) — **6 runs complete in 25 min** (2026-02-23)
- [x] Analyze results: parameter correlations, stream network density, timing

### Wrap-up

- [x] Document results with memory/time usage and parameter comparison
- [x] Update STATUS.md with resolution decision

## Deliverable

Determined processing resolution with scientific justification. Either: (a) full 1m works and we use it, (b) 2m is the practical limit with quantified parameter impact vs 1m, or (c) alternative DEM conditioning tool identified and tested.

## Scripts

```
scripts/phase_b/
├── test_scalability.py           # Scalability test
├── test_scalability.sh           # SLURM wrapper (override --mem at submit)
├── test_resolution_comparison.py # Resolution comparison
└── test_resolution_comparison.sh # SLURM wrapper (64GB, 1hr)
```

Output: `output/osbs/phase_b/`

### Running

```bash
# Scalability: incremental memory testing
sbatch --mem=64gb  scripts/phase_b/test_scalability.sh
sbatch --mem=128gb scripts/phase_b/test_scalability.sh
sbatch --mem=256gb --qos=gerber-b scripts/phase_b/test_scalability.sh

# Resolution comparison
sbatch scripts/phase_b/test_resolution_comparison.sh
```

## Log

### 2026-02-23: Scripts created

Created both test scripts and SLURM wrappers. Scripts are standalone — no imports from the existing pipeline (avoids coupling to broken code paths). Hillslope parameter functions inlined from `merit_regression.py` in the resolution comparison script (temporary duplication, Phase D will extract shared module).

Key design decisions:
- Scalability test: DEM conditioning + flow routing + HAND only. No hillslope params, FFT, or plots. Reads contiguous interior region via rasterio window from OSBS_interior.tif.
- Resolution comparison: Block averaging (not decimation) for subsampling. Uses np.gradient for slope/aspect with the N/S fix applied. Includes catchment-level aspect averaging and DTND tail removal.
- Both scripts use `/proc/self/status` VmHWM for peak RSS tracking.

### 2026-02-23: Scalability test — 64GB PASS

Fixed `test_scalability.py` DEM loading: replaced mosaic-based loading (read from `OSBS_interior.tif` via rasterio window) with on-the-fly tile merging via `rasterio.merge`. The mosaic file is gitignored and didn't exist on disk. New approach merges 90 tiles directly, matching the pattern used in `validate_lc_physical.py`.

**Result: Full 1m resolution completes at 64GB on the 90-tile contiguous sub-region.** The original OOM (job 23793378) was on the full interior mosaic (~189M pixels with 37.5% nodata) — over 2x larger, and the nodata tile gaps create additional artificial flats after depression filling. The contiguous sub-region (0% nodata) avoids the nodata-induced flats but still contains real flat surfaces (lakes, wetlands) that `resolve_flats` must process.

Job 25530167, node c0706a-s5. Full results: `output/osbs/phase_b/scalability_64gb.json`.

**Step timing (90M pixels, 9000x10000 at 1m):**

| Step | Seconds | Peak RSS (GB) |
|------|---------|---------------|
| load_dem (tile merge) | 2.6 | 0.9 |
| fill_pits | 15.4 | 21.1 |
| fill_depressions | 55.6 | 21.1 |
| resolve_flats | 136.7 | 22.7 |
| flowdir | 18.9 | 29.2 |
| accumulation | 57.4 | 29.2 |
| create_channel_mask | 17.9 | 29.2 |
| compute_hand | 45.1 | 29.2 |
| **Total** | **351.1** | **29.2** |

**Key observations:**
- `resolve_flats` is the single longest step (2m17s) but only peaked at 22.7 GB — well within the 64GB budget.
- The real memory ceiling is `flowdir`/`accumulation` at 29.2 GB.
- Total wall time under 6 minutes. No need for 128GB or 256GB runs.

**Hydrology summary (Lc=300m, A_thresh=45,000):**
- Stream pixels: 254,253 (0.28% of domain)
- HAND: mean 3.74m, max 27.31m, P95 13.32m
- DTND: mean 189.70m, max 1989.09m, P95 471.37m

**Implication for Phase B:** 1m is feasible at 64GB for the contiguous sub-region (90M pixels). `resolve_flats` cost is driven by total flat area from two sources: real water bodies (lakes, wetlands — present in any domain) and artificial flats from nodata fill (only in domains with tile gaps). The original 189M-pixel OOM had both; this test had only the former. Synthetic lake bottoms (see `docs/synthetic_lake_bottoms.md`) would reduce `resolve_flats` cost on any domain by replacing flat water surfaces with sloped bathymetry before conditioning. For the contiguous domain, 128GB/256GB runs are unnecessary. The remaining question is whether 1m produces meaningfully better hillslope parameters than 2m or 4m (Script 2).

### 2026-02-23: Resolution comparison — 1m is the clear choice

Updated `test_resolution_comparison.py`: replaced `np.gradient` slope/aspect workaround with pgrid's `slope_aspect("dem")` (Phase A Horn 1981 UTM-aware), and added dual-domain support (5x5 + full 90-tile region). 6 total runs: {5x5, full} x {1m, 2m, 4m}.

Job 25532242, node c0704a-s5. Full results: `output/osbs/phase_b/resolution_comparison/`.

**Timing and resources:**

| Domain | Res | Shape | Streams | Catchments | Time (s) | Peak RSS (GB) |
|--------|-----|-------|---------|------------|----------|---------------|
| 5x5 | 1m | 5000x5000 | 56,217 | 247 | 154 | 16.4 |
| 5x5 | 2m | 2500x2500 | 26,952 | 245 | 35 | 16.4 |
| 5x5 | 4m | 1250x1250 | 11,874 | 225 | 8 | 16.4 |
| full | 1m | 9000x10000 | 254,253 | 1,119 | 1027 | 58.0 |
| full | 2m | 4500x5000 | 121,717 | 1,055 | 220 | 58.0 |
| full | 4m | 2250x2500 | 58,932 | 1,027 | 51 | 58.0 |

**Parameter correlations (1m vs 2m / 1m vs 4m):**

| Parameter | 5x5: r(1m,2m) | 5x5: r(1m,4m) | full: r(1m,2m) | full: r(1m,4m) |
|-----------|---------------|---------------|----------------|----------------|
| Height | 0.9990 | 0.9974 | 0.9998 | 0.9998 |
| Distance | 0.9989 | 0.9980 | 0.9998 | 0.9992 |
| Area | 0.6363 | 0.5944 | 0.8222 | 0.6569 |
| Slope | 0.9688 | 0.9291 | 0.9492 | 0.9264 |
| Aspect | 0.6413 | 0.3457 | 0.9999 | 0.6770 |
| Width | 0.8962 | 0.9157 | 0.9877 | 0.9723 |

**Key findings:**

- **Height and distance** are resolution-insensitive (>0.99 everywhere). These are the parameters that drive lateral flow in CTSM.
- **Slope** systematically decreases with coarser resolution (smoothing artifact). 1m slopes are ~50% higher than 4m in the lowest HAND bin. Correlations still >0.93 but the bias is one-directional — subsampling always underestimates slope.
- **Area and aspect** are the weakest correlations. Area fractions shift because HAND bin boundaries and catchment delineation change with resolution. Aspect instability on the 5x5 domain (r=0.35 for 1m vs 4m) is a small-sample effect — the full domain (more catchments) is much more stable.
- **Full domain consistently produces higher correlations** than 5x5, confirming that more catchments = more robust statistics.
- **Computational cost is not a barrier.** Full domain at 1m: 17 min, 58 GB peak RSS. Well within 64GB / 1hr.

**Decision: Use 1m resolution.** No subsampling. The computational cost is negligible, slope is more accurate at 1m, and there is no parameter that improves with coarser resolution. The 4x subsampling in the original pipeline was a premature optimization for an OOM that doesn't occur on the contiguous domain.

