# OSBS Pipeline Divergence Audit — 2026-03-16

## How to Use This Document

This is a combined plan + log for an exhaustive divergence audit of the OSBS hillslope pipeline. A fresh Claude session should:

1. Read this document first to understand scope and context
2. Read the files listed in "Files to Read" below
3. Work through each section sequentially, checking boxes and filling in divergence tables
4. Fill in the Summary (section 10) when all comparisons are complete

**Do not write code or make changes.** This is a read-only audit. The deliverable is this document with all tables filled in.

---

## Context

### What this project does

We are implementing Swenson & Lawrence (2025) representative hillslope methodology to generate custom hillslope parameters for the OSBS (Ordway-Swisher Biological Station) site using 1m NEON LIDAR data. The output is a NetCDF file that CTSM (Community Terrestrial Systems Model) reads to define 16 hillslope columns per gridcell (4 aspects x 4 elevation bins), enabling lateral subsurface flow and aspect-dependent radiation in the land model.

### Why divergences exist

Swenson's code (`Representative_Hillslopes/`) was written for global 90m MERIT DEM data in geographic CRS (EPSG:4326). Our pipeline processes 1m NEON LIDAR in UTM CRS (EPSG:32617). Key differences:

- **CRS**: Geographic (lat/lon, haversine distances) vs projected (meters, Euclidean distances)
- **Resolution**: 90m (~12K pixels per gridcell) vs 1m (90M pixels for our domain)
- **Data source**: Global MERIT DEM (continuous) vs mosaicked NEON tiles (gaps between tiles)
- **Environment**: NCAR HPC, older Python vs UF HiPerGator, Python 3.12 with current numpy/scipy
- **Scope**: Global processing (many gridcells) vs single-site (one gridcell, OSBS)

### The correct lens

Swenson's paper is the scientific reference. Swenson's code is a guide. It is NOT ground truth — there are too many divergences dictated by the 1m UTM data to treat it as such. The question for each divergence is: **does this make sense, is it justifiable, and why did we do it?**

We are using Swenson's pipeline to define **macro-scale** drainage properties (hillslope geometry, stream networks), not to resolve microtopography. The 1m data reveals micro-features (tree-throw mounds, animal burrows) that Swenson's methodology was never designed to handle. Where we diverge to handle these, the divergence should be documented and justified.

### Phase decisions baked into the pipeline

These are established decisions from earlier project phases. They explain most expected divergences:

- **Phase A (pysheds UTM CRS):** Modified `pgrid.py` to detect CRS and use Euclidean distance/spacing for UTM. Replaced haversine-based DTND with Euclidean DTND (still hydrologically linked via D8 trace). Replaced haversine-based Horn 1981 slope/aspect with uniform pixel spacing. Tested with 3 synthetic DEMs + geographic regression (1000+ pytest assertions, 100% mutation testing score).

- **Phase B (resolution):** Full 1m resolution, no subsampling. 90M pixels completes in ~22 min, peak 29 GB. Height and distance correlate >0.999 across 1m/2m/4m. Slope is systematically underestimated at coarser resolutions.

- **Phase C (characteristic length scale):** Lc = 356m (range 285-356m). The Laplacian spectrum at 1m has a micro-topographic artifact at ~8m from k^2 amplification. Real drainage peak emerges at ~300m when wavelengths < 20m are excluded via `min_wavelength=20`. Physical validation: P95 DTND/Lc = 1.17 (PASS), mean catchment/Lc^2 = 0.876 (PASS).

- **Phase D (pipeline rebuild):** Extracted shared modules (`hillslope_params.py`, `spatial_scale.py`, `dem_processing.py`). Replaced EDT-based DTND with pysheds hydrological DTND. Replaced `np.gradient` slope/aspect with pgrid Horn 1981. All equations verified against paper. Tiered SLURM wrappers. Production domain: R4C5-R12C14 (90 tiles, 0 nodata).

### Prior audit

`docs/osbs-pipeline-audit-260310.md` verified:
- All 12 pipeline steps match Figure 9 ordering
- All 7 key equations (Eq 6-12, 14, 16-17) implemented correctly
- 15 issues found (2 significant domain selection issues — now resolved; 6 moderate; 7 minor)
- OSBS vs MERIT pipeline consistency table (no unintentional divergences between our two pipelines)

**What the prior audit did NOT do:** Line-by-line implementation comparison of our functions against Swenson's originals. That is what this audit does.

### What NOT to audit

- **pysheds fork (`$PYSHEDS_FORK/pysheds/pgrid.py`)**: Covered by comprehensive pytest suite (`test_utm.py`, `test_split_valley.py`, `test_depression_basin.py`, `test_hillslope.py`). Just confirm tests pass.
- **Plotting functions**: Visual output only, no effect on parameters.
- **SLURM wrappers**: Shell scripts for job submission, reviewed in prior audit.

---

## Files to Read

### Our code (read ALL of these)

| File | Path | Lines |
|------|------|-------|
| hillslope_params.py | `$SWENSON/scripts/hillslope_params.py` | 481 |
| spatial_scale.py | `$SWENSON/scripts/spatial_scale.py` | 826 |
| dem_processing.py | `$SWENSON/scripts/dem_processing.py` | 122 |
| run_pipeline.py | `$SWENSON/scripts/osbs/run_pipeline.py` | 1543 |

### Swenson's code (read ALL of these)

| File | Path | Lines |
|------|------|-------|
| representative_hillslope.py | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/representative_hillslope.py` | 1754 |
| terrain_utils.py | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/terrain_utils.py` | 443 |
| geospatial_utils.py | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/geospatial_utils.py` | 305 |
| spatial_scale.py | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/spatial_scale.py` | 806 |

### Reference documents (read for context if needed)

| Document | Path |
|----------|------|
| Prior audit | `$SWENSON/docs/osbs-pipeline-audit-260310.md` |
| Project status | `$SWENSON/STATUS.md` |
| Paper summary | `$TOOLS/docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |

Where `$SWENSON` = `/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson` and `$TOOLS` = `/blue/gerber/cdevaneprugh/hpg-esm-tools`.

---

## Evaluation Framework

For every divergence found, fill in:

| Column | Description |
|--------|-------------|
| **What** | Concise description of the difference |
| **Why** | Category: CRS, Resolution, Bug fix, Python/numpy, Code org, or Unknown |
| **Justifiable** | Yes / No / Needs investigation |
| **Affects output** | Yes (changes parameter values) / No (same result) / Edge case only |

**Expected divergence categories:**

1. **CRS adaptation** — UTM code paths, `pixel_size` parameters, Euclidean vs haversine. Phase A work.
2. **Resolution adaptation** — `min_wavelength=20m`, larger edge windows, no subsampling. Phase B/C work.
3. **Bug fix** — N/S aspect swap, w^2→w^1 weighting, width calculation. Documented in STATUS.md.
4. **Python modernization** — `np.linalg.solve` vs `np.linalg.inv`, type hints, Path objects. Should not affect output.
5. **Code organization** — Functions extracted into modules, renamed, different parameter passing. Should not affect output.
6. **Defensive coding** — MemoryError handling, basin masking fallback, empty-bin guards. Different from Swenson but appropriate for 90M pixel domains.
7. **Unknown** — The ones we're hunting for. Flag these prominently.

---

## Source File Mapping

| Our file | Lines | Swenson counterpart(s) | Lines |
|----------|-------|----------------------|-------|
| `scripts/hillslope_params.py` | 481 | `terrain_utils.py` + `geospatial_utils.py` + `representative_hillslope.py` | 443 + 305 + 1754 |
| `scripts/spatial_scale.py` | 826 | `spatial_scale.py` | 806 |
| `scripts/dem_processing.py` | 122 | `geospatial_utils.py` | 305 |
| `scripts/osbs/run_pipeline.py` | 1543 | `representative_hillslope.py` | 1754 |
| `$PYSHEDS_FORK/pysheds/pgrid.py` | 4365 | (tested via pytest, not audited here) | — |

---

## 1. hillslope_params.py vs Swenson

### 1.1 get_aspect_mask()

- [x] Compare to Swenson inline code (`representative_hillslope.py:744-757`)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Standalone function vs Swenson's inline `if asp_ndx == 0` check; we detect wrapping via `lower > upper` generically | Code org | Yes — more general, works for any bin ordering | No |
| 2 | Returns boolean mask directly; Swenson uses `np.where(np.logical_or(...))` returning index arrays | Code org | Yes — callers can use boolean masks for indexing | No |
| 3 | Uses `\|` and `&` operators; Swenson uses `np.logical_or`/`np.logical_and` | Python | Yes — equivalent for numpy boolean arrays | No |

**Notes:** Functionally identical. The wrapping detection (`lower > upper`) is more robust than `asp_ndx == 0` because it works regardless of bin ordering. The standard aspect bins always have North at index 0 with (315, 45), so in practice neither approach fails.


### 1.2 compute_hand_bins() vs SpecifyHandBounds()

- [x] Compare to `terrain_utils.py:299-412` (fastsort branch only — Swenson calls with `BinMethod="fastsort"`)
- [x] Note: Swenson has 3 BinMethod options (fithand, explicitsum, fastsort). We only implement fastsort.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | We add `np.isfinite(hand)` check (line 76); Swenson only filters `fhand > 0` (tu:353) | Defensive — UTM can produce NaN HAND at nodata boundaries | Yes | Edge case only (UTM) |
| 2 | Swenson computes `std_hand = std_dev(fhand[fhand > 0])` (tu:307) but never uses it in fastsort; we omit it | Code org — dead code removal | Yes | No |
| 3 | Fallback for `hand_valid.size == 0` returns heuristic bounds; Swenson has no such guard (would crash) | Defensive | Yes | Edge case only |
| 4 | Fallback for `above_bin1.size == 0` returns heuristic bounds; Swenson would hit `int(-1)` index crash | Defensive | Yes | Edge case only |
| 5 | Swenson has `warning()` calls (tu:389-394, 399-401); we omit logging | Code org | Yes | No |
| 6 | Final length check `if (len(hand_bin_bounds) - 1) != 4` (tu:410-411) omitted; our structure guarantees length | Code org | Yes — structurally guaranteed | No |
| 7 | `min_aspect_fraction` is a parameter (default 0.01); Swenson hardcodes at tu:367 | Code org — parameterization | Yes | No |

**Notes:** Core fastsort logic faithfully reproduced. When `initial_q25 == bin1_max` exactly, both take the common-case branch (`>` is false). The high-relief branch aspect-loop sorting at tu:380 includes HAND=0 pixels; ours at line 103 does the same — both consistent.


### 1.3 fit_trapezoidal_width() vs calc_width_parameters() + _fit_polynomial()

- [x] Compare to `representative_hillslope.py:54-96` (calc_width_parameters, trapezoid branch)
- [x] Compare to `representative_hillslope.py:113-136` (_fit_polynomial)
- [x] Check: area normalization — we divide by n_hillslopes internally (line 207); Swenson divides at call site (`rh:768: farea[aind] / number_of_hillslopes[asp_ndx]`)
- [x] Check: w^1 weighting — our fix matches Swenson's actual _fit_polynomial (W = diag(weights), w^1). The bug was that an earlier version used w^2. Verify the fix is correct.
- [x] Check: `np.linalg.solve` vs `np.linalg.inv` + `np.dot` — different numerical method, should give same results
- [x] Check: fallback/error handling — we return a dict with heuristic values; Swenson raises ValueError

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Area normalization: we divide internally (hp:207); Swenson at call site (rh:768) | Code org — mathematically identical | Yes | No |
| 2 | `np.linalg.solve(GtWG, GtWy)` (hp:227) vs `np.linalg.inv(gtg) @ gtd` (rh:133-134) | Python — `solve` is more numerically stable; identical for 3x3 | Yes | No |
| 3 | W = diag(A_cumsum) with w^1 weighting matches Swenson's `_fit_polynomial` (rh:130-131) | N/A — verified match | N/A | No |
| 4 | Vandermonde: `np.column_stack([ones, d, d**2])` vs loop `g[:, n] = np.power(x, n)` | Python | Yes | No |
| 5 | Degenerate input: we return heuristic fallback (hp:186-191); Swenson raises ValueError (rh:61) | Defensive — silent fallback vs crash | Needs investigation — may mask data problems | Edge case only |
| 6 | Exception fallback: `except Exception` (hp:248) catches singular matrix etc.; Swenson would crash | Defensive | Needs investigation — same concern as #5 | Edge case only |
| 7 | **`max(trap_width, 1)` floor at hp:247; Swenson returns raw fitted width** | Defensive — prevents zero/negative widths reaching quadratic solver | Needs investigation — width < 1 may clip valid fits at very small scales; at OSBS 1m pixels, sub-meter width indicates bad fit | **Yes — clips any fitted width below 1** |
| 8 | `<=` comparison (hp:186 `np.max(dtnd) <= min_dtnd`) vs `<` (rh:60). When max(dtnd)==min_dtnd exactly, we take fallback; Swenson proceeds to near-degenerate fit | Bug fix — our behavior is more conservative | Yes | Edge case only |
| 9 | Convergent width adjustment (hp:242-245) matches Swenson (rh:87-94) identically | N/A | N/A | No |
| 10 | d=0 prepend: `np.concatenate` (hp:212-213) vs `[0] + list` (rh:73-75) | Python | Yes | No |

**Notes:** The core weighted-least-squares fit is mathematically equivalent. Two behavioral changes: (a) silent fallback instead of crashing, (b) `max(trap_width, 1)` floor. The floor never triggers for MERIT (widths are thousands of meters). For OSBS at 1m, sub-meter fitted widths almost certainly indicate a bad fit rather than a real physical width.


### 1.4 quadratic() vs quadratic()

- [x] Compare to `geospatial_utils.py:168-188`
- [x] Check: eps parameter for near-zero discriminants — our addition, Swenson doesn't have it

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | We compute `discriminant` once and reuse; Swenson recomputes `bk**2 - 4*ak*ck` three times | Python — avoids redundant computation | Yes | No |
| 2 | Error message: `f"Cannot solve quadratic: discriminant={discriminant:.2f}"` vs Swenson reports coefficients | Code org | Yes | No (exception path) |
| 3 | Swenson has `debug("quadratic roots ", dm_roots)` logging; we omit | Code org | Yes | No |
| 4 | Both have `eps=1e-6` default; both adjust `ck` identically for near-zero discriminants | N/A — match | N/A | No |

**Notes:** Functionally identical. Only cosmetic differences.


### 1.5 circular_mean_aspect() vs _calculate_hillslope_mean_aspect()

- [x] Compare to `terrain_utils.py:155-170`

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Standalone circular mean function; Swenson's includes drainage_id/hillslope iteration | Code org — math extracted, iteration in `catchment_mean_aspect()` | Yes | No |
| 2 | Math identical: `np.mean(np.sin(DTR * aspects))`, `arctan2(sin, cos) / DTR`, negative adjustment | N/A — match | N/A | No |

**Notes:** Mathematically identical. Just the extracted circular mean kernel.


### 1.6 catchment_mean_aspect() vs set_aspect_to_hillslope_mean_serial()

- [x] Compare to `terrain_utils.py:236-279`
- [x] This is a complex function with chunked processing. Compare chunk logic carefully.
- [x] Check: `np.where` usage differences (Swenson uses `np.logical_and.reduce`, we use `&`)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `np.isfinite(drainage_id) & (drainage_id > 0)` vs `np.logical_and(np.isfinite(...), ... > 0)` | Python — equivalent for numpy arrays | Yes | No |
| 2 | Empty `uid` guard at hp:349 returns zero array; Swenson serial version has no guard (parallel does at tu:184) | Defensive — prevents crash on empty input | Yes | Edge case only |
| 3 | Chunk boolean: `valid_aspect & (drainage_id.flat >= uid[n1]) & (drainage_id.flat < uid[n2])` vs `np.logical_and.reduce((l1, ..., ...))` | Python — `&` chaining equivalent to `logical_and.reduce` | Yes | No |
| 4 | Inner drainage matching: `cind[drainage_id.flat[cind] == did]` vs `cind[np.where(drainage_id.flat[cind] == did)[0]]` | Python — boolean indexing equivalent to np.where indexing | Yes | No |
| 5 | `hillslope_types[: hillslope_types.size - 1]` identical in both | N/A — match | N/A | No |
| 6 | Union with channels: `(hillslope.flat[dind] == 4) \| (hillslope.flat[dind] == ht)` vs `np.logical_or(...)` | Python — equivalent | Yes | No |
| 7 | Circular mean inlined (hp:387-395) rather than calling `circular_mean_aspect()`; Swenson also inlines | Code org — consistent with original | Yes | No |

**Notes:** Faithful line-by-line port of Swenson's serial version. Every Python idiom difference produces identical results. Chunksize default is 500 in both.


### 1.7 compute_pixel_areas() vs inline area computation

- [x] Compare to `representative_hillslope.py:1708-1715`
- [x] Note: this function is geographic-CRS only. UTM callers use `pixel_size^2` instead. Verify it's not called in the OSBS pipeline.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `np.tile(sin_theta.reshape(-1, 1), (1, ncols))` vs Swenson's `np.tile(np.sin(th), (im, 1)).T` | Python — both produce (nrows, ncols) array with sin(theta) along lat axis | Yes | No |
| 2 | `RE**2` vs `np.power(re, 2)` | Python — equivalent for scalar | Yes | No |
| 3 | Swenson allocates `farea = np.zeros(...)` then immediately overwrites — dead code; we omit | Code org — dead code removal | Yes | No |

**Notes:** Mathematically identical. Geographic CRS only — confirmed NOT called by the OSBS pipeline (UTM uses uniform `pixel_size^2`).


### 1.8 tail_index() vs TailIndex()

- [x] Compare to `terrain_utils.py:286-296`
- [x] Check: `np.std` (our ddof=0 default) vs Swenson's custom `std_dev()` function (tu:282-283). Both should be population std.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `np.std(dtnd_pos)` (ddof=0) vs Swenson's `std_dev()`: `sqrt(mean((x-mean(x))^2))` — numerically equivalent | Python — `np.std` may be more numerically stable internally | Yes | No |
| 2 | `std_dtnd == 0` guard (hp:466-467) returns all indices; Swenson has no guard — would get division by zero | Defensive — zero std means all DTND identical, no tail to remove | Yes | Edge case only |
| 3 | `np.sum(positive_hand) == 0` guard (hp:459-460); Swenson has no guard — empty array to `std_dev` would crash | Defensive — handles degenerate case | Yes | Edge case only |

**Notes:** Functionally identical with two defensive guards added.


---

## 2. spatial_scale.py vs Swenson spatial_scale.py

### 2.1 calc_gradient()

- [x] Compare to Swenson's `calc_gradient()` (geospatial_utils.py:129-162)
- [x] Check: UTM code path (pixel_size parameter) — new, no Swenson equivalent
- [x] Check: geographic path should be identical to Swenson

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | UTM code path added via `pixel_size` parameter (ss:81-88, 117-120) — uniform spacing in both axes | CRS | Yes | Yes (UTM mode uses different spacing) |
| 2 | CRS validation raises ValueError for invalid args; Swenson has no validation | Defensive | Yes | No |
| 3 | Return type: tuple vs Swenson's list | Python | Yes | No (both indexable by [0], [1]) |
| 4 | `else` for Horn1981 branch vs Swenson's second `if` (both equivalent given validation) | Code org | Yes | No |

**Notes:** Geographic code path (ss:121-130) is character-for-character equivalent to Swenson gu:155-162. Horn1981 smoothing loop (ss:96-114) matches Swenson gu:137-153 exactly.


### 2.2 smooth_2d_array()

- [x] Compare to Swenson's `smooth_2d_array()` (geospatial_utils.py:44-56)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Type annotations added | Python | Yes | No |

**Notes:** Functionally identical. Every line of computation matches exactly.


### 2.3 fit_planar_surface()

- [x] Compare to Swenson's `fit_planar_surface()` (geospatial_utils.py:59-75)
- [x] Check: optional coordinates (pixel indices for UTM) — new behavior

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Optional `x_coords`/`y_coords` with default to `np.arange` (ss:152-185); Swenson requires `elon`/`elat` | CRS — enables UTM mode with pixel indices | Yes | No (backward compatible — positional call with elon/elat still works) |
| 2 | Variable names: `x2d`/`y2d` vs `elon2d`/`elat2d` | Code org | Yes | No |

**Notes:** Backward compatible. Planar detrending residuals are identical regardless of whether coordinates are degrees, meters, or pixel indices — least-squares fit is invariant to affine coordinate transforms.


### 2.4 blend_edges()

- [x] Compare to Swenson's `blend_edges()` (geospatial_utils.py:77-112)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Type annotations added; minor comment changes | Python / Code org | Yes | No |

**Notes:** Functionally identical. Every computation line matches exactly.


### 2.5 _fit_polynomial()

- [x] Compare to Swenson's `_fit_polynomial()` (spatial_scale.py:37-61)
- [x] Note: this is the spatial_scale.py version used for FFT peak fitting, NOT the hillslope_params.py version used for trapezoidal fitting

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Swenson's `coefs = np.zeros(...)` initial allocation (sss:43) is dead code — reassigned at line 59; we omit | Code org | Yes | No |
| 2 | `weights is None` vs `type(weights) == type(None)` | Python — idiomatic; equivalent | Yes | No |
| 3 | **Swenson's `if y.size != weights.size` check (sss:52-53) dropped** | Regression — defensive check removed | **No — dropping a guard is a regression** | Edge case only (mismatch would cause unclear errors downstream) |
| 4 | f-string error message vs string concatenation | Python | Yes | No |

**Notes:** Divergence #3 is a minor regression — we dropped a defensive check that Swenson had. Worth restoring.


### 2.6 _synth_polynomial()

- [x] Compare to Swenson's `_synth_polynomial()` (spatial_scale.py:64-70)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Swenson stores `im = x.size`, `ncoefs = coefs.size` and uses `np.zeros((im))` tuple syntax; we use `x.size` directly | Python — `(im)` is not a tuple in Python, evaluates to scalar | Yes | No |

**Notes:** Functionally identical.


### 2.7 _bin_amplitude_spectrum()

- [x] Compare to Swenson's `_bin_amplitude_spectrum()` (spatial_scale.py:73-90)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `np.zeros(nlambda)` vs `np.zeros((nlambda))` | Python — parenthesized scalar same as bare scalar | Yes | No |

**Notes:** Functionally identical.


### 2.8 _gaussian_no_norm()

- [x] Compare to Swenson's `_gaussian_no_norm()` (spatial_scale.py:239-240)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Type annotations and docstring added | Python | Yes | No |

**Notes:** Formula on ss:289 matches Swenson sss:240 exactly.


### 2.9 _log_normal()

- [x] Compare to Swenson's `_log_normal()` (spatial_scale.py:93-102)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `np.zeros(x.size)` vs `np.zeros((x.size))` | Python — equivalent | Yes | No |

**Notes:** Functionally identical.


### 2.10 _fit_peak_gaussian()

- [x] Compare to Swenson's `_fit_peak_gaussian()` (spatial_scale.py:243-364)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `verbose: bool = False` parameter added but **never used** in body | Code org — dead parameter | Yes | No |
| 2 | Omit explicit `None` kwargs to `signal.find_peaks` (Swenson passes `threshold=None, distance=None, wlen=None, plateau_size=None`) | Python — all default to None | Yes | No |
| 3 | All `debug()` calls omitted | Code org | Yes | No |
| 4 | `useIndividualWidths = True` variable dropped; hardcode the True branch (always-True variable is dead code) | Code org — dead code removal | Yes | No |
| 5 | `max(minw, ...)` vs `np.max([minw, ...])` for scalar comparisons | Python | Yes | No |
| 6 | `gsigma = 1 * np.mean(...)` dead `1 *` multiplier dropped | Code org | Yes | No |
| 7 | `pcov_1gauss` unused from curve_fit; we use `_` | Python | Yes | No |

**Notes:** All divergences are cosmetic or dead-code removal. Computational path for `useIndividualWidths=True` (always taken) is identical.


### 2.11 _fit_peak_lognormal()

- [x] Compare to Swenson's `_fit_peak_lognormal()` (spatial_scale.py:105-236)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Same dead-parameter, dead-code, Python idiom changes as 2.10 | Code org / Python | Yes | No |
| 2 | **`np.log(center) if center > 0 else 0` guard (ss:412); Swenson has bare `mu = np.log(center)` (sss:177)** | Defensive — prevents `np.log(0) = -inf` or `np.log(<0) = nan` for edge peaks at smallest wavelength bin | Yes | Edge case only |

**Notes:** The `center > 0` guard (#2) is the only substantive addition.


### 2.12 _locate_peak()

- [x] Compare to Swenson's `_LocatePeak()` (spatial_scale.py:367-508)
- [x] Check: return dict keys — we added `psharp_ga`, `psharp_ln`, `gof_ga`, `gof_ln`, `tscore` for diagnostics

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `min_wavelength` parameter exposed to caller (default 1, same as Swenson's `minWavelength=1`) | Resolution — enables restricted-wavelength FFT | Yes | Yes (when caller passes min_wavelength > 1) |
| 2 | **Swenson's exponential fit (sss:398-399) dropped entirely — coefficients computed but never used** | Code org — dead code removal | Yes | No |
| 3 | **Swenson's `gof` array (sss:386-389) dropped — allocated, initialized to 1e6, but never populated or read** | Code org — dead code removal | Yes | No |
| 4 | `se` zero-division guard: `if den > 0 else 1e10` (ss:503); Swenson has bare `np.sqrt(num / den)` | Defensive | Yes | Edge case only |
| 5 | Return dict adds `psharp_ga`, `psharp_ln`, `gof_ga`, `gof_ln`, `tscore` diagnostic keys | Code org — additional diagnostics don't change computation | Yes | No |
| 6 | Swenson's `if model == "None": raise RuntimeError` (sss:487-488) omitted — unreachable in both | Code org | Yes | No |

**Notes:** Substantive divergences are (1) `min_wavelength` parameter exposed for Phase C restricted-wavelength FFT, and (4) zero-division guard. The exponential fit (#2) and `gof` array (#3) are provably dead code in Swenson.


### 2.13 identify_spatial_scale_laplacian_dem() vs IdentifySpatialScaleLaplacian()

- [x] Compare to Swenson's `IdentifySpatialScaleLaplacian()` (spatial_scale.py:511-681)
- [x] Compare to Swenson's `IdentifySpatialScaleLaplacianDEM()` (spatial_scale.py:683-806)
- [x] Check: UTM code path — pixel_size parameter, nodata fill with mean elevation, larger default blend/zero windows (50px vs 4/5px)
- [x] Check: min_wavelength parameter — Phase C addition, filters k^2 artifact. Swenson doesn't have this.
- [x] Check: return dict — we added `spatialScale_m` and diagnostic fields
- [x] This is the most heavily modified function. Compare carefully.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Full UTM code path via `pixel_size` parameter: CRS detection, nodata fill with mean elevation, uniform spacing | CRS | Yes | Yes (UTM mode) |
| 2 | CRS validation raises ValueError if both/neither CRS provided | Defensive | Yes | No |
| 3 | **Swenson's `IdentifySpatialScaleLaplacianDEM` references undefined `selon`/`selat`/`selev` in coastal path (sss:725-730) — broken code that crashes at runtime** | Bug in Swenson | N/A — we don't reproduce the bug | No (for our use cases) |
| 4 | Swenson's coastal gridcell expansion (sss:566-612) reads larger DEM region — we don't have this | Code org — UTM doesn't need it; geographic path never hits coastal cases in our usage | Acceptable | No (for our use cases) |
| 5 | **Blend-edges window: `50 if has_utm else 4` (ss:728) vs Swenson's triple-assignment `win=33%, win=7, win=4` where first two are dead code** | Resolution — 50px at 1m = 50m, comparable to 4px at ~90m = ~360m physically | Yes | Yes (UTM gets different window) |
| 6 | **Swenson's `IdentifySpatialScaleLaplacianDEM` does NOT zero edges — missing feature. Swenson's `IdentifySpatialScaleLaplacian` DOES zero edges (sss:641-643, n=5). We always offer zero-edges with default True** | Bug fix — Swenson's omission from DEM variant is likely an oversight per "causing bad fits" comment | Yes | Yes (geographic-mode from array-input differs from Swenson's DEM variant) |
| 7 | Configurable `zero_edges_n`: geographic default 5 matches Swenson; UTM default 50 | Resolution | Yes | Yes (UTM gets larger window) |
| 8 | Configurable `blend_edges_n`; Swenson hardcodes 4 | Resolution | Yes | No (geographic default matches) |
| 9 | `min_wavelength` passed to `_locate_peak` (ss:790-796); Swenson uses default `minWavelength=1` | Resolution — Phase C addition | Yes | Yes (restricts peak search) |
| 10 | Return dict adds `spatialScale_m`, `psharp_ga`, `psharp_ln`, `gof_ga`, `gof_ln`, `tscore` | Code org — diagnostics | Yes | No |
| 11 | File-reading entry point (`IdentifySpatialScaleLaplacian`) dropped — ours only takes arrays | Code org — I/O in pipeline, not spatial scale module | Yes | No |
| 12 | **Swenson prints "Planar surface removed" inside `doBlendEdges` block, NOT inside `detrendElevation` (sss:626, 746) — misplaced print; we print correctly in detrend block** | Bug fix — print only, no computation | Yes | No |

**Notes:** Most important divergences: (6) zero-edges bug fix, (1/5/7/9) UTM adaptations, and (3) documented broken code in Swenson's `IdentifySpatialScaleLaplacianDEM`.


---

## 3. dem_processing.py vs Swenson geospatial_utils.py

These functions were copied from `merit_regression.py` (which was copied from Swenson) for intentional decoupling. They should be near-identical to Swenson's originals.

### 3.1 _four_point_laplacian()

- [x] Compare to `geospatial_utils.py:_four_point_laplacian()` (gu:191-205)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `im = mask.shape[1]` dropped — Swenson assigns but never uses it (dead code) | Code org | Yes | No |
| 2 | Type annotation added | Python | Yes | No |

**Notes:** Functionally identical.


### 3.2 _expand_mask_buffer()

- [x] Compare to `geospatial_utils.py:_expand_mask_buffer()` (gu:235-253 via `_inside_indices_buffer` gu:208-232)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `_inside_indices_buffer()` logic inlined; Swenson has it as a separate function | Code org — no external callers | Yes | No |
| 2 | Swenson's `mask` kwarg in `_inside_indices_buffer(data, buf=1, mask=None)` dropped — never passed by any caller (defaults to empty array) | Code org — dead parameter | Yes | No |

**Notes:** Functionally identical. The inlining is clean.


### 3.3 erode_dilate_mask()

- [x] Compare to `geospatial_utils.py:erode_dilate_mask()` (gu:255-261)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `(x > 0) & (mask > 0)` vs `np.logical_and(x > 0, mask > 0)` | Python | Yes | No |

**Notes:** Functionally identical.


### 3.4 identify_basins()

- [x] Compare to `geospatial_utils.py:identify_basins()` (gu:263-296)
- [x] Note: at 1m, this returns zero detections (no single elevation value exceeds 25% histogram frequency). Effectively dead code for OSBS but still runs.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `nodata is not None` vs `type(nodata) != type(None)` | Python | Yes | No |
| 2 | Swenson's inner `for i in ind:` loop (gu:290) with `eps` re-computation is dead code inside niter loop — line 294 `imask[_four_point_laplacian(1 - imask) >= 3] = 0` doesn't use `i` or `eps`. We simplify to run it once per niter iteration | Code org — dead code simplification; running idempotent operation once vs `len(ind)` times | Yes | No |

**Notes:** The inner loop simplification produces identical results because `_four_point_laplacian(1 - imask) >= 3` is idempotent — after the first execution, subsequent runs within the same niter iteration are no-ops.


### 3.5 identify_open_water()

- [x] Compare to `geospatial_utils.py:identify_open_water()` (gu:298-305)
- [x] Note: at 1m, this returns zero detections (LIDAR noise on water produces slopes >> 1e-4 threshold). Effectively dead code for OSBS.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Return type: tuple vs Swenson's list | Python | Yes — callers unpack identically | No |

**Notes:** Functionally identical.


---

## 4. run_pipeline.py main loop vs representative_hillslope.py

The pipeline main loop (`run_pipeline.py:main()`) corresponds to Swenson's `CalcGeoparamsGridcell()` and `CalcRepresentativeHillslopeForm()` in `representative_hillslope.py`. Swenson processes multiple gridcells in a loop; we process one site.

### 4.1 DEM I/O and mosaic creation

- [x] Compare mosaic/tile handling to Swenson's DEM I/O (`dem_io.py`)
- [x] Note: tile grid system, NEON format, mosaic stitching — entirely new code. Swenson reads pre-existing global DEMs. No Swenson equivalent to compare against.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Entirely new code: tile selection, mosaic stitching, connected component extraction, nodata edge trimming — no Swenson equivalent | CRS / data source | Yes — required for NEON tile data | N/A |

**Notes:** No comparison possible. Swenson's DEM I/O reads pre-tiled global rasters in geographic CRS. Our pipeline stitches individual NEON 1m GeoTIFFs and extracts the largest connected component.


### 4.2 FFT call and Lc computation

- [x] Compare FFT invocation (`run_pipeline.py:820-835`) to Swenson's call (~`representative_hillslope.py:1580`)
- [x] Check: `min_wavelength=20m` — Phase C addition, not in Swenson
- [x] Check: `blend_edges_n=50, zero_edges_n=50` — larger than Swenson's defaults (4/5) because 1m pixels
- [x] Check: `pixel_size=PIXEL_SIZE` — triggers UTM mode
- [x] Check: `accum_threshold = int(0.5 * lc_px**2)` — uses pixel-unit Lc (at 1m, cells = m^2)

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `min_wavelength=20` (rp:823); Swenson passes default 1 | Resolution — Phase C filters k^2 artifact at 1m | Yes | Yes — shifts Lc from ~8m to ~300m |
| 2 | `blend_edges_n=50, zero_edges_n=50` (rp:824-825) vs Swenson's 4/5 | Resolution — 50px at 1m = 50m, comparable to 4px at 90m = 360m | Yes — Phase C showed Lc insensitive | No |
| 3 | `pixel_size=PIXEL_SIZE` (rp:822) triggers UTM mode | CRS | Yes | Yes |
| 4 | `accum_threshold = int(0.5 * lc_px**2)` (rp:835) vs Swenson `accum_thresh = 0.5 * (spatialScale**2)` (rh:451) — identical formula in pixel units | N/A — match | N/A | No |

**Notes:** `nlambda` defaults match (30 in both).


### 4.3 DEM conditioning chain

- [x] Compare conditioning sequence to `representative_hillslope.py:~1600-1680`
- [x] Check: order — basin detection → fill_pits → fill_depressions → slope_aspect → open_water → basin lowering → resolve_flats
- [x] Check: resolve_flats fallback — we fall back to flooded DEM; Swenson falls back to raw DEM (`rh:405-408`). At 1m, raw DEM has millions of LIDAR noise pits that would fragment the stream network. Flooded DEM preserves connected drainage.
- [x] Check: slope_aspect computed on ORIGINAL DEM (before conditioning) — prevents false gradients from fill artifacts. Does Swenson do the same?
- [x] Check: basin_mask lowering of flooded DEM by 0.1m (line 968) — forces flow through detected water bodies
- [x] Check: re-masking basins after flowdir (lines 999-1007) — sets inflated DEM to NaN at basin pixels
- [x] Check: basin_boundary forcing into stream network (line 1014) — forces water body boundaries to be stream cells

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | **resolve_flats fallback**: we fall back to flooded DEM (rp:982-992); Swenson falls back to raw DEM (rh:1567) | Defensive — at 1m, raw DEM has millions of noise pits that fragment drainage; flooded DEM preserves connectivity | Yes | Yes — affects flow routing quality when resolve_flats fails |
| 2 | slope_aspect on original DEM: both call on original unconditioned DEM (rp:955, rh:1547/1593) | N/A — match | N/A | No |
| 3 | We call slope_aspect ONCE (rp:955) and reuse; Swenson calls TWICE (rh:1547, 1593) on same input — redundant | Code org — optimization | Yes | No |
| 4 | Basin lowering: `flooded_arr[basin_mask > 0] -= 0.1` (rp:968) matches Swenson (rh:1549) | N/A — match | N/A | No |
| 5 | Re-masking: we set `inflated_arr[basin_mask > 0] = np.nan` (rp:1000); Swenson also nullifies flooded DEM (rh:1582-1583) — flooded DEM unused after this in both | Code org — functionally equivalent | Yes | No |

**Notes:** Conditioning chain order matches Swenson's exactly: basin detection -> fill_pits -> fill_depressions -> slope_aspect (original) -> open water -> basin lowering -> resolve_flats -> flowdir -> re-mask -> accumulation -> force basin boundaries.


### 4.4 Flow routing and accumulation

- [x] Compare to `representative_hillslope.py:~1680-1700`
- [x] Check: A_thresh safety valve (lines 1025-1028) — divides max_acc by 100 if max_acc < accum_threshold. Does Swenson have an equivalent guard?
- [x] Check: MemoryError handling — we catch and continue with `branches=None`; Swenson calls `sys.exit(1)`

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | A_thresh safety valve: both divide `max_acc / 100` when `max_acc < accum_threshold` (rp:1025-1028, rh:1597-1601) | N/A — match | N/A | No |
| 2 | MemoryError in `extract_river_network`: we continue with `branches = None` (rp:1048-1056); Swenson returns -1 and skips gridcell (rh:1612-1614) | Defensive — single-site pipeline can't skip | Yes | Only if MemoryError occurs |
| 3 | MemoryError in `river_network_length_and_slope`: we continue with `net_stats = None` (rp:1070-1077); Swenson returns -1 (rh:1662-1664) | Defensive — same reasoning | Yes | Only if MemoryError occurs |
| 4 | Basin boundary forcing: identical `acc_arr[basin_boundary > 0] = accum_threshold + 1` (rp:1014, rh:1590) | N/A — match | N/A | No |

**Notes:** No discrepancies in flow routing logic. Both use same `flowdir` -> `accumulation` sequence with `dirmap = (64, 128, 1, 2, 4, 8, 16, 32)`.


### 4.5 HAND/DTND computation

- [x] Compare compute_hand call to Swenson's usage
- [x] Note: pysheds fork handles CRS internally (tested via pytest suite with 1000+ assertions)
- [x] Check: what arguments does Swenson pass vs what we pass?

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | String-based API: `grid.compute_hand("fdir", "inflated", ...)` (rp:1090-1097) vs Swenson passes arrays directly (rh:1685) | Code org — fork supports both | No algorithmic difference | No |
| 2 | CRS: pysheds fork detects UTM and uses Euclidean distance for DTND (Phase A) vs haversine in Swenson's original | CRS — essential fix for STATUS.md #1 | Yes | Yes — DTND now correct for UTM |

**Notes:** The compute_hand call is the single most important fix in the pipeline — replaces EDT-based DTND with hydrologically-linked Euclidean DTND.


### 4.6 Hillslope classification and catchment aspect averaging

- [x] Compare to `representative_hillslope.py:1692, 1725-1751`
- [x] Check: compute_hillslope call arguments
- [x] Check: catchment_mean_aspect call — we pass `np.array(grid.drainage_id)`, `aspect`, `np.array(grid.hillslope)`. Does Swenson pass the same?

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `catchment_mean_aspect()` (hp:310-398) is serial implementation matching Swenson's `set_aspect_to_hillslope_mean_serial` (tu:236-279) | Code org | Yes | No |
| 2 | Parallel version (`set_aspect_to_hillslope_mean_parallel`) not implemented; serial sufficient at OSBS scale | Code org — performance, not correctness | Yes | No |

**Notes:** `chunksize` default is 500 in both.


### 4.7 Filtering: flood filter, DTND clip, valid mask, basin masking, tail removal

- [x] Compare flood filter (lines 1148-1162) to `representative_hillslope.py:680-697`
- [x] Note: flood filter is dead code at 1m (n_flooded=0, identify_open_water finds nothing). Compare logic anyway — it will matter when synthetic lake bottoms enables water detection.
- [x] Compare DTND minimum clip (line 1175-1176: `smallest_dtnd = 1.0`) to `rh:699-700`
- [x] Compare valid mask construction (line 1179: `valid = np.isfinite(hand_flat) & valid_mask_flat`). Swenson uses `np.isfinite(hand_flat)` only — we add `valid_mask_flat` for nodata gaps in mosaics.
- [x] Compare basin masking logic (lines 1182-1192). Swenson raises ValueError when region too flat; we warn and continue.
- [x] Compare tail removal — we call `tail_index()` on filtered arrays; Swenson calls `TailIndex()` similarly.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Flood filter structurally equivalent (rp:1148-1162 vs rh:678-697); dead code at 1m | N/A — dead code for OSBS | N/A | No |
| 2 | DTND clip: `smallest_dtnd = 1.0` (rp:1175) matches Swenson (rh:700) | N/A — match | N/A | No |
| 3 | Valid mask: we add `valid_mask_flat` for nodata gaps; Swenson uses `np.isfinite(hand_flat)` only | Defensive — necessary for mosaic handling | Yes | No (additive filter) |
| 4 | Basin masking: we warn and continue (rp:1184-1192); Swenson raises ValueError / `continue` to skip subregion (rh:575-593) | Defensive — single-site can't skip | Yes | Only if site is almost entirely basin |
| 5 | **Tail removal order: we apply AFTER basin masking and valid filtering (rp:1195-1201); Swenson applies BEFORE flood filtering (rh:666-676)** | Unknown — different pixel population fed to exponential fit could shift the 5% cutoff | Needs investigation | Potentially yes — minor effect |

**Notes:** The tail removal order difference (#5) is the most notable finding. Our order (valid mask -> basin mask -> tail removal) feeds a cleaner pixel population to the exponential fit. Swenson's order (tail removal -> flood filter -> DTND clip) includes basin pixels in the fit. The effect should be minor since tail removal targets extreme high DTND values.


### 4.8 Per-aspect parameter computation loop

- [x] Compare to `representative_hillslope.py:736-880`
- [x] Check: n_hillslopes counting — we use `len(np.unique(drainage_id_flat[asp_indices]))` (line 1272); Swenson uses `np.unique(fdid[aind]).size` (rh:761). Should be equivalent.
- [x] Check: trapezoidal fit call — we pass `min_dtnd=PIXEL_SIZE` (1m); Swenson passes `mindtnd=ares` (~90m). Both set min_dtnd to the DEM resolution.
- [x] Check: area fraction computation — we compute `bin_raw_areas[i] / total_raw` (lines 1322-1326); Swenson computes `np.sum(farea[cind]) / np.sum(farea[aind])` (rh:836). Should be equivalent.
- [x] Check: fitted area scaling — we compute `trap_area * area_fraction` (line 1327); Swenson does the same (rh:837).
- [x] Check: width at bin edge — we solve `quadratic([trap_slope, trap_width, -da_width])` then `width = trap_width + 2 * trap_slope * le` (lines 1373-1374); compare to Swenson rh:840-844.
- [x] Check: distance at bin midpoint — we compute `da_dist = sum(fitted_areas[:h_idx+1]) - fitted_areas[h_idx] / 2` (line 1382); compare to Swenson rh:848-858. This implements Eq 17.
- [x] Check: HAND <= 0 bin skip (lines 1351-1360) — compare to Swenson rh:819-821.
- [x] Check: empty bin handling — we append zero_element; does Swenson skip or zero-fill?
- [x] Check: median DTND (line 1366) — we compute it but then OVERRIDE it with the trapezoid-derived distance (lines 1383-1389). Swenson does the same (rh:824-828 computes median, rh:858 overwrites with quadratic solution). Verify the fallback to median_dtnd when quadratic fails.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | `n_hillslopes`: `len(np.unique(drainage_id_flat[asp_indices]))` (rp:1272) vs `np.unique(fdid[aind]).size` (rh:761) — identical logic | N/A — match | N/A | No |
| 2 | **`min_dtnd=PIXEL_SIZE` (=1m, rp:1286) vs `mindtnd=ares` (~90m, rh:770)** | CRS — both use DEM pixel resolution. PIXEL_SIZE is the correct UTM analog of `ares` | Yes | Yes — changes trapezoidal fit binning grid |
| 3 | Area fraction: `bin_raw_areas[i] / total_raw` (rp:1322-1326) vs `np.sum(farea[cind]) / np.sum(farea[aind])` (rh:836) — identical logic | N/A — match | N/A | No |
| 4 | Fitted area scaling: `trap_area * area_fraction` identical in both | N/A — match | N/A | No |
| 5 | Width at bin edge: `quadratic([trap_slope, trap_width, -da_width])`, `width = trap_width + 2 * trap_slope * le` (rp:1370-1374) matches Swenson (rh:840-844) | N/A — match | N/A | No |
| 6 | Distance at bin midpoint (Eq 17): `da_dist = sum(fitted_areas[:h_idx+1]) - fitted_areas[h_idx] / 2` (rp:1382) matches Swenson (rh:848-859) | N/A — match | N/A | No |
| 7 | HAND <= 0 bin skip: `np.mean(hand_flat[bin_indices]) <= 0` (rp:1351) matches Swenson (rh:819) | N/A — match | N/A | No |
| 8 | **Empty bin handling: we append placeholder with non-zero height/aspect (rp:1334-1348); Swenson leaves pre-initialized zeros (rh:384-390). Swenson's compression step (rh:972) removes zero columns; we have no compression step** | Defensive — structural difference | Needs investigation — see notes | Yes — empty bins with non-zero height/aspect appear in NetCDF |
| 9 | Median DTND computed then overridden by trapezoidal distance in both (rp:1366/1383-1389, rh:824-828/858-859); fallback to median when quadratic fails | N/A — match | N/A | No |
| 10 | **Missing minimum-hillslope-count check: Swenson discards all data if < 3 of 4 aspects populated (rh:1019-1036); we have no such check** | Defensive — structural difference | Low risk at OSBS (all 4 aspects always populated at 90M pixels) | Edge case only |

**Notes on #8:** Swenson's pipeline has a compression step (rh:971-1012) that removes empty bins and compacts arrays. Our pipeline always writes 16 columns. For OSBS (flat terrain, uniform aspect distribution), empty bins are unlikely, but the structural difference exists. If any bin were empty, our NetCDF would contain a column with `height=midpoint, area=0, width=0` where Swenson's would have that column removed.


### 4.9 Stream parameters

- [x] Compare to `representative_hillslope.py:~1100-1120`
- [x] Check: stream slope — we use `net_stats["slope"]` from `river_network_length_and_slope()`; Swenson averages per-subregion network slopes.
- [x] Check: depth/width power law — `0.001 * total_area_m2**0.4` and `0.001 * total_area_m2**0.6` (lines 1438-1439). `total_area_m2` is the sum of 16 fitted element areas (~235K m^2), which is per-hillslope area, NOT drainage area. Labeled INTERIM, deferred to Phase E.
- [x] Check: what does Swenson use as the area input to the power law?

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | **Stream slope: our `river_network_length_and_slope()` (pgrid.py:3170-3315 via extract_profiles) vs Swenson's `calc_network_length` (tu:30-144 via manual D8 walk/haversine)**. Fork also filters `elevation_difference > 0` | CRS — fork version is CRS-aware (Euclidean for UTM) | Yes | Yes — algorithmic differences in reach tracing |
| 2 | Depth/width power law: identical `0.001 * area**0.4` / `0.001 * area**0.6` coefficients. Area input: our `total_area_m2 = sum(16 fitted areas)` vs Swenson `uharea = np.sum(area[:])` — both sum all column areas | N/A — match in formula | N/A | No |
| 3 | We use single-domain call; Swenson averages over subregions | Code org — we process one site | Yes | No |

**Notes:** Swenson's latest code also uses `river_network_length_and_slope()` (rh:1656-1661) — `calc_network_length` in terrain_utils.py is the older implementation. So both use the same pgrid function; the difference is CRS handling (haversine vs Euclidean).


### 4.10 NetCDF output

- [x] Compare `write_hillslope_netcdf()` (lines 490-726) to Swenson's NetCDF writing
- [x] Check: variable names, dimensions, units, dtypes against Swenson reference file
- [x] Check: aspect conversion — degrees → radians at line 549 (`elem["aspect"] * np.pi / 180`)
- [x] Check: downhill_column_index logic (lines 556-559) — bin_idx==0 gets -9999, others point to previous column
- [x] Check: pct_hillslope computation (lines 561-568) — per-aspect area / total area * 100
- [x] Check: bedrock_depth = 1e6 placeholder (line 570) vs Swenson's 0. Both are placeholders.
- [x] Check: AREA field — we write trapezoidal sum; Swenson writes gridcell area (post-hoc via ncks). CTSM doesn't read this field.

**Divergences:**

| # | What | Why | Justifiable | Affects output |
|---|------|-----|-------------|----------------|
| 1 | Variable names and dimensions: `lsmlat`, `lsmlon`, `nhillslope`, `nmaxhillcol` — match | N/A | N/A | No |
| 2 | Aspect degrees → radians: `aspect * np.pi / 180` (rp:549) vs `aspect * dtr` (rh:1351) — identical | N/A — match | N/A | No |
| 3 | **`downhill_column_index`: sequential 0-indexed (rp:556-559, bin 0 = -9999, others = i) assumes all bins populated. Swenson uses running `col_cnt` that skips empty bins (rh:915-920), with post-pass correction (rh:931-940)** | Defensive — structural difference, consequence of no compression step | Correct when all 16 bins populated (expected at OSBS); incorrect if bins empty | Potentially yes |
| 4 | `pct_hillslope`: identical `100 * aspect_area / total_area` in both | N/A — match | N/A | No |
| 5 | **`bedrock_depth = 1e6` (rp:570) vs Swenson `0` (rh:390)** | Both placeholders — STATUS.md #7 | Needs investigation — CTSM may use for soil column depth | Yes — different values in NetCDF |
| 6 | **`AREA` variable written (rp:610-613); Swenson does not write it** | Code org — additional variable | Verify whether CTSM reads this | Possibly |
| 7 | 1D coordinate variables: `lsmlat`/`lsmlon` (ours) vs `latitude`/`longitude` (Swenson) — 2D `LONGXY`/`LATIXY` match | Code org | Yes — CTSM reads 2D fields for matching | No |
| 8 | Swenson writes `chunk_mask` diagnostic (rh:1308-1317); we omit — used for subregion processing | Code org — not needed for single site | Yes | No |
| 9 | `column_index`: always 1-16 sequential (rp:553); Swenson may be 1-N after compression | Code org — self-consistent within each pipeline | Yes — as long as `nhillcolumns` correct | No |
| 10 | Additional global attributes: `pixel_size_m`, `characteristic_length_m`, `accumulation_threshold`, `fft_min_wavelength_m` — informational metadata | Code org | Yes | No |

**Notes:** The structurally different handling of empty bins (#3) and the bedrock_depth placeholder (#5) are the main concerns. Both are low-risk for OSBS (all bins populated, bedrock_depth behavior in CTSM needs verification).


---

## 5. Functions Swenson has that we don't use

- [x] Review each for whether we should be using it

| Swenson function | File | Why we skip it | Risk of skipping |
|---|---|---|---|
| `calc_width_parameters(..., form="annular")` | `rh:98+` | We use trapezoid only, matching Swenson's default and published global dataset. Annular models a ring-sector geometry useful for very unequal aspect distributions. | None — OSBS has near-uniform aspects; trapezoidal is appropriate |
| `set_aspect_to_hillslope_mean_parallel()` | `tu:173-233` | Serial version sufficient at OSBS scale (~10-50K unique drainage IDs). Parallel adds multiprocessing overhead for small workloads. | Low — performance only, not correctness |
| `SpecifyHandBoundsNoAspect()` | `tu:415-443` | Simple quartile binning without aspect. Not needed for 4x4 config. **Would be required for 1x8 configuration** (PI question #2). | None currently; required if 1x8 adopted |
| `calc_network_length()` | `tu:30-154` | Legacy implementation superseded by `river_network_length_and_slope()` in pgrid. Swenson's own code also switched to the pgrid version (rh:1656-1661). | None — we use the same function Swenson uses in his latest code |
| `hand_analysis_global.py` (entire file) | 364 lines | Global driver script — loops over gridcells, calls `CalcGeoparamsGridcell()`, writes per-chunk NetCDF. We process a single OSBS gridcell. | None — pure orchestration, no science code missing |
| `read_MERIT_dem_data()` etc. | `dem_io.py` | MERIT tile I/O in geographic CRS. We have our own NEON tile/mosaic system. | None |
| `CalcRepresentativeHillslopeForm()` (circular/triangular sections) | `rh:181-338` | Alternative geometric models for width/distance. We use trapezoidal (Swenson's default). | None — trapezoidal matches published dataset |
| Column compression step (rh:971-1012) | `rh:971-1012` | Removes empty bins and compacts arrays. We always write 16 columns. | Low for OSBS (empty bins unlikely). See 4.8 #8. |
| Minimum hillslope count check (rh:1019-1036) | `rh:1014-1036` | Discards all data if < 3 of 4 aspects populated. | Low for OSBS (all 4 aspects populated at 90M pixels). See 4.8 #10. |

---

## 6. Functions we have that Swenson doesn't

- [x] Review each for correctness (no Swenson reference to compare against)

| Function | File | Purpose | Verified by |
|---|---|---|---|
| `parse_tile_range()` / `parse_all_tile_ranges()` | `rp:150-186` | NEON tile grid selection (R#C# syntax) | Tier 1/2/3 runs, tile count matches expected |
| `create_custom_mosaic()` | `rp:197-260` | Mosaic stitching from tiles via rasterio | Heatmap visualization, rasterio bounds/shape checks |
| Connected component extraction | `rp:856-923` | `scipy.ndimage.label` isolates largest contiguous region | Phase B/C runs on domains with/without nodata; stream network continuity |
| Nodata edge trimming | `rp:888-895` | Remove all-nodata rows/cols (pysheds fails on them) | Discovered empirically; documented in STATUS.md |
| `generate_mosaic_heatmap()` | `rp:262-290` | Input domain elevation visualization | Visual QC |
| `create_spectral_plot()` | `rp:292-321` | FFT power spectrum + fitted models | Visual QC of Phase C |
| `create_stream_network_plot()` | `rp:323-361` | DEM with stream network overlay | Visual QC |
| `create_hand_map_plot()` | `rp:363-401` | HAND field visualization | Visual QC |
| `create_hillslope_params_plot()` | `rp:403-482` | 4-panel summary of 6 hillslope params x 16 elements | Visual QC |
| `write_hillslope_netcdf()` | `rp:484-740` | CTSM-compatible NetCDF output (Swenson has equivalent inline code) | `ncdump` comparison against Swenson reference file |
| resolve_flats fallback | `rp:979-992` | Falls back to flooded DEM (not raw DEM); preserves connected drainage at 1m | Phase D documentation |
| `river_network_length_and_slope()` | pgrid.py:3170-3315 | CRS-aware stream slope computation (Swenson also uses this in latest code) | Pysheds test suite |

---

## 7. Data Flow Trace

Trace a hypothetical pixel through the pipeline: non-stream, non-edge, mid-HAND range (~3m), in the interior of the R6C10 tile.

- [x] Select a pixel with known properties (not on stream, not on edge, mid-HAND range)
- [x] Trace through each pipeline step:

| Step | Expected behavior | OK? |
|------|------------------|-----|
| Raw DEM elevation | Read from NEON DTM tile; float meters above WGS84 ellipsoid. Interior pixel, no nodata. | Yes |
| After basin detection | Unchanged — `identify_basins()` returns empty mask at 1m (no single elevation > 25% histogram frequency) | Yes |
| After fill_pits | Unchanged or sub-cm increase — only noise pits filled; mid-HAND pixels are not in pits | Yes |
| After fill_depressions | Unchanged or small increase if pixel was in a minor closed depression; most mid-HAND pixels unaffected | Yes |
| After resolve_flats | Unchanged or tiny epsilon increment to break ties in flat areas; most non-flat pixels unaffected | Yes |
| Flow direction | D8 direction to steepest downhill neighbor (one of 8 compass directions). Deterministic from inflated DEM. | Yes |
| Flow accumulation | Count of upstream pixels. Non-stream pixel: accumulation < A_thresh (45,000). Mid-hillslope: moderate accumulation (100s-1000s). | Yes |
| Stream/non-stream | Non-stream — accumulation < A_thresh = 45,000 | Yes |
| HAND | Positive value: height above the D8-traced nearest stream pixel. Mid-range for OSBS: ~2-5m. Computed by pysheds fork using Euclidean distance (Phase A). | Yes |
| DTND | Positive value: hydrologically-linked distance to nearest stream (Euclidean on UTM). Mid-range: ~100-200m. NOT the geographically nearest stream — follows D8 trace. | Yes |
| Slope | From Horn 1981 on original (unconditioned) DEM. OSBS typical: 0.03-0.05 m/m. | Yes |
| Aspect (raw) | From Horn 1981 on original DEM. Any direction 0-360. Sign convention correct (Phase A/D). | Yes |
| Aspect (catchment mean) | Replaced by circular mean of all pixels on same side of same catchment (headwater/left/right combined with channel type 4). Reduces local noise. | Yes |
| Aspect bin | One of N(315-45)/E(45-135)/S(135-225)/W(225-315) based on catchment mean aspect | Yes |
| HAND bin | One of 4 bins based on HAND value relative to quartile boundaries [0, Q25, Q50, Q75, max]. Mid-HAND (~3m) likely falls in bin 2 or 3. | Yes |
| Valid after tail removal? | Yes — mid-HAND, mid-DTND pixels are well within the exponential tail threshold (5% of peak PDF). Only extreme high-DTND pixels are removed. | Yes |
| Element parameters | Contributes to one of 16 elements (4 aspects x 4 HAND bins). Pixel's area, slope, and aspect averaged with other pixels in same element. DTND and HAND contribute to bin statistics. | Yes |

**Notes:** No divergences found in the data flow trace. Each step operates as documented. The key Phase A/D changes (Euclidean DTND, corrected slope/aspect) are properly integrated.

---

## 8. Numerical Robustness

Focus on `hillslope_params.py` and `run_pipeline.py` parameter computation only. pysheds edge cases are covered by the fork's pytest suite.

- [x] `quadratic()`: ak=0 handling, discriminant near-zero, negative discriminant
- [x] `fit_trapezoidal_width()`: n_hillslopes=0, singular GtWG matrix, all DTND below min_dtnd
- [x] `compute_hand_bins()`: initial_q25 == bin1_max exactly, empty aspect bins, zero-size hand_valid
- [x] `tail_index()`: std_dtnd near-zero, all HAND <= 0
- [x] `circular_mean_aspect()`: uniform distribution (returns arbitrary angle), single value, empty array
- [x] Parameter computation loop: empty bins, zero fitted areas, trap_slope=0

**Findings:**

| Function | Edge case | Handled? | Notes |
|----------|-----------|----------|-------|
| `quadratic()` | `ak=0` (division by zero in `2*ak`) | **No** | Neither our code nor Swenson's guards against `ak=0`. When trap_slope=0, equation is linear. Inherited gap from Swenson. |
| `quadratic()` | Discriminant near-zero (within eps) | Yes | Both adjust `ck` to force tangent solution. eps=1e-6 threshold identical. |
| `quadratic()` | Discriminant negative beyond eps | Yes | Raises RuntimeError. Same in both. |
| `fit_trapezoidal_width()` | `n_hillslopes=0` | **No** | Division by zero at hp:189-190, 207, 213, 253-254. In practice always >= 1 when `aind.size > 0`, but no guard. |
| `fit_trapezoidal_width()` | Singular GtWG matrix | Yes | Caught by `except Exception` (hp:248); returns heuristic fallback. Swenson would crash. |
| `fit_trapezoidal_width()` | All DTND below min_dtnd | Yes | Caught by `np.max(dtnd) <= min_dtnd` (hp:186); returns heuristic fallback. |
| `fit_trapezoidal_width()` | All weights (A_cumsum) zero | Partial | Singular matrix caught by except; fallback divides by n_hillslopes and returns width=0, area=0. |
| `compute_hand_bins()` | `initial_q25 == bin1_max` exactly | Yes | Takes common-case branch (> is false). Same in both. |
| `compute_hand_bins()` | Empty aspect bins | Partial | `hand_asp_sorted.size > 0` check prevents crash. Empty bin gets `bmin = bin1_max`. |
| `compute_hand_bins()` | `hand_valid.size == 0` | Yes | Returns hardcoded fallback (hp:80). |
| `compute_hand_bins()` | `hand_valid.size` 1-3 | Partial | `int(0.25 * n) - 1` = -1 when n < 4, indexes last element. All quartiles same — degenerate bins. Same in Swenson. |
| `tail_index()` | `std_dtnd == 0` exactly | Yes | Returns all indices (hp:466-467). Swenson would crash. |
| `tail_index()` | `std_dtnd` near-zero (1e-300) | Partial | Division produces huge values; `expon.fit` may misbehave. Extremely unlikely. |
| `tail_index()` | All HAND <= 0 | Yes | Returns all indices (hp:459-460). |
| `circular_mean_aspect()` | Uniform distribution | Yes | sin/cos sums approach 0; `arctan2(~0, ~0)` returns value near 0. Mathematically undefined but doesn't crash. |
| `circular_mean_aspect()` | Single value | Yes | Returns that value after sin/cos round-trip. |
| `circular_mean_aspect()` | Empty array | **No** | `np.mean(empty)` returns NaN with warning. NaN propagates silently. Same in Swenson. |
| Parameter loop | Empty bins | Partial | Placeholder element appended (non-zero height/aspect, zero area/width). No compression step. |
| Parameter loop | Zero fitted areas | Partial | `quadratic([0, trap_width, 0])` → `ak=0` → unhandled (see quadratic row). |
| Parameter loop | `trap_slope=0` | Partial | Passed to `quadratic` as `ak=0` → division by zero. Fallback to median_dtnd catches some cases. |

---

## 9. Output Validation

Quick physical plausibility checks on tier 3 output (`output/osbs/2026-02-26_tier3_contiguous/hillslope_params.json`).

- [x] Elevation increases monotonically bin 1 → bin 4 within each aspect
- [x] Distance increases monotonically bin 1 → bin 4 within each aspect
- [x] All slopes positive and physically reasonable (OSBS: 0.01-0.06 m/m)
- [x] All aspects in [0, 2π] radians (in NetCDF) / [0, 360] degrees (in JSON)
- [x] All areas positive
- [x] pct_hillslope sums to 100
- [x] 4 aspects have similar parameters (OSBS is nearly flat, nearly uniform aspect distribution — large asymmetries would indicate a bug)
- [x] width × distance ≈ area for each element (rough trapezoidal cross-check)

**Findings:**

| Check | Result | Notes |
|-------|--------|-------|
| Elevation monotonicity | PASS | All 4 aspects strictly increasing bin 1→4. Range: ~0.00007m (bin 1) to 8.8-9.6m (bin 4). |
| Distance monotonicity | PASS | All 4 aspects strictly increasing. Range: 34-44m (bin 1) to 351-367m (bin 4). |
| Slope range | PASS | min=0.0438 (N bin 1), max=0.0574 (N bin 2). All within [0.01, 0.06] m/m. |
| Aspect range | PASS | JSON in degrees, NetCDF in radians. All fall within expected quadrant. North aspects cluster near 0/360 (correct wrap-around). |
| Positive areas | PASS | All 16 areas positive. Range: 13,362-16,474 m^2. |
| pct_hillslope sum | PASS | sum = 100.00% (24.67 + 24.63 + 25.32 + 25.38). Max deviation from 25% is 0.38 pp. |
| Cross-aspect symmetry | PASS | Heights vary <4% CV across aspects. Distances vary 2-12% (N/W slightly larger than S/E in lower bins — mild, not alarming). Areas uniform (~20% spread from equal-area binning). |
| width x distance ~ area | N/A | This check is not valid for these parameters. Width is at the downslope *interface* (not average across bin), and distance is the *mean* DTND (not bin depth). width x distance is not geometrically equivalent to area for the trapezoidal model. |

**Additional observations:**

1. **Bin 1 heights effectively zero** (~0.00007 m): HAND bin boundary is [0, 0.00027 m], so bin 1 captures only pixels essentially at stream level. Consistent with the equal-area binning constraint and OSBS's large fraction of near-stream pixels.

2. **HAND bin boundaries very unequal**: [0, 0.00027, 1.61, 5.29, 25.10] m. Reflects OSBS's low-relief topography with a long tail of higher pixels.

3. **Width decreases with distance**: 192-204m (bin 1) to 107-115m (bin 4) — convergent hillslope geometry (wider near stream, narrower near ridge). Physically correct.

4. **Stream parameters**: depth=0.141m, width=1.67m, slope=0.00476 m/m. Labeled "INTERIM power law." Reasonable for small headwater streams in low-relief setting.

5. **Gridcell AREA = 0.235 km^2**: Sum of all 16 element areas = 235,211 m^2 = 0.235 km^2. This is mean catchment area, not domain area (90 km^2). Consistent.

6. **Bedrock depth = 1,000,000 m**: The placeholder noted in STATUS.md #7.

---

## 10. Summary

### Divergence counts

| Category | Count | Output-affecting |
|----------|-------|-----------------|
| CRS adaptation | 10 | 8 (all in UTM mode) |
| Resolution adaptation | 7 | 4 |
| Bug fix | 3 | 1 (zero-edges in DEM variant) |
| Python modernization | 35 | 0 |
| Code organization | 38 | 0 |
| Defensive coding | 19 | 2 (edge cases) |
| Unknown / needs investigation | 2 | 1 (tail removal order) |
| **Total** | **114** | **16** |

### Issues found

| # | Severity | Description | Action |
|---|----------|-------------|--------|
| 1 | **Moderate** | No compression step for empty bins — all columns always written, with placeholder values where Swenson would remove the column. `downhill_column_index` assumes all bins populated. | **Decision (2026-03-17):** Not an issue for this research. Moving to 1-aspect x 8-16 bins pools all 90M pixels into a single aspect — empty bins impossible. Pipeline is OSBS-only. Comment added to `write_hillslope_netcdf()` noting the assumption and pointing here. Would need a compression step if applied to other sites. |
| 2 | **Moderate** | `bedrock_depth = 1e6` vs Swenson's 0 — both placeholders, but CTSM may use this to limit soil column depth. | **Decision (2026-03-17):** Changed to 0 to match Swenson. Research confirmed both are no-ops: osbs2 uses `hillslope_soil_profile_method='Uniform'`, which skips `SetHillslopeSoilThickness` entirely — `hillslope_bedrock_depth` is never read. Even under `FromFile`, neither 0 nor 1e6 brackets any real soil layer, so `nbedrock` would retain the surfdata default. Using 0 for baseline consistency. |
| 3 | **Low** | Tail removal order differs (after vs before basin masking) — slightly different pixel population for exponential fit. | **Decision (2026-03-17):** Fixed — moved tail removal before DTND clip and basin masking to match Swenson's order. Basin pixels should be in the exponential fit so the tail threshold accounts for their long flow paths. Currently a no-op at OSBS (basin detection empty at 1m), but matters once synthetic lake bottoms is implemented. |
| 4 | **Low** | `_fit_polynomial` dropped Swenson's `weights.size` validation check. | **Fixed (2026-03-17):** Restored `y.size != weights.size` guard in `spatial_scale.py`. |
| 5 | **Low** | `quadratic()` unguarded for `ak=0` (trap_slope=0) — inherited from Swenson. | **Fixed (2026-03-17):** Added `ak=0` linear branch (`-ck/bk`) in `hillslope_params.py`. |
| 6 | **Low** | `fit_trapezoidal_width()` unguarded for `n_hillslopes=0` — division by zero. | **Accepted (2026-03-17):** Unreachable — loop only runs when pixels exist, guaranteeing >= 1 drainage ID. |
| 7 | **Low** | `max(trap_width, 1)` floor clips any fitted width below 1m — absent in Swenson. | **Accepted (2026-03-17):** Intentionally more conservative. Sub-meter fitted width at 1m resolution indicates a bad fit. |
| 8 | **Low** | `circular_mean_aspect()` on empty array produces silent NaN — inherited from Swenson. | **Accepted (2026-03-17):** Cannot trigger with 1-aspect configuration and 90M pixels. |
| 9 | **Info** | `AREA` variable written in NetCDF — Swenson does not. | **Decision (2026-03-17):** Intentional addition. Leave as-is. |
| 10 | **Info** | Broken code in Swenson's `IdentifySpatialScaleLaplacianDEM` — references undefined `selon`/`selat`/`selev` variables in coastal gridcell path. | Not our bug. Document for reference. |

### Pysheds fork confirmation

- [x] `cd $PYSHEDS_FORK && conda activate ctsm && python -m pytest tests/ -v` passes

**Result:** 83 passed, 33 skipped, 0 failed, 12 warnings (numpy cast warnings). The 33 skips are in `test_grid.py` (pre-existing API mismatch from Swenson's additions) and `test_hillslope.py` — both documented in STATUS.md #10.

### Overall assessment

The OSBS pipeline is a faithful adaptation of Swenson & Lawrence (2025) for 1m UTM LIDAR data. Of 114 divergences found, 98 are cosmetic (Python idiom, code organization, dead code removal, type annotations, logging changes) with no effect on computed parameters.

The 16 output-affecting divergences are all justified:
- **10 CRS adaptations** (UTM pixel_size, Euclidean DTND, Horn 1981 uniform spacing) are Phase A work, tested via 1000+ pytest assertions
- **4 resolution adaptations** (min_wavelength=20m, blend/zero=50px, min_dtnd=1m) are Phase B/C decisions with sensitivity analysis
- **1 bug fix** (zero-edges in DEM variant) corrects an oversight in Swenson's code
- **1 unknown** (tail removal order) warrants investigation but expected minor effect

No **Unknown** divergences of concern were found. The two items flagged as "Needs investigation" are:
1. Tail removal order (low impact — affects which pixels are included in the exponential tail fit, but both approaches are defensible)
2. `bedrock_depth` placeholder (already tracked in STATUS.md #7)

The structural gap of most concern is the absent **compression step** (#1 in issues). At OSBS with 90M pixels and uniform aspect distribution, all 16 bins will be populated, making this a non-issue for current production. However, if the pipeline is applied to smaller domains or sites with non-uniform topography, empty bins could produce incorrect `downhill_column_index` values. This should be addressed before generalizing beyond OSBS.

**Dead code identified in Swenson's codebase:**
- `_fit_polynomial`: unused `coefs` allocation, commented-out return
- `_fit_peak_gaussian`/`_fit_peak_lognormal`: `useIndividualWidths` variable (always True)
- `_LocatePeak`: exponential fit (computed, never used), `gof` array (allocated, never populated)
- `IdentifySpatialScaleLaplacianDEM`: blend-edges window triple-assignment (first two overwritten), broken coastal path (undefined variables)
- `identify_basins`: inner `for i in ind:` loop runs idempotent operation redundantly
- `compute_pixel_areas` (Swenson rh:1709): `farea = np.zeros(...)` immediately overwritten
