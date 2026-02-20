# MERIT Validation Pipeline Audit — By Pipeline Phase

Systematic comparison of Swenson's complete codebase against our MERIT
validation pipeline (`merit_regression.py`), organized by pipeline phase.
Each section consolidates Swenson's approach, ours, divergences, shared
function verification, and DEM conditioning notes under one roof.

**Date:** 2026-02-20
**Source files compared:**

| Abbreviation | File |
|--------------|------|
| rh | `$BLUE/Representative_Hillslopes/representative_hillslope.py` |
| ss | `$BLUE/Representative_Hillslopes/spatial_scale.py` |
| gu | `$BLUE/Representative_Hillslopes/geospatial_utils.py` |
| tu | `$BLUE/Representative_Hillslopes/terrain_utils.py` |
| mr | `scripts/merit_validation/merit_regression.py` |
| afd | `scripts/merit_validation/area_fraction_diagnostics.py` |
| our_ss | `scripts/spatial_scale.py` |
| pgrid | `$BLUE/pysheds_fork/pysheds/pgrid.py` |

---

## Summary

### Counts

| Category | Count |
|----------|-------|
| Total items compared | 48 |
| MATCH | 42 |
| DIVERGENCE | 2 |
| OMISSION | 1 |
| N/A (post-processing) | 3 |

### Current Expected Correlations

| Parameter | Correlation |
|-----------|-------------|
| Height | 0.9977 |
| Distance | 0.9987 |
| Slope | 0.9850 |
| Aspect | 1.0000 |
| Width | 0.9894 |
| Area fraction | 0.9221 |

### Pipeline Fidelity Rating

**High fidelity for the geographic CRS path.** The MERIT validation confirms
that our pipeline correctly implements Swenson's methodology for geographic
(lat/lon) data. All 6 parameters match at >=0.92 correlation, with 5 at >=0.98.
The remaining gap in area fraction (0.92) is attributable to stream network
differences (A_thresh, domain expansion) rather than algorithmic errors.

**Untested for OSBS (UTM).** The 6 DEM conditioning steps that are no-ops at
90m MERIT would become active at 1m OSBS data. The UTM code path in pgrid.py
(Phase A) has not been validated against this audit.

---

## 1. Spectral Analysis (FFT / Lc)

### Swenson

**Entry point:** `IdentifySpatialScaleLaplacian()` (ss:511-681)

**Region selection:** Called from `CalcGeoparamsGridcell` (rh:428-437) with
`scorners` -- 4 corners of the gridcell at `+/-0.5 * dlon/dlat` from center
(rh:413-418). A single FFT region per gridcell.

**DEM loading:** `dem_reader(dem_file_template, corners, zeroFill=True)` (ss:537)

**Resolution:** `ares = abs(elat[0] - elat[1]) * (re * pi/180)` (ss:547)

**Max wavelength:** `maxWavelength = 2 * maxHillslopeLength / ares` (ss:548)

**Land masking:** `lmask = where(elev > min_land_elevation, 1, 0)` (ss:553).
If `land_frac < land_threshold` (0.75), reads a larger region (sf=0.75 of
domain), computes `smooth_2d_array()` (gu:44-56), and subtracts (ss:609-612).

**Detrending:** `fit_planar_surface(elev, elon, elat)` (gu:59-75) -- 3-coefficient
planar fit via least squares (ss:615-617).

**Edge blending:** `blend_edges(elev, n=4)` (gu:77-112). Note: `win` is
hardcoded to 4 despite the formula `int(min(ejm,eim)//33)` appearing first
(ss:621-624). The formula result is overwritten on lines 622-623.

**Laplacian computation:** Two passes of `calc_gradient()` (gu:129-162):
```
grad = calc_gradient(elev, elon, elat)       # [dzdx, dzdy]
x = calc_gradient(grad[0], elon, elat)
laplac = x[0]                                # d^2z/dx^2
x = calc_gradient(grad[1], elon, elat)
laplac += x[1]                               # d^2z/dx^2 + d^2z/dy^2
```

**`calc_gradient` internals** (gu:129-162):
- `np.gradient(z)` returns `[dzdy, dzdx]` (axis-0, axis-1)
- Horn 1981: 4-point averaging `ind = [-1, 0, 0, 1]` at interior points
- Edge points: 3-point averaging `eind = [0, 0, 1]` / `[-2, -1, -1]`
- Spacing: `dx = re * dtr * abs(lon[0]-lon[1])`, latitude-dependent:
  `dx2d = dx * tile(cos(dtr*lat), (lon.size,1)).T`
  `dy2d = dy * ones((lat.size, lon.size))`

**Edge zeroing:** 5 pixels on each edge of the Laplacian (ss:641-643).

**FFT:** `np.fft.rfft2(laplac, norm='ortho')` (ss:645).

**Wavelength grid:** `radialfreq = sqrt(colfreq^2 + rowfreq^2)`;
`wavelength = 1/radialfreq` where `radialfreq > 0`;
`wavelength[0,0] = 2*max(wavelength)` (ss:651-662).

**Spectral binning:** `_bin_amplitude_spectrum(amp_fft, wavelength, nlambda=30)`
(ss:73-90). Log-spaced wavelength bins: `linspace(0, max(log10(wavelength)),
nlambda+1)`. Mean amplitude and wavelength per bin.

**Peak fitting:** `_LocatePeak(lambda_1d, amp_1d, maxWavelength)` (ss:367-508):
1. Linear fit: `_fit_polynomial(log10(lambda), amp, ncoefs=2)` (ss:393-395)
2. Gaussian fit: `_fit_peak_gaussian()` (ss:243-364)
3. Log-normal fit: `_fit_peak_lognormal()` (ss:105-236)
4. T-score for linear (ss:421-433)
5. Selection hierarchy:
   - If `psharp_ga >= 1.5` or `psharp_ln >= 1.5`: choose gaussian or lognormal
     by lower GOF (ss:451-466)
   - Else if `tscore > 2`: linear model, spatialScale = maxWavelength or
     minWavelength depending on slope sign (ss:469-478)
   - Else: flat model, spatialScale = minWavelength (ss:480-485)

**Output:** `spatialScale` in pixels, bounded by `[minWavelength, maxWavelength]`
(ss:676-677).

**A_thresh:** `accum_thresh = 0.5 * spatialScale^2` (rh:451).

### Ours

**Entry point:** `compute_lc(dem_path)` (mr:112-174)

**Region selection:** 3 center crops (500^2, 1000^2, 3000^2) from the full MERIT
tile + full tile at 4x subsampling (mr:120-171). Takes **median** of all 4 Lc
values in meters (mr:173).

Each region calls `identify_spatial_scale_laplacian_dem()` (our_ss:467-630) with:
```python
max_hillslope_length=10000, nlambda=30, detrend_elevation=True,
blend_edges_flag=True, zero_edges=True
```

**Internal logic** (our_ss:467-630) is identical to Swenson's
`IdentifySpatialScaleLaplacianDEM()` (ss:683-806):
- Same `calc_gradient` (our_ss:23-76 = gu:129-162)
- Same Laplacian: `d^2z/dx^2 + d^2z/dy^2`
- Same edge zeroing (5 pixels)
- Same FFT: `rfft2(laplac, norm='ortho')`
- Same spectral binning: `_bin_amplitude_spectrum()`
- Same peak fitting: `_locate_peak()` = `_LocatePeak()`
- Same model selection thresholds (psharp=1.5, tscore=2)

**A_thresh:** `accum_threshold = int(0.5 * median_lc_px^2)` (mr:845)

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 1 | Gradient computation | MATCH | -- | -- |
| 2 | Edge blending (win=4) | MATCH | -- | -- |
| 3 | Laplacian | MATCH | -- | -- |
| 4 | rfft2, norm='ortho' | MATCH | -- | -- |
| 5 | Spectral binning | MATCH | -- | -- |
| 6 | Peak fitting (gaussian/lognormal/linear) | MATCH | -- | -- |
| 7 | Model selection thresholds | MATCH | -- | -- |
| 8 | **Region selection** | **DIVERGENCE** | No | See analysis below |
| 9 | A_thresh formula | MATCH | -- | -- |

**#8 — FFT Region Selection (DIVERGENCE)**

**Swenson:** Single FFT on the gridcell-sized region (rh:413-418, ss:537).
One `scorners` defined as `+/-0.5 * dlon/dlat` from center. The
`IdentifySpatialScaleLaplacian` function reads the DEM for exactly this
region and computes one Lc.

**Ours:** 3 center crops (500^2, 1000^2, 3000^2 pixels) + full tile at 4x
subsampling (mr:120-171). Takes the **median** of 4 Lc values in meters.

**Difference:** We compute 4 independent FFTs on different spatial extents and
take the median. Swenson computes 1 FFT on the gridcell footprint.

**Tested?** Not directly. However, area_fraction_research.md Test O showed
that the full-tile native FFT (Lc = 9.8 px, A_thresh = 48) would **worsen**
area fraction, while the center crops give Lc ~ 8.1-8.3 px (A_thresh ~ 33-34).
The median of our 4 regions (8.25 px, A_thresh = 34) happens to produce good
results.

**Impact:** Moderate on area fraction (which is sensitive to A_thresh), minimal
on the other 5 parameters. Test O showed height, distance, slope, aspect all
stay above 0.995 across A_thresh = 20-100. Width is stable at ~0.91-0.95.
Area fraction varies from 0.84 (A_thresh=20) to -0.02 (A_thresh=100).

**Scientific validity:** Our multi-region median approach is more robust than
a single FFT (less sensitive to local anomalies), but it does not replicate
Swenson's exact Lc. This is acceptable for validation purposes -- the goal is
to verify the pipeline produces correct results for a given A_thresh, not to
reproduce Swenson's exact A_thresh.

### Shared Functions

**calc_gradient** — Swenson: gu:129-162. Ours: our_ss:23-76. **MATCH.**
Both use identical logic:
1. `np.gradient(z)` -> `[dzdy, dzdx]`
2. Horn 1981 4-point averaging with `ind = [-1, 0, 0, 1]`
3. 3-point edges with `[0, 0, 1]` and `[-2, -1, -1]`
4. Spherical spacing: `dx = re*dtr*abs(lon[0]-lon[1])`,
   `dx2d = dx * cos(lat)`, `dy2d = dy * ones()`
Variable naming differs (`dtr` vs `DTR`, `re` vs `RE`) but values are
identical (pi/180, 6.371e6).

**blend_edges** — Swenson: gu:77-112. Ours: our_ss:113-140. **MATCH.**
Identical algorithm: progressive weighted averaging from edges inward, applied
to both axes.

**smooth_2d_array** — Swenson: gu:44-56. Ours: our_ss:79-92. **MATCH.**
Identical FFT-based smoothing with `hw = scalar/(land_frac^2 * min(shape))`.

**fit_planar_surface** — Swenson: gu:59-75. Ours: our_ss:95-110. **MATCH.**
Identical 3-coefficient least-squares planar fit.

**_fit_polynomial** — Swenson (rh): rh:113-136. Swenson (ss): ss:37-61.
Ours: our_ss:143-163. **MATCH.**
All three are identical: construct Vandermonde matrix `G`, apply optional
`diag(weights)`, solve via `inv(G^T W G) * G^T W y`.
The critical w^1 weighting: `weights = A` passes through `diag(A)`, producing
`G^T * diag(A) * G` and `G^T * diag(A) * y`. This minimizes
`sum_i w_i * r_i^2`, not `sum_i w_i^2 * r_i^2`.

**_bin_amplitude_spectrum** — Swenson: ss:73-90. Ours: our_ss:174-194. **MATCH.**
Identical: log-spaced bins, mean amplitude and wavelength per bin.

**_LocatePeak / _locate_peak** — Swenson: ss:367-508. Ours: our_ss:363-464.
**MATCH.** Renamed but identical logic. Same thresholds (psharp=1.5, tscore=2),
same model hierarchy, same spatialScale bounds.

**_gaussian_no_norm** — Swenson: ss:239-240. Ours: our_ss:197-201. **MATCH.**
Identical: `amp * exp(-(x-cen)^2 / (2*sigma^2))`.

**_log_normal** — Swenson: ss:93-102. Ours: our_ss:204-214. **MATCH.**
Identical: `amp * exp(-(ln(x-shift) - mu)^2 / (2*sigma^2))` for `x > shift`.

**_fit_peak_gaussian** — Swenson: ss:243-364. Ours: our_ss:217-287. **MATCH.**
Same `signal.find_peaks` parameters, same edge peak addition, same width-based
window, same curve_fit with pdist check, same psharp = ramp/rwid.

**_fit_peak_lognormal** — Swenson: ss:105-236. Ours: our_ss:290-360. **MATCH.**
Same logic as gaussian variant but with lognormal model and lognormal variance
for sharpness.

### Deep Audit Findings (2026-02-20)

Line-by-line comparison of Swenson's `spatial_scale.py` (ss:1-807),
`geospatial_utils.py` (gu:1-162), and `representative_hillslope.py`
(rh:410-464) against our `spatial_scale.py` (our_ss:1-631) and
`merit_regression.py` (mr:1-180, 1130-1191). Findings A-I supplement
the divergence table above.

**Finding A: Exponential fit omitted (benign)**

Swenson's `_LocatePeak` (ss:397-400) fits an exponential model alongside
linear/gaussian/lognormal:
```python
ecoefs = _fit_polynomial(logLambda[lmin:lmax+1],
                         np.log(ratio_var_to_lambda[lmin:lmax+1]), 2)
```
But `ecoefs` is never used in the model selection logic (ss:447-485). It's
computed and discarded. Our `_locate_peak` omits it entirely. No functional
difference.

**Finding B: Division-by-zero guards in t-score (defensive improvement)**

Swenson (ss:432-433):
```python
se = np.sqrt(num / den)
tscore = np.abs(lcoefs[1]) / se
```
Ours (our_ss:415-416):
```python
se = np.sqrt(num / den) if den > 0 else 1e10
tscore = np.abs(lcoefs[1]) / se if se > 0 else 0
```
We add guards against division by zero. These would only trigger with
degenerate input (constant wavelengths). Defensive, correct.

**Finding C: Edge zeroing source discrepancy (matches production path)**

Swenson has TWO versions of the FFT function:
- `IdentifySpatialScaleLaplacian` (ss:511-681, file-reader): has `zeroEdges`
  parameter (default True), zeros 5px border at ss:641-643
- `IdentifySpatialScaleLaplacianDEM` (ss:683-806, array-input): has NO
  `zeroEdges` parameter, no edge zeroing

Our `identify_spatial_scale_laplacian_dem` (our_ss:467-630) adds `zero_edges`
parameter (default True) and applies it at our_ss:574-579.

The published data was generated via `CalcGeoparamsGridcell` which calls the
file-reader version (with edge zeroing). Our code matches the production path.
The DEM variant in Swenson's code is a testing function that lacks this step.

**Finding D: Coastal adjustment simplification (no-op for MERIT)**

Swenson's file-reader version (ss:563-612): If `land_frac < 0.75`, expands
domain by 0.75x, reads a larger DEM, computes `smooth_2d_array()` (gu:44-56),
and subtracts.

Swenson's DEM variant (ss:720-731): References undefined variables (`selon`,
`selat`, `selev`) -- this code would crash with NameError if triggered. Bug
in Swenson's testing function.

Ours (our_ss:546-550): Smooths the input array directly without domain
expansion. Simpler but less correct for coastal cells.

No-op for MERIT validation (land_frac >= 0.75 for this inland gridcell).
Would matter for coastal gridcells only.

**Finding E: `detrendElevation` default (documented insensitive)**

Swenson: `detrendElevation=False` default in both FFT functions. The actual
value used for published data depends on the caller -- unknown.

Ours: Always passes `detrend_elevation=True` (mr:133).

Phase C Test E showed Lc is insensitive to this setting. No functional
difference at MERIT. Worth noting for documentation completeness.

**Finding F: Lognormal edge peak log(0) guard (defensive improvement)**

Swenson (ss:177): `mu = np.log(center)` -- no guard for `center <= 0`
Ours (our_ss:324): `mu = np.log(center) if center > 0 else 0`

When the edge peak (index 0) is tested, `center = x[0]` is the shortest
log-wavelength, which could approach 0. Our guard prevents RuntimeWarning.

**Finding G: Dead code in main() (cleanup)**

mr:1162-1163 computes an A_thresh value using `median_lc_m / 92.3` that
is immediately overwritten on mr:1166-1168 with the correct pixel-based
computation. The dead code is harmless but confusing. **Removed 2026-02-20.**

**Finding H: find_peaks keyword arguments (style only)**

Swenson passes all keyword arguments explicitly including defaults
(`threshold=None, distance=None, wlen=None, plateau_size=None`).
Ours omits default-valued kwargs. Identical behavior.

**Finding I: Bare except vs except Exception (style only)**

Swenson: `except:` (ss:198, 330) -- catches all exceptions including
SystemExit, KeyboardInterrupt.
Ours: `except Exception:` (our_ss:263, 335) -- only catches standard
exceptions. More correct but same practical behavior (curve_fit only
raises RuntimeError/ValueError/TypeError).

---

## 2. Domain Expansion

### Swenson

**Dynamic sizing:** `sf = 1 + 4 * (ares * spatialScale) / grid_spacing` (rh:462).
Where `grid_spacing = abs(dlon * dtr * re)`.

**Subregion splitting:** If `gs_ratio = grid_spacing / scale_in_meters > 400`,
split into 4 subregions (rh:490-495).

### Ours

**Fixed expansion:** `EXPANSION_FACTOR = 1.5` (mr:67).
Load with rasterio window (mr:375-381).
Compute gridcell extraction slices (mr:384-393).

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 10 | **Dynamic vs fixed expansion** | **DIVERGENCE** | No | See analysis below |

**#10 — Domain Expansion (DIVERGENCE)**

**Swenson:** Dynamic: `sf = 1 + 4 * (ares * spatialScale) / grid_spacing`
(rh:462). For this gridcell with `ares ~ 92m`, `spatialScale ~ 8.3 px`,
`grid_spacing ~ 110 km`: `sf ~ 1 + 4 * 764 / 110000 ~ 1.028`. The expanded
region is only ~2.8% larger than the gridcell in each direction.

**Ours:** Fixed `EXPANSION_FACTOR = 1.5` (mr:67). The expanded region is 50%
larger than the gridcell.

**Tested?** Not directly. However, a larger expansion means more DEM data for
flow routing, which reduces edge effects -- more catchments near the gridcell
boundary will be fully resolved. This should only improve quality.

**Impact:** Minimal to positive. A larger buffer means the extracted gridcell
interior is further from the routing domain edge. Edge effects (misrouted
flow near domain boundaries) are reduced.

**Scientific validity:** Our larger expansion is conservative and appropriate.
Swenson's dynamic formula produces a very small buffer (~3%) that could leave
border catchments partially resolved, but he handles this via subregion splitting
for very fine grids (gs_ratio > 400). For this MERIT gridcell, neither approach
causes problems.

### Deep Audit Findings (2026-02-20)

Line-by-line comparison of Swenson's `representative_hillslope.py`
(rh:341-609), `geospatial_utils.py` (gu:1-306), and our
`merit_regression.py` (mr:370-740) domain expansion code. Findings A-E
supplement the divergence table above.

**Finding A: DEM loading — zeroFill vs native nodata (no-op for MERIT)**

Swenson (rh:1475): `dem_reader(template, corners, zeroFill=True)` converts
nodata pixels (`<= -9999`) to elevation 0, then uses `fill_value=np.nan`
for basin masking later.

Ours (mr:560-580): rasterio window read keeps native nodata (-9999 for
MERIT) throughout.

No nodata pixels exist in this fully-land MERIT gridcell — both approaches
read the same elevation values. Would matter for coastal cells where
zero-filled ocean pixels differ from -9999 nodata in basin detection logic.

**Finding B: No longitude wrapping (benign)**

Swenson (rh:420-426, 482-487): Wraps corner longitudes to [0, 360] before
DEM reading.

Ours: Uses native negative longitudes from the MERIT tile. rasterio handles
negative longitudes natively for window-based reads.

Both produce the same pixel window. The wrapping only matters for tiles
crossing the antimeridian, which our single-gridcell MERIT test does not.

**Finding C: No subregion splitting (no-op for MERIT)**

Swenson (rh:489-495): If `gs_ratio = grid_spacing / scale_in_meters > 400`,
splits the gridcell into 4 subregions and processes them separately. This
handles very-fine-resolution DEMs relative to gridcell size.

For this MERIT gridcell, `gs_ratio ~ 144` (110 km gridcell / 764m Lc), well
below the 400 threshold. Swenson would not split either. No-op.

**Finding D: Gridcell extraction — transform-based vs arg_closest_point
(equivalent)**

Swenson (rh:547-567): Uses `arg_closest_point()` (gu:115-126), which
iterates through the coordinate array in 32-bit float to find the nearest
pixel to each corner coordinate.

Ours (mr:568-578): Computes pixel indices algebraically from the rasterio
affine transform: `col = (lon - origin_x) / pixel_width`.

Both identify the same pixel range. The transform-based approach is exact
for regular grids (which MERIT is). The `arg_closest_point` search is
equivalent but slower for large arrays.

**Finding E: A_thresh type — int vs float (benign)**

Swenson (rh:451): `accum_thresh = 0.5 * spatialScale**2` — produces a
float (e.g., 34.03).

Ours (mr:1164): `int(0.5 * median_lc_px**2)` — truncates to int (e.g., 34).

Less than 1 pixel difference. Flow accumulation values are integers, so the
threshold comparison (`acc > thresh`) rounds the same way regardless of
whether thresh is 34 or 34.03.

---

## 3. DEM Conditioning

### Swenson

Defined at rh:1457-1754 (`CalcLandscapeCharacteristicsPysheds`).

1. **DEM loading:** `dem_reader(dem_file_template, corners, zeroFill=True)` (rh:1475)
2. **Basin identification:** `identify_basins(elev, nodata=fill_value)` (rh:1514) ->
   sets basin pixels to `fill_value` (NaN) (rh:1519)
3. **Grid creation:** `Grid.from_array(data=elev, affine=eaffine, crs=ecrs,
   nodata=fill_value)` (rh:1522-1528)
4. **Pit filling:** `grid.fill_pits(dem)` (rh:1543)
5. **Depression filling:** `grid.fill_depressions(pit_filled_dem)` -> `flooded_dem`
   (rh:1544)
6. **Open water detection:** `slope, _ = grid.slope_aspect(dem)` (rh:1547) ->
   `identify_open_water(slope, max_slope=1e-4)` (rh:1548) -> `basin_boundary`,
   `basin_mask`
7. **Basin lowering:** `flooded_dem[basin_mask > 0] -= 0.1` (rh:1549) -- forces
   flow through open water instead of around it
8. **Flat resolution:** `grid.resolve_flats(flooded_dem)` -> `inflated_dem` (rh:1563)

`resolve_flats` is wrapped in try/except ValueError (rh:1562-1567) -- on
failure, falls back to the flooded DEM and continues.

### Ours

1. `identify_basins(dem_data, nodata=nodata_val)` (mr:512) -> mask basin pixels
   as nodata before pysheds
2. `grid.fill_pits("dem", out_name="pit_filled")` (mr:529)
3. `grid.fill_depressions("pit_filled", out_name="flooded")` (mr:530)
4. `grid.slope_aspect("dem")` (mr:534) -- compute slope/aspect on original DEM
   (used for water detection + final output)
5. `identify_open_water(slope)` (mr:540) -> `basin_boundary`, `basin_mask`
6. `flooded_arr[basin_mask > 0] -= 0.1` (mr:547) -- lower basin pixels in
   flooded DEM to force flow through open water
7. `grid.resolve_flats("flooded", out_name="inflated")` (mr:557) -- on modified
   flooded DEM, with try/except ValueError fallback

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 11 | fill_pits / fill_depressions / resolve_flats | MATCH* | -- | *Fallback target differs (Finding A). Not triggered. |
| 12 | identify_basins | MATCH | Yes (N) | No-op at 90m. Implemented 2026-02-20. |
| 13 | identify_open_water | MATCH | Yes (C, N) | No-op at 90m. Implemented 2026-02-20. |
| 14 | 0.1m basin lowering | MATCH | Yes (N) | No-op at 90m. Implemented 2026-02-20. |

Items 12-14 were implemented as part of the basin/water handling chain
(2026-02-20). At 90m MERIT resolution, 0 basin pixels and 0 open water
pixels are detected -- these are confirmed no-ops. They will become active
at 1m OSBS resolution where sinkholes, wetlands, ponds, and swamps are
contiguous flat regions.

### DEM Conditioning Notes

**resolve_flats ValueError fallback** (dem_conditioning_todo item 1):
Swenson wraps `resolve_flats` in try/except ValueError (rh:1562-1567) and
falls back to the **original** DEM on failure (not the flooded DEM). Implemented in our pipeline.
No regression impact -- `resolve_flats` succeeds on this MERIT gridcell.

**acc_mask isfinite check** (dem_conditioning_todo item 2):
Swenson's accumulation mask includes `np.isfinite(inflated_dem)` (rh:1604-1606)
to exclude NaN basin pixels from the stream network. Implemented in our
pipeline. No-op at MERIT (0 NaN pixels in inflated DEM). Will matter for
OSBS where basin interiors are NaN.

### Deep Audit Findings (2026-02-20)

Line-by-line comparison of Swenson's `representative_hillslope.py`
(rh:1457-1614) and `geospatial_utils.py` (gu:1-306) DEM conditioning code
against our `merit_regression.py` (mr:370-740). Findings A-I supplement
the divergence table above.

**Finding A: resolve_flats fallback target (functional divergence, dormant)**

Swenson (rh:1562-1567): The `resolve_flats` try/except ValueError block
falls back to `grid.add_gridded_data(dem)` — the **original** basin-masked
DEM (before `fill_pits` / `fill_depressions`). This reverts all DEM
conditioning on failure.

Ours (mr:636-642): Falls back to `grid.flooded` — the depression-filled,
basin-lowered DEM. This preserves pit-filling and depression-filling.

Only triggers on ValueError from `resolve_flats`. Does not trigger for
this MERIT gridcell. Our approach is arguably more sensible (preserves
the more-processed DEM), while Swenson's reverts to the rawest state.

**Note:** The DEM Conditioning Notes above originally stated Swenson falls
back to the flooded DEM — this was incorrect and has been corrected.

**Finding B: No land fraction checks (no-op for MERIT)**

Swenson (rh:1481-1484): Two guards before processing:
1. If land fraction < `min_land_fraction` (0.01): skip gridcell entirely
2. If max elevation - min elevation < `min_relief` (0.001): skip gridcell

Ours: No equivalent checks. We process the single MERIT gridcell
unconditionally.

Both guards would pass for this gridcell (land fraction = 1.0, relief
>> 0.001m). These are production filters for the global pipeline that
skip ocean cells and perfectly flat cells. Not needed for single-gridcell
validation.

**Finding C: No validity check after fill_pits (no-op for MERIT)**

Swenson (rh:1540-1542): After `fill_pits`, checks `s1 = dem.size` and
`s2 = pit_filled.size`. If either `<= 1`, skips the gridcell.

Ours: No equivalent check.

A safety valve for degenerate grids where pit-filling produces an empty
or single-pixel result. Cannot trigger for the 1200x1200+ pixel MERIT
gridcell. Production filter only.

**Finding D: flooded_dem not re-masked after flowdir (dead code in Swenson)**

Swenson (rh:1582): `flooded_dem[basin_mask > 0] = fill_value` — re-masks
basin pixels in the flooded DEM with NaN after `flowdir()`.

This is a write to an array that is never read again after this point.
The inflated DEM (from `resolve_flats`) is the one passed to
`compute_hand`. The flooded DEM is only used as input to `resolve_flats`
(rh:1563), which has already completed.

Ours: We do the same re-masking (mr:564), which is equally dead code.
Harmless but not functionally meaningful.

**Finding E: Redundant slope_aspect call in Swenson (ours is more efficient)**

Swenson calls `grid.slope_aspect(dem)` twice:
1. rh:1547 — for `identify_open_water()` (uses slope only)
2. rh:1593 — for the output aspect array (uses both slope and aspect)

Ours (mr:608-609): Calls `slope_aspect("dem")` once, before `resolve_flats`,
and reuses the result for both water detection and final output.

Functionally equivalent — `slope_aspect` is deterministic on the same
input. Ours avoids the redundant computation.

**Finding F: Grid creation API difference (equivalent)**

Swenson (rh:1522-1528): `Grid.from_array(data=elev, affine=eaffine,
crs=ecrs, nodata=fill_value)` — class method that creates a Grid from
an array.

Ours (mr:515-518): `Grid()` + `grid.add_gridded_data(dem_data, ...)` —
creates an empty Grid, then adds data.

Both produce the same internal state: a pysheds Grid with the DEM as the
`"dem"` view. The `from_array` class method is syntactic sugar around the
same underlying logic.

**Finding G: nanmax vs max for A_thresh check (defensive improvement,
refines item #17)**

Swenson (rh:1597): `if np.max(acc) > accum_thresh` — uses plain `max`,
positive condition (keep thresh if max exceeds it, else reduce).

Ours (mr:674): `if np.nanmax(acc_arr) < accum_threshold` — uses
`nanmax`, inverted condition. Equivalent logic.

For this MERIT gridcell, no NaN values exist in the accumulation array,
so `max` and `nanmax` return the same value. At OSBS with basin-masked
NaN pixels, plain `max` returns NaN, and `NaN > thresh` evaluates False
— causing Swenson's code to take the reduce branch with
`np.max(acc) / 100 = NaN`, breaking the stream network. Our `nanmax`
handles this correctly.

**Note:** The previous version of this finding (2026-02-20 Sections 2-3
audit) had both sides reversed — it claimed Swenson used `nanmax` and
we used `max`. Corrected after verification against rh:1597 and mr:674.

**Finding H: identify_basins cleanup loop (no-op for MERIT, refines
item #12)**

Swenson's `identify_basins()` (gu:263-296) has a cascading cleanup loop
that re-runs Laplacian basin detection after each removal, because removing
one basin can expose new flat regions that look like basins. The loop runs
until no new basins are found.

Ours (mr:416-452): Same cascading loop implemented identically.

Both find 0 basins at 90m MERIT resolution. The cascading behavior matters
at 1m OSBS resolution where removing a large wetland basin can create
apparent flat zones at its boundary.

**Finding I: identify_open_water return type (style only, refines item #13)**

Swenson's `identify_open_water()` (gu:298-306) returns NumPy arrays.
Ours (mr:454-467): Returns NumPy arrays with identical values.

The function signatures and return values are functionally identical.
Stylistic differences in variable naming (`basin_boundary` vs
`open_water_boundary`) but same semantics.

---

## 4. Flow Routing

### Swenson

1. **Flow direction:** `grid.flowdir(inflated_dem, dirmap=dirmap)` (rh:1578)
2. **Re-mask basins:** `flooded_dem[basin_mask > 0] = NaN`,
   `inflated_dem[basin_mask > 0] = NaN` (rh:1582-1583)
3. **Accumulation:** `grid.accumulation(fdir, dirmap=dirmap)` (rh:1586)
4. **Force basin boundaries into stream network:**
   `acc[basin_boundary > 0] = accum_thresh + 1` (rh:1590)
5. **A_thresh adjustment:** If `max(acc) < accum_thresh`, use `max(acc)/100`
   (rh:1597-1601)

### Ours

1. `grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP)` (mr:561)
2. Re-mask basins: `inflated_arr[basin_mask > 0] = np.nan` (mr:564-572)
3. `grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP)` (mr:575)
4. Force basin boundaries into stream network:
   `acc_arr[basin_boundary > 0] = accum_threshold + 1` (mr:579)
5. A_thresh safety valve: if `max(acc) < thresh`, use `max(acc)/100` (mr:582-585)

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 15 | Basin re-masking after flowdir | MATCH | Yes (N) | No-op at 90m. Implemented 2026-02-20. |
| 16 | Basin boundary stream forcing | MATCH | Yes (N) | No-op at 90m. Implemented 2026-02-20. |
| 17 | A_thresh adjustment (max(acc)/100) | MATCH | Yes | No-op at 90m. Implemented 2026-02-20. |

Items 15-17 are part of the basin/water handling chain. At 90m MERIT, no
basins exist, so these are confirmed no-ops. At OSBS 1m: item 15 would
create stream outlets at wetland boundaries, item 16 forces basin perimeters
into the stream network, and item 17 could trigger in very flat terrain where
accumulation never reaches the threshold.

**#17 — A_thresh Adjustment**

**Swenson:** If `max(acc) < accum_thresh`, use `max(acc) / 100` (rh:1597-1601).
This is a safety valve for very flat areas where flow never accumulates enough
to exceed the threshold.

**Ours:** Now implemented (mr:582-585).

**Tested?** Not directly, but for this MERIT gridcell `max(acc)` far exceeds
any reasonable threshold. The adjustment never triggers.

**Impact:** None for this gridcell. Could matter for flat terrain (e.g., OSBS)
where accumulation might be lower.

---

## 5. Slope and Aspect

### Swenson

`slope, aspect = grid.slope_aspect(dem)` -- from **original** DEM (rh:1593).

Uses `_gradient_horn_1981()` (pgrid:4172-4226):
- 8-neighbor stencil indexed as `[N, NE, E, SE, S, SW, W, NW]`
- CRS-aware spacing: geographic uses `re * abs(dtr * dx * cos(lat))`,
  projected uses `abs(dx)` directly (pgrid:4199-4208)
- `dzdx = (sum(E_neighbors) - sum(W_neighbors)) / (8 * mean_cell_dx)` (pgrid:4224)
- `dzdy = (sum(N_neighbors) - sum(S_neighbors)) / (8 * mean_cell_dy)` (pgrid:4225)
- `slope = sqrt(dzdx^2 + dzdy^2)` (pgrid:2296)
- `aspect = (180/pi) * arctan2(-dzdx, -dzdy)` -> compass bearing (pgrid:2300)
- Converted to [0, 360] (pgrid:2303)

### Ours

`grid.slope_aspect("dem")` (mr:608-609) -- from original DEM. Computed early
(before resolve_flats) for water detection; same values reused as input to
catchment-level aspect averaging (see section 7).
Same `_gradient_horn_1981()` implementation in pgrid.py.

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 18 | slope_aspect on original DEM | MATCH | -- | -- |
| 19 | Horn 1981 stencil | MATCH | -- | -- |

---

## 6. Stream Network and HAND/DTND

### Swenson

1. **Accumulation mask:** `acc_mask = (acc > thresh) & isfinite(inflated_dem)`
   (rh:1604-1606)
2. **River network extraction:** `grid.extract_river_network(fdir, mask=acc_mask,
   dirmap=dirmap)` (rh:1609) -- used for stream statistics
3. **River network length and slope:**
   `grid.river_network_length_and_slope(dem=inflated_dem, fdir=fdir, acc=acc,
   mask=acc_mask, dirmap=dirmap)` (rh:1655-1673) -- total network length,
   mean slope, per-reach arrays. Wrapped in try/except MemoryError.
4. **Channel mask:** `grid.create_channel_mask(fdir, mask=acc_mask, dirmap=dirmap)`
   -> `channel_mask, channel_id, bank_mask` (rh:1679)
5. **HAND/DTND:** `grid.compute_hand(fdir, inflated_dem, channel_mask,
   channel_id, dirmap=dirmap)` (rh:1685) -- note: uses **inflated_dem**

`compute_hand` internals (pgrid:1934-2047):
- Breadth-first upstream trace from channel pixels
- `hndx[pixel] = flat_index_of_drainage_channel_pixel` (pgrid:1954)
- `HAND = dem[pixel] - dem.flat[hndx[pixel]]` (pgrid:1956)
- `drainage_id = channel_id.flat[hndx[pixel]]` (pgrid:1959-1961)
- **DTND:** Haversine distance from pixel to its drainage channel pixel
  (pgrid:1989-1992) for geographic CRS; Euclidean for projected (pgrid:2007)

### Ours

1. `acc_mask = (grid.acc > accum_threshold) & isfinite(inflated)` (mr:680)
2. `grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)` (mr:681)
3. `grid.extract_river_network("fdir", mask=acc_mask, dirmap=DIRMAP)` (mr:686)
   -- GeoJSON stream reaches (Swenson rh:1608-1614)
4. `grid.inflated_dem = grid.inflated` (mr:699) -- alias for pgrid compatibility
5. `grid.river_network_length_and_slope("fdir", mask=acc_mask, dirmap=DIRMAP)`
   (mr:702) -- total network length, mean slope, per-reach arrays
   (Swenson rh:1655-1673)
6. `grid.compute_hand("fdir", "inflated", grid.channel_mask, grid.channel_id,
   dirmap=DIRMAP)` (mr:715-722) -- uses **inflated DEM** (matches Swenson rh:1685)

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 21 | DEM passed to compute_hand | MATCH | Yes | Now passes `"inflated"` per rh:1685 |
| 22 | DTND formula (haversine for geographic) | MATCH | -- | -- |
| 23 | drainage_id from BFS | MATCH | -- | -- |
| 47 | River network extraction (stats) | MATCH | Yes (N) | `extract_river_network` + `river_network_length_and_slope` added |

**#21 — DEM Passed to compute_hand (RESOLVED)**

**Swenson:** `grid.compute_hand(fdir, inflated_dem, channel_mask, channel_id)`
(rh:1685) -- passes the pit-filled, depression-filled, flat-resolved DEM.

**Ours:** Now matches Swenson. `grid.compute_hand("fdir", "inflated", ...)`
(mr:715-722).

**Resolution (commit 01a8532, 2026-02-20):** Switched from `"dem"` to
`"inflated"` to match Swenson rh:1685. HAND is now computed relative to the
same surface that determined flow directions.

**Impact (measured, with basin chain in place):**

| Parameter | Before (`"dem"`) | After (`"inflated"`) | Delta |
|-----------|------------------|----------------------|-------|
| Height | 0.9999 | 0.9977 | -0.0022 |
| Distance | 0.9994 | 0.9986 | -0.0008 |
| Slope | 0.9966 | 0.9827 | -0.0139 |
| Aspect | 0.9999 | 0.9999 | +0.0000 |
| Width | 0.9419 | 0.9471 | +0.0052 |
| Area fraction | 0.8174 | 0.8284 | +0.0110 |

Height and slope correlations decrease slightly while width and area fraction
improve. The net effect was considered acceptable -- matching Swenson's actual
code is more important than optimizing correlations against published data
that may have been generated with a different pipeline version.

**Previous analysis (pre-basin-chain, "Test N"):** Results were similar but
from a different baseline. The interpretation about "paradoxical" behavior
(original DEM giving better height/slope) was correct -- the published data
likely used a slightly different DEM conditioning path. Now that we match
Swenson's code exactly, the remaining correlation gaps reflect inherent
differences between our stream network and Swenson's (different A_thresh,
different domain expansion), not algorithmic divergences.

### DEM Conditioning Notes

**Switch compute_hand to inflated DEM** (dem_conditioning_todo item 3):
Swenson passes inflated_dem to `compute_hand` (rh:1685). The inflated
(flat-resolved) DEM is the correct surface for HAND computation because it's
consistent with the flow direction surface. Implemented in commit 01a8532.

**Stream network extraction + slope** (dem_conditioning_todo item 5):
`extract_river_network` (rh:1608-1614) and `river_network_length_and_slope`
(rh:1655-1673) added in commit 712f6c5. Stream slope is one of the hardcoded
parameters in the OSBS pipeline (STATUS.md problem #6) -- computing it from
the DEM provides a physically motivated value instead of a guess. No change
to correlations (informational output only).

### Deep Audit Findings: Sections 4, 5, 6 (2026-02-20)

Line-by-line comparison of Swenson's `representative_hillslope.py`
(rh:1575-1703) and `pgrid.py` key functions (`flowdir`, `accumulation`,
`slope_aspect`, `compute_hand`, `create_channel_mask`, `compute_hillslope`,
`extract_river_network`, `river_network_length_and_slope`) against our
`merit_regression.py` (mr:340-740). Findings A-J supplement the
divergence tables in Sections 4, 5, and 6 above.

**Finding A: pysheds API — string names vs array passing (equivalent)**

Swenson passes arrays directly to pysheds methods:
```python
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)       # rh:1578
acc = grid.accumulation(fdir, dirmap=dirmap)            # rh:1586
hand, dtnd, drainage_id = grid.compute_hand(fdir, inflated_dem, ...)  # rh:1685
```

Ours passes string view names:
```python
grid.flowdir("inflated", out_name="fdir", dirmap=DIRMAP, routing="d8")   # mr:646
grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")   # mr:660
grid.compute_hand("fdir", "inflated", ..., dirmap=DIRMAP, routing="d8")  # mr:715
```

Both are valid pysheds API patterns. The string-name API resolves the
view internally via `grid.__getattr__`. The array-passing API uses the
array directly. Functionally equivalent — same data reaches the same
computation. Our pattern is more explicit about where results are stored.

**Finding B: routing="d8" and dirmap explicit vs default (benign)**

Our code explicitly passes `routing="d8"` to every pysheds call
(mr:646, 660, 681, 687, 702, 715, 729). Swenson omits it (default is
`"d8"` in all pgrid methods). Similarly, we pass `dirmap=DIRMAP` where
`DIRMAP` is the standard pysheds 8-direction map; Swenson passes
`dirmap=dirmap` where `dirmap` comes from `_set_dirmap()` (the same
default). No functional difference.

**Finding C: MemoryError handling — continue vs abort (behavioral
divergence, dormant)**

Swenson aborts the gridcell on MemoryError:
```python
except MemoryError:
    warning("Memory Error in extract_river_network, skipping")
    return -1    # rh:1612-1614
```
Same pattern at rh:1662-1664 for `river_network_length_and_slope`.

Ours continues with degraded output:
```python
except MemoryError:
    print("  WARNING: MemoryError in extract_river_network, skipping")
    branches = None    # mr:691-693
```
Same pattern at mr:710-712 for `river_network_length_and_slope`.

Swenson's `return -1` skips the entire gridcell (the caller checks for
this). Ours sets the result to `None` and continues computing the
remaining parameters. Neither triggers at this gridcell size. Would
matter for very large domains where the GeoJSON construction exhausts
memory.

Swenson's approach is more conservative (no partial output). Ours is
more forgiving (stream stats are informational, not used by the 6-param
computation). Both are reasonable — the stream network extraction and
length/slope computation are independent of HAND/DTND/hillslope
classification.

**Finding D: Section 3 Finding G correction — nanmax/max reversed
(factual correction)**

The previous Section 3 Finding G (from the 2026-02-20 Sections 2-3
audit) had both sides reversed:
- Claimed Swenson uses `nanmax` → **actually uses `np.max`** (rh:1597)
- Claimed we use `np.max` → **actually use `np.nanmax`** (mr:674)

Verified against source. Swenson's logic (rh:1597-1600):
```python
if np.max(acc) > accum_thresh:
    self.thresh = accum_thresh
else:
    self.thresh = np.max(acc) / 100
```

Ours (mr:674-677):
```python
if np.nanmax(acc_arr) < accum_threshold:
    accum_threshold = int(np.nanmax(acc_arr) / 100)
```

Equivalent logic (inverted condition), but our `nanmax` is the
defensive improvement — not Swenson's. The recommendation "should change
to `nanmax` for OSBS" was moot; already done.

**Section 3 Finding G has been corrected in place** (see above).

**Finding E: Section 5 — no new findings (confirmed match)**

Sections 4, 5, and 6 were audited as a combined block because they are
tightly coupled. Section 5 (Slope and Aspect) had no findings beyond
what was already documented:

- Both use `slope_aspect()` on the original DEM (items #18, #19)
- The redundant second `slope_aspect` call in Swenson (rh:1593 vs
  rh:1547) was already documented in Section 3 Finding E
- Horn 1981 stencil is identical (verified in Section 5 description)

No new divergences.

**Finding F: Operation reorder — create_channel_mask before
extract_river_network (benign)**

Swenson's order (rh:1608-1679):
1. `extract_river_network` (rh:1609)
2. Stream network post-processing (rh:1616-1646)
3. `river_network_length_and_slope` (rh:1658)
4. `create_channel_mask` (rh:1679)

Our order (mr:681-722):
1. `create_channel_mask` (mr:681)
2. `extract_river_network` (mr:686)
3. `river_network_length_and_slope` (mr:702)
4. `compute_hand` (mr:715)

We call `create_channel_mask` first (before river network extraction).
This is safe because `create_channel_mask` depends only on `fdir` and
`acc_mask`, not on the river network output. And `compute_hand` needs
`channel_mask` and `channel_id` from `create_channel_mask`, so calling
it earlier is logically cleaner. No functional difference — all
operations read from the same `fdir` and `acc_mask` inputs.

**Finding G: river_network_length_and_slope — implicit attribute access
(fragile but correct)**

`river_network_length_and_slope` signature (pgrid:3170-3171):
```python
def river_network_length_and_slope(self, fdir, mask, dirmap=None, ..., **kwargs):
```

No `dem` or `acc` parameter. Internally accesses `self.inflated_dem`
(pgrid:3269) and `self.acc` (pgrid:3284) by hardcoded attribute name.

Swenson's call (rh:1658):
```python
grid.river_network_length_and_slope(dem=inflated_dem, fdir=fdir,
    acc=acc, mask=acc_mask, dirmap=dirmap)
```
The `dem=` and `acc=` keyword arguments are captured by `**kwargs` and
**silently ignored**. The function uses `self.inflated_dem` and
`self.acc` regardless of what's passed.

Our workaround (mr:699):
```python
grid.inflated_dem = grid.inflated  # alias for pgrid compat
```
This sets `self.inflated_dem` to point at the same array as
`self.inflated`, so the hardcoded attribute access at pgrid:3269 finds
the correct data.

Correct but fragile — any pgrid attribute rename would break silently
(Swenson's kwargs would still be ignored; our alias would point to a
stale name). The existing comment at mr:696-698 documents this.

**Finding H: compute_hand returns HAND only; stores DTND/drainage_id
as attributes (API divergence, equivalent)**

Our fork's `compute_hand` (pgrid:1794-2047):
- Returns only HAND via `_output_handler` (pgrid:2046-2047)
- Stores DTND as `grid.dtnd` via `_output_handler(data=dtnd,
  out_name='dtnd', ...)` (pgrid:2010)
- Stores drainage_id as `grid.drainage_id` via `_output_handler(
  data=drainage_id, out_name='drainage_id', ...)` (pgrid:1961)
- Stores AZND as `grid.aznd` via `_output_handler(data=aznd,
  out_name='aznd', ...)` (pgrid:2040)

Swenson's original pysheds API (rh:1685):
```python
hand, dtnd, drainage_id = grid.compute_hand(fdir, inflated_dem,
    channel_mask, channel_id, dirmap=dirmap)
```
Destructures 3 return values — the original API before our fork
modified it.

Our code (mr:715-724) correctly uses the attribute access pattern:
```python
grid.compute_hand("fdir", "inflated", grid.channel_mask,
    grid.channel_id, dirmap=DIRMAP, routing="d8")
hand = grid.hand
dtnd = grid.dtnd
```

This is a consequence of our fork's Phase A modifications. The
`_output_handler` pattern stores results as grid attributes rather than
returning them directly. Functionally equivalent — same arrays, accessed
differently.

**Finding I: Swenson's stream network post-processing omitted (benign)**

Swenson (rh:1616-1646) unpacks the GeoJSON river network into arrays:
```python
branch_id = np.zeros((nreach))
branch_xy = np.zeros((nreach, 2))
for branch in network:
    for pt in branch["geometry"]["coordinates"]:
        branch_id[n] = branch["id"]
        branch_xy[n] = pt
```
Then stores as `self.network`, `self.network_id`, `self.nreach`,
`self.nstreams`. Also determines latitude direction (`latdir`).

Ours (mr:689): Just counts features for logging:
```python
n_streams = len(branches["features"])
```

These arrays are used by Swenson for visualization and NetCDF coordinate
output, not for any of the 6 hillslope parameter computations. The
latitude direction is used internally by `river_network_length_and_slope`
(which reads it from the coordinate arrays directly). Our omission has
no effect on parameter validation.

**Finding J: AZND computed but unused (informational)**

Our fork's `compute_hand` computes AZND (azimuth to nearest drainage)
at pgrid:2012-2040 and stores it as `grid.aznd`. Neither Swenson's
parameter computation nor ours uses AZND for any of the 6 hillslope
parameters. It's a byproduct of the distance computation (uses the same
dx/dy displacement vectors as DTND).

Swenson's original code also computed AZND. Our fork added the projected
CRS branch (pgrid:2027-2037) for completeness alongside the DTND
projected branch.

---

## 7. Hillslope Classification and Aspect Averaging

### Swenson

1. **Hillslope classification:** `grid.compute_hillslope(fdir, channel_mask,
   bank_mask)` (rh:1692) -- assigns 1=headwater, 2=right bank, 3=left bank,
   4=channel
2. **Catchment-level aspect averaging:**
   `set_aspect_to_hillslope_mean_parallel(drainage_id, aspect, hillslope)`
   (rh:1725) -- for each unique `(drainage_id, hillslope_type)` combination,
   computes circular mean of all pixel aspects, then replaces all pixel aspects
   with that mean (tu:147-171). Channel pixels (type 4) are combined with each
   other type.

**This smoothed aspect is used for ALL downstream binning.** The per-pixel
aspect values are overwritten at rh:1751.

### Ours

1. `grid.compute_hillslope("fdir", "channel_mask", "bank_mask", dirmap=DIRMAP)`
   (mr:728-729) -- classifies every pixel as headwater (1), right bank (2),
   left bank (3), or channel (4). Matches Swenson rh:1692.
2. `catchment_mean_aspect(drainage_id, aspect, hillslope)` (mr:737-739) --
   for each `(drainage_id, hillslope_type)` pair, computes circular mean of
   all pixel aspects and replaces individual values with group mean. Channel
   pixels (type 4) are merged with each hillslope side. Matches Swenson's
   `set_aspect_to_hillslope_mean_parallel()` (rh:1725, tu:174-233).

**This smoothed aspect is used for ALL downstream binning** -- the per-pixel
aspect values are overwritten before gridcell extraction.

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 20 | Catchment-level aspect averaging | MATCH | Yes | `catchment_mean_aspect()` ported; area_frac 0.8284->0.9047 |

**#20 — Catchment-Level Aspect Averaging (RESOLVED)**

**Swenson:** After computing per-pixel aspect via Horn 1981, replaces all pixel
aspects with their catchment-side circular mean:
`set_aspect_to_hillslope_mean_parallel(drainage_id, aspect, hillslope)`
(rh:1725, tu:174-233). For each `(drainage_id, hillslope_type)` pair, computes
the circular mean of all pixel aspects and sets every pixel in that group to
that value. Channel pixels (type 4) are combined with each hillslope type.

**Ours:** Now matches Swenson. `catchment_mean_aspect()` (mr:346-413) ports
`set_aspect_to_hillslope_mean_serial` (tu:236-279), the serial-chunked
variant. Called at mr:737-739 after `compute_hillslope()` classifies pixels.

**Resolution (commit b4e9ff0, 2026-02-20):** Ported as `catchment_mean_aspect()`
inline in `merit_regression.py`. Uses the serial chunked algorithm: for each
`drainage_id`, for each hillslope type (1-3), compute circular mean aspect of
that group plus channel pixels (type 4).

**Impact (measured):**

| Parameter | Before | After | Delta |
|-----------|--------|-------|-------|
| Height | 0.9977 | 0.9977 | +0.0000 |
| Distance | 0.9986 | 0.9987 | +0.0001 |
| Slope | 0.9827 | 0.9850 | +0.0023 |
| Aspect | 0.9999 | 1.0000 | +0.0001 |
| Width | 0.9471 | 0.9465 | -0.0006 |
| Area fraction | 0.8284 | 0.9047 | **+0.0763** |

Area fraction improved substantially -- the largest single-change improvement
in the entire audit. Catchment averaging stabilizes N/E/S/W bin assignments by
replacing noisy per-pixel aspects with catchment-side means, preventing pixels
near bin boundaries from landing in wrong bins.

**Previous analysis (pre-resolution):** The pre-implementation hypothesis that
this would be "small at 90m" was wrong. Even at 90m, enough pixels sit near
aspect bin boundaries (44/45 deg, 134/135 deg, etc.) that catchment averaging moves
them into the correct bin, substantially improving area fractions.

### DEM Conditioning Notes

**Catchment-level aspect averaging** (dem_conditioning_todo item 4):
Full implementation details including the serial-chunked algorithm and
`compute_hillslope` prerequisites are documented in `dem_conditioning_todo.md`
item 4. Ported as `catchment_mean_aspect()` in commit b4e9ff0.

### Deep Audit (2026-02-20)

Line-by-line comparison of hillslope classification and aspect averaging.
Swenson: tu:174-279 (`set_aspect_to_hillslope_mean_parallel`).
Ours: mr:373-414 (`catchment_mean_aspect`).

| # | Finding | Label | Impact |
|---|---------|-------|--------|
| A | Missing empty `uid` early return guard | Dormant | None at this gridcell |

**A: Missing empty `uid` guard.** Swenson's parallel version (tu:184-185)
has `if uid.size == 0: return`. Our `catchment_mean_aspect` (mr:373-414)
has no such guard. If `uid.size == 0`, the chunking math would produce
`cs = min(500, -1) = -1` and the loop body would attempt `uid[0]` on an
empty array (IndexError). Dormant: pysheds always produces non-empty
`drainage_id` after `compute_hand`.

---

## 8. Pixel Area

### Swenson

Spherical formula (rh:1708-1715):
```python
phi = dtr * lon
th = dtr * (90.0 - lat)
dphi = abs(phi[1] - phi[0])
dth = abs(th[0] - th[1])
farea = tile(sin(th), (im, 1)).T
area = farea * dth * dphi * re**2
```

### Ours

`compute_pixel_areas(lon, lat)` (mr:342-352):
```python
phi = DTR * lon
theta = DTR * (90.0 - lat)
dphi = abs(phi[1] - phi[0])
dtheta = abs(theta[0] - theta[1])
area = tile(sin(theta), (1, ncols)) * dtheta * dphi * RE**2
```
Same spherical formula as Swenson.

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 24 | Spherical formula | MATCH | -- | -- |

### Shared Functions

**compute_pixel_areas** — Swenson: rh:1708-1715. Ours: mr:342-352. **MATCH.**
Both: `sin(colatitude) * d_theta * d_phi * re^2`.

---

## 9. Gridcell Extraction

### Swenson

Extract interior arrays from expanded grid using `arg_closest_point()` (rh:547-567).
If subregions were split, aggregate via `extend()` (rh:599-609).

### Ours

Extract from expanded grid using precomputed row/col slices (mr:440-448).
Compute coordinate arrays, flatten to 1D (mr:451-455).

### Divergences

None. No divergence items for this stage.

---

## 10. Data Filtering

### Swenson

Applied in order:
1. **Basin masking** (optional, `flagBasins=True`): `identify_basins()` on
   extracted DEM, remove basin pixels (rh:570-596)
2. **NaN removal:** `ind = where(isfinite(fhand))` (rh:653) -- removes NaN-HAND
   pixels; does NOT filter on `hand >= 0`
3. **DTND tail removal** (`removeTailDTND=True`): `TailIndex(fdtnd, fhand)`
   (tu:286-296) -- exponential fit, remove pixels where PDF < 5% of max
   (rh:666-676)
4. **Flood filter:** Threshold sweep on `fflood` (= `basin_mask`) for
   `HAND < 2m` pixels; sets HAND = -1 for flooded pixels (rh:678-697)
5. **DTND minimum clipping:** `fdtnd[fdtnd < 1.0] = 1.0` (rh:700-701)

### Ours

**Valid mask:** `valid = (hand_flat >= 0) & isfinite(hand_flat)` (mr:456)

**Flood filter:** Basin pixels with HAND < 2m marked invalid before binning
(mr:590). Implemented 2026-02-20 as part of basin/water handling chain.

**Not implemented:** DTND tail removal, DTND min clipping, basin masking at
extraction (tested as harmful/no-effect/no-op respectively).

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 25 | NaN removal (isfinite vs hand>=0) | MATCH | Yes (F) | Negligible. Fixed 2026-02-20. |
| 26 | **DTND tail removal** | **OMISSION** | Yes (A) | Harmful (-0.010) |
| 27 | Flood filter | MATCH | Yes (C) | No-op at 90m. Implemented 2026-02-20. |
| 28 | DTND min clipping | MATCH | Yes (B) | No effect. Fixed 2026-02-20. |

**#25 — Valid Mask (DIVERGENCE)**

Swenson: `isfinite(fhand)` only (rh:653). Does NOT filter on `hand >= 0`.
Ours: `(hand_flat >= 0) & isfinite(hand_flat)` (mr:456). Adds `>= 0` check.

Tested (Test F): Negligible impact (-0.0001 on area fraction). Essentially
no pixels have finite negative HAND at this gridcell.

**#26 — DTND Tail Removal (OMISSION)**

Swenson: `TailIndex(fdtnd, fhand)` (tu:286-296) removes pixels at the far end
of the DTND distribution where the exponential PDF drops below 5% of max.

Tested (Test A): Harmful -- area fraction drops -0.010, width drops -0.054.
Removing long-distance pixels biases the trapezoidal fit. Intentionally omitted.

**#28 — DTND Min Clipping (OMISSION)**

Swenson: `fdtnd[fdtnd < 1.0] = 1.0` (rh:700-701).

Tested (Test B): No effect. Few or no DTND values below 1.0 at this resolution.
Intentionally omitted.

---

## 11. HAND Binning

### Swenson

`SpecifyHandBounds(fhand, faspect, aspect_bins, bin1_max=2,
BinMethod="fastsort")` (tu:299-412, called at rh:727).

**fastsort method** (tu:350-408):
1. Sort `fhand[fhand > 0]`, compute quartiles at [0.25, 0.5, 0.75, 1.0]
2. If `Q25 > bin1_max` (2m): **forced branch**
   - Check each aspect has at least 1% of pixels below `bin1_max` (tu:368-395)
   - If any aspect's 1%-ile > `bin1_max`, raise `bin1_max` (tu:388-395)
   - Remaining bins: 33rd/66th percentiles of `hand > bin1_max` (tu:397-408)
   - Bounds: `[0, bin1_max, b33, b66, 1e6]`
3. If `Q25 <= bin1_max`: **quartile branch**
   - Bounds: `[0, Q25, Q50, Q75, 1e6]`

### Ours

**HAND binning:** `compute_hand_bins()` (mr:188-246) -- same fastsort algorithm
as `SpecifyHandBounds`:
- Sort `hand[hand > 0 & isfinite]`
- If `Q25 > bin1_max`: forced branch with aspect-aware minimum check
- Else: quartile branch

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 29 | SpecifyHandBounds algorithm | MATCH | -- | -- |
| 30 | fastsort method | MATCH | -- | -- |
| 31 | bin1_max=2 constraint | MATCH | -- | -- |
| 32 | Aspect-aware minimum | MATCH | -- | -- |

### Shared Functions

**SpecifyHandBounds / compute_hand_bins** — Swenson: tu:299-412.
Ours: mr:188-246. **MATCH.**
After the fix to align with Swenson's `SpecifyHandBounds()`, these are
functionally identical:
- Same fastsort method: sort `hand[hand > 0]`, compute quartiles
- Same forced/quartile branch selection at `Q25 > bin1_max`
- Same aspect-aware minimum check in forced branch
- Same `b33 == b66` deduplication

---

## 12. Trapezoidal Width Fitting

### Swenson

`calc_width_parameters(fdtnd[aind], farea[aind] / n_hillslopes, mindtnd=ares,
nhisto=10)` (rh:766-775).

Internals (rh:54-110):
1. Create 10 DTND bins from `[mindtnd, max(dtnd)+1]` (rh:63)
2. Cumulative area: `A(d) = sum(area[dtnd >= d])` (rh:67-70)
3. Prepend d=0, total area if `mindtnd > 0` (rh:73-75)
4. Weighted polynomial fit: `_fit_polynomial(d, A, ncoefs=3, weights=A)` (rh:78)
   -- **w^1 weighting** via `G^T * diag(A) * G` (rh:130-131)
5. Extract: `trap_slope = -coefs[2]`, `trap_width = -coefs[1]`,
   `trap_area = coefs[0]` (rh:83-85)
6. **Width guard:** If `trap_slope < 0`, compute `Atri = -(width^2)/(4*slope)`;
   if `Atri < trap_area`, adjust: `width = sqrt(-4 * slope * area)` (rh:91-94)

### Ours

Per aspect (mr:465-589):
1. **Aspect mask:** `get_aspect_mask()` & valid (mr:466)
2. **n_hillslopes:** From `drainage_id` unique count (mr:483-492)
3. **Trapezoidal fit:** `fit_trapezoidal_width()` (mr:249-308) -- same formula
   as Swenson with w^1 weighting (mr:286-291)

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 33 | 10-bin DTND histogram | MATCH | -- | -- |
| 34 | Cumulative area (reverse CDF) | MATCH | -- | -- |
| 35 | Weighted polynomial (w^1) | MATCH | After fix | -- |
| 36 | Width guard | MATCH | -- | -- |

**#35 — Polynomial Weighting (MATCH after fix)**

Originally used w^2 weighting (our `lstsq`-based implementation applied
`diag(A^2)`). Fixed to match Swenson's w^1 normal equations (`G^T * diag(A) * G`).
Tested (Test I): Area fraction +0.002, width -0.018. Small impact but it was
a genuine code-level bug.

### Shared Functions

**calc_width_parameters / fit_trapezoidal_width** — Swenson: rh:54-96.
Ours: mr:249-308. **MATCH (after w^1 fix).**
Both use:
1. 10 DTND bins from `[min_dtnd, max(dtnd)+1]`
2. Reverse cumulative area: `A(d) = sum(area[dtnd >= d])`
3. Prepend d=0, total area if `min_dtnd > 0`
4. w^1 weighted polynomial: `G^T * diag(weights) * G`
5. Same width guard: `if slope < 0 and Atri < Atrap: width = sqrt(-4*slope*area)`

**quadratic** — Swenson: gu:168-188. Ours: mr:311-329. **MATCH.**
Identical: solve `ax^2 + bx + c = 0` with discriminant check and `ck` adjustment
for near-zero discriminant (`eps=1e-6`). Returns selected root.

---

## 13. Element Computation

### Swenson

For each aspect bin (rh:736-920), for each HAND bin (rh:811-920):

1. **Bin selection:** `hind = (hand[aind] >= b1) & (hand[aind] < b2)` (rh:814)
2. **Mean-HAND skip:** `if mean(fhand[cind]) <= 0: continue` (rh:819)
3. **Height:** `mean(fhand[cind])` (rh:823)
4. **Median DTND (raw):** `dtnd_sorted[int(0.5*size - 1)]` (rh:825-828) -- later
   overwritten by quadratic
5. **Slope:** `mean(tmp[isfinite(tmp)])` where `tmp = fslope[cind]` (rh:831-832)
6. **Area:** `trap_area * (sum(farea[cind]) / sum(farea[aind]))` (rh:836-837)
7. **Width (quadratic):** cumulative fitted area up to current bin ->
   `le = quadratic([trap_slope, trap_width, -da])` ->
   `we = trap_width + le * trap_slope * 2` (rh:840-845)
8. **Distance (quadratic):** cumulative fitted area to midpoint ->
   `ld = quadratic([trap_slope, trap_width, -da])` (rh:847-859)
9. **Aspect:** Circular mean via `arctan2(mean(sin), mean(cos))` (rh:894-902)

### Ours

Per aspect (mr:465-589):
1. **First pass -- raw areas per bin** (mr:502-528)
2. **Area fractions and fitted areas** (mr:522-528)
3. **Second pass -- 6 parameters per bin** (mr:530-588):
   - `height = mean(hand)` (mr:548)
   - `slope = nanmean(slope)` (mr:549)
   - `aspect = circular_mean_aspect()` (mr:550)
   - `width = quadratic([trap_slope, trap_width, -da_width])` -> width at lower
     edge (mr:556-564)
   - `distance = quadratic([trap_slope, trap_width, -da_dist])` -> midpoint
     distance (mr:570-577)
   - `area = trap_area * fraction` (mr:553)

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 37 | height = mean(HAND) | MATCH | -- | -- |
| 38 | slope = nanmean(slope) | MATCH | -- | -- |
| 39 | aspect = circular mean | MATCH | -- | -- |
| 40 | area = trap_area x fraction | MATCH | -- | -- |
| 41 | width = quadratic at lower edge | MATCH | -- | -- |
| 42 | distance = quadratic at midpoint | MATCH | -- | -- |
| 43 | Mean-HAND bin skip | MATCH | Yes (D) | No-op for this cell. Fixed 2026-02-20. |

**#43 — Mean-HAND Bin Skip (OMISSION)**

Swenson: `if mean(fhand[cind]) <= 0: continue` (rh:819) -- skips bins where
mean HAND is non-positive.

Ours: Not implemented.

Tested (Test D): No-op -- no bins have mean HAND <= 0 at this gridcell.

### Shared Functions

**circular_mean_aspect** — Swenson: inline at rh:894-903. Ours: mr:332-339.
**MATCH.** Both: `arctan2(mean(sin(dtr*asp)), mean(cos(dtr*asp))) / dtr`,
wrap to [0,360].

---

## 14. Post-processing

### Swenson

1. **Column compression:** Remove unused slots (rh:972-1012)
2. **Minimum aspect validation:** If `< 3` unique hillslope indices, zero all
   parameters (rh:1015-1036)
3. **Stream channel parameters:** `depth = 1e-3 * area^0.4`,
   `width = 1e-3 * area^0.6`, `slope = mean_network_slope` (rh:1104-1114)
4. **NetCDF output:** Full variable set with dimensions
   `(nmaxhillcol, lsmlat, lsmlon)` (rh:1117-1399)

### Ours

- No column compression or minimum-aspect validation (not needed for regression)
- No stream channel parameters (not part of validation)
- Comparison: standard Pearson correlation for height, distance, slope, width.
  Circular correlation for aspect. Area as fractions (mr:620-688).

### Divergences

| # | Item | Status | Tested? | Impact |
|---|------|--------|---------|--------|
| 44 | Column compression | N/A | -- | -- |
| 45 | Minimum 3-aspect validation | N/A | -- | -- |
| 46 | Stream channel params | N/A | -- | -- |
| 48 | **Basin masking at extraction (flagBasins)** | **OMISSION** | No | See analysis below |

Items 44-46 are post-processing steps not relevant to the 6-parameter
validation. They would be needed for a production pipeline generating
CTSM-compatible NetCDF output.

**#48 — Basin Masking at Extraction (OMISSION)**

**Swenson:** Optional (`flagBasins=True`): After extracting gridcell arrays,
runs `identify_basins()` on the DEM and removes basin pixels from all arrays
(rh:570-596). Requires >1% non-flat pixels.

**Ours:** Not implemented.

**Tested?** Partially -- Test N showed `identify_basins()` finds 0 pixels at
this gridcell. However, the `flagBasins` parameter is set per-run, and it's
unclear whether Swenson's published data was generated with it enabled.

**Impact:** None for this gridcell (no basins detected). Would matter for flat
terrain like OSBS where wetland depressions would be flagged.

### Deep Audit of Sections 8-14 (2026-02-20)

Line-by-line comparison of pixel area, gridcell extraction, data
filtering, HAND binning, trapezoidal width fitting, element computation,
and post-processing. Swenson: rh:54-136, 340-400, 547-706, 735-920;
tu:174-279, 286-413; gu:115-188. Ours: mr:175-415, 570-913;
spatial_scale.py:143-163.

16 findings (A-P). One real bug (M). Rest are dormant divergences,
equivalent approaches, or defensive improvements.

| # | Section | Finding | Label |
|---|---------|---------|-------|
| A | 8 | `np.tile` axis syntax differs | Equivalent |
| B | 9 | Extraction: `arg_closest_point` vs affine slicing | Equivalent |
| C | 10 | NaN pre-filtering vs inclusive arrays | Dormant |
| D | 10 | Flood filter denominator: total vs basin-only | Dormant |
| E | 11 | Quartile upper bound: `max(hand)` vs `1e6` | Negligible |
| F | 11 | Aspect mask on NaN-filtered vs full array | Dormant |
| G | 11 | Empty `above_bin1`: negative index vs size guard | Equivalent |
| H | 12 | `max(dtnd) < min_dtnd`: ValueError vs fallback | Dormant |
| I | 12 | Linear algebra: `inv + dot` vs `solve` | Equivalent (ours better) |
| J | 12 | Width minimum clamp `max(width, 1)` | Dormant |
| K | 12 | Dead code: duplicate width guard at rh:782-785 | Informational |
| L | 12 | Exception handling around fit | Defensive improvement |
| **M** | **13** | **n_hillslopes: wrong array indexing** | **BUG (FIXED 2026-02-20)** |
| N | 13 | Median DTND index off by 1 | Dormant |
| O | 13 | Empty bin: `continue` vs explicit zero dict | Equivalent |
| P | 13 | Quadratic fallbacks in ours, Swenson crashes | Defensive improvement |

**A: `np.tile` axis syntax (Section 8).** Swenson (rh:1713):
`tile(sin(th), (im, 1)).T` — tiles columns then transposes. Ours
(mr:349): `tile(sin(theta), (1, ncols))` — tiles directly along
columns. Produces the same (nrows, ncols) area array. Equivalent.

**B: Gridcell extraction method (Section 9).** Swenson uses
`arg_closest_point(lon, lat, ...)` (gu:115-148) to find nearest row/col
indices from coordinates. Ours uses precomputed affine-transform slices
(mr:440-448). Both extract the same sub-array. Equivalent.

**C: NaN pre-filtering (Section 10).** Swenson (rh:647-653) removes
NaN-HAND pixels via `isfinite` from ALL arrays before passing to
`SpecifyHandBounds`. The arrays `fhand`, `fdtnd`, `fslope`, `faspect`,
`fdid` are all NaN-filtered together. Ours passes full gridcell arrays
to `compute_hand_bins`, which internally filters with
`valid = (hand > 0) & np.isfinite(hand)`. The aspect mask check at
mr:220 (`hand[asp_mask]`) includes zeros and NaN, but `np.sort` puts
NaN at end so the 1%-ile from the sorted front is unaffected.
Functionally equivalent. Dormant.

**D: Flood filter denominator (Section 10).** Swenson (rh:576):
`non_flat_fraction = ind.size / fhand.size` — ratio of non-basin pixels
to ALL gridcell pixels. If >99% is basin, skip. Ours (mr:764):
`np.sum(flooded_low_hand) / n_flooded_gc < 0.95` — ratio of
flooded-with-low-HAND pixels to BASIN pixels only. Different logic
(threshold sweep vs binary fraction). Dormant for binary basin masks
(0/1 values). Would diverge for graded basin masks, but neither
codebase uses those.

**E: Quartile upper bound (Section 11).** Swenson's quartile branch
(tu:354-360) returns `[0, Q25, Q50, Q75, max(hand)]`. Last bin:
`hand >= Q75 AND hand < max(hand)` — excludes pixel(s) at exactly
max(hand). Ours (mr:242-248) returns `[0, Q25, Q50, Q75, 1e6]`. Last
bin: `hand >= Q75 AND hand < 1e6` — includes all pixels. Difference: a
handful of pixels at exactly max(hand). Negligible impact.

**F: Aspect mask on NaN-filtered vs full array (Section 11).** Swenson
applies the aspect mask to NaN-filtered arrays (finding C above means
no NaN in `fhand`, `faspect`, etc.). Ours applies the aspect mask
to the full gridcell arrays which may contain NaN. The NaN values
are excluded later by the valid-pixel filters in binning. Dormant:
same pixels end up in each bin.

**G: Empty `above_bin1` handling (Section 11).** Swenson (tu:309-313):
If `above_bin1` is empty, `Q25 = fhand[0]` — index 0 of an empty array
would crash, but the `if len(above_bin1) < 4` guard at tu:346
catches this first and forces the quartile branch. Ours (mr:228-230):
`if len(above_bin1) == 0`, explicitly sets `bin1_max = hand_max`.
Both reach the same quartile fallback. Equivalent.

**H: `max(dtnd) < min_dtnd` fallback (Section 12).** Swenson (rh:745):
If `max(dtnd[aind]) < min_dtnd`, skips this aspect entirely (no width
fit). Ours (mr:800): No explicit check. If max DTND is below the
minimum, the trapezoidal fit receives a degenerate range and `solve`
may produce nonsensical coefficients. Dormant: min_dtnd = 1.0 after
DTND clipping, and DTND is always >= 1.0 post-clipping. Would only
matter if clipping were removed.

**I: Linear algebra (Section 12).** Swenson (rh:133-134):
`covm = np.linalg.inv(gtg); coefs = dot(covm, gtd)` — explicit matrix
inversion. Ours (mr:295): `coeffs = np.linalg.solve(GtWG, GtWy)` —
direct solve without inversion. `solve` avoids explicit matrix
inversion and is numerically more stable for near-singular systems.
Same results for well-conditioned 2x2 matrices. Ours is the
improvement.

**J: Width minimum clamp (Section 12).** Swenson (rh:779):
`width = max(width, 1)` — clamps fitted base width to minimum 1.
Ours: No explicit clamp on `trap_width`. The quadratic solver handles
negative widths via the discriminant check. Dormant: the fit almost
always produces positive widths for real terrain. Would matter for
degenerate catchments with very few pixels.

**K: Dead code in Swenson (Section 12).** Swenson (rh:782-785):
```python
if slope < 0 and Atri < Atrap:
    width = sqrt(-4 * slope * area)
```
This duplicates the same guard already applied at rh:136.
Informational only.

**L: Exception handling around fit (Section 12).** Swenson: No
try/except around the trapezoidal fit. If `inv` fails on a singular
matrix, the gridcell crashes. Ours (mr:788-798): Wraps the fit in
try/except, falls back to uniform width on failure. Defensive
improvement.

**M: n_hillslopes — wrong array indexing (Section 13). BUG (dormant).**

Swenson (rh:559, 761):
```python
fdid = lc.drainage_id[j1:j2, i1:i2].flatten()  # extract gridcell
# ... later ...
number_of_hillslopes[asp_ndx] = np.unique(fdid[aind]).size
```
Correctly extracts gridcell region from `drainage_id`, then indexes
with `aind` (indices into the extracted gridcell arrays).

Ours (mr:806-815):
```python
n_hillslopes = max(
    len(np.unique(
        grid.drainage_id.flatten()[asp_indices]  # FULL expanded grid
    )), 1)
```
`asp_indices` are indices into `hand_gc.flatten()` (gridcell-level,
~1.4M elements). `grid.drainage_id.flatten()` is the full expanded
grid (~3.2M elements, different row width). Index 1366 in the
gridcell = position (1,0), but index 1366 in the expanded grid =
position (0,1366) — wrong spatial location.

**Impact:** Produces wrong unique `drainage_id` counts → wrong
`n_hillslopes` → wrong per-hillslope area normalization in the
trapezoidal fit. Dormant for correlation-based validation because
`n_hillslopes` is a uniform divisor within each aspect (ratios
preserved). Would affect absolute area values in production.

**Fix (for Phase D):** Extract `drainage_id` to gridcell first:
```python
drainage_id_gc = np.array(grid.drainage_id)[gc_row_slice, gc_col_slice].flatten()
n_hillslopes = max(len(np.unique(drainage_id_gc[asp_indices])), 1)
```

**N: Median DTND index off by 1 (Section 13).** Swenson (rh:826-828):
`dtnd_sorted[int(0.5 * dtnd_sorted.size - 1)]` — for N=10, index=4
(just below median). Ours (mr:875):
`dtnd_sorted[len(dtnd_sorted) // 2]` — for N=10, index=5 (just above
median). Both values are overwritten by the quadratic solver at
rh:858 / mr:896. Completely dormant.

**O: Empty bin handling (Section 13).** Swenson: `if mean(hand) <= 0:
continue` (rh:819) — skips bins, leaves dict entries unset. Ours:
explicit zero dict for empty bins (mr:843-858). Same effect: empty
bins produce zero parameters. Equivalent.

**P: Quadratic fallbacks (Section 13).** Swenson: If the quadratic
solver fails (negative discriminant, etc.), the code crashes — no
fallback. Ours (mr:560-564, 572-577): Wraps quadratic calls in
try/except, falls back to linear interpolation for width and midpoint
for distance. Defensive improvement — handles degenerate bins that
Swenson's code would crash on.

---

## Appendix A: Cross-Reference to area_fraction_research.md Tests

| Audit item | Test | Section |
|------------|------|---------|
| DEM conditioning (5 steps) | Test N | S7, "Test N" |
| inflated vs original DEM | Test N | S7, "Test N" |
| Valid mask | Test F | S4.6, "Test F" |
| DTND tail removal | Test A | S4.1, "Test A" |
| Flood filter | Test C | S4.3, "Test C" |
| DTND min clipping | Test B | S4.2, "Test B" |
| Mean-HAND bin skip | Test D | S4.4, "Test D" |
| Polynomial weighting | Test I | S4.9, "Test I" |
| bin1_max sensitivity | Test L | S4.12, "Test L" |
| A_thresh sensitivity | Test O | S7, "Test O" |

## Appendix B: Swenson Function Call Graph

```
CalcGeoparamsGridcell (rh:341)
+-- IdentifySpatialScaleLaplacian (ss:511)
|   +-- dem_reader()
|   +-- smooth_2d_array() [coastal only]
|   +-- fit_planar_surface() [if detrend]
|   +-- blend_edges()
|   +-- calc_gradient() x 4 (Laplacian)
|   +-- rfft2()
|   +-- _bin_amplitude_spectrum()
|   +-- _LocatePeak()
|       +-- _fit_polynomial() [linear]
|       +-- _fit_peak_gaussian()
|       +-- _fit_peak_lognormal()
|
+-- CalcLandscapeCharacteristicsPysheds (rh:1457)
|   +-- dem_reader()
|   +-- identify_basins() (gu:263)
|   +-- Grid.from_array()
|   +-- fill_pits()
|   +-- fill_depressions()
|   +-- slope_aspect(dem) -> identify_open_water() (gu:298)
|   +-- flooded_dem -= 0.1 [basin regions]
|   +-- resolve_flats()
|   +-- flowdir()
|   +-- accumulation()
|   +-- slope_aspect(dem) [for output]
|   +-- extract_river_network()
|   +-- river_network_length_and_slope()
|   +-- create_channel_mask()
|   +-- compute_hand(inflated_dem)
|   +-- compute_hillslope()
|   +-- set_aspect_to_hillslope_mean_parallel() (tu:174)
|
+-- [identify_basins() at extraction] [if flagBasins]
+-- [NaN removal, tail removal, flood filter, DTND clipping]
+-- SpecifyHandBounds (tu:299)
|
+-- [per-aspect loop]
|   +-- calc_width_parameters (rh:54)
|   |   +-- _fit_polynomial()
|   +-- [per-HAND-bin loop]
|       +-- mean(HAND), nanmean(slope), circular_mean(aspect)
|       +-- quadratic() x 2 (width, distance)
|
+-- [column compression, min-aspect check]
+-- CalcRepresentativeHillslopeForm [if CircularSection/TriangularSection]
+-- [stream channel params]
+-- [NetCDF output]
```

## Appendix C: Our Function Call Graph

```
main (mr:818)
+-- compute_lc (mr:112)
|   +-- load_dem_with_coords() x 4
|   +-- identify_spatial_scale_laplacian_dem() x 4 (our_ss:467)
|       +-- smooth_2d_array() [coastal only]
|       +-- fit_planar_surface() [if detrend]
|       +-- blend_edges()
|       +-- calc_gradient() x 4 (Laplacian)
|       +-- rfft2()
|       +-- _bin_amplitude_spectrum()
|       +-- _locate_peak()
|           +-- _fit_polynomial() [linear]
|           +-- _fit_peak_gaussian()
|           +-- _fit_peak_lognormal()
|
+-- compute_hillslope_params (mr:470)
|   +-- rasterio.open() + window read
|   +-- identify_basins() -> mask basins as nodata
|   +-- Grid() + add_gridded_data()
|   +-- fill_pits()
|   +-- fill_depressions()
|   +-- slope_aspect("dem") -> slope, aspect (for water detection + output)
|   +-- identify_open_water(slope) -> basin_boundary, basin_mask
|   +-- flooded -= 0.1 [basin regions]
|   +-- resolve_flats()
|   +-- flowdir()
|   +-- [re-mask basins in inflated DEM]
|   +-- accumulation()
|   +-- [force basin boundaries into acc]
|   +-- [A_thresh safety valve]
|   +-- create_channel_mask()
|   +-- extract_river_network()
|   +-- grid.inflated_dem = grid.inflated  [alias for pgrid compat]
|   +-- river_network_length_and_slope()
|   +-- compute_hand("inflated")
|   +-- compute_hillslope()
|   +-- catchment_mean_aspect()  -> overwrites per-pixel aspect
|   +-- compute_pixel_areas()
|   +-- [flood filter: basin pixels with HAND < 2m -> invalid]
|   +-- compute_hand_bins()
|   |
|   +-- [per-aspect loop]
|       +-- fit_trapezoidal_width()
|       +-- [per-HAND-bin loop]
|           +-- mean(HAND), nanmean(slope), circular_mean_aspect()
|           +-- quadratic() x 2 (width, distance)
|
+-- compare_to_published (mr:620)
    +-- [Pearson + circular correlation]
```

---

## Log

### 2026-02-19 — Initial audit

Created pipeline_audit.md with full line-by-line comparison of Swenson's
codebase against merit_regression.py. Catalogued 48 items: 26 MATCHes,
6 DIVERGENCEs, 10 OMISSIONs, 6 N/A.

### 2026-02-20 — Basin/water handling chain implemented

Ported 5 functions from `geospatial_utils.py` into `merit_regression.py`:
`_four_point_laplacian`, `_expand_mask_buffer`, `erode_dilate_mask`,
`identify_basins`, `identify_open_water`.

Integrated the full basin/water chain into `compute_hillslope_params()`:
1. `identify_basins()` on raw DEM before pysheds grid creation
2. `slope_aspect("dem")` moved before resolve_flats (serves water detection
   and final output)
3. `identify_open_water(slope)` detects coherent flat regions
4. Basin pixels lowered by 0.1m in flooded DEM to force flow through them
5. Basin pixels re-masked as NaN in inflated DEM after flowdir
6. Basin boundaries forced into stream network (acc set above threshold)
7. A_thresh safety valve (reduce threshold if max accumulation is too low)
8. Flood filter: basin pixels with HAND < 2m marked invalid before binning

**Regression result (job 25335704):** All confirmed as no-ops at 90m MERIT.
0 basin pixels detected, 0 open water pixels. All 6 correlations PASS:

| Parameter | Expected | Actual | Delta |
|-----------|----------|--------|-------|
| Height | 0.9999 | 0.9999 | -0.0000 |
| Distance | 0.9990 | 0.9994 | +0.0004 |
| Slope | 0.9966 | 0.9966 | -0.0000 |
| Aspect | 0.9999 | 0.9999 | -0.0000 |
| Width | 0.9410 | 0.9419 | +0.0009 |
| Area fraction | 0.8215 | 0.8174 | -0.0041 |

Small delta shifts (distance +0.0004, width +0.0009, area_fraction -0.0041)
are from moving `slope_aspect()` before `resolve_flats()` and re-adding
modified arrays to the grid — all within tolerance.

Updated divergence catalog: items 12-17 and 27 changed from OMISSION to
MATCH. New totals: 33 MATCHes, 5 DIVERGENCEs, 4 OMISSIONs, 6 N/A.

### 2026-02-20 — Three remaining DEM conditioning items resolved

Updated audit to reflect commits 01a8532, b4e9ff0, 712f6c5 which resolved
`dem_conditioning_todo.md` items 3, 4, and 5:

1. **#21 DIVERGENCE->MATCH:** `compute_hand` now passes `"inflated"` DEM
   (commit 01a8532, matching rh:1685).
2. **#20 OMISSION->MATCH:** `catchment_mean_aspect()` ported from
   `set_aspect_to_hillslope_mean_serial` (commit b4e9ff0, matching rh:1725-1751).
   Area fraction improved 0.8284->0.9047, the largest single-change improvement.
3. **#47 OMISSION->MATCH:** `extract_river_network` and
   `river_network_length_and_slope` added (commit 712f6c5, matching rh:1608-1673).

Updated sections: divergence table (Part 3), Part 2 pipeline description
(new S2.7 for hillslope classification + aspect averaging, renumbered
S2.8-2.13), Part 4 detailed analysis (S4.4, S4.5 marked resolved),
Part 6 summary counts, Appendix C call graph.

New totals: 38 MATCHes, 3 DIVERGENCEs, 4 OMISSIONs, 3 N/A. (Counts
recalculated from the table — the previous totals had a pre-existing
discrepancy with the table entries.)

Current expected correlations:

| Parameter | Correlation |
|-----------|-------------|
| Height | 0.9977 |
| Distance | 0.9987 |
| Slope | 0.9850 |
| Aspect | 1.0000 |
| Width | 0.9465 |
| Area fraction | 0.9047 |

### 2026-02-20 — Created full_pipeline_audit.md

Reorganized pipeline_audit.md from perspective-based structure (5 parts) into
phase-based structure (14 sections). Each section consolidates Swenson's
approach, ours, divergences, shared functions, and DEM conditioning notes.
All 48 divergence items preserved. Content reorganized, not rewritten.

### 2026-02-20 — Deep FFT section audit (Section 1)

Line-by-line comparison of Swenson's `spatial_scale.py`, `geospatial_utils.py`,
and `representative_hillslope.py` FFT calling convention against our
`spatial_scale.py` and `merit_regression.py`. Added 9 findings (A-I) as a
new "Deep Audit Findings" subsection in Section 1. Key findings:

- **A:** Exponential fit computed but unused in Swenson (benign omission)
- **B, F:** Our defensive guards against division-by-zero and log(0) (correct)
- **C:** Edge zeroing: our code matches Swenson's production path, not the
  testing-only DEM variant
- **D:** Coastal adjustment simplified (no-op for inland MERIT gridcell;
  Swenson's DEM variant has a NameError bug)
- **E:** detrendElevation default differs (insensitive per Phase C Test E)
- **G:** Dead code removed from mr:1162-1163 (A_thresh computed then overwritten)
- **H, I:** Style-only differences (find_peaks kwargs, except clause)

No divergence table changes. No regression impact.

### 2026-02-20 — Deep audit of Sections 2 & 3

Line-by-line comparison of Swenson's domain expansion (rh:341-609) and DEM
conditioning (rh:1457-1614) code against our pipeline. Also read all of
`geospatial_utils.py` (gu:1-306).

**Section 2 (Domain Expansion):** 5 findings (A-E). All benign/no-op/equivalent
for this MERIT gridcell: DEM loading nodata handling (A), longitude wrapping (B),
subregion splitting (C), gridcell extraction method (D), A_thresh int vs float (E).

**Section 3 (DEM Conditioning):** 9 findings (A-I). 1 genuine functional
divergence (dormant): `resolve_flats` fallback target differs — Swenson reverts
to original DEM, we keep the flooded DEM (Finding A). Corrected a factual error
in the existing DEM Conditioning Notes that stated Swenson falls back to the
flooded DEM. Added asterisk to item #11 noting the fallback divergence.

Other Section 3 findings: production guards we omit (B, C — no-op), dead code
in both codebases (D), our efficiency improvement (E — single slope_aspect call
vs Swenson's two), equivalent Grid creation API (F), `nanmax` vs `max` for
A_thresh safety valve (G — corrected in Sections 4-6 audit), cascading basin detection (H),
and return type style (I).

No divergence table changes. No regression impact.

### 2026-02-20 — Deep audit of Sections 4, 5, 6

Line-by-line comparison of Swenson's flow routing (rh:1575-1703) and
pgrid.py stream/HAND functions against our pipeline (mr:340-740).
10 findings (A-J) as a combined block covering Sections 4, 5, and 6.

**Key findings:**

- **A, B:** API-level differences (string names vs arrays, explicit
  routing="d8" vs default) — functionally equivalent
- **C:** MemoryError handling diverges: Swenson aborts gridcell (`return -1`),
  we continue with `None`. Dormant — not triggered at this gridcell size.
- **D:** Corrected Section 3 Finding G (nanmax/max had both sides reversed).
  Swenson uses `np.max`, we use `np.nanmax`. Our version is the defensive
  improvement, not Swenson's. Section 3 text updated in place.
- **E:** Section 5 (Slope/Aspect) confirmed — no new findings
- **F:** Operation reorder (create_channel_mask before extract_river_network)
  — benign, inputs are independent
- **G:** `river_network_length_and_slope` accesses `self.inflated_dem` and
  `self.acc` by hardcoded name. Swenson passes `dem=` and `acc=` kwargs
  that are silently captured by `**kwargs` and ignored. Our alias workaround
  (`grid.inflated_dem = grid.inflated`) is correct but fragile.
- **H:** Our fork's `compute_hand` returns only HAND; stores DTND and
  drainage_id as grid attributes. Swenson destructures 3 return values
  (original API). Equivalent.
- **I:** Swenson's stream network post-processing (GeoJSON → arrays) omitted.
  Used for visualization, not parameter computation.
- **J:** AZND computed by `compute_hand` but unused by either codebase for
  hillslope parameters.

No divergence table changes. No regression impact. No bugs found.

### 2026-02-20 — Deep audit of Sections 7-14

Line-by-line comparison of hillslope classification / aspect averaging
(Section 7), pixel area (8), gridcell extraction (9), data filtering (10),
HAND binning (11), trapezoidal width fitting (12), element computation (13),
and post-processing (14).

17 findings total: 1 in Section 7 (A), 16 in Sections 8-14 (A-P).

**One real bug found (Section 8-14, Finding M):** `n_hillslopes` at
mr:806-815 indexes `grid.drainage_id.flatten()` (full expanded grid)
with gridcell-level indices (`asp_indices`), picking drainage IDs from
wrong spatial positions. Dormant for correlation-based validation
(n_hillslopes is a uniform divisor — ratios preserved), but would
produce wrong absolute areas in a production pipeline. Fix deferred to
Phase D.

Other notable findings:
- **I:** Our `np.linalg.solve` is numerically superior to Swenson's
  explicit `inv + dot`.
- **L, P:** Our exception handling around trapezoidal fit and quadratic
  solver are defensive improvements over Swenson's crash-on-failure.
- **N:** Median DTND index off by 1 — completely dormant (overwritten
  by quadratic).
- **C, D, F:** NaN filtering and flood filter denominator differences —
  dormant, same pixels end up in each bin.

No divergence table changes. No regression impact (Finding M dormant
for correlations). Audit of all 14 sections now complete.

### 2026-02-20 — Six easy-win fixes applied

Applied 6 fixes identified in the mismatch analysis:

1. **Finding M (n_hillslopes BUG):** Extracted `drainage_id` to gridcell
   region before indexing. Was indexing the full expanded grid (~3.2M elements)
   with gridcell-level indices (~1.4M elements). **Not dormant** — width
   improved 0.9465→0.9894, area fraction improved 0.9047→0.9221.
2. **#25 (valid mask):** Removed `hand >= 0` check, now `isfinite` only.
3. **#28 (DTND min clipping):** Added `dtnd[dtnd < 1.0] = 1.0`.
4. **#43 (mean-HAND bin skip):** Added guard for bins with mean HAND <= 0.
5. **Finding E (quartile upper bound):** Removed `bounds[-1] = 1e6` override
   in quartile branch to match Swenson (tu:353-361).
6. **Finding A (empty uid guard):** Added `if uid.size == 0: return out`
   in `catchment_mean_aspect`.

Also dropped two planned fixes after verification:
- Width min clamp: already implemented at mr:306 and mr:889.
- Dead code removal: inflated_dem re-masking at mr:649-657 is NOT dead code.

**Regression result (job 25366112):** Lc PASS (763m). All 6 correlations
within tolerance. Width and area fraction both improved — the n_hillslopes
bug was NOT dormant for correlations as predicted.

| Parameter | Before | After | Delta |
|-----------|--------|-------|-------|
| Height | 0.9977 | 0.9977 | +0.0000 |
| Distance | 0.9987 | 0.9987 | +0.0000 |
| Slope | 0.9850 | 0.9850 | -0.0000 |
| Aspect | 1.0000 | 1.0000 | -0.0000 |
| Width | 0.9465 | 0.9894 | +0.0429 |
| Area fraction | 0.9047 | 0.9221 | +0.0174 |

Updated expected correlations in `merit_regression.py` to new baseline.
Updated counts: 42 MATCH, 2 DIVERGENCE, 1 OMISSION, 3 N/A.
