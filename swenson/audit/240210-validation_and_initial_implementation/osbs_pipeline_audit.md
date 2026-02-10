# Audit of OSBS Pipeline Scripts

Date: 2026-02-10

## Files Audited

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/osbs/run_pipeline.py` | 1817 | Main hillslope processing pipeline |
| `scripts/osbs/run_pipeline.sh` | 53 | SLURM job wrapper |
| `scripts/osbs/stitch_mosaic.py` | 145 | Merges DTM tiles into single GeoTIFF |
| `scripts/osbs/extract_subset.py` | 278 | Extracts rectangular subset from mosaic |

KML-related scripts were excluded from this audit.

---

## Critical Issues

### 1. DTND Uses Wrong Algorithm

**File:** `run_pipeline.py` lines 1480-1483

```python
stream_binary = stream_mask > 0
dtnd_pixels = distance_transform_edt(~stream_binary)
dtnd = dtnd_pixels * pixel_size
```

`distance_transform_edt` computes the Euclidean distance from each pixel to the
**geographically nearest** stream pixel. Swenson's method computes the distance
to the **hydrologically nearest** stream pixel --- the one each pixel actually
drains to via D8 flow routing. These differ whenever a pixel is geographically
close to a stream on the opposite side of a divide but hydrologically drains to
a farther stream.

**Why the workaround exists:** pysheds' `compute_hand()` computes the correct
hydrologically-linked DTND internally (pgrid.py lines 1928-1942), but the
implementation uses the haversine formula, which assumes geographic (lat/lon)
coordinates. Our data is UTM. Feeding UTM easting/northing into haversine
produces garbage, so the pipeline sidesteps pysheds' DTND entirely.

The result is that neither DTND is correct:

| Approach | Concept | Math |
|----------|---------|------|
| pysheds DTND | Correct (flow-path-linked stream pixel) | Wrong (haversine on UTM) |
| Pipeline EDT | Wrong (nearest stream regardless of drainage) | Correct (Euclidean on UTM) |

**Fix options:**
- Modify pysheds `compute_hand()` to detect UTM CRS and use Euclidean distance
  instead of haversine
- Extract the internal `hndx` array from `compute_hand()` (the index mapping each
  pixel to its drainage stream pixel) and compute the UTM distance externally
- Add a `return_index=True` path that exposes `hndx` (the parameter already exists
  in the function signature but bypasses the DTND calculation)

### 2. Slope/Aspect Not Validated

**File:** `run_pipeline.py` lines 1514-1528

```python
dzdy, dzdx = np.gradient(dem_for_slope, pixel_size)
slope = np.sqrt(dzdx**2 + dzdy**2)
aspect = np.degrees(np.arctan2(-dzdx, -dzdy))
```

Stage 8 of the MERIT validation discovered that `np.gradient`-based aspect
calculation had a Y-axis sign inversion that systematically swapped North and
South aspects. The fix was to replace it with pgrid's `grid.slope_aspect("dem")`
(Horn 1981 8-neighbor stencil). That fix was applied to `stage3_hillslope_params.py`
but **not** propagated to the OSBS pipeline.

The pipeline's `arctan2(-dzdx, -dzdy)` argument order differs from the buggy MERIT
code and may compensate for the N/S issue, but it was **never validated** against
the stage 8 findings. The comment at line 1514 ("pysheds assumes geographic coords")
acknowledges why pgrid's method wasn't used directly, but no alternative validation
was performed.

See the [Stage 8 section of the merit validation audit](claude-audit.md#stage-8)
for the full background.

### 3. No Trustworthy Characteristic Length Scale (Lc)

**File:** `run_pipeline.py` lines 1275-1287

The FFT is 4x subsampled (`subsample_factor = 4` at line 1275), and a floor of
`MIN_LC_PIXELS = 100` (100m) is imposed when the FFT fails to find a real peak.

Stage 2 of the MERIT validation showed that subsampling below the Nyquist limit
produces spectral artifacts (the ~2700m full-tile result vs ~760m at center
regions). The full-resolution FFT on the OSBS interior mosaic has never been run.

Everything downstream depends on Lc:
- Accumulation threshold = 0.5 * Lc^2
- Stream network density
- HAND and DTND values
- All 6 hillslope parameters

See the [Stage 9 section of the merit validation audit](claude-audit.md#stage-9)
for the Lc comparison table.

### 4. 4x Subsampling for Flow Routing

**File:** `run_pipeline.py` line 1354

```python
subsample = 4  # Process at 4m resolution instead of 1m
```

Discards 93.75% of the 1m LIDAR data before the core hydrology computation
(flow direction, accumulation, stream network, HAND). The bottleneck is pysheds'
`resolve_flats()`, which has poor scaling behavior on large flat regions common
at OSBS.

Documented separately in [flow-routing-resolution.md](flow-routing-resolution.md).

---

## Scientific Concerns

### 5. DEM Conditioning May Erase Real Geomorphic Features

**File:** `run_pipeline.py` lines 1436-1440

```python
grid.fill_pits("dem", out_name="pit_filled")
grid.fill_depressions("pit_filled", out_name="flooded")
grid.resolve_flats("flooded", out_name="inflated")
```

At 1m resolution, pits and depressions include real features: sinkholes,
wetland depressions, karst dissolution features. Filling them forces a
continuous drainage network but destroys information about closed basins
that are central to OSBS's wetlandscape hydrology.

**Status:** Open question for PI. This is a fundamental methodological
choice, not a bug.

### 6. Stream Channel Parameters Are Hardcoded Guesses

**File:** `run_pipeline.py` lines 1013-1022

```python
stream_depth = 0.3
stream_width = 5.0
stream_slope = np.mean(lowest_bin_slopes) * 0.5 if any(lowest_bin_slopes) else 0.002
```

Comparison to Swenson's reference file (`hillslopes_osbs_c240416.nc`):

| Parameter | Pipeline | Swenson reference |
|-----------|----------|-------------------|
| Stream depth | 0.3 m | 0.269 m |
| Stream width | 5.0 m | 4.414 m |
| Stream slope | heuristic | 0.00233 |

The guesses are in the right ballpark, but there is no methodology behind them.
Stream slope could be computed from actual DEM elevation drops along the
identified stream network. Stream depth and width could be derived from regional
empirical relationships or from the MERIT Hydro dataset that Swenson used for his
global values.

These parameters directly affect CTSM's lateral subsurface flow and stream-groundwater
exchange calculations.

### 7. Bedrock Depth Is a Placeholder

**File:** `run_pipeline.py` line 1025

```python
bedrock_depth = np.full(n_columns, 1e6)
```

This is effectively infinite depth. The Swenson reference file has all zeros
for bedrock depth. Neither value is physically meaningful. CTSM uses bedrock
depth to limit soil column depth and lateral flow.

**Action needed:** Determine what CTSM does with bedrock_depth = 0 vs 1e6.
Check whether this parameter matters for OSBS simulations, and if so, identify
a reasonable source (SoilGrids, OSBS site data, etc).

### 8. FFT Parameters Not Validated for OSBS

**File:** `run_pipeline.py` lines 80-85, 489-506

| Parameter | Value | Concern |
|-----------|-------|---------|
| `blend_edges(n=50)` | 50px = 50m | Swenson uses n=4 at 90m = 360m of geographic smoothing. 50m is 7x less. |
| `edge_zero = 50` | 50px = 50m | Same scaling mismatch. |
| `NLAMBDA = 30` | 30 bins | Copied from Swenson. May need more bins at 1m (wider wavelength range). |
| `MAX_HILLSLOPE_LENGTH = 2000` | 2000m | Reduced from Swenson's 10000 (reasonable for OSBS, but not validated). |

See the [FFT parameter sensitivity test plan](claude-audit.md#claude-info-when-pressed)
in the merit validation audit.

---

## Code Quality Issues

### 9. ~400 Lines of Duplicated Spatial Scale Code

**File:** `run_pipeline.py` lines 275-568

The following functions are copied and adapted from `scripts/spatial_scale.py`:
- `calc_gradient_utm`
- `blend_edges`
- `fit_planar_surface_utm`
- `bin_amplitude_spectrum`
- `log_normal`
- `fit_peak_lognormal`
- `identify_spatial_scale_utm`

The pipeline version has diverged: it drops Gaussian and linear peak fitting,
keeping only lognormal. Any bug fix or improvement must be applied to both copies.

### 10. Duplicated Hillslope Computation Code

**File:** `run_pipeline.py` lines 666-742

`fit_trapezoidal_width`, `quadratic`, and the width calculation logic (lines
1659-1696) are duplicated from `stage3_hillslope_params.py`.

See the [Stage 5/6 section of the merit validation audit](claude-audit.md#stage-56)
for the proposed shared module structure that would resolve issues 9 and 10.

### 11. `n_hillslopes` Fallback Is Fragile

**File:** `run_pipeline.py` lines 1608-1617

```python
n_hillslopes = max(1, len(np.unique(
    grid.drainage_id.flatten()[asp_indices]
    if hasattr(grid, 'drainage_id') else [1]
)))
```

`drainage_id` is set by `compute_hand()` (pgrid.py line 1926), which IS called
before this code runs. So the `hasattr` guard should never trigger. But if it
does (e.g., pysheds API changes), `n_hillslopes=1` would silently produce
incorrect area normalization in the trapezoidal fit.

---

## Things Done Right

- **Width fix** is properly implemented (lines 1651-1696): uses fitted areas
  from the trapezoidal model with the quadratic solver, matching stage 6 fix.
- **Aspect deg-to-rad conversion** is present (line 985) for NetCDF output.
- **Valid mask** properly excludes nodata/fill regions from all statistics.
- **Connected-component extraction** (lines 1311-1328) correctly isolates the
  largest contiguous data region, avoiding flow fragmentation across mosaic gaps.
- **NetCDF structure** matches the Swenson reference file exactly (verified via
  `ncdump` against `hillslopes_osbs_c240416.nc`): all variable names, dimensions,
  units, and downhill column indexing are correct.
- **Circular mean** for aspect (lines 746-752) correctly handles the 0/360
  wraparound.
- **HAND bin computation** (lines 585-663) implements the mandatory 2m lowest-bin
  constraint from Swenson.
- **Slope computed from original DEM** (line 1380-1381), not the pysheds-conditioned
  DEM. This was a lesson from the stage 9 debugging (pysheds fills nodata with
  high values, creating false gradients at boundaries).

---

## Supporting Scripts

### `run_pipeline.sh`
SLURM wrapper. Requests 128GB, 4hr, gerber-b QOS. Sets `TILE_SELECTION_MODE=interior`
and `OUTPUT_DESCRIPTOR=interior`. No issues found.

### `stitch_mosaic.py`
Straightforward `rasterio.merge` of all DTM tiles into a single GeoTIFF. No issues found.

### `extract_subset.py`
Extracts a rectangular region from the mosaic using WGS84 corner coordinates
converted to UTM via pyproj. Functionally correct. Note: this script is used
for one-off extractions, not part of the main pipeline.

---

## Priority Summary

| # | Issue | Scientific Impact | Fix Complexity |
|---|-------|-------------------|----------------|
| 1 | DTND algorithm wrong | High | Medium (modify pysheds or extract hndx) |
| 2 | Slope/aspect not validated | High | Low (validate, then adapt pgrid for UTM) |
| 3 | No trustworthy Lc | High | Low (run full-res FFT) |
| 4 | 4x subsampling | High | Medium (optimize resolve_flats) |
| 5 | DEM conditioning | Medium | N/A (science question for PI) |
| 6 | Stream params guessed | Medium | Low (compute from stream network) |
| 7 | Bedrock depth placeholder | Unknown | Research CTSM behavior |
| 8 | FFT params untested | Medium | Low (sensitivity testing) |
| 9-10 | Code duplication | Maintenance risk | Medium (create shared module) |
| 11 | n_hillslopes fallback | Low | Low (add assertion) |
