# DEM Conditioning Todo: Remaining Swenson Alignment Items

After implementing the basin/water handling chain, 5 differences remain between
Swenson's DEM conditioning pipeline (`representative_hillslope.py`, abbreviated
`rh`) and ours (`merit_regression.py`, abbreviated `mr`). Each item is tested
incrementally via the MERIT regression.

Items are ordered by dependency and risk: safety-net changes first, then
no-ops, then correlation-affecting changes, then new functionality.

---

## 1. `resolve_flats` ValueError fallback

**Risk:** None. Pure safety net.

**What Swenson does (rh:1562-1567):**
```python
try:
    inflated_dem = grid.resolve_flats(flooded_dem)
except ValueError:
    warning("flats cannot be resolved")
    inflated_dem = grid.add_gridded_data(dem)
```
If `resolve_flats` throws a `ValueError`, Swenson falls back to the original
(flooded) DEM and continues. This allows the pipeline to survive gridcells
where flat resolution fails.

**What we do (mr:557):**
```python
grid.resolve_flats("flooded", out_name="inflated")
```
No try/except. A `ValueError` would crash the pipeline.

**Change:** Wrap in try/except. On `ValueError`, fall back to using `"flooded"`
as the `"inflated"` surface:
```python
try:
    grid.resolve_flats("flooded", out_name="inflated")
except ValueError:
    print("  WARNING: resolve_flats failed, using flooded DEM as inflated")
    grid.add_gridded_data(
        np.array(grid.flooded),
        data_name="inflated",
        affine=grid.affine,
        crs=grid.crs,
        nodata=grid.nodata,
    )
```

**Expected regression impact:** No change. `resolve_flats` succeeds on this
MERIT gridcell, so the fallback never triggers. All 6 correlations unchanged.

- [x] Implement
- [x] Run regression, confirm PASS with identical correlations

---

## 2. `acc_mask` isfinite check

**Risk:** None at 90m MERIT. Defensive for OSBS.

**What Swenson does (rh:1604-1606):**
```python
acc_mask = np.logical_and(
    (acc > self.thresh), np.isfinite(inflated_dem)
)
```
The `isfinite` check excludes basin pixels (which were set to NaN/fill_value
in the inflated DEM at rh:1583) from the stream network mask. This prevents
NaN-valued basin interiors from being classified as stream pixels based solely
on their accumulation value.

**What we do (mr:595):**
```python
acc_mask = grid.acc > accum_threshold
```
No `isfinite` check. Basin pixels whose accumulation exceeds the threshold
could be included in the stream mask.

**Why it's a no-op at MERIT:** At 90m resolution, the MERIT gridcell has no
basin pixels in the inflated DEM after the basin chain runs. The `isfinite`
filter removes zero additional pixels.

**Why it matters for OSBS:** At 1m resolution with real lake/wetland basins,
basin-interior pixels will have NaN in the inflated DEM. Without the
`isfinite` check, these could leak into the stream mask.

**Change (mr:595):**
```python
acc_mask = (grid.acc > accum_threshold) & np.isfinite(np.array(grid.inflated))
```

**Expected regression impact:** No change. 0 pixels removed by the new filter
at MERIT. All 6 correlations unchanged.

- [x] Implement
- [x] Run regression, confirm PASS with identical correlations

---

## 3. Switch `compute_hand` to inflated DEM

**Risk:** Medium. WILL change HAND values and correlations.

**What Swenson does (rh:1685):**
```python
hand, dtnd, drainage_id = grid.compute_hand(
    fdir, inflated_dem, channel_mask, channel_id, dirmap=dirmap
)
```
HAND is computed relative to the **inflated** (flat-resolved) DEM. This means
the elevation reference for "height above nearest drainage" uses the same
surface that determined flow directions. Flat regions that were inflated to
create gradient get correspondingly adjusted HAND values.

**What we do (mr:599-601):**
```python
grid.compute_hand(
    "fdir", "dem", grid.channel_mask, grid.channel_id,
    dirmap=DIRMAP, routing="d8"
)
```
HAND is computed relative to the **original** (pit-filled but not
flat-resolved) DEM. Flow directions were computed from the inflated DEM, but
height differences are measured on the original surface. This creates a
mismatch: a pixel's drainage target was determined by the inflated surface,
but its height-above-drainage uses the pre-inflation surface.

**Previous test results (pre-basin-chain, "Test N"):**

| Parameter | `"dem"` | `"inflated"` | Delta |
|-----------|---------|--------------|-------|
| Height    | 0.9999  | 0.9977       | -0.0022 |
| Slope     | 0.9966  | 0.9825       | -0.0141 |
| Area frac | 0.8215  | 0.8289       | +0.0074 |

These results are from before the basin/water handling chain was added.
The basin chain changes the inflated DEM (basins are lowered then re-NaN'd),
so correlations will differ. Must retest.

**Change (mr:600):** `"dem"` -> `"inflated"`
```python
grid.compute_hand(
    "fdir", "inflated", grid.channel_mask, grid.channel_id,
    dirmap=DIRMAP, routing="d8"
)
```

**Expected regression impact:** Height, slope, and area fraction correlations
will change. Direction unknown with the basin chain in place. The expected
correlation values in the regression must be updated after this change.

- [x] Implement
- [x] Run regression, record new correlations
- [x] Update expected correlations in `merit_regression.py`
- [x] Verify all correlations still within scientifically reasonable range

---

## 4. Catchment-level aspect averaging

**Risk:** Medium-high. Largest code change. Replaces per-pixel aspect with
catchment-side circular mean.

**What Swenson does (rh:1692-1751):**

After computing HAND/DTND, Swenson runs two additional steps:

**Step 4a: Classify hillslope sides (rh:1692)**
```python
hillslope = grid.compute_hillslope(fdir, channel_mask, bank_mask)
```
`compute_hillslope` (pgrid.py:2102) classifies every pixel as:
- 1 = headwater (upstream of channel head)
- 2 = right bank
- 3 = left bank
- 4 = channel

This uses `bank_mask` from `create_channel_mask` (already stored as
`grid.bank_mask` by our pipeline at mr:596).

**Step 4b: Average aspect within catchment sides (rh:1725-1751)**
```python
aspect2d_catchment_mean = set_aspect_to_hillslope_mean_parallel(
    drainage_id, aspect, hillslope, npools=npools
)
self.aspect = aspect2d_catchment_mean
```
For each unique `drainage_id` (catchment), for each hillslope side
(headwater, right bank, left bank), compute the circular mean aspect of all
pixels in that group. Replace every pixel's aspect with its group mean.

The helper `_calculate_hillslope_mean_aspect` (terrain_utils.py:147-171)
does the actual circular mean:
```python
mean_aspect = np.arctan2(
    np.mean(np.sin(dtr * aspect[ind])),
    np.mean(np.cos(dtr * aspect[ind])),
) / dtr
```
Channel pixels (type 4) are merged with each hillslope side for averaging.

**What we do (mr:536):**
```python
aspect = np.array(grid.aspect)
```
Raw per-pixel aspect. No catchment averaging.

**What this changes conceptually:** Instead of each pixel having its own
aspect, all pixels on the same side of the same catchment share one aspect
value. This smooths out local aspect noise and ensures consistent aspect
binning within catchment units.

**Prerequisites:**
- `grid.bank_mask` — already available from `create_channel_mask` (mr:596)
- `grid.drainage_id` — already available from `compute_hand` (returned by
  pgrid, stored as `grid.drainage_id`)
- `grid.compute_hillslope()` — available in pgrid.py:2102, not yet called
- `_calculate_hillslope_mean_aspect` — in terrain_utils.py:147, must port
- `set_aspect_to_hillslope_mean_parallel` — in terrain_utils.py:174, must port

**Implementation plan:**
1. After `compute_hand`, call `grid.compute_hillslope("fdir",
   grid.channel_mask, grid.bank_mask)`
2. Port `_calculate_hillslope_mean_aspect` into `merit_regression.py`
   (or a shared module)
3. Port `set_aspect_to_hillslope_mean_parallel` (uses multiprocessing Pool)
4. Replace `aspect` array with the catchment-averaged version before
   hillslope parameter computation

**Expected regression impact:** Small at 90m. Aspect correlation is already
0.9999 — catchment averaging at coarse resolution changes few pixels
meaningfully. May slightly affect area fraction since aspect binning
(N/E/S/W assignment) uses the averaged values.

- [ ] Port `_calculate_hillslope_mean_aspect` from terrain_utils.py
- [ ] Port `set_aspect_to_hillslope_mean_parallel` from terrain_utils.py
- [ ] Add `compute_hillslope` call after `compute_hand`
- [ ] Apply catchment aspect averaging before parameter computation
- [ ] Run regression, record new correlations
- [ ] Update expected correlations if changed

---

## 5. Stream network extraction + slope

**Risk:** Low. Adds new output, does not change existing 6 parameters.

**What Swenson does:**

**Step 5a: Extract river network (rh:1608-1614)**
```python
try:
    branches = grid.extract_river_network(
        fdir=fdir, mask=acc_mask, dirmap=dirmap
    )
except MemoryError:
    warning("Memory Error in extract_river_network, skipping")
    return -1
```
`extract_river_network` (pgrid.py:3446) generates GeoJSON river segments
from the flow direction and accumulation mask. Returns a dict with
`"features"` containing individual stream reaches.

**Step 5b: Compute network length and slope (rh:1655-1673)**
```python
try:
    x = grid.river_network_length_and_slope(
        dem=inflated_dem, fdir=fdir, acc=acc,
        mask=acc_mask, dirmap=dirmap
    )
except MemoryError:
    warning("Memory Error in river_network_length_and_slope, skipping")
    return -1

self.network_length = x["length"]
self.mean_network_slope = x["slope"]
self.reach_slopes = x["reach_slopes"]
self.reach_lengths = x["reach_lengths"]
```
`river_network_length_and_slope` (pgrid.py:3170) traces each stream reach
and computes elevation-based slope along the profile. Returns total network
length, mean slope, and per-reach arrays.

**What we do:** Nothing. We don't extract the river network or compute stream
slope.

**Why this matters:** Stream slope is one of the hardcoded parameters in the
OSBS pipeline (STATUS.md problem #6). Computing it from the DEM provides a
physically motivated value instead of a guess.

**Implementation plan:**
1. After the accumulation mask is computed, call `extract_river_network`
   wrapped in try/except MemoryError
2. Call `river_network_length_and_slope` wrapped in try/except MemoryError
3. Print summary stats: number of stream reaches, total network length,
   mean network slope
4. These are informational — not part of the 6-parameter correlation test

**Expected regression impact:** No change to correlations. New stdout output
only.

- [ ] Add `extract_river_network` call with MemoryError handling
- [ ] Add `river_network_length_and_slope` call with MemoryError handling
- [ ] Print stream network stats (reach count, length, slope)
- [ ] Run regression, confirm correlations unchanged
