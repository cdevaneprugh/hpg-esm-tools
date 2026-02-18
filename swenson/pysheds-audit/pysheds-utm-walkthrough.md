# Pysheds Fork: UTM CRS Support Walkthrough

**Branch:** `feature/utm-crs-support` in `$PYSHEDS_FORK`
**Base commit:** `7b20e8e` (Swenson's functions ported to upstream pysheds 0.4)
**Phase:** A of the Swenson hillslope implementation

---

## Background

### Why this fork exists

CTSM's hillslope hydrology needs six geomorphic parameters per hillslope element. Swenson & Lawrence (2025) developed the methodology using pysheds on the MERIT DEM, which uses a geographic CRS (EPSG:4326 — lon/lat in degrees). Their code was ported into our pysheds fork as `pgrid.py` functions.

Our OSBS data is 1m NEON LIDAR in UTM (EPSG:32617 — easting/northing in **meters**). Every function that converts between pixel indices and physical distances assumed geographic coordinates and used the haversine formula to do so. Haversine interprets coordinate values as degrees — feeding it `400000` (a UTM easting in meters) means `400000°`, producing nonsensical distances.

### What was broken

Four locations in `pgrid.py` contained unconditional haversine/spherical-bearing math:

| Location | Computation | Effect on OSBS |
|----------|-------------|----------------|
| `compute_hand()` — DTND | Distance to nearest drainage | Garbage distances |
| `compute_hand()` — AZND | Azimuth to nearest drainage | Garbage bearings |
| `_gradient_horn_1981()` | Pixel spacing for slope/aspect | Garbage gradients → wrong slope & aspect |
| `river_network_length_and_slope()` | Reach segment lengths | Garbage stream lengths & slopes |

The OSBS pipeline (`run_pipeline.py`) worked around two of these by reimplementing DTND with `scipy.ndimage.distance_transform_edt` (correct math but wrong algorithm — Euclidean to nearest channel, not hydrologically linked channel) and slope/aspect with `np.gradient` (correct math but introduced an N/S aspect swap bug from sign convention mismatch). The other two had no workaround.

### Design pattern

All four fixes follow the same pattern: detect the CRS once, then branch at each distance/angle computation. Geographic CRS gets the original haversine code unchanged; projected CRS gets the Euclidean equivalent. This preserves bit-identical behavior for geographic data (MERIT validation still passes) while producing correct results for projected data (UTM).

The CRS detection leverages an existing pyproj compatibility layer at `pgrid.py:32-34` that was already present in Swenson's port.

---

## 1. Pyproj Compatibility Layer (line 32-34) — Pre-existing

This infrastructure was present before Phase A. It is the foundation that `_crs_is_geographic()` builds on.

```python
# pgrid.py lines 32-34
#
# [ORIGINAL] Swenson's port included this pyproj version-compatibility layer.
# pyproj 2.2+ changed the API for CRS objects:
#   - Old (<2.2): CRS is a Proj object, geographic test is .is_latlong
#   - New (>=2.2): CRS has a .crs attribute, geographic test is .is_geographic
#
# These three lines abstract the difference so downstream code can use:
#   _pyproj_crs(self.crs)          → get the CRS object (works with any pyproj)
#   _pyproj_crs_is_geographic      → attribute name to check ('is_latlong' or 'is_geographic')
#
_OLD_PYPROJ = LooseVersion(pyproj.__version__) < LooseVersion('2.2')
_pyproj_crs = lambda Proj: Proj.crs if not _OLD_PYPROJ else Proj
_pyproj_crs_is_geographic = 'is_latlong' if _OLD_PYPROJ else 'is_geographic'
```

No code was changed here. Phase A changed only the import path for `LooseVersion` (see Section 9).

---

## 2. CRS Detection: `_crs_is_geographic()` (line 129-148)

**This is the central addition.** Every CRS branch in the fork calls this method.

### What it does

Returns `True` for geographic CRS (lat/lon in degrees), `False` for projected CRS (e.g. UTM in meters). The result determines whether to use haversine (spherical) or Euclidean (planar) distance math.

### Diff

**Before:** Method did not exist. Each haversine call assumed geographic CRS unconditionally.

**After:**

```python
# pgrid.py lines 129-148
#
# [ADDED - Phase A] New method. All CRS branches call this.
#
def _crs_is_geographic(self):
    """Check whether the grid's CRS is geographic (lat/lon) or projected (e.g. UTM).

    This distinction controls how spatial distances are computed throughout
    the codebase:

    - Geographic CRS (e.g. EPSG:4326): Coordinates are longitude/latitude
      in degrees. Distances must be computed via the haversine formula,
      which accounts for Earth's curvature.

    - Projected CRS (e.g. UTM EPSG:32617): Coordinates are easting/northing
      in linear units (meters for UTM). Distances can be computed via simple
      Euclidean math — sqrt(dx² + dy²) — because the projection has already
      mapped the curved surface to a flat plane.

    Edge case: A local/engineering CRS with no EPSG code will return False
    (not geographic), which is the correct default since local CRS use
    linear units and Euclidean distance is appropriate.
    """
    # [WHY] Uses the pyproj compatibility layer (lines 32-34) to work
    # with any pyproj version. The getattr call resolves to either
    # self.crs.is_geographic (pyproj >= 2.2) or self.crs.is_latlong
    # (pyproj < 2.2). Both return a boolean.
    return getattr(_pyproj_crs(self.crs), _pyproj_crs_is_geographic)
```

### Why not check the CRS at `__init__` time?

The Grid's CRS can be changed after construction (e.g. by `add_gridded_data()` or `read_raster()`), so checking at each use site ensures correctness regardless of when data is loaded.

---

## 3. DTND in `compute_hand()` (lines 1958-1997)

**Problem #1 from STATUS.md.** This was the most impactful fix — it was what made the pipeline use the wrong DTND algorithm entirely.

### What DTND is

Distance To Nearest Drainage: the straight-line distance from each pixel to the specific channel pixel it drains to (found by tracing D8 flow directions downstream). This is *not* the shortest distance to any channel pixel — it is the distance to the *hydrologically linked* channel pixel.

### Why this matters

The original code used haversine unconditionally. On UTM data this produced garbage, which forced the OSBS pipeline to replace it with `scipy.ndimage.distance_transform_edt`. That workaround used correct Euclidean math but the *wrong algorithm* — EDT finds the nearest channel pixel by straight-line distance, ignoring watershed divides. A pixel on one side of a ridge might be assigned the distance to a channel on the other side. With the fork fix, pysheds computes the correct hydrological DTND using Euclidean math on UTM coordinates.

### How it works

The key insight is that the *topology* (which pixel drains to which channel) is CRS-independent — it's determined entirely by D8 flow directions. The `hndx` array (line 1949) maps every pixel to its drainage outlet's flat array index. This routing step is unchanged. Only the *distance computation* between a pixel and its `hndx` target needs CRS-aware math.

### Diff

**Before (original Swenson code):**

```python
# calculate distance to nearest channel
dtnd = np.zeros(dem.shape)

dlon = lon2d-lon2d.flat[hndx]
dlat = lat2d-lat2d.flat[hndx]
dtr = np.pi/180
# haversine formula
dtnd = np.power(np.sin(dtr*dlat/2),2) + np.cos(dtr*lat2d) * np.cos(dtr*lat2d.flat[hndx]) * np.power(np.sin(dtr*dlon/2),2)
dtnd[dtnd > 1] = 1
dtnd[dtnd < 0] = 0
dtnd = (6.371e6 * 2 * np.arctan2( np.sqrt(dtnd), np.sqrt(1-dtnd)))
#dtnd = np.where(hndx != -1, dtnd, nodata_out)
dtnd = np.where(hndx != -1, dtnd, 0)
```

**After:**

```python
# pgrid.py lines 1958-1997
#
# --- Distance To Nearest Drainage (DTND) ---
#
# [ADDED - Phase A] Block comment explaining the computation.
#
# Compute the straight-line distance from each pixel to the
# drainage pixel it flows to (its hydrologically nearest channel
# pixel). This is NOT the shortest Euclidean distance to any
# channel pixel — it's the distance to the specific channel pixel
# found by tracing D8 flow directions downstream.
#
# hndx (computed above at line ~1923) stores the flat array index
# of each pixel's drainage outlet. It's CRS-independent — purely
# topological, determined by the D8 flow direction graph.
#
# dx/dy below are coordinate differences FROM drainage pixel
# TO current pixel (i.e. current - drainage). The sign doesn't
# matter for distance (we square them), but it does matter for
# AZND below.

# [ORIGINAL] These two lines are unchanged — they compute the
# coordinate offset between each pixel and its drainage target.
# In the original Swenson code these were named dlon/dlat; they
# were renamed to dx/dy to be CRS-neutral (for UTM they contain
# easting/northing offsets in meters, not longitude/latitude).
dx = x2d-x2d.flat[hndx]
dy = y2d-y2d.flat[hndx]

# [ADDED - Phase A] CRS branch for distance computation.
if self._crs_is_geographic():
    # [ORIGINAL] Geographic CRS: coordinates are lon/lat in degrees.
    # Must use the haversine formula to compute great-circle
    # distance on the sphere. Feeding UTM meter values into
    # this formula produces garbage (it interprets e.g.
    # 400000m as 400000 degrees).
    #
    # [WHY] This branch is identical to Swenson's original code.
    # The local dtr/re constants were replaced by module-level
    # _DEG_TO_RAD/_EARTH_RADIUS_M but the math is unchanged.
    # MERIT validation (stage 1-9) confirms bit-identical results.
    dtnd = np.power(np.sin(_DEG_TO_RAD*dy/2),2) + np.cos(_DEG_TO_RAD*y2d) \
           * np.cos(_DEG_TO_RAD*y2d.flat[hndx]) \
           * np.power(np.sin(_DEG_TO_RAD*dx/2),2)
    dtnd[dtnd > 1] = 1
    dtnd[dtnd < 0] = 0
    dtnd = (_EARTH_RADIUS_M * 2 * np.arctan2( np.sqrt(dtnd), np.sqrt(1-dtnd)))
else:
    # [ADDED - Phase A] Projected CRS (e.g. UTM): coordinates are
    # already in linear units (meters for UTM). Euclidean distance
    # is exact on the projected plane. UTM distortion is < 0.04%
    # within a single zone, so the error is negligible.
    #
    # [WHY] This is the simple Pythagorean formula. It replaces
    # the EDT workaround in run_pipeline.py (which computed
    # distance to the geographically nearest channel, ignoring
    # watershed topology). Here dx/dy point to the correct
    # (hydrologically linked) channel pixel via hndx.
    dtnd = np.sqrt(dx**2 + dy**2)

# [ORIGINAL] Mask unrouted pixels (hndx == -1) to zero.
# The commented-out nodata_out line was already commented in
# Swenson's original; retained as-is.
dtnd = np.where(hndx != -1, dtnd, 0)

self._output_handler(data=dtnd, out_name='dtnd',
                     properties=properties,
                     inplace=inplace, metadata=metadata)
```

### What changed vs what didn't

| Element | Changed? | Notes |
|---------|----------|-------|
| `hndx` D8 routing (line 1929-1949) | No | Topology is CRS-independent |
| `dx`/`dy` computation (was `dlon`/`dlat`) | **Renamed** | Same subtraction, different physical interpretation; renamed to be CRS-neutral |
| `x2d`/`y2d` (was `lon2d`/`lat2d`) | **Renamed** | CRS-neutral names from `_2d_crs_coordinates()` |
| Haversine formula | No | Moved into `if` branch, logic identical |
| `dtr`/`re` constants | **Replaced** | Now uses module-level `_DEG_TO_RAD`/`_EARTH_RADIUS_M` |
| Euclidean branch | **Added** | New `else` clause |
| `dtnd = np.zeros(...)` initialization | **Removed** | No longer needed — both branches produce the full array |

---

## 4. AZND in `compute_hand()` (lines 1999-2029)

Immediately follows the DTND computation and reuses the same `dx`/`dy` values.

### What AZND is

Azimuth to Nearest Drainage: the compass bearing from each pixel to its drainage outlet. Convention: 0° = north, 90° = east, 180° = south, 270° = west.

### Diff

**Before (original Swenson code):**

```python
# calculate angle wrt nearest channel
# points toward hndx, so reverse dlon
aznd = np.zeros(dem.shape)
aznd = np.arctan2(np.sin(-dtr*dlon),(np.cos(dtr*lat2d)*np.tan(dtr*lat2d.flat[hndx]) - np.sin(dtr*lat2d)*np.cos(-dtr*dlon)))
aznd = aznd/dtr
aznd[aznd < 0] += 360
#aznd = np.where(hndx != -1, aznd, 0)
```

**After:**

```python
# pgrid.py lines 1999-2029
#
# --- Azimuth to Nearest Drainage (AZND) ---
#
# [ADDED - Phase A] Block comment explaining the sign convention.
#
# Compute the compass bearing FROM each pixel TO its drainage
# outlet. Convention: 0° = north, 90° = east, 180° = south,
# 270° = west (standard geographic azimuth).
#
# Sign convention for dx/dy (computed above):
#   dx = x_pixel - x_drainage  (FROM drainage TO pixel)
#   dy = y_pixel - y_drainage  (FROM drainage TO pixel)
#
# We want the bearing FROM pixel TO drainage, which is the
# REVERSE direction. Hence we negate both components.
#
# arctan2(x, y) with x = east component, y = north component
# gives clockwise-from-north azimuth (geographic convention).

# [ADDED - Phase A] CRS branch for azimuth computation.
if self._crs_is_geographic():
    # [ORIGINAL] Spherical bearing formula. Identical to Swenson's
    # original code. The sin(-dx) and cos(-dx) terms negate
    # dx for the same reason: reversing the vector from
    # "drainage→pixel" to "pixel→drainage".
    #
    # [WHY] The spherical bearing formula accounts for meridian
    # convergence. On a sphere, a constant easting offset
    # corresponds to different angular bearings at different
    # latitudes. This formula handles that correctly.
    aznd = np.arctan2(
        np.sin(-_DEG_TO_RAD*dx),
        (np.cos(_DEG_TO_RAD*y2d) * np.tan(_DEG_TO_RAD*y2d.flat[hndx])
         - np.sin(_DEG_TO_RAD*y2d) * np.cos(-_DEG_TO_RAD*dx))
    )
    aznd = aznd/_DEG_TO_RAD
else:
    # [ADDED - Phase A] Projected CRS: planar azimuth.
    # Negate dx/dy to reverse from "drainage→pixel" to
    # "pixel→drainage". arctan2(-dx, -dy) gives the
    # clockwise-from-north bearing in radians.
    #
    # [WHY] On a projected plane, bearing is simply the angle
    # of the displacement vector. arctan2(east, north) gives
    # clockwise-from-north by convention. The negation reverses
    # the direction since dx/dy point drainage→pixel.
    aznd = np.degrees(np.arctan2(-dx, -dy))

# [ORIGINAL] Normalize from [-180, 360) to [0, 360).
# Moved outside the if/else since both branches need it.
aznd[aznd < 0] += 360

self._output_handler(data=aznd, out_name='aznd',
                     properties=properties,
                     inplace=inplace, metadata=metadata)
```

### What changed vs what didn't

| Element | Changed? | Notes |
|---------|----------|-------|
| `dx`/`dy` (was `dlon`/`dlat`) | **Renamed** | Reused from DTND section above |
| Spherical bearing formula | No | Moved into `if` branch, logic identical |
| `dtr` constant | **Replaced** | Now uses module-level `_DEG_TO_RAD` |
| Planar bearing branch | **Added** | New `else` clause using `np.degrees(np.arctan2(-dx, -dy))` |
| `aznd = np.zeros(...)` initialization | **Removed** | Both branches produce the full array |
| `aznd[aznd < 0] += 360` normalization | **Moved** | Was after the formula; now after the `if/else` block (applies to both branches) |

---

## 5. `_gradient_horn_1981()` (lines 4175-4236)

**Addresses problem #4 from STATUS.md** (slope/aspect). This is the internal method called by `slope_aspect()`.

### What it does

Computes the spatial gradient (dz/dx, dz/dy) of a DEM using the Horn 1981 3x3 stencil. The stencil uses 8 neighbors with specific weights:

```
dz/dx = (z_NE + 2*z_E + z_SE - z_NW - 2*z_W - z_SW) / (8 * cell_dx)
dz/dy = (z_NE + 2*z_N + z_NW - z_SE - 2*z_S - z_SW) / (8 * cell_dy)
```

The `cell_dx` and `cell_dy` denominators must be physical distances in meters. For geographic CRS, this requires converting degree offsets to meters using Earth's radius and a `cos(lat)` correction for longitude. For projected CRS, the affine transform already gives coordinates in meters, so no conversion is needed.

### Why this matters

The OSBS pipeline couldn't use `slope_aspect()` because `_gradient_horn_1981()` unconditionally applied the haversine-based degree-to-meter conversion. On UTM data, feeding meter values into `re * dtr * cos(lat_in_meters)` produces nonsense. The pipeline worked around this with `np.gradient(dem, pixel_size)`, but that introduced the N/S aspect sign bug (STATUS.md problem #4) because `np.gradient` and Horn 1981 have opposite `dzdy` sign conventions along the y-axis.

### Diff

**Before (original Swenson code):**

```python
def _gradient_horn_1981(self, dem, inside):
    """
    Calculate gradient of a dem.
    """
    warnings.filterwarnings(action='ignore', message='Invalid value encountered',
                               category=RuntimeWarning)
    # eight surrounding indices ordered as [N,NE,E,SE,S,SW,W,NW]
    inner_neighbors = self._select_surround_ravel(inside, dem.shape).T

    # elevation of central gridpoint's neighbors
    elev_neighbors = dem.flat[inner_neighbors]

    lon2d, lat2d = self._2d_crs_coordinates()
    dlon = np.subtract(lon2d.flat[inner_neighbors], lon2d.flat[inside])
    dlat = np.subtract(lat2d.flat[inner_neighbors], lat2d.flat[inside])

    # convert to meters
    re = 6.371e6
    dtr = np.pi/180
    dx = re * np.abs(np.multiply(dtr*dlon,np.cos(dtr*lat2d.flat[inside])))
    dy = re * np.abs(dtr*dlat)

    mean_dx = 0.5 * np.sum(dx[[2,6],:],axis=0) # average dx west and east
    mean_dy = 0.5 * np.sum(dy[[0,4],:],axis=0) # average dy south and north

    # for x gradient sum [NE,2xE,SE,-NW,-2xW,-SW]
    # for y gradient sum [NE,2xN,NW,-SE,-2xS,-SW]
    haxindices = [1,2,2,3]  #add
    hsxindices = [5,6,6,7]  #subtract
    hayindices = [0,0,1,7]  #add
    hsyindices = [3,4,4,5]  #subtract

    dzdx = (np.sum(elev_neighbors[haxindices,:],axis=0) - np.sum(elev_neighbors[hsxindices,:],axis=0)) / (8.*mean_dx)
    dzdy = (np.sum(elev_neighbors[hayindices,:],axis=0) - np.sum(elev_neighbors[hsyindices,:],axis=0)) / (8.*mean_dy)
    return [dzdx,dzdy]
```

**After:**

```python
# pgrid.py lines 4175-4236
#
def _gradient_horn_1981(self, dem, inside):
    """
    Calculate gradient of a dem.
    """
    # [ORIGINAL] Suppress runtime warnings from division in flat areas
    # where the gradient is zero/undefined.
    warnings.filterwarnings(action='ignore', message='Invalid value encountered',
                               category=RuntimeWarning)

    # [ORIGINAL] Eight surrounding indices ordered as [N,NE,E,SE,S,SW,W,NW].
    # This ordering is critical — the Horn stencil indices below (haxindices
    # etc.) reference positions in this specific order.
    inner_neighbors = self._select_surround_ravel(inside, dem.shape).T

    # [ORIGINAL] Elevation of central gridpoint's neighbors.
    elev_neighbors = dem.flat[inner_neighbors]

    # [ADDED - Phase A] Block comment explaining coordinate system.
    #
    # --- Convert pixel-index differences to physical distances ---
    #
    # _2d_crs_coordinates() returns CRS coordinates at pixel centers:
    #   - Geographic CRS → lon/lat in degrees
    #   - Projected CRS (e.g. UTM) → easting/northing in meters
    #
    # dx[k, i] and dy[k, i] are the coordinate offsets from pixel i
    # to its k-th neighbor (k in [N, NE, E, SE, S, SW, W, NW]).

    # [ORIGINAL] Get 2D coordinates and compute offsets to neighbors.
    # In the original Swenson code these were lon2d/lat2d and dlon/dlat;
    # renamed to x2d/y2d and dx/dy to be CRS-neutral.
    # For geographic CRS, dx/dy are in degrees.
    # For projected CRS, dx/dy are in meters.
    x2d, y2d = self._2d_crs_coordinates()
    dx = np.subtract(x2d.flat[inner_neighbors], x2d.flat[inside])
    dy = np.subtract(y2d.flat[inner_neighbors], y2d.flat[inside])

    # [ADDED - Phase A] CRS branch for unit conversion.
    #
    # Convert coordinate differences to physical cell spacings in meters.
    # abs() is taken because the Horn stencil cares about spacing magnitude,
    # not direction — the directional information is encoded in which
    # neighbors are summed vs subtracted (haxindices/hsxindices below).
    #
    # In the original Swenson code, the physical spacings were named dx/dy.
    # They are now named cell_dx/cell_dy to distinguish them from the
    # coordinate offsets dx/dy above.
    if self._crs_is_geographic():
        # [ORIGINAL] Geographic CRS: approximate meter distances from
        # degree offsets.
        # cell_dx uses a cos(lat) correction for longitude convergence
        # toward poles: 1 degree of longitude is shorter at high
        # latitudes.
        # cell_dy is simply arc length along a meridian: always
        # re * dtr * dlat regardless of latitude.
        cell_dx = _EARTH_RADIUS_M * np.abs(np.multiply(_DEG_TO_RAD*dx,
                                     np.cos(_DEG_TO_RAD*y2d.flat[inside])))
        cell_dy = _EARTH_RADIUS_M * np.abs(_DEG_TO_RAD*dy)
    else:
        # [ADDED - Phase A] Projected CRS (e.g. UTM): the affine
        # transform maps pixel indices directly to CRS coordinates in
        # linear units (meters for UTM). Coordinate differences are
        # already physical distances — no conversion needed. UTM
        # distortion is < 0.04% within a zone.
        #
        # [WHY] This is the key fix for slope/aspect on UTM data.
        # The original code's `re * dtr * cos(lat)` conversion is
        # nonsensical when lat is already in meters (e.g. 3286000m).
        # For UTM, abs(dx) and abs(dy) are already the physical
        # spacing between pixel centers.
        cell_dx = np.abs(dx)
        cell_dy = np.abs(dy)

    # [ORIGINAL] Horn 1981 uses the average spacing of the two cardinal
    # neighbors along each axis to normalize the weighted finite
    # difference. This makes the stencil symmetric for non-square pixels.
    #   mean_cell_dx = average of east (index 2) and west (index 6) spacings
    #   mean_cell_dy = average of north (index 0) and south (index 4) spacings
    #
    # [ADDED - Phase A] Comments clarifying the index references.
    # In the original code these were mean_dx/mean_dy; renamed to
    # mean_cell_dx/mean_cell_dy for consistency with cell_dx/cell_dy.
    mean_cell_dx = 0.5 * np.sum(cell_dx[[2,6],:],axis=0)
    mean_cell_dy = 0.5 * np.sum(cell_dy[[0,4],:],axis=0)

    # [ORIGINAL] Horn 1981 stencil weights.
    # For x gradient: sum [NE, 2xE, SE] - sum [NW, 2xW, SW]
    # For y gradient: sum [NE, 2xN, NW] - sum [SE, 2xS, SW]
    # The doubled indices (E appears twice, N appears twice) implement
    # the 2x weighting of cardinal neighbors.
    haxindices = [1,2,2,3]  #add
    hsxindices = [5,6,6,7]  #subtract
    hayindices = [0,0,1,7]  #add
    hsyindices = [3,4,4,5]  #subtract

    # [ORIGINAL] Final gradient computation. The 8 in the denominator
    # is the sum of Horn stencil weights (1+2+1+1+2+1 = 8).
    dzdx = (np.sum(elev_neighbors[haxindices,:],axis=0)
            - np.sum(elev_neighbors[hsxindices,:],axis=0)) / (8.*mean_cell_dx)
    dzdy = (np.sum(elev_neighbors[hayindices,:],axis=0)
            - np.sum(elev_neighbors[hsyindices,:],axis=0)) / (8.*mean_cell_dy)
    return [dzdx,dzdy]
```

### What changed vs what didn't

| Element | Changed? | Notes |
|---------|----------|-------|
| Neighbor selection (`_select_surround_ravel`) | No | Purely topological |
| Elevation lookup | No | |
| `x2d`/`y2d` (was `lon2d`/`lat2d`) | **Renamed** | CRS-neutral names from `_2d_crs_coordinates()` |
| `dx`/`dy` coordinate offsets (was `dlon`/`dlat`) | **Renamed** | Same subtraction, CRS-neutral names |
| `cell_dx`/`cell_dy` physical spacings (was `dx`/`dy`) | **Renamed** | Disambiguated from coordinate offsets above |
| `mean_cell_dx`/`mean_cell_dy` (was `mean_dx`/`mean_dy`) | **Renamed** | Consistent with `cell_dx`/`cell_dy` |
| `re`/`dtr` constants | **Replaced** | Now uses module-level `_EARTH_RADIUS_M`/`_DEG_TO_RAD` |
| Degree-to-meter conversion | No | Moved into `if` branch, uses `cell_dx`/`cell_dy` |
| UTM branch (`abs(dx)`, `abs(dy)`) | **Added** | New `else` clause, stored as `cell_dx`/`cell_dy` |
| Horn stencil weights and final division | No | Now uses `mean_cell_dx`/`mean_cell_dy` |

### Why `abs()` in the UTM branch?

The Horn stencil encodes direction through which neighbors are *added* vs *subtracted* (haxindices vs hsxindices). The `cell_dx`/`cell_dy` denominators should be pure spacing magnitudes — always positive. For geographic CRS, `abs()` was already applied to the degree offsets. For UTM, the affine transform can produce negative `dy` (northing decreases with row index in a standard north-up GeoTIFF), so `abs()` is needed to get the magnitude.

---

## 6. `slope_aspect()` (lines 2240-2312)

### No code change — CRS awareness is encapsulated in `_gradient_horn_1981()`

`slope_aspect()` is the public-facing method that calls `_gradient_horn_1981()` and converts the raw gradient into slope magnitude and aspect angle. Because all CRS-specific logic lives in `_gradient_horn_1981()`, `slope_aspect()` itself needed no modification beyond a docstring update.

Shown here for completeness to demonstrate the call chain.

### Diff

**Before docstring:**
```python
def slope_aspect(self, dem, ...):
    """
    Computes the slope and aspect from a digital elevation grid.
    ...
```

**After docstring:**
```python
def slope_aspect(self, dem, ...):
    """
    Computes the slope and aspect from a digital elevation grid.
    The Horn 1981 stencil uses CRS-appropriate distance normalization
    (haversine+cos(lat) for geographic, uniform pixel spacing for projected/UTM).
    ...
```

### Full current code

```python
# pgrid.py lines 2240-2312
#
def slope_aspect(self, dem, slope_out_name='slope', aspect_out_name='aspect',
                 nodata_in_dem=None, nodata_out=np.nan,
                 inplace=True, apply_mask=False, ignore_metadata=False,
                 **kwargs):
    """
    Computes the slope and aspect from a digital elevation grid.
    The Horn 1981 stencil uses CRS-appropriate distance normalization
    (haversine+cos(lat) for geographic, uniform pixel spacing for
    projected/UTM).
    ...
    """

    nodata_in_dem = self._check_nodata_in(dem, nodata_in_dem)
    properties = {'nodata' : nodata_out}
    metadata = {}

    # initialize array
    dem = self._input_handler(dem, apply_mask=apply_mask,
                              nodata_view=nodata_in_dem,
                              properties=properties,
                              ignore_metadata=ignore_metadata, **kwargs)

    try:
        if nodata_in_dem is None:
            dem_mask = np.array([]).astype(int)
        else:
            if np.isnan(nodata_in_dem):
                dem_mask = np.where(np.isnan(dem.ravel()))[0]
            else:
                dem_mask = np.where(dem.ravel() == nodata_in_dem)[0]
        # Make sure nothing flows to the nodata cells
        dem.flat[dem_mask] = dem.max() + 1
        inside = self._inside_indices(dem, mask=dem_mask)

        # [ORIGINAL] This is where CRS awareness enters the call chain.
        # _gradient_horn_1981 detects CRS and uses appropriate distance
        # formulas internally. slope_aspect doesn't need to know.
        grad = self._gradient_horn_1981(dem, inside)

        dzdx = grad[0]
        dzdy = grad[1]

        # [ORIGINAL] Calculate slope from gradient.
        # Slope = magnitude of the gradient vector. Sign-independent,
        # so CRS handling doesn't affect this step.
        slope = np.zeros(dem.shape)
        slope.flat[inside] = np.sqrt(dzdx*dzdx+dzdy*dzdy)

        # [ORIGINAL] Calculate aspect from gradient.
        # Steepest descent is along the negative of the gradient.
        # arctan2(-dzdx, -dzdy) gives clockwise-from-north azimuth.
        #
        # [WHY this doesn't have the N/S bug] _gradient_horn_1981()
        # references neighbors by compass direction (N=0, NE=1, E=2,
        # etc.) and normalizes spacings with abs(). This always returns
        # dzdy = d(elev)/d(north) regardless of CRS or array ordering.
        # So arctan2(-dzdx, -dzdy) is always correct here.
        #
        # Compare to np.gradient(dem, pixel_size) which follows array
        # index order: dzdy = d(elev)/d(row) = d(elev)/d(south) for
        # north-up rasters — opposite sign. That's what caused the
        # N/S swap in run_pipeline.py.
        aspect = np.zeros(dem.shape)
        aspect.flat[inside] = (180.0/np.pi)*np.arctan2(-dzdx,-dzdy)

        # convert from [-180,180] to [0-360]
        aspect[(aspect < 0)]+=360

        self._output_handler(data=slope, out_name=slope_out_name,
                             properties=properties,
                             inplace=inplace, metadata=metadata)
        self._output_handler(data=aspect, out_name=aspect_out_name,
                             properties=properties,
                             inplace=inplace, metadata=metadata)

    except:
        raise
    return
```

---

## 7. `river_network_length_and_slope()` (lines 3176-3290)

Found during audit as a "missed haversine" — not called by the OSBS pipeline at the time of the audit, but would have produced wrong results if used with UTM data.

### What it does

Estimates total river network length and mean slope by summing segment-by-segment distances along extracted stream profiles. Each segment connects consecutive pixels along a reach.

### Diff

**Before (original Swenson code):**

```python
dtr = np.pi/180.
re = 6.371e6
# ... (inside the per-profile loop) ...
plon = np.asarray((fdir.affine * (xi, yi))[0])
plat = np.asarray((fdir.affine * (xi, yi))[1])
plon,plat = plon[pmask],plat[pmask]
dlon = plon[:-1] - plon[1:]
dlat = plat[:-1] - plat[1:]
dist = np.power(np.sin(dtr*dlat/2),2) + np.cos(dtr*plat[:-1]) \
       * np.cos(dtr*plat[1:]) \
       * np.power(np.sin(dtr*dlon/2),2)
length = np.sum(re * 2 * np.arctan2(np.sqrt(dist),np.sqrt(1-dist)))
```

**After:**

```python
# pgrid.py lines 3176-3290 (relevant excerpt)
#
# [ADDED - Phase A] CRS check once, outside the loop.
# Avoids repeated method calls per profile.
is_geographic = self._crs_is_geographic()

# [ORIGINAL] Loop over extracted stream profiles.
for index, profile in enumerate(profiles):
    endpoint = profiles[connections[index]][0]
    yi, xi = np.unravel_index(profile.tolist(), fdir.shape)

    # extract_profiles does not mask out missing values; apply mask here
    pmask = mask[yi,xi]

    # [ADDED - Phase A] Clarifying comment.
    # px/py are affine-transformed coordinates along the stream
    # reach profile. For geographic CRS these are lon/lat in degrees;
    # for projected CRS (e.g. UTM) these are easting/northing in meters.

    # [ORIGINAL] Get coordinates and compute consecutive differences.
    # In the original Swenson code these were named plon/plat and
    # dlon/dlat; renamed to px/py and dx/dy to be CRS-neutral.
    px = np.asarray((fdir.affine * (xi, yi))[0])
    py = np.asarray((fdir.affine * (xi, yi))[1])
    px,py = px[pmask],py[pmask]
    dx = px[:-1] - px[1:]
    dy = py[:-1] - py[1:]

    # [ADDED - Phase A] CRS branch for segment length computation.
    if is_geographic:
        # [ORIGINAL] Geographic CRS: coordinates are degrees. Use
        # haversine to compute great-circle segment lengths along
        # the reach. Identical to Swenson's original code.
        dist = np.power(np.sin(_DEG_TO_RAD*dy/2),2) \
               + np.cos(_DEG_TO_RAD*py[:-1]) \
               * np.cos(_DEG_TO_RAD*py[1:]) \
               * np.power(np.sin(_DEG_TO_RAD*dx/2),2)
        length = np.sum(_EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(dist),
                                             np.sqrt(1-dist)))
    else:
        # [ADDED - Phase A] Projected CRS: coordinates are already
        # in linear units (meters for UTM). Euclidean distance
        # between consecutive profile pixels.
        length = np.sum(np.sqrt(dx**2 + dy**2))

    # [ORIGINAL] Rest of loop body (elevation, slope) unchanged...
    elevation = self.inflated_dem[yi,xi]
    elevation = elevation[pmask]
    # ...
```

### What changed vs what didn't

| Element | Changed? | Notes |
|---------|----------|-------|
| `dtr`/`re` constants | **Replaced** | Now uses module-level `_DEG_TO_RAD`/`_EARTH_RADIUS_M`; moved from before loop into `if` branch |
| `is_geographic` check | **Added** | Computed once before the loop |
| `px`/`py` (was `plon`/`plat`) | **Renamed** | CRS-neutral names |
| `dx`/`dy` (was `dlon`/`dlat`) | **Renamed** | CRS-neutral names |
| Haversine segment lengths | No | Moved into `if` branch, logic identical |
| Euclidean segment lengths | **Added** | New `else` clause |
| `rx`/`ry` midpoint coords (was `rlon`/`rlat`) | **Renamed** | CRS-neutral names (used later in loop for reach midpoints) |
| Profile extraction, elevation, slope logic | No | |

---

## 8. `_2d_crs_coordinates()` (lines 1762-1787)

### Variable rename — docstring updated, return values renamed

Despite the method name suggesting geographic coordinates, it returns coordinates in whatever CRS the grid uses. For geographic CRS, these are lon/lat in degrees. For projected CRS (UTM), these are easting/northing in meters.

The original Swenson code named the return values `lon2d`/`lat2d` and used `geocoords` for the intermediate affine result. These were renamed to `x2d`/`y2d` and `coords` to be CRS-neutral -- the old names were misleading for projected CRS where the values are easting/northing in meters, not longitude/latitude.

### Diff

**Before docstring:**
```python
def _2d_crs_coordinates(self):
    """
    2D geographic coordinate arrays

    """
```

**After docstring:**
```python
def _2d_crs_coordinates(self):
    """Return 2D coordinate arrays in the grid's CRS.

    Returns [x2d, y2d] — 2D arrays of CRS coordinates at pixel centers.
      - Geographic CRS: x2d = longitude (degrees), y2d = latitude (degrees)
      - Projected CRS:  x2d = easting (meters),    y2d = northing (meters)

    The affine transform maps pixel indices (col, row) to CRS coordinates.
    The 0.5*dx / 0.5*dy offset shifts from pixel corner to pixel center.
    """
```

### Full current code

```python
# pgrid.py lines 1762-1787
#
def _2d_crs_coordinates(self):
    """Return 2D coordinate arrays in the grid's CRS.

    Returns [x2d, y2d] — 2D arrays of CRS coordinates at pixel centers.
      - Geographic CRS: x2d = longitude (degrees), y2d = latitude (degrees)
      - Projected CRS:  x2d = easting (meters),    y2d = northing (meters)

    The affine transform maps pixel indices (col, row) to CRS coordinates.
    The 0.5*dx / 0.5*dy offset shifts from pixel corner to pixel center.
    """

    # [ORIGINAL] All code below is unchanged in logic; variable names
    # were updated to be CRS-neutral.
    # The affine transform encodes the CRS-to-pixel mapping:
    #   x.c = x origin (left edge), x.f = y origin (top edge)
    #   x.a = pixel width (dx), x.e = pixel height (dy, typically negative)
    x = self.affine
    x0, y0, dx, dy = x.c, x.f, x.a, x.e
    ys, xs = self.shape

    # Create 2D index arrays
    i2d = np.tile(range(xs),(ys,1))        # column indices
    j2d = np.tile(range(ys),(xs,1)).T      # row indices

    # Apply affine transform: (col, row) → (x_coord, y_coord)
    coords = self.affine * (i2d.flatten(),j2d.flatten())

    # Shift from pixel corner to pixel center
    x2d = coords[0].reshape(self.shape) + 0.5*dx
    y2d = coords[1].reshape(self.shape) + 0.5*dy

    return [x2d,y2d]
```

### Why this matters for understanding the UTM branches

When `_gradient_horn_1981()` or `compute_hand()` compute:
```python
dx = x2d - x2d.flat[neighbor]
```

For geographic CRS, `dx` is a degree offset that needs haversine conversion. For UTM, `dx` is already a meter offset that needs no conversion. The CRS branches exist because this one method returns values with fundamentally different units depending on the CRS.

---

## 9. Deprecation Fixes (non-UTM)

These five changes fix deprecation warnings unrelated to CRS handling. They were done in the same branch to establish a clean working state before the UTM modifications.

### 9.1. `LooseVersion` import (line 9)

```python
# Before:
from distutils.version import LooseVersion

# After:
from looseversion import LooseVersion
```

`distutils` was removed in Python 3.12. The `looseversion` package is a standalone replacement. This was the only deprecation warning that was a hard error on Python 3.12+; the others were warnings.

### 9.2-9.4. `np.in1d` → `np.isin` (lines 1292, 3129, 3142)

```python
# Before (3 locations):
invalid_cells = ~np.in1d(fdir.ravel(), dirmap)     # line 1292
is_fork = np.in1d(end, forks_end)                   # line 3129
no_pred = ~np.isin(start, end)                      # line 3142  (was np.in1d)

# After:
invalid_cells = ~np.isin(fdir.ravel(), dirmap)      # line 1292
is_fork = np.isin(end, forks_end)                    # line 3129
no_pred = ~np.isin(start, end)                       # line 3142
```

`np.in1d` was deprecated in NumPy 1.24 in favor of `np.isin`. Identical behavior; `np.isin` is the recommended replacement.

### 9.5. `Series._append` → `pd.concat` (line 3970)

```python
# Before:
gradfactor = (0.9 * (minsteps / gradmax)).replace(np.inf, 0)._append(pd.Series({0 : 0}))

# After:
gradfactor = pd.concat([(0.9 * (minsteps / gradmax)).replace(np.inf, 0), pd.Series({0 : 0})])
```

`Series._append` (private method) was removed in pandas 2.0. `pd.concat` is the standard replacement.

---

## Summary of All Changes

| Section | Lines | Type | Geographic branch | Projected branch |
|---------|-------|------|-------------------|-----------------|
| `_crs_is_geographic()` | 129-148 | **New method** | Returns `True` | Returns `False` |
| DTND in `compute_hand()` | 1958-1997 | **CRS branch** | Haversine (original) | Euclidean (added) |
| AZND in `compute_hand()` | 1999-2029 | **CRS branch** | Spherical bearing (original) | Planar arctan2 (added) |
| `_gradient_horn_1981()` | 4175-4236 | **CRS branch** | `re * dtr * cos(lat)` (original) | `abs(dx)` / `abs(dy)` (added) |
| `slope_aspect()` | 2240-2312 | Docstring only | — | — |
| `river_network_length_and_slope()` | 3176-3290 | **CRS branch** | Haversine (original) | Euclidean (added) |
| `_2d_crs_coordinates()` | 1762-1787 | Variable rename + docstring | — | — |
| Deprecation fixes | various | Non-UTM | — | — |

All geographic branches preserve the original Swenson code exactly — MERIT validation (stages 1-9, >0.95 correlation on 5/6 parameters) remains bit-identical.
