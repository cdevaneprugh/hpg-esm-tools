# The N/S Aspect Swap Bug

A standalone technical reference explaining the sign-convention bug that
swapped north and south aspects in the OSBS hillslope pipeline. This document
traces the bug from first principles — data layout conventions, gradient
algorithms, trigonometry — through to its downstream effects on hillslope
parameters.

**Status:** The bug has been fixed. An interim sign correction was applied to
`run_pipeline.py` (2026-02-10), and Phase A added full UTM CRS support to
pgrid (2026-02-12). Phase D will replace `np.gradient` entirely with pgrid's
Horn 1981 stencil.

---

## Table of Contents

1. [What Aspect Means Physically](#1-what-aspect-means-physically)
2. [The Aspect Formula](#2-the-aspect-formula)
3. [How GeoTIFF Rasters Are Stored](#3-how-geotiff-rasters-are-stored)
4. [How np.gradient Works](#4-how-npgradient-works)
5. [How pgrid's Horn 1981 Works](#5-how-pgrids-horn-1981-works)
6. [Swenson's calc_gradient (Full Trace)](#6-swensons-calc_gradient-full-trace)
7. [The Sign Convention Table](#7-the-sign-convention-table)
8. [The Bug: Double Negation](#8-the-bug-double-negation)
9. [Worked Example: V-Valley DEM](#9-worked-example-v-valley-dem)
10. [Why the Error is Small in the V-Valley](#10-why-the-error-is-small-in-the-v-valley)
11. [The Reflection Pattern](#11-the-reflection-pattern)
12. [Where the Bug Matters](#12-where-the-bug-matters)
13. [Where the Bug Does NOT Matter](#13-where-the-bug-does-not-matter)
14. [The Four Code Paths and Their Status](#14-the-four-code-paths-and-their-status)
15. [Why pgrid's Horn 1981 is Immune](#15-why-pgrids-horn-1981-is-immune)
16. [History and Resolution](#16-history-and-resolution)

---

## 1. What Aspect Means Physically

Aspect is the compass direction of steepest descent from a point on a
surface. It answers the question: "if I pour water on this pixel, which
compass direction does it initially flow?"

The standard convention measures aspect in **degrees clockwise from north**,
with a range of 0 to 360:

| Direction | Aspect (degrees) |
|-----------|-------------------|
| North     | 0 (or 360)        |
| East      | 90                |
| South     | 180               |
| West      | 270               |

A pixel whose steepest descent points due east has aspect = 90. Due south =
180. Due west = 270. Northeast = 45. Southwest = 225.

Aspect is undefined on a perfectly flat surface (no slope, no preferred
direction). In practice, floating-point DEMs rarely have exactly zero slope,
so aspect is computed everywhere and the flat-surface case is handled by
convention (often 0 or NaN).

---

## 2. The Aspect Formula

### From gradient to aspect

The elevation gradient at a point is a 2D vector pointing in the direction of
steepest **ascent**:

```
gradient = (dz/dx_east, dz/dy_north)
```

where `dz/dx_east` is the rate of elevation change per unit distance
eastward, and `dz/dy_north` is the rate of elevation change per unit distance
northward. Steepest descent is the **negative** of the gradient:

```
steepest_descent = (-dz/dx_east, -dz/dy_north)
```

Aspect is the compass bearing of this descent vector. The standard formula
is:

```
aspect = arctan2(east_component_of_descent, north_component_of_descent)
       = arctan2(-dz/dx_east, -dz/dy_north)
```

### What arctan2 does

`arctan2(y, x)` returns the angle in radians of the vector `(x, y)` measured
counterclockwise from the positive x-axis. Its range is `(-pi, pi]`.

But we are calling it as `arctan2(east, north)` — the east component is
passed as the first argument (`y`) and the north component as the second
(`x`). This swaps the axes relative to the standard math convention.

To see why this gives a compass bearing, consider what happens:
- The north axis plays the role of the x-axis.
- The east axis plays the role of the y-axis.
- `arctan2(east, north)` measures the angle counterclockwise from the north
  axis toward the east axis.

"Counterclockwise from north toward east" **is** the definition of "clockwise
from north" on a map (because east is to the right of north on a standard
map). So `arctan2(east, north)` directly gives the compass bearing.

### Quadrant-by-quadrant verification

Using `arctan2(east_descent, north_descent)`:

| Descent direction | east_descent | north_descent | arctan2 result | After +360 if <0 |
|-------------------|--------------|---------------|----------------|-------------------|
| Pure North        | 0            | +1            | 0.0            | 0                 |
| Pure East         | +1           | 0             | 90.0           | 90                |
| Pure South        | 0            | -1            | 180.0          | 180               |
| Pure West         | -1           | 0             | -90.0          | 270               |
| Northeast         | +1           | +1            | 45.0           | 45                |
| Southeast         | +1           | -1            | 135.0          | 135               |
| Southwest         | -1           | -1            | -135.0         | 225               |
| Northwest         | -1           | +1            | -45.0          | 315               |

The raw `arctan2` output ranges from -180 to +180 degrees. Adding 360 to
negative values converts to the 0-360 convention. Every row matches the
expected compass bearing.

---

## 3. How GeoTIFF Rasters Are Stored

### The standard layout

A GeoTIFF stores elevation as a 2D array with this convention:

- **Row 0** is the **northernmost** row of pixels.
- **Row index increases southward.** Row 1 is south of row 0.
- **Column 0** is the **westernmost** column.
- **Column index increases eastward.** Column 1 is east of column 0.

This means the array is stored "upside down" relative to a map: the top of
the array (low row indices) is the north, the bottom (high row indices) is
the south.

### The affine transform

The mapping from pixel coordinates `(col, row)` to geographic/projected
coordinates `(x, y)` is encoded in a 6-parameter affine transform:

```
x = a * col + b * row + c
y = d * col + e * row + f
```

For a north-up raster with no rotation, `b = d = 0`, and:

```
Affine(a, 0, c, 0, e, f)
```

where:
- `a` = pixel width (positive, e.g., 1.0 for 1m pixels)
- `c` = x-coordinate of the upper-left pixel corner (easting)
- `e` = pixel height (**negative**, e.g., -1.0 for 1m pixels)
- `f` = y-coordinate of the upper-left pixel corner (northing)

The **negative `e`** is the key: it encodes the fact that northing
**decreases** as row index **increases**. Moving one row down in the array
moves one pixel south in geographic space.

### Concrete example: NEON DTM tile

The NEON DTM tiles at OSBS use EPSG:32617 (UTM zone 17N) with 1m pixels.
A typical tile has the affine transform:

```python
Affine(1.0, 0.0, 404000.0,
       0.0, -1.0, 3287000.0)
```

This means:
- Easting starts at 404000 m, increases by 1m per column
- Northing starts at 3287000 m, **decreases** by 1m per row
- Pixel at `[row=0, col=0]` is at (404000.5, 3286999.5) — the 0.5 offset
  is the pixel center
- Pixel at `[row=999, col=0]` is at (404000.5, 3286000.5) — 999m further
  south

The critical implication for gradient calculations: **the row axis of the
array runs south, not north.** Any derivative computed along axis 0 (the row
axis) is a derivative with respect to the southward direction.

---

## 4. How np.gradient Works

### Basic behavior

`np.gradient(dem, pixel_size)` computes centered finite differences along
each array axis. For a 2D array, it returns a tuple of two arrays:

```python
dzdy, dzdx = np.gradient(dem, pixel_size)
```

- `dzdy` = derivative along axis 0 (rows)
- `dzdx` = derivative along axis 1 (columns)

The naming `dzdy, dzdx` is a common convention when unpacking, but numpy
itself just returns `(d/d(axis_0), d/d(axis_1))`. It has no concept of
geographic directions.

### What "axis 0" means geographically

For a standard GeoTIFF (row 0 = north, row index increases southward):

- **Axis 0 = rows = southward.** So `dzdy = d(elev)/d(row) = d(elev)/d(south)`.
- **Axis 1 = columns = eastward.** So `dzdx = d(elev)/d(col) = d(elev)/d(east)`.

The x-component (`dzdx`) is in the correct geographic orientation — positive
means elevation increases eastward.

The y-component (`dzdy`) is in the **opposite** direction from geographic
north:
```
dzdy_numpy = d(elev)/d(south) = -d(elev)/d(north)
```

### The central difference formula

For an interior pixel at `[r, c]`, numpy computes:

```
dzdy[r, c] = (dem[r+1, c] - dem[r-1, c]) / (2 * pixel_size)
```

This is `(south_neighbor - north_neighbor) / (2 * spacing)`.

### Worked example: 3x3 DEM with elevation higher to the north

```
dem = [[12, 12, 12],   # row 0 = north (high)
       [10, 10, 10],   # row 1 = center
       [ 8,  8,  8]]   # row 2 = south (low)
```

At the center pixel `[1, 1]`:
```
dzdy[1,1] = (dem[2,1] - dem[0,1]) / (2 * 1.0)
          = (8 - 12) / 2
          = -2.0
```

The result is **-2.0**, meaning elevation **decreases** as row index
increases (i.e., decreases southward). This is `d(elev)/d(south) = -2.0`.

Converting to the northward convention:
```
d(elev)/d(north) = -d(elev)/d(south) = -(-2.0) = +2.0
```

Which is correct: elevation increases northward at 2.0 m/m.

The sign flip between numpy's row-axis convention and geographic north is the
root cause of the aspect bug.

---

## 5. How pgrid's Horn 1981 Works

pgrid's `_gradient_horn_1981()` (`pgrid.py:4172-4226`) implements the Horn
1981 weighted finite-difference stencil. Unlike `np.gradient`, it explicitly
references compass directions and produces `dzdy = d(elev)/d(north)`
regardless of how the array is stored.

### Step 1: Get 8 neighbor indices

`_select_surround_ravel()` (`pgrid.py:3728-3740`) maps each interior pixel's
flat index `i` to its 8 neighbors:

```python
def _select_surround_ravel(self, i, shape):
    offset = shape[1]       # number of columns
    return np.array([
        i + 0 - offset,     # [0] N:  one row up (more north)
        i + 1 - offset,     # [1] NE: one row up, one col right
        i + 1 + 0,          # [2] E:  one col right
        i + 1 + offset,     # [3] SE: one row down, one col right
        i + 0 + offset,     # [4] S:  one row down (more south)
        i - 1 + offset,     # [5] SW: one row down, one col left
        i - 1 + 0,          # [6] W:  one col left
        i - 1 - offset,     # [7] NW: one row up, one col left
    ]).T
```

The neighbor indices are ordered by compass direction: **[N, NE, E, SE, S,
SW, W, NW]**. The index arithmetic encodes the GeoTIFF layout:
- `i - offset` = one row up = one row north (since row 0 is north)
- `i + offset` = one row down = one row south

This is where the geographic meaning enters: the *labels* N, S, E, W are
embedded in the code structure.

### Step 2: Get CRS coordinates

`_2d_crs_coordinates()` (`pgrid.py:1769-1792`) builds 2D coordinate arrays
from the affine transform:

```python
def _2d_crs_coordinates(self):
    x = self.affine
    x0, y0, dx, dy = x.c, x.f, x.a, x.e
    # ...
    x2d = coords[0].reshape(self.shape) + 0.5*dx
    y2d = coords[1].reshape(self.shape) + 0.5*dy
    return [x2d, y2d]
```

For UTM, `x2d` contains eastings and `y2d` contains northings. Because the
affine has negative y-scaling (`e = -1.0`), northing decreases with row
index: `y2d[0,:] > y2d[1,:] > ... > y2d[-1,:]`.

### Step 3: Compute coordinate offsets

Back in `_gradient_horn_1981()`, the coordinate offsets from each pixel to
its neighbors are computed:

```python
x2d, y2d = self._2d_crs_coordinates()
dx = np.subtract(x2d.flat[inner_neighbors], x2d.flat[inside])
dy = np.subtract(y2d.flat[inner_neighbors], y2d.flat[inside])
```

For the N neighbor (index 0): `dy[0, :] = y2d[N_neighbor] - y2d[center]`.
Since the N neighbor has higher northing, `dy[0, :] > 0`.

For the S neighbor (index 4): `dy[4, :] = y2d[S_neighbor] - y2d[center]`.
Since the S neighbor has lower northing, `dy[4, :] < 0`.

### Step 4: Take absolute value of spacings (critical)

```python
if self._crs_is_geographic():
    cell_dx = _EARTH_RADIUS_M * np.abs(np.multiply(_DEG_TO_RAD*dx, ...))
    cell_dy = _EARTH_RADIUS_M * np.abs(_DEG_TO_RAD*dy)
else:
    # Projected (UTM): offsets are already in meters
    cell_dx = np.abs(dx)
    cell_dy = np.abs(dy)
```

**`np.abs()` discards the sign of the offset.** After this step, `cell_dy`
is a positive distance in meters for all neighbors, regardless of whether
they are north or south.

This is the design choice that makes the Horn stencil sign-convention-safe:
directional information is encoded entirely in **which** neighbors get
added vs subtracted in the stencil, **not** in the sign of the spacing.

### Step 5: Apply the Horn 1981 stencil

```python
# Mean spacing from cardinal neighbors
mean_cell_dy = 0.5 * np.sum(cell_dy[[0,4],:], axis=0)   # avg of N and S

# Y-gradient stencil:
hayindices = [0, 0, 1, 7]    # add: N, N, NE, NW (north-side neighbors)
hsyindices = [3, 4, 4, 5]    # subtract: SE, S, S, SW (south-side neighbors)

dzdy = (np.sum(elev_neighbors[hayindices,:], axis=0)
      - np.sum(elev_neighbors[hsyindices,:], axis=0)) / (8. * mean_cell_dy)
```

Expanding the stencil explicitly:

```
dzdy = (elev_N + elev_N + elev_NE + elev_NW
      - elev_SE - elev_S - elev_S - elev_SW) / (8 * mean_dy)
```

The numerator is: `(sum of north-side elevations) - (sum of south-side
elevations)`. If the terrain is higher to the north, this is positive.
The denominator `8 * mean_dy` is always positive (absolute spacing).

Therefore: **`dzdy` is positive when elevation increases northward.**

```
dzdy_horn = d(elev)/d(north)
```

### Step 6: The same logic for dzdx

```python
haxindices = [1, 2, 2, 3]    # add: NE, E, E, SE (east-side)
hsxindices = [5, 6, 6, 7]    # subtract: SW, W, W, NW (west-side)

dzdx = (np.sum(elev_neighbors[haxindices,:], axis=0)
      - np.sum(elev_neighbors[hsxindices,:], axis=0)) / (8. * mean_cell_dx)
```

Similarly: **`dzdx` is positive when elevation increases eastward.**

```
dzdx_horn = d(elev)/d(east)
```

### Result

pgrid's `_gradient_horn_1981()` returns `[dzdx, dzdy]` where:
- `dzdx = d(elev)/d(east)` — same as np.gradient
- `dzdy = d(elev)/d(north)` — **opposite** to np.gradient

---

## 6. Swenson's calc_gradient (Full Trace)

`calc_gradient()` (`geospatial_utils.py:129-162`) is the third gradient
implementation in the codebase. It is NOT used for aspect computation — only
for the FFT Laplacian. We trace it fully to complete the sign-convention
inventory.

### Step 1: np.gradient call (line 137)

```python
dzdy, dzdx = np.gradient(z)
```

Same as Section 4: `dzdy = d(elev)/d(row) = d(elev)/d(south)`. No spacing
argument is passed, so derivatives are in units of "per row index" (not per
meter). The sign convention is identical to bare `np.gradient`.

### Step 2: Horn 1981 averaging (lines 139-153)

```python
dzdy2, dzdx2 = np.zeros(dzdy.shape), np.zeros(dzdx.shape)

# Edge handling: 3-point average
eind = np.asarray([0, 0, 1])
dzdx2[0, :] = np.mean(dzdx[eind, :], axis=0)
dzdy2[:, 0] = np.mean(dzdy[:, eind], axis=1)

eind = np.asarray([-2, -1, -1])
dzdx2[-1, :] = np.mean(dzdx[eind, :], axis=0)
dzdy2[:, -1] = np.mean(dzdy[:, eind], axis=1)

# Interior: 4-point average with weights [-1, 0, 0, 1]
ind = np.asarray([-1, 0, 0, 1])
for n in range(1, dzdx.shape[0] - 1):
    dzdx2[n, :] = np.mean(dzdx[n + ind, :], axis=0)
for n in range(1, dzdy.shape[1] - 1):
    dzdy2[:, n] = np.mean(dzdy[:, n + ind], axis=1)
```

This is a 1D smoothing pass applied to np.gradient's output. For each
interior row `n`, the x-gradient is averaged across rows `[n-1, n, n, n+1]`
(the double `n` gives the center pixel twice the weight). Similarly, for each
interior column, the y-gradient is averaged across columns.

This is NOT the same as pgrid's 8-neighbor 2D stencil — it is two
independent 1D smoothing passes along the row and column axes, respectively.
Swenson calls this "Horn 1981" in the method name string, referring to the
fact that Horn's original stencil can be decomposed as a product of two 1D
operations, but the result differs from pgrid's implementation because the
smoothing directions are applied separately rather than jointly.

Crucially, **the sign convention is unchanged.** Averaging values across
neighboring rows does not flip signs. `dzdy2` is still
`d(elev)/d(row) = d(elev)/d(south)`.

### Step 3: Physical spacing (lines 155-162)

```python
# Calculate spacing (geographic CRS: degrees to meters)
dx = re * dtr * np.abs(lon[0] - lon[1])
dy = re * dtr * np.abs(lat[0] - lat[1])

dx2d = dx * np.tile(np.cos(dtr * lat), (lon.size, 1)).T
dy2d = dy * np.ones((lat.size, lon.size))

return [dzdx2 / dx2d, dzdy2 / dy2d]
```

- `re` is the Earth's radius (~6,371 km).
- `dtr` is degrees to radians.
- `np.abs(lon[0] - lon[1])` gives the unsigned degree spacing between
  adjacent grid columns.
- `dx` is the equatorial arc length for that longitude spacing. The next
  line, `dx2d = dx * cos(lat)`, applies the latitude-dependent correction
  for meridian convergence.
- `dy` is the arc length for the latitude spacing (constant, since meridians
  are great circles).
- `dy2d` is a constant positive array (same `dy` at all grid points).

Division by `dy2d` converts units from "per row index" to "per meter" but
**does not change the sign**. A positive `dzdy2` divided by a positive `dy2d`
remains positive.

### Result

`calc_gradient` returns `[dzdx/dx2d, dzdy/dy2d]` where dzdy is still
`d(elev)/d(south)`. The physical spacing conversion changes units but
preserves sign.

### Why this doesn't cause an aspect bug

`calc_gradient` is ONLY called for the FFT Laplacian computation:

```python
grad = calc_gradient(elev, lon, lat)       # [dzdx, dzdy]
x = calc_gradient(grad[0], lon, lat)       # [d²z/dxdx, d²z/dydx]
laplac = x[0]                              # d²z/dx²
x = calc_gradient(grad[1], lon, lat)       # [d²z/dxdy, d²z/dydy]
laplac += x[1]                             # d²z/dx² + d²z/dy²
```

The Laplacian is `d^2z/dx^2 + d^2z/dy^2`. Since `d^2z/dy^2 = d/dy(dz/dy)`,
applying the same derivative operator twice causes the sign to cancel:

```
d/d(south)[d/d(south)(z)] = d^2z/d(south)^2 = d^2z/d(north)^2
```

The Laplacian is invariant under coordinate reflection. It does not matter
whether the y-derivative is computed as d/d(north) or d/d(south) — the
second derivative is the same either way. So `calc_gradient`'s sign
convention is irrelevant for this use case.

Aspect comes exclusively from `slope_aspect()` -> `_gradient_horn_1981()`,
which has a different sign convention. These two gradient implementations
never interact.

### The OSBS pipeline's calc_gradient_utm

The OSBS pipeline has its own copy at `run_pipeline.py:275-296`:

```python
def calc_gradient_utm(z, dx=1.0):
    dzdy, dzdx = np.gradient(z)
    # ... same Horn 1981 averaging ...
    return (dzdx2 / dx, dzdy2 / dx)
```

This is a direct port of Swenson's `calc_gradient` adapted for UTM (uniform
spacing `dx` in meters instead of lat/lon-dependent spacing). The sign
convention is identical: `dzdy = d(elev)/d(south)`. And like Swenson's
version, it is only used for the FFT Laplacian, where the sign doesn't
matter.

---

## 7. The Sign Convention Table

Four independent gradient implementations exist in the codebase:

| Source | Returns | dzdy meaning | Sign vs d(elev)/d(north) | Used for |
|--------|---------|-------------|--------------------------|----------|
| pgrid `_gradient_horn_1981()` | `[dzdx, dzdy]` | d(elev)/d(north) | **same** | slope, aspect |
| `np.gradient(dem, px)` | `(dzdy, dzdx)` | d(elev)/d(south) | **opposite** | OSBS pipeline slope/aspect |
| Swenson `calc_gradient()` | `[dzdx, dzdy]` | d(elev)/d(south) | **opposite** | FFT Laplacian only |
| OSBS `calc_gradient_utm()` | `(dzdx, dzdy)` | d(elev)/d(south) | **opposite** | FFT Laplacian only |

The aspect formula `arctan2(-dzdx, -dzdy)` is correct **only** when
`dzdy = d(elev)/d(north)`. Using it with `dzdy = d(elev)/d(south)` produces
the wrong sign on the north component of steepest descent.

---

## 8. The Bug: Double Negation

### pgrid's correct formula

In `slope_aspect()` (`pgrid.py:2289-2303`):

```python
grad = self._gradient_horn_1981(dem, inside)
dzdx = grad[0]    # d(elev)/d(east)
dzdy = grad[1]    # d(elev)/d(north)

aspect.flat[inside] = (180.0/np.pi)*np.arctan2(-dzdx, -dzdy)
aspect[(aspect < 0)] += 360
```

What each negation does:
- `-dzdx` = `-(d(elev)/d(east))` = east component of steepest **descent**
- `-dzdy` = `-(d(elev)/d(north))` = north component of steepest **descent**

So `arctan2(-dzdx, -dzdy)` = `arctan2(east_descent, north_descent)` = the
compass bearing of steepest descent. This is correct.

### The OSBS pipeline's buggy formula (before fix)

In `run_pipeline.py` (original code, before the 2026-02-10 fix):

```python
dzdy, dzdx = np.gradient(dem_for_slope, pixel_size)

# BUGGY: copied pgrid's formula without accounting for sign difference
aspect = np.degrees(np.arctan2(-dzdx, -dzdy))
aspect[aspect < 0] += 360
```

What each negation does here:
- `-dzdx` = `-(d(elev)/d(east))` = east component of descent — **still
  correct**, because np.gradient's x-component has the same sign convention
  as pgrid's
- `-dzdy` = `-(d(elev)/d(south))` = `-(-d(elev)/d(north))` =
  `+d(elev)/d(north)` = north component of steepest **ascent**, not descent

The `-dzdy` term was supposed to negate `d(elev)/d(north)` to get
`north_descent`. But np.gradient returns `d(elev)/d(south)`, which is
*already* `-d(elev)/d(north)`. Negating it a second time produces
`+d(elev)/d(north)` — that is, the north component of ascent. The two
negations cancel out, leaving the wrong sign.

### The fix

The corrected formula (`run_pipeline.py:1523-1528`, current code):

```python
# np.gradient axis 0 follows row index (increases southward in GeoTIFF):
#   dzdy = d(elev)/d(south) = -d(elev)/d(north)
# Therefore: north_downhill = -d(elev)/d(north) = dzdy (no negation needed)
#            east_downhill  = -d(elev)/d(east)  = -dzdx
aspect = np.degrees(np.arctan2(-dzdx, dzdy))
```

By dropping the negation on `dzdy`, the formula correctly uses
`d(elev)/d(south)` directly as the north component of descent — because
"downhill in the north component" means "losing elevation as you go north"
means "gaining elevation as you go south", which is exactly what
`d(elev)/d(south)` measures when positive.

---

## 9. Worked Example: V-Valley DEM

The synthetic V-valley DEM (`generate_synthetic_dem.py`) provides
analytically computable aspect values. The geometry:
- Cross-valley slope: 0.03 m/m (steep, perpendicular to channel)
- Downstream slope: 0.001 m/m (gentle, along channel, north to south)
- Channel runs north-south along the center column

### Elevation formula

```
elev[r, c] = base_elevation
           + downstream_slope * (nrows - 1 - r) * pixel_size
           + cross_slope * abs(c - channel_col) * pixel_size
```

Row 0 is the north (upstream) end. Row `nrows-1` is the south (downstream)
end.

### Tracing the east side of the valley (c > channel_col)

**What np.gradient returns:**

```
dzdx = d(elev)/d(col) = d(elev)/d(east)
     = cross_slope * sign(c - channel_col)
     = +0.03
```

Elevation increases eastward (away from the channel). This is correct.

```
dzdy = d(elev)/d(row) = d(elev)/d(south)
     = downstream_slope * (-1)
     = -0.001
```

The `(nrows - 1 - r)` term in the elevation formula decreases as row
increases, so the derivative with respect to row is negative: elevation
decreases southward (the upstream end at row 0 is higher). `d(elev)/d(south)
= -0.001`.

**What the correct aspect should be:**

The steepest descent on the east side of the valley points west (toward the
channel) and south (downstream). To compute the exact bearing:

First, convert to the northward convention:
```
d(elev)/d(north) = -d(elev)/d(south) = -(-0.001) = +0.001
```

The descent components are:
```
east_descent = -d(elev)/d(east) = -(+0.03) = -0.03   (westward)
north_descent = -d(elev)/d(north) = -(+0.001) = -0.001 (southward)
```

The correct aspect:
```
aspect = arctan2(-0.03, -0.001)
       = arctan2(east_descent, north_descent)
       = 268.09 degrees  (west-southwest)
```

**What the buggy formula computes:**

```
arctan2(-dzdx, -dzdy) = arctan2(-(+0.03), -(-0.001))
                      = arctan2(-0.03, +0.001)
                      = 271.91 degrees  (west-northwest)
```

The north-south component is flipped: instead of pointing slightly south of
west, the buggy result points slightly north of west.

**Error: 271.91 - 268.09 = 3.82 degrees.**

**What the fixed formula computes:**

```
arctan2(-dzdx, dzdy) = arctan2(-(+0.03), -0.001)
                     = arctan2(-0.03, -0.001)
                     = 268.09 degrees  (correct)
```

### Verifying the west side (c < channel_col)

On the west side, `dzdx = d(elev)/d(east) = cross_slope * sign(c - channel)
= -0.03` (elevation decreases eastward toward the channel). The aspect should
point east and south.

Correct: `arctan2(-(-0.03), -0.001)` = `arctan2(+0.03, -0.001)` = 91.91
degrees (east-southeast).

Buggy: `arctan2(-(-0.03), -(-0.001))` = `arctan2(+0.03, +0.001)` = 88.09
degrees (east-northeast).

Error: 3.82 degrees. Same magnitude, same direction of error.

---

## 10. Why the Error is Small in the V-Valley

The 3.82-degree error seems minor. It is small because the cross-slope
(0.03) dominates the downstream slope (0.001) by a factor of 30. The aspect
is nearly pure east/west, and only the small north/south component gets
flipped.

Geometrically, the correct and buggy descent vectors are:

```
correct: (-0.03, -0.001)    buggy: (-0.03, +0.001)
```

These two vectors differ only in the sign of their second component. The
angle between them is:

```
2 * arctan(0.001 / 0.03) = 2 * 1.909 degrees = 3.82 degrees
```

The error is proportional to the ratio of the flipped component to the
dominant component. When that ratio is small, the error is small.

At OSBS, the situation is very different. Slopes are 0.01-0.06 m/m with no
strongly dominant axis. Many pixels have aspect near 0 (north) or 180
(south), where the north-south component is the dominant one. For these
pixels, the error approaches 180 degrees — a complete reversal. A pixel
whose steepest descent truly points north would be reported as pointing
south.

---

## 11. The Reflection Pattern

The bug negates the north component of the descent vector while leaving the
east component unchanged. Geometrically, this reflects the vector across the
east-west axis (the line from 90 to 270 degrees).

The complete reflection table:

| True aspect | Buggy aspect | Error (degrees) |
|-------------|-------------|-----------------|
| 0 (N)       | 180 (S)     | 180             |
| 45 (NE)     | 135 (SE)    | 90              |
| 90 (E)      | 90 (E)      | 0               |
| 135 (SE)    | 45 (NE)     | 90              |
| 180 (S)     | 0 (N)       | 180             |
| 225 (SW)    | 315 (NW)    | 90              |
| 270 (W)     | 270 (W)     | 0               |
| 315 (NW)    | 225 (SW)    | 90              |

Key observations:

- **East (90) and West (270) are unaffected.** They lie on the east-west
  axis, which is the axis of reflection. Reflecting across an axis leaves
  points on that axis unchanged.

- **North (0) and South (180) are maximally affected.** They are
  perpendicular to the axis of reflection. North becomes South and South
  becomes North — a 180-degree error.

- **Diagonal directions are rotated by 90 degrees.** NE (45) becomes SE
  (135): the east component stays the same, but the north component flips to
  south.

- **The error is always a reflection, not a rotation.** The transformation
  negates the north component of the descent vector. If the true descent
  direction is `(sin(theta), cos(theta))`, the buggy version is
  `(sin(theta), -cos(theta))`. Since `sin(180 - theta) = sin(theta)` and
  `cos(180 - theta) = -cos(theta)`, the buggy angle is `(180 - theta) mod
  360`. This is reflection across the east-west axis (90-270 line), which
  maps N to S and vice versa while leaving E and W fixed.

---

## 12. Where the Bug Matters

The aspect value is consumed by the pipeline's aspect-binning step, which
assigns each pixel to one of four hillslopes (North, East, South, West). The
bug corrupts this assignment, and the corruption propagates through every
subsequent computation.

### Downstream corruption chain in the OSBS pipeline

**Step 1: Aspect values (line 1528)** — Every pixel's aspect is reflected
across the E-W axis. At OSBS with roughly uniform aspect distribution, ~25%
of pixels are in each quadrant. After the bug, N-aspect pixels are labeled as
S, and S-aspect pixels are labeled as N. E and W are correct.

**Step 2: Aspect binning (line 1588)** — `get_aspect_mask()` assigns pixels
to N/E/S/W bins based on the (now-wrong) aspect values. The N bin receives
what should be S-bin pixels, and vice versa. E and W bins receive the correct
pixels.

**Step 3: Area fractions (lines 1620, 1652-1657)** — The fraction of total
area assigned to each hillslope depends on how many pixels are in each bin.
Because the N and S pixel sets are swapped, the N and S area fractions are
also swapped. At OSBS where the aspect distribution is nearly uniform, the
error is subtle (both fractions are near 25%). But even small differences
between the true N and S pixel counts are assigned to the wrong hillslope.

**Step 4: Trapezoidal width fit (lines 1625-1627)** — The trapezoidal model
is fitted on the DTND vs accumulated-area curve for each aspect bin's pixel
population. Since the N and S bins contain the wrong pixels, the fitted
trapezoidal parameters (base width, divergence slope, total area) are wrong.

**Step 5: HAND binning (lines 1632-1650)** — Within each aspect bin, pixels
are further divided into elevation bins by their HAND values. Since the aspect
bins contain the wrong pixels, the HAND distributions within each bin are also
wrong. This affects the bin boundaries and the number of pixels in each
elevation band.

**Step 6: All 6 parameters x 16 elements (lines 1660-1711)** — For each of
the 16 hillslope elements (4 aspects x 4 elevation bins), the pipeline
computes mean height (HAND), median distance (DTND), mean slope, circular
mean aspect, fitted area, and width from the trapezoidal model. All of these
are computed from the wrong pixel populations, so all are wrong for the N and
S hillslopes.

**Step 7: NetCDF output** — The wrong parameters are written to the output
file, which would be used by CTSM.

### Observed impact

In the MERIT validation, fixing the aspect bug improved the area fraction
correlation from 0.64 to 0.82 (a gain of 0.18 in correlation). This was the
largest single-fix improvement across all 6 parameters.

At OSBS, the impact would be more severe for any analysis that depends on
distinguishing north-facing from south-facing terrain. For the hillslope
hydrology use case, the N and S hillslope columns would have swapped
parameter sets — the "north-facing" column in CTSM would actually have
south-facing terrain properties and vice versa. This would affect
aspect-dependent insolation corrections and any analysis of aspect-dependent
water/carbon dynamics.

---

## 13. Where the Bug Does NOT Matter

Several pipeline computations are immune to the aspect sign bug:

**Slope magnitude** (`run_pipeline.py:1521`):

```python
slope = np.sqrt(dzdx**2 + dzdy**2)
```

Squaring eliminates the sign. Whether `dzdy` represents d/d(north) or
d/d(south), its square is the same. Slope magnitude is correct regardless of
the sign convention.

**HAND (Height Above Nearest Drainage):** HAND is a pure elevation
difference — the elevation of each pixel minus the elevation of its drainage
outlet. No gradient is involved in the computation. HAND is computed by
pysheds' `compute_hand()`, which traces flow paths using the D8 flow
direction grid.

**DTND (Distance To Nearest Drainage):** DTND is a pure distance
measurement — either Euclidean (EDT) or flow-path-based. No gradient is
involved. (The DTND has its own separate problem — see STATUS.md problem #1
— but that problem is unrelated to the aspect sign bug.)

**Flow routing (D8):** The D8 algorithm assigns flow direction by comparing a
pixel's elevation to its 8 neighbors and choosing the steepest descent
neighbor. This is a direct elevation comparison, not a gradient computation.
The flow direction grid is correct regardless of the aspect bug.

**FFT Laplacian:** The Laplacian `d^2z/dx^2 + d^2z/dy^2` uses second
derivatives where the sign of the first derivative cancels. Both
`calc_gradient()` (Swenson's) and `calc_gradient_utm()` (OSBS pipeline) have
`dzdy = d(elev)/d(south)`, but applying the derivative operator twice gives
`d^2z/d(south)^2 = d^2z/d(north)^2`. The Laplacian is sign-invariant under
coordinate reflection. The characteristic length scale (Lc) computed from the
FFT of the Laplacian is unaffected.

---

## 14. The Four Code Paths and Their Status

| Code path | Method | dzdy convention | Formula | Status |
|-----------|--------|-----------------|---------|--------|
| pgrid `slope_aspect()` | Horn 1981 via `_gradient_horn_1981()` | d(elev)/d(north) | `arctan2(-dzdx, -dzdy)` | Correct (both CRS) |
| OSBS pipeline `run_pipeline.py:1518-1528` | `np.gradient()` | d(elev)/d(south) | `arctan2(-dzdx, dzdy)` | Fixed (interim sign correction) |
| MERIT regression `merit_regression.py:665` | pgrid `slope_aspect()` | d(elev)/d(north) | `arctan2(-dzdx, -dzdy)` | Correct |
| Swenson `calc_gradient()` / OSBS `calc_gradient_utm()` | `np.gradient()` + Horn averaging | d(elev)/d(south) | N/A (Laplacian only) | N/A (sign irrelevant) |

The OSBS pipeline's fix is marked "interim" because Phase D will replace
`np.gradient` entirely with pgrid's `slope_aspect()`, now that Phase A has
added UTM CRS support to pgrid. At that point, the sign-convention mismatch
disappears — both the MERIT and OSBS pipelines will use the same Horn 1981
stencil through pgrid.

---

## 15. Why pgrid's Horn 1981 is Immune

The fundamental difference between pgrid's approach and np.gradient's approach
is where the geographic direction information lives.

**pgrid's `_gradient_horn_1981()`** encodes direction in two places:

1. **Neighbor ordering.** `_select_surround_ravel()` returns indices ordered
   as [N, NE, E, SE, S, SW, W, NW]. The labels "N" and "S" are embedded in
   the index arithmetic: N = `i - ncols` (one row up), S = `i + ncols` (one
   row down). This relies on the GeoTIFF convention (row 0 = north), but the
   mapping from row arithmetic to compass direction is explicit and fixed.

2. **Stencil coefficients.** The Horn stencil sums north-side neighbors and
   subtracts south-side neighbors. The indices `hayindices = [0, 0, 1, 7]`
   select N, N, NE, NW (all north-side). The indices `hsyindices = [3, 4, 4,
   5]` select SE, S, S, SW (all south-side). This produces `d(elev)/d(north)`
   by construction.

The `abs()` on spacings discards the sign of coordinate offsets, ensuring
that the physical distance normalization is always positive. Direction comes
purely from the stencil structure, not from the sign of any spacing.

**`np.gradient()`** has no concept of compass directions. It computes
`d(array)/d(axis_0)` and `d(array)/d(axis_1)` — finite differences along
array axes, following array indexing order. The geographic meaning of "axis 0"
depends on how the raster is stored in memory:

- Standard GeoTIFF (row 0 = north): axis 0 runs south, so
  `d/d(axis_0) = d/d(south)`
- If the raster were stored with row 0 = south (non-standard): axis 0 would
  run north, and the same formula would give `d/d(north)`

This makes np.gradient's sign convention dependent on an external convention
(the raster's storage order) that is not visible in the code that calls it.
The caller must know the storage convention and adjust signs accordingly.
pgrid's approach is self-contained: the compass directions are encoded in the
code itself.

---

## 16. History and Resolution

### Timeline

**MERIT validation stage 8 (2026-02-07):** The N/S swap was first discovered
during MERIT validation. Area fraction correlation was 0.64 — anomalously
low compared to all other parameters. Investigation revealed that
north-aspect pixels were being assigned to the south bin and vice versa. The
fix was to replace the pipeline's `np.gradient` + manual aspect formula with
pgrid's `slope_aspect()` method, which uses the Horn 1981 stencil with
correct compass-direction indexing. Area fraction jumped to 0.82.

**OSBS pipeline could not use the same fix:** The OSBS pipeline processes
UTM data (EPSG:32617), but pgrid's `_gradient_horn_1981()` computed physical
spacings using haversine math that assumes geographic coordinates (latitude
and longitude in degrees). Feeding UTM meter-valued coordinates into
haversine produces garbage distances. The pipeline therefore continued using
`np.gradient` with the buggy `arctan2(-dzdx, -dzdy)` formula.

**Interim sign fix (2026-02-10):** Applied the sign correction to
`run_pipeline.py`: changed `arctan2(-dzdx, -dzdy)` to `arctan2(-dzdx, dzdy)`.
This produces correct aspect values from np.gradient's output by accounting
for the sign-convention difference. Added detailed comments explaining the
reasoning.

**Phase A: UTM CRS support in pgrid (2026-02-12):** Modified pgrid's
`_gradient_horn_1981()` to detect the CRS type and use either
haversine-based (geographic) or Euclidean (projected) distance computation.
The core stencil logic is unchanged — only the spacing calculation differs.
This allows the OSBS pipeline to use pgrid's `slope_aspect()` directly on UTM
data, eliminating the need for the np.gradient workaround entirely.

**Phase D (planned):** Will replace the remaining `np.gradient` usage in
the OSBS pipeline's slope/aspect computation with pgrid's `slope_aspect()`.
At that point, both the MERIT and OSBS pipelines will use the same gradient
implementation with the same sign convention. The interim sign fix becomes
unnecessary and will be removed.

### Root cause

The bug arose from copying a formula across a sign-convention boundary
without adjusting for the difference. pgrid's `arctan2(-dzdx, -dzdy)` is
correct because pgrid's dzdy means d(elev)/d(north). The pipeline copied the
formula but used np.gradient's dzdy, which means d(elev)/d(south). The
formula's `-dzdy` negation, which was supposed to convert from
gradient-of-ascent to gradient-of-descent, instead converted from
gradient-of-descent back to gradient-of-ascent — a double negation.

The original comment in the pipeline code showed that the author correctly
identified the sign difference ("numpy y-axis increases downward") but
reached the wrong conclusion about what adjustment was needed. This is a
common hazard when multiple negations interact: each individual step looks
reasonable, but the chain has one too many (or one too few) sign flips.

---

## Source Files Reference

| File | Lines | Content |
|------|-------|---------|
| `$PYSHEDS_FORK/pysheds/pgrid.py` | 2289-2306 | `slope_aspect()` — calls Horn 1981, applies arctan2 |
| `$PYSHEDS_FORK/pysheds/pgrid.py` | 4172-4226 | `_gradient_horn_1981()` — full Horn stencil implementation |
| `$PYSHEDS_FORK/pysheds/pgrid.py` | 3728-3740 | `_select_surround_ravel()` — 8-neighbor compass indexing |
| `$PYSHEDS_FORK/pysheds/pgrid.py` | 1769-1792 | `_2d_crs_coordinates()` — CRS coordinate grids |
| `scripts/osbs/run_pipeline.py` | 1514-1534 | Fixed aspect computation (np.gradient + corrected formula) |
| `scripts/osbs/run_pipeline.py` | 275-296 | `calc_gradient_utm()` — FFT Laplacian only |
| `scripts/merit_validation/merit_regression.py` | 665 | `grid.slope_aspect("dem")` — uses pgrid directly |
| `$BLUE/Representative_Hillslopes/geospatial_utils.py` | 129-162 | `calc_gradient()` — Swenson's original, FFT Laplacian only |
| `$PYSHEDS_FORK/data/synthetic_valley/generate_synthetic_dem.py` | 34-165 | V-valley DEM with analytical expectations |
