# Swenson Representative Hillslope Implementation

Working document for implementing the Swenson et al. (2025) representative hillslope methodology.

**Repository**: `/blue/gerber/cdevaneprugh/Representative_Hillslopes/`

---

## Method Overview

The Swenson methodology generates representative hillslope geomorphic parameters from high-resolution DEMs for use in CTSM hillslope hydrology. The approach:

1. Identifies characteristic spatial scales using spectral analysis of DEM curvature
2. Delineates catchments using pysheds with spatially-varying accumulation thresholds
3. Calculates 6 geomorphic parameters per DEM pixel: height (HAND), distance (DTND), width, slope, aspect, area
4. Aggregates into representative hillslopes (4 aspects x 4 elevation bins = 16 columns per gridcell)

### Output Parameters (per hillslope column)

| Parameter | Description | Units |
|-----------|-------------|-------|
| `nhillcolumns` | Number of hillslope columns per landunit | - |
| `hillslope_index` | Column identifier (1-16) | - |
| `column_height` | HAND - height above nearest drainage | m |
| `column_width` | Width at lower edge of element | m |
| `column_length` | Distance from channel to element centroid | m |
| `column_area` | Area of hillslope element | m² |
| `column_slope` | Mean surface slope | m/m |
| `column_aspect` | Hillslope-mean aspect (N/E/S/W bin) | radians |
| `stream_channel_length` | Total channel length in gridcell | m |
| `stream_channel_slope` | Mean channel slope | m/m |

---

## Algorithm Pipeline

### Step 1: Spatial Scale Identification

**Function**: `IdentifySpatialScaleLaplacian()` in `spatial_scale.py:511`

**Purpose**: Determine the characteristic hillslope length scale from DEM using spectral analysis.

**Algorithm**:
1. Calculate Laplacian of DEM (∇²z = ∂²z/∂x² + ∂²z/∂y²)
2. Compute 2D FFT of Laplacian field
3. Bin amplitude spectrum by wavelength
4. Fit models (linear, Gaussian, lognormal) to variance-wavelength relationship
5. Peak location indicates characteristic spatial scale

**Key equations**:
```
Laplacian: ∇²z = ∂²z/∂x² + ∂²z/∂y²

Accumulation threshold: A_thresh = 0.5 × (spatial_scale)²
```

**Output**: `spatialScale` (characteristic length in pixels), `accum_thresh` (accumulation threshold)

### Step 2: Landscape Characteristics Calculation

**Class**: `LandscapeCharacteristics` in `representative_hillslope.py:1409`

**Method**: `CalcLandscapeCharacteristicsPysheds()` at line 1457

**Pipeline**:
1. Read DEM for gridcell region (expanded to capture full catchments)
2. Create pysheds Grid object
3. Fill pits and depressions
4. Resolve flats (inflate DEM)
5. Calculate flow directions (D8 algorithm)
6. Calculate flow accumulation
7. Extract stream network where accumulation > threshold
8. Create channel mask with bank identification
9. Compute HAND (Height Above Nearest Drainage)
10. Compute DTND (Distance To Nearest Drainage)
11. Assign drainage IDs to each pixel
12. Classify hillslopes (headwater, left bank, right bank)
13. Calculate slope and aspect
14. Average aspect across each catchment-hillslope unit

### Step 3: Representative Hillslope Aggregation

**Function**: `CalcGeoparamsGridcell()` in `representative_hillslope.py:341`

**Process**:
1. Bin pixels by aspect (N, E, S, W quadrants)
2. Within each aspect, bin by HAND into elevation classes
3. Calculate representative parameters for each bin:
   - Median distance from channel
   - Total area
   - Mean slope
   - Width from area-distance relationship

**Width calculation** (trapezoid form):
```python
# Fit cumulative area vs distance relationship
A(d) = a₀ + a₁d + a₂d²

# Width at distance d:
w(d) = -a₁ - 2a₂d

# Trapezoid parameters:
base_width = -a₁
slope_angle = -a₂
```

---

## Key Code Locations

### Main Scripts

| Script | Purpose | Lines |
|--------|---------|-------|
| `representative_hillslope.py` | Main processing pipeline | ~1750 |
| `spatial_scale.py` | Spectral analysis for spatial scale | ~800 |
| `dem_io.py` | DEM reading (MERIT, ASTER, FAB) | - |
| `terrain_utils.py` | HAND binning, aspect processing | - |
| `geospatial_utils.py` | Gradient calculations, utilities | - |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `CalcGeoparamsGridcell()` | representative_hillslope.py:341 | Main gridcell processing |
| `CalcRepresentativeHillslopeForm()` | representative_hillslope.py:181 | Calculate widths/distances |
| `IdentifySpatialScaleLaplacian()` | spatial_scale.py:511 | Spectral scale identification |
| `LandscapeCharacteristics.CalcLandscapeCharacteristicsPysheds()` | representative_hillslope.py:1457 | Pysheds-based terrain analysis |

### Output Writing

| Function | Location | Purpose |
|----------|----------|---------|
| `WriteHillslopeToNetCDF()` | representative_hillslope.py | Write hillslope parameters to surface data |

---

## Pysheds Fork Requirements

**Fork**: https://github.com/swensosc/pysheds (commit d98738d)

**Local path**: `/blue/gerber/cdevaneprugh/Representative_Hillslopes/pysheds/`

**Usage**: Add to PYTHONPATH (not conda install)

```bash
export PYTHONPATH="/blue/gerber/cdevaneprugh/Representative_Hillslopes/pysheds/pysheds:$PYTHONPATH"
```

### Fork-Specific Methods

The Swenson fork adds methods not available in official pysheds:

| Method | Returns | Official pysheds |
|--------|---------|------------------|
| `grid.compute_hand()` | HAND, DTND, drainage_id | Returns HAND only |
| `grid.compute_hillslope()` | Hillslope classification (1-4) | Does not exist |
| `grid.create_channel_mask()` | channel_mask, channel_id, bank_mask | Does not exist |
| `grid.river_network_length_and_slope()` | Network geometry stats | Does not exist |

### Method Details

**`compute_hand(fdir, dem, channel_mask, channel_id, dirmap)`**
- Returns tuple: (hand, dtnd, drainage_id)
- `hand`: Height Above Nearest Drainage [m]
- `dtnd`: Distance To Nearest Drainage [m]
- `drainage_id`: Unique catchment identifier per pixel

**`compute_hillslope(fdir, channel_mask, bank_mask)`**
- Returns array with values:
  - 1 = headwater hillslope
  - 2 = right bank hillslope
  - 3 = left bank hillslope
  - 4 = channel

**`create_channel_mask(fdir, mask, dirmap)`**
- Returns tuple: (channel_mask, channel_id, bank_mask)
- `channel_mask`: Boolean mask of channel pixels
- `channel_id`: Unique ID for each channel reach
- `bank_mask`: Left/right bank identification

**`river_network_length_and_slope(dem, fdir, acc, mask, dirmap)`**
- Returns dict with network statistics:
  - `length`: Total network length [m]
  - `slope`: Mean network slope [m/m]
  - `reach_slopes`, `reach_lengths`: Per-reach values

---

## OSBS Adaptation Notes

### Site Information
- **Location**: Ordway-Swisher Biological Station, North-central Florida
- **Coordinates**: ~29.68°N, 82.00°W
- **Terrain**: Sandhills with wetland depressions
- **DEM Source**: 1m LIDAR available

### Considerations for OSBS

1. **Low relief**: Florida sandhills have subtle topography
   - May need lower accumulation thresholds
   - Spectral analysis may find longer characteristic scales

2. **Wetland depressions**: Karst-influenced closed basins
   - Basin identification/removal is critical (see `identify_basins()`)
   - May need to adjust `flood_thresh` parameters

3. **DEM format**: Need to adapt `dem_io.py` for local LIDAR format
   - Current readers: MERIT, ASTER, FAB
   - May need custom GeoTIFF reader for LIDAR

4. **Scale**: 1m resolution is much finer than global DEMs
   - Processing may be memory-intensive
   - Consider processing in tiles

### Recommended Workflow for OSBS

1. Obtain 1m LIDAR DEM covering OSBS site
2. Create DEM reader function for local format
3. Test spectral analysis to identify characteristic scale
4. Run catchment delineation on test region
5. Validate HAND/DTND against known wetland locations
6. Generate hillslope parameters for full gridcell
7. Create CTSM surface dataset with hillslope columns

---

## Environment Setup

### Conda Environment Update

```bash
conda activate ctsm
conda env update -f environment.yml --prune
```

**Note**: The environment requires numpy < 2.0 due to pysheds fork using deprecated numpy type
aliases (`np.bool8`, etc.). This constraint is specified in `environment.yml`.

### PYTHONPATH for Pysheds Fork

Add to `~/.bashrc` or run before processing:

```bash
export PYTHONPATH="/blue/gerber/cdevaneprugh/Representative_Hillslopes/pysheds:$PYTHONPATH"
```

**Note**: The path points to the repository root, not the inner `pysheds/` directory.

### Test Imports

```python
# Test pysheds fork - use pgrid.Grid for full method support
from pysheds.pgrid import Grid
g = Grid()
print("Pysheds fork methods available:")
print("  compute_hand:", hasattr(g, 'compute_hand'))
print("  compute_hillslope:", hasattr(g, 'compute_hillslope'))
print("  create_channel_mask:", hasattr(g, 'create_channel_mask'))
print("  river_network_length_and_slope:", hasattr(g, 'river_network_length_and_slope'))
```

**Important**: Use `pysheds.pgrid.Grid`, not `pysheds.sgrid.sGrid`. The fork's `pgrid.py` module
contains the extended methods needed for the Swenson workflow. A local fix was applied to
`pgrid.py` to correct import statements (changed `from pysheds.view` to `from pysheds.pview`).

---

## References

- Swenson et al. (2025) - Representative hillslope methodology paper
- CTSM hillslope hydrology documentation (see `$BLUE/ctsm5.3/src/biogeophys/CLAUDE.md`)
- pysheds documentation: https://mattbartos.com/pysheds/

---

## Status

- [x] Repository cloned
- [x] Pysheds fork cloned
- [x] Environment dependencies identified
- [x] Conda environment updated (gdal, rasterio, geojson, pyproj, scikit-image, numba)
- [x] Pysheds fork imports verified (using `pgrid.Grid` with local fix)
- [x] All Swenson modules import successfully
- [ ] OSBS DEM acquired
- [ ] DEM reader adapted for LIDAR format
- [ ] Test run on OSBS region
