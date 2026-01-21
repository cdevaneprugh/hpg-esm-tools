# Swenson Representative Hillslope Implementation

Implementation of Swenson & Lawrence (2025) representative hillslope methodology for OSBS.

## Current Status

**Phase 1 (Setup): COMPLETE**
**Phase 2 (Port Methods): COMPLETE**
**Phase 3 (OSBS Plan): PENDING**
**Phase 4 (Implement OSBS): PENDING**

## Background

@../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md

## Key Resources

| Resource | Location |
|----------|----------|
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |
| Implementation notes | `IMPLEMENTATION.md` (this directory) |
| Swenson's codebase | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/` |
| Our pysheds fork | `$BLUE/pysheds_fork` (branch: uf-hillslope) |
| Processing scripts | `scripts/` (this directory) |

## Directory Structure

```
swenson/
├── CLAUDE.md           # This file - context loader
├── IMPLEMENTATION.md   # Detailed implementation notes
├── swenson-todo.md     # Project task tracking
└── scripts/            # Processing scripts (future)
```

## Pysheds Fork: HillslopeGrid Class

Our pysheds fork (`$BLUE/pysheds_fork`) contains the `HillslopeGrid` class with Swenson's methods:

### Methods Implemented

| Method | Purpose | Returns |
|--------|---------|---------|
| `create_channel_mask()` | Identify channels and banks | (channel_mask, channel_id, bank_mask) |
| `compute_hand_extended()` | HAND + distance + drainage ID | (hand, dtnd, drainage_id) |
| `compute_hillslope()` | Classify hillslope position | hillslope (1=head, 2=right, 3=left, 4=channel) |
| `river_network_length_and_slope()` | Network statistics | dict with length, slope, etc. |

### Usage

```python
from pysheds.hillslope import HillslopeGrid

# Load DEM
grid = HillslopeGrid.from_raster('dem.tif')
dem = grid.read_raster('dem.tif')

# Process DEM
inflated_dem = grid.resolve_flats(dem)
fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing='d8')
acc = grid.accumulation(fdir, dirmap=dirmap, routing='d8')

# Create channel mask (threshold at accumulation of 500 cells)
channel_mask = acc > 500

# Run hillslope methods
channel_mask_out, channel_id, bank_mask = grid.create_channel_mask(
    fdir, channel_mask, dirmap=dirmap
)
hand, dtnd, drainage_id = grid.compute_hand_extended(
    fdir, inflated_dem, channel_mask_out, channel_id, dirmap=dirmap
)
hillslope = grid.compute_hillslope(
    fdir, channel_mask_out, bank_mask, dirmap=dirmap
)
```

### Testing

```bash
# Activate environment and run tests
cd $BLUE/pysheds_fork
pysheds-env
python -m pytest tests/test_hillslope.py -v  # 20 tests

# Run verification script
python tests/verify_outputs.py
```

## Test Site: OSBS

- **Location**: Ordway-Swisher Biological Station, North-central Florida
- **Coordinates**: ~29.68°N, 82.00°W
- **Terrain**: Sandhills with wetland depressions
- **DEM sources**:
  - 90m: MERIT DEM (requires registration)
  - 1m: NEON LIDAR (to be obtained)

### MERIT DEM Access

Registration required at: https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/

Tile needed: **n25w085** (covers N25-N30, W85-W80)

## Environment

DEM processing dependencies are in `environment.yml`:
- gdal, rasterio, geojson, pyproj, scikit-image, numba, looseversion

Activate with:
```bash
pysheds-env  # alias adds pysheds_fork to PYTHONPATH
```

## Git Repository

```bash
cd $BLUE/pysheds_fork
git log --oneline -5  # See recent commits

# Key commits:
# 7f0d9e0 Add hillslope.py with HillslopeGrid class
# 7f55394 Fix profile handling and add unit tests
# 2ff1a68 Add verification scripts
```
