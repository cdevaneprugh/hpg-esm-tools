# Swenson Hillslope Implementation for OSBS

Progress tracking and task log.

---

## Goal

Implement Swenson's methods for calculating hillslope parameters from DEM data for our OSBS site.

**Key differences from Swenson's work:**
- Swenson used the 90m MERIT dataset. We will use the 1m NEON dataset for OSBS.
- Swenson forked pysheds and added custom methods. We will port those methods to our own fork.

**Guiding principle:** Do not reinvent the wheel. Use Swenson's paper and methods as a template.

**Overall arc:**
1. Port Swenson's code to our pysheds fork
2. Replicate his experiment to validate our implementation
3. Plan how to apply our tools to OSBS data
4. Download OSBS data and implement

---

## Phase 1: Setup

Basic housekeeping and environment preparation.

- [x] Load the full Swenson paper into context
- [x] Identify required programs and dependencies
    - pysheds (our fork) - **core dependency**
    - numba, rasterio, gdal, scikit-image - **already in environment.yml**
- [x] Clone pysheds fork to `$BLUE/pysheds_fork`
    - [x] Ensure master is up to date with upstream
    - [x] Create `uf-development` branch for development
- [x] Update CLAUDE.md with pysheds setup instructions

### Summary

**Completed 2025-01-22**

Repository setup:
- Cloned `cdevaneprugh/pysheds` to `$BLUE/pysheds_fork`
- Added upstream remote (`mdbartos/pysheds`)
- Created `uf-development` branch (pushed to origin)
- Fork is synced with upstream master

Verification:
```bash
pysheds-env && python -c "from pysheds.sgrid import sGrid"  # OK
```

Key finding: Swenson's fork uses a separate `pgrid.py` file (~200KB) containing all hillslope-specific methods. Upstream `sgrid.py` has a basic `compute_hand()`, but Swenson's version adds DTND (Distance To Nearest Drainage) and AZND (Azimuth to Nearest Drainage) calculations.

---

## Phase 2: Implement Missing Swenson Functions

**Guiding principle:** Do not reinvent the wheel. Copy Swenson's implementation exactly.

### Analysis Summary

#### What Swenson Added to His Fork

Swenson created `pgrid.py` (~200KB, 4,209 lines) - a pure-Python Grid class with hillslope-specific methods:

**Core Hillslope Methods (NEW):**

| Method | Lines | Purpose |
|--------|-------|---------|
| `compute_hand()` | 1750-1950 | Extended HAND + DTND + AZND + drainage_id |
| `compute_hillslope()` | 1952-2150 | Classify pixels: headwater, left bank, right bank, channel |
| `slope_aspect()` | 2152-2222 | Calculate slope and aspect from DEM (Horn 1981 method) |
| `extract_profiles()` | 2968-3084 | Delineate river segments with connectivity |
| `river_network_length_and_slope()` | 3086-3213 | Calculate network length and mean slope |
| `create_channel_mask()` | 3225-3342 | Create channel mask, IDs, and bank masks |

**Supporting Methods (NEW):**

| Method | Lines | Purpose |
|--------|-------|---------|
| `_2d_geographic_coordinates()` | 1730-1748 | Generate 2D lon/lat arrays from affine |
| `_translate_dict()` | 3215-3223 | Map direction values to indices |
| `_gradient_horn_1981()` | 4070-4104 | Horn (1981) gradient for slope/aspect |

#### What Upstream pysheds Has

| Category | Status |
|----------|--------|
| Flow Direction (D8, Dinf, MFD) | Complete |
| Flow Accumulation | Complete |
| Catchment Delineation | Complete |
| HAND (Basic) | **Partial** - height only, no DTND/AZND |
| DEM Preprocessing | Complete |

#### What Upstream Lacks (We Need to Add)

| Feature | Swenson's Solution |
|---------|-------------------|
| DTND (Distance To Nearest Drainage) | In `compute_hand()` |
| AZND (Azimuth To Nearest Drainage) | In `compute_hand()` |
| Drainage ID per pixel | In `compute_hand()` |
| Hillslope classification (L/R bank, headwater) | `compute_hillslope()` |
| Slope/Aspect from DEM | `slope_aspect()` |
| Channel mask with IDs | `create_channel_mask()` |
| River network length/slope | `river_network_length_and_slope()` |

### Implementation Strategy (DECIDED)

**Approach:** Copy pgrid.py to our fork (mirrors Swenson exactly)

**Decisions:**
- Copy entire pgrid.py (not surgical porting)
- Pure Python first (no numba optimization)
- Match Swenson's API signatures for Representative_Hillslopes compatibility

### Implementation Checklist

**Setup Tasks:**
- [x] Copy `pgrid.py` from Swenson's fork to `$PYSHEDS_FORK/pysheds/`
- [x] Update `grid.py` to import Grid from pgrid
- [x] Verify import works: `from pysheds.pgrid import Grid`
- [x] Commit changes to `uf-development` branch

**Methods Verification (all come with pgrid.py):**
- [x] `_2d_geographic_coordinates()`
- [x] `_gradient_horn_1981()`
- [x] `_translate_dict()`
- [x] `slope_aspect()`
- [x] `extract_profiles()`
- [x] `create_channel_mask()`
- [x] `compute_hand()` (extended)
- [x] `compute_hillslope()`

### Existing Test Suite

Swenson's fork includes a pytest test suite at `Representative_Hillslopes/pysheds/tests/`:

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures (grid, dem, fdir, test data paths) |
| `test_grid.py` | ~40 test functions for Grid class |

**Test data included:** `dem.tif`, `dir.asc`, `roi.tif`, `cogeo.tiff`

**Existing coverage:**
- Core pysheds (flowdir, accumulation, catchment): **Tested**
- Basic HAND, extract_profiles: **Tested**
- Hillslope methods (slope_aspect, create_channel_mask, compute_hillslope): **NOT tested**

### Tests Checklist

**Copy from Swenson's fork:**
- [x] Copy `tests/` directory to `$PYSHEDS_FORK/tests/`
- [x] Copy `data/` directory to `$PYSHEDS_FORK/data/`
- [x] Run existing tests: `pytest tests/test_grid.py` (note: tests designed for sgrid API, skipped)

**New tests to write (hillslope-specific):**
- [x] `test_slope_aspect()` - Tests slope/aspect calculation
- [x] `test_create_channel_mask()` - Tests channel IDs, bank classification
- [x] `test_compute_hand()` - Tests HAND calculation
- [x] `test_compute_hillslope()` - Tests hillslope classification
- [x] `test_extract_profiles()` - Tests river segment extraction
- [x] Integration test: Full pipeline on test DEM

### Summary

**Completed 2025-01-22**

#### What Was Done

1. **Copied pgrid.py** from Swenson's fork (`Representative_Hillslopes/pysheds/pysheds/pgrid.py`) to our fork at `$PYSHEDS_FORK/pysheds/pgrid.py`

2. **Updated grid.py** to fall back to pgrid when numba is unavailable (matches Swenson's pattern)

3. **Fixed NumPy 2.0 compatibility issues:**
   - Replaced deprecated `np.warnings` with `warnings` module (12 occurrences)
   - Replaced deprecated `np.bool` with `bool` (13 occurrences)
   - Replaced deprecated `np.float` with `np.float64` (2 occurrences)
   - Fixed unsigned integer overflow in `_flatten_fdir` by adding `_signed_mintype()` helper function

4. **Copied test infrastructure** from Swenson's fork:
   - `tests/conftest.py` - pytest fixtures
   - `tests/test_grid.py` - original test suite (designed for sgrid API)
   - `data/` directory - test DEMs and flow direction grids

5. **Created comprehensive hillslope test suite** at `tests/test_hillslope.py`:
   - TestSlopeAspect: 4 tests for slope/aspect calculation
   - TestChannelMask: 3 tests for channel mask and bank classification
   - TestComputeHand: 2 tests for HAND calculation
   - TestComputeHillslope: 2 tests for hillslope classification
   - TestExtractProfiles: 2 tests for river segment extraction
   - TestIntegration: 1 full workflow test

#### Test Results

```
================== 14 passed, 1 skipped, 33 warnings in 3.63s ==================
```

All hillslope-specific methods verified working.

#### Commit

```
7b20e8e - Add Swenson's hillslope-specific methods from pgrid.py
```
Branch: `uf-development` (pushed to origin)

#### Verified Methods

| Method | Status | Purpose |
|--------|--------|---------|
| `slope_aspect()` | Working | Calculate slope and aspect from DEM |
| `create_channel_mask()` | Working | Channel mask, IDs, and bank classification |
| `compute_hand()` | Working | Extended HAND with DTND, AZND, drainage_id |
| `compute_hillslope()` | Working | Hillslope classification (L/R bank, headwater) |
| `extract_profiles()` | Working | River segment extraction with connectivity |

#### Usage

```bash
pysheds-env
python -c "from pysheds.pgrid import Grid; print('OK')"
```

#### API Notes

The pgrid API uses an inplace pattern by default:
- Methods store results as grid attributes (e.g., `grid.slope`, `grid.aspect`)
- Methods return `None` when `inplace=True` (default)
- Access results via attributes after calling methods

---

## Phase 3: Recreate Swenson Results

Validate our implementation by recreating Swenson's hillslope output.

- [ ] Use sample MERIT data (already downloaded) for SE United States
- [ ] Bin to ~1 degree gridcells (follow Swenson's methods exactly)
- [ ] Generate hillslope NetCDF file
- [ ] Compare results to Swenson's published data

**Note:** If binning context/code/methods are not provided, we can design our own approach.

### Results

*(To be populated)*

---

## Phase 4: OSBS Implementation

Apply what we learned to our OSBS dataset.

*(To be planned after Phase 3)*
