# OSBS Hillslope Analysis

Analysis scripts for CTSM (Community Terrestrial Systems Model) hillslope hydrology simulation at Ordway-Swisher Biological Station (OSBS).

## Overview

This analysis examines hillslope hydrological processes in CTSM simulations with explicit hillslope representation. The OSBS cases use a single-aspect layout with variable bin count:
- **1 aspect** (OSBS is too flat for meaningful aspect differentiation)
- **N HAND elevation bins** (8 or 16 depending on the case) + 1 bareground column
- Scripts auto-detect column count and layout from the NetCDF data

## Data Files

### Input Data
- **`data/combined_h0.nc`** - Monthly gridcell-averaged output (h0 stream), 867 years
- **`data/combined_h0_1yr.nc`** - Annual binned h0 data, 869 years
- **`data/combined_h0_20yr.nc`** - 20-year binned h0 data, 43 bins (years 10-850)
- **`data/combined_h1.nc`** - Monthly column-level output (h1 stream), 10,410 timesteps
- **`data/combined_h1_1yr.nc`** - Annual binned h1 data, 867 years
- **`data/combined_h1_20yr.nc`** - 20-year binned h1 data, 43 bins (years 10-850)
- **`data/osbs_hillslopes_0.9x1.25_c240416.nc`** - Surface data file with hillslope geometry

### File Structure
- **h0 stream**: Gridcell-averaged variables, no spatial resolution
  - Dimensions: `(time, lndgrid)` where lndgrid=1
  - Monthly frequency (12 timesteps per year)

- **h1 stream**: Column-level variables with spatial detail
  - Dimensions: `(time, column)` where column=17
  - Monthly frequency (12 timesteps per year)
  - Contains hillslope geometry variables

## Column Organization

### Column Detection
Scripts auto-detect hillslope columns using: `hillslope_index > 0 & < 9000`. The remaining column(s) are bareground (`hillslope_index == -9999`, `cols1d_itype_lunit == 2`). Hillslope columns are sorted by `hillslope_elev` ascending (stream-to-ridge).

### OSBS Cases
| Case | Columns | Layout |
|------|---------|--------|
| osbs2.branch.v4 | 9 | 1x8 HAND bins + bareground (Swenson reference) |
| osbs4.branch.v1 | 9 | 1x8 HAND bins + bareground (PI spillheight) |
| osbs4.branch.v2 | 17 | 1x16 HAND bins + bareground (16-bin hybrid + spillheight) |

### HAND Zone Grouping
For analysis, columns are grouped by HAND elevation:
- **TAI zone**: HAND elevation < 0.5m (near-stream, wetland)
- **Transition zone**: 0.5m <= HAND < 5.0m
- **Upland zone**: HAND >= 5.0m

### Column Weights (Area Fractions)
The variable `cols1d_wtgcell` provides area fractions for each column relative to the total gridcell. Weights are renormalized within each zone for proper averaging.

## Weighted Averaging Methodology

### Concept
Hillslope columns have **different areas**. When computing averages across multiple columns, larger columns should contribute more than smaller columns. This is accomplished through **area-weighted averaging** using the `cols1d_wtgcell` variable.

### Formula
For a variable `VAR` and a set of columns `cols`:

```python
# Get weights for selected columns
weights = cols1d_wtgcell[cols]

# Renormalize weights to sum to 1.0 within this group
renorm_weights = weights / weights.sum()

# Calculate weighted average
weighted_avg = (VAR[:, cols] * renorm_weights).sum(axis=1)
```

### Example: TAI Zone Average
```python
# Auto-detect hillslope columns
h_idx = ds["hillslope_index"].values
h_cols = np.where((h_idx > 0) & (h_idx < 9000))[0]
elevations = ds["hillslope_elev"].values[h_cols]

# TAI zone: HAND < 0.5m
tai_mask = elevations < 0.5
tai_cols = h_cols[tai_mask]
tai_weights = cols1d_wtgcell[tai_cols]
renorm = tai_weights / tai_weights.sum()
tai_avg = (VAR[:, tai_cols] * renorm).sum(axis=1)
```

#### Gridcell Average
Include all columns (hillslope + bareground):
```python
n_total = VAR.shape[1]
gridcell_avg = (VAR * cols1d_wtgcell[:n_total]).sum(axis=1)
```

**Important**: Bareground column is **excluded** from zone grouping but **included** in gridcell averages.

### Validation
Gridcell averages calculated from h1 column data match h0 gridcell values to numerical precision (< 1e-6 relative error).

## Binning Scripts

### `bin_temporal.sh`
Creates N-year binned averages from monthly data. Replaces separate `bin_1yr.sh` and `bin_20yr.sh` scripts.

**Usage:**
```bash
./bin_temporal.sh <input_file> <output_file> [--years=N]
./bin_temporal.sh -h | --help
```

**Arguments:**
- `input_file` - Input NetCDF file (concatenated monthly history)
- `output_file` - Output NetCDF file (binned time series)
- `--years=N` - Years per bin (default: 1 for annual averages)

**Examples:**
```bash
# Annual averages (default)
./bin_temporal.sh data/combined_h0.nc data/combined_h0_1yr.nc

# 20-year averages
./bin_temporal.sh data/combined_h0.nc data/combined_h0_20yr.nc --years=20

# 5-year averages
./bin_temporal.sh data/combined_h1.nc data/combined_h1_5yr.nc --years=5
```

**Process:**
- Averages every N×12 consecutive months
- Discards incomplete final bin
- Uses NCO tools (ncra, ncrcat)
- Automatic progress reporting

**Important Note**: The `mcdate` variable in binned files represents the **center/average** of each bin, not the end. For a bin covering years 1-20, mcdate will show year ~10.

## Plotting Scripts

### 1. `plot_timeseries_full.py`
Plots full simulation time series using 20-year binned h0 data (gridcell average).

**Usage:**
```bash
python3 plot_timeseries_full.py <input_file> <output_file> <variable>
```

**Example:**
```bash
python3 plot_timeseries_full.py data/combined_h0_20yr.nc plots/GPP_full.png GPP
```

**Features:**
- X-axis: Simulation years (ticks every 100 years starting at 0)
- Y-axis: Variable value with units
- Single line plot (no markers)
- Title: `"{variable} (20 year bins)"`

**Input:** h0 20-year binned file (gridcell-averaged data)

---

### 2. `plot_timeseries_last20.py`
Plots last N years with HAND-zone breakdown using annual binned h1 data.

**Usage:**
```bash
python3 plot_timeseries_last20.py <input_file> <output_file> <variable> [--years=N]
```

**Example:**
```bash
python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/GPP_last20.png GPP
```

**Features:**
- **Top panel**: By HAND zone (TAI / Transition / Upland)
- **Bottom panel**: Individual column traces colored by HAND elevation
- **Black dashed line**: Gridcell average (includes bareground)
- Auto-detects column count and layout

**Input:** h1 annual binned file (column-level data)

---

### 3. `plot_zwt_hillslope_profile.py`
Plots water table depth (ZWT) against hillslope elevation profile for early and recent periods.

**Usage:**
```bash
python3 plot_zwt_hillslope_profile.py <input_file> <output_file>
```

**Features:**
- **2 panels**: Early and recent periods
- Surface elevation and water table lines with zone fills
- Auto-detects N columns, labels by HAND elevation
- Dynamic y-axis

**Input:** h1 binned file

---

### 4. `plot_hillslope_cross_section.py` (NEW)
Filled cross-section through the hillslope showing surface, water table, and ponding.

**Usage:**
```bash
python3 plot_hillslope_cross_section.py <input_file> <output_file>
```

**Features:**
- Brown surface line, blue water table line
- Tan/blue fills for unsaturated/saturated zones
- Surface ponding shown if H2OSFC present in data
- Smooth interpolated profiles
- **2 panels**: Early and recent periods

**Input:** h1 binned file

---

### 5. `plot_tai_heatmap.py` (NEW)
Space-time heatmap showing variable evolution across the hillslope.

**Usage:**
```bash
python3 plot_tai_heatmap.py <input_file> <output_file> [variable]
```

**Features:**
- X = time, Y = HAND elevation, color = variable value
- Diverging colormap for ZWT; variable-specific colormaps for GPP, H2OSFC
- TAI/Transition zone boundary lines
- Default variable: ZWT

**Input:** h1 annual binned file

---

### 6. `plot_carbon_water_coupling.py` (NEW)
Multi-panel showing water-carbon coupling at three hillslope positions.

**Usage:**
```bash
python3 plot_carbon_water_coupling.py <input_file> <output_file> [--carbon-var=TOTSOMC]
```

**Features:**
- 3 rows: TAI (lowest), Transition (median), Upland (highest)
- Twin axes: ZWT (left, inverted) and carbon variable (right)
- Default carbon variable: TOTSOMC

**Input:** h1 annual binned file

---

### 7. `plot_vr_profile.py`
Depth profiles of vertically resolved variables (soil temperature, moisture, carbon).

**Usage:**
```bash
python3 plot_vr_profile.py <input_file> <output_file> <variable>
```

**Features:**
- ~5 representative columns from stream to ridge
- Labels by HAND elevation
- Auto-detects column count
- Note: current h1a data has no VR variables; ready for future cases

**Input:** h1 binned file (variable must have `levsoi`, `levgrnd`, or `levdcmp` dimension)

---

### Archived Scripts
Moved to `archive/` (redundant with pipeline geometry plots):
- `plot_elevation_width_overlay.py`
- `plot_col_areas.py`
- `plot_pft_distribution.py`

---

## Key Variables

### Spatial Variables (in h1 files)
- `hillslope_index(column)` - Hillslope ID (1=hillslope, -9999=bareground)
- `hillslope_elev(column)` - Elevation above stream (m)
- `hillslope_distance(column)` - Distance from stream (m)
- `hillslope_width(column)` - Column width (m)
- `hillslope_area(column)` - Column area (m²)
- `hillslope_slope(column)` - Slope (m/m)
- `hillslope_aspect(column)` - Aspect (radians)
- `cols1d_wtgcell(column)` - Column weight relative to gridcell (area fraction)

### Time Dependent Variables (examples)
- `GPP` - Gross Primary Production (gC/m²/s)
- `TOTECOSYSC` - Total Ecosystem Carbon (gC/m²)
- `ZWT` - Water Table Depth below surface (m)
- `NPP` - Net Primary Production (gC/m²/s)
- `NEE` - Net Ecosystem Exchange (gC/m²/s)

### Time Variabls
- `mcdate` - Date in YYYYMMDD format
  - In binned files: represents center/average of bin
  - Example: mcdate=10 for bin covering years 1-20
- `time` - Days since 0001-01-01
- `time_bounds` - Time interval endpoints

## Important Scientific Findings

### Stream Column Behavior
The stream column (1.24% of gridcell) exhibits very different behavior from hillslopes:

**GPP (Gross Primary Production):**
- Hillslopes: ~0.000055 gC/m²/s
- Stream: ~0.000063 gC/m²/s (14% higher)
- Result: Gridcell average slightly **higher** than hillslope averages

**TOTECOSYSC (Total Ecosystem Carbon):**
- Hillslopes: ~21,860 gC/m²
- Stream: ~5,660 gC/m² (74% lower)
- Result: Gridcell average significantly **lower** than hillslope averages

**Explanation:**
- Stream has higher productivity (GPP) but much lower carbon storage (TOTECOSYSC)
- Hillslopes have vegetation and deep soil → high carbon storage
- Stream is water body with minimal soil/vegetation → low carbon storage

### Water Table Evolution
From ZWT analysis:
- **Early period (years 1-20)**: Deep water table (3.6-3.8m depth)
- **Recent period (years 841-860)**: Shallower, stable water table (1.9-2.4m depth)
- Water table rises significantly during simulation and stabilizes in later years

## Utility Scripts

### `generate_all_plots.py`
Batch generation of all standard hillslope analysis plots.

**Usage:**
```bash
python3 generate_all_plots.py
```

**Prerequisites:**
Data files in `data/` directory:
- `combined_h0_20yr.nc` - gridcell-level, 20-year bins
- `combined_h1.nc` - column-level, full resolution
- `combined_h1_1yr.nc` - column-level, annual bins
- `combined_h1_20yr.nc` - column-level, 20-year bins

**Output:**
All plots saved to `plots/` directory:
- `{VAR}_full.png` - Full simulation timeseries
- `{VAR}_last20.png` - Last 20 years by hillslope group
- `ZWT_hillslope_profile.png` - Water table vs elevation profile
- `elevation_width_overlay.png` - Hillslope geometry
- `column_areas.png` - Column area distribution
- `pft_distribution.png` - PFT pie chart

**Notes:**
- Modify `VARIABLES` list in script to change which variables are plotted
- Creates `plots/` directory if it doesn't exist

---

### `get_gridcell.py`
Calculate grid indices for a coordinate on the 0.9x1.25° CTSM grid.

**Usage:**
```bash
python3 get_gridcell.py
```

Prints lat/lon indices for OSBS site (hardcoded) and NCO verification commands. Modify `TARGET_LAT` and `TARGET_LON` constants to use for other sites.

## Dependencies

- **Python 3** with:
  - xarray
  - numpy
  - matplotlib

- **NCO (NetCDF Operators)**:
  - ncra (averaging)
  - ncrcat (concatenation)
  - ncdump (inspection)
  - ncks (subsetting)

## Notes for Future Sessions

- All weighted averaging has been validated against h0 reference files
- Binning scripts correctly average data; mcdate represents bin center
- Stream column behavior differs significantly from hillslopes (higher GPP, lower carbon)
- Always exclude stream (column 16) from hillslope binning
- Always include stream in gridcell averages for validation
- Water table shows significant evolution over simulation period
- All 4 hillslopes have similar geometry despite different aspects

## References

- CTSM Documentation: https://escomp.github.io/ctsm-docs/
- NCO Documentation: http://nco.sourceforge.net/
- OSBS Site Information: https://ameriflux.lbl.gov/sites/siteinfo/US-Osh
