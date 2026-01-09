# OSBS Hillslope Analysis

Analysis scripts for CTSM (Community Terrestrial Systems Model) hillslope hydrology simulation at Ordway-Swisher Biological Station (OSBS).

## Overview

This analysis examines hillslope hydrological processes in a ~867-year CTSM simulation with explicit hillslope representation. The model divides the gridcell into:
- **4 hillslopes** representing cardinal directions (North, East, South, West)
- **4 elevation positions per hillslope** (Outlet, Lower, Upper, Ridge)
- **16 total hillslope columns** + 1 stream column (column 16)

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

### Column-to-Hillslope Mapping
```
Columns 0-3:   North hillslope  (hillslope_index = 1, aspect ~0°)
Columns 4-7:   East hillslope   (hillslope_index = 2, aspect ~90°)
Columns 8-11:  South hillslope  (hillslope_index = 3, aspect ~180°)
Columns 12-15: West hillslope   (hillslope_index = 4, aspect ~270°)
Column 16:     Stream/river     (hillslope_index = -9999)
```

### Column-to-Elevation Mapping
Each hillslope has 4 positions sorted by elevation:
```
Position  North  East   South  West   Elevation (m)
Outlet    0      4      8      12     ~0.17-0.19
Lower     1      5      9      13     ~1.24
Upper     2      6      10     14     ~2.79-2.82
Ridge     3      7      11     15     ~8.07-8.14
```

### Column Weights (Area Fractions)
The variable `cols1d_wtgcell` provides area fractions for each column relative to the total gridcell:

```
Hillslope  Total Weight  % of Gridcell
North      0.2422        24.22%
East       0.2429        24.29%
South      0.2495        24.95%
West       0.2531        25.31%
Stream     0.0124        1.24%
TOTAL      1.0000        100.00%
```

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

### Example: North Hillslope Average
```python
# North hillslope columns
north_cols = [0, 1, 2, 3]

# Original weights (sum = 0.2422)
weights = [0.0760, 0.0529, 0.0560, 0.0574]

# Renormalized weights (sum = 1.0)
renorm_weights = [0.3136, 0.2183, 0.2311, 0.2370]

# Weighted average
north_avg = sum(VAR[:, col] * w for col, w in zip(north_cols, renorm_weights))
```

### Binning Strategies

#### By Cardinal Direction (4 bins)
Group all 4 elevation positions within each hillslope:
```python
direction_cols = {
    'North': [0, 1, 2, 3],
    'East':  [4, 5, 6, 7],
    'South': [8, 9, 10, 11],
    'West':  [12, 13, 14, 15]
}
```

#### By Elevation Position (4 bins)
Group same elevation positions across all hillslopes:
```python
elevation_cols = {
    'Outlet': [0, 4, 8, 12],
    'Lower':  [1, 5, 9, 13],
    'Upper':  [2, 6, 10, 14],
    'Ridge':  [3, 7, 11, 15]
}
```

#### Gridcell Average
Include all 17 columns (hillslopes + stream):
```python
# Use original weights (already sum to 1.0)
gridcell_avg = (VAR[:, 0:17] * cols1d_wtgcell[0:17]).sum(axis=1)
```

**Important**: Stream column is **excluded** from hillslope binning but **included** in gridcell averages.

### Validation
Gridcell averages calculated from h1 column data match h0 gridcell values to numerical precision (< 1e-6 relative error).

## Binning Scripts

### `bin_1yr.sh`
Creates annual (1-year) binned averages from monthly data.

**Usage:**
```bash
./bin_1yr.sh <input_file> <output_file>
```

**Example:**
```bash
./bin_1yr.sh data/combined_h1.nc data/combined_h1_1yr.nc
```

**Process:**
- Averages every 12 consecutive months (1 year)
- Discards incomplete final year
- Preserves all spatial dimensions

### `bin_20yr.sh`
Creates N-year binned averages from monthly data (default: 20 years).

**Usage:**
```bash
./bin_20yr.sh <input_file> <output_file> [years_per_bin]
```

**Example:**
```bash
./bin_20yr.sh data/combined_h1.nc data/combined_h1_20yr.nc 20
```

**Process:**
- Averages every 240 consecutive months (20 years)
- Discards incomplete final bin
- Uses NCO tools (ncra, ncrcat)

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
Plots last 20 years with direction/elevation breakdown using annual binned h1 data.

**Usage:**
```bash
python3 plot_timeseries_last20.py <input_file> <output_file> <variable>
```

**Example:**
```bash
python3 plot_timeseries_last20.py data/combined_h1_1yr.nc plots/GPP_last20.png GPP
```

**Features:**
- **Top panel**: By cardinal direction (North, East, South, West)
- **Bottom panel**: By elevation position (Outlet, Lower, Upper, Ridge)
- **Black dashed line**: Gridcell average (includes stream column)
- Uses area-weighted averaging (see methodology above)
- Line plots (no markers)

**Input:** h1 annual binned file (column-level data)

**Calculation Details:**
- Hillslope bins: Weighted average of columns 0-15 (excludes stream)
- Gridcell average: Weighted average of all 17 columns (includes stream)
- Extracts last 20 years: `data[-20:, :]`

---

### 3. `plot_zwt_hillslope_profile.py`
Plots water table depth (ZWT) against hillslope elevation profile for early and recent periods.

**Usage:**
```bash
python3 plot_zwt_hillslope_profile.py [input_file] [output_file] [hillslope]
```

**Example:**
```bash
python3 plot_zwt_hillslope_profile.py data/combined_h1_20yr.nc plots/zwt_profile.png North
```

**Features:**
- **2 panels**: Early period (years 1-20) and Recent period (last 20 years)
- Shows hillslope surface elevation and water table elevation
- Colored zones:
  - Tan shading: Unsaturated zone (surface to water table)
  - Light blue shading: Saturated zone (below water table)
  - Dashed blue line: Stream level reference (elevation = 0)
- Position labels: Outlet, Lower, Upper, Ridge
- ZWT values annotated at each position

**Input:** h1 20-year binned file (column-level 20-year averages)

**Calculation:**
```
Water table elevation = hillslope_elev - ZWT
```
where ZWT is depth below surface (positive downward).

---

### 4. `plot_elevation_width_overlay.py`
Plots hillslope elevation and width profiles with all 4 aspects overlaid.

**Usage:**
```bash
python3 plot_elevation_width_overlay.py [input_file] [output_file]
```

**Example:**
```bash
python3 plot_elevation_width_overlay.py data/combined_h1.nc plots/elevation_width_overlay.png
```

**Features:**
- **Left panel**: Elevation profiles (all 4 hillslopes overlaid)
- **Right panel**: Width profiles (all 4 hillslopes overlaid)
- Shows geometric similarity across aspects
- 4 positions per hillslope (ridge to outlet)

**Input:** h1 file (any time resolution - only uses spatial metadata)

---

### 5. `plot_col_areas.py`
Bar chart of hillslope column areas.

**Usage:**
```bash
python3 plot_col_areas.py [input_file] [output_file]
```

**Example:**
```bash
python3 plot_col_areas.py data/combined_h1.nc plots/column_areas.png
```

**Features:**
- Bar chart showing `hillslope_area` for each column
- Color-coded by hillslope
- Percentage labels on bars
- Vertical lines separate hillslopes
- Excludes stream column (column 16)

**Input:** h1 file (uses `hillslope_area` variable)

---

### 6. `plot_pft_distribution.py`
Pie chart of Plant Functional Type (PFT) distribution.

**Usage:**
```bash
python3 plot_pft_distribution.py [input_file] [output_file]
```

**Example:**
```bash
python3 plot_pft_distribution.py data/combined_h1.nc plots/pft_distribution.png
```

**Features:**
- Shows PFT distribution for hillslope columns
- All hillslope columns have identical PFT distribution
- PFT names from CTSM parameter file

**Input:** h1 file (uses `pfts1d_itype_veg` and `pfts1d_wtcol`)

---

## Key Variables

### Spatial Variables (in h1 files)
- `hillslope_index(column)` - Hillslope ID (1=N, 2=E, 3=S, 4=W, -9999=stream)
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

## Verification Files

### `verify_gridcell_calc.py`
Comprehensive verification script that validates weighted averaging calculations.

**Features:**
- Compares h1 column calculations to h0 gridcell reference values
- Tests for GPP variable
- Shows column-by-column weighted contributions
- Validates that weights sum to 1.0
- Confirms numerical accuracy (< 1e-12 relative error)

**Output:** `gridcell_verification.txt`

### `verify_totecosysc_vs_h0.py`
Specific verification for TOTECOSYSC variable.

**Features:**
- Verifies TOTECOSYSC gridcell calculation against h0 reference
- Demonstrates stream column effect on gridcell average
- Matches h0 values with < 1e-6 relative error

**Output:** `totecosysc_h0_verification.txt`

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
