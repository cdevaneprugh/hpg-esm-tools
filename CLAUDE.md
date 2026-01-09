## IMPORTANT
This is the remnants of the CLAUDE file that was previously used in $CASES.

Many of the scripts that used to live there are now located in the general $BLUE space.

NOTE: Some of the scripts/files mentioned may have been moved or deleted. Do NOT use this file as an infallible source. Use it as a starting point.
The goal is to take what is useful here and apply it to new a new CLAUDE.md file and new documentation for scripts.

## Configuration Scripts

### scripts/test_config.sh
Standard test configuration for 5-year runs:
- Start date: 1901-01-01
- Duration: 60 months (5 years)
- Forcing end: 1906
- Wallclock: 2:30:00

Apply with: `cd <case_directory> && bash ../scripts/test_config.sh`

## Case Analyzer Tool

### Overview

The `case.analyzer/` directory contains a hybrid toolkit for analyzing CTSM case directories:
- Extracts case configuration from XML
- Catalogs history files and output (all streams: h0-h5)
- Generates time series plots of key variables
- **Fast NCO-based concatenation + Python plotting**

**Status: ✅ PRODUCTION READY** (v0.7.1 - Incomplete file handling)

### Architecture

**Hybrid Bash + NCO + Python (479 lines total):**
- `analyze_case` - Main wrapper (131 lines)
- `generate_case_summary.sh` - Summary with multi-stream detection (138 lines)
- `concat_hist_stream` - Standalone NCO concatenation script (96 lines)
- `plot_variables.py` - Streamlined Python plotting (66 lines)
- `default.conf` - Configuration for XML vars and plot variables (52 lines)

**Dependencies:** bash, NCO (ncrcat, ncks, ncdump), Python (xarray, matplotlib, numpy)

**Key Features:**
- ✅ Hybrid design (bash orchestration, NCO concat, Python plotting)
- ✅ Fixed output location (no timestamp directories)
- ✅ Standalone concat_hist_stream script (can be run independently)
- ✅ Handles incomplete files at simulation end (automatically excludes time=0 files)
- ✅ Proper cftime handling in plots (noleap calendars)
- ✅ Multi-stream support (auto-detects h0-h5 history streams)
- ✅ Summary displays to terminal in plot mode
- ✅ Minimal defensive checks (54% code reduction from v0.5.0)

### Usage

**Quick analysis (print summary to terminal):**
```bash
cd /blue/gerber/cdevaneprugh/cases

# Fast summary to stdout (< 1 second, no files created)
./case.analyzer/analyze_case <case_directory>
```

**Full analysis with plots:**
```bash
# Generate summary + plots (fixed output location)
./case.analyzer/analyze_case <case_directory> --plot

# Custom configuration
./case.analyzer/analyze_case <case_directory> --plot --config myconfig.conf
```

**Advanced: Manual concatenation and plotting:**
```bash
# Concatenate specific stream
./case.analyzer/concat_hist_stream <case_directory> h1

# Plot from concatenated file
./case.analyzer/plot_variables.py \
  <case_directory>/analysis/concat/combined_h1.nc \
  <output_dir> \
  GPP NPP NEE
```

**Output Structure:**
```
$CASEDIR/analysis/           # Fixed location (always overwrites)
├── summary.txt              # Case configuration and file catalog
├── concat/                  # Concatenated NetCDF files
│   ├── combined_h0.nc       # Monthly data (h0 stream)
│   └── combined_h1.nc       # Daily data (if manually concatenated)
└── plots/                   # Time series PNG plots
    ├── GPP.png
    ├── NPP.png
    └── ...
```

### Performance

**20-year case (240 monthly files, 6 variables):**
- Summary generation: ~1 second
- Concatenation (h0): ~2 seconds
- Plotting (6 variables): ~6 seconds
- **Total: ~9 seconds**

### Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `analyze_case` | Main wrapper - orchestrates analysis | `./analyze_case <CASEDIR> [--plot]` |
| `generate_case_summary.sh` | Extract XML config, catalog files | Called by analyze_case |
| `concat_hist_stream` | Concatenate NetCDF history files | `./concat_hist_stream <CASEDIR> [STREAM]` |
| `plot_variables.py` | Generate time series plots | `./plot_variables.py <NC_FILE> <OUTDIR> <VARS...>` |
| `bin_temporal` | Create annual averages from monthly data | `./bin_temporal <CASEDIR> [STREAM]` |
| `default.conf` | Configuration for XML vars and plot variables | Edit to customize |

### Development Status

**Latest Release: v0.7.1 (Session 16 - 2025-11-04)**
- ✅ Incomplete file handling in concat_hist_stream
- ✅ Automatically detects and excludes files with time=0
- ✅ Enables concatenation of any stream (h0-h5) regardless of simulation end state
- ✅ Minimal code addition: 14 lines
- ✅ Total codebase: 479 lines (3% increase from v0.7.0)
- ✅ Production ready, fully tested

**Previous Release: v0.7.0 (Session 12 - 2025-10-21)**
- Streamlined plotting: 48% reduction (128 → 66 lines)
- Fast smoke-test oriented plotting (numeric time index, no date parsing)
- Non-interactive matplotlib backend (faster rendering)
- Simplified dimension handling (works for all variable types)
- Total codebase: 465 lines (12% reduction from v0.6.0)

See `case.analyzer/PROJECT_PLAN.md` for:
- Complete architecture documentation
- Implementation status and history
- Design decisions and rationale
- Testing results and performance metrics

## Hillslope Analysis Tool

### Overview

The `hillslope.analysis/` directory contains scripts for analyzing CTSM hillslope hydrology simulations. Designed for column-level (h1 stream) data with explicit hillslope representation.

**Key Concepts:**
- **4 hillslopes** per gridcell (North, East, South, West aspects)
- **4 elevation positions** per hillslope (Outlet, Lower, Upper, Ridge)
- **16 hillslope columns** + 1 stream column (column 16)
- **Area-weighted averaging** using `cols1d_wtgcell` variable

### Column Organization

```
Columns 0-3:   North hillslope  (aspect ~0°)
Columns 4-7:   East hillslope   (aspect ~90°)
Columns 8-11:  South hillslope  (aspect ~180°)
Columns 12-15: West hillslope   (aspect ~270°)
Column 16:     Stream/river
```

### Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `bin_1yr.sh` | Create annual averages from monthly data | `./bin_1yr.sh <input.nc> <output.nc>` |
| `bin_20yr.sh` | Create N-year binned averages | `./bin_20yr.sh <input.nc> <output.nc> [years]` |
| `plot_timeseries_full.py` | Full simulation time series (h0) | `python3 plot_timeseries_full.py <h0_20yr.nc> <out.png> <VAR>` |
| `plot_timeseries_last20.py` | Last 20 years by direction/elevation | `python3 plot_timeseries_last20.py <h1_1yr.nc> <out.png> <VAR>` |
| `plot_zwt_hillslope_profile.py` | Water table vs elevation profile | `python3 plot_zwt_hillslope_profile.py <h1_20yr.nc> <out.png> [hillslope]` |
| `plot_elevation_width_overlay.py` | Hillslope geometry profiles | `python3 plot_elevation_width_overlay.py <h1.nc> <out.png>` |
| `plot_col_areas.py` | Bar chart of column areas | `python3 plot_col_areas.py <h1.nc> <out.png>` |
| `plot_pft_distribution.py` | PFT distribution pie chart | `python3 plot_pft_distribution.py <h1.nc> <out.png>` |
| `generate_all_plots.py` | Generate all standard plots | `python3 generate_all_plots.py` |
| `get_gridcell.py` | Extract gridcell info from h1 data | `python3 get_gridcell.py` |

### Data Files

Standard data files in `hillslope.analysis/data/`:
- `combined_h0.nc` - Monthly gridcell-averaged data
- `combined_h1.nc` - Monthly column-level data
- `combined_h0_1yr.nc` / `combined_h1_1yr.nc` - Annual binned
- `combined_h0_20yr.nc` / `combined_h1_20yr.nc` - 20-year binned

### Key Variables

**Spatial (h1 files):**
- `hillslope_elev(column)` - Elevation above stream (m)
- `hillslope_distance(column)` - Distance from stream (m)
- `hillslope_area(column)` - Column area (m²)
- `cols1d_wtgcell(column)` - Area fraction for weighted averaging

**Hydrological:**
- `ZWT` - Water table depth below surface (m)
- `H2OSFC` - Surface water depth (mm)
- `QRUNOFF` - Total runoff (mm/s)

### Weighted Averaging

When computing averages across columns, use area-weighted averaging:

```python
# Example: North hillslope average
north_cols = [0, 1, 2, 3]
weights = cols1d_wtgcell[north_cols]
renorm_weights = weights / weights.sum()
north_avg = (VAR[:, north_cols] * renorm_weights).sum(axis=1)
```

**Important:** Exclude stream column (16) from hillslope binning; include in gridcell averages.

See `hillslope.analysis/README.md` for complete methodology and scientific findings.

## Documentation

- `docs/TEST_Cases.md`: Summary of test and custom case configurations and run status
- `docs/CTSM_Deterministic_Testing_Analysis.md`: Analysis of deterministic behavior and hash comparison testing
- `docs/CTSM_CPRNC_Deterministic_Analysis.md`: CPRNC tool usage and NetCDF comparison analysis
- `docs/SPILLHEIGHT_IMPLEMENTATION.md`: Spillheight mechanism for hillslope hydrology (wetland bank height effect)
- `case.analyzer/PROJECT_PLAN.md`: Case analyzer tool development plan and documentation
- `hillslope.analysis/README.md`: Hillslope analysis methodology and scientific findings
