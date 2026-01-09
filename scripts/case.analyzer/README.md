# CTSM Case Analyzer

Toolkit for analyzing CTSM cases. Extracts xml variables, catalogs output files, and generates time series plots.

---

## Quick Start

**Install dependencies**
To install the environment, ensure the conda module is loaded and you are in the same directory as `environment.yml`.

Then run `conda env create -f environment.yml`.

This will setup a conda environment named "ctsm-tools". This can be changed in the yml file if desired.

### Basic Use
```bash
# Print case summary (no files created)
./case.analyzer/analyze_case <CASEDIR>

# Generate summary + plots
./case.analyzer/analyze_case <CASEDIR> --plot
```

---

## Architecture

**Hybrid:** Bash orchestration + NCO concatenation + Python plotting


`analyze_case` (wrapper)
1. `generate_case_summary.sh`   --> Extracts XML config, catalogs files
2. `concat_hist_stream`         --> Concatenates NetCDF with NCO (ncrcat)
3. `plot_variables.py`          --> Plots time series with matplotlib

- `bin_temporal` (standalone)   --> Annual averaging for long simulations

---

## Components

### `analyze_case` (Main Script)

Orchestrates the entire analysis workflow.

```bash
./analyze_case <CASEDIR> [--plot] [--output-dir DIR] [--config FILE]
```

**Options:**
- `--plot` - Generate plots (otherwise just prints summary)
- `--output-dir DIR` - Custom output location (default: `CASEDIR/analysis`)
- `--config FILE` - Custom config (default: `default.conf`)

### `concat_hist_stream` (Standalone)

Concatenate history files for a specific stream.

```bash
./concat_hist_stream <CASEDIR> [STREAM] [OUTPUT_DIR]
```

**Streams:** h0 (monthly), h1 (daily), h2-h5 (custom)
**Output:** `CASEDIR/analysis/concat/combined_h{N}.nc`

**Incomplete File Handling:**
Automatically detects and excludes incomplete files (time=0) that occur when simulations end mid-cycle. This is common for streams with `hist_mfilt > 1` where files accumulate multiple timesteps before writing.

### `plot_variables.py` (Standalone)

Generate plots from concatenated NetCDF files.

```bash
./plot_variables.py <CONCAT_FILE> <OUTPUT_DIR> <VAR1> [VAR2 ...]
```

**Example:**
```bash
./plot_variables.py my_case/analysis/concat/combined_h0.nc plots/ GPP NPP NEE
```

### `bin_temporal` (Standalone)

Create annual averages from monthly history files.

```bash
./bin_temporal <CASEDIR> [STREAM] [OUTPUT_DIR]
```

**Arguments:**
- `CASEDIR` - Path to case directory (required)
- `STREAM` - History stream (default: h0)
- `OUTPUT_DIR` - Output location (default: `CASEDIR/analysis/binned`)

**Output:** `CASEDIR/analysis/binned/combined_annual_h0.nc`

**Example:**
```bash
# Create annual averages for 500-year simulation
./bin_temporal my_case/ h0

# Plot annual trends (much cleaner than monthly)
./plot_variables.py my_case/analysis/binned/combined_annual_h0.nc custom.plots/ GPP NPP
```

### `default.conf` (Configuration)

Defines which XML variables to query and which variables to plot.

```bash
XML_VARS=(CASE COMPSET RUN_STARTDATE STOP_N ...)
PLOT_VARS=(GPP NPP NEE ER TOTECOSYSC TSA)
```

---

## Output Structure

```
$CASEDIR/analysis/
├── summary.txt              # Case configuration
├── concat/
│   └── combined_h0.nc       # Concatenated monthly NetCDF
├── binned/                  # (optional) Annual averages
│   ├── annual/
│   │   ├── annual_1901.nc
│   │   └── ...
│   └── combined_annual_h0.nc
└── plots/
    ├── GPP.png
    ├── NPP.png
    └── ...
```

---

## Dependencies

- **NCO** - NetCDF Operators (ncrcat, ncks)
- **Python 3** - xarray, matplotlib, numpy

---

## Performance

**20-year case (240 files, 6 variables):** ~9 seconds total
- Summary: ~1s
- Concatenation: ~2s
- Plotting: ~6s

---

## Manual Workflow

Run components independently for custom analysis:

```bash
# 1. Concatenate daily data (h1 stream)
./concat_hist_stream my_case/ h1

# 2. Plot custom variables
./plot_variables.py my_case/analysis/concat/combined_h1.nc plots/ GPP NEE
```

### Long Term Simulation Workflow

For multi-century spin ups and simulations, use annual binning for cleaner trend analysis:

```bash
# 1. Generate annual averages (removes seasonal variability)
./bin_temporal my_case/ h0

# 2. Plot long-term trends
./plot_variables.py my_case/analysis/binned/combined_annual_h0.nc plots/ GPP NPP TOTECOSYSC

```

## Final Notes
**Performance:** ~4 seconds per year. A 500-year simulation processes in ~35 minutes.
- Still looking at ways to improve this. For now it may be best to send the script to a compute node and monitor the log it generates.

### To Do
1. Improve performance of annual binning.
2. Explore other binning methods.
3. Investigate whether CTSM uses a 360 or 365 day year. This could affect weighted average of monthly data.
4. Investigate teasing out and plotting column data.
