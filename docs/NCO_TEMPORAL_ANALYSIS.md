# NCO Temporal Analysis for CTSM History Files

> **Note**: Historical analysis document. Test case directories referenced may no longer
> exist. The NCO workflows and commands remain valid and applicable.

**Date:** 2025-10-29
**Test Case:** `subset_data.78pfts.no-dompft.I1850Clm60BgcCru.OSBS.251006-113206`
**Duration:** 20 years (1901-1920), 240 monthly files

---

## Overview

This document explores using NCO (NetCDF Operators) for temporal binning and data analysis of CTSM history files. The primary goal is to remove seasonal variability for long-term trend analysis in multi-century simulations.

---

## 1. Annual Binning from Monthly Data

### Workflow

**Objective:** Convert 240 monthly files → 20 annual average files → 1 concatenated time series

```bash
# Step 1: Create annual averages (one file per year)
HISTDIR="/path/to/archive/lnd/hist"
OUTDIR="/path/to/output"

for year in {1901..1920}; do
    ncra -O "$HISTDIR"/*clm2.h0.${year}-*.nc "$OUTDIR/annual_${year}.nc"
done

# Step 2: Concatenate annual files into time series
ncrcat -O "$OUTDIR"/annual_*.nc "$OUTDIR"/combined_annual_timeseries.nc
```

### Performance

| Operation | Time | Output |
|-----------|------|--------|
| Annual averaging (20 years) | ~40 seconds | 20 files × 213 KB |
| Concatenation | ~2 seconds | 318 KB total |
| **Total workflow** | **~42 seconds** | **318 KB** |

### File Size Comparison

| Data Type | Size | Time Points | Reduction |
|-----------|------|-------------|-----------|
| Monthly concatenated | 1.5 MB | 240 | baseline |
| Annual concatenated | 318 KB | 20 | **78% smaller** |
| Variable subset (3 vars) | 16 KB | 20 | **99% smaller** |
| Climatology mean | 213 KB | 1 | **86% smaller** |

### Benefits for Long Simulations

For a 500-year simulation:
- **Monthly files:** 6,000 files → ~31 MB concatenated
- **Annual files:** 500 files → ~8 MB concatenated (**74% reduction**)
- **Plotting:** 500 points vs 6,000 points (cleaner trends)

---

## 2. Other Temporal Binning Options

### 2.1 Seasonal Averaging

**Use case:** Compare winter vs summer patterns

```bash
# Winter (DJF) average
ncra -O hist/*clm2.h0.1901-12.nc \
        hist/*clm2.h0.1902-01.nc \
        hist/*clm2.h0.1902-02.nc \
        winter_djf_1901-02.nc

# Summer (JJA) average
ncra -O hist/*clm2.h0.1901-0[6-8].nc summer_jja_1901.nc

# Create all seasons for all years, then concatenate
```

### 2.2 Decadal Averaging

**Use case:** Multi-century trends

```bash
# Decade 1901-1910
ncra -O annual_190*.nc annual_191[0].nc decadal_1901-1910.nc

# Or from monthly data directly
ncra -O hist/*clm2.h0.190*.nc hist/*clm2.h0.191[0]-*.nc decadal_1901-1910.nc
```

### 2.3 Running Means (Smoothing)

**Use case:** Remove inter-annual variability

```bash
# 3-year running mean
ncra -O -F -d time,1,3 combined_annual.nc running_mean_year1.nc
ncra -O -F -d time,2,4 combined_annual.nc running_mean_year2.nc
ncra -O -F -d time,3,5 combined_annual.nc running_mean_year3.nc
# ... continue for all years
```

**Note:** `-F` enables Fortran-style 1-based indexing

### 2.4 Climatology (Long-term Mean)

**Use case:** Baseline comparison

```bash
# Mean of entire simulation period
ncwa -O -a time combined_annual_timeseries.nc climatology_mean.nc

# Result: Single time point representing 20-year average
```

---

## 3. Useful NCO Utilities for Data Interpretation

### 3.1 ncra - NetCDF Record Averager

**Purpose:** Average files along record (time) dimension

**Common uses:**
- Annual/seasonal/decadal averaging
- Multi-file ensemble means
- Temporal smoothing

```bash
# Average multiple files
ncra -O file1.nc file2.nc file3.nc average.nc

# Average specific time range
ncra -O -F -d time,start,end input.nc output.nc
```

### 3.2 ncrcat - NetCDF Record Concatenator

**Purpose:** Concatenate files along time dimension (current case.analyzer use)

```bash
# Concatenate monthly files
ncrcat -O hist/*clm2.h0.*.nc combined_monthly.nc

# Concatenate annual averages
ncrcat -O annual_*.nc combined_annual.nc
```

### 3.3 ncwa - NetCDF Weighted Averager

**Purpose:** Average over specified dimensions (with optional weights)

```bash
# Remove time dimension (climatology)
ncwa -O -a time input.nc climatology.nc

# Spatial average (if gridded data)
ncwa -O -a lat,lon input.nc timeseries.nc
```

### 3.4 ncbo - NetCDF Binary Operator

**Purpose:** Arithmetic operations between files (+, -, ×, ÷)

**Use case:** Compute anomalies, differences, ratios

```bash
# Anomaly: year - climatology
ncbo -O --op_typ=sub annual_1901.nc climatology_mean.nc anomaly_1901.nc

# Difference between scenarios
ncbo -O --op_typ=sub scenario_A.nc scenario_B.nc difference.nc

# Ratio of two variables (in same file)
ncbo -O --op_typ=dvd var1.nc var2.nc ratio.nc
```

### 3.5 ncks - NetCDF Kitchen Sink

**Purpose:** Subsetting, extraction, metadata operations

```bash
# Extract specific variables
ncks -O -v GPP,NPP,TOTECOSYSC input.nc subset.nc

# Extract time range (years 5-10)
ncks -O -F -d time,5,10 input.nc subset_time.nc

# Extract metadata only
ncks -m input.nc | grep "variable_name"

# Check if variable exists (for scripting)
ncks -m -v GPP input.nc >/dev/null 2>&1 && echo "GPP found"
```

### 3.6 ncap2 - NetCDF Arithmetic Processor

**Purpose:** Compute derived variables using expressions

```bash
# Compute NEP = GPP - ER
ncap2 -O -s 'NEP=GPP-ER' input.nc output.nc

# Multiple derived variables
ncap2 -O -s 'NEP=GPP-ER; NBP=NEP-FIRE_C' input.nc output.nc

# Conditional operations
ncap2 -O -s 'GPP_pos=GPP.clamp(0.0,1e20)' input.nc output.nc
```

### 3.7 ncdump / ncgen

**Purpose:** Text conversion and inspection

```bash
# Inspect file header
ncdump -h input.nc

# View specific variable
ncdump -v time input.nc

# Convert to text (CDL format)
ncdump input.nc > file.cdl

# Convert back to NetCDF
ncgen -o output.nc file.cdl
```

### 3.8 ncstat (if available)

**Purpose:** Statistical summaries

```bash
# Statistics for all variables
ncstat input.nc

# Statistics for specific variable
ncstat -v GPP input.nc
```

---

## 4. Integration with case.analyzer

### Option 1: New Script - `bin_temporal.sh`

Create standalone script similar to `concat_hist_stream`:

```bash
#!/bin/bash
# Script: bin_temporal.sh
# Purpose: Temporal binning of CTSM history files
# Usage: ./bin_temporal.sh <CASEDIR> <STREAM> <BINNING> [OUTPUT_DIR]

CASEDIR=$1
STREAM=${2:-h0}
BINNING=${3:-annual}  # annual, seasonal, decadal, climatology
OUTPUT_DIR=${4:-$CASEDIR/analysis/binned}

# Implementation...
```

**Pros:**
- Standalone, reusable
- Can be called manually or by wrapper
- Flexible binning options

**Cons:**
- Adds complexity
- Need to decide on interface

### Option 2: Add to `analyze_case` Wrapper

Add `--binning` flag to existing wrapper:

```bash
./analyze_case <CASEDIR> --plot --binning annual
./analyze_case <CASEDIR> --plot --binning seasonal
./analyze_case <CASEDIR> --plot --binning climatology
```

**Pros:**
- Integrated workflow
- Single command for users

**Cons:**
- Increases wrapper complexity
- Plotting would need to handle different time scales

### Option 3: Separate Tool Suite

Create `case.timeseries/` directory with specialized tools:

```
case.timeseries/
├── annual_average.sh       # Annual binning
├── seasonal_split.sh       # Seasonal averages
├── decadal_average.sh      # Decadal binning
├── climatology.sh          # Long-term means
├── running_mean.sh         # Smoothing
└── anomaly.sh              # Anomaly calculation
```

**Pros:**
- Clear separation of concerns
- Easy to understand and maintain
- Users pick tools they need

**Cons:**
- More files to manage
- Potential code duplication

---

## 5. Recommended Workflows

### For Short Runs (<50 years)

**Use monthly data directly:**
```bash
# Current approach works well
./case.analyzer/analyze_case <CASEDIR> --plot
```

### For Long Runs (50-500 years)

**Use annual averaging:**
```bash
# Step 1: Create annual binning script (new)
./case.analyzer/bin_temporal.sh <CASEDIR> h0 annual

# Step 2: Plot annual data
./case.analyzer/plot_variables.py \
    <CASEDIR>/analysis/binned/combined_annual.nc \
    <CASEDIR>/analysis/plots_annual/ \
    GPP NPP TOTECOSYSC
```

### For Very Long Runs (>500 years)

**Use decadal averaging:**
```bash
./case.analyzer/bin_temporal.sh <CASEDIR> h0 decadal
```

**Or use variable subsetting:**
```bash
# Extract only carbon variables to reduce size
ncks -O -v GPP,NPP,ER,NEE,TOTECOSYSC \
    combined_annual.nc \
    carbon_only_annual.nc
```

---

## 6. Additional NCO Features Worth Exploring

### 6.1 Ensemble Statistics (nces)

Compare multiple cases or ensemble members:

```bash
# Ensemble mean
nces -O case1_annual.nc case2_annual.nc case3_annual.nc ensemble_mean.nc

# Ensemble standard deviation
nces -O -y rmssdn case1.nc case2.nc case3.nc ensemble_std.nc
```

### 6.2 Variable Packing (ncpdq)

Reduce file size with packing/compression:

```bash
# Pack floating point to short integers
ncpdq -O -P all_dbl input.nc output_packed.nc

# Can reduce file size by 50-70%
```

### 6.3 Time Coordinate Manipulation

Adjust time values/units:

```bash
# Change time units
ncap2 -O -s 'time=time*365' input.nc output.nc

# Add time bounds
ncap2 -O -s 'time_bounds=...' input.nc output.nc
```

---

## 7. Performance Considerations

### Scaling Estimates

Based on 20-year test case (240 monthly files):

| Simulation Length | Monthly Files | Annual Binning Time | Annual File Size |
|-------------------|---------------|---------------------|------------------|
| 20 years | 240 | 40 sec | 318 KB |
| 100 years | 1,200 | 3.3 min | 1.6 MB |
| 500 years | 6,000 | 17 min | 8 MB |
| 1,000 years | 12,000 | 33 min | 16 MB |

### Optimization Tips

1. **Parallel processing:** Process years independently
```bash
parallel -j 8 ncra -O hist/*clm2.h0.{}-*.nc annual_{}.nc ::: {1901..1920}
```

2. **Variable subsetting first:** Extract variables before averaging
```bash
# Faster if you only need a few variables
ncks -v GPP,NPP hist/*.nc | ncra -O - annual.nc
```

3. **Use deflation:** Compress output files
```bash
ncra -O -4 -L 1 input*.nc output.nc  # NetCDF4 with compression
```

---

## 8. Example: Complete Multi-Century Workflow

For a 500-year simulation with seasonal removal:

```bash
#!/bin/bash
# Complete workflow for 500-year CTSM case

CASEDIR="/path/to/case"
HISTDIR="$(cd $CASEDIR && ./xmlquery DOUT_S_ROOT --value)/lnd/hist"
OUTDIR="$CASEDIR/analysis"

echo "Processing 500-year simulation..."

# Step 1: Annual binning (500 files)
mkdir -p "$OUTDIR/annual"
for year in {1501..2000}; do
    ncra -O "$HISTDIR"/*clm2.h0.${year}-*.nc "$OUTDIR/annual/annual_${year}.nc"
done

# Step 2: Concatenate
ncrcat -O "$OUTDIR"/annual/annual_*.nc "$OUTDIR/combined_annual_500yr.nc"

# Step 3: Extract carbon variables only
ncks -O -v GPP,NPP,ER,NEE,TOTECOSYSC,TOTVEGC,TOTSOMC \
    "$OUTDIR/combined_annual_500yr.nc" \
    "$OUTDIR/carbon_annual_500yr.nc"

# Step 4: Create decadal averages for very long-term trends
mkdir -p "$OUTDIR/decadal"
for decade_start in {1501..1991..10}; do
    decade_end=$((decade_start + 9))
    ncra -O "$OUTDIR"/annual/annual_{${decade_start}..${decade_end}}.nc \
        "$OUTDIR/decadal/decadal_${decade_start}-${decade_end}.nc"
done

ncrcat -O "$OUTDIR"/decadal/decadal_*.nc "$OUTDIR/combined_decadal_500yr.nc"

# Step 5: Compute climatology
ncwa -O -a time "$OUTDIR/combined_annual_500yr.nc" "$OUTDIR/climatology_500yr.nc"

# Step 6: Compute anomalies for each year
mkdir -p "$OUTDIR/anomalies"
for file in "$OUTDIR"/annual/annual_*.nc; do
    year=$(basename $file .nc | sed 's/annual_//')
    ncbo -O --op_typ=sub "$file" "$OUTDIR/climatology_500yr.nc" \
        "$OUTDIR/anomalies/anomaly_${year}.nc"
done

echo "Processing complete!"
echo "Outputs:"
echo "  Annual: $OUTDIR/combined_annual_500yr.nc (500 time points)"
echo "  Decadal: $OUTDIR/combined_decadal_500yr.nc (50 time points)"
echo "  Climatology: $OUTDIR/climatology_500yr.nc (1 time point)"
echo "  Anomalies: $OUTDIR/anomalies/ (500 files)"
```

---

## 9. Summary and Recommendations

### Key Findings

1. **Annual binning reduces file size by 78%** while removing seasonal variability
2. **NCO workflow is fast:** ~40 seconds for 20 years
3. **Multiple binning strategies available:** annual, seasonal, decadal, climatology
4. **Rich toolkit:** ncra, ncwa, ncbo, ncap2, ncks for diverse analyses

### Recommended Approach for case.analyzer

**Option: Create `bin_temporal.sh` as standalone script**

**Rationale:**
- Keeps main `analyze_case` simple
- Provides flexibility for advanced users
- Similar to existing `concat_hist_stream` design
- Can be integrated into wrapper later if needed

**Interface:**
```bash
./case.analyzer/bin_temporal.sh <CASEDIR> [STREAM] [BINNING] [OUTPUT_DIR]

# Examples:
./case.analyzer/bin_temporal.sh my_case h0 annual
./case.analyzer/bin_temporal.sh my_case h0 seasonal my_case/analysis/seasonal
./case.analyzer/bin_temporal.sh my_case h0 climatology
```

**Binning options:**
- `annual` - 12-month averages
- `seasonal` - DJF, MAM, JJA, SON averages
- `decadal` - 10-year averages
- `climatology` - Single mean over entire period
- `running3` - 3-year running mean (future)

### Next Steps

1. **Immediate:** User decides on integration approach
2. **Implementation:** Create `bin_temporal.sh` script (if approved)
3. **Testing:** Validate with 100+ year cases
4. **Documentation:** Update CLAUDE.md with new workflow
5. **Optional:** Extend `plot_variables.py` to handle different time scales

---

## 10. References

- **NCO Documentation:** http://nco.sourceforge.net/
- **NCO Examples:** http://nco.sourceforge.net/nco.html#Examples
- **CTSM History Fields:** https://escomp.github.io/ctsm-docs/

---

**Test files location:** `/blue/gerber/cdevaneprugh/cases/tmp/nco_exploration/`
