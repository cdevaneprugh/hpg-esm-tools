# Swenson Representative Hillslope Implementation

Implementation of Swenson & Lawrence (2025) representative hillslope methodology for OSBS.

## Background

@../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md

## Key Resources

| Resource | Location |
|----------|----------|
| Progress tracking | `progress-tracking.md` |
| Tile grid reference | `tile_grid.md` |
| Google Earth grid | `osbs_tile_grid.kml` |
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |
| Swenson's codebase | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/` |
| Our pysheds fork | `$BLUE/pysheds_fork` |
| pysheds documentation | https://mattbartos.com/pysheds/ |

## Directory Structure

```
swenson/
├── CLAUDE.md                  # This file
├── progress-tracking.md       # Progress tracking and reference docs
├── tile_grid.md               # Tile reference system (R#C# format)
├── osbs_tile_grid.kml         # Google Earth tile grid
│
├── scripts/
│   ├── merit_validation/      # Stage 1-9 scripts (MERIT DEM validation)
│   ├── osbs/                  # OSBS processing scripts
│   │   ├── run_pipeline.py    # Main hillslope pipeline
│   │   ├── run_pipeline.sh    # SLURM job wrapper
│   │   ├── stitch_mosaic.py   # Create mosaic from tiles
│   │   ├── extract_subset.py  # Extract subset regions
│   │   └── export_kml.py      # Export to Google Earth
│   └── spatial_scale.py       # Shared FFT utilities
│
├── data/
│   ├── tiles/                 # Raw NEON DTM tiles (233 tiles)
│   ├── mosaics/               # Generated mosaics
│   │   ├── OSBS_full.tif      # All 233 tiles stitched
│   │   └── OSBS_*.tif         # Custom selections
│   ├── merit/                 # MERIT DEM for validation
│   └── reference/             # Reference datasets (Swenson global)
│
├── output/
│   ├── merit_validation/      # Stage 1-9 results (validation)
│   └── osbs/                  # OSBS pipeline runs
│       └── YYYY-MM-DD_<desc>/ # Timestamped output directories
│
└── logs/                      # SLURM job logs
```

## Running the Pipeline

```bash
cd $TOOLS/swenson

# Set output descriptor (optional, defaults to "full")
export OUTPUT_DESCRIPTOR=interior

# Submit job
sbatch scripts/osbs/run_pipeline.sh

# Or run interactively
python scripts/osbs/run_pipeline.py
```

Output goes to: `output/osbs/YYYY-MM-DD_<descriptor>/`

## pysheds Setup

Our pysheds fork: `$BLUE/pysheds_fork` (env var: `$PYSHEDS_FORK`)

**Branches:**
- `master` - synced with upstream
- `uf-development` - our development branch

**To use:** Run `pysheds-env` before running scripts (adds to PYTHONPATH).

```bash
pysheds-env
python -c "from pysheds.sgrid import sGrid; print('OK')"
```

## Test Case Setup (osbs2)

Reference case for validating custom OSBS hillslope data.

### Case Locations

| Case | Path | Description |
|------|------|-------------|
| osbs2 (original) | `/blue/gerber/sgerber/earth_model_output/cime_output_root/osbs2/` | 860+ year spinup with hillslopes |
| osbs2.branch.v2 | `$CASES/osbs2.branch.v2` | Branch case for testing |

### Input Data

**Directory:** `/blue/gerber/sgerber/CTSM/subset_input/`

| File | Purpose |
|------|---------|
| `surfdata_OSBS_hist_1850_78pfts_c251002.nc` | Surface data (78 PFTs) |
| `hillslopes_osbs_c240416.nc` | Hillslope parameters (Swenson global data) |
| `datmdata/` | Atmospheric forcing |

### Key Namelist Settings (user_nl_clm)

```fortran
fsurdat = '$CLM_USRDAT_DIR/surfdata_OSBS_hist_1850_78pfts_c251002.nc'
hillslope_file = '$CLM_USRDAT_DIR/hillslopes_osbs_c240416.nc'
use_hillslope = .true.
use_init_interp = .false.
```

### Creating a Test Branch

```bash
cd $CIME_SCRIPTS
./create_newcase --case $CASES/osbs2.test.custom-hillslope \
    --compset 1850_DATM%CRUv7_CLM60%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP \
    --res CLM_USRDAT --run-unsupported

cd $CASES/osbs2.test.custom-hillslope
./xmlchange RUN_TYPE=branch
./xmlchange RUN_REFCASE=osbs2
./xmlchange RUN_REFDIR=/blue/gerber/sgerber/earth_model_output/cime_output_root/osbs2/run
./xmlchange RUN_REFDATE=0861-01-01
./xmlchange CLM_USRDAT_DIR=/path/to/custom/input
```

Then modify `user_nl_clm` to point to your custom hillslope file.
