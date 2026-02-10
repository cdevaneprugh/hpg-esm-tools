# Swenson Representative Hillslope Implementation

Implementation of Swenson & Lawrence (2025) representative hillslope methodology for OSBS.

## Background

@../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md

## Key Resources

| Resource | Location |
|----------|----------|
| Project status | `STATUS.md` |
| Tile grid reference | `tile_grid.md` |
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |
| Swenson's codebase | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/` |
| Our pysheds fork | `$BLUE/pysheds_fork` |
| pysheds documentation | https://mattbartos.com/pysheds/ |

## Directory Structure

```
swenson/
├── CLAUDE.md                  # This file
├── STATUS.md                  # Living project status document
├── tile_grid.md               # Tile reference system (R#C# format)
│
├── audit/
│   └── 240210-validation_and_initial_implementation/
│       ├── audit-summary-240210.md         # Frozen snapshot of STATUS.md at audit time
│       ├── claude-audit.md                 # Personal audit notes
│       ├── osbs_pipeline_audit.md          # Issue catalog
│       ├── flow-routing-resolution.md      # Testing plan
│       ├── progress-tracking.md            # Historical implementation log
│       ├── claude_merit_validation_audit/  # Audit re-run results
│       └── logs/                           # Historical SLURM logs
│
├── data/
│   ├── neon/
│   │   ├── dtm/               # 233 NEON DTM tiles (1m, EPSG:32617)
│   │   └── README.md          # NEON data product catalog
│   ├── mosaics/               # Generated mosaics (OSBS_full.tif, OSBS_interior.tif)
│   ├── merit/                 # MERIT DEM for validation
│   ├── reference/             # Swenson published data
│   └── .gitignore             # Ignores *.tif, *.nc
│
├── output/
│   ├── google-earth/          # KML files for Google Earth
│   ├── osbs/                  # Pipeline runs (YYYY-MM-DD_<desc>/)
│   └── plots/                 # Comparison plots
│
├── scripts/
│   ├── spatial_scale.py       # Shared FFT module
│   ├── merit_validation/      # Stages 1-9 (MERIT DEM validation)
│   ├── osbs/                  # Pipeline scripts
│   │   ├── run_pipeline.py    # Main hillslope pipeline
│   │   ├── run_pipeline.sh    # SLURM job wrapper
│   │   ├── stitch_mosaic.py   # Create mosaic from tiles
│   │   └── extract_subset.py  # Extract subset regions
│   └── visualization/         # KML generation scripts
│       ├── export_kml.py      # Tile grid KML
│       └── export_perimeter_kml.py  # Selection perimeter KML
│
└── logs/                      # Future SLURM output
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
