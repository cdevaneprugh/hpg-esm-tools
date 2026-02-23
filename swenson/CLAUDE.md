# Swenson Representative Hillslope Implementation

Implementation of Swenson & Lawrence (2025) representative hillslope methodology for OSBS.

## Background

@../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md
@STATUS.md

## Pipeline Orientation

The central technical problem is that pysheds assumes a geographic CRS while our NEON LIDAR data is in UTM (EPSG:32617). STATUS.md has the full problem catalog and phase plan — read it before starting work. The main OSBS pipeline is `scripts/osbs/run_pipeline.py`. The MERIT validation pipeline has been consolidated into a single regression script (`scripts/merit_validation/merit_regression.py`) that validates the geographic CRS code path after pysheds fork changes. The original 9 stage scripts are archived in `audit/merit_validation_stages/`.

## Phase Workflow

Work is organized into phases A-F, tracked in `phases/`. Each phase file has a `Status:` header, task checkboxes, and a `## Log` section.

**When working on a phase, update its tracking file:**

- **Starting work:** Set `Status: In progress`, add a dated log entry describing what you're doing.
- **During work:** Check off tasks as completed. Add dated log entries with results, decisions, and issues encountered.
- **Finishing:** Set `Status: Complete`, write a summary log entry.
- **Scope changes:** If work reveals new problems or changes scope, update the phase file AND STATUS.md.

Phase files are the **primary record** of what was done and why. After completing a phase, update STATUS.md to reflect the new project state.

## Key Resources

| Resource | Location |
|----------|----------|
| Project status | `STATUS.md` |
| Tile coverage reference | `data/neon/tile_coverage.md` |
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |
| Swenson's codebase | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/` |
| Our pysheds fork | `$BLUE/pysheds_fork` |
| pysheds documentation | https://mattbartos.com/pysheds/ |

## Standard Test Data

| Dataset | Description | Location |
|---------|-------------|----------|
| Single-tile smoke test | R6C10 — representative tile with lake, swamp, upland | `data/neon/dtm/NEON_D03_OSBS_DP3_404000_3286000_DTM.tif` |
| Contiguous mosaic region | R4-R12, C5-C14 (90 tiles, 9x10 km, 0 nodata) | Pixel coords `[3000:12000, 4000:14000]` in `data/mosaics/OSBS_interior.tif` |
| Full tile coverage map | Nodata percentages, row/column assignments | `data/neon/tile_coverage.md` |

## Directory Structure

```
swenson/
├── CLAUDE.md                  # This file
├── STATUS.md                  # Living project status document
│
├── phases/                    # Phase tracking files (A-F)
│   ├── A-pysheds-utm.md       # Fix pysheds for UTM CRS
│   ├── B-flow-resolution.md   # Resolve flow routing resolution
│   ├── C-characteristic-length.md  # Establish trustworthy Lc
│   ├── C-archive/             # Completed Phase C scripts (4 .py + 4 .sh)
│   ├── D-rebuild-pipeline.md  # Rebuild pipeline with fixes
│   ├── E-complete-parameters.md    # Complete the parameter set
│   └── F-validate-deploy.md   # Validate and deploy
│
├── audit/
│   ├── 240210-validation_and_initial_implementation/
│   │   ├── audit-summary-240210.md         # Frozen snapshot of STATUS.md at audit time
│   │   ├── claude-audit.md                 # Personal audit notes
│   │   ├── osbs_pipeline_audit.md          # Issue catalog
│   │   ├── flow-routing-resolution.md      # Testing plan
│   │   ├── progress-tracking.md            # Historical implementation log
│   │   ├── claude_merit_validation_audit/  # Audit re-run results
│   │   └── logs/                           # Historical SLURM logs
│   ├── 250223-pysheds_and_merit_pipeline_audit/
│   │   ├── merit_validation/              # MERIT regression audit docs and diagnostics
│   │   └── pysheds/                       # pysheds fork refactoring and test audit
│   └── merit_validation_stages/            # Archived stage scripts (1-9) and SLURM wrappers
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
│   ├── merit_validation/      # MERIT geographic regression test
│   │   ├── merit_regression.py  # Single-file regression (Lc + 6 params vs published)
│   │   ├── merit_regression.sh  # SLURM wrapper
│   │   ├── README.md            # Script purpose and flowchart
│   │   └── output/              # results.json, summary.txt, SLURM logs
│   ├── osbs/                  # Pipeline scripts
│   │   ├── run_pipeline.py    # Main hillslope pipeline
│   │   ├── run_pipeline.sh    # SLURM job wrapper
│   │   ├── stitch_mosaic.py   # Create mosaic from tiles
│   │   └── extract_subset.py  # Extract subset regions
│   ├── smoke_tests/           # Phase A smoke tests (future)
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

## MERIT Regression Test

Validates the pysheds fork's geographic CRS code path. Run after any changes to `$PYSHEDS_FORK/pysheds/pgrid.py`.

```bash
cd $TOOLS/swenson
sbatch scripts/merit_validation/merit_regression.sh
```

Computes Lc via FFT and 6 hillslope parameters for a known MERIT gridcell, then compares to Swenson's published data. Outputs `scripts/merit_validation/output/results.json` and `summary.txt`. Exits 0 on PASS, 1 on FAIL. Expected runtime: ~10-20 min.

**Pass criteria:** All 6 parameter correlations within 0.01 of expected, Lc within 5% of 763m.

| Parameter | Expected correlation |
|-----------|---------------------|
| Height (HAND) | 0.9979 |
| Distance (DTND) | 0.9992 |
| Slope | 0.9839 |
| Aspect (circular) | 1.0000 |
| Width | 0.9919 |
| Area fraction | 0.9244 |

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
