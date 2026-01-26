# Swenson Representative Hillslope Implementation

Implementation of Swenson & Lawrence (2025) representative hillslope methodology for OSBS.

## Background

@../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md

## Key Resources

| Resource | Location |
|----------|----------|
| Progress tracking | `progress-tracking.md` (this directory) |
| Paper summary | `../docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` |
| Swenson's codebase | `/blue/gerber/cdevaneprugh/Representative_Hillslopes/` |
| Our pysheds fork | `$BLUE/pysheds_fork` |
| pysheds documentation | https://mattbartos.com/pysheds/ |
| Processing scripts | `scripts/` (this directory) |

## Directory Structure

```
swenson/
├── CLAUDE.md              # This file - context loader
├── progress-tracking.md   # Progress tracking and reference docs
├── scripts/               # Processing scripts
├── data/                  # Input data (NEON tiles, mosaics)
└── output/                # Pipeline outputs and visualizations
```

## Output Files

Key outputs in `output/full_mosaic/`:

| File | Purpose |
|------|---------|
| `osbs_tile_grid.kml` | Google Earth tile grid with row/col labels |
| `tile_grid_reference.png` | PNG reference grid for quick viewing |
| `stream_network.png` | Delineated stream network visualization |
| `hand_map.png` | Height Above Nearest Drainage map |
| `hillslope_params.json` | Computed hillslope parameters |
| `hillslope_params.png` | Hillslope parameter summary plot |

**Tile Grid (KML):** Use `scripts/export_tile_grid_kml.py` to regenerate. Green = existing tiles, red = missing. Two toggleable layers for outlines and labels.

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

**Remotes:**
- `origin` - `git@github.com:cdevaneprugh/pysheds.git`
- `upstream` - `https://github.com/mdbartos/pysheds.git`

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

### Hillslope File Details

**Source:** Extracted from Swenson's global dataset:
```bash
ncks -d lsmlon,222 -d lsmlat,127 hillslopes_0.9x1.25_c240416.nc hillslopes_osbs_c240416.nc
```

**Coordinates:** 278.0066°E, 29.6893°N

**Structure:**
- 16 hillslope columns (4 aspects × 4 elevation bins)
- 4 hillslopes: N (24.5%), E (24.6%), S (25.3%), W (25.6%)

### Key Namelist Settings (user_nl_clm)

```fortran
fsurdat = '$CLM_USRDAT_DIR/surfdata_OSBS_hist_1850_78pfts_c251002.nc'
hillslope_file = '$CLM_USRDAT_DIR/hillslopes_osbs_c240416.nc'
use_hillslope = .true.
use_init_interp = .false.
```

**Additional hillslope settings (from lnd_in):**
```fortran
use_hillslope = .true.
use_hillslope_routing = .false.
downscale_hillslope_meteorology = .true.
hillslope_fsat_equals_zero = .false.
```

### Case Configuration

| Setting | Value |
|---------|-------|
| Compset | `1850_DATM%CRUv7_CLM60%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP` |
| CO2 | 284.7 ppmv |
| Branch point | Year 861 |

### Creating a Test Branch

To create a new branch from osbs2 for testing custom hillslope data:

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
./xmlchange CLM_USRDAT_DIR=/path/to/custom/input  # Point to your custom data
```

Then modify `user_nl_clm` to point to your custom hillslope file.
