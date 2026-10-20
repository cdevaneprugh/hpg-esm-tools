# hpg-esm-tools

Scripts and utilities for working with CESM/CTSM on HiPerGator.

## Repository Structure

```
hpg-esm-tools/
├── swenson/                 # Swenson representative-hillslope implementation — the bulk of the repo (own CLAUDE.md + STATUS.md)
├── scripts/
│   ├── case.analyzer/       # Case analysis toolkit
│   ├── hillslope.analysis/  # OSBS hillslope hydrology plotting
│   └── inputdata.downloading/  # Input data download utilities (SVN-based, legacy)
├── docs/                    # Technical documentation + paper summaries
├── CLAUDE.md               # This file
├── LICENSE
└── environment.yml         # Conda environment specification
```

Environment variables are documented in `$BLUE/CLAUDE.md`.

## swenson/ — The Main Project

The Swenson & Lawrence (2025) representative-hillslope implementation for OSBS
lives at `swenson/` and is the bulk of this repo by size and activity. It has its
own documentation set: `swenson/CLAUDE.md` (index), `swenson/STATUS.md`
(scientific record: locked decisions, open questions, change log), and
`swenson/phases/` (detailed per-phase records). **Start there for anything
OSBS / hillslope-pipeline / NEON-forcing related.**

## Scripts Reference

### case.analyzer/

Fast hybrid toolkit for analyzing CTSM case directories. Stable; unmaintained
since its initial import (2026-01).

| Script | Purpose |
|--------|---------|
| `analyze_case` | Main wrapper - orchestrates analysis |
| `generate_case_summary.sh` | Extract XML config, catalog files |
| `concat_hist_stream` | Concatenate NetCDF history files |
| `bin_temporal` | Create annual averages from monthly data |
| `plot_variables.py` | Generate time series plots |

**Usage:**
```bash
# Quick summary (to terminal)
./scripts/case.analyzer/analyze_case $CASES/<case_name>

# With plots
./scripts/case.analyzer/analyze_case $CASES/<case_name> --plot
```

### hillslope.analysis/

Scripts for analyzing the OSBS CTSM hillslope hydrology cases with column-level
(h1 stream) data. OSBS/Swenson-specific (see `swenson/`), not generic.

**Key concepts (OSBS single-aspect layout):**
- 1 aspect per gridcell (OSBS is too flat for meaningful aspect differentiation)
- N HAND elevation bins (8 or 16) + 1 bareground column; scripts auto-detect the
  column count
- The older 4-aspect / 16-column layout survives only in the `archive/` scripts

| Script | Purpose |
|--------|---------|
| `generate_case_plots.sh` | SLURM driver — all plots for the OSBS branch cases |
| `generate_all_plots.py` | Glue — runs the plot modules over binned data |
| `bin_temporal.sh` | Temporal binning (NCO N-year averages; generic) |
| `plot_timeseries_full.py` / `plot_timeseries_last20.py` | Time series plots (generic) |
| `plot_col_timeseries.py` | Per-column spinup time series (Phase E.5+ column structure) |
| `plot_zwt_hillslope_profile.py` | Water table vs elevation |
| `plot_hillslope_cross_section.py` | Hillslope cross-section |
| `plot_tai_heatmap.py` | TAI / HAND-zone heatmap |
| `plot_carbon_water_coupling.py` | Carbon–water coupling |
| `plot_vr_profile.py` | Vertically-resolved profiles (currently inert — no case emits VR vars) |
| `get_gridcell.py` | OSBS gridcell locator (one-off utility) |
| `docs/carbon-state-variables.md` | Reference: CTSM carbon state variables |
| `archive/` | Superseded 4-aspect-era plotters |

### inputdata.downloading/

Utilities for downloading CTSM input data from NCAR servers. SVN-based (legacy
NCAR inputdata server); unused since 2025.

## Documentation

| File | Content |
|------|---------|
| `CTSM_DEVELOPMENT_GUIDE.md` | Quick reference for CTSM development |
| `CTSM_RESEARCH_NOTES.md` | Detailed findings from Phase 4.3 research |
| `NCO_TEMPORAL_ANALYSIS.md` | NCO tools for temporal analysis (historical) |
| `CTSM_Deterministic_Testing_Analysis.md` | Deterministic behavior and hash comparison (historical) |
| `CTSM_CPRNC_Deterministic_Analysis.md` | CPRNC tool usage for NetCDF comparison (historical) |
| `clm.hist.names/` | CLM history field reference |
| `papers/` | Paper PDFs (gitignored) + markdown summaries (DOE grant, Fan 2019, Swenson 2025) |

### CTSM Fork Documentation

Comprehensive CLAUDE.md files exist throughout the CTSM fork at `$BLUE/ctsm5.3/`.
See the root `CLAUDE.md` there for navigation.

## Conda Environment

Use `environment.yml` to create the development environment:

```bash
module load conda
conda env create -f environment.yml
```

**Quick activation (defined in bashrc):**
```bash
esm    # activates ctsm environment
```

Includes:
- **Python 3.12** (pinned to match HiPerGator module used for CTSM builds)
- **Data:** xarray, netCDF4, matplotlib, numpy, pandas, NCO tools
- **Geospatial (swenson pipeline):** gdal, rasterio, geopandas, pyproj, scikit-image, numba
- **Python linting:** ruff, mypy
- **Shell linting:** shellcheck
- **Fortran tools:** fprettify (formatter)
- **Testing:** pytest

## Related Repositories

Repository details (GitHub URLs, branches, local paths) are in `$BLUE/CLAUDE.md`.
Each repository has its own `CLAUDE.md` with detailed guidance.

## Working with Claude Code

Project tracking lives in `swenson/STATUS.md` and `swenson/phases/` (there is no
repo-level to-do file).

### Slash Commands

Defined user-globally in `~/.claude/commands/` — **not part of this repo** (they
won't exist for anyone else who clones it).

| Command | Purpose |
|---------|---------|
| `/case-check <path>` | Smoke test a case - verify output files, quick plots |
| `/upstream-check <path\|ctsm>` | Compare local repo to upstream, identify relevant changes |

### When Modifying Scripts

1. Use the `ctsm` conda environment
2. Run `ruff check` and `ruff format` on Python files
3. Run `shellcheck` on bash scripts
