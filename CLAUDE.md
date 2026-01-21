# hpg-esm-tools

Scripts and utilities for working with CESM/CTSM on HiPerGator.

## Repository Structure

```
hpg-esm-tools/
├── scripts/
│   ├── case.analyzer/       # Case analysis toolkit (production ready)
│   ├── hillslope.analysis/  # Hillslope hydrology plotting
│   ├── inputdata.downloading/  # Input data download utilities
│   └── porting/             # Module setup and upstream tracking
├── docs/                    # Technical documentation
├── CLAUDE.md               # This file
├── claude-todo.md          # Working to-do list for Claude Code sessions
└── environment.yml         # Conda environment specification
```

Environment variables are documented in `$BLUE/CLAUDE.md`.

## Scripts Reference

### case.analyzer/ (v0.7.1 - Production Ready)

Fast hybrid toolkit for analyzing CTSM case directories.

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

Scripts for analyzing CTSM hillslope hydrology simulations with column-level (h1 stream) data.

**Key concepts:**
- 4 hillslopes per gridcell (N, E, S, W aspects)
- 4 elevation positions per hillslope (Outlet, Lower, Upper, Ridge)
- 16 hillslope columns + 1 stream column

| Script | Purpose |
|--------|---------|
| `bin_temporal.sh` | Temporal binning (N-year averages) |
| `plot_timeseries_*.py` | Time series plots |
| `plot_zwt_hillslope_profile.py` | Water table vs elevation |
| `plot_elevation_width_overlay.py` | Hillslope geometry |
| `plot_col_areas.py` | Column area bar charts |
| `plot_pft_distribution.py` | PFT distribution |

### inputdata.downloading/

Utilities for downloading CTSM input data from NCAR servers.

### porting/

- `module_env_setup.sh` - Load required modules for CTSM builds

## Documentation

| File | Content |
|------|---------|
| `CTSM_DEVELOPMENT_GUIDE.md` | Quick reference for CTSM development |
| `CTSM_RESEARCH_NOTES.md` | Detailed findings from Phase 4.3 research |
| `SPILLHEIGHT_IMPLEMENTATION.md` | Spillheight mechanism for hillslope hydrology |
| `NCO_TEMPORAL_ANALYSIS.md` | NCO tools for temporal analysis |
| `CTSM_Deterministic_Testing_Analysis.md` | Deterministic behavior and hash comparison |
| `CTSM_CPRNC_Deterministic_Analysis.md` | CPRNC tool usage for NetCDF comparison |
| `clm.hist.names/` | CLM history field reference |

### Archive (historical)

| Document | Location |
|----------|----------|
| Fork Audit | `docs/archive/FORK_AUDIT_2025-01-11.md` |
| Modification Analysis | `docs/archive/CTSM_MODIFICATION_ANALYSIS_2025-01-12.md` |
| Upstream Check | `docs/archive/CTSM_UPSTREAM_CHECK_2025-01-11.md` |

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
- **Python linting:** ruff, mypy
- **Shell linting:** shellcheck
- **Fortran tools:** fprettify (formatter)
- **Testing:** pytest

## Related Repositories

- **hpg-esm-docs** - Official documentation (MkDocs/GitHub Pages)
- **CTSM fork** - `github.com/cdevaneprugh/CTSM` branch `uf-ctsm5.3.085`
  - Local: `/blue/gerber/cdevaneprugh/ctsm5.3`
  - Contains HiPerGator-specific tool fixes (mksurfdata, subset_data paths)
- **ccs_config_cesm fork** - `github.com/cdevaneprugh/ccs_config_cesm` branch `uf-hipergator`
  - Submodule of CTSM fork
  - Contains HiPerGator machine configuration

## Working with Claude Code

See `claude-todo.md` for the current task list and planning document.

### Slash Commands

| Command | Purpose |
|---------|---------|
| `/case-check <path>` | Smoke test a case - verify output files, quick plots |
| `/upstream-check <path\|ctsm>` | Compare local repo to upstream, identify relevant changes |

### When Modifying Scripts

1. Use the `ctsm` conda environment
2. Run `ruff check` and `ruff format` on Python files
3. Run `shellcheck` on bash scripts
