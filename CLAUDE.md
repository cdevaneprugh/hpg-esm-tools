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

## Key Environment Variables

Defined in user's bashrc:

| Variable | Path | Purpose |
|----------|------|---------|
| `$BLUE` | `/blue/gerber/cdevaneprugh` | User's primary workspace |
| `$CASES` | `/blue/gerber/cdevaneprugh/cases` | CTSM case directories |
| `$ESM_OUTPUT` | `/blue/gerber/cdevaneprugh/earth_model_output/cime_output_root` | Model output root |
| `$CIME_SCRIPTS` | `/blue/gerber/cdevaneprugh/ctsm5.3/cime/scripts` | CIME scripts location |
| `$INPUT_DATA` | `/blue/gerber/earth_models/inputdata` | Shared input data |
| `$SUBSET_DATA` | `/blue/gerber/earth_models/shared.subset.data` | Shared subset data |

## Case Troubleshooting

When debugging a failed CTSM case:

1. **Start with CaseStatus** - Read `CaseStatus` in the case root directory. It logs all workflow steps with timestamps and provides absolute paths to relevant log files when errors occur.

2. **Follow the path** - CaseStatus points directly to the log containing the error. Read that file and diagnose.

3. **Find directories if needed** - Use `./xmlquery EXEROOT RUNDIR DOUT_S_ROOT` from the case directory to locate build, run, and archive directories.

Note: Detailed case structure and HiPerGator-specific configuration will be documented in Phase 4.

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
| `CTSM_Deterministic_Testing_Analysis.md` | Deterministic behavior and hash comparison |
| `CTSM_CPRNC_Deterministic_Analysis.md` | CPRNC tool usage for NetCDF comparison |
| `NCO_TEMPORAL_ANALYSIS.md` | NCO tools for temporal analysis |
| `SPILLHEIGHT_IMPLEMENTATION.md` | Spillheight mechanism for hillslope hydrology |
| `TEST_Cases.md` | Test case configurations |
| `clm.hist.names/` | CLM history field reference |

## Documentation Index

### hpg-esm-tools Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| CTSM Development Guide | `docs/CTSM_DEVELOPMENT_GUIDE.md` | Quick reference for CTSM development |
| CTSM Research Notes | `docs/CTSM_RESEARCH_NOTES.md` | Detailed findings from Phase 4.3 research |
| Fork Audit | `docs/FORK_AUDIT_2025-01-11.md` | Local modifications analysis |
| Modification Analysis | `docs/CTSM_MODIFICATION_ANALYSIS_2025-01-12.md` | Root cause analysis of fixes |
| Upstream Check | `docs/CTSM_UPSTREAM_CHECK_2025-01-11.md` | Upstream comparison results |

### CTSM Repository Documentation (Phase 4.4)

Comprehensive CLAUDE.md documentation in the CTSM fork (`/blue/gerber/cdevaneprugh/ctsm5.3/`):

| Document | Location | Purpose |
|----------|----------|---------|
| Root CLAUDE.md | `CLAUDE.md` | Repository overview, navigation hub |
| Testing Guide | `TESTING.md` | All 5 testing systems |
| Progress Tracking | `claude-todo.md` | Documentation progress |
| **tools/** | | |
| tools/ overview | `tools/CLAUDE.md` | Tool inventory, decision tree |
| mksurfdata_esmf | `tools/mksurfdata_esmf/CLAUDE.md` | Build process, HiPerGator fixes |
| site_and_regional | `tools/site_and_regional/CLAUDE.md` | subset_data, mesh tools |
| **python/** | | |
| Python package | `python/CLAUDE.md` | Package structure, module map |
| site_and_regional impl | `python/ctsm/site_and_regional/CLAUDE.md` | Implementation details |
| **src/** | | |
| Source overview | `src/CLAUDE.md` | Fortran organization, types |
| main/ | `src/main/CLAUDE.md` | Driver, types, control |
| biogeophys/ | `src/biogeophys/CLAUDE.md` | Hydrology, energy, hillslope |
| biogeochem/ | `src/biogeochem/CLAUDE.md` | Carbon-nitrogen cycling |
| **libraries/** | | |
| Libraries | `libraries/CLAUDE.md` | mpi-serial, PIO research |

## Conda Environment

Use `environment.yml` to create the development environment:

```bash
module load conda
conda env create -f environment.yml
```

**Quick activation (defined in bashrc):**
```bash
esm    # activates esm-tools environment
```

Includes:
- **Python 3.12** (pinned to match HiPerGator module used for CTSM builds)
- **Neovim 0.11+** with treesitter syntax highlighting
- **Data:** xarray, netCDF4, matplotlib, numpy, pandas, NCO tools
- **Python linting:** ruff, mypy
- **Shell linting:** shellcheck
- **Fortran tools:** fprettify (formatter)
- **Testing:** pytest

## Editor Setup

Neovim is included in the conda environment (no lmod module needed).

**Config location:** `~/.config/nvim/`

**Features:**
- Treesitter syntax highlighting (Python, Fortran, Bash, Lua, Markdown)
- Gruvbox colorscheme (auto-detects true color support)
- Lualine statusline
- Built-in completion (Ctrl-N/Ctrl-P)
- 80-column marker, line numbers

**Usage:**
```bash
esm           # activate environment (includes neovim)
vim file.py   # opens neovim with syntax highlighting
vim file.f90  # opens neovim with Fortran syntax highlighting
```

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

1. Use the `esm-tools` conda environment
2. Run `ruff check` and `ruff format` on Python files
3. Run `shellcheck` on bash scripts
