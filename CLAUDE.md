# hpg-esm-tools

Scripts and utilities for working with CESM/CTSM on HiPerGator.

## Repository Structure

```
hpg-esm-tools/
├── scripts/
│   ├── case.analyzer/       # Case analysis toolkit (production ready)
│   ├── hillslope.analysis/  # Hillslope hydrology plotting
│   ├── inputdata.downloading/  # Input data download utilities
│   ├── porting/             # Module setup and upstream tracking
│   └── deploy.custom.files/ # HiPerGator config deployment (legacy)
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

### deploy.custom.files/ (Legacy)

Deployment scripts for HiPerGator-specific configs. Being replaced by proper CTSM fork strategy.

## Documentation

| File | Content |
|------|---------|
| `CTSM_Deterministic_Testing_Analysis.md` | Deterministic behavior and hash comparison |
| `CTSM_CPRNC_Deterministic_Analysis.md` | CPRNC tool usage for NetCDF comparison |
| `NCO_TEMPORAL_ANALYSIS.md` | NCO tools for temporal analysis |
| `SPILLHEIGHT_IMPLEMENTATION.md` | Spillheight mechanism for hillslope hydrology |
| `TEST_Cases.md` | Test case configurations |
| `clm.hist.names/` | CLM history field reference |

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
- **ctsm5.3** - CTSM checkout at `$CIME_SCRIPTS/../..`

## Working with Claude Code

See `claude-todo.md` for the current task list and planning document.

When modifying scripts:
1. Use the `esm-tools` conda environment
2. Run `ruff check` and `ruff format` on Python files
3. Run `shellcheck` on bash scripts
