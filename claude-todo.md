# Claude Code To-Do List

This document serves as a working to-do list for the user and Claude Code when reorganizing the workspace, refactoring scripts, revising documentation, and creating agents/commands.

---

## Phase 1: Foundation

### 1.1 Conda Environment Setup - COMPLETE
Set up a new default conda environment for development work. This is prerequisite for script refactoring.

**Location:** `/blue/gerber/cdevaneprugh/.conda/envs/`

- [x] Create environment.yml specification
- [x] Create environment: `module load conda && conda env create -f environment.yml`
- [x] Test environment with existing scripts
- [x] Verify linters work (ruff, shellcheck, mypy, fprettify, fortls)
- [x] Clean up old environments (removed: claude, ctsm5.3.059, ctsm.tools)

**Active environments:** `esm-tools` (default), `ferret`

**Usage:** `module load conda && conda activate esm-tools`

### 1.2 Workspace Organization - COMPLETE
- [x] Set up git repos (hpg-esm-tools, hpg-esm-docs)
- [x] Review existing scripts and docs in hpg-esm-tools (see docs/SCRIPT_REVIEW_2025-01-10.md)
  - case.analyzer/ - KEEP AS-IS (production ready, 8.6/10)
  - hillslope.analysis/ - REFACTOR (consolidate bin scripts, enable colors)
  - inputdata.downloading/ - KEEP AS-IS (production ready, 9/10)
  - porting/ - CONSOLIDATED (removed get_upstream_changes.sh)
  - deploy.custom.files/ - KEEP (transition to fork strategy planned)
  - docs/ - KEEP ALL (excellent documentation)
- [x] Remove redundant/outdated information from docs
- [x] Determine which scripts to: keep, remove, merge, refactor, or convert to agents/commands
- [x] Add .nc files to .gitignore (created .gitignore)

### 1.3 Bashrc/Vimrc Cleanup - COMPLETE
- [x] Remove defunct `newcase` alias
- [x] Fix `deploy_configs` path
- [x] Add `esm` alias for conda activation
- [x] Add `vim`/`vi` aliases for neovim
- [x] Migrate to neovim with full lua config
- [x] Treesitter syntax highlighting (Python, Fortran, Bash, Lua, Markdown)
- [x] Gruvbox theme, lualine statusline
- [x] Built-in completion (Ctrl-N/Ctrl-P)
- [x] Column 80 marker, line numbers
- [x] Move neovim to conda (v0.11.5) - no lmod dependency
- [x] Add true color detection to options.lua
- [x] Track nvim config in home directory git repo

**Usage:** `esm` (single command activates conda env with neovim + all tools)

---

## Phase 2: Script Refactoring

After conda environment is set up and scripts have been reviewed:

### 2.1 Hillslope Analysis Refactoring - COMPLETE
- [x] Consolidate bin_1yr.sh + bin_20yr.sh into `bin_temporal.sh --years=N`
- [x] Enable color schemes in plot_elevation_width_overlay.py and plot_col_areas.py
- [x] Parameterize plot_vr_profile.py for any hillslope (moved from vr_variables/, added --hillslope flag)
- [x] Fix 3-period vs 2-period mismatch in plot_zwt_hillslope_profile.py docstring
- [x] Standardize all plotting scripts with uniform structure (argparse, section headers, docstrings)
- [x] Update generate_all_plots.py for new function signatures
- [x] Run linters (shellcheck, ruff) on all modified scripts

### 2.2 General Cleanup - COMPLETE
- [x] Verify doc paths are current (added historical notes to docs with outdated paths)
- [x] Run linters and formatters on all scripts
- [x] Add comments to existing code where needed

---

## Phase 3: Agents & Slash Commands

Tools for Claude Code to use. Where functionality overlaps, design as a single tool that can run either in-context (slash command style) or as background agent.

### Case Management Tools
1. **Case Troubleshooter/Debugger**
   - Input: case directory
   - Reads logs, diagnoses failure, suggests fixes
   - Optional: fix and resubmit (user confirmation required)

2. **Case Creator**
   - Input: compset, grid, specific variables
   - Creates and builds case
   - Optional: monitor until execution starts

3. **Hist File & Case Parser**
   - Query case variables from XML configs
   - Examine and report on history files
   - Could leverage/replace case_analyzer scripts

### Research & Documentation Tools
4. **Documentation Reconciler** (HIGH PRIORITY for Phase 4)
   - Reconcile discrepancies between: online docs, local READMEs, code behavior
   - Generate findings file with solutions, observations, insights
   - Critical for CTSM deep dive work

5. **Paper Summarizer & Info Extractor**
   - Summarize scientific papers
   - Extract relevant information for research

6. **GitHub Project Status Checker**
   - Compare local version to upstream commits
   - Identify notable changes relevant to our use case
   - Recommend whether to update, with analysis of advantages/risks

### Source Code Tools (for Phase 4)
7. **CTSM Module Analyzer**
   - Find specific parameters or submodules in CTSM source
   - Limited use case - mainly when context window retention isn't needed

---

## Phase 4: CTSM Deep Work

Long-term efforts requiring the tools built in Phase 3.

### 4.1 CTSM Fork Strategy
- [ ] Analyze best approach to fork CTSM
  - Fork main repo + ccs_config (machine configs)?
  - Manage gitmodules complexity
- [ ] Add CLAUDE.md files for tracking
- [ ] Create user.src/ directory for source modifications
- [ ] Consider script directory for relevant user scripts
- [ ] Document the deploy_configs approach and why it's being replaced

### 4.2 Version Analysis
- [ ] Evaluate current ctsm5.3 against newer versions
- [ ] What has changed? Worth upgrading?
- [ ] Use GitHub Status Checker tool for this analysis

### 4.3 Claude Code Best Practices for CTSM Development
- [ ] Pull documentation from NCAR/CESM wikis and PDFs
- [ ] Parse what's relevant vs outdated
- [ ] Create local summaries with source links
- [ ] Establish reference priority: 1) local summaries, 2) provided sources, 3) CESM forums
- [ ] Implement guide for Claude Code when modifying Fortran source code

### 4.4 CTSM Source Code Deep Dive
- [ ] Use Documentation Reconciler to identify: broken docs, outdated but useful docs, up-to-date docs
- [ ] Write Claude-specific documentation for reference
- [ ] Focus on commonly used scripts, source code, build directions
- [ ] Work with user to systematically document findings

### 4.5 hpg-esm-docs Integration
- [ ] Design documentation structure with user
- [ ] Integrate findings from deep dive
- [ ] Build out MkDocs site with GitHub Pages
- [ ] Target audience: future researchers and students

---

## Phase 5: Low Priority / Deferred

### 5.1 X11 Forwarding Improvements
Streamline plot viewing workflow. Current approach is crude.

### 5.2 Input Data Reconciliation
For tower run script - maximize use of local data files.

### 5.3 Input Data Ownership & Permissions
- [ ] Change ownership of downloaded inputdata to group leader (for when user leaves)
- [ ] Verify permissions are correct (Lustre filesystem has had chmod issues)
- [ ] Location: `/blue/gerber/earth_models/inputdata`

### 5.4 Shared PIO Build Strategy
- [ ] Figure out optimal shared PIO build for subset data and single point scripts
- [ ] Current approach: ad-hoc shared build at `/blue/gerber/earth_models/shared/parallelio/`
- [ ] Issues to address:
  - Shipped PIO version breaks subset/single-point scripts
  - Shared PIO enables faster CTSM rebuilds (PIO doesn't need to rebuild each time)
- [ ] Consider documenting in CTSM fork strategy

---

## Notes

- ~~The CLAUDE.md in this repo is remnants from $CASES - needs to be updated/replaced~~ **DONE** - CLAUDE.md updated with current repo structure
- ~~Large .nc files in hillslope.analysis/ should not be on GitHub~~ **DONE** - .gitignore created
- deploy.custom.files/ was moved to scripts/ in this directory (legacy approach)
- Group's current workflow (copying source mods between directories) will be replaced by proper fork management

## Session Log

**2025-01-09:** Initial planning session
- Reviewed and reorganized claude-todo.md into phased approach
- Explored repository structure: scripts/, docs/
- Created environment.yml with linters (ruff, mypy, shellcheck, fprettify, fortls) and data tools
- Created .gitignore to exclude .nc files and caches
- Updated CLAUDE.md with accurate repo documentation
- Created `esm-tools` conda environment (Python 3.12)
- Cleaned up old conda environments (removed claude, ctsm5.3.059, ctsm.tools)
- Cleaned up bashrc: removed defunct newcase alias, fixed deploy_configs path, added esm and vim aliases
- Migrated from vim to neovim with lua config:
  - lazy.nvim plugin manager
  - treesitter syntax highlighting
  - gruvbox theme, lualine statusline
  - 80-column marker, line numbers
- Moved neovim from lmod module to conda environment:
  - Added nvim>=0.11.0 to environment.yml (now at v0.11.5)
  - Updated options.lua with true color detection (COLORTERM check)
  - Single `esm` command now activates everything
  - No more `module load neovim` needed
  - Updated CLAUDE.md with new editor setup documentation

**2025-01-10:** Neovim simplification
- Removed LSP (pylsp, fortls, lua_ls) - too noisy with false positives on CTSM codebase
- Removed nvim-cmp - using built-in Ctrl-N/Ctrl-P completion instead
- Removed LSP packages from environment.yml (fortran-language-server, python-lsp-server)
- Kept: treesitter (syntax highlighting), gruvbox (theme), lualine (statusline)
- Disabled mouse in neovim
- Added nvim config to home directory git tracking
- Updated CLAUDE.md files to reflect simplified setup

**2025-01-10:** Script and documentation review
- Completed comprehensive review of all scripts/ and docs/ directories
- Created docs/SCRIPT_REVIEW_2025-01-10.md with detailed findings
- Findings summary:
  - case.analyzer/ - Production ready (8.6/10), keep as-is
  - hillslope.analysis/ - Good (7/10), needs minor refactoring
  - inputdata.downloading/ - Production ready (9/10), keep as-is
  - deploy.custom.files/ - Excellent (9/10), keep until fork strategy ready
  - docs/ - Excellent, all current and valuable
- Removed scripts/porting/get_upstream_changes.sh (minimal utility)
- Added hillslope refactoring items to Phase 2
- Added shared PIO build strategy to Phase 5

**2025-01-10:** Hillslope script refactoring (Phase 2 complete)
- Consolidated bin_1yr.sh + bin_20yr.sh into bin_temporal.sh with --years=N parameter
- Refactored all plotting scripts with uniform structure:
  - Comprehensive docstrings (PURPOSE, USAGE, ARGUMENTS, EXAMPLE, OUTPUT, NOTES)
  - Section headers (Imports, Constants, Main function, CLI)
  - argparse-based CLI with -h help support
  - Type hints on main functions
- Enabled color schemes in plot_elevation_width_overlay.py and plot_col_areas.py
- Fixed plot_zwt_hillslope_profile.py docstring (said "3 periods" but implemented 2)
- Moved plot_vr_profile.py from vr_variables/ to main directory, added --hillslope parameter
- Updated generate_all_plots.py for new function signatures
- Removed vr_variables/ subdirectory
- All scripts pass shellcheck and ruff
