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
1. **Case Troubleshooter/Debugger** - DONE (documentation approach)
   - Added troubleshooting guidance to CLAUDE.md
   - Workflow: CaseStatus → follow log paths → diagnose
   - More comprehensive docs planned for Phase 4

2. **Case Creator** - DEFERRED (handle in-context)
   - Simple enough to do conversationally
   - No separate tool needed
   - **Subset data workflow learned:**
     - `create_newcase --res CLM_USRDAT --run-unsupported --user-mods-dirs <subset_dir>/user_mods/`
     - Subset data structure: `datmdata/`, `surfdata_*.nc`, `user_mods/`
     - `user_mods/` contains: `shell_commands`, `user_nl_clm`, `user_nl_datm_streams`
     - Example compset: `1850_DATM%CRUv7_CLM60%BGC_SICE_SOCN_MOSART_SGLC_SWAV_SESP`
   - **Key concepts confirmed:** spinup (AD/post-AD), branch vs hybrid runs, restart files

3. **Case Smoke Test** - DONE (`/case-check` slash command)
   - Location: `~/.claude/commands/case-check.md`
   - Verifies output files match expected (count, completeness)
   - Quick plots for sanity check (spinup trending, etc.)
   - Interactive follow-ups in main context
   - Leverages case.analyzer scripts

### Research & Documentation Tools
4. **Documentation Reconciler** - DONE (agent, draft)
   - Location: `~/.claude/agents/doc-reconciler.md`
   - Reconciles: online docs, local READMEs, code behavior
   - Uses opus model for complexity
   - Primary sources: CTSM docs, CESM forums, CLM page
   - Will be refined during Phase 4 deep dive

5. **Paper Summarizer & Info Extractor** - DEFERRED
   - Needs better research context first
   - Revisit after Phase 4 establishes scientific goals

6. **GitHub Project Status Checker** - DONE (`/upstream-check` slash command)
   - Location: `~/.claude/commands/upstream-check.md`
   - Compares local repo to upstream
   - Flags: hillslope, biogeochem, biogeophys, src/ changes
   - Interactive follow-up in main context

### Source Code Tools (for Phase 4)
7. **CTSM Module Analyzer** - DEFERRED
   - Limited use case
   - Can handle in-context when needed

---

## Phase 4: CTSM Deep Work

Long-term efforts requiring the tools built in Phase 3.

### 4.1 CTSM Fork Strategy - IN PROGRESS

**Status:** Phase 0 (Audit) complete. Ready for Phase A (Create GitHub Forks).

**Plan file:** `~/.claude/plans/indexed-chasing-hammock.md`
**Audit results:** `docs/FORK_AUDIT_2025-01-11.md`

#### Decisions Made
- Fork to personal GitHub account (cdevaneprugh/ctsm, cdevaneprugh/ccs_config)
- Jump straight to ctsm5.3.085 (skip setting up on 5.3.059 first)
- Keep lmod modules for now, evaluate conda migration later (Phase 5)
- No current SourceMods (src.mods directory for future use)

#### Phase 0: Audit - COMPLETE
- [x] Audited all local modifications in ctsm5.3.059
- [x] Compared deploy.custom.files to upstream 5.3.085
- [x] Identified what's still needed vs fixed upstream
- [x] Verified HiPerGator configs are current
- [x] Verified prerequisites (GCC 14.2.0, shared PIO, cprnc)

#### Key Audit Findings (see docs/FORK_AUDIT_2025-01-11.md)
**No longer needed (fixed in upstream 5.3.085):**
- Longitude TypeError fixes in single_point_case.py

**Still needed (carry forward):**
- `MPILIB=openmpi` in single_point_case.py (HiPerGator-specific)
- DATM type default change in subset_data.py
- HiPerGator input paths in default_data_*.cfg
- mksurfdata Fortran format specifier fixes
- CMakeLists.txt shared PIO linking
- gen_mksurfdata_build GCC 14 compiler flags
- PIO version in .gitmodules (pio2_6_6)
- spillheight namelist default (research-specific)

**ccs_config (needs fork):**
- machines/hipergator/config_machines.xml
- machines/hipergator/config_batch.xml
- machines/hipergator/gnu_hipergator.cmake

#### Remaining Phases
- [ ] **Phase A:** Create GitHub forks (CTSM + ccs_config) - USER ACTION REQUIRED
- [ ] **Phase B:** Set up local CTSM fork on 5.3.085
- [ ] **Phase C:** Set up ccs_config fork with HiPerGator configs
- [ ] **Phase D:** Test and validate (build, subset_data, run)
- [ ] **Phase E:** Clean up (swap dirs, retire deploy.custom.files)
- [ ] **Phase F:** Create src.mods directory (future use)

#### Next Session Pickup Point
**User needs to create GitHub forks first:**
1. Go to https://github.com/ESCOMP/CTSM → Fork to cdevaneprugh/ctsm
2. Go to https://github.com/ESMCI/ccs_config → Fork to cdevaneprugh/ccs_config

**Then Claude will:**
1. Clone fresh from fork
2. Create uf-ctsm5.3.085 branch from upstream tag
3. Apply modifications identified in audit (only what's still needed)
4. Update .gitmodules to point to ccs_config fork
5. Add CLAUDE.md to CTSM root
6. Set up ccs_config fork with HiPerGator configs
7. Test build and run
8. Swap directories and retire deploy.custom.files

### 4.2 Version Analysis - DONE
- [x] Evaluated ctsm5.3.059 against upstream (1114 commits behind)
- [x] Latest available: ctsm5.3.085 (26 patches), ctsm5.4.007 (new major)
- [x] Used `/upstream-check ctsm` slash command
- [x] **Recommendation:** Upgrade to ctsm5.3.085
  - No hillslope changes (safe for current work)
  - Important subset_data fixes (1-PFT, Longitude TypeError)
  - RRTMGP temperature bug fix
- [x] Analysis saved: `docs/CTSM_UPSTREAM_CHECK_2025-01-11.md`

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

### 4.6 ESM Guidance File for Claude Code
- [ ] Complete `~/.claude/esm-guidance.md` with comprehensive content
- [ ] Reference from global CLAUDE.md via `@~/.claude/esm-guidance.md`
- [ ] Sections to complete:
  - Scientific Goals (research context, what we're studying)
  - Directory Structure & Model Layout (paths, organization)
  - Input Data (sources, subset data, NEON data)
  - Case Creation (compsets, namelists, spinup, runtime estimation)
  - Troubleshooting (CaseStatus workflow, common errors)
  - Case Analysis (tools, key variables, workflows)
- [ ] Draft created at `~/.claude/esm-guidance.md` - needs content after CTSM deep dive

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

**2025-01-10:** Phase 3 Case Management Tools
- Added Case Troubleshooting section to hpg-esm-tools/CLAUDE.md
- Updated hillslope script references (bin_temporal.sh, removed get_upstream_changes.sh)
- Explored subset data case workflow: examined working case, user_mods structure
- Decided Case Creator not needed as separate tool (handle in-context)
- Created `/case-check` slash command (`~/.claude/commands/case-check.md`):
  - Smoke test workflow for CTSM cases
  - File inventory (expected vs actual)
  - Quick plots for sanity check
  - Leverages existing case.analyzer scripts
- Added Phase 4.6: ESM Guidance File task
- Created `~/.claude/esm-guidance.md` draft with section placeholders
- Added `@~/.claude/esm-guidance.md` reference to global CLAUDE.md
- Discussed SLURM integration possibilities (future capability)

**2025-01-11:** Phase 3 Research & Documentation Tools
- Created `/upstream-check` slash command (`~/.claude/commands/upstream-check.md`):
  - Compares local repo to upstream remote
  - Flags relevant changes: hillslope, biogeochem, biogeophys, src/
  - Provides upgrade recommendations
- Created `doc-reconciler` agent (`~/.claude/agents/doc-reconciler.md`):
  - Uses opus model for complexity
  - Primary sources: CTSM docs, CESM forums, CLM page
  - Generates reconciliation reports for documentation discrepancies
  - Draft - will refine during Phase 4
- Deferred: Paper Summarizer (needs research context), CTSM Module Analyzer (limited use)
- Phase 3 complete - all tools either implemented or intentionally deferred
- Ran `/upstream-check ctsm` - first real use of the tool:
  - Local: ctsm5.3.059, upstream: 1114 commits behind
  - Recommendation: upgrade to ctsm5.3.085
  - Key finding: subset_data fixes, no hillslope changes
  - Saved analysis to `docs/CTSM_UPSTREAM_CHECK_2025-01-11.md`
- Phase 4.2 (Version Analysis) complete

**2025-01-11:** Phase 4.1 Fork Strategy - Audit Complete
- Read hpg-esm-docs porting documentation (shared-utils, forking-ctsm, ctsm-porting)
- Entered plan mode to design fork strategy
- User decisions: personal GitHub account, jump to 5.3.085, create ccs_config fork
- Completed Phase 0 audit of all local modifications:
  - Found modifications already applied directly to ctsm5.3 checkout (not just in deploy.custom.files)
  - Identified Longitude TypeError fixes (same as upstream 5.3.085 - can drop)
  - Identified HiPerGator-specific changes that must be kept (MPILIB, paths, mksurfdata fixes)
  - Verified HiPerGator machine configs are current (GCC 14.2.0, OpenMPI 5.0.7, ESMF 8.8.1)
  - Verified prerequisites available (shared PIO, cprnc)
- Created `docs/FORK_AUDIT_2025-01-11.md` with detailed findings
- Plan saved at `~/.claude/plans/indexed-chasing-hammock.md`
- **Stopping point:** Ready for Phase A - user needs to create GitHub forks before next session
