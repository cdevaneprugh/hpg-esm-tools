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

**Active environments:** `ctsm` (default), `ferret`

**Usage:** `module load conda && conda activate ctsm`

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

## Phase 4: CTSM Deep Work - COMPLETE

Long-term efforts requiring the tools built in Phase 3. All subsections complete.

### 4.1 CTSM Fork Strategy - COMPLETE

**Status:** All phases complete. Fork strategy operational.

**Plan file:** `~/.claude/plans/indexed-chasing-hammock.md`
**Audit results:** `docs/FORK_AUDIT_2025-01-11.md`

#### Final Setup
| Component | Location |
|-----------|----------|
| CTSM fork | `github.com/cdevaneprugh/CTSM` branch `uf-ctsm5.3.085` |
| ccs_config fork | `github.com/cdevaneprugh/ccs_config_cesm` branch `uf-hipergator` |
| Local checkout | `/blue/gerber/cdevaneprugh/ctsm5.3` |
| Old checkout | `/blue/gerber/cdevaneprugh/ctsm5.3.old` (archived) |
| Legacy scripts | `scripts/deploy.custom.files.archived/` |

#### Completed Phases
- [x] **Phase 0:** Audit local modifications
- [x] **Phase A:** Create GitHub forks (CTSM + ccs_config_cesm)
- [x] **Phase B:** Set up local CTSM fork on 5.3.085
- [x] **Phase C:** Set up ccs_config fork with HiPerGator configs
- [x] **Phase D:** Test and validate (case built and ran successfully)
- [x] **Phase E:** Clean up (directories swapped, deploy.custom.files archived)

#### Key Notes
- ccs_config repo was renamed to `ccs_config_cesm` by ESMCI
- Removed `--exclusive` from default SLURM config (HiPerGator uses shared nodes)
- Existing cases don't pick up batch config changes - must create fresh case
- Test case: `ctsm-fork.I1850Clm60SpCru.f09_g17.260112-122859`

### 4.2 Version Analysis - DONE
- [x] Evaluated ctsm5.3.059 against upstream (1114 commits behind)
- [x] Latest available: ctsm5.3.085 (26 patches), ctsm5.4.007 (new major)
- [x] Used `/upstream-check ctsm` slash command
- [x] **Recommendation:** Upgrade to ctsm5.3.085
  - No hillslope changes (safe for current work)
  - Important subset_data fixes (1-PFT, Longitude TypeError)
  - RRTMGP temperature bug fix
- [x] Analysis saved: `docs/CTSM_UPSTREAM_CHECK_2025-01-11.md`

### 4.3 Claude Code Best Practices for CTSM Development - COMPLETE

**Research complete.** Extensive documentation crawled and compiled.

**Resources Crawled:**
- CTSM GitHub Wiki (69 pages)
- UCAR CLM Developers Guide + subpages
- CESM Tutorial (SourceMods, debugging, output)
- DiscussCESM Forums

**Documentation Created:**
- `docs/CTSM_DEVELOPMENT_GUIDE.md` - Quick reference guide
- `~/.claude/esm-guidance.md` - Updated with workflow sections
- `~/.claude/plans/indexed-chasing-hammock.md` - Full research findings

**Reference Priority Established:**
1. Local summaries (esm-guidance.md, CTSM_DEVELOPMENT_GUIDE.md)
2. Official docs (CTSM wiki, tech note, user guide)
3. CESM forums (community knowledge)
4. Third-party sources (verify carefully)

**Topics Covered:**
- Git workflow and PR requirements
- Coding conventions (indentation, comments, argument passing)
- Testing (aux_clm, test categories, baselines)
- SourceMods usage
- Adding features (namelist items, restart variables, parameters)
- History output configuration
- Debugging techniques
- Spinup procedures
- git-fleximod submodule management
- FATES integration
- Machine porting

### 4.4 CTSM Source Code Deep Dive - COMPLETE

**Status:** Comprehensive CLAUDE.md documentation created across CTSM repository.

**Plan file:** `~/.claude/plans/stateful-brewing-mist.md`

- [x] Explore tools/, python/, src/, libraries/ directories
- [x] Document wrapper→implementation architecture (tools/ vs python/ctsm/)
- [x] Document Fortran source organization (subgrid hierarchy, type system)
- [x] Document testing infrastructure (5 testing systems)
- [x] Research mpi-serial and PIO library status
- [x] Create CLAUDE.md files for Claude Code context loading

**Files Created (13 total):**
```
/blue/gerber/cdevaneprugh/ctsm5.3/
├── claude-todo.md                    # Progress tracking
├── TESTING.md                        # 5 testing systems guide
├── tools/CLAUDE.md                   # Tool inventory, decision tree
├── tools/mksurfdata_esmf/CLAUDE.md   # Build process, HiPerGator mods
├── tools/site_and_regional/CLAUDE.md # subset_data documentation
├── python/CLAUDE.md                  # Package structure
├── python/ctsm/site_and_regional/CLAUDE.md  # Implementation details
├── src/CLAUDE.md                     # Fortran organization
├── src/main/CLAUDE.md                # Driver, types, control vars
├── src/biogeophys/CLAUDE.md          # Hydrology, energy, hillslope
├── src/biogeochem/CLAUDE.md          # Carbon-nitrogen cycling
└── libraries/CLAUDE.md               # mpi-serial/PIO research
```

**Key Findings:**
- tools/ contains thin wrappers (20-40 lines), python/ctsm/ has implementations (~5,600 LOC)
- 5 testing systems: run_sys_tests, CIME create_test, Fortran pFUnit, Python pytest, FATES
- mpi-serial not used on HiPerGator (openmpi workaround sufficient)
- Shared PIO already configured at `/blue/gerber/earth_models/shared/parallelio/bld`

### 4.5 hpg-esm-docs Integration - COMPLETE

**Status:** Documentation overhauled and deployed to GitHub Pages.

**Live site:** https://cdevaneprugh.github.io/hpg-esm-docs/

**Tone shift:** From "Gerber group docs" → "HiPerGator community resource"
- Core message: "Here's how to set this up yourself (and here's our fork if you want it)"
- Generalized all paths with `<group>` placeholders
- Reframed environment variables as recommendations

**New pages created:**
- `installation/quickstart.md` - Clean step-by-step installation guide
- `installation/cime-config.md` - Detailed config file explanation

**Pages updated:**
- `index.md` - Removed personal paths, reworded to "Reference Fork"
- `onboarding.md` - Generalized paths, reframed env vars
- `prerequisites.md` - Reframed as example, added data size warning
- `fork-setup.md` → "Fork Reference" - Expanded "Why We Fork" section

**Tracking:** `hpg-esm-docs/DOCUMENTATION_TODO.md` for ongoing improvements

### 4.6 ESM Guidance File for Claude Code - COMPLETE
- [x] Complete `~/.claude/docs/esm-guidance.md` with comprehensive content
- [x] Reference from global CLAUDE.md via `@~/.claude/docs/esm-guidance.md`
- [x] Sections completed:
  - CTSM Fork Setup (repositories, modifications, upstream tracking)
  - CTSM Source Structure (directories, subgrid hierarchy)
  - Case Workflow (lifecycle, directories, run types)
  - Spinup Procedures (AD spinup, monitoring, equilibrium)
  - History Output (configuration, frequency, averaging)
  - Troubleshooting (CaseStatus, debug mode, common issues)
  - SourceMods (when to use, workflow)
  - Documentation Reference Priority
  - Scientific Goals (DOE project, wetlandscapes, TAI, OSBS)
  - Hillslope Hydrology (structure, parameters, physical principles)
  - Input Data (strategy, hillslope data)
  - Case Analysis (h0/h1/h2 streams, key variables, column mapping)

### 4.7 Upstream Contributions - COMPLETE
- [x] Evaluate and submit PRs for genuine bugs found during porting
- [x] Analysis saved: `docs/CTSM_MODIFICATION_ANALYSIS_2025-01-12.md`

**PRs Submitted (2026-01-21):**

1. **CMakeLists.txt STATIC/SHARED fix**
   - PR: https://github.com/ESCOMP/CTSM/pull/3700
   - Branch: `fix/mksurfdata-cmake-shared-pio`
   - Bug: declares `STATIC IMPORTED` but points to `.so` files
   - Fix: change to `SHARED IMPORTED`

2. **mksurfdata.F90 format specifiers**
   - PR: https://github.com/ESCOMP/CTSM/pull/3701
   - Branch: `fix/mksurfdata-format-specifiers`
   - Bug: `I` format without width fails on GCC 10+
   - Fix: `I0` for write statements, `I6` for read statements (matches CTSM style)

**Not submitted (workarounds, not fixes):**
- gen_mksurfdata_build GCC flags - masks underlying legacy Fortran issues

### 4.8 Conda Environment Configuration - RESOLVED
- [x] Explored running CTSM entirely with conda environments
- [x] **Conclusion: Hybrid approach is optimal for HPC**
- Research findings (2026-01-13):
  - conda MPI has SLURM compatibility issues (PMIx support often missing)
  - System MPI (lmod) has native SLURM integration and cluster optimizations
  - ESMF requires MPI, so can't use conda for full isolation
- **Adopted approach:**
  - lmod: Compilers, MPI, NetCDF, HDF5, ESMF for CTSM builds
  - Conda (ctsm): Python 3.12, editor, linters, dev tools for Claude Code sessions
- User workflow:
  1. Load module collection with CTSM build deps + conda module
  2. Activate `ctsm` conda for development work
- This provides best of both worlds: optimized builds + modern dev tools

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

### 5.4 Shared PIO Build Strategy - COMPLETE
- [x] Figure out optimal shared PIO build for subset data and single point scripts
- [x] Current approach: shared build at `/blue/gerber/earth_models/shared/parallelio/bld`
- [x] Solution implemented (2026-01-13):
  - Added `PIO_VERSION_MAJOR=2` to config_machines.xml
  - Added `PIO_TYPENAME_VALID_VALUES=netcdf`
  - Added `LD_LIBRARY_PATH` for runtime dynamic linking
  - case.build now uses external PIO instead of rebuilding
- [x] Committed to ccs_config fork (ed750f2)

### 5.5 MPI-Serial Library - ABANDONED
- [x] Investigated mpi-serial as alternative to OpenMPI for single-core runs
- [x] **Conclusion: Not feasible with modern CTSM**
- Findings (2026-01-13):
  - CTSM 5.3+ uses CDEPS/CMEPS data model (replaced deprecated data models)
  - CDEPS/CMEPS requires ESMF
  - ESMF on HiPerGator is built with OpenMPI, incompatible with mpi-serial
  - Build fails at link stage: "undefined reference to symbol 'ompi_mpi_unsigned_short'"
- Workaround: Use OpenMPI with NTASKS=1 for single-point runs (current approach works)

### 5.6 Possible Bug Fixes (Upstream Contributions)

Potential CTSM issues we could contribute fixes for.

#### 5.6.1 #2263 - SMINN_TO_PLANT_FUN units incorrect ⭐ EASY
- **File:** `src/biogeochem/CNVegNitrogenFluxType.F90`
- **Issue:** Vertically-resolved variant `sminn_to_plant_fun_no3_vr` has wrong units in history metadata
- **Fix:** Change `units='gN/m^2/s'` to `units='gN/m^3/s'`
- **Labels:** bfb, bug (bit-for-bit, won't change model answers)
- **Link:** https://github.com/ESCOMP/CTSM/issues/2263

#### 5.6.2 #2767 - Broken external links in docs
- **Location:** `doc/` directory `.rst` files
- **Issue:** Multiple broken URLs in Technical Note and User's Guide
- **Fix:** Find and correct broken external links
- **Labels:** good first issue, size: small, documentation
- **Note:** Already assigned to Adrianna Foster - check status before starting
- **Link:** https://github.com/ESCOMP/CTSM/issues/2767

#### 5.6.3 #793 - htvp determination looks wrong
- **File:** `src/biogeophys/CanopyTemperatureMod.F90` (lines 409-410)
- **Issue:** Binary switch between `hvap`/`hsub` when liquid water == 0 exactly; should use weighted average
- **Labels:** bug, science (changes model answers)
- **Complexity:** Medium - requires scientific justification
- **Note:** Open since 2019 - may be controversial or intentionally deferred
- **Link:** https://github.com/ESCOMP/CTSM/issues/793

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
- Created `ctsm` conda environment (Python 3.12)
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

**2025-01-12:** Modification Root Cause Analysis
- Analyzed why local CTSM modifications were necessary
- CMakeLists.txt STATIC/SHARED: upstream bug - declares STATIC but points to .so files
- mksurfdata.F90 format specifiers: legacy Fortran without width fails on GCC 10+
- gen_mksurfdata_build flags: GCC 14 workarounds for legacy code
- Identified 2 strong upstream contribution candidates, 1 possible
- Created `docs/CTSM_MODIFICATION_ANALYSIS_2025-01-12.md`
- Added Phase 4.7 (Upstream Contributions) to todo list

**2026-01-12:** Phase 4.1 Fork Strategy - COMPLETE
- User created GitHub forks: `cdevaneprugh/CTSM`, `cdevaneprugh/ccs_config_cesm`
- Note: ccs_config was renamed to ccs_config_cesm by ESMCI
- Phase A: Cloned CTSM fork, created `uf-ctsm5.3.085` branch from tag
- Phase B: Applied all HiPerGator-specific modifications to CTSM fork
  - mksurfdata Fortran fixes, CMakeLists.txt, gen_mksurfdata_build
  - single_point_case.py MPILIB, default_data paths, spillheight
  - Added CLAUDE.md to CTSM root
- Phase C: Set up ccs_config fork
  - Initially tried ~/.cime approach - hit CIME v3 bugs
  - `--share` directive not recognized by HiPerGator SLURM
  - Forked ccs_config_cesm, created `uf-hipergator` branch
  - Added HiPerGator machine config (config_machines.xml, config_batch.xml, gnu_hipergator.cmake)
  - Removed `--exclusive` from default SLURM config
  - Updated CTSM .gitmodules to point to ccs_config fork
- Phase D: Test validation
  - First case failed: `--share` flag error (stale batch config in existing case)
  - Created fresh case: `ctsm-fork.I1850Clm60SpCru.f09_g17.260112-122859`
  - Build succeeded, model ran 5 days successfully
- Phase E: Cleanup
  - Swapped directories: ctsm5.3 → ctsm5.3.old, ctsm5.3.new → ctsm5.3
  - Archived deploy.custom.files to deploy.custom.files.archived
  - Updated CLAUDE.md with fork information
  - Updated esm-guidance.md with fork setup section

**2026-01-12:** Phase 4.3 CTSM Development Best Practices - COMPLETE
- Crawled primary resources:
  - CTSM GitHub Wiki (69 pages covering development workflow, coding guidelines, testing)
  - UCAR CLM Developers Guide (coding conventions, namelist items, restart variables)
  - CESM Tutorial (SourceMods, debugging, output configuration)
  - DiscussCESM Forums (debugging tips, community solutions)
- Topics researched:
  - Git workflow and PR requirements
  - CLM coding conventions (indentation, comments, argument passing)
  - Testing (aux_clm, baselines, status codes)
  - SourceMods usage and workflow
  - Adding features (namelist items, restart variables, parameters)
  - History output (hist_fincl, averaging flags)
  - Debugging (DEBUG mode, INFO_DBUG, log analysis)
  - Spinup procedures (AD spinup, equilibrium criteria)
  - git-fleximod submodule management
  - FATES integration protocols
  - Machine porting (CTSM 5.2+ structure)
- Documentation created:
  - `docs/CTSM_DEVELOPMENT_GUIDE.md` - Quick reference guide
  - `~/.claude/esm-guidance.md` - Updated with workflow sections
  - `~/.claude/plans/indexed-chasing-hammock.md` - Full research findings
- Established reference priority:
  1. Local summaries (esm-guidance.md, CTSM_DEVELOPMENT_GUIDE.md)
  2. Official docs (CTSM wiki, tech note)
  3. CESM forums
  4. Third-party (verify carefully)
- Documentation organization:
  - Created `docs/CTSM_RESEARCH_NOTES.md` with full research findings
  - Organized `~/.claude/docs/` directory structure
  - Moved `esm-guidance.md` to `~/.claude/docs/`
  - Updated `~/.claude/CLAUDE.md` reference path
  - Added Documentation Index to `hpg-esm-tools/CLAUDE.md`

**2026-01-13:** Phase 4.5 hpg-esm-docs Integration - COMPLETE
- Major documentation overhaul for broader HiPerGator audience
- Tone shift: "Gerber group docs" → "HiPerGator community resource"
- Core message: "Here's how to set this up yourself (and here's our fork if you want it)"
- New pages created:
  - `installation/quickstart.md` - Clean step-by-step installation guide
  - `installation/cime-config.md` - Detailed CIME config file explanation
- Pages updated:
  - `index.md` - Removed personal paths, "Our Setup" → "Reference Fork"
  - `onboarding.md` - Generalized paths with `<group>` placeholders
  - `prerequisites.md` - Reframed as example structure, added input data size warning
  - `fork-setup.md` - Renamed to "Fork Reference", expanded "Why We Fork"
- Navigation reorganized: Quick Start first in Installation section
- Created `DOCUMENTATION_TODO.md` for ongoing tracking
- Set up dark mode (slate theme)
- Deployed to GitHub Pages: https://cdevaneprugh.github.io/hpg-esm-docs/
- Added future tasks: 4.8 (Conda env configuration), 5.5 (MPI-serial)

**2026-01-13:** Phase 4.4 CTSM Source Code Deep Dive - COMPLETE
- Created comprehensive CLAUDE.md documentation across CTSM repository (13 files, ~3,440 lines)
- Explored and documented 4 major areas: tools/, python/, src/, libraries/
- Key architectural finding: tools/ contains thin CLI wrappers (20-40 lines), python/ctsm/ has actual implementations (~5,600 LOC)
- Documented 5 testing systems with invocations and HiPerGator notes:
  - run_sys_tests (CTSM orchestration)
  - CIME create_test (200+ integration tests)
  - Fortran pFUnit (module unit tests)
  - Python pytest (~12,300 LOC)
  - FATES testing suite
- Library research findings:
  - mpi-serial: Not used on HiPerGator (openmpi workaround is sufficient)
  - PIO: Shared build already configured at /blue/gerber/earth_models/shared/parallelio/bld
- Created TESTING.md at CTSM root for comprehensive testing guidance
- Documented Fortran source organization:
  - Subgrid hierarchy: Gridcell → LandUnit → Column → Patch
  - Type system and clm_instMod.F90 (~50+ instances)
  - Driver calling sequence and initialization order
  - biogeophys/ (hydrology, energy, hillslope) and biogeochem/ (C-N cycling)
- Updated hpg-esm-tools/CLAUDE.md documentation index to reference new CTSM docs
- All documentation committed and pushed to CTSM fork

**2026-01-13:** Phase 4.8 Environment Modernization - PIO Complete, mpi-serial Abandoned
- Investigated conda-based CTSM environment:
  - Research showed conda MPI has SLURM compatibility issues on HPC
  - User chose hybrid approach: lmod for builds, conda for dev tools (ctsm)
- Enabled shared PIO library usage:
  - Added `PIO_VERSION_MAJOR=2` to config_machines.xml (tells case.build to use external PIO)
  - Added `PIO_TYPENAME_VALID_VALUES=netcdf` (specifies supported I/O backends)
  - Added `LD_LIBRARY_PATH` for runtime dynamic linking to PIO shared libs
  - Test case (f45_g37 global, 5 days) ran successfully with shared PIO
- Investigated mpi-serial for single-point interactive runs:
  - mpi-serial build failed: ESMF on HiPerGator requires OpenMPI
  - Modern CTSM with CDEPS/CMEPS requires ESMF, blocking mpi-serial usage
  - User chose to abandon mpi-serial entirely
- Added comprehensive documentation comments to config files:
  - config_machines.xml: Machine identification, modules, PIO paths, environment variables
  - config_batch.xml: QOS queues, partition settings, removed --exclusive explanation
- Committed changes:
  - ccs_config: ed750f2 "Add shared PIO support and documentation to HiPerGator config"
  - CTSM: b0a18e442 "Update ccs_config submodule: shared PIO support"
- Both repositories pushed to GitHub
- Test case cleaned up: `claude-test.I1850Clm60Sp.f45_g37.pio-test`
- Updated Phase 5.4 (Shared PIO) and 5.5 (MPI-Serial) - both now documented as resolved/abandoned

**2026-01-14:** Phase 4.6 ESM Guidance - COMPLETE
- Created paper summaries for research context:
  - `docs/papers/DOE_Grant_Summary.md` - Project scope and actual focus
  - `docs/papers/Fan_2019_Hillslope_Hydrology_ESM_Summary.md` - Scientific foundation
  - `docs/papers/Swenson_2025_Hillslope_Dataset_Summary.md` - Dataset methodology
- Updated `~/.claude/docs/esm-guidance.md` with comprehensive project context:
  - Scientific Goals (DOE project, wetlandscapes, TAI, OSBS site)
  - Hillslope Hydrology section (structure, parameters, physical principles)
  - Input Data details (current strategy, hillslope data workflow)
  - Case Analysis (h0/h1/h2 streams, key variables, column mapping)
- Updated hpg-esm-docs research section:
  - `docs/research/overview.md` - DOE project context, research areas
  - `docs/research/hillslope.md` - Comprehensive hillslope documentation
  - `docs/research/neon-sites.md` - OSBS details, data workflow
- Phase 4.6 now complete - all TODO sections filled in

**2026-01-21:** Phase 4.7 Upstream Contributions - COMPLETE
- Submitted two PRs to ESCOMP/CTSM for mksurfdata_esmf bugs:
  1. CMakeLists.txt STATIC→SHARED fix for shared PIO libraries
  2. Format specifier fix (I→I0/I6) for GCC 10+ compatibility
- Key finding during testing: `I0` works for write but not read (requires positive width)
- Correct fix: `I0` for write statements, `I6` for read statements
- Verified build succeeds on HiPerGator with GCC 14.2.0
- Searched GitHub issues - no existing issues for these bugs
- Fork branches: `fix/mksurfdata-cmake-shared-pio`, `fix/mksurfdata-format-specifiers`
- Fork branch `uf-ctsm5.3.085` also updated with corrected format specifiers
- Phase 4 now fully complete
