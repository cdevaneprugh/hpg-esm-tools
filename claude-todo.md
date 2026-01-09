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

### 1.2 Workspace Organization
- [x] Set up git repos (hpg-esm-tools, hpg-esm-docs)
- [ ] Review existing scripts and docs in hpg-esm-tools
  - case.analyzer/ - case analysis toolkit
  - hillslope.analysis/ - hillslope hydrology analysis (note: contains large .nc files that should be gitignored)
  - Old scripts - assess for useful code snippets
- [ ] Remove redundant/outdated information from docs
- [ ] Determine which scripts to: keep, remove, merge, refactor, or convert to agents/commands
- [x] Add .nc files to .gitignore (created .gitignore)

### 1.3 Bashrc/Vimrc Cleanup - COMPLETE
- [x] Remove defunct `newcase` alias
- [x] Fix `deploy_configs` path
- [x] Add `esm` alias for conda activation
- [x] Add `vim`/`vi` aliases for neovim
- [x] Migrate to neovim with full lua config
- [x] LSP support: pylsp (Python), fortls (Fortran)
- [x] Completion via nvim-cmp
- [x] Column 80 marker, spell check, gruvbox theme
- [x] Move neovim to conda (v0.11.5) - no lmod dependency
- [x] Add true color detection to options.lua

**Usage:** `esm` (single command activates conda env with neovim + all tools)

---

## Phase 2: Script Refactoring

After conda environment is set up and scripts have been reviewed:

- [ ] Create consistent format for plotting scripts
- [ ] Add comments to existing code where needed
- [ ] Create/revise documentation for scripts
- [ ] Run linters and formatters on all scripts

**Key question to answer during refactor:** For scripts being converted to agents, should we keep code snippets for Claude to leverage, or is a clean agent specification document sufficient?

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
- Migrated from vim to neovim with full lua config:
  - lazy.nvim plugin manager
  - LSP: pylsp (Python), fortls (Fortran)
  - nvim-cmp completion
  - treesitter syntax highlighting
  - gruvbox theme, 80-column marker, spell check
- Moved neovim from lmod module to conda environment:
  - Added nvim>=0.11.0 to environment.yml (now at v0.11.5)
  - Updated options.lua with true color detection (COLORTERM check)
  - Single `esm` command now activates everything: neovim + LSP servers + linters
  - No more `module load neovim` needed
  - Updated CLAUDE.md with new editor setup documentation
