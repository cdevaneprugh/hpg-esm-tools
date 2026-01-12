# CTSM Research Notes

Detailed findings from Phase 4.3 research on CTSM development best practices for Claude Code.

**Date:** 2026-01-12

---

## Primary Documentation Sources

### Official Documentation (Tier 1 - Authoritative)
| Resource | URL | Content |
|----------|-----|---------|
| CTSM GitHub Wiki | https://github.com/ESCOMP/ctsm/wiki | 69 pages of development documentation |
| CTSM Technical Note | https://escomp.github.io/CTSM/tech_note/index.html | 32 chapters on model science |
| CTSM User's Guide | https://escomp.github.io/CTSM/users_guide/overview/introduction.html | Setup and running |
| UCAR CLM Developers Guide | https://wiki.ucar.edu/spaces/ccsm/pages/37431066 | Development workflow |
| CESM Tutorial | https://ncar.github.io/CESM-Tutorial/ | Hands-on guides |

### CTSM Wiki Key Pages (Tier 1)
| Page | Purpose |
|------|---------|
| CTSM development workflow | Testing and PR requirements |
| CTSM coding guidelines | Links to CLM coding conventions |
| Quick start to CTSM development with git | Git workflow, branching, PRs |
| CTSM PR Expectations | Requirements for contributions |
| System Testing Guide | Test categories and commands |
| List of common problems | Common bugs and fixes |
| Moving a parameter to the params file | NetCDF parameter workflow |
| Adding biogeochem changes to CN Matrix | Matrix solution integration |
| Protocols on updating FATES within CTSM | FATES submodule management |

### UCAR Wiki Key Pages (Tier 1)
| Page | Purpose |
|------|---------|
| CLM Coding Conventions | Fortran style, comments, indentation |
| CLM Testing | Test commands and categories |
| Adding New Namelist Items to CLM | 7-step process for namelists |
| Adding Variables to the Restart File | restartvar() usage |

### Forum Resources (Tier 2 - Community)
| Resource | URL |
|----------|-----|
| DiscussCESM Forums | https://bb.cgd.ucar.edu/cesm/ |
| CLM tag on forums | https://bb.cgd.ucar.edu/cesm/tags/clm/ |

---

## CTSM Development Workflow Summary

### Git Workflow
1. Clone with `--origin escomp` from ESCOMP/CTSM
2. Create feature branch from master: `git checkout -b MYBRANCH master`
3. Run `./bin/git-fleximod update` to populate submodules
4. Never commit directly to master or release branches
5. Push to personal fork, create PR against ESCOMP/CTSM

### Testing Requirements
**Before PR submission:**
- Run `aux_clm` test suite on supported machines (cheyenne, izumi)
- Compare against baselines to detect answer changes
- Pass all tests or document expected failures

**Test Categories:**
| Category | Purpose |
|----------|---------|
| aux_clm | Primary suite for development |
| clm_short | Quick subset for iteration |
| fates | FATES-specific tests |
| aux_cime_baselines | Nightly CIME tests |

**Test Command:**
```bash
./create_test --xml-category aux_clm -c COMPARE -g GENERATE
```

**Status Codes:** PASS, SFAIL (script), CFAIL (compile), BFAIL (baseline)

### PR Requirements
1. Scientific value OR bug fix/refactoring
2. No harm to other components
3. Code quality per CTSM coding guidelines
4. Pass test suite + add new tests
5. Documentation for new science capabilities

---

## CLM Coding Conventions

### Comments
- Comment after `else`/`endif` for nested blocks
- Document pointer intent (IN, OUT, INOUT)
- Explain algorithms and processes

### Argument Passing (Ranked by Preference)
1. **Subroutine interface with `intent`** (preferred)
2. **Pointer with associate + intent comments**
3. **Global data** (avoid for new code)

### Indentation
| Block Type | Spaces |
|------------|--------|
| do, if, while | 3 |
| program, module, subroutine | 2 |
| associate | 0 |
| continuation lines | 5 |

### Key Rules
- **No CPP tokens** - use conditional logic instead
- **No code duplication** - extract to subroutines
- **Remove trailing whitespace**
- **Single statement per line**
- **Keep subroutines short**

---

## Common Problems and Solutions

### Model Crashes
| Problem | Solution |
|---------|----------|
| Divide by zero | Conditional check for 0 values |
| Floating point exceptions | Handle impossible cases |
| Negative rounding | Use `max(foo, 0._r8)` or `truncate_small_values` |

### Parallelization Issues (PEM/ERP failures)
| Problem | Solution |
|---------|----------|
| Missing namelist broadcast | Add `shr_mpi_bcast` calls |
| Subgrid variable indexing | Match index type (patch, column, gridcell) |
| Uninitialized scalars | Initialize before use in loops |

### Threading Bugs
| Problem | Solution |
|---------|----------|
| Full array initialization | Use bounds: `foo(bounds%begc:bounds%endc)` |
| Missing array indices | Use `variable_patch(p)` not `variable_patch` |
| Private variable lists | Add all temporaries to PRIVATE clause |

---

## SourceMods Usage

### Directory Structure
```
case_directory/SourceMods/src.clm/   # CLM modifications
case_directory/SourceMods/src.cam/   # CAM modifications
```

### Workflow
1. Locate subroutine in CTSM source
2. Copy entire file to appropriate SourceMods directory
3. Make modifications
4. Rebuild and run

### When to Use
- Case-specific science changes
- Experimental modifications
- Runtime Fortran changes

### When NOT to Use (Use Fork Instead)
- Build tool fixes (mksurfdata, subset_data)
- HiPerGator-specific changes
- Modifications needed across all cases

---

## Adding New Features

### New Namelist Items (7 Steps)
1. Identify target module
2. Add data as local to module (private)
3. Add/modify initialization subroutine
4. Call from clm_initializeMod.F90
5. Add to namelist read with broadcast
6. Add entry to namelist_definition_clm4_5.xml
7. Update build-namelist if new group

### New Restart Variables
```fortran
call restartvar(ncid=ncid, flag=flag, varname='VARIABLE_NAME',
     xtype=ncd_double, dim1name='column',
     long_name='description', units='K',
     interpinic_flag='interp', readvar=readvar, data=pointer)
```

**Rule:** Only prognostic state variables belong on restart files.

### New Parameters (to params file)
1. Extract with `ncdump -p 9,17 params.nc > params.cdl`
2. Edit CDL to add variable + data
3. Generate with `ncgen -o new_params.nc params.cdl`
4. Create patch file: `diff -u old.cdl new.cdl > patch`
5. Update namelist_defaults_ctsm.xml
6. Add code to read parameter

---

## History Output Configuration

### Key Namelist Variables
| Variable | Purpose |
|----------|---------|
| `hist_nhtfrq` | Output frequency (0=monthly, >0=timesteps, <0=hours) |
| `hist_mfilt` | Time samples per file |
| `hist_fincl1-10` | Variables for h0-h9 files |
| `hist_empty_htapes` | Clear default output |

### Averaging Flags
| Flag | Meaning |
|------|---------|
| A | Average |
| I | Instantaneous |
| M | Minimum |
| X | Maximum |
| SUM | Sum |

**Example:** `hist_fincl1 = 'TSOI:X'` outputs maximum soil temperature.

### Example user_nl_clm
```fortran
hist_empty_htapes = .true.
hist_fincl1 = 'TSA', 'TSKIN', 'GPP', 'EFLX_LH_TOT'
hist_nhtfrq = -24
hist_mfilt = 365
```

---

## Debugging Techniques

### Runtime Debugging (No Rebuild)
```bash
./xmlchange INFO_DBUG=2
```
Adds diagnostic output to cpl.log.

### Compile-Time Debugging
```bash
./xmlchange DEBUG=TRUE
./case.build --clean-all
./case.build
```
Enables bounds checking, FPE trapping. **Runs significantly slower.**

### Log File Analysis
1. Check cesm.log for component errors
2. Check lnd.log for CLM-specific issues
3. Check CaseStatus for workflow failures

### Debugger Tools
- **DDT** (ARM Forge) - available on NCAR systems
- **Totalview** - commercial debugger
- **Write statements** - traditional approach

---

## Spinup Procedures

### AD Spinup (Accelerated Decomposition)
```bash
./xmlchange CLM_ACCELERATED_SPINUP=on
```
~200 years to approximate steady state.

### Post-AD Spinup
```bash
./xmlchange CLM_ACCELERATED_SPINUP=off
```
Several hundred years to final equilibrium.

### Equilibrium Criteria
- Less than 3% of land surface in carbon disequilibrium
- Monitor: TOTECOSYSC, TOTSOMC, TOTVEGC, GPP, TWS
- Arctic regions take longest (~1000 years)

---

## Submodule Management (git-fleximod)

### Basic Commands
```bash
./bin/git-fleximod status    # Check submodule status
./bin/git-fleximod update    # Populate/update submodules
./bin/git-fleximod --help    # Usage guide
```

### When to Rerun
- After `git checkout` to different tag/branch
- After `git merge` from master
- Whenever `.gitmodules` changes

### Customizing Submodules
1. Edit `.gitmodules` (change fxtag)
2. Run `./bin/git-fleximod update <submodule>`
3. Commit `.gitmodules` change

**Warning:** Changing submodule versions may produce invalid builds.

---

## FATES Integration

### Updating FATES Version
```bash
cd src/fates
git checkout <fates_tag>
```

### Development with Custom FATES
1. Add fork as remote: `git remote add $USER git@github.com:$USER/fates.git`
2. Create branch: `git checkout -b <branch> origin/master`
3. Push to fork: `git push -u $USER <branch>`

### API Changes
Require coordination between FATES and CTSM repositories with parallel PRs.

---

## Machine Porting (CTSM 5.2+)

### Required Files
| File | Location |
|------|----------|
| config_machines.xml | ccs_config/machines/ (add NODENAME_REGEX) |
| config_machines.xml | ccs_config/machines/<machine>/ (full definition) |
| <compiler>_<machine>.cmake | ccs_config/machines/<machine>/ |

### Key Differences from CESM
- CTSM uses `ccs_config/machines/` not `$CIMEROOT/config/`
- Uses cmake macros instead of config_compilers.xml
- NODENAME_REGEX in separate file from machine config

---

## Reference Priority for Claude Code

1. **Local summaries** (CTSM_DEVELOPMENT_GUIDE.md, esm-guidance.md)
2. **Official docs** (CTSM wiki, tech note, user guide)
3. **CESM forums** (community knowledge)
4. **Third-party sources** (verify carefully)
