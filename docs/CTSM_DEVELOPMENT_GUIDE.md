# CTSM Development Guide

Quick reference for CTSM development best practices, compiled from official documentation.

**Primary Sources:**
- [CTSM GitHub Wiki](https://github.com/ESCOMP/ctsm/wiki)
- [CLM Developers Guide](https://wiki.ucar.edu/spaces/ccsm/pages/37431066)
- [CESM Tutorial](https://ncar.github.io/CESM-Tutorial/)

---

## Git Workflow

```bash
# Clone with named remote
git clone --origin escomp https://github.com/ESCOMP/CTSM.git

# Create feature branch
git checkout -b MYBRANCH master

# Populate submodules
./bin/git-fleximod update

# Never commit directly to master
```

**Key Rules:**
- Create separate clones for each branch to avoid case interference
- Run `git-fleximod update` after cloning and after `.gitmodules` changes
- Push to personal fork, create PR against ESCOMP/CTSM

---

## Testing

### Test Categories
| Category | Purpose |
|----------|---------|
| `aux_clm` | Primary suite for development (required before PR) |
| `clm_short` | Quick subset for iteration |
| `fates` | FATES-specific tests |

### Commands
```bash
# Run test suite
./create_test --xml-category aux_clm -c COMPARE -g GENERATE

# Run single test
./create_test TESTNAME

# Check results
./cs.status.TESTID
./cs.status.fails  # Only failures
```

### Status Codes
- **PASS** - Test successful
- **SFAIL** - Script creation failure
- **CFAIL** - Compilation failure
- **BFAIL** - Baseline comparison failure

---

## Coding Conventions

### Indentation
| Block Type | Spaces |
|------------|--------|
| do, if, while | 3 |
| program, module, subroutine | 2 |
| associate | 0 |
| continuation lines | 5 |

### Key Rules
- Comment after `else`/`endif` for nested blocks
- Document pointer intent (IN, OUT, INOUT)
- No CPP tokens - use conditional logic
- No code duplication - extract to subroutines
- Remove trailing whitespace
- Keep subroutines short

### Argument Passing (Ranked)
1. **Subroutine interface with `intent`** (preferred)
2. **Pointer with associate + intent comments**
3. **Global data** (avoid for new code)

---

## SourceMods

### Directory Structure
```
case_directory/SourceMods/src.clm/   # CLM modifications
```

### Workflow
1. Locate subroutine in CTSM source (`src/`)
2. Copy **entire file** to SourceMods directory
3. Make modifications
4. Rebuild and run

### When to Use
- Case-specific science changes
- Experimental modifications
- Runtime Fortran changes

### When NOT to Use
- Build tool fixes (use fork)
- Machine-specific changes (use ccs_config)
- Changes needed across all cases

---

## Adding New Features

### New Namelist Items (7 Steps)
1. Identify target module
2. Add data as local to module (private)
3. Add/modify initialization subroutine
4. Call from `clm_initializeMod.F90`
5. Add to namelist read with `shr_mpi_bcast`
6. Add entry to `namelist_definition_clm4_5.xml`
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
```bash
# Extract
ncdump -p 9,17 params.nc > params.cdl

# Edit CDL, add variable + data

# Generate
ncgen -o new_params.nc params.cdl

# Create patch
diff -u old.cdl new.cdl > patch

# Update namelist_defaults_ctsm.xml
# Add code to read parameter
```

---

## History Output

### Key Variables in user_nl_clm
| Variable | Purpose |
|----------|---------|
| `hist_nhtfrq` | Frequency: 0=monthly, >0=timesteps, <0=hours |
| `hist_mfilt` | Time samples per file |
| `hist_fincl1-10` | Variables for h0-h9 files |
| `hist_empty_htapes` | Clear default output |

### Averaging Flags
`A`=Average, `I`=Instantaneous, `M`=Minimum, `X`=Maximum, `SUM`=Sum

### Example
```fortran
hist_empty_htapes = .true.
hist_fincl1 = 'TSA', 'TSKIN', 'GPP', 'EFLX_LH_TOT'
hist_nhtfrq = -24
hist_mfilt = 365
```

---

## Debugging

### Runtime (No Rebuild)
```bash
./xmlchange INFO_DBUG=2
```

### Compile-Time
```bash
./xmlchange DEBUG=TRUE
./case.build --clean-all
./case.build
```
**Warning:** Runs significantly slower.

### Log Files
1. `cesm.log` - Component errors
2. `lnd.log` - CLM-specific issues
3. `CaseStatus` - Workflow failures

---

## Spinup

### AD Spinup (~200 years)
```bash
./xmlchange CLM_ACCELERATED_SPINUP=on
```

### Post-AD Spinup (several hundred years)
```bash
./xmlchange CLM_ACCELERATED_SPINUP=off
```

### Variables to Monitor
- TOTECOSYSC (total ecosystem carbon)
- TOTSOMC (soil organic matter carbon)
- TOTVEGC (vegetation carbon)
- GPP (gross primary production)
- TWS (total water storage)

---

## Submodule Management

### Commands
```bash
./bin/git-fleximod status    # Check status
./bin/git-fleximod update    # Update submodules
```

### When to Rerun
- After `git checkout` to different tag/branch
- After `git merge` from master
- Whenever `.gitmodules` changes

---

## Common Problems

### Model Crashes
| Problem | Solution |
|---------|----------|
| Divide by zero | Conditional check for 0 |
| Negative rounding | `max(foo, 0._r8)` or `truncate_small_values` |

### Parallelization (PEM/ERP failures)
| Problem | Solution |
|---------|----------|
| Missing broadcast | Add `shr_mpi_bcast` calls |
| Wrong index type | Match patch/column/gridcell |
| Uninitialized scalars | Initialize before loops |

### Threading
| Problem | Solution |
|---------|----------|
| Full array init | Use `foo(bounds%begc:bounds%endc)` |
| Missing indices | Use `variable_patch(p)` not `variable_patch` |

---

## PR Requirements

1. Scientific value OR bug fix/refactoring
2. No harm to other components
3. Code quality per coding guidelines
4. Pass test suite + add new tests
5. Documentation for new science

---

## Reference Links

- [CTSM Wiki](https://github.com/ESCOMP/ctsm/wiki)
- [CTSM Technical Note](https://escomp.github.io/CTSM/tech_note/index.html)
- [CLM Coding Conventions](https://wiki.ucar.edu/spaces/ccsm/pages/274337742)
- [DiscussCESM Forums](https://bb.cgd.ucar.edu/cesm/)
