# CTSM Fork Audit - 2025-01-11

## Summary

Audit of local modifications in ctsm5.3.059 to determine what needs to be carried forward to the new uf-ctsm5.3.085 fork.

---

## Modifications Found

### Category 1: NO LONGER NEEDED (Fixed in Upstream 5.3.085)

| File | Modification | Upstream Fix |
|------|--------------|--------------|
| `python/ctsm/site_and_regional/single_point_case.py` | Longitude TypeError fixes (`str(self.plon)` → `self.plon.get(360)`) | Fixed in commits 367317ecb, daa218c09, 3f5d157ac |

**Note:** The local Longitude fixes are identical to what was merged upstream. These can be dropped.

---

### Category 2: STILL NEEDED (Carry Forward to Fork)

#### A. HiPerGator-Specific Changes

| File | Modification | Reason |
|------|--------------|--------|
| `python/ctsm/site_and_regional/single_point_case.py` | `MPILIB=mpi-serial` → `MPILIB=openmpi` | HiPerGator uses openmpi |
| `python/ctsm/subset_data.py` | Default DATM type: `datm_crujra` → `datm_cruncep` | Local data availability preference |
| `tools/site_and_regional/default_data_2000.cfg` | Input path: `/glade/campaign/...` → `/blue/gerber/earth_models/inputdata` | HiPerGator paths |
| `tools/site_and_regional/default_data_2000.cfg` | Added `[datm_cruncep]` section | CRUNCEP forcing configuration |
| `tools/site_and_regional/default_data_1850.cfg` | Similar path changes | HiPerGator paths |

#### B. mksurfdata Build Fixes

| File | Modification | Reason |
|------|--------------|--------|
| `tools/mksurfdata_esmf/src/mksurfdata.F90` | Format specifiers: `I` → `I12`, `i` → `I6` | GCC Fortran requires explicit widths |
| `tools/mksurfdata_esmf/src/CMakeLists.txt` | `STATIC IMPORTED` → `SHARED IMPORTED` | Link to shared PIO libraries (.so) |
| `tools/mksurfdata_esmf/gen_mksurfdata_build` | Added `-fallow-argument-mismatch -fallow-invalid-boz -ffree-line-length-none` | GCC 14 compiler compatibility |

#### C. PIO Version

| File | Modification | Reason |
|------|--------------|--------|
| `.gitmodules` | ParallelIO: `pio2_6_4` → `pio2_6_6` | Use newer PIO version |

#### D. Research-Specific Changes

| File | Modification | Reason |
|------|--------------|--------|
| `bld/namelist_files/namelist_defaults_ctsm.xml` | Added `<spillheight>0.2</spillheight>` | Hillslope hydrology research |

---

### Category 3: NEW FILES TO ADD

| File | Purpose |
|------|---------|
| `tools/site_and_regional/osbs.cfg` | OSBS site configuration |
| `CLAUDE.md` (root) | Claude Code guidance for CTSM |
| `tools/site_and_regional/CLAUDE.md` | Claude guidance for subset tools |

---

### Category 4: ccs_config (Machine Configs)

These need to be committed to a forked ccs_config repository.

**Files:**
- `machines/hipergator/config_machines.xml`
- `machines/hipergator/config_batch.xml`
- `machines/hipergator/gnu_hipergator.cmake`

**Status:** Current configs look up-to-date for HiPerGator:
- GCC 14.2.0, OpenMPI 5.0.7, ESMF 8.8.1
- Shared PIO build at `/blue/gerber/earth_models/shared/parallelio/bld`
- SLURM batch config with gerber and gerber-b queues

**Recommendation:** These configs appear current. Verify module versions are still available on HiPerGator before committing.

---

## Action Items for Fork

### Must Apply to uf-ctsm5.3.085:

1. **single_point_case.py** - Only the `MPILIB=openmpi` change (Longitude fixes are in upstream)
2. **subset_data.py** - DATM type default change
3. **default_data_*.cfg** - HiPerGator input paths and CRUNCEP section
4. **mksurfdata.F90** - Fortran format specifier fixes
5. **CMakeLists.txt** - Shared PIO library linking
6. **gen_mksurfdata_build** - GCC 14 compiler flags
7. **.gitmodules** - PIO version (evaluate if still needed)
8. **namelist_defaults_ctsm.xml** - spillheight parameter (optional, research-specific)

### Must Apply to ccs_config Fork:

1. Create `machines/hipergator/` directory
2. Add config_machines.xml
3. Add config_batch.xml
4. Add gnu_hipergator.cmake

### Optional:

- Add CLAUDE.md files for Claude Code navigation
- Add osbs.cfg site configuration

---

## Verification Before Proceeding

- [ ] Confirm module versions still available: `module spider gcc/14.2.0 openmpi/5.0.7 esmf/8.8.1`
- [ ] Confirm shared PIO still works: `/blue/gerber/earth_models/shared/parallelio/bld`
- [ ] Confirm cprnc still works: `/blue/gerber/earth_models/shared/cprnc/bld/cprnc`
