# CTSM Modification Analysis - 2025-01-12

Analysis of local CTSM modifications to determine root causes and upstream contribution opportunities.

---

## 1. CMakeLists.txt: STATIC vs SHARED Library Declaration

**File:** `tools/mksurfdata_esmf/src/CMakeLists.txt`

**The Bug:**
```cmake
add_library(pioc STATIC IMPORTED)  # declares STATIC
set_property(TARGET pioc PROPERTY IMPORTED_LOCATION $ENV{PIO}/lib/libpioc.so)  # points to .so (SHARED)
```

This is an internal inconsistency in the upstream code that still exists in 5.3.085. You cannot declare a library as `STATIC IMPORTED` and then point it to a `.so` (shared library) file.

**Our Fix:** Change `STATIC IMPORTED` to `SHARED IMPORTED`

**Why does it work at NCAR?**
- NCAR's systems likely have PIO built as static libraries (`.a` files) at the same path
- Or a CMake version that's more lenient about this mismatch
- Or they use `ESMF_PIO=internal` which builds PIO differently

**Evidence:** According to [CESM forums](https://bb.cgd.ucar.edu/cesm/threads/the-error-about-pio.9238/), users need to manually edit CMakeCache.txt to change PIO library references, suggesting this is a known pain point.

**Contribution opportunity:** Yes - genuine bug affecting anyone using shared PIO libraries.

---

## 2. mksurfdata.F90 Format Specifiers

**File:** `tools/mksurfdata_esmf/src/mksurfdata.F90`

**The Bug:**
```fortran
write(ndiag,'(2(a,I))') ' npes = ', npes, ' grid size = ', grid_size
```

The `I` format descriptor without a width is a legacy Fortran extension.

According to [GCC documentation](https://gcc.gnu.org/onlinedocs/gfortran/Default-widths-for-F_002fG_002fI-format-descriptors.html):
> "To support legacy codes, GNU Fortran allows width to be omitted from format specifications if and only if `-fdec-format-defaults` is given"

**Our Fix:** Add explicit widths: `I` → `I12`, `i` → `I6`

**Why does it work at NCAR?**
- They likely use Intel compilers (ifort) which accept this
- Or older GCC versions with different defaults
- Or they add `-fdec-format-defaults` somewhere in their build chain

**Contribution opportunity:** Yes - portability fix for modern GCC compilers.

---

## 3. gen_mksurfdata_build GCC 14 Flags

**File:** `tools/mksurfdata_esmf/gen_mksurfdata_build`

**Our Addition:**
```bash
-DCMAKE_Fortran_FLAGS=" -fallow-argument-mismatch -fallow-invalid-boz -ffree-line-length-none"
```

**Purpose:**
- `-fallow-argument-mismatch`: Allow type mismatches in procedure calls (legacy Fortran)
- `-fallow-invalid-boz`: Allow invalid BOZ literal constants
- `-ffree-line-length-none`: No limit on line length

**Assessment:** These are workarounds for legacy Fortran code issues. GCC 10+ became stricter about these.

**Contribution opportunity:** Maybe - could be framed as "GCC 14 compatibility" but they're masking underlying code issues.

---

## 4. All Modified Files Summary

| File | Change | Upstream Fix? | Keep? | Contribution? |
|------|--------|---------------|-------|---------------|
| **CMakeLists.txt** | `STATIC` → `SHARED` | No | Yes | **Yes** - genuine bug |
| **mksurfdata.F90** | `I` → `I12` format specs | No | Yes | **Yes** - portability fix |
| **gen_mksurfdata_build** | GCC 14 flags | No | Yes | Maybe - GCC workarounds |
| **single_point_case.py** | Longitude `.get(360)` fixes | **Yes (5.3.085)** | **No** | Already upstream |
| **single_point_case.py** | `MPILIB=openmpi` | No | Yes | No - HiPerGator-specific |
| **subset_data.py** | Default `datm_crujra` → `datm_cruncep` | No | Optional | No - preference |
| **default_data_*.cfg** | HiPerGator paths + CRUNCEP section | No | Yes | No - site-specific |
| **.gitmodules** | PIO `pio2_6_4` → `pio2_6_6` | No | Evaluate | Probably not |
| **namelist_defaults_ctsm.xml** | `spillheight=0.2` | No | Yes | No - research-specific |
| **namelist_definition_ctsm.xml** | spillheight definition | No | Yes | No - research-specific |

---

## 5. Contribution Candidates

### Strong Candidates for Upstream PR

1. **CMakeLists.txt STATIC/SHARED fix**
   - Clear bug affecting anyone using shared PIO
   - Simple one-line fix
   - Easy to justify

2. **mksurfdata.F90 format specifiers**
   - Portability issue with modern GCC
   - Affects anyone building with GCC 10+
   - Standard-compliant fix

### Possible Candidates

3. **gen_mksurfdata_build GCC flags**
   - Workarounds for legacy Fortran issues
   - Could be framed as "GCC 14 compatibility"
   - Upstream might prefer fixing underlying code instead

---

## 6. Sources

- [CESM Forums: PIO Error](https://bb.cgd.ucar.edu/cesm/threads/the-error-about-pio.9238/)
- [GCC: Default widths for F/G/I format descriptors](https://gcc.gnu.org/onlinedocs/gfortran/Default-widths-for-F_002fG_002fI-format-descriptors.html)
- [GCC: Fortran Dialect Options](https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html)
- [MSU HPCC: CTSM Installation Lab Notebook](https://docs.icer.msu.edu/2024-06-13_LabNotebook_CTSM_Install/)
