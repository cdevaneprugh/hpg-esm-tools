# Spillheight Implementation for Hillslope Hydrology

**Status:** In Development
**Last Updated:** 2025-12-15
**Location:** `osbs2.spillheight/`

## Overview

The spillheight modification creates a "wetland bank height" effect for hillslope hydrology simulations. Water must accumulate to a specified threshold height before it can drain to the stream channel. This is intended to simulate wetland/riparian zone water retention.

## Concept

```
                    UPLAND
                      |
                      v
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  <- spillheight threshold
    |  accumulated surface water   |
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    |        LOWLAND COLUMN        |
    --------------------------------
              STREAM
```

By subtracting `spillheight` from hillslope column elevations, the effective elevation relative to the stream is lowered, reducing the hydraulic head gradient and thus reducing subsurface drainage to the stream.

## Modified Source Files

All modified files are in `osbs2.spillheight/` for use as SourceMods.

### 1. HillslopeHydrologyMod.F90 (Primary Implementation)

**Changes:**
- Line 55: Added module variable `spillheight = 0.2_r8` (default value)
- Lines 89-92: Added `spillheight` to `hillslope_properties_inparm` namelist
- Line 140: MPI broadcast of spillheight
- Line 147: Log output of spillheight value
- **Line 363:** Key change - subtracts spillheight from elevation:
  ```fortran
  hill_elev(l,:) = fhillslope_in(g,:) - spillheight
  ```

**Effect:** Reduces hydraulic head gradient for subsurface flow to stream.

### 2. SurfaceWaterMod.F90 (Surface Runoff Suppression)

**Changes (lines 498-502):**
```fortran
!sg 2025-11-25 disallow surface runoff (accumulate)
if ( qflx_h2osfc_surf(c) < 1.0e-8) then
   qflx_h2osfc_surf(c) = 0._r8
end if
```

**Effect:** Prevents small surface runoff values, encouraging water accumulation.

### 3. SaturatedExcessRunoffMod.F90 (Saturated Excess Disabled)

**Changes (lines 294-295):**
```fortran
!sg 2025-11-26, disallow surface runoff, assume it rolls down the hill until it finds a pond
qflx_sat_excess_surf(c) = 0._r8
```

**Effect:** Disables saturated excess surface runoff entirely.

### 4. InfiltrationExcessRunoffMod.F90 (Infiltration Excess Disabled)

**Changes (lines 189-190):**
```fortran
!sg: 2025-11-26, allow for max infiltration anyway
this%qinmax_method = QINMAX_METHOD_NONE
```

**Effect:** Sets unlimited infiltration rate (no infiltration excess runoff).

### 5. SoilHydrologyMod.F90

No modifications (included for reference).

## Namelist Configuration

### Definition (`bld/namelist_files/namelist_definition_ctsm.xml`)

```xml
<entry id="spillheight" type="real" category="physics"
        group="hillslope_properties_inparm">
Height threshold (m) for water to accumulate before draining to stream.
Subtracts from hillslope_elevation to create a bank height effect.
Default: 0.2 m
</entry>
```

### Default (`bld/namelist_files/namelist_defaults_ctsm.xml`)

```xml
<spillheight>0.2</spillheight>
```

### Usage in user_nl_clm

```fortran
spillheight = 0.5   ! Override default (meters)
```

## How It Works

### Subsurface Flow to Stream

The elevation change affects head gradient calculations in `SoilHydrologyMod.F90`:

```fortran
! PerchedLateralFlow (lines 1833-1836)
head_gradient = (col%hill_elev(c)-zwt_perched(c)) &
     - max(min((stream_water_depth - stream_channel_depth),0._r8), &
     (col%hill_elev(c)-frost_table(c)))

! SubsurfaceLateralFlow (lines 2276-2278)
head_gradient = (col%hill_elev(c)-zwt(c)) &
     - min((stream_water_depth - stream_channel_depth),0._r8)
```

Lowering `hill_elev` reduces these head gradients, reducing flow to stream.

### Transmissivity to Stream

```fortran
! Lines 1887, 2325
if ((col%hill_elev(c_src)-z(c_src,k)) > (-stream_channel_depth)) then
   transmis = transmis + ...
```

Lowering elevation means fewer soil layers qualify as "above" stream channel bottom.

### Upland-to-Upland Flow

```fortran
head_gradient = (col%hill_elev(c)-zwt(c)) - (col%hill_elev(col%cold(c))-zwt(col%cold(c)))
```

**No effect** - uniform spillheight subtraction cancels out between columns.

## Known Limitations

1. **Uniform elevation lowering:** All columns are lowered equally, which may affect meteorological downscaling calculations unnecessarily.

2. **Surface runoff completely disabled:** Current implementation sets surface runoff to 0 unconditionally, rather than allowing overflow when water exceeds spillheight.

3. **No coordination with h2osfc_thresh:** The existing `h2osfc_thresh` mechanism could be used for natural spillover behavior.

## Recommendations for Future Development

### Option 1: Only Modify Lowland Column Elevation

Instead of lowering all columns, only raise the lowland column (stream interface) or treat spillheight as a bank height:

```fortran
! In InitHillslope, only modify lowland columns:
if (col_dndx(l,ci) <= -999) then  ! lowland column
   hill_elev(l,ci) = fhillslope_in(g,ci) + spillheight
else
   hill_elev(l,ci) = fhillslope_in(g,ci)
endif
```

### Option 2: Use h2osfc_thresh for Surface Water Spillover

Set `h2osfc_thresh = spillheight * 1000` (convert m to mm) for lowland columns in `SoilHydrologyInitTimeConstMod.F90`:

```fortran
if (col%is_hillslope_column(c) .and. col%cold(c) == ispval) then
   ! Lowland column - use spillheight as threshold
   soilhydrology_inst%h2osfc_thresh_col(c) = spillheight * 1000._r8
endif
```

This would allow natural spillover behavior using existing model infrastructure.

### Option 3: Conditional Surface Runoff

Instead of setting `qflx_sat_excess_surf = 0`, make it conditional:

```fortran
! In SaturatedExcessRunoffMod.F90
if (col%is_hillslope_column(c) .and. h2osfc(c) < spillheight * 1000._r8) then
   qflx_sat_excess_surf(c) = 0._r8
endif
```

## Testing Checklist

- [ ] Build case with SourceMods from `osbs2.spillheight/`
- [ ] Verify spillheight appears in CLM log output
- [ ] Compare h2osfc accumulation with/without spillheight
- [ ] Check stream water volume behavior
- [ ] Verify subsurface drainage reduction
- [ ] Test different spillheight values (0.1, 0.2, 0.5 m)

## Related Files

- **SourceMods:** `osbs2.spillheight/`
- **user_nl_clm template:** `osbs2.spillheight/user_nl_clm`
- **Namelist definitions:** `ctsm5.3/bld/namelist_files/namelist_definition_ctsm.xml`
- **Namelist defaults:** `ctsm5.3/bld/namelist_files/namelist_defaults_ctsm.xml`

## References

- CTSM hillslope hydrology: `src/biogeophys/HillslopeHydrologyMod.F90`
- Surface water module: `src/biogeophys/SurfaceWaterMod.F90`
- Soil hydrology: `src/biogeophys/SoilHydrologyMod.F90`
- h2osfc_thresh initialization: `src/biogeophys/SoilHydrologyInitTimeConstMod.F90`
