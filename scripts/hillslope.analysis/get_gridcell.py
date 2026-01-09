#!/usr/bin/env python3

# find lat/lon indices for a given coordinate on a regular grid
# CORRECTED: Actual grid spacing for 0.9x1.25 degree resolution
# grid has 192 lat cells (indices 0-191) spanning -90° to +90°
# 191 intervals, so spacing = 180° / 191 = 0.942408°
LAT_N_CELLS = 192
LON_N_CELLS = 288
LAT_SPACING = 180.0 / (LAT_N_CELLS - 1)  # 0.942408°
LON_SPACING = 360.0 / (LON_N_CELLS - 1)  # 1.254355°

# OSBS site
TARGET_LAT = 29.689
TARGET_LON = 278.007

# Grid starting coordinates
START_LAT = -90
START_LON = 0

# Calculate indices using nearest neighbor (round to find closest cell center)
LAT_IDX = int(round((TARGET_LAT - START_LAT) / LAT_SPACING))
LON_IDX = int(round((TARGET_LON - START_LON) / LON_SPACING))

print("Grid parameters:")
print(f"  Latitude: {LAT_N_CELLS} cells, spacing = {LAT_SPACING:.6f}")
print(f"  Longitude: {LON_N_CELLS} cells, spacing = {LON_SPACING:.6f}")
print("")
print("Calculated indices:")
print(f"  lat_idx={LAT_IDX}, lon_idx={LON_IDX}")
print("")
print("To verify:")
print(f"  ncks -H -v LATIXY -d lsmlat,{LAT_IDX} -d lsmlon,{LON_IDX} hillslopes_0.9x1.25_c240416.nc")
print(f"  ncks -H -v LONGXY -d lsmlat,{LAT_IDX} -d lsmlon,{LON_IDX} hillslopes_0.9x1.25_c240416.nc")
