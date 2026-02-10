# OSBS Tile Coverage Reference

Nodata analysis of all 233 NEON DTM tiles. Generated 2026-02-10 during Phase C follow-up.

## Grid Diagram

Nodata percentage per tile. `X` = 0% nodata (fully valid), `*` = <1% nodata, number = nodata %, `.` = no tile, `**` = ~100% nodata (tile exists but essentially empty).

```
         Cols:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
Row  0          .   .   .   .   .   .   .   .   .   .  91  63  66  67  90   .   .   .   .
Row  1          .   .   .   .   .   .   .   .   .   .  71   X   X   X  64   .   .   .   .
Row  2          .   .   .   .  **  **   .   .   .   .  67   X   X   X  64   .   .   .   .
Row  3          .   .   .   .  80   6  12   6  11  13   2   X   X   X  11  17  99   .   .
Row  4         94  69  71  72  56   X   X   X   X   X   X   X   X   X   X   *  97   .   .
Row  5         71   X   X   X   X   X   X   X   X   X   X   X   X   X   X   *  98   .   .
Row  6         72   X   X   X   X   X   X   X   X   X   X   X   X   X   X   *  99   .   .
Row  7         71   X   X   X   X   X   X   X   X   X   X   X   X   X   X   1  100   .   .
Row  8         69   X   X   X   X   X   X   X   X   X   X   X   X   X   X   2  99   .   .
Row  9         67   X   X   X   X   X   X   X   X   X   X   X   X   X   X   1  29  34  95
Row 10         66   X   X   X   X   X   X   X   X   X   X   X   X   X   X   X   X   *  95
Row 11         68   *   *   X   X   X   X   X   X   X   X   X   X   X   X   X   X   *  95
Row 12         99  99  97  95  70   X   X   X   X   X   X   X   X   X   X   X   X   X  96
Row 13          .   .   .   .  72   X   X   X   X   X   X   X  25  58  56  57  63  66  99
Row 14          .   .   .   .  73   X   X   X   X   X   X   X  41   .   .   .   .   .   .
Row 15          .   .   .   .  71   X   X   X   X   X   X   X  41   .   .   .   .   .   .
Row 16          .   .   .   .  83  31  25  26  30  22  25  27  58   .   .   .   .   .   .
```

## Summary

| Category | Count |
|----------|-------|
| Total tiles | 233 |
| Fully valid (0% nodata) | 153 |
| Nearly valid (<= 5% nodata) | 11 |
| Partial (> 5% nodata) | 69 |

## Contiguous Regions

### Largest fully contiguous rectangle (0% nodata)

**R4-R12, C5-C14** = 9 rows x 10 cols = 90 tiles (9 km x 10 km).

Verified: 0 nodata pixels out of 90,000,000. Elevation range: 23.53 - 55.50 m.

Mosaic pixel coordinates in `OSBS_interior.tif`: `elev[3000:12000, 4000:14000]`

This is the recommended region for any FFT or spectral analysis that requires contiguous data.

### Near-contiguous extensions

R5-R11, C1-C14 (14 cols x 7 rows) appears clean per individual tile checks, but the mosaic has ~4% nodata from merge boundary effects at the C1 edge. Not recommended without further verification.

### Single-tile candidates

For tests requiring a single contiguous tile with representative landscape:

| Tile | Location | Size | Nodata | Notes |
|------|----------|------|--------|-------|
| R6C10 | 404000E, 3286000N | 1000x1000 | 0% | Lake, swamp, and upland — representative |
| R8C8 | 402000E, 3284000N | 1000x1000 | 0% | Deep interior |
| R7C7 | 401000E, 3285000N | 1000x1000 | 0% | Deep interior |

## Full Nodata Table

All 80 tiles with nodata > 0%, sorted by row then column.

| Tile | UTM Easting | UTM Northing | Nodata % |
|------|-------------|--------------|----------|
| R0C10 | 404000 | 3292000 | 91 |
| R0C11 | 405000 | 3292000 | 63 |
| R0C12 | 406000 | 3292000 | 66 |
| R0C13 | 407000 | 3292000 | 67 |
| R0C14 | 408000 | 3292000 | 90 |
| R1C10 | 404000 | 3291000 | 71 |
| R1C14 | 408000 | 3291000 | 64 |
| R2C4 | 398000 | 3290000 | ~100 |
| R2C5 | 399000 | 3290000 | ~100 |
| R2C10 | 404000 | 3290000 | 67 |
| R2C14 | 408000 | 3290000 | 64 |
| R3C4 | 398000 | 3289000 | 80 |
| R3C5 | 399000 | 3289000 | 6 |
| R3C6 | 400000 | 3289000 | 12 |
| R3C7 | 401000 | 3289000 | 6 |
| R3C8 | 402000 | 3289000 | 11 |
| R3C9 | 403000 | 3289000 | 13 |
| R3C10 | 404000 | 3289000 | 2 |
| R3C14 | 408000 | 3289000 | 11 |
| R3C15 | 409000 | 3289000 | 17 |
| R3C16 | 410000 | 3289000 | 99 |
| R4C0 | 394000 | 3288000 | 94 |
| R4C1 | 395000 | 3288000 | 69 |
| R4C2 | 396000 | 3288000 | 71 |
| R4C3 | 397000 | 3288000 | 72 |
| R4C4 | 398000 | 3288000 | 56 |
| R4C15 | 409000 | 3288000 | <1 |
| R4C16 | 410000 | 3288000 | 97 |
| R5C0 | 394000 | 3287000 | 71 |
| R5C15 | 409000 | 3287000 | <1 |
| R5C16 | 410000 | 3287000 | 98 |
| R6C0 | 394000 | 3286000 | 72 |
| R6C15 | 409000 | 3286000 | <1 |
| R6C16 | 410000 | 3286000 | 99 |
| R7C0 | 394000 | 3285000 | 71 |
| R7C15 | 409000 | 3285000 | 1 |
| R7C16 | 410000 | 3285000 | 100 |
| R8C0 | 394000 | 3284000 | 69 |
| R8C15 | 409000 | 3284000 | 2 |
| R8C16 | 410000 | 3284000 | 99 |
| R9C0 | 394000 | 3283000 | 67 |
| R9C15 | 409000 | 3283000 | 1 |
| R9C16 | 410000 | 3283000 | 29 |
| R9C17 | 411000 | 3283000 | 34 |
| R9C18 | 412000 | 3283000 | 95 |
| R10C0 | 394000 | 3282000 | 66 |
| R10C17 | 411000 | 3282000 | <1 |
| R10C18 | 412000 | 3282000 | 95 |
| R11C0 | 394000 | 3281000 | 68 |
| R11C1 | 395000 | 3281000 | <1 |
| R11C2 | 396000 | 3281000 | <1 |
| R11C17 | 411000 | 3281000 | <1 |
| R11C18 | 412000 | 3281000 | 95 |
| R12C0 | 394000 | 3280000 | 99 |
| R12C1 | 395000 | 3280000 | 99 |
| R12C2 | 396000 | 3280000 | 97 |
| R12C3 | 397000 | 3280000 | 95 |
| R12C4 | 398000 | 3280000 | 70 |
| R12C18 | 412000 | 3280000 | 96 |
| R13C4 | 398000 | 3279000 | 72 |
| R13C12 | 406000 | 3279000 | 25 |
| R13C13 | 407000 | 3279000 | 58 |
| R13C14 | 408000 | 3279000 | 56 |
| R13C15 | 409000 | 3279000 | 57 |
| R13C16 | 410000 | 3279000 | 63 |
| R13C17 | 411000 | 3279000 | 66 |
| R13C18 | 412000 | 3279000 | 99 |
| R14C4 | 398000 | 3278000 | 73 |
| R14C12 | 406000 | 3278000 | 41 |
| R15C4 | 398000 | 3277000 | 71 |
| R15C12 | 406000 | 3277000 | 41 |
| R16C4 | 398000 | 3276000 | 83 |
| R16C5 | 399000 | 3276000 | 31 |
| R16C6 | 400000 | 3276000 | 25 |
| R16C7 | 401000 | 3276000 | 26 |
| R16C8 | 402000 | 3276000 | 30 |
| R16C9 | 403000 | 3276000 | 22 |
| R16C10 | 404000 | 3276000 | 25 |
| R16C11 | 405000 | 3276000 | 27 |
| R16C12 | 406000 | 3276000 | 58 |
