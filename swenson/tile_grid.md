# OSBS Tile Grid Reference

Quick reference for NEON LIDAR tile selection.

## Grid Dimensions

- **17 rows × 19 columns** (potential grid)
- **233 tiles** of 323 possible (72% coverage)
- Each tile: 1 km × 1 km (1000 × 1000 pixels at 1m resolution)

## Reference Format

| Format | Example | Description |
|--------|---------|-------------|
| Single tile | `R5C7` | Row 5, Column 7 |
| Range | `R4-12,C4-16` | Rows 4-12, Columns 4-16 |
| List | `R5C7, R5C8, R6C7` | Specific tiles |
| Exclude | `R4-12,C4-16 except R5C7` | Range with exclusions |

## Tile Grid Map

```
        Columns:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
                 (W) ──────────────────────────────────────────────> (E)
                 394k                                              412k
      Row  0 (N) .  .  .  .  .  .  .  .  .  .  X  X  X  X  X  .  .  .  .   3292k
      Row  1     .  .  .  .  .  .  .  .  .  .  X  X  X  X  X  .  .  .  .   3291k
      Row  2     .  .  .  .  X  X  .  .  .  .  X  X  X  X  X  .  .  .  .   3290k
      Row  3     .  .  .  .  X  X  X  X  X  X  X  X  X  X  X  X  X  .  .   3289k
      Row  4     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  .  .   3288k
      Row  5     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  .  .   3287k
      Row  6     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  .  .   3286k
      Row  7     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  .  .   3285k
      Row  8     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  .  .   3284k
      Row  9     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X   3283k
      Row 10     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X   3282k
      Row 11     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X   3281k
      Row 12     X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X   3280k
      Row 13     .  .  .  .  X  X  X  X  X  X  X  X  X  X  X  X  X  X  X   3279k
      Row 14     .  .  .  .  X  X  X  X  X  X  X  X  X  .  .  .  .  .  .   3278k
      Row 15     .  .  .  .  X  X  X  X  X  X  X  X  X  .  .  .  .  .  .   3277k
      Row 16 (S) .  .  .  .  X  X  X  X  X  X  X  X  X  .  .  .  .  .  .   3276k

Legend: X = tile exists, . = no tile
```

## Coordinate Mapping

### Columns (West to East)

| Col | UTM Easting | Col | UTM Easting |
|-----|-------------|-----|-------------|
| C0 | 394000 | C10 | 404000 |
| C1 | 395000 | C11 | 405000 |
| C2 | 396000 | C12 | 406000 |
| C3 | 397000 | C13 | 407000 |
| C4 | 398000 | C14 | 408000 |
| C5 | 399000 | C15 | 409000 |
| C6 | 400000 | C16 | 410000 |
| C7 | 401000 | C17 | 411000 |
| C8 | 402000 | C18 | 412000 |
| C9 | 403000 | | |

### Rows (North to South)

| Row | UTM Northing | Row | UTM Northing |
|-----|--------------|-----|--------------|
| R0 | 3292000 | R9 | 3283000 |
| R1 | 3291000 | R10 | 3282000 |
| R2 | 3290000 | R11 | 3281000 |
| R3 | 3289000 | R12 | 3280000 |
| R4 | 3288000 | R13 | 3279000 |
| R5 | 3287000 | R14 | 3278000 |
| R6 | 3286000 | R15 | 3277000 |
| R7 | 3285000 | R16 | 3276000 |
| R8 | 3284000 | | |

**CRS:** EPSG:32617 (UTM Zone 17N)

## Google Earth Reference

**File:** `osbs_tile_grid.kml` (in swenson root)

- Green polygons = tile exists
- Red polygons = no tile
- Labels show R#C# reference

## Corner Tiles

| Corner | Tile | Notes |
|--------|------|-------|
| Northwest-most | R2C4 | Isolated pair with R2C5 |
| Northeast-most | R0C14 | |
| Southwest-most | R16C4 | |
| Southeast-most | R13C18 | |

## Common Selections

| Selection | Tiles | Description |
|-----------|-------|-------------|
| Interior only | `R4-12,C4-16` | Fully surrounded tiles, no edge effects |
| Full coverage | All 233 tiles | Everything available |
| Dense core | `R4-12,C4-14` | Best continuous coverage |
