# NEON Data Products for OSBS

| Product | DP Number | Directory | Resolution | Tiles | Description |
|---------|-----------|-----------|------------|-------|-------------|
| DTM | DP3.30024.001 | dtm/ | 1m | 233 | Digital Terrain Model (bare earth) |
| Slope | DP3.30025.001 | slope/ | 1m | 231 | Slope in degrees (Horn 1981, 3x3 pre-filtered DTM) |
| Aspect | DP3.30025.001 | aspect/ | 1m | 231 | Aspect in degrees CW from grid north |

All products: OSBS 2023-05 collection, RELEASE-2026.

CRS: EPSG:32617 (UTM 17N)
Tile size: 1000m x 1000m
Tile naming: NEON_D03_OSBS_DP3_<easting>_<northing>_{DTM|Slope|Aspect}.tif
Coverage: 19x17 grid, 233 of 323 possible DTM tiles

Slope/aspect have 231 tiles (2 fewer than DTM). Missing tiles:
- 399000_3290000 (R14C5) — outside production domain
- 410000_3285000 (R9C16) — outside production domain

All 90 production tiles (R4C5-R12C14) have matching DTM, slope, and aspect data.

NEON slope/aspect uses Horn 1981 on a 3x3 averaged DTM (reduces TIN interpolation noise
on flat terrain). Computed with 20m tile-edge buffer — no boundary artifacts.
See ATBD: NEON.DOC.003791vB.
