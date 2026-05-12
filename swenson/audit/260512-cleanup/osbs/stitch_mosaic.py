#!/usr/bin/env python3
"""
Create production mosaics for OSBS from NEON tiles.

Merges DTM, slope, and aspect tiles for the production domain (R4C5-R12C14,
90 tiles, 9x10 km) into data/mosaics/production/. Skips products whose
mosaics already exist.

Products:
  - DTM (DP3.30024.001): bare earth elevation (m)
  - Slope (DP3.30025.001): terrain slope (degrees)
  - Aspect (DP3.30025.001): terrain aspect (degrees CW from north)

Usage:
    python scripts/osbs/stitch_mosaic.py
"""

from pathlib import Path

import rasterio
from rasterio.merge import merge

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "mosaics" / "production"

# Tile grid parameters
TILE_GRID_ORIGIN_EASTING = 394000
TILE_GRID_ORIGIN_NORTHING = 3292000
TILE_SIZE = 1000

# Production domain: R4C5-R12C14 (90 tiles, 0 nodata)
PROD_ROW_RANGE = range(4, 13)  # R4-R12
PROD_COL_RANGE = range(5, 15)  # C5-C14

# Products: (source_dir, filename_suffix, output_name)
PRODUCTS = [
    (DATA_DIR / "neon" / "dtm", "DTM", "dtm.tif"),
    (DATA_DIR / "neon" / "slope", "Slope", "slope.tif"),
    (DATA_DIR / "neon" / "aspect", "Aspect", "aspect.tif"),
]


def production_tile_paths(tile_dir: Path, suffix: str) -> list[Path]:
    """Return paths to production-domain tiles for a given product."""
    paths = []
    for row in PROD_ROW_RANGE:
        for col in PROD_COL_RANGE:
            easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
            northing = TILE_GRID_ORIGIN_NORTHING - row * TILE_SIZE
            path = tile_dir / f"NEON_D03_OSBS_DP3_{easting}_{northing}_{suffix}.tif"
            if path.exists():
                paths.append(path)
    return sorted(paths)


def mosaic_tiles(tile_paths: list[Path], output_path: Path) -> None:
    """Merge tiles into a single GeoTIFF with LZW compression."""
    datasets = [rasterio.open(str(p)) for p in tile_paths]
    mosaic_arr, mosaic_transform = merge(datasets)

    meta = datasets[0].meta.copy()
    meta.update(
        driver="GTiff",
        height=mosaic_arr.shape[1],
        width=mosaic_arr.shape[2],
        transform=mosaic_transform,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mosaic_arr)

    for ds in datasets:
        ds.close()

    print(
        f"  Written: {output_path.name} "
        f"({mosaic_arr.shape[2]}x{mosaic_arr.shape[1]} pixels)"
    )


def main():
    """Create all production mosaics."""
    n_expected = len(PROD_ROW_RANGE) * len(PROD_COL_RANGE)
    print("=" * 60)
    print(f"OSBS Production Mosaic Creation ({n_expected} tiles)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for tile_dir, suffix, output_name in PRODUCTS:
        output_path = OUTPUT_DIR / output_name
        print(f"\n{output_name}:")

        if output_path.exists():
            print(f"  Already exists: {output_path}")
            continue

        paths = production_tile_paths(tile_dir, suffix)
        print(f"  Found {len(paths)}/{n_expected} tiles")

        if len(paths) < n_expected:
            missing = n_expected - len(paths)
            print(f"  WARNING: {missing} tiles missing from {tile_dir.name}/")

        if not paths:
            print(f"  ERROR: No tiles found in {tile_dir}")
            continue

        mosaic_tiles(paths, output_path)

    print("\n" + "=" * 60)
    print("Done!")
    for _, _, output_name in PRODUCTS:
        path = OUTPUT_DIR / output_name
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path}")


if __name__ == "__main__":
    main()
