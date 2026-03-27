#!/usr/bin/env python3
"""
Generate NWI open water mask raster for the production domain.

Reads the NWI shapefile, filters to open water (Lacustrine + Palustrine
Unconsolidated Bottom), reprojects to UTM, and rasterizes onto the
production DTM grid. Output is a single-band uint8 GeoTIFF (1 = water,
0 = land) matching the DTM resolution and extent.

Usage:
    python generate_water_mask.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/

NWI_SHAPEFILE = (
    BASE_DIR / "data" / "HU8_03080103_Watershed" / "HU8_03080103_Wetlands.shp"
)
DTM_MOSAIC = BASE_DIR / "data" / "mosaics" / "production" / "dtm.tif"
OUTPUT_MASK = BASE_DIR / "data" / "mosaics" / "production" / "water_mask.tif"

# NWI Cowardin code prefixes for open water
OPEN_WATER_PREFIXES = ("L", "PUB")


def main():
    # --- Read and filter NWI ---
    print(f"Reading NWI shapefile: {NWI_SHAPEFILE.name}")
    gdf = gpd.read_file(NWI_SHAPEFILE)
    print(f"  Total features: {len(gdf):,}")

    mask = gdf["ATTRIBUTE"].str.startswith(OPEN_WATER_PREFIXES)
    water = gdf[mask].copy()
    print(f"  Open water features (L*, PUB*): {len(water):,}")

    # --- Reproject to UTM (all rasterization in UTM, no WGS84 clipping) ---
    print("\nReprojecting to EPSG:32617...")
    water_utm = water.to_crs("EPSG:32617")

    # --- Read DTM grid parameters ---
    print(f"Reading DTM grid: {DTM_MOSAIC.name}")
    with rasterio.open(DTM_MOSAIC) as src:
        dtm_transform = src.transform
        dtm_crs = src.crs
        dtm_shape = (src.height, src.width)
    print(f"  Shape: {dtm_shape[0]} x {dtm_shape[1]}")
    print(f"  CRS: {dtm_crs}")

    # --- Rasterize polygons onto DTM grid ---
    print("\nRasterizing water polygons...")
    shapes = [(geom, 1) for geom in water_utm.geometry if geom is not None]
    water_mask = rasterize(
        shapes,
        out_shape=dtm_shape,
        transform=dtm_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    # --- Summary ---
    n_water = int(np.sum(water_mask == 1))
    n_total = water_mask.size
    pct = 100.0 * n_water / n_total
    print(f"\n  Water pixels: {n_water:,} / {n_total:,} ({pct:.1f}%)")
    print(f"  Water area: {n_water / 1e4:.1f} ha ({n_water / 1e6:.3f} km²)")

    # --- Write mask ---
    print(f"\nWriting: {OUTPUT_MASK.name}")
    with rasterio.open(
        OUTPUT_MASK,
        "w",
        driver="GTiff",
        height=dtm_shape[0],
        width=dtm_shape[1],
        count=1,
        dtype=np.uint8,
        crs=dtm_crs,
        transform=dtm_transform,
        compress="lzw",
    ) as dst:
        dst.write(water_mask, 1)

    print(f"Saved: {OUTPUT_MASK}")


if __name__ == "__main__":
    main()
