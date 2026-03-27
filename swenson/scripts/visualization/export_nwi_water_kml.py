#!/usr/bin/env python3
"""
Export NWI open water polygons as KML for Google Earth.

Reads the NWI shapefile, filters to open water (Lacustrine + Palustrine
Unconsolidated Bottom), clips to the production domain, and exports as KML.
Uses LineString geometry (no polygon fill).

Usage:
    python export_nwi_water_kml.py
"""

from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import MultiPolygon, Polygon, box

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/

NWI_SHAPEFILE = (
    BASE_DIR / "data" / "HU8_03080103_Watershed" / "HU8_03080103_Wetlands.shp"
)
DTM_MOSAIC = BASE_DIR / "data" / "mosaics" / "production" / "dtm.tif"
OUTPUT_KML = BASE_DIR / "output" / "google-earth" / "osbs_nwi_water.kml"

# NWI Cowardin code prefixes for open water
OPEN_WATER_PREFIXES = ("L", "PUB")


def polygon_to_linestring_coords(geom: Polygon | MultiPolygon) -> list[str]:
    """Extract exterior ring coordinates as KML LineString coordinate strings."""
    rings = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        return rings

    for poly in polys:
        coords = " ".join(f"{x},{y},0" for x, y in poly.exterior.coords)
        rings.append(coords)
    return rings


def write_kml(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write GeoDataFrame as KML with LineString outlines (no fill)."""
    placemarks = []
    for _, row in gdf.iterrows():
        code = row["ATTRIBUTE"]
        wtype = row["WETLAND_TY"]
        rings = polygon_to_linestring_coords(row.geometry)
        for coords_str in rings:
            placemarks.append(
                f"""    <Placemark>
      <name>{code}</name>
      <description>{wtype}</description>
      <styleUrl>#waterStyle</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coords_str}</coordinates>
      </LineString>
    </Placemark>"""
            )

    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>NWI Open Water (L*, PUB*)</name>
    <description>NWI open water polygons clipped to OSBS production domain</description>
    <Style id="waterStyle">
      <LineStyle>
        <color>ff00ffff</color>
        <width>2</width>
      </LineStyle>
    </Style>
{chr(10).join(placemarks)}
  </Document>
</kml>
"""
    with open(output_path, "w") as f:
        f.write(kml_content)


def main():
    # --- Read and filter NWI ---
    print(f"Reading NWI shapefile: {NWI_SHAPEFILE.name}")
    gdf = gpd.read_file(NWI_SHAPEFILE)
    print(f"  Total features: {len(gdf):,}")

    mask = gdf["ATTRIBUTE"].str.startswith(OPEN_WATER_PREFIXES)
    water = gdf[mask].copy()
    print(f"  Open water features (L*, PUB*): {len(water):,}")

    # --- Reproject to UTM and clip (clip in UTM to match production perimeter) ---
    print("\nReprojecting to UTM...")
    water_utm = water.to_crs("EPSG:32617")

    with rasterio.open(DTM_MOSAIC) as src:
        b = src.bounds
    print("Clipping to production domain (UTM)...")
    clip_box = box(b.left, b.bottom, b.right, b.top)
    water_clipped = water_utm.clip(clip_box)
    print(f"  Features in domain: {len(water_clipped)}")

    if len(water_clipped) == 0:
        print("WARNING: No open water features found in production domain.")
        return

    # --- Summary ---
    codes = water_clipped["ATTRIBUTE"].value_counts()
    total_area_m2 = water_clipped.geometry.area.sum()
    print(f"  Total water area: {total_area_m2 / 1e4:.1f} ha")
    print("  Cowardin codes:")
    for code, count in codes.items():
        print(f"    {code}: {count}")

    # --- Reproject to WGS84 for KML export ---
    print("\nReprojecting to WGS84 for KML...")
    water_wgs84 = water_clipped.to_crs("EPSG:4326")

    # --- Export KML ---
    OUTPUT_KML.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting KML: {OUTPUT_KML.name}")
    write_kml(water_wgs84, OUTPUT_KML)
    print(f"Saved: {OUTPUT_KML}")


if __name__ == "__main__":
    main()
