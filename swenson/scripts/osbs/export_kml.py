#!/usr/bin/env python3
"""
Export OSBS tile grid as KML for viewing in Google Earth.

Creates a KML file with:
- Polygon for each tile (1km x 1km)
- Label showing tile reference (e.g., "R5C7")
- Different styling for existing vs missing tiles

Output: output/full_mosaic/osbs_tile_grid.kml
"""

import re
from pathlib import Path

from pyproj import Transformer

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/
DATA_DIR = BASE_DIR / "data"

DTM_DIR = DATA_DIR / "tiles"
OUTPUT_KML = BASE_DIR / "osbs_tile_grid.kml"  # Root for easy access

# Tile size in meters
TILE_SIZE = 1000


def parse_tile_filenames(tile_dir: Path) -> set[tuple[int, int]]:
    """
    Scan tile directory and extract UTM coordinates from filenames.

    Filename format: NEON_D03_OSBS_DP3_{EASTING}_{NORTHING}_DTM.tif

    Returns:
        Set of (easting, northing) tuples for existing tiles
    """
    pattern = re.compile(r"NEON_D03_OSBS_DP3_(\d+)_(\d+)_DTM\.tif")
    tiles = set()

    for f in tile_dir.glob("*.tif"):
        match = pattern.match(f.name)
        if match:
            easting = int(match.group(1))
            northing = int(match.group(2))
            tiles.add((easting, northing))

    return tiles


def get_tile_corners_wgs84(
    easting: int, northing: int, transformer: Transformer
) -> list[tuple[float, float]]:
    """
    Get tile corner coordinates in WGS84 (lon, lat).

    Tile corners are ordered counter-clockwise starting from SW corner,
    which is what KML expects for polygon coordinates.

    Args:
        easting: UTM easting of tile SW corner
        northing: UTM northing of tile SW corner
        transformer: pyproj Transformer from UTM to WGS84

    Returns:
        List of (lon, lat) tuples for corners [SW, SE, NE, NW, SW]
    """
    # Corner coordinates in UTM (SW, SE, NE, NW, back to SW to close polygon)
    corners_utm = [
        (easting, northing),  # SW
        (easting + TILE_SIZE, northing),  # SE
        (easting + TILE_SIZE, northing + TILE_SIZE),  # NE
        (easting, northing + TILE_SIZE),  # NW
        (easting, northing),  # SW (close polygon)
    ]

    # Transform to WGS84
    corners_wgs84 = []
    for e, n in corners_utm:
        lon, lat = transformer.transform(e, n)
        corners_wgs84.append((lon, lat))

    return corners_wgs84


def get_tile_center_wgs84(
    easting: int, northing: int, transformer: Transformer
) -> tuple[float, float]:
    """
    Get tile center coordinates in WGS84 (lon, lat).

    Args:
        easting: UTM easting of tile SW corner
        northing: UTM northing of tile SW corner
        transformer: pyproj Transformer from UTM to WGS84

    Returns:
        (lon, lat) tuple for tile center
    """
    center_e = easting + TILE_SIZE / 2
    center_n = northing + TILE_SIZE / 2
    lon, lat = transformer.transform(center_e, center_n)
    return lon, lat


def generate_kml_header() -> str:
    """Generate KML document header with styles."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>OSBS Tile Grid</name>
    <description>1km x 1km tile grid for NEON OSBS DTM data. Green = tile exists, Red = tile missing.</description>

    <!-- Style for existing tiles (green polygon) -->
    <Style id="tileExists">
      <PolyStyle>
        <color>4000ff00</color>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>ff00ff00</color>
        <width>2</width>
      </LineStyle>
    </Style>

    <!-- Style for missing tiles (red polygon) -->
    <Style id="tileMissing">
      <PolyStyle>
        <color>400000ff</color>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>ff0000ff</color>
        <width>1</width>
      </LineStyle>
    </Style>

    <!-- Style for existing tile labels (green text) -->
    <Style id="labelExists">
      <IconStyle>
        <scale>0</scale>
      </IconStyle>
      <LabelStyle>
        <color>ff00ff00</color>
        <scale>0.8</scale>
      </LabelStyle>
    </Style>

    <!-- Style for missing tile labels (red text) -->
    <Style id="labelMissing">
      <IconStyle>
        <scale>0</scale>
      </IconStyle>
      <LabelStyle>
        <color>ff0000ff</color>
        <scale>0.7</scale>
      </LabelStyle>
    </Style>

"""


def generate_polygon_placemark(
    label: str, corners: list[tuple[float, float]], exists: bool
) -> str:
    """
    Generate KML Placemark polygon for a single tile.

    Args:
        label: Tile label (e.g., "R5C7")
        corners: List of (lon, lat) tuples for corners
        exists: Whether tile data exists

    Returns:
        KML Placemark XML string
    """
    style = "tileExists" if exists else "tileMissing"

    # Format coordinates as "lon,lat,0" separated by spaces
    coords_str = " ".join(f"{lon},{lat},0" for lon, lat in corners)

    return f"""      <Placemark>
        <name>{label}</name>
        <styleUrl>#{style}</styleUrl>
        <Polygon>
          <tessellate>1</tessellate>
          <outerBoundaryIs>
            <LinearRing>
              <coordinates>{coords_str}</coordinates>
            </LinearRing>
          </outerBoundaryIs>
        </Polygon>
      </Placemark>
"""


def generate_label_placemark(
    label: str, center: tuple[float, float], exists: bool
) -> str:
    """
    Generate KML Placemark point label for a single tile.

    Args:
        label: Tile label (e.g., "R5C7")
        center: (lon, lat) tuple for tile center
        exists: Whether tile data exists

    Returns:
        KML Placemark XML string
    """
    style = "labelExists" if exists else "labelMissing"
    lon, lat = center

    return f"""      <Placemark>
        <name>{label}</name>
        <styleUrl>#{style}</styleUrl>
        <Point>
          <coordinates>{lon},{lat},0</coordinates>
        </Point>
      </Placemark>
"""


def generate_kml_footer() -> str:
    """Generate KML document footer."""
    return """  </Document>
</kml>
"""


def main():
    """Generate KML file for OSBS tile grid."""
    print("Scanning tile directory...")
    existing_tiles = parse_tile_filenames(DTM_DIR)
    print(f"Found {len(existing_tiles)} existing tiles")

    if not existing_tiles:
        print("ERROR: No tiles found in", DTM_DIR)
        return

    # Get grid extents
    eastings = sorted(set(e for e, n in existing_tiles))
    northings = sorted(set(n for e, n in existing_tiles), reverse=True)  # N to S

    print(f"Grid extent: {len(eastings)} columns x {len(northings)} rows")
    print(f"Easting range: {min(eastings)} - {max(eastings)}")
    print(f"Northing range: {min(northings)} - {max(northings)}")

    # Setup coordinate transformer (UTM Zone 17N to WGS84)
    transformer = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)

    # Generate KML
    print("Generating KML...")
    kml_content = generate_kml_header()

    # Collect placemarks for polygons and labels separately
    polygon_placemarks = []
    label_placemarks = []

    tile_count = 0
    existing_count = 0
    missing_count = 0

    for row, n in enumerate(northings):
        for col, e in enumerate(eastings):
            # Get tile corners and center in WGS84
            corners = get_tile_corners_wgs84(e, n, transformer)
            center = get_tile_center_wgs84(e, n, transformer)

            # Check if tile exists
            exists = (e, n) in existing_tiles
            label = f"R{row}C{col}"

            polygon_placemarks.append(
                generate_polygon_placemark(label, corners, exists)
            )
            label_placemarks.append(generate_label_placemark(label, center, exists))

            tile_count += 1
            if exists:
                existing_count += 1
            else:
                missing_count += 1

    # Add polygons folder
    kml_content += """    <Folder>
      <name>Tile Outlines</name>
      <description>Polygon outlines for each tile</description>
"""
    kml_content += "".join(polygon_placemarks)
    kml_content += """    </Folder>
"""

    # Add labels folder
    kml_content += """    <Folder>
      <name>Tile Labels</name>
      <description>Row/Column labels for each tile</description>
"""
    kml_content += "".join(label_placemarks)
    kml_content += """    </Folder>
"""

    kml_content += generate_kml_footer()

    # Save KML
    print(f"Writing KML to {OUTPUT_KML}...")
    with open(OUTPUT_KML, "w") as f:
        f.write(kml_content)

    print("\nSummary:")
    print(f"  Total tiles in grid: {tile_count}")
    print(f"  Existing tiles: {existing_count} (green)")
    print(f"  Missing tiles: {missing_count} (red)")
    print(f"\nOpen {OUTPUT_KML} in Google Earth to view.")


if __name__ == "__main__":
    main()
