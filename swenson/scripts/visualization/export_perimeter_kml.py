#!/usr/bin/env python3
"""
Export the outer perimeter of a tile selection as KML with red outline.

Creates a KML file with:
- Single polygon showing the outer perimeter of all selected tiles
- Red outline styling (no fill, thick red line)
- Can be overlaid on Google Earth with the existing tile grid KML

Usage:
    python export_perimeter_kml.py [output_file]

The tile selection is configured via TILE_RANGES constant (same format as run_pipeline.py).
"""

import re
import sys
from pathlib import Path

from pyproj import Transformer

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/

# Default output
DEFAULT_OUTPUT = BASE_DIR / "output" / "google-earth" / "osbs_trimmed_perimeter.kml"

# Tile grid parameters (from tile_grid.md)
TILE_GRID_ORIGIN_EASTING = 394000  # UTM easting for column 0
TILE_GRID_ORIGIN_NORTHING = 3292000  # UTM northing for row 0
TILE_SIZE = 1000  # meters per tile

# Tile selection (same format as run_pipeline.py INTERIOR_TILE_RANGES)
TILE_RANGES = [
    "R4C10-R4C12",  # 3 tiles
    "R5C9-R5C12",  # 4 tiles
    "R6C9-R6C14",  # 6 tiles
    "R7C7-R7C14",  # 8 tiles
    "R8C6-R8C14",  # 9 tiles
    "R9C6-R9C14",  # 9 tiles
]


def parse_tile_range(range_str: str) -> list[tuple[int, int]]:
    """
    Parse a tile range string into (row, col) tuples.

    Supported formats:
    - "R5C7" -> [(5, 7)]
    - "R5C7-R5C9" -> [(5, 7), (5, 8), (5, 9)]
    """
    # Single tile: R#C#
    single_match = re.match(r"^R(\d+)C(\d+)$", range_str)
    if single_match:
        return [(int(single_match.group(1)), int(single_match.group(2)))]

    # Range: R#C#-R#C#
    range_match = re.match(r"^R(\d+)C(\d+)-R(\d+)C(\d+)$", range_str)
    if range_match:
        r1, c1, r2, c2 = map(int, range_match.groups())
        tiles = []
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                tiles.append((r, c))
        return tiles

    raise ValueError(f"Invalid tile range format: {range_str}")


def parse_all_tile_ranges(ranges: list[str]) -> set[tuple[int, int]]:
    """Parse all tile ranges and return unique (row, col) tuples."""
    tiles = set()
    for range_str in ranges:
        tiles.update(parse_tile_range(range_str))
    return tiles


def tile_to_utm(row: int, col: int) -> tuple[int, int]:
    """
    Convert (row, col) to UTM coordinates of the tile's SW corner.

    Returns (easting, northing).
    """
    easting = TILE_GRID_ORIGIN_EASTING + col * TILE_SIZE
    northing = TILE_GRID_ORIGIN_NORTHING - (row + 1) * TILE_SIZE  # SW corner
    return easting, northing


def compute_outer_perimeter(tiles: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Compute the outer perimeter of a tile selection as a list of UTM coordinates.

    Uses a grid edge traversal algorithm:
    1. Build a set of all edge segments (shared edges between tile and non-tile)
    2. Traverse edges to form a closed polygon

    Returns list of (easting, northing) coordinates forming a closed polygon.
    """
    # Build set of all edges
    # Each edge is represented as ((x1, y1), (x2, y2)) with points ordered consistently
    edges = set()

    for row, col in tiles:
        # Get tile corners in UTM
        sw_e, sw_n = tile_to_utm(row, col)
        se_e, se_n = sw_e + TILE_SIZE, sw_n
        ne_e, ne_n = sw_e + TILE_SIZE, sw_n + TILE_SIZE
        nw_e, nw_n = sw_e, sw_n + TILE_SIZE

        # Define the 4 edges of this tile
        tile_edges = [
            ((sw_e, sw_n), (se_e, se_n)),  # South edge
            ((se_e, se_n), (ne_e, ne_n)),  # East edge
            ((ne_e, ne_n), (nw_e, nw_n)),  # North edge
            ((nw_e, nw_n), (sw_e, sw_n)),  # West edge
        ]

        # Check each edge - if neighbor tile exists, it's an internal edge
        neighbors = [
            (row + 1, col),  # South neighbor
            (row, col + 1),  # East neighbor
            (row - 1, col),  # North neighbor
            (row, col - 1),  # West neighbor
        ]

        for edge, neighbor in zip(tile_edges, neighbors):
            if neighbor not in tiles:
                # This is a boundary edge - add it
                # Normalize edge direction for consistent representation
                edges.add(edge)

    # Traverse edges to form polygon
    # Start with an arbitrary edge
    if not edges:
        return []

    edges_list = list(edges)
    current_edge = edges_list[0]
    edges.remove(current_edge)

    polygon = [current_edge[0], current_edge[1]]

    while edges:
        # Find edge that connects to the last point
        last_point = polygon[-1]
        found = False

        for edge in list(edges):
            if edge[0] == last_point:
                polygon.append(edge[1])
                edges.remove(edge)
                found = True
                break
            elif edge[1] == last_point:
                polygon.append(edge[0])
                edges.remove(edge)
                found = True
                break

        if not found:
            # No connecting edge found - might be a disjoint region
            print(f"Warning: Disjoint region detected, {len(edges)} edges remaining")
            break

    # Close the polygon if needed
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    return polygon


def generate_kml(polygon_utm: list[tuple[int, int]], output_path: Path) -> None:
    """
    Generate KML file with red perimeter polygon.

    Args:
        polygon_utm: List of (easting, northing) coordinates
        output_path: Output KML file path
    """
    # Setup coordinate transformer (UTM Zone 17N to WGS84)
    transformer = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)

    # Transform polygon to WGS84
    polygon_wgs84 = []
    for e, n in polygon_utm:
        lon, lat = transformer.transform(e, n)
        polygon_wgs84.append((lon, lat))

    # Format coordinates as "lon,lat,0" separated by spaces
    coords_str = " ".join(f"{lon},{lat},0" for lon, lat in polygon_wgs84)

    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>OSBS Trimmed Selection Perimeter</name>
    <description>Outer perimeter of selected tiles for hillslope analysis</description>

    <!-- Style for perimeter (red outline, no fill) -->
    <Style id="perimeterStyle">
      <PolyStyle>
        <color>00000000</color>
        <fill>0</fill>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>ff0000ff</color>
        <width>4</width>
      </LineStyle>
    </Style>

    <Placemark>
      <name>Selection Perimeter</name>
      <description>Outer boundary of the trimmed tile selection (39 tiles)</description>
      <styleUrl>#perimeterStyle</styleUrl>
      <Polygon>
        <tessellate>1</tessellate>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coords_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>

  </Document>
</kml>
"""

    with open(output_path, "w") as f:
        f.write(kml_content)


def main():
    """Generate perimeter KML for tile selection."""
    # Determine output path
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = DEFAULT_OUTPUT

    print("Parsing tile selection...")
    tiles = parse_all_tile_ranges(TILE_RANGES)
    print(f"Selected {len(tiles)} tiles")

    # Print visual representation
    rows = sorted(set(r for r, c in tiles))
    cols = sorted(set(c for r, c in tiles))
    print(
        f"\nTile selection (rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}):"
    )
    print("        Columns: ", end="")
    for c in range(min(cols), max(cols) + 1):
        print(f"{c:3}", end="")
    print()

    for r in range(min(rows), max(rows) + 1):
        print(f"  Row {r:2}         ", end="")
        for c in range(min(cols), max(cols) + 1):
            if (r, c) in tiles:
                print("  X", end="")
            else:
                print("  .", end="")
        print()

    print("\nComputing outer perimeter...")
    perimeter = compute_outer_perimeter(tiles)
    print(f"Perimeter has {len(perimeter)} vertices")

    print(f"\nGenerating KML: {output_path}")
    generate_kml(perimeter, output_path)

    print(f"\nDone. Open {output_path} in Google Earth to view.")
    print("Overlay with osbs_tile_grid.kml to see selection context.")


if __name__ == "__main__":
    main()
