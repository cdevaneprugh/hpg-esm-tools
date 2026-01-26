#!/usr/bin/env python3
"""
Extract a 4x4 km smoke test subset from the OSBS DTM mosaic.

Uses user-provided WGS84 corner coordinates to identify the 4x4 tile grid,
then extracts from the mosaic and generates a verification heatmap.

Creates:
- data/osbs_smoke_test_4x4.tif - 4000x4000 pixel subset DEM
- output/smoke_test/elevation_heatmap.png - verification plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "smoke_test"

MOSAIC_PATH = DATA_DIR / "NEON_OSBS_DTM_mosaic.tif"
SUBSET_PATH = DATA_DIR / "osbs_smoke_test_4x4.tif"
HEATMAP_PATH = OUTPUT_DIR / "elevation_heatmap.png"

# User-provided corner coordinates (WGS84)
# Points contained within corner tiles
CORNERS_WGS84 = {
    "upper_left": {"lon": -81.9972, "lat": 29.7139},  # 29°42'50"N 81°59'50"W
    "upper_right": {"lon": -81.9669, "lat": 29.7139},  # 29°42'50"N 81°58'01"W
    "lower_left": {"lon": -81.9975, "lat": 29.6864},  # 29°41'11"N 81°59'51"W
    "lower_right": {"lon": -81.9669, "lat": 29.6864},  # 29°41'11"N 81°58'01"W
}

# NEON tile size (1 km = 1000 m = 1000 pixels at 1m resolution)
TILE_SIZE = 1000


def convert_wgs84_to_utm(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 (lon, lat) to UTM Zone 17N (easting, northing)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    return transformer.transform(lon, lat)


def get_tile_bounds(easting: float, northing: float) -> tuple[int, int]:
    """
    Get the tile origin (southwest corner) for a given UTM coordinate.

    NEON tiles are 1000m x 1000m aligned to 1000m boundaries.
    """
    tile_e = int(easting // TILE_SIZE) * TILE_SIZE
    tile_n = int(northing // TILE_SIZE) * TILE_SIZE
    return tile_e, tile_n


def extract_subset(
    src: rasterio.DatasetReader, bounds: dict
) -> tuple[np.ndarray, dict]:
    """
    Extract a rectangular subset from the mosaic based on UTM bounds.

    Returns:
        dem: 2D array of elevation values
        profile: rasterio profile for the subset
    """
    # Convert bounds to pixel coordinates
    transform = src.transform
    col_min = int((bounds["west"] - transform.c) / transform.a)
    col_max = int((bounds["east"] - transform.c) / transform.a)
    row_min = int((transform.f - bounds["north"]) / abs(transform.e))
    row_max = int((transform.f - bounds["south"]) / abs(transform.e))

    # Create window
    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)

    # Read data
    dem = src.read(1, window=window)

    # Create new profile
    new_transform = rasterio.Affine(
        transform.a,
        transform.b,
        bounds["west"],
        transform.d,
        transform.e,
        bounds["north"],
    )

    profile = src.profile.copy()
    profile.update(
        width=dem.shape[1],
        height=dem.shape[0],
        transform=new_transform,
    )

    return dem, profile


def generate_heatmap(dem: np.ndarray, bounds: dict, output_path: Path):
    """Generate elevation heatmap for verification."""
    # Mask nodata values
    dem_masked = np.ma.masked_less_equal(dem, -9000)

    # Calculate statistics
    elev_min = float(dem_masked.min())
    elev_max = float(dem_masked.max())
    elev_mean = float(dem_masked.mean())

    # Calculate extent
    extent_e = (bounds["east"] - bounds["west"]) / 1000  # km
    extent_n = (bounds["north"] - bounds["south"]) / 1000  # km

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot with terrain colormap
    im = ax.imshow(
        dem_masked,
        cmap="terrain",
        vmin=elev_min,
        vmax=elev_max,
        extent=[bounds["west"], bounds["east"], bounds["south"], bounds["north"]],
        aspect="equal",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Elevation (m)", fontsize=12)

    # Labels
    ax.set_title(
        f"OSBS Smoke Test Subset (1m NEON LIDAR)\n"
        f"Extent: {extent_e:.1f} x {extent_n:.1f} km | "
        f"Elevation: {elev_min:.1f} - {elev_max:.1f} m | "
        f"Shape: {dem.shape[1]} x {dem.shape[0]} px",
        fontsize=12,
    )
    ax.set_xlabel("Easting (m UTM 17N)", fontsize=11)
    ax.set_ylabel("Northing (m UTM 17N)", fontsize=11)

    # Add corner coordinates annotation
    ax.annotate(
        f"NW: {bounds['west']:.0f}E, {bounds['north']:.0f}N",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.annotate(
        f"SE: {bounds['east']:.0f}E, {bounds['south']:.0f}N",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return elev_min, elev_max, elev_mean


def main():
    """Main entry point."""
    print("=" * 60)
    print("OSBS Smoke Test Subset Extraction")
    print("=" * 60)

    # Step 1: Convert corners to UTM
    print("\n1. Converting corner coordinates WGS84 -> UTM 17N...")
    corners_utm = {}
    for name, coords in CORNERS_WGS84.items():
        e, n = convert_wgs84_to_utm(coords["lon"], coords["lat"])
        corners_utm[name] = {"easting": e, "northing": n}
        print(
            f"   {name}: ({coords['lon']:.4f}, {coords['lat']:.4f}) -> ({e:.0f}E, {n:.0f}N)"
        )

    # Step 2: Identify tile boundaries
    print("\n2. Identifying tile boundaries (1000m grid)...")
    ul_tile = get_tile_bounds(
        corners_utm["upper_left"]["easting"], corners_utm["upper_left"]["northing"]
    )
    lr_tile = get_tile_bounds(
        corners_utm["lower_right"]["easting"], corners_utm["lower_right"]["northing"]
    )
    print(f"   Upper-left tile origin:  {ul_tile[0]}E, {ul_tile[1]}N")
    print(f"   Lower-right tile origin: {lr_tile[0]}E, {lr_tile[1]}N")

    # Calculate bounds for 4x4 tile grid
    # West edge: western tile origin
    # East edge: eastern tile origin + tile size (to include full tile)
    # South edge: southern tile origin
    # North edge: northern tile origin + tile size (to include full tile)
    bounds = {
        "west": ul_tile[0],
        "east": lr_tile[0] + TILE_SIZE,
        "south": lr_tile[1],
        "north": ul_tile[1] + TILE_SIZE,
    }

    n_tiles_e = (bounds["east"] - bounds["west"]) // TILE_SIZE
    n_tiles_n = (bounds["north"] - bounds["south"]) // TILE_SIZE
    print(
        f"   Grid size: {n_tiles_e} x {n_tiles_n} tiles ({bounds['east'] - bounds['west']}m x {bounds['north'] - bounds['south']}m)"
    )

    print("\n   Final bounds:")
    print(f"     West:  {bounds['west']:.0f}E")
    print(f"     East:  {bounds['east']:.0f}E")
    print(f"     South: {bounds['south']:.0f}N")
    print(f"     North: {bounds['north']:.0f}N")

    # Step 3: Check against mosaic extent
    print("\n3. Checking against mosaic extent...")
    with rasterio.open(MOSAIC_PATH) as src:
        mosaic_bounds = src.bounds
        print(
            f"   Mosaic extent: {mosaic_bounds.left:.0f}E - {mosaic_bounds.right:.0f}E, "
            f"{mosaic_bounds.bottom:.0f}N - {mosaic_bounds.top:.0f}N"
        )

        # Verify subset is within mosaic
        if bounds["west"] < mosaic_bounds.left or bounds["east"] > mosaic_bounds.right:
            raise ValueError("Subset east-west bounds exceed mosaic extent!")
        if (
            bounds["south"] < mosaic_bounds.bottom
            or bounds["north"] > mosaic_bounds.top
        ):
            raise ValueError("Subset north-south bounds exceed mosaic extent!")
        print("   OK - subset is within mosaic bounds")

        # Step 4: Extract subset
        print("\n4. Extracting subset from mosaic...")
        dem, profile = extract_subset(src, bounds)
        print(f"   Extracted shape: {dem.shape[1]} x {dem.shape[0]} pixels")
        print(
            f"   Data range: {dem[dem > -9000].min():.1f} - {dem[dem > -9000].max():.1f} m"
        )

    # Step 5: Save subset GeoTIFF
    print(f"\n5. Saving subset to {SUBSET_PATH}...")
    with rasterio.open(SUBSET_PATH, "w", **profile) as dst:
        dst.write(dem, 1)
    print(f"   File size: {SUBSET_PATH.stat().st_size / 1e6:.1f} MB")

    # Step 6: Generate verification heatmap
    print("\n6. Generating verification heatmap...")
    elev_min, elev_max, elev_mean = generate_heatmap(dem, bounds, HEATMAP_PATH)
    print(f"   Saved to {HEATMAP_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Subset file:     {SUBSET_PATH}")
    print(f"Heatmap:         {HEATMAP_PATH}")
    print(f"Grid size:       {n_tiles_e} x {n_tiles_n} tiles")
    print(f"Pixel size:      {dem.shape[1]} x {dem.shape[0]} px")
    print(
        f"Geographic size: {(bounds['east'] - bounds['west']) / 1000:.1f} x {(bounds['north'] - bounds['south']) / 1000:.1f} km"
    )
    print(f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m")
    print(f"Mean elevation:  {elev_mean:.1f} m")
    print()
    print(
        "VERIFICATION: Review elevation_heatmap.png to confirm correct area extracted."
    )


if __name__ == "__main__":
    main()
