#!/usr/bin/env python3
"""
Mosaic NEON OSBS DTM tiles and generate elevation heatmap.

Creates:
- data/NEON_OSBS_DTM_mosaic.tif - merged DEM
- output/osbs_dtm_elevation.png - elevation visualization
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DTM_DIR = DATA_DIR / "NEON_OSBS_DTM"
MOSAIC_PATH = DATA_DIR / "NEON_OSBS_DTM_mosaic.tif"
PLOT_PATH = OUTPUT_DIR / "osbs_dtm_elevation.png"


def mosaic_tiles():
    """Merge all DTM tiles into a single GeoTIFF."""
    print("Finding DTM tiles...")
    dtm_files = sorted(glob.glob(str(DTM_DIR / "*.tif")))
    print(f"Found {len(dtm_files)} tiles")

    print("Opening datasets...")
    datasets = [rasterio.open(f) for f in dtm_files]

    print("Merging tiles...")
    mosaic_arr, mosaic_transform = merge(datasets)

    # Get metadata from first file
    meta = datasets[0].meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": mosaic_arr.shape[1],
            "width": mosaic_arr.shape[2],
            "transform": mosaic_transform,
            "compress": "lzw",
        }
    )

    print(f"Writing mosaic to {MOSAIC_PATH}...")
    with rasterio.open(MOSAIC_PATH, "w", **meta) as dst:
        dst.write(mosaic_arr)

    # Close all datasets
    for ds in datasets:
        ds.close()

    print(f"Mosaic shape: {mosaic_arr.shape[1]} x {mosaic_arr.shape[2]} pixels")
    return mosaic_arr[0], mosaic_transform


def generate_heatmap(dem: np.ndarray, transform):
    """Generate elevation heatmap visualization."""
    print("Generating elevation heatmap...")

    # Mask nodata values (-9999)
    dem_masked = np.ma.masked_less_equal(dem, -9000)

    # Calculate statistics
    elev_min = dem_masked.min()
    elev_max = dem_masked.max()
    elev_mean = dem_masked.mean()
    print(
        f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m (mean: {elev_mean:.1f} m)"
    )

    # Calculate extent in km (UTM coordinates)
    height, width = dem.shape
    x_extent = width / 1000  # km
    y_extent = height / 1000  # km
    print(f"Spatial extent: {x_extent:.1f} x {y_extent:.1f} km")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot with terrain colormap
    im = ax.imshow(
        dem_masked,
        cmap="terrain",
        vmin=elev_min,
        vmax=elev_max,
        aspect="equal",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Elevation (m)", fontsize=12)

    # Labels
    ax.set_title(
        f"OSBS Digital Terrain Model (1m NEON LIDAR)\n"
        f"Extent: {x_extent:.1f} x {y_extent:.1f} km | "
        f"Elevation: {elev_min:.1f} - {elev_max:.1f} m",
        fontsize=14,
    )
    ax.set_xlabel("Easting (pixels)", fontsize=11)
    ax.set_ylabel("Northing (pixels)", fontsize=11)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {PLOT_PATH}")
    plt.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("OSBS DTM Mosaic and Visualization")
    print("=" * 60)

    # Check if mosaic already exists
    if MOSAIC_PATH.exists():
        print(f"Mosaic already exists at {MOSAIC_PATH}")
        print("Loading existing mosaic...")
        with rasterio.open(MOSAIC_PATH) as src:
            dem = src.read(1)
            transform = src.transform
    else:
        dem, transform = mosaic_tiles()

    generate_heatmap(dem, transform)

    print("=" * 60)
    print("Done!")
    print(f"  Mosaic: {MOSAIC_PATH}")
    print(f"  Heatmap: {PLOT_PATH}")


if __name__ == "__main__":
    main()
