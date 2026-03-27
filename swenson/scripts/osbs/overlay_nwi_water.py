#!/usr/bin/env python3
"""
Overlay NWI water mask on the production DEM hillshade.

Loads the production DTM and water mask rasters, generates a hillshade,
and overlays the water mask boundary as black contour lines.

Usage:
    python overlay_nwi_water.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from matplotlib.patches import Patch

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/

DTM_MOSAIC = BASE_DIR / "data" / "mosaics" / "production" / "dtm.tif"
WATER_MASK_PATH = BASE_DIR / "data" / "mosaics" / "production" / "water_mask.tif"
OUTPUT_DIR = BASE_DIR / "output" / "osbs"
OUTPUT_PNG = OUTPUT_DIR / "nwi_water_overlay.png"


def make_hillshade(dem: np.ndarray, nodata: float) -> np.ndarray:
    """Generate high-contrast hillshade from DEM."""
    ls = LightSource(azdeg=315, altdeg=45)
    masked = np.where(dem == nodata, np.nan, dem)
    return ls.hillshade(masked, vert_exag=15, dx=1, dy=1)


def main():
    # --- Load production rasters ---
    print(f"Reading DTM: {DTM_MOSAIC.name}")
    with rasterio.open(DTM_MOSAIC) as src:
        dtm_bounds = src.bounds
        dtm = src.read(1)
        dtm_nodata = src.nodata
    print(f"  Shape: {dtm.shape}")

    print(f"Reading water mask: {WATER_MASK_PATH.name}")
    with rasterio.open(WATER_MASK_PATH) as src:
        water_mask = src.read(1)
    print(f"  Shape: {water_mask.shape}")

    assert dtm.shape == water_mask.shape, (
        f"Shape mismatch: DTM {dtm.shape}, mask {water_mask.shape}"
    )

    # --- Summary ---
    n_water = int(np.sum(water_mask == 1))
    n_total = water_mask.size
    pct = 100.0 * n_water / n_total
    area_ha = n_water / 1e4
    print(f"\n  Water pixels: {n_water:,} / {n_total:,} ({pct:.1f}%)")
    print(f"  Water area: {area_ha:.1f} ha")

    # --- Generate hillshade ---
    print("\nGenerating hillshade...")
    hillshade = make_hillshade(dtm, dtm_nodata)

    # --- Plot overlay ---
    print("Plotting overlay...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    extent = [dtm_bounds.left, dtm_bounds.right, dtm_bounds.bottom, dtm_bounds.top]
    ax.imshow(hillshade, cmap="gray", extent=extent, origin="upper", vmin=0.0, vmax=1.0)

    # Water mask boundary as contour lines
    ax.contour(
        water_mask,
        levels=[0.5],
        colors="blue",
        linewidths=1.2,
        extent=extent,
        origin="upper",
    )

    legend_handle = Patch(facecolor="none", edgecolor="blue", linewidth=1.2)
    ax.legend(
        [legend_handle],
        ["NWI Water Mask"],
        loc="lower left",
        fontsize=10,
        framealpha=0.8,
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        f"NWI Water Mask over Production DEM\n"
        f"{n_water:,} water pixels ({pct:.1f}%), {area_ha:.1f} ha"
    )
    ax.set_aspect("equal")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
