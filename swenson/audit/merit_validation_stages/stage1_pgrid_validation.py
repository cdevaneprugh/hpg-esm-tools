#!/usr/bin/env python
"""
Stage 1: pgrid Validation on MERIT DEM

Validate that our pysheds fork's pgrid methods work correctly on a full
MERIT DEM tile (6000x6000 pixels, ~90m resolution).

This script:
1. Loads the full MERIT DEM tile (n30w095)
2. Computes flow direction and accumulation using pgrid
3. Creates a stream network mask using an accumulation threshold
4. Computes HAND (Height Above Nearest Drainage) and DTND (Distance To Nearest Drainage)
5. Saves intermediate outputs as GeoTIFFs
6. Generates diagnostic plots

Data paths:
- Input: /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif
- Output: swenson/output/merit_validation/stage1/

Expected runtime: ~5 minutes on 4 cores with 32GB RAM
"""

import os
import sys
import time
import numpy as np

# Add pysheds fork to path
pysheds_fork = os.environ.get("PYSHEDS_FORK", "/blue/gerber/cdevaneprugh/pysheds_fork")
sys.path.insert(0, pysheds_fork)

from pysheds.pgrid import Grid

# Optional imports for output
try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available, cannot save GeoTIFFs")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, cannot generate plots")


# Configuration
MERIT_DEM_PATH = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif"
)
OUTPUT_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage1"

# Accumulation threshold for stream network
# Paper uses A_thresh = 0.5 * Lc^2 where Lc is characteristic length
# For initial validation, use a fixed threshold (in number of cells)
# 1000 cells * 90m * 90m = ~8.1 km^2 contributing area
ACCUM_THRESHOLD = 1000


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def save_geotiff(
    data: np.ndarray, filepath: str, transform, crs: str = "EPSG:4326"
) -> None:
    """Save array as GeoTIFF."""
    if not HAS_RASTERIO:
        print(f"  Skipping {filepath} (rasterio not available)")
        return

    # Handle nodata
    nodata = -9999.0
    data_out = np.where(np.isnan(data), nodata, data).astype(np.float32)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data_out, 1)

    print(f"  Saved: {filepath}")


def create_diagnostic_plots(
    dem: np.ndarray,
    acc: np.ndarray,
    hand: np.ndarray,
    dtnd: np.ndarray,
    stream_mask: np.ndarray,
    output_dir: str,
) -> None:
    """Generate diagnostic plots."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
        return

    print_section("Generating Diagnostic Plots")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. DEM
    ax = axes[0, 0]
    im = ax.imshow(dem, cmap="terrain")
    ax.set_title("Digital Elevation Model (m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Flow accumulation (log scale)
    ax = axes[0, 1]
    acc_log = np.log10(np.maximum(acc, 1))
    im = ax.imshow(acc_log, cmap="Blues")
    ax.set_title("Flow Accumulation (log10 cells)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Stream network
    ax = axes[0, 2]
    ax.imshow(dem, cmap="terrain", alpha=0.5)
    ax.imshow(
        np.ma.masked_where(stream_mask == 0, stream_mask), cmap="Blues", alpha=0.8
    )
    ax.set_title(f"Stream Network (threshold={ACCUM_THRESHOLD} cells)")

    # 4. HAND
    ax = axes[1, 0]
    hand_plot = np.where(hand < 0, np.nan, hand)
    im = ax.imshow(
        hand_plot, cmap="viridis", vmin=0, vmax=np.nanpercentile(hand_plot, 99)
    )
    ax.set_title("HAND - Height Above Nearest Drainage (m)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 5. DTND
    ax = axes[1, 1]
    dtnd_km = dtnd / 1000.0  # Convert to km
    im = ax.imshow(dtnd_km, cmap="magma", vmin=0, vmax=np.nanpercentile(dtnd_km, 99))
    ax.set_title("DTND - Distance To Nearest Drainage (km)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 6. HAND histogram
    ax = axes[1, 2]
    hand_valid = hand_plot[~np.isnan(hand_plot)]
    ax.hist(hand_valid, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("HAND (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("HAND Distribution")
    ax.axvline(
        np.median(hand_valid),
        color="r",
        linestyle="--",
        label=f"Median: {np.median(hand_valid):.1f} m",
    )
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "stage1_diagnostic_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    # Additional plot: Zoom into a representative area
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Zoom region (center 1000x1000 pixels)
    y_center, x_center = dem.shape[0] // 2, dem.shape[1] // 2
    y_slice = slice(y_center - 500, y_center + 500)
    x_slice = slice(x_center - 500, x_center + 500)

    # Zoomed DEM
    ax = axes[0]
    im = ax.imshow(dem[y_slice, x_slice], cmap="terrain")
    ax.set_title("DEM (zoomed center 1000x1000)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Zoomed HAND
    ax = axes[1]
    im = ax.imshow(
        hand_plot[y_slice, x_slice],
        cmap="viridis",
        vmin=0,
        vmax=np.nanpercentile(hand_plot, 95),
    )
    ax.set_title("HAND (zoomed)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Zoomed stream network with DTND
    ax = axes[2]
    ax.imshow(
        dtnd_km[y_slice, x_slice],
        cmap="magma",
        vmin=0,
        vmax=np.nanpercentile(dtnd_km, 95),
    )
    stream_zoom = stream_mask[y_slice, x_slice]
    ax.contour(stream_zoom, levels=[0.5], colors="cyan", linewidths=0.5)
    ax.set_title("DTND with stream network (cyan)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="DTND (km)")

    plt.tight_layout()
    zoom_path = os.path.join(output_dir, "stage1_zoomed_plots.png")
    plt.savefig(zoom_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {zoom_path}")


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("Stage 1: pgrid Validation on MERIT DEM")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check input file
    if not os.path.exists(MERIT_DEM_PATH):
        print(f"ERROR: DEM file not found: {MERIT_DEM_PATH}")
        sys.exit(1)

    print(f"Input DEM: {MERIT_DEM_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Accumulation threshold: {ACCUM_THRESHOLD} cells")

    # -------------------------------------------------------------------------
    # Step 1: Load DEM with pgrid
    # -------------------------------------------------------------------------
    print_section("Step 1: Loading DEM with pgrid")

    t0 = time.time()
    grid = Grid.from_raster(MERIT_DEM_PATH, "dem")
    dem = grid.dem  # Access the loaded data as attribute

    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM dtype: {dem.dtype}")
    print(f"  DEM range: [{np.nanmin(dem):.2f}, {np.nanmax(dem):.2f}] m")
    print(f"  Grid CRS: {grid.crs}")
    print(f"  Grid affine:\n    {grid.affine}")
    print(f"  Load time: {time.time() - t0:.1f} seconds")

    # Get coordinate arrays for later use
    transform = grid.affine

    # -------------------------------------------------------------------------
    # Step 2: Condition DEM (fill pits)
    # -------------------------------------------------------------------------
    print_section("Step 2: Conditioning DEM (fill pits)")

    t0 = time.time()

    # Fill pits - pgrid method (stores as grid.pit_filled)
    grid.fill_pits("dem", out_name="pit_filled")
    print("  Pits filled")

    # Fill depressions (stores as grid.flooded)
    grid.fill_depressions("pit_filled", out_name="flooded")
    print("  Depressions filled")

    # Resolve flats (stores as grid.inflated)
    grid.resolve_flats("flooded", out_name="inflated")
    print("  Flats resolved")

    print(f"  Conditioning time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 3: Compute flow direction
    # -------------------------------------------------------------------------
    print_section("Step 3: Computing Flow Direction")

    t0 = time.time()

    # D8 direction map (pysheds convention)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # D8 flow direction (stores as grid.fdir)
    grid.flowdir("inflated", out_name="fdir", dirmap=dirmap, routing="d8")
    fdir = grid.fdir
    print("  Flow direction computed")
    print(f"  fdir shape: {fdir.shape}")
    print(f"  fdir dtype: {fdir.dtype}")
    print(f"  Unique directions: {np.unique(fdir)}")
    print(f"  Flow direction time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 4: Compute flow accumulation
    # -------------------------------------------------------------------------
    print_section("Step 4: Computing Flow Accumulation")

    t0 = time.time()

    # Flow accumulation (stores as grid.acc)
    grid.accumulation("fdir", out_name="acc", dirmap=dirmap, routing="d8")
    acc = grid.acc
    print("  Flow accumulation computed")
    print(f"  acc shape: {acc.shape}")
    print(f"  acc range: [{np.min(acc)}, {np.max(acc)}] cells")
    print(f"  Accumulation time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 5: Create stream network mask
    # -------------------------------------------------------------------------
    print_section("Step 5: Creating Stream Network Mask")

    t0 = time.time()

    # Create mask where accumulation exceeds threshold
    acc_mask = acc > ACCUM_THRESHOLD

    # Use pgrid's create_channel_mask to get channel_mask, channel_id, and bank_mask
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=dirmap, routing="d8")

    stream_mask = grid.channel_mask
    channel_id = grid.channel_id

    stream_cells = np.sum(stream_mask > 0)
    num_channels = int(np.nanmax(channel_id)) if np.any(~np.isnan(channel_id)) else 0
    print(f"  Number of channel segments: {num_channels}")
    print(
        f"  Stream cells: {stream_cells} ({100 * stream_cells / stream_mask.size:.2f}%)"
    )
    print(f"  Stream mask time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 6: Compute HAND and DTND
    # -------------------------------------------------------------------------
    print_section("Step 6: Computing HAND and DTND")

    t0 = time.time()

    # Compute HAND using pgrid method (also computes DTND internally)
    # Stores results as grid.hand and grid.dtnd
    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=dirmap,
        routing="d8",
    )

    hand = grid.hand
    dtnd = grid.dtnd

    print("  HAND computed")
    print(f"  HAND shape: {hand.shape}")
    print(f"  HAND range: [{np.nanmin(hand):.2f}, {np.nanmax(hand):.2f}] m")

    hand_valid = hand[~np.isnan(hand) & (hand >= 0)]
    print("  HAND statistics (valid cells):")
    print(f"    Mean: {np.mean(hand_valid):.2f} m")
    print(f"    Median: {np.median(hand_valid):.2f} m")
    print(f"    Std: {np.std(hand_valid):.2f} m")
    print(f"    95th percentile: {np.percentile(hand_valid, 95):.2f} m")

    print("\n  DTND computed")
    print(f"  DTND shape: {dtnd.shape}")
    print(f"  DTND range: [{np.nanmin(dtnd):.0f}, {np.nanmax(dtnd):.0f}] m")

    dtnd_valid = dtnd[~np.isnan(dtnd) & (dtnd >= 0)]
    print("  DTND statistics (valid cells):")
    print(f"    Mean: {np.mean(dtnd_valid) / 1000:.2f} km")
    print(f"    Median: {np.median(dtnd_valid) / 1000:.2f} km")
    print(f"    Std: {np.std(dtnd_valid) / 1000:.2f} km")
    print(f"    95th percentile: {np.percentile(dtnd_valid, 95) / 1000:.2f} km")

    print(f"\n  HAND/DTND computation time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------------------------
    print_section("Step 7: Saving Outputs")

    # Save GeoTIFFs
    save_geotiff(np.array(dem), os.path.join(OUTPUT_DIR, "dem.tif"), transform)
    save_geotiff(
        np.log10(np.maximum(acc, 1)).astype(np.float32),
        os.path.join(OUTPUT_DIR, "accumulation_log10.tif"),
        transform,
    )
    save_geotiff(
        stream_mask.astype(np.float32),
        os.path.join(OUTPUT_DIR, "stream_mask.tif"),
        transform,
    )
    save_geotiff(np.array(hand), os.path.join(OUTPUT_DIR, "hand.tif"), transform)
    save_geotiff(np.array(dtnd), os.path.join(OUTPUT_DIR, "dtnd.tif"), transform)

    # Save summary statistics to text file
    summary_path = os.path.join(OUTPUT_DIR, "stage1_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 1: pgrid Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input DEM: {MERIT_DEM_PATH}\n")
        f.write(f"DEM shape: {dem.shape}\n")
        f.write(f"DEM range: [{np.nanmin(dem):.2f}, {np.nanmax(dem):.2f}] m\n\n")
        f.write(f"Accumulation threshold: {ACCUM_THRESHOLD} cells\n")
        f.write(
            f"Stream cells: {stream_cells} ({100 * stream_cells / stream_mask.size:.2f}%)\n\n"
        )
        f.write("HAND Statistics:\n")
        f.write(f"  Range: [{np.nanmin(hand):.2f}, {np.nanmax(hand):.2f}] m\n")
        f.write(f"  Mean: {np.mean(hand_valid):.2f} m\n")
        f.write(f"  Median: {np.median(hand_valid):.2f} m\n")
        f.write(f"  Std: {np.std(hand_valid):.2f} m\n")
        f.write(f"  95th percentile: {np.percentile(hand_valid, 95):.2f} m\n\n")
        f.write("DTND Statistics:\n")
        f.write(f"  Range: [{np.nanmin(dtnd):.0f}, {np.nanmax(dtnd):.0f}] m\n")
        f.write(f"  Mean: {np.mean(dtnd_valid) / 1000:.2f} km\n")
        f.write(f"  Median: {np.median(dtnd_valid) / 1000:.2f} km\n")
        f.write(f"  Std: {np.std(dtnd_valid) / 1000:.2f} km\n")
        f.write(f"  95th percentile: {np.percentile(dtnd_valid, 95) / 1000:.2f} km\n\n")
        f.write(f"Total processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # -------------------------------------------------------------------------
    # Step 8: Generate diagnostic plots
    # -------------------------------------------------------------------------
    create_diagnostic_plots(
        dem=np.array(dem),
        acc=np.array(acc),
        hand=np.array(hand),
        dtnd=np.array(dtnd),
        stream_mask=stream_mask,
        output_dir=OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Stage 1 Complete")

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nValidation criteria:")
    print(
        f"  - HAND range: [0, ~500m] for this region ({'PASS' if np.nanmax(hand) < 1000 else 'CHECK'})"
    )
    print(
        f"  - DTND range: [0, ~50km] for this region ({'PASS' if np.nanmax(dtnd) / 1000 < 100 else 'CHECK'})"
    )
    print(
        f"  - Stream coverage: ~1-5% of cells ({'PASS' if 0.5 < 100 * stream_cells / stream_mask.size < 10 else 'CHECK'})"
    )


if __name__ == "__main__":
    main()
