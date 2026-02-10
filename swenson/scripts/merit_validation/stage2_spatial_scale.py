#!/usr/bin/env python
"""
Stage 2: Spatial Scale Analysis using FFT

Determine the characteristic spatial scale for the MERIT DEM tile using
FFT analysis of the Laplacian, following Swenson & Lawrence (2025).

This script:
1. Loads the MERIT DEM (or a subset for faster testing)
2. Computes the Laplacian of the elevation field
3. Applies FFT to identify the wavelength with maximum amplitude
4. Determines the characteristic length scale (Lc)
5. Calculates the accumulation threshold: A_thresh = 0.5 * Lc^2

The spatial scale is used in Stage 3 to set the stream network threshold.

Data paths:
- Input: /blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif
- Output: swenson/output/merit_validation/stage2/

Expected runtime: ~5-15 minutes on 4 cores with 32GB RAM
"""

import os
import sys
import time
import json
import numpy as np

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_scale import identify_spatial_scale_laplacian_dem

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Configuration
MERIT_DEM_PATH = (
    "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/merit/n30w095_dem.tif"
)
OUTPUT_DIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/merit_validation/stage2"

# Analysis parameters
MAX_HILLSLOPE_LENGTH = 10000  # meters (10 km max)
NLAMBDA = 30  # number of wavelength bins

# For testing, process subregions at different sizes
# Set to None to process full tile
SUBREGION_SIZES = [500, 1000, 3000]  # pixels per side


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def load_dem_with_coords(
    filepath: str, subsample: int = None, center_region: int = None
) -> dict:
    """
    Load DEM and compute coordinate arrays.

    Parameters
    ----------
    filepath : str
        Path to GeoTIFF file
    subsample : int, optional
        Subsample factor (e.g., 2 means take every 2nd pixel)
    center_region : int, optional
        Extract center region of this size (pixels per side)

    Returns
    -------
    dict with keys: elev, lon, lat, transform, crs
    """
    with rasterio.open(filepath) as src:
        elev = src.read(1)
        transform = src.transform
        crs = src.crs

        # Get coordinate arrays
        nrows, ncols = elev.shape

        # Create 1D coordinate arrays from transform
        lon = np.array([transform.c + transform.a * (i + 0.5) for i in range(ncols)])
        lat = np.array([transform.f + transform.e * (j + 0.5) for j in range(nrows)])

    # Extract center region if requested
    if center_region is not None:
        cy, cx = nrows // 2, ncols // 2
        half = center_region // 2
        y_slice = slice(cy - half, cy + half)
        x_slice = slice(cx - half, cx + half)
        elev = elev[y_slice, x_slice]
        lon = lon[x_slice]
        lat = lat[y_slice]

    # Subsample if requested
    if subsample is not None and subsample > 1:
        elev = elev[::subsample, ::subsample]
        lon = lon[::subsample]
        lat = lat[::subsample]

    return {"elev": elev, "lon": lon, "lat": lat, "transform": transform, "crs": crs}


def create_spectral_plots(results: list, output_dir: str) -> None:
    """Generate spectral analysis plots."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
        return

    print_section("Generating Spectral Analysis Plots")

    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(6 * n_results, 10))

    if n_results == 1:
        axes = axes.reshape(2, 1)

    for i, res in enumerate(results):
        label = res.get("label", f"Region {i + 1}")
        lambda_1d = res["lambda_1d"]
        laplac_amp_1d = res["laplac_amp_1d"]
        spatial_scale = res["spatialScale"]
        model = res["model"]
        ares = res["res"]

        # Top plot: Amplitude spectrum
        ax = axes[0, i]
        ax.semilogy(lambda_1d, laplac_amp_1d, "b.-", linewidth=1.5, markersize=4)
        ax.axvline(
            spatial_scale,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Lc = {spatial_scale:.1f} px ({spatial_scale * ares:.0f} m)",
        )
        ax.set_xlabel("Wavelength (pixels)")
        ax.set_ylabel("Laplacian Amplitude")
        ax.set_title(f"{label}\nModel: {model}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom plot: Amplitude vs wavelength in meters
        ax = axes[1, i]
        lambda_m = lambda_1d * ares / 1000  # Convert to km
        ax.semilogy(lambda_m, laplac_amp_1d, "b.-", linewidth=1.5, markersize=4)
        ax.axvline(
            spatial_scale * ares / 1000,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Lc = {spatial_scale * ares / 1000:.1f} km",
        )
        ax.set_xlabel("Wavelength (km)")
        ax.set_ylabel("Laplacian Amplitude")
        ax.set_title(f"Accumulation threshold: {0.5 * spatial_scale**2:.0f} cells")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "stage2_spectral_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")


def main():
    """Main processing function."""
    start_time = time.time()

    print_section("Stage 2: Spatial Scale Analysis")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not HAS_RASTERIO:
        print("ERROR: rasterio not available, cannot read GeoTIFF")
        sys.exit(1)

    if not os.path.exists(MERIT_DEM_PATH):
        print(f"ERROR: DEM file not found: {MERIT_DEM_PATH}")
        sys.exit(1)

    print(f"Input DEM: {MERIT_DEM_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max hillslope length: {MAX_HILLSLOPE_LENGTH} m")

    # -------------------------------------------------------------------------
    # Step 1: Analyze multiple region sizes
    # -------------------------------------------------------------------------
    print_section("Step 1: Analyzing Spatial Scales at Different Region Sizes")

    results = []

    for size in SUBREGION_SIZES:
        print(f"\n--- Processing {size}x{size} center region ---")

        t0 = time.time()

        # Load center region
        data = load_dem_with_coords(MERIT_DEM_PATH, center_region=size)
        elev = data["elev"]
        lon = data["lon"]
        lat = data["lat"]

        print(f"  Loaded region: {elev.shape}")
        print(f"  Elevation range: [{np.nanmin(elev):.1f}, {np.nanmax(elev):.1f}] m")
        print(f"  Lon range: [{lon[0]:.4f}, {lon[-1]:.4f}]")
        print(f"  Lat range: [{lat[0]:.4f}, {lat[-1]:.4f}]")

        # Run spatial scale analysis
        result = identify_spatial_scale_laplacian_dem(
            elev=elev,
            elon=lon,
            elat=lat,
            max_hillslope_length=MAX_HILLSLOPE_LENGTH,
            land_threshold=0.75,
            min_land_elevation=0,
            detrend_elevation=True,
            blend_edges_flag=True,
            zero_edges=True,
            nlambda=NLAMBDA,
            verbose=True,
        )

        result["label"] = f"{size}x{size} region"
        result["region_size"] = size
        results.append(result)

        # Print results
        Lc = result["spatialScale"]
        ares = result["res"]
        A_thresh = 0.5 * Lc**2

        print(f"\n  Results for {size}x{size}:")
        print(f"    Model: {result['model']}")
        print(f"    Spatial scale: {Lc:.1f} pixels")
        print(f"    Spatial scale: {Lc * ares:.0f} m = {Lc * ares / 1000:.2f} km")
        print(f"    Accumulation threshold: {A_thresh:.0f} cells")
        print(f"    Processing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 2: Full tile analysis (if not too slow)
    # -------------------------------------------------------------------------
    print_section("Step 2: Full Tile Analysis (subsampled)")

    # Subsample to ~1500x1500 for reasonable memory/time
    subsample_factor = 4
    t0 = time.time()

    data = load_dem_with_coords(MERIT_DEM_PATH, subsample=subsample_factor)
    elev = data["elev"]
    lon = data["lon"]
    lat = data["lat"]

    print(f"  Loaded subsampled ({subsample_factor}x) tile: {elev.shape}")
    print(f"  Effective resolution: {subsample_factor * 90:.0f} m")

    # Run spatial scale analysis
    result_full = identify_spatial_scale_laplacian_dem(
        elev=elev,
        elon=lon,
        elat=lat,
        max_hillslope_length=MAX_HILLSLOPE_LENGTH,
        land_threshold=0.75,
        min_land_elevation=0,
        detrend_elevation=True,
        blend_edges_flag=True,
        zero_edges=True,
        nlambda=NLAMBDA,
        verbose=True,
    )

    # Scale spatial scale back to original resolution
    result_full["spatialScale"] *= subsample_factor
    result_full["res"] /= subsample_factor
    result_full["lambda_1d"] *= subsample_factor
    result_full["label"] = f"Full tile ({subsample_factor}x subsampled)"
    result_full["region_size"] = "full"
    results.append(result_full)

    Lc = result_full["spatialScale"]
    ares = result_full["res"]
    A_thresh = 0.5 * Lc**2

    print("\n  Results for full tile:")
    print(f"    Model: {result_full['model']}")
    print(f"    Spatial scale: {Lc:.1f} pixels")
    print(f"    Spatial scale: {Lc * ares:.0f} m = {Lc * ares / 1000:.2f} km")
    print(f"    Accumulation threshold: {A_thresh:.0f} cells")
    print(f"    Processing time: {time.time() - t0:.1f} seconds")

    # -------------------------------------------------------------------------
    # Step 3: Determine best estimate
    # -------------------------------------------------------------------------
    print_section("Step 3: Summary and Best Estimate")

    # Calculate statistics across regions
    spatial_scales_px = [r["spatialScale"] for r in results]
    spatial_scales_m = [r["spatialScale"] * r["res"] for r in results]

    # Use median as best estimate
    best_Lc_px = np.median(spatial_scales_px)
    best_Lc_m = np.median(spatial_scales_m)
    best_A_thresh = 0.5 * best_Lc_px**2

    print("Spatial scale estimates:")
    for r in results:
        Lc = r["spatialScale"]
        ares = r["res"]
        print(f"  {r['label']}: {Lc:.1f} px ({Lc * ares:.0f} m), model={r['model']}")

    print("\nBest estimate (median):")
    print(f"  Lc = {best_Lc_px:.1f} pixels")
    print(f"  Lc = {best_Lc_m:.0f} m = {best_Lc_m / 1000:.2f} km")
    print(f"  A_thresh = {best_A_thresh:.0f} cells")

    # -------------------------------------------------------------------------
    # Step 4: Save outputs
    # -------------------------------------------------------------------------
    print_section("Step 4: Saving Outputs")

    # Save summary JSON
    summary = {
        "input_dem": MERIT_DEM_PATH,
        "max_hillslope_length_m": MAX_HILLSLOPE_LENGTH,
        "nlambda": NLAMBDA,
        "results": [],
    }

    for r in results:
        summary["results"].append(
            {
                "label": r["label"],
                "region_size": r["region_size"],
                "model": r["model"],
                "spatial_scale_px": float(r["spatialScale"]),
                "spatial_scale_m": float(r["spatialScale"] * r["res"]),
                "resolution_m": float(r["res"]),
                "accum_threshold_cells": float(0.5 * r["spatialScale"] ** 2),
            }
        )

    summary["best_estimate"] = {
        "spatial_scale_px": float(best_Lc_px),
        "spatial_scale_m": float(best_Lc_m),
        "accum_threshold_cells": float(best_A_thresh),
    }

    json_path = os.path.join(OUTPUT_DIR, "stage2_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save text summary
    summary_path = os.path.join(OUTPUT_DIR, "stage2_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Stage 2: Spatial Scale Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input DEM: {MERIT_DEM_PATH}\n")
        f.write(f"Max hillslope length: {MAX_HILLSLOPE_LENGTH} m\n\n")

        f.write("Results by region:\n")
        f.write("-" * 40 + "\n")
        for r in results:
            Lc = r["spatialScale"]
            ares = r["res"]
            A_thresh = 0.5 * Lc**2
            f.write(f"\n{r['label']}:\n")
            f.write(f"  Model: {r['model']}\n")
            f.write(f"  Spatial scale: {Lc:.1f} px ({Lc * ares:.0f} m)\n")
            f.write(f"  Accum threshold: {A_thresh:.0f} cells\n")

        f.write("\n\nBest estimate (median):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Lc = {best_Lc_px:.1f} pixels\n")
        f.write(f"  Lc = {best_Lc_m:.0f} m = {best_Lc_m / 1000:.2f} km\n")
        f.write(f"  A_thresh = {best_A_thresh:.0f} cells\n")
        f.write(f"\nTotal processing time: {time.time() - start_time:.1f} seconds\n")

    print(f"  Saved: {summary_path}")

    # Generate plots
    create_spectral_plots(results, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Stage 2 Complete")

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey result for Stage 3:")
    print(f"  Recommended accumulation threshold: {best_A_thresh:.0f} cells")
    print("  (This replaces the fixed threshold of 1000 cells used in Stage 1)")


if __name__ == "__main__":
    main()
