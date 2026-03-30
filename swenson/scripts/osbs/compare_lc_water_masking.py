#!/usr/bin/env python3
"""
Compare FFT-derived Lc with and without lake masking.

Runs the spatial scale analysis (FFT of DEM Laplacian) three ways:
  1. Raw DEM — current pipeline method, no special lake treatment
  2. Mean-fill — lake pixels replaced with mean land elevation before FFT
  3. Zero Laplacian — Laplacian zeroed at lake pixels before FFT

All three use the same FFT parameters as the production pipeline (1m,
min_wavelength=20m, blend_edges=50, zero_edges=50). The only variable
is how lake pixels are handled.

Usage:
    python compare_lc_water_masking.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Add parent directory for shared module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_scale import identify_spatial_scale_laplacian_dem  # noqa: E402

# =============================================================================
# Constants (mirror run_pipeline.py)
# =============================================================================

PIXEL_SIZE = 1.0  # meters
MIN_WAVELENGTH = 20  # meters
NODATA_THRESHOLD = -9000

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent  # swenson/

MOSAIC_PATH = BASE_DIR / "data" / "mosaics" / "production" / "dtm.tif"
WATER_MASK_PATH = BASE_DIR / "data" / "mosaics" / "production" / "water_mask.tif"
OUTPUT_DIR = BASE_DIR / "output" / "osbs" / "water_masking_comparison"

# FFT parameters matching production pipeline (run_pipeline.py lines 670-677)
FFT_KWARGS = {
    "pixel_size": PIXEL_SIZE,
    "min_wavelength": MIN_WAVELENGTH,
    "blend_edges_n": 50,
    "zero_edges_n": 50,
    "verbose": True,
}


# =============================================================================
# Plotting
# =============================================================================


def create_spectral_comparison_plot(
    methods: dict[str, dict], output_path: Path
) -> None:
    """Overlay spectral curves from all three methods on one figure."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "Raw DEM": "#2c3e50",
        "Mean-fill": "#e74c3c",
        "Zero Laplacian": "#3498db",
    }

    for label, result in methods.items():
        lam = result["lambda_1d"]
        amp = result["laplac_amp_1d"]
        lc = result["spatialScale"]
        lc_m = result["spatialScale_m"]
        color = colors[label]

        ax.semilogy(
            lam, amp, ".-", color=color, linewidth=1.5, markersize=4, label=label
        )
        ax.axvline(
            lc,
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"Lc = {lc:.0f} px ({lc_m:.0f} m)",
        )

    ax.set_xlabel("Wavelength (pixels)")
    ax.set_ylabel("Laplacian Amplitude")
    ax.set_title("OSBS Spatial Scale: Water Masking Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    t_start = time.time()

    # --- Load data ---
    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    print(f"  DEM: {MOSAIC_PATH.name}")
    with rasterio.open(MOSAIC_PATH) as src:
        dem = src.read(1)
    print(f"  Shape: {dem.shape[0]} x {dem.shape[1]} = {dem.size:,} pixels")
    print(f"  Memory: {dem.nbytes / 1e9:.2f} GB")

    n_nodata = int(np.sum(dem < NODATA_THRESHOLD))
    print(f"  Nodata pixels: {n_nodata:,}")

    print(f"\n  Water mask: {WATER_MASK_PATH.name}")
    with rasterio.open(WATER_MASK_PATH) as src:
        water_mask = src.read(1)
    assert dem.shape == water_mask.shape, (
        f"Shape mismatch: DEM {dem.shape}, mask {water_mask.shape}"
    )

    n_water = int(np.sum(water_mask == 1))
    water_frac = n_water / water_mask.size
    print(f"  Water pixels: {n_water:,} ({100 * water_frac:.2f}%)")

    # --- Method 1: Raw DEM ---
    print("\n" + "=" * 60)
    print("Method 1: Raw DEM (current pipeline)")
    print("=" * 60)

    t0 = time.time()
    result_raw = identify_spatial_scale_laplacian_dem(dem, **FFT_KWARGS)
    t_raw = time.time() - t0
    print(f"  Time: {t_raw:.1f} s")

    # --- Method 2: Mean-fill lakes ---
    print("\n" + "=" * 60)
    print("Method 2: Mean-fill lake pixels")
    print("=" * 60)

    dem_filled = dem.copy()
    valid_land = (dem > NODATA_THRESHOLD) & (water_mask == 0)
    mean_elev = float(np.mean(dem[valid_land]))
    dem_filled[water_mask == 1] = mean_elev
    print(f"  Fill value: {mean_elev:.2f} m (mean land elevation)")

    t0 = time.time()
    result_fill = identify_spatial_scale_laplacian_dem(dem_filled, **FFT_KWARGS)
    t_fill = time.time() - t0
    print(f"  Time: {t_fill:.1f} s")
    del dem_filled

    # --- Method 3: Zero Laplacian at lakes ---
    print("\n" + "=" * 60)
    print("Method 3: Zero Laplacian at lake pixels")
    print("=" * 60)

    t0 = time.time()
    result_zero = identify_spatial_scale_laplacian_dem(
        dem, **FFT_KWARGS, laplacian_mask=water_mask
    )
    t_zero = time.time() - t0
    print(f"  Time: {t_zero:.1f} s")

    # --- Comparison table ---
    methods = {
        "Raw DEM": result_raw,
        "Mean-fill": result_fill,
        "Zero Laplacian": result_zero,
    }

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    header = f"{'Method':<18s} {'Lc (m)':>8s} {'Lc (px)':>8s} {'A_thresh':>9s} {'Model':>10s} {'Psharp':>8s}"
    print(header)
    print("-" * len(header))

    for label, r in methods.items():
        lc_m = r["spatialScale_m"]
        lc_px = r["spatialScale"]
        a_thresh = int(0.5 * lc_px**2)
        model = r["model"]
        psharp = max(r["psharp_ga"], r["psharp_ln"])
        print(
            f"{label:<18s} {lc_m:>8.1f} {lc_px:>8.1f} {a_thresh:>9d} {model:>10s} {psharp:>8.2f}"
        )

    print(f"\nTotal time: {time.time() - t_start:.1f} s")

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Spectral plot
    plot_path = OUTPUT_DIR / "spectral_comparison.png"
    create_spectral_comparison_plot(methods, plot_path)
    print(f"\nSaved: {plot_path}")

    # JSON results
    results_json = {
        "metadata": {
            "description": (
                "Lc comparison: effect of NWI lake masking on FFT "
                "characteristic length scale"
            ),
            "dem_path": str(MOSAIC_PATH),
            "water_mask_path": str(WATER_MASK_PATH),
            "dem_shape": list(dem.shape),
            "water_pixels": n_water,
            "water_fraction": round(water_frac, 4),
            "pixel_size_m": PIXEL_SIZE,
            "min_wavelength_m": MIN_WAVELENGTH,
            "timestamp": datetime.now().isoformat(),
        },
        "methods": {},
    }

    for label, r in methods.items():
        lc_px = r["spatialScale"]
        results_json["methods"][label] = {
            "Lc_m": float(r["spatialScale_m"]),
            "Lc_px": float(lc_px),
            "A_thresh": int(0.5 * lc_px**2),
            "model": r["model"],
            "selection": int(r["selection"]),
            "psharp_ga": float(r["psharp_ga"]),
            "psharp_ln": float(r["psharp_ln"]),
            "gof_ga": float(r["gof_ga"]),
            "gof_ln": float(r["gof_ln"]),
            "tscore": float(r["tscore"]),
            "spectra": {
                "lambda_px": r["lambda_1d"].tolist(),
                "amplitude": r["laplac_amp_1d"].tolist(),
            },
        }

    json_path = OUTPUT_DIR / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
