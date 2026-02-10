#!/usr/bin/env python3
"""
Phase C Follow-up: Restricted Wavelength Sweep

Three tests on clean, contiguous data:

Part A — Single-tile FFT (mosaic artifact check)
  Run Laplacian and raw spectra on tile R6C10 (1000x1000, 0% nodata).
  Confirms spectral features are intrinsic to the terrain, not from
  mosaic stitching or nodata fill.

Part B — Contiguous mosaic FFT (clean baseline)
  Run Laplacian and raw spectra on R4-R12, C5-C14 (9000x10000, 0% nodata).
  Hard verification gate: script aborts if ANY nodata pixel exists.

Part C — Restricted wavelength sweep (the key test)
  For cutoff wavelengths [10, 20, 50, 100, 180, 300, 500] m, exclude short
  wavelengths from the peak fitting and re-run model selection. Tests whether
  the 200-500m Laplacian hump becomes a proper spectral peak when micro-
  topographic wavelengths are excluded — mimicking what 90m resolution does
  implicitly.

Output:
  - 2x2 comparison plot (single tile vs mosaic, Laplacian vs raw)
  - Restricted wavelength sweep table and plot
  - Best-candidate spectra at cutoffs where peaked models fit
  All saved to output/osbs/phase_c/

Usage:
  python scripts/phase_c_followup_restricted_wavelength.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from phase_c_lc_analysis import (
    DEFAULTS,
    compute_lc,
    gaussian,
    locate_peak,
    log,
    log_normal,
    synth_polynomial,
)

# ============================================================================
# Paths
# ============================================================================

SWENSON_DIR = Path(__file__).parent.parent
MOSAIC_PATH = SWENSON_DIR / "data" / "mosaics" / "OSBS_interior.tif"
TILE_R6C10 = (
    SWENSON_DIR / "data" / "neon" / "dtm" / "NEON_D03_OSBS_DP3_404000_3286000_DTM.tif"
)
DEFAULT_PLOT_DIR = SWENSON_DIR / "output" / "osbs" / "phase_c"

# Contiguous block: R4-R12, C5-C14 in mosaic pixel coordinates
# Mosaic origin: (395000E, 3292000N), rows increase southward
CLEAN_BLOCK_ROW_SLICE = slice(3000, 12000)  # rows 3000-11999 (9000 px)
CLEAN_BLOCK_COL_SLICE = slice(4000, 14000)  # cols 4000-13999 (10000 px)

# Restricted wavelength sweep cutoffs (meters)
CUTOFF_WAVELENGTHS = [10, 20, 50, 100, 180, 300, 500]

# Shared FFT parameters (defaults from phase_c_lc_analysis)
FFT_PARAMS = {
    "blend_edges_n": DEFAULTS["blend_edges"],
    "zero_edges_n": DEFAULTS["zero_edges"],
    "nlambda": DEFAULTS["nlambda"],
    "max_hillslope_length": DEFAULTS["max_hillslope_length"],
    "detrend": DEFAULTS["detrend"],
}


# ============================================================================
# Part A: Single-tile FFT
# ============================================================================


def run_single_tile(tile_path: Path) -> dict:
    """Run Laplacian and raw FFT on a single tile."""
    log(f"Loading tile: {tile_path.name}")
    with rasterio.open(tile_path) as src:
        elev = src.read(1).astype(np.float64)
        pixel_size = src.res[0]
        nodata = src.nodata

    if nodata is not None:
        elev[elev == nodata] = -9999

    log(f"  Shape: {elev.shape}, pixel_size: {pixel_size}m")

    # Check for nodata
    valid = elev > -9000
    n_nodata = np.sum(~valid)
    if n_nodata > 0:
        log(f"  WARNING: {n_nodata} nodata pixels ({n_nodata / elev.size:.2%})")

    log("  Running Laplacian spectrum...")
    laplacian = compute_lc(
        elev, pixel_size=pixel_size, use_laplacian=True, **FFT_PARAMS
    )

    log("  Running raw elevation spectrum...")
    raw = compute_lc(elev, pixel_size=pixel_size, use_laplacian=False, **FFT_PARAMS)

    return {"laplacian": laplacian, "raw": raw, "shape": elev.shape}


# ============================================================================
# Part B: Contiguous mosaic FFT
# ============================================================================


def run_contiguous_mosaic(mosaic_path: Path) -> dict:
    """Run Laplacian and raw FFT on the verified-clean mosaic block."""
    log(f"Loading mosaic: {mosaic_path}")
    with rasterio.open(mosaic_path) as src:
        elev_full = src.read(1).astype(np.float64)
        pixel_size = src.res[0]
        nodata = src.nodata

    if nodata is not None:
        elev_full[elev_full == nodata] = -9999

    # Extract contiguous block
    elev = elev_full[CLEAN_BLOCK_ROW_SLICE, CLEAN_BLOCK_COL_SLICE]
    log(f"  Extracted block: {elev.shape} (R4-R12, C5-C14)")

    # Hard verification gate
    valid = elev > -9000
    n_nodata = np.sum(~valid)
    pct_nodata = n_nodata / elev.size * 100
    log(f"  Nodata pixels: {n_nodata} ({pct_nodata:.4f}%)")

    if n_nodata > 0:
        raise RuntimeError(
            f"ABORT: Contiguous block has {n_nodata} nodata pixels "
            f"({pct_nodata:.4f}%). Expected 0. "
            f"The R4-R12, C5-C14 block is not clean — investigate before proceeding."
        )

    log("  Verification passed: 0 nodata pixels.")
    log(f"  Elevation range: {elev.min():.2f} - {elev.max():.2f} m")

    log("  Running Laplacian spectrum...")
    laplacian = compute_lc(
        elev, pixel_size=pixel_size, use_laplacian=True, **FFT_PARAMS
    )

    log("  Running raw elevation spectrum...")
    raw = compute_lc(elev, pixel_size=pixel_size, use_laplacian=False, **FFT_PARAMS)

    return {"laplacian": laplacian, "raw": raw, "shape": elev.shape}


# ============================================================================
# Part C: Restricted wavelength sweep
# ============================================================================


def run_restricted_sweep(mosaic_laplacian: dict) -> list[dict]:
    """
    Re-run peak fitting on the mosaic Laplacian spectrum with short wavelengths
    excluded.

    For each cutoff, filter lambda_1d and amp_1d to wavelengths >= cutoff,
    then run locate_peak on the filtered spectrum.
    """
    lambda_1d = mosaic_laplacian["lambda_1d"]
    amp_1d = mosaic_laplacian["amp_1d"]
    pixel_size = mosaic_laplacian["pixel_size"]
    max_wavelength = 2 * DEFAULTS["max_hillslope_length"] / pixel_size

    wl_meters = lambda_1d * pixel_size

    results = []
    for cutoff in CUTOFF_WAVELENGTHS:
        mask = wl_meters >= cutoff
        n_bins = np.sum(mask)

        if n_bins < 4:
            log(f"  Cutoff {cutoff}m: only {n_bins} bins remaining — skipping")
            results.append(
                {
                    "cutoff_m": cutoff,
                    "n_bins": int(n_bins),
                    "lc_m": float("nan"),
                    "model": "insufficient_data",
                    "psharp_ga": 0.0,
                    "psharp_ln": 0.0,
                    "gof_ga": 0.0,
                    "gof_ln": 0.0,
                    "tscore": 0.0,
                    "selection": -1,
                    "coefs": [],
                    "lambda_1d": lambda_1d[mask],
                    "amp_1d": amp_1d[mask],
                }
            )
            continue

        filtered_lambda = lambda_1d[mask]
        filtered_amp = amp_1d[mask]

        peak_result = locate_peak(filtered_lambda, filtered_amp, max_wavelength)

        lc_m = peak_result["spatialScale"] * pixel_size

        log(
            f"  Cutoff {cutoff:>4d}m: {n_bins:>2d} bins, "
            f"Lc={lc_m:>8.1f}m, model={peak_result['model']:<10s}, "
            f"psharp(Ga/Ln)={peak_result['psharp_ga']:.2f}/{peak_result['psharp_ln']:.2f}"
        )

        results.append(
            {
                "cutoff_m": cutoff,
                "n_bins": int(n_bins),
                "lc_m": lc_m,
                "model": peak_result["model"],
                "psharp_ga": peak_result["psharp_ga"],
                "psharp_ln": peak_result["psharp_ln"],
                "gof_ga": peak_result["gof_ga"],
                "gof_ln": peak_result["gof_ln"],
                "tscore": peak_result["tscore"],
                "selection": peak_result["selection"],
                "coefs": peak_result["coefs"],
                "lambda_1d": filtered_lambda,
                "amp_1d": filtered_amp,
            }
        )

    return results


# ============================================================================
# Plotting
# ============================================================================


def _plot_spectrum_panel(
    ax: plt.Axes,
    result: dict,
    title: str,
) -> None:
    """Plot a single spectrum panel with fitted model overlay."""
    lambda_1d = result["lambda_1d"]
    amp_1d = result["amp_1d"]
    pixel_size = result["pixel_size"]
    wl_meters = lambda_1d * pixel_size
    log_lambda = np.log10(lambda_1d)

    ax.plot(wl_meters, amp_1d, "k-o", ms=3, label="Binned amplitude")

    model = result["model"]
    coefs = result["coefs"]
    x_fine = np.linspace(log_lambda[0], log_lambda[-1], 200)
    lambda_fine = 10**x_fine * pixel_size

    if model == "gaussian":
        y_fit = gaussian(x_fine, *coefs)
        peak_wl = 10 ** coefs[1] * pixel_size
        ax.plot(lambda_fine, y_fit, "r-", lw=2, label=f"Gaussian ({peak_wl:.1f} m)")
        ax.axvline(peak_wl, color="r", ls="--", alpha=0.5)
    elif model == "lognormal":
        y_fit = log_normal(x_fine, *coefs)
        peak_wl = 10 ** np.exp(coefs[2]) * pixel_size
        ax.plot(lambda_fine, y_fit, "b-", lw=2, label=f"Lognormal ({peak_wl:.1f} m)")
        ax.axvline(peak_wl, color="b", ls="--", alpha=0.5)
    elif model == "linear":
        y_fit = synth_polynomial(x_fine, np.array(coefs))
        ax.plot(lambda_fine, y_fit, "g-", lw=2, label="Linear trend")

    ax.set_xscale("log")
    ax.set_xlabel("Wavelength (m)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")

    info = (
        f"Lc = {result['lc_meters']:.1f} m\n"
        f"Model: {model} (sel={result['selection']})\n"
        f"psharp: Ga={result['psharp_ga']:.2f}, Ln={result['psharp_ln']:.2f}\n"
        f"T-score: {result['tscore']:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        fontfamily="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )


def plot_tile_vs_mosaic(
    tile: dict,
    mosaic: dict,
    save_path: Path,
) -> None:
    """2x2 comparison: tile vs mosaic, Laplacian vs raw."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_spectrum_panel(
        axes[0, 0],
        tile["laplacian"],
        f"Single tile R6C10 — Laplacian ({tile['shape'][0]}x{tile['shape'][1]})",
    )
    _plot_spectrum_panel(
        axes[0, 1],
        tile["raw"],
        f"Single tile R6C10 — Raw elevation ({tile['shape'][0]}x{tile['shape'][1]})",
    )
    _plot_spectrum_panel(
        axes[1, 0],
        mosaic["laplacian"],
        f"Contiguous mosaic — Laplacian ({mosaic['shape'][0]}x{mosaic['shape'][1]})",
    )
    _plot_spectrum_panel(
        axes[1, 1],
        mosaic["raw"],
        f"Contiguous mosaic — Raw elevation ({mosaic['shape'][0]}x{mosaic['shape'][1]})",
    )

    fig.suptitle(
        "Phase C Follow-up: Single Tile vs Contiguous Mosaic Spectra (OSBS, 1m)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"  Saved: {save_path}")


def plot_restricted_sweep(
    sweep_results: list[dict],
    save_path: Path,
) -> None:
    """Lc vs cutoff wavelength, color-coded by model type."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    model_colors = {
        "gaussian": "red",
        "lognormal": "blue",
        "linear": "green",
        "flat": "gray",
        "insufficient_data": "white",
    }
    model_markers = {
        "gaussian": "o",
        "lognormal": "s",
        "linear": "^",
        "flat": "x",
        "insufficient_data": ".",
    }

    for r in sweep_results:
        if r["model"] == "insufficient_data":
            continue
        color = model_colors.get(r["model"], "black")
        marker = model_markers.get(r["model"], "o")
        ax.scatter(
            r["cutoff_m"],
            r["lc_m"],
            c=color,
            marker=marker,
            s=100,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    # Connect with line
    valid = [r for r in sweep_results if r["model"] != "insufficient_data"]
    if valid:
        cutoffs = [r["cutoff_m"] for r in valid]
        lcs = [r["lc_m"] for r in valid]
        ax.plot(cutoffs, lcs, "k--", alpha=0.3, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Minimum wavelength cutoff (m)")
    ax.set_ylabel("Lc (m)")
    ax.set_title("Restricted Wavelength Sweep: Lc vs Cutoff")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            ms=10,
            label="Lognormal",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            ms=10,
            label="Gaussian",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="green",
            ms=10,
            label="Linear",
        ),
        Line2D(
            [0], [0], marker="x", color="w", markerfacecolor="gray", ms=10, label="Flat"
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")

    # Annotate each point with model and psharp
    for r in sweep_results:
        if r["model"] == "insufficient_data":
            continue
        psharp = max(r["psharp_ga"], r["psharp_ln"])
        label = f"{r['model'][:3]}\np={psharp:.1f}"
        ax.annotate(
            label,
            (r["cutoff_m"], r["lc_m"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=6,
            fontfamily="monospace",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"  Saved: {save_path}")


def plot_best_candidate_spectra(
    sweep_results: list[dict],
    pixel_size: float,
    save_path: Path,
) -> None:
    """
    Plot the Laplacian spectrum at cutoffs where a peaked model (Gaussian or
    Lognormal) fits, showing the fitted curve.
    """
    peaked = [r for r in sweep_results if r["model"] in ("gaussian", "lognormal")]

    if not peaked:
        log("  No peaked models found in sweep — skipping best-candidate plot.")
        return

    n_panels = len(peaked)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )

    for idx, r in enumerate(peaked):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        lambda_1d = r["lambda_1d"]
        amp_1d = r["amp_1d"]
        wl_meters = lambda_1d * pixel_size
        log_lambda = np.log10(lambda_1d)

        ax.plot(wl_meters, amp_1d, "k-o", ms=4, label="Binned amplitude")

        # Fitted model overlay
        coefs = r["coefs"]
        x_fine = np.linspace(log_lambda[0], log_lambda[-1], 200)
        lambda_fine = 10**x_fine * pixel_size

        if r["model"] == "gaussian":
            y_fit = gaussian(x_fine, *coefs)
            peak_wl = 10 ** coefs[1] * pixel_size
            ax.plot(lambda_fine, y_fit, "r-", lw=2, label=f"Gaussian ({peak_wl:.1f} m)")
            ax.axvline(peak_wl, color="r", ls="--", alpha=0.5)
        elif r["model"] == "lognormal":
            y_fit = log_normal(x_fine, *coefs)
            peak_wl = 10 ** np.exp(coefs[2]) * pixel_size
            ax.plot(
                lambda_fine, y_fit, "b-", lw=2, label=f"Lognormal ({peak_wl:.1f} m)"
            )
            ax.axvline(peak_wl, color="b", ls="--", alpha=0.5)

        ax.set_xscale("log")
        ax.set_xlabel("Wavelength (m)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Cutoff >= {r['cutoff_m']}m (Lc = {r['lc_m']:.1f}m)")
        ax.legend(fontsize=8)

        psharp = max(r["psharp_ga"], r["psharp_ln"])
        info = (
            f"Model: {r['model']}\n"
            f"psharp: {psharp:.2f}\n"
            f"GoF(Ga): {r['gof_ga']:.2e}\n"
            f"GoF(Ln): {r['gof_ln']:.2e}\n"
            f"n_bins: {r['n_bins']}"
        )
        ax.text(
            0.02,
            0.98,
            info,
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            fontfamily="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Restricted Wavelength Sweep: Best Candidate Spectra (peaked models only)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"  Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Phase C Follow-up: Restricted Wavelength Sweep"
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory for output plots",
    )
    args = parser.parse_args()

    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Part A: Single-tile FFT
    # ==================================================================
    log("")
    log("=" * 70)
    log("PART A: Single-tile FFT (R6C10 — mosaic artifact check)")
    log("=" * 70)

    tile_results = run_single_tile(TILE_R6C10)

    # ==================================================================
    # Part B: Contiguous mosaic FFT
    # ==================================================================
    log("")
    log("=" * 70)
    log("PART B: Contiguous mosaic FFT (R4-R12, C5-C14 — clean baseline)")
    log("=" * 70)

    mosaic_results = run_contiguous_mosaic(MOSAIC_PATH)

    # ==================================================================
    # Part A+B comparison
    # ==================================================================
    print("\n" + "=" * 90)
    print("PART A+B: SINGLE TILE vs CONTIGUOUS MOSAIC")
    print("=" * 90)
    print(
        f"{'Source':<30} {'Spectrum':<12} {'Lc(m)':<8} {'Model':<12} "
        f"{'psharp_ga':<11} {'psharp_ln':<11} {'T-score':<8}"
    )
    print("-" * 92)
    for label, data in [
        ("Tile R6C10 (1x1 km)", tile_results),
        ("Mosaic R4-R12,C5-C14 (9x10 km)", mosaic_results),
    ]:
        for spec_type in ["laplacian", "raw"]:
            r = data[spec_type]
            print(
                f"{label:<30} {spec_type:<12} {r['lc_meters']:<8.1f} {r['model']:<12} "
                f"{r['psharp_ga']:<11.3f} {r['psharp_ln']:<11.3f} {r['tscore']:<8.3f}"
            )

    # Save comparison plot
    plot_tile_vs_mosaic(
        tile_results,
        mosaic_results,
        plot_dir / "tile_vs_mosaic_comparison.png",
    )

    # ==================================================================
    # Part C: Restricted wavelength sweep
    # ==================================================================
    log("")
    log("=" * 70)
    log("PART C: Restricted wavelength sweep (contiguous mosaic Laplacian)")
    log("=" * 70)

    sweep_results = run_restricted_sweep(mosaic_results["laplacian"])

    # Print sweep table
    print("\n" + "=" * 100)
    print("RESTRICTED WAVELENGTH SWEEP RESULTS")
    print("=" * 100)
    print(
        f"{'Cutoff(m)':<12} {'n_bins':<8} {'Lc(m)':<10} {'Model':<14} "
        f"{'psharp_ga':<11} {'psharp_ln':<11} {'GoF_ga':<12} {'GoF_ln':<12} {'T-score':<8}"
    )
    print("-" * 100)

    for r in sweep_results:
        lc_str = f"{r['lc_m']:.1f}" if not np.isnan(r["lc_m"]) else "N/A"
        print(
            f"{r['cutoff_m']:<12} {r['n_bins']:<8} {lc_str:<10} {r['model']:<14} "
            f"{r['psharp_ga']:<11.3f} {r['psharp_ln']:<11.3f} "
            f"{r['gof_ga']:<12.4e} {r['gof_ln']:<12.4e} {r['tscore']:<8.3f}"
        )

    # Summary interpretation
    peaked = [r for r in sweep_results if r["model"] in ("gaussian", "lognormal")]
    linear = [r for r in sweep_results if r["model"] == "linear"]

    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if peaked:
        print(f"Peaked models found at {len(peaked)} cutoff(s):")
        for r in peaked:
            psharp = max(r["psharp_ga"], r["psharp_ln"])
            print(
                f"  cutoff >= {r['cutoff_m']}m: Lc = {r['lc_m']:.1f}m "
                f"({r['model']}, psharp={psharp:.2f})"
            )
        best = max(peaked, key=lambda r: max(r["psharp_ga"], r["psharp_ln"]))
        best_psharp = max(best["psharp_ga"], best["psharp_ln"])
        print(
            f"\nBest candidate: cutoff >= {best['cutoff_m']}m -> "
            f"Lc = {best['lc_m']:.1f}m ({best['model']}, psharp={best_psharp:.2f})"
        )
    else:
        print("No peaked models found at any cutoff.")

    if linear:
        print(f"\nLinear (no peak) at {len(linear)} cutoff(s):")
        for r in linear:
            print(
                f"  cutoff >= {r['cutoff_m']}m: Lc = {r['lc_m']:.1f}m (T={r['tscore']:.2f})"
            )

    # Save sweep plots
    plot_restricted_sweep(sweep_results, plot_dir / "restricted_wavelength_sweep.png")
    plot_best_candidate_spectra(
        sweep_results,
        mosaic_results["laplacian"]["pixel_size"],
        plot_dir / "restricted_wavelength_best_candidates.png",
    )

    log("\nPhase C follow-up (restricted wavelength) complete.")


if __name__ == "__main__":
    main()
