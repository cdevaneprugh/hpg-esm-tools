#!/usr/bin/env python3
"""
Phase C Follow-up: Raw DEM Spectrum Test

Compares three spectral views of the same elevation data:
1. Laplacian spectrum (baseline reproduction — Swenson's standard method)
2. Raw elevation spectrum (no Laplacian — actual energy distribution)
3. k²-corrected Laplacian spectrum (Laplacian amplitude / k² per bin)

The Laplacian amplifies frequency by k² = (2π/λ)², boosting short wavelengths.
This test determines whether the 8m peak found in Phase C is a genuine
topographic signal or appears dominant only due to k² amplification.

Output:
  - 3-panel comparison plot with peak fits and metrics
  - Results table comparing Lc, model, psharp, GoF across all three views
  - Amplitude ratio of 8m vs 200-500m features in each spectrum

Usage:
  python scripts/phase_c_followup_raw_spectrum.py
  python scripts/phase_c_followup_raw_spectrum.py --plot-dir output/osbs/phase_c
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

MOSAIC_PATH = Path(__file__).parent.parent / "data" / "mosaics" / "OSBS_interior.tif"
DEFAULT_PLOT_DIR = Path(__file__).parent.parent / "output" / "osbs" / "phase_c"


def compute_k2_corrected(laplacian_result: dict) -> dict:
    """
    Divide Laplacian binned amplitudes by k² = (2π/λ)² per wavelength bin.

    This approximately recovers the raw elevation spectrum shape, since the
    continuous Laplacian in Fourier space is -k² * Z(k). The discrete
    gradient stencil deviates from ideal k² at high frequencies, so the
    match is approximate — documenting the deviation is itself useful.
    """
    lambda_1d = laplacian_result["lambda_1d"].copy()
    amp_1d = laplacian_result["amp_1d"].copy()
    pixel_size = laplacian_result["pixel_size"]

    # lambda_1d is in pixels; convert to physical wavenumber for k²
    # k = 2π / (λ * pixel_size), but since pixel_size=1m for OSBS,
    # k = 2π / λ in units of rad/pixel
    k = 2 * np.pi / lambda_1d
    k_squared = k**2

    corrected_amp = amp_1d / k_squared

    # Run peak fitting on the corrected spectrum
    max_wavelength = 2 * DEFAULTS["max_hillslope_length"] / pixel_size
    peak_result = locate_peak(lambda_1d, corrected_amp, max_wavelength=max_wavelength)

    lc_pixels = peak_result["spatialScale"]
    lc_meters = lc_pixels * pixel_size

    return {
        "lc_pixels": lc_pixels,
        "lc_meters": lc_meters,
        "pixel_size": pixel_size,
        "model": peak_result["model"],
        "selection": peak_result["selection"],
        "coefs": peak_result["coefs"],
        "psharp_ga": peak_result["psharp_ga"],
        "psharp_ln": peak_result["psharp_ln"],
        "gof_ga": peak_result["gof_ga"],
        "gof_ln": peak_result["gof_ln"],
        "tscore": peak_result["tscore"],
        "lambda_1d": lambda_1d,
        "amp_1d": corrected_amp,
        "use_laplacian": "k2_corrected",
    }


def measure_feature_amplitudes(result: dict) -> dict:
    """
    Measure amplitude of the 8m and 200-500m spectral features.

    Returns dict with amp_8m, amp_200_500m, and their ratio.
    """
    lambda_1d = result["lambda_1d"]
    amp_1d = result["amp_1d"]
    pixel_size = result["pixel_size"]
    wl_meters = lambda_1d * pixel_size

    # 8m feature: bins with wavelength 4-16m (centered on ~8m)
    mask_8m = (wl_meters >= 4) & (wl_meters <= 16)
    amp_8m = np.max(amp_1d[mask_8m]) if np.any(mask_8m) else 0.0

    # 200-500m feature
    mask_200_500 = (wl_meters >= 200) & (wl_meters <= 500)
    amp_200_500 = np.max(amp_1d[mask_200_500]) if np.any(mask_200_500) else 0.0

    ratio = amp_8m / amp_200_500 if amp_200_500 > 0 else float("inf")

    return {
        "amp_8m": amp_8m,
        "amp_200_500m": amp_200_500,
        "ratio_8m_to_200_500m": ratio,
    }


def plot_comparison(
    results: dict[str, dict],
    features: dict[str, dict],
    save_path: Path,
) -> None:
    """3-panel comparison plot: Laplacian, raw elevation, k²-corrected."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        ("laplacian", "Laplacian Spectrum (Swenson standard)"),
        ("raw", "Raw Elevation Spectrum (no Laplacian)"),
        ("k2_corrected", "k²-corrected Laplacian Spectrum"),
    ]

    for ax, (key, title) in zip(axes, panels):
        result = results[key]
        feat = features[key]
        lambda_1d = result["lambda_1d"]
        amp_1d = result["amp_1d"]
        pixel_size = result["pixel_size"]
        wl_meters = lambda_1d * pixel_size
        log_lambda = np.log10(lambda_1d)

        ax.plot(wl_meters, amp_1d, "k-o", ms=3, label="Binned amplitude")

        # Overlay fitted model
        model = result["model"]
        coefs = result["coefs"]
        x_fine = np.linspace(log_lambda[0], log_lambda[-1], 200)
        lambda_fine = 10**x_fine * pixel_size

        if model == "gaussian":
            y_fit = gaussian(x_fine, *coefs)
            peak_wl = 10 ** coefs[1] * pixel_size
            ax.plot(
                lambda_fine, y_fit, "r-", lw=2, label=f"Gaussian (peak={peak_wl:.1f} m)"
            )
            ax.axvline(peak_wl, color="r", ls="--", alpha=0.5)
        elif model == "lognormal":
            y_fit = log_normal(x_fine, *coefs)
            peak_wl = 10 ** np.exp(coefs[2]) * pixel_size
            ax.plot(
                lambda_fine,
                y_fit,
                "b-",
                lw=2,
                label=f"Lognormal (peak={peak_wl:.1f} m)",
            )
            ax.axvline(peak_wl, color="b", ls="--", alpha=0.5)
        elif model == "linear":
            y_fit = synth_polynomial(x_fine, np.array(coefs))
            ax.plot(lambda_fine, y_fit, "g-", lw=2, label="Linear trend")

        ax.set_xscale("log")
        ax.set_xlabel("Wavelength (m)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

        # Annotation
        info_text = (
            f"Lc = {result['lc_meters']:.1f} m\n"
            f"Model: {model} (sel={result['selection']})\n"
            f"psharp: Ga={result['psharp_ga']:.2f}, Ln={result['psharp_ln']:.2f}\n"
            f"GoF: Ga={result['gof_ga']:.2e}, Ln={result['gof_ln']:.2e}\n"
            f"T-score: {result['tscore']:.2f}\n"
            f"---\n"
            f"Amp(8m): {feat['amp_8m']:.3e}\n"
            f"Amp(200-500m): {feat['amp_200_500m']:.3e}\n"
            f"Ratio(8m/200-500m): {feat['ratio_8m_to_200_500m']:.2f}"
        )
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            fontfamily="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )

    fig.suptitle(
        "Phase C Follow-up: Raw DEM Spectrum Test — OSBS Interior (1m)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"  Saved comparison plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase C Follow-up: Raw Spectrum Test")
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=MOSAIC_PATH,
        help="Path to interior mosaic GeoTIFF",
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

    # ---- Load mosaic ----
    log(f"Reading mosaic: {args.mosaic}")
    with rasterio.open(args.mosaic) as src:
        elev = src.read(1).astype(np.float64)
        pixel_size = src.res[0]
        crs = src.crs
        nodata = src.nodata

    log(f"  Shape: {elev.shape}, pixel_size: {pixel_size}m, CRS: {crs}")

    if nodata is not None:
        elev[elev == nodata] = -9999

    # Shared parameters
    params = {
        "pixel_size": pixel_size,
        "blend_edges_n": DEFAULTS["blend_edges"],
        "zero_edges_n": DEFAULTS["zero_edges"],
        "nlambda": DEFAULTS["nlambda"],
        "max_hillslope_length": DEFAULTS["max_hillslope_length"],
        "detrend": DEFAULTS["detrend"],
    }

    # ---- 1. Laplacian spectrum (baseline reproduction) ----
    log("")
    log("=" * 60)
    log("1. LAPLACIAN SPECTRUM (Swenson standard)")
    log("=" * 60)
    laplacian_result = compute_lc(elev, use_laplacian=True, **params)

    # ---- 2. Raw elevation spectrum ----
    log("")
    log("=" * 60)
    log("2. RAW ELEVATION SPECTRUM (no Laplacian)")
    log("=" * 60)
    raw_result = compute_lc(elev, use_laplacian=False, **params)

    # ---- 3. k²-corrected Laplacian spectrum ----
    log("")
    log("=" * 60)
    log("3. k²-CORRECTED LAPLACIAN SPECTRUM")
    log("=" * 60)
    k2_result = compute_k2_corrected(laplacian_result)
    log(
        f"  Result: model={k2_result['model']}, "
        f"Lc={k2_result['lc_pixels']:.1f} px ({k2_result['lc_meters']:.1f} m), "
        f"selection={k2_result['selection']}"
    )

    # ---- Feature amplitude measurements ----
    results = {
        "laplacian": laplacian_result,
        "raw": raw_result,
        "k2_corrected": k2_result,
    }
    features = {key: measure_feature_amplitudes(r) for key, r in results.items()}

    # ---- Results table ----
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)
    header = (
        f"{'Spectrum':<25} {'Lc(m)':<8} {'Model':<12} {'psharp_ga':<11} "
        f"{'psharp_ln':<11} {'GoF_ga':<12} {'GoF_ln':<12} {'T-score':<8}"
    )
    print(header)
    print("-" * 100)

    for key, label in [
        ("laplacian", "Laplacian (standard)"),
        ("raw", "Raw elevation"),
        ("k2_corrected", "k²-corrected Laplacian"),
    ]:
        r = results[key]
        print(
            f"{label:<25} {r['lc_meters']:<8.1f} {r['model']:<12} "
            f"{r['psharp_ga']:<11.3f} {r['psharp_ln']:<11.3f} "
            f"{r['gof_ga']:<12.4e} {r['gof_ln']:<12.4e} {r['tscore']:<8.3f}"
        )

    # ---- Feature amplitude comparison ----
    print("\n" + "=" * 100)
    print("FEATURE AMPLITUDE COMPARISON")
    print("=" * 100)
    print(
        f"{'Spectrum':<25} {'Amp(8m)':<14} {'Amp(200-500m)':<16} {'Ratio(8m/200-500m)':<20}"
    )
    print("-" * 75)

    for key, label in [
        ("laplacian", "Laplacian (standard)"),
        ("raw", "Raw elevation"),
        ("k2_corrected", "k²-corrected Laplacian"),
    ]:
        f = features[key]
        ratio_str = (
            f"{f['ratio_8m_to_200_500m']:.2f}"
            if f["ratio_8m_to_200_500m"] != float("inf")
            else "inf"
        )
        print(
            f"{label:<25} {f['amp_8m']:<14.4e} {f['amp_200_500m']:<16.4e} {ratio_str:<20}"
        )

    # ---- Interpretation ----
    raw_feat = features["raw"]
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    ratio = raw_feat["ratio_8m_to_200_500m"]
    if ratio < 0.5:
        scenario = "A"
        desc = (
            "200-500m feature dominates raw spectrum, 8m is small/absent.\n"
            "  The 8m Laplacian peak is primarily a k² amplification artifact.\n"
            "  Drainage-scale Lc is likely in the 200-500m range."
        )
    elif ratio < 2.0:
        scenario = "B"
        desc = (
            "Both features present in raw spectrum, 200-500m dominates.\n"
            "  8m is real topographic structure but weaker than drainage-scale feature.\n"
            "  Laplacian k² amplification inverted their relative prominence.\n"
            "  The raw spectrum's peak is likely the better Lc for hillslope delineation."
        )
    else:
        scenario = "C"
        desc = (
            "8m feature dominates even without k² amplification.\n"
            "  OSBS genuinely has strong ~8m topographic periodicity.\n"
            "  Needs visual/physical validation (DEM inspection, aerial imagery)."
        )

    print(f"Scenario {scenario}: {desc}")
    print(f"\nRaw spectrum 8m/200-500m amplitude ratio: {ratio:.2f}")
    print(f"Raw spectrum peak (Lc): {raw_result['lc_meters']:.1f} m")
    print(f"Laplacian spectrum peak (Lc): {laplacian_result['lc_meters']:.1f} m")

    # k² correction consistency check
    print("\n--- k² correction consistency ---")
    k2_feat = features["k2_corrected"]
    raw_ratio = raw_feat["ratio_8m_to_200_500m"]
    k2_ratio = k2_feat["ratio_8m_to_200_500m"]
    print(f"Raw spectrum 8m/200-500m ratio:         {raw_ratio:.2f}")
    print(f"k²-corrected Laplacian 8m/200-500m ratio: {k2_ratio:.2f}")
    if raw_ratio > 0 and k2_ratio > 0 and raw_ratio != float("inf"):
        agreement = min(raw_ratio, k2_ratio) / max(raw_ratio, k2_ratio)
        print(f"Agreement: {agreement:.1%}")
        if agreement > 0.5:
            print(
                "  k² correction matches raw spectrum well — discrete stencil ≈ ideal k²"
            )
        else:
            print(
                "  k² correction deviates from raw spectrum — "
                "discrete gradient chain transfer function differs from ideal k²"
            )

    # ---- Save comparison plot ----
    plot_comparison(results, features, plot_dir / "raw_spectrum_comparison.png")

    log("\nPhase C follow-up (raw spectrum) complete.")


if __name__ == "__main__":
    main()
