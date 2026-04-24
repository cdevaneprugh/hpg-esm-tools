#!/usr/bin/env python3
"""
Diagnose HAND bin 1 contamination from conditioning artifacts.

Loads intermediate arrays saved by run_pipeline.py when SAVE_DIAGNOSTICS=1
and produces six plots plus a summary JSON. The plots answer:

1. Fill decomposition  — separates the four conditioning stages so
   depression_fill_depth (the real contamination signal) can be seen cleanly.
2. HAND distribution   — overall shape with bin boundaries overlaid.
3. Bin 1 DTND          — the actual DTND distribution of pixels that land in
   the lowest HAND bin (becomes col 2's stats post-lake-insertion).
4. Bin 1 DTND split    — overlays clean vs. depression-filled subpopulations
   in bin 1. Indicates whether filtering would change col 2's DTND stats.
5. Spatial contamination map — 2D image of depression_fill_depth with NWI
   outlines. Shows whether contamination clusters inside NWI (expected) or
   outside (unmapped wetlands).
6. NWI-interior depression_fill — compares modeled basin depth inside NWI
   polygons to Lee et al. 2023's measured mean spill depth (2.64m).

See swenson/docs/lake-column-ctsm-audit.md Section 6 for methodology.

Usage:
    python scripts/osbs/diagnose_hand_contamination.py <run_dir>

Example:
    python scripts/osbs/diagnose_hand_contamination.py \\
        output/osbs/2026-04-24_diagnostic

Expects <run_dir>/diagnostics/ to contain the 8 .npy files written by
run_pipeline.py with SAVE_DIAGNOSTICS=1. Writes plots and summary.json to
<run_dir>/diagnostics/plots/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Import bin-boundary function for consistency with production pipeline.
# Done via sys.path manipulation matching run_pipeline.py's pattern.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hillslope_params import compute_hand_bins_hybrid  # noqa: E402


# Reference constants ---------------------------------------------------------

LEE_2023_MEAN_SPILL_DEPTH_M = 2.641  # OSBS mean spill depth ± 0.950m (Table 1)
WATER_LOWERING_M = 0.1  # pipeline's NWI water-pixel lowering amount
BIN1_UPPER_M = 0.1  # HAND upper bound for bin 1 (lowest bin in hybrid scheme)

# Plot styling — match run_pipeline.py conventions
DPI = 150
HIST_COLOR_PIT = "#3498db"  # blue
HIST_COLOR_DEP = "#e74c3c"  # red — the signal of interest
HIST_COLOR_RES = "#2ecc71"  # green
HIST_COLOR_CLEAN = "#2c3e50"  # dark — clean baseline
HIST_COLOR_DIRTY = "#e74c3c"  # red — contaminated subpopulation


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_diagnostic_arrays(diag_dir: Path) -> dict:
    """Load the 8 .npy files saved by run_pipeline.py SAVE_DIAGNOSTICS mode."""
    required = [
        "dem",
        "pit_filled",
        "flooded_orig",
        "water_mask",
        "inflated",
        "hand",
        "dtnd",
        "wide_channel_mask",
    ]
    missing = [name for name in required if not (diag_dir / f"{name}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing diagnostic arrays in {diag_dir}: {missing}. "
            f"Re-run pipeline with SAVE_DIAGNOSTICS=1."
        )

    arrs = {name: np.load(diag_dir / f"{name}.npy") for name in required}

    # Shape consistency check
    ref_shape = arrs["dem"].shape
    for name, arr in arrs.items():
        if arr.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: {name} is {arr.shape}, expected {ref_shape}"
            )
    print(f"Loaded 8 arrays of shape {ref_shape}")
    return arrs


def decompose_fill(arrs: dict) -> dict:
    """Compute per-stage fill contributions."""
    dem = arrs["dem"]
    pit_filled = arrs["pit_filled"]
    flooded_orig = arrs["flooded_orig"]
    inflated = arrs["inflated"]
    water_mask = arrs["water_mask"]

    # Reconstruct the water-lowered flooded DEM that resolve_flats saw
    flooded_lowered = flooded_orig - WATER_LOWERING_M * (water_mask > 0)

    return {
        "pit_fill_depth": pit_filled - dem,
        "depression_fill_depth": flooded_orig - pit_filled,
        "resolve_flat_depth": inflated - flooded_lowered,
    }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_fill_decomposition(decomp: dict, water_mask: np.ndarray, out_path: Path):
    """Plot 1: Three-panel fill-depth histogram for non-water pixels."""
    land = (water_mask == 0).flatten()
    pit = decomp["pit_fill_depth"].flatten()[land]
    dep = decomp["depression_fill_depth"].flatten()[land]
    res = decomp["resolve_flat_depth"].flatten()[land]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, arr, color, title in [
        (axes[0], pit, HIST_COLOR_PIT, "pit_fill_depth (Stage 1)"),
        (axes[1], dep, HIST_COLOR_DEP, "depression_fill_depth (Stage 2)"),
        (axes[2], res, HIST_COLOR_RES, "resolve_flat_depth (Stage 4)"),
    ]:
        positive = arr[arr > 0]
        n_zero = int(np.sum(arr == 0))
        n_pos = int(positive.size)
        if positive.size > 0:
            max_val = np.max(positive)
            if max_val <= 0:
                bins = np.linspace(0, 1, 50)
            else:
                bins = np.logspace(
                    np.log10(max(1e-6, np.min(positive))), np.log10(max_val), 60
                )
            ax.hist(positive, bins=bins, color=color, alpha=0.75, edgecolor="black")
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Fill depth (m)")
        ax.set_ylabel("Pixel count (log)")
        ax.set_title(
            f"{title}\nzero: {n_zero:,} ({100 * n_zero / land.sum():.1f}%)  "
            f"positive: {n_pos:,}"
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fill-Depth Decomposition (Non-Water Land Pixels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hand_histogram(
    hand: np.ndarray, water_mask: np.ndarray, bin_edges: np.ndarray, out_path: Path
):
    """Plot 2: HAND histogram (non-water) with bin boundaries overlaid."""
    land = (water_mask == 0) & np.isfinite(hand) & (hand > 0)
    hand_vals = hand[land]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

    # Full range view
    for ax, hi, title in [
        (ax_full, np.max(hand_vals), "Full HAND range"),
        (ax_zoom, 2.0, "HAND 0-2m (TAI zone)"),
    ]:
        hist_bins = np.linspace(0, hi, 200)
        ax.hist(
            hand_vals, bins=hist_bins, color="#3498db", alpha=0.75, edgecolor="none"
        )
        ax.set_yscale("log")
        ax.set_xlim(0, hi)
        ax.set_xlabel("HAND (m)")
        ax.set_ylabel("Pixel count (log)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Overlay bin boundaries
        for bound in bin_edges:
            if bound <= hi:
                ax.axvline(bound, color="red", linewidth=0.6, alpha=0.6)

    fig.suptitle(
        f"HAND Distribution with {len(bin_edges) - 1} Bin Boundaries "
        f"(Non-Water, {hand_vals.size:,} pixels)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bin1_dtnd(
    hand: np.ndarray, dtnd: np.ndarray, water_mask: np.ndarray, out_path: Path
):
    """Plot 3: DTND histogram for pixels with HAND in [0, 0.1m]."""
    in_bin1 = (
        (water_mask == 0) & np.isfinite(hand) & (hand > 0) & (hand <= BIN1_UPPER_M)
    )
    dtnd_bin1 = dtnd[in_bin1]

    fig, ax = plt.subplots(figsize=(10, 6))
    if dtnd_bin1.size > 0:
        p995 = np.percentile(dtnd_bin1, 99.5)
        bins = np.linspace(0, p995, 100)
        ax.hist(
            dtnd_bin1, bins=bins, color=HIST_COLOR_CLEAN, alpha=0.8, edgecolor="none"
        )
        ax.axvline(
            np.median(dtnd_bin1),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"median = {np.median(dtnd_bin1):.1f} m",
        )
        ax.axvline(
            np.mean(dtnd_bin1),
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"mean = {np.mean(dtnd_bin1):.1f} m",
        )
        ax.legend()
    ax.set_xlabel("DTND (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title(
        f"Bin 1 DTND Distribution (HAND <= {BIN1_UPPER_M}m, "
        f"{dtnd_bin1.size:,} pixels)\n"
        "This is col 2's actual DTND distribution post-lake-insertion"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bin1_dtnd_split(
    hand: np.ndarray,
    dtnd: np.ndarray,
    depression_fill: np.ndarray,
    water_mask: np.ndarray,
    out_path: Path,
):
    """Plot 4: Bin 1 DTND split by depression_fill_depth == 0 vs > 0."""
    in_bin1 = (
        (water_mask == 0) & np.isfinite(hand) & (hand > 0) & (hand <= BIN1_UPPER_M)
    )
    clean = in_bin1 & (depression_fill == 0)
    dirty = in_bin1 & (depression_fill > 0)

    dtnd_clean = dtnd[clean]
    dtnd_dirty = dtnd[dirty]

    fig, ax = plt.subplots(figsize=(10, 6))

    all_dtnd = (
        np.concatenate([dtnd_clean, dtnd_dirty]) if dtnd_dirty.size else dtnd_clean
    )
    hi = np.percentile(all_dtnd, 99.5) if all_dtnd.size else 100
    bins = np.linspace(0, hi, 80)

    if dtnd_clean.size > 0:
        ax.hist(
            dtnd_clean,
            bins=bins,
            color=HIST_COLOR_CLEAN,
            alpha=0.7,
            label=f"depression_fill = 0 ({dtnd_clean.size:,}, "
            f"med={np.median(dtnd_clean):.1f}m)",
            edgecolor="none",
        )
    if dtnd_dirty.size > 0:
        ax.hist(
            dtnd_dirty,
            bins=bins,
            color=HIST_COLOR_DIRTY,
            alpha=0.6,
            label=f"depression_fill > 0 ({dtnd_dirty.size:,}, "
            f"med={np.median(dtnd_dirty):.1f}m)",
            edgecolor="none",
        )
    ax.legend()
    ax.set_xlabel("DTND (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title(
        "Bin 1 DTND Split by Contamination Status\n"
        "Overlapping distributions ⇒ no filter needed; divergent ⇒ filter recommended"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_contamination(
    depression_fill: np.ndarray,
    water_mask: np.ndarray,
    out_path: Path,
    downsample: int = 5,
):
    """Plot 5: 2D map of depression_fill_depth with water mask outlined."""
    dep_ds = depression_fill[::downsample, ::downsample]
    water_ds = water_mask[::downsample, ::downsample]

    # Log-norm colormap; tiny pixel for zero to avoid log(0)
    dep_for_plot = np.where(dep_ds > 1e-6, dep_ds, np.nan)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Guard against all-NaN (no depression-filled pixels in downsampled view)
    n_finite = int(np.sum(np.isfinite(dep_for_plot)))
    if n_finite > 0:
        vmax = max(0.01, float(np.nanmax(dep_for_plot)))
        im = ax.imshow(
            dep_for_plot,
            cmap="hot_r",
            norm=LogNorm(vmin=0.001, vmax=vmax),
            interpolation="none",
        )
        fig.colorbar(im, ax=ax, label="depression_fill_depth (m)")
    else:
        ax.text(
            0.5,
            0.5,
            "No depression-filled pixels in downsampled view",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    # Overlay water mask as faint blue
    ax.imshow(
        np.where(water_ds > 0, 1, np.nan),
        cmap="Blues",
        alpha=0.35,
        interpolation="none",
    )

    ax.set_title(
        "Spatial Contamination Map\n"
        "Hot colors = depression-filled pixels. Blue overlay = NWI water mask.\n"
        "Hot pixels INSIDE blue = expected. Hot pixels OUTSIDE blue = unmapped wetlands."
    )
    ax.set_xlabel(f"Pixel column (downsampled x{downsample})")
    ax.set_ylabel("Pixel row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_nwi_interior_depth(
    depression_fill: np.ndarray, water_mask: np.ndarray, out_path: Path
):
    """Plot 6: depression_fill_depth inside NWI polygons vs Lee 2023 reference."""
    inside_nwi = (water_mask > 0) & (depression_fill > 0)
    vals = depression_fill[inside_nwi]

    fig, ax = plt.subplots(figsize=(10, 6))
    if vals.size > 0:
        hi = max(LEE_2023_MEAN_SPILL_DEPTH_M * 1.5, np.percentile(vals, 99))
        bins = np.linspace(0, hi, 80)
        ax.hist(
            vals,
            bins=bins,
            color=HIST_COLOR_DEP,
            alpha=0.8,
            edgecolor="none",
            label=f"NWI-interior depression_fill ({vals.size:,} pixels)",
        )
        ax.axvline(
            LEE_2023_MEAN_SPILL_DEPTH_M,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Lee et al. 2023 mean spill depth = "
            f"{LEE_2023_MEAN_SPILL_DEPTH_M:.2f}m",
        )
        ax.axvline(
            np.mean(vals),
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Modeled mean = {np.mean(vals):.2f}m",
        )
        ax.legend()
    ax.set_xlabel("depression_fill_depth (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title(
        "Modeled vs. Measured OSBS Basin Depth\n"
        "Inside-NWI fill depth ≈ CTSM's estimate of each lake's spill depth"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------


def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "q05": None, "q95": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q05": float(np.percentile(arr, 5)),
        "q95": float(np.percentile(arr, 95)),
    }


def compute_summary(arrs: dict, decomp: dict, bin_edges: np.ndarray) -> dict:
    hand = arrs["hand"]
    dtnd = arrs["dtnd"]
    water_mask = arrs["water_mask"]
    dep = decomp["depression_fill_depth"]
    pit = decomp["pit_fill_depth"]
    res = decomp["resolve_flat_depth"]

    land = (water_mask == 0) & np.isfinite(hand)
    land_positive = land & (hand > 0)
    in_bin1 = land_positive & (hand <= BIN1_UPPER_M)
    bin1_clean = in_bin1 & (dep == 0)
    bin1_dirty = in_bin1 & (dep > 0)
    nwi_interior = (water_mask > 0) & (dep > 0)

    return {
        "pixel_counts": {
            "total": int(water_mask.size),
            "water_nwi": int(np.sum(water_mask > 0)),
            "land_positive_hand": int(np.sum(land_positive)),
            "bin1_total": int(np.sum(in_bin1)),
            "bin1_clean_dep_zero": int(np.sum(bin1_clean)),
            "bin1_dirty_dep_positive": int(np.sum(bin1_dirty)),
            "nwi_interior_depression_filled": int(np.sum(nwi_interior)),
        },
        "fractions": {
            "land_with_depression_fill": float(
                np.sum(land & (dep > 0)) / max(1, np.sum(land))
            ),
            "bin1_with_depression_fill": float(
                np.sum(bin1_dirty) / max(1, np.sum(in_bin1))
            ),
        },
        "fill_stage_stats_m": {
            "pit_fill_depth_positive": _stats(pit[land & (pit > 0)]),
            "depression_fill_depth_positive": _stats(dep[land & (dep > 0)]),
            "resolve_flat_depth_positive": _stats(res[land & (res > 0)]),
        },
        "bin1_dtnd_stats_m": {
            "all": _stats(dtnd[in_bin1]),
            "clean_depression_fill_zero": _stats(dtnd[bin1_clean]),
            "dirty_depression_fill_positive": _stats(dtnd[bin1_dirty]),
        },
        "nwi_interior_fill_stats_m": _stats(dep[nwi_interior]),
        "lee_2023_mean_spill_depth_m": LEE_2023_MEAN_SPILL_DEPTH_M,
        "bin_edges_m": bin_edges.tolist(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Pipeline run directory containing diagnostics/ subdir",
    )
    args = parser.parse_args()

    diag_dir = args.run_dir / "diagnostics"
    plot_dir = diag_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading arrays from {diag_dir}")
    arrs = load_diagnostic_arrays(diag_dir)

    print("Computing fill-depth decomposition...")
    decomp = decompose_fill(arrs)

    # Compute bin edges using the same function as the production pipeline
    hand = arrs["hand"]
    water_mask = arrs["water_mask"]
    valid = (water_mask == 0) & np.isfinite(hand) & (hand > 0)
    bin_edges = np.array(compute_hand_bins_hybrid(hand[valid]))

    print(f"HAND bin edges ({len(bin_edges)}): {bin_edges}")

    print("Generating plots...")
    plot_fill_decomposition(
        decomp, arrs["water_mask"], plot_dir / "1_fill_decomposition.png"
    )
    print(f"  ✓ {plot_dir / '1_fill_decomposition.png'}")

    plot_hand_histogram(
        arrs["hand"], arrs["water_mask"], bin_edges, plot_dir / "2_hand_histogram.png"
    )
    print(f"  ✓ {plot_dir / '2_hand_histogram.png'}")

    plot_bin1_dtnd(
        arrs["hand"], arrs["dtnd"], arrs["water_mask"], plot_dir / "3_bin1_dtnd.png"
    )
    print(f"  ✓ {plot_dir / '3_bin1_dtnd.png'}")

    plot_bin1_dtnd_split(
        arrs["hand"],
        arrs["dtnd"],
        decomp["depression_fill_depth"],
        arrs["water_mask"],
        plot_dir / "4_bin1_dtnd_split.png",
    )
    print(f"  ✓ {plot_dir / '4_bin1_dtnd_split.png'}")

    plot_spatial_contamination(
        decomp["depression_fill_depth"],
        arrs["water_mask"],
        plot_dir / "5_spatial_contamination.png",
    )
    print(f"  ✓ {plot_dir / '5_spatial_contamination.png'}")

    plot_nwi_interior_depth(
        decomp["depression_fill_depth"],
        arrs["water_mask"],
        plot_dir / "6_nwi_interior_depth.png",
    )
    print(f"  ✓ {plot_dir / '6_nwi_interior_depth.png'}")

    print("Writing summary.json...")
    summary = compute_summary(arrs, decomp, bin_edges)
    with open(plot_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {plot_dir / 'summary.json'}")

    # Key takeaways printed to stdout
    frac_bin1_dirty = summary["fractions"]["bin1_with_depression_fill"]
    bin1_n = summary["pixel_counts"]["bin1_total"]
    print("\n=== Key findings ===")
    print(f"Bin 1 pixels: {bin1_n:,}")
    print(
        f"  Fraction with depression_fill > 0: {frac_bin1_dirty:.1%} "
        f"({summary['pixel_counts']['bin1_dirty_dep_positive']:,})"
    )
    nwi = summary["nwi_interior_fill_stats_m"]
    if nwi["mean"] is not None:
        print(
            f"NWI-interior mean depression_fill: {nwi['mean']:.2f}m "
            f"(Lee 2023 measured: {LEE_2023_MEAN_SPILL_DEPTH_M:.2f}m)"
        )


if __name__ == "__main__":
    main()
