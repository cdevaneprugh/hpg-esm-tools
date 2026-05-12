#!/usr/bin/env python3
"""
Diagnose holes in the NWI water mask.

Loads water_mask.tif, computes hole statistics (pixels enclosed by water but
labeled land), and renders a simple blue-on-white plot of the mask.

Usage:
    python diagnose_water_mask.py --label before
    python diagnose_water_mask.py --label after --mask-path /path/to/mask.tif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_fill_holes, label

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_MASK = BASE_DIR / "data" / "mosaics" / "production" / "water_mask.tif"
DEFAULT_OUTDIR = BASE_DIR / "output" / "osbs" / "water_mask_diagnostic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label", required=True, help="Output filename label (e.g. before, after)"
    )
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading mask: {args.mask_path}")
    with rasterio.open(args.mask_path) as src:
        mask = src.read(1)
    n_water = int(np.sum(mask > 0))
    n_total = mask.size
    print(f"  Shape: {mask.shape}")
    print(
        f"  Water pixels: {n_water:,} / {n_total:,} ({100.0 * n_water / n_total:.2f}%)"
    )

    print("\nComputing hole statistics...")
    filled = binary_fill_holes(mask > 0)
    holes = filled & ~(mask > 0)
    n_holes_px = int(np.sum(holes))
    print(f"  Hole pixels (enclosed by water, labeled land): {n_holes_px:,}")

    if n_holes_px > 0:
        labeled, n_components = label(holes)
        print(f"  Distinct hole components: {n_components}")
        if n_components > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            top = np.sort(sizes)[::-1][:5]
            print(f"  Top 5 hole sizes (pixels): {', '.join(f'{s:,}' for s in top)}")
    else:
        n_components = 0

    print(f"\nRendering plot: {args.label}.png")
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    ax.imshow(mask, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")

    title = f"NWI water mask ({args.label})"
    annotation = (
        f"water pixels: {n_water:,}\n"
        f"hole pixels:  {n_holes_px:,}\n"
        f"hole components: {n_components}"
    )
    ax.set_title(title, fontsize=12, loc="left")
    ax.text(
        0.99,
        0.01,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", linewidth=0.5
        ),
    )

    out_path = args.outdir / f"{args.label}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
