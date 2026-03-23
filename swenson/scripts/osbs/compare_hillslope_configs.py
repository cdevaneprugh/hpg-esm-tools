#!/usr/bin/env python3
"""
Compare hillslope profiles from two pipeline configurations.

Reads two hillslope_params.json files (e.g., 4x4 equal-area vs 1x8 log-spaced)
and produces a 2x2 figure comparing height, width, area, and slope profiles.

For multi-aspect configurations (4x4), computes an area-weighted average across
aspects to produce a single comparable profile line alongside per-aspect lines.

Usage:
    python scripts/osbs/compare_hillslope_configs.py <old.json> <new.json> <output.png>

Example:
    python scripts/osbs/compare_hillslope_configs.py \
        output/osbs/2026-03-17_tier3_contiguous/hillslope_params.json \
        output/osbs/2026-03-19_tier3_contiguous_1x8/hillslope_params.json \
        output/plots/4x4_vs_1x8_tier3.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASPECT_COLORS = {
    "North": "#3498db",
    "East": "#e74c3c",
    "South": "#f39c12",
    "West": "#2ecc71",
    "All": "#2c3e50",
}


def load_params(path: str) -> dict:
    """Load hillslope_params.json and extract non-zero elements."""
    with open(path) as f:
        params = json.load(f)

    meta = params["metadata"]
    elements = params["elements"]

    # Group elements by aspect, filter zero-area
    aspects = {}
    for elem in elements:
        name = elem["aspect_name"]
        if elem["area"] > 0:
            aspects.setdefault(name, []).append(elem)

    # Sort each aspect's elements by distance
    for name in aspects:
        aspects[name].sort(key=lambda e: e["distance"])

    label = f"{meta['n_aspect_bins']}x{meta['n_hand_bins']}"
    return {"meta": meta, "aspects": aspects, "label": label, "path": path}


def compute_averaged_profile(data: dict) -> list[dict] | None:
    """Area-weighted average across aspects, aligned by bin position.

    For a 4x4 config, each aspect has 4 bins (lowest to highest). This averages
    bin 1 across all aspects, bin 2 across all aspects, etc., weighting by area.

    Returns a list of averaged elements sorted by distance, or None if only
    one aspect has data.
    """
    aspects_with_data = {k: v for k, v in data["aspects"].items() if v}
    if len(aspects_with_data) <= 1:
        return None

    # Find the number of bins per aspect (use the mode)
    bin_counts = [len(elems) for elems in aspects_with_data.values()]
    n_bins = max(bin_counts)

    averaged = []
    for bin_idx in range(n_bins):
        total_area = 0
        weighted = {"height": 0, "distance": 0, "slope": 0, "width": 0}

        for elems in aspects_with_data.values():
            if bin_idx >= len(elems):
                continue
            e = elems[bin_idx]
            a = e["area"]
            total_area += a
            for key in weighted:
                weighted[key] += e[key] * a

        if total_area > 0:
            for key in weighted:
                weighted[key] /= total_area
            averaged.append(
                {
                    "height": weighted["height"],
                    "distance": weighted["distance"],
                    "slope": weighted["slope"],
                    "width": weighted["width"],
                    "area": total_area,
                    "hand_bin": bin_idx,
                }
            )

    averaged.sort(key=lambda e: e["distance"])
    return averaged


def plot_comparison(old: dict, new: dict, output_path: str) -> None:
    """Create 2x2 comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute averaged profile for old config (if multi-aspect)
    old_avg = compute_averaged_profile(old)

    panels = [
        (axes[0, 0], "height", "distance", "HAND (m)", "Height Profile"),
        (axes[0, 1], "width", "distance", "Width (m)", "Width Profile"),
        (axes[1, 0], "slope", "distance", "Slope (m/m)", "Slope Profile"),
    ]

    # Panels 1-3: line profiles (parameter vs distance)
    for ax, ykey, xkey, ylabel, title in panels:
        # Old config: thin colored lines per aspect
        for name, elems in old["aspects"].items():
            x = [e[xkey] for e in elems]
            y = [e[ykey] for e in elems]
            color = ASPECT_COLORS.get(name, "#999999")
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1,
                markersize=4,
                color=color,
                alpha=0.4,
                label=f"{old['label']} {name}",
            )

        # Old config: averaged profile (if multi-aspect)
        if old_avg is not None:
            x = [e[xkey] for e in old_avg]
            y = [e[ykey] for e in old_avg]
            ax.plot(
                x,
                y,
                marker="D",
                linewidth=2,
                markersize=7,
                color="#7f8c8d",
                label=f"{old['label']} avg",
                zorder=5,
            )

        # New config: thick dark line
        for name, elems in new["aspects"].items():
            x = [e[xkey] for e in elems]
            y = [e[ykey] for e in elems]
            color = ASPECT_COLORS.get(name, "#2c3e50")
            ax.plot(
                x,
                y,
                marker="s",
                linewidth=2.5,
                markersize=6,
                color=color,
                label=f"{new['label']}",
            )

        ax.set_xlabel("Distance from Stream (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")

    # Panel 4: area fraction distribution (bar chart)
    ax = axes[1, 1]

    old_total = sum(e["area"] for elems in old["aspects"].values() for e in elems)
    new_total = sum(e["area"] for elems in new["aspects"].values() for e in elems)

    # Old config: averaged area fractions (one bar per bin position)
    if old_avg is not None:
        old_avg_fracs = [e["area"] / old_total for e in old_avg]
        old_avg_labels = [f"Bin {e['hand_bin'] + 1}" for e in old_avg]
    else:
        old_avg_fracs = []
        old_avg_labels = []
        for name, elems in old["aspects"].items():
            for e in elems:
                old_avg_fracs.append(e["area"] / old_total if old_total > 0 else 0)
                old_avg_labels.append(f"{e['hand_bin'] + 1}")

    # New config bars
    new_fracs = []
    new_labels = []
    for name, elems in new["aspects"].items():
        for e in elems:
            new_fracs.append(e["area"] / new_total if new_total > 0 else 0)
            new_labels.append(f"Bin {e['hand_bin'] + 1}")

    x_old = np.arange(len(old_avg_fracs))
    x_new = np.arange(len(new_fracs)) + len(old_avg_fracs) + 1.5

    ax.bar(
        x_old, old_avg_fracs, color="#7f8c8d", alpha=0.7, width=0.8, label=old["label"]
    )
    ax.bar(
        x_new,
        new_fracs,
        color=ASPECT_COLORS.get("All", "#2c3e50"),
        alpha=0.7,
        width=0.8,
        label=new["label"],
    )

    all_ticks = list(x_old) + list(x_new)
    all_labels = old_avg_labels + new_labels
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Area Fraction")
    ax.set_title("Area Distribution (summed across aspects for 4x4)")
    ax.legend(fontsize=8)

    # Add separator
    sep_x = len(old_avg_fracs) + 0.5
    ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Metadata annotation
    old_meta = old["meta"]
    new_meta = new["meta"]
    info = (
        f"Old: {old['label']}, Lc={old_meta.get('spatial_scale_m', 0):.0f}m, "
        f"total area={old_total / 1e6:.3f} km\u00b2\n"
        f"New: {new['label']}, Lc={new_meta.get('spatial_scale_m', 0):.0f}m, "
        f"total area={new_total / 1e6:.3f} km\u00b2"
    )
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, style="italic")

    fig.suptitle("Hillslope Configuration Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare hillslope profiles from two configurations"
    )
    parser.add_argument("old_json", help="Path to old config hillslope_params.json")
    parser.add_argument("new_json", help="Path to new config hillslope_params.json")
    parser.add_argument("output_png", help="Output plot path")
    args = parser.parse_args()

    for p in [args.old_json, args.new_json]:
        if not Path(p).exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    old = load_params(args.old_json)
    new = load_params(args.new_json)

    print(f"Old: {old['label']} ({len(old['aspects'])} aspects with data)")
    print(f"New: {new['label']} ({len(new['aspects'])} aspects with data)")

    # Report averaged profile if computed
    avg = compute_averaged_profile(old)
    if avg:
        print(f"Old averaged: {len(avg)} bins with data")

    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(old, new, args.output_png)


if __name__ == "__main__":
    main()
