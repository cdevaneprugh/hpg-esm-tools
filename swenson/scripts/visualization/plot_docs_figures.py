#!/usr/bin/env python3
"""Generate OSBS-side comparison plots for docs/swenson/dataset-comparison.md.

Reads the production `hillslope_params.json` and writes two PNGs meant to
sit next to `swenson_elevation_width.png` and `swenson_col_areas.png` in
the docs. Runs in seconds on a login node — no pipeline rerun, no
CTSM/pysheds dependencies.

Usage:
    python scripts/visualization/plot_docs_figures.py \\
        [--input PATH] [--output-dir DIR]

Defaults:
    input      = swenson/output/osbs/2026-05-05_production/hillslope_params.json
    output-dir = /blue/gerber/cdevaneprugh/hpg-esm-docs/docs/swenson/images/

Produces:
    production_elevation_width.png   (2-panel line chart, HAND + width vs distance)
    production_col_areas.png         (bar chart, column area with % labels)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Palette matches production_hillslope_params.png
COLOR_LAKE = "#A6C8DB"
COLOR_FLOOD = "#8B2C2C"
COLOR_UPLAND = "#4C9A80"
COLOR_LINE = "#333333"

SCRIPT_DIR = Path(__file__).resolve().parent
SWENSON_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = (
    SWENSON_DIR
    / "output"
    / "osbs"
    / "2026-05-05_production"
    / "hillslope_params.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/blue/gerber/cdevaneprugh/hpg-esm-docs/docs/swenson/images"
)


def zone_color(element: dict) -> str:
    """Lake / flood-zone (raw HAND < 0) / upland (raw HAND ≥ 0)."""
    if element.get("is_lake", False):
        return COLOR_LAKE
    if element["height"] < 0:
        return COLOR_FLOOD
    return COLOR_UPLAND


def load_elements(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["elements"]


def plot_elevation_width(elements: list[dict], out_path: Path) -> None:
    """Two-panel line chart: elevation and width vs distance from stream."""
    lake = [e for e in elements if e.get("is_lake", False)]
    land = sorted(
        [e for e in elements if not e.get("is_lake", False)],
        key=lambda e: e["distance"],
    )

    land_dist = np.array([e["distance"] for e in land])
    land_height = np.array([e["height"] for e in land])
    land_width = np.array([e["width"] for e in land])
    land_colors = [zone_color(e) for e in land]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: HAND
    ax1.plot(
        land_dist, land_height,
        color=COLOR_LINE, linewidth=1.4, alpha=0.55, zorder=1,
    )
    ax1.scatter(
        land_dist, land_height,
        c=land_colors, s=70, edgecolors="black", linewidths=0.5, zorder=2,
    )
    for e in lake:
        ax1.scatter(
            [e["distance"]], [e["height"]],
            c=COLOR_LAKE, s=260, marker="*",
            edgecolors="black", linewidths=0.9, zorder=3,
        )
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle=":", zorder=0)
    ax1.set_xlabel("Distance from Stream (m)", fontsize=12)
    ax1.set_ylabel("HAND above Stream (m)", fontsize=12)
    ax1.set_title("Elevation Profile", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right panel: width
    ax2.plot(
        land_dist, land_width,
        color=COLOR_LINE, linewidth=1.4, alpha=0.55, zorder=1,
    )
    ax2.scatter(
        land_dist, land_width,
        c=land_colors, s=70, edgecolors="black", linewidths=0.5, zorder=2,
    )
    for e in lake:
        ax2.scatter(
            [e["distance"]], [e["width"]],
            c=COLOR_LAKE, s=260, marker="*",
            edgecolors="black", linewidths=0.9, zorder=3,
        )
    ax2.set_xlabel("Distance from Stream (m)", fontsize=12)
    ax2.set_ylabel("Hillslope Width (m)", fontsize=12)
    ax2.set_title("Width Profile", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")

    legend_handles = [
        Line2D(
            [0], [0], marker="*", color="w", markerfacecolor=COLOR_LAKE,
            markeredgecolor="black", markersize=16, label="Lake column",
        ),
        Patch(
            facecolor=COLOR_FLOOD, edgecolor="black",
            label="Flood zone (raw HAND < 0)",
        ),
        Patch(
            facecolor=COLOR_UPLAND, edgecolor="black",
            label="Upland (raw HAND ≥ 0)",
        ),
    ]
    ax1.legend(
        handles=legend_handles, loc="upper left",
        fontsize=10, framealpha=0.9,
    )

    fig.suptitle(
        "Hillslope Geometry: OSBS 1 m NEON LIDAR (production, 25 columns)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_col_areas(elements: list[dict], out_path: Path) -> None:
    """Bar chart: per-rep column area with percent-of-total labels."""
    labels: list[str] = []
    areas: list[float] = []
    colors: list[str] = []
    for e in elements:
        if e.get("is_lake", False):
            labels.append("Lake")
        else:
            labels.append(str(e["hand_bin"] + 1))
        areas.append(e["area"])
        colors.append(zone_color(e))

    areas_arr = np.array(areas)
    total = areas_arr.sum()
    pcts = 100.0 * areas_arr / total

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, areas_arr, color=colors, edgecolor="black", linewidth=0.5,
    )

    ymax = areas_arr.max()
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel(
        "Column (chain order: Lake, then bins 1–24 low → high HAND)",
        fontsize=12,
    )
    ax.set_ylabel(
        "Column area per representative hillslope (m²)",
        fontsize=12,
    )
    ax.set_title(
        "Hillslope Column Areas: OSBS 1 m production (25 columns, single aspect)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_ylim(0, ymax * 1.15)

    legend_handles = [
        Patch(facecolor=COLOR_LAKE, edgecolor="black", label="Lake"),
        Patch(
            facecolor=COLOR_FLOOD, edgecolor="black",
            label="Flood zone (raw HAND < 0)",
        ),
        Patch(
            facecolor=COLOR_UPLAND, edgecolor="black",
            label="Upland (raw HAND ≥ 0)",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        fontsize=10, framealpha=0.9,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Production hillslope_params.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for PNGs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    elements = load_elements(args.input)
    print(f"Loaded {len(elements)} elements from {args.input}")

    plot_elevation_width(
        elements, args.output_dir / "production_elevation_width.png"
    )
    plot_col_areas(
        elements, args.output_dir / "production_col_areas.png"
    )


if __name__ == "__main__":
    main()
