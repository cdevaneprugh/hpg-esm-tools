"""Generate three diagnostic plots from the 2026-04-24 saved arrays.

1. Spatial contamination map at full resolution: depression_fill_depth heatmap
   + NWI water filled blue + current pipeline bin 1 in neon green + legend.
2. Histogram of raw HAND values for hot pixels that change bin under the
   raw-HAND fix. Annotates the proposed flood-zone bin edges.
3. Column area comparison (old vs new structure) including the lake column.

Reads from:  output/osbs/2026-04-24_diagnostic/diagnostics/*.npy
Writes to:   output/osbs/HAND-diagnostic-2026-04-25/
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path("/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson")
SOURCE = REPO / "output/osbs/2026-04-24_diagnostic/diagnostics"
OUT = REPO / "output/osbs/HAND-diagnostic-2026-04-25"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading saved arrays...")
hand = np.load(SOURCE / "hand.npy")
flooded_orig = np.load(SOURCE / "flooded_orig.npy")
pit_filled = np.load(SOURCE / "pit_filled.npy")
water_mask = np.load(SOURCE / "water_mask.npy")

with open(SOURCE / "plots/summary.json") as f:
    summary = json.load(f)
current_edges = np.array(summary["bin_edges_m"])

print(f"  shape = {hand.shape}, total pixels = {hand.size:,}")

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

dep_fill = flooded_orig - pit_filled  # Stage 2 conditioning depth
raw_hand = hand - dep_fill  # corrected HAND

# Hot mask (depression-filled non-water land pixels with finite positive HAND)
hot = (water_mask == 0) & np.isfinite(hand) & (hand > 0) & (dep_fill > 0)
print(f"  hot pixels: {int(hot.sum()):,}")

# ---------------------------------------------------------------------------
# New bin scheme (proposed)
# ---------------------------------------------------------------------------

neg_edges = np.array([-np.inf, -2.0, -1.0, -0.5, -0.2, 0.0])
neg_labels = ["FZ_DEEP", "FZ_MOD", "FZ_SHALLOW", "FZ_RIM_NEAR", "FZ_MARGIN"]
pos_edges = current_edges  # starts at 0
pos_labels = ["True B1"] + [f"B{i + 1}" for i in range(1, len(pos_edges) - 1)]

all_edges = np.concatenate([neg_edges, pos_edges[1:]])
all_labels = neg_labels + pos_labels

# Simplified scheme used by plot 3: collapse all flood-zone pixels into a
# single "FZ" column (matches Setup 1). Used only for the column-area
# comparison; other plots still use the 5-bin breakdown for stratification.
simple_neg_edges = np.array([-np.inf, 0.0])
simple_neg_labels = ["FZ"]
simple_all_edges = np.concatenate([simple_neg_edges, pos_edges[1:]])
simple_all_labels = simple_neg_labels + pos_labels

# ---------------------------------------------------------------------------
# Plot 1: Spatial contamination map (full resolution)
# ---------------------------------------------------------------------------


def plot_spatial():
    print("Plot 1: spatial contamination map (full resolution)...")

    # Depression fill heat map (NaN where no fill, so pixels show through)
    dep_for_plot = np.where(dep_fill > 1e-6, dep_fill, np.nan)

    # Bin 1 pixels (current pipeline scheme): land, finite, HAND in (0, 0.1m]
    bin1_mask = (water_mask == 0) & np.isfinite(hand) & (hand > 0) & (hand <= 0.1)
    bin1_overlay = np.where(bin1_mask, 1.0, np.nan)
    n_bin1 = int(bin1_mask.sum())

    # NWI water filled blue
    nwi_overlay = np.where(water_mask > 0, 1.0, np.nan)
    n_nwi = int((water_mask > 0).sum())

    fig, ax = plt.subplots(figsize=(20, 18))

    # Layer 1 (bottom): depression_fill_depth heatmap
    n_finite = int(np.sum(np.isfinite(dep_for_plot)))
    if n_finite > 0:
        vmax = max(0.01, float(np.nanmax(dep_for_plot)))
        im = ax.imshow(
            dep_for_plot,
            cmap="hot_r",
            norm=LogNorm(vmin=0.001, vmax=vmax),
            interpolation="none",
            aspect="equal",
        )
        cbar = fig.colorbar(im, ax=ax, label="depression_fill_depth (m)", shrink=0.7)
        cbar.ax.tick_params(labelsize=11)

    # Layer 2: current bin 1 pixels in neon green (under NWI overlay)
    green_cmap = ListedColormap(["#39FF14"])
    ax.imshow(
        bin1_overlay,
        cmap=green_cmap,
        alpha=0.65,
        interpolation="none",
        aspect="equal",
    )

    # Layer 3 (top): NWI water mask filled blue — drawn last so it sits above
    # everything (including the bin 1 highlight, even though by construction
    # bin 1 pixels and NWI pixels don't overlap).
    blue_cmap = ListedColormap(["#1f77b4"])
    ax.imshow(
        nwi_overlay,
        cmap=blue_cmap,
        alpha=0.85,
        interpolation="none",
        aspect="equal",
    )

    # Legend
    legend_elements = [
        Patch(
            facecolor="#1f77b4",
            alpha=0.85,
            edgecolor="black",
            label=f"NWI water mask (excluded from binning): {n_nwi:,} pixels",
        ),
        Patch(
            facecolor="#39FF14",
            alpha=0.65,
            edgecolor="black",
            label=f"Pipeline bin 1 (HAND ≤ 0.1m, non-water): {n_bin1:,} pixels",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=12, framealpha=0.95)

    ax.set_xlabel("Pixel column (1m)")
    ax.set_ylabel("Pixel row (1m)")

    fig.tight_layout()
    out_path = OUT / "1_spatial_contamination.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Histogram of raw HAND for hot pixels that change bin
# ---------------------------------------------------------------------------


def plot_movers_histogram():
    print("Plot 2: raw HAND histogram for hot movers...")

    raw_h_hot = raw_hand[hot]
    pipe_h_hot = hand[hot]

    # New bin assignment under raw HAND
    new_idx = np.digitize(raw_h_hot, all_edges) - 1
    # Old (pipeline) bin assignment, mapped to all_labels space
    old_pipe_idx = np.digitize(pipe_h_hot, current_edges) - 1
    pipeline_in_new_space = old_pipe_idx + len(neg_labels)

    movers = new_idx != pipeline_in_new_space
    raw_h_movers = raw_h_hot[movers]
    n_movers = int(movers.sum())
    n_stayers = int((~movers).sum())
    print(f"  hot movers: {n_movers:,}")
    print(f"  hot stayers: {n_stayers:,}")

    fig, ax = plt.subplots(figsize=(13, 7))

    # Histogram with fine bins across mover range
    lo, hi = float(raw_h_movers.min()), float(raw_h_movers.max())
    bins = np.linspace(lo, hi, 200)
    ax.hist(
        raw_h_movers,
        bins=bins,
        color="#e74c3c",
        alpha=0.85,
        edgecolor="none",
    )

    # Reference: median and mean of movers
    med = float(np.median(raw_h_movers))
    mean = float(np.mean(raw_h_movers))
    ax.axvline(
        med, color="cyan", linestyle="-", linewidth=1.5, label=f"median = {med:.2f} m"
    )
    ax.axvline(
        mean,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"mean = {mean:.2f} m",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Raw HAND (m)")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title(
        f"Raw HAND distribution of hot pixels that change bin under the fix\n"
        f"({n_movers:,} pixels, vs {n_stayers:,} hot pixels that stay in their current bin)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = OUT / "2_movers_raw_hand_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Column area comparison (old vs new)
# ---------------------------------------------------------------------------


def plot_column_areas():
    print("Plot 3: column area comparison...")

    # Land mask (non-water, finite, positive pipeline HAND)
    land = (water_mask == 0) & np.isfinite(hand) & (hand > 0)

    # Old bin assignment (pipeline HAND)
    old_idx = np.digitize(hand[land], current_edges) - 1
    old_areas = np.bincount(old_idx, minlength=len(current_edges) - 1)

    # New bin assignment (raw HAND) — use the SIMPLE scheme (single FZ column)
    # for the area comparison so the structural change reads cleanly:
    # lake + 1 flood zone + True B1 + B2-B16 = 18 columns total.
    new_idx = np.digitize(raw_hand[land], simple_all_edges) - 1
    new_areas = np.bincount(new_idx, minlength=len(simple_all_labels))

    lake_area = int((water_mask > 0).sum())

    # Build labels and area arrays
    old_labels_full = ["LAKE"] + [f"B{i + 1}" for i in range(len(old_areas))]
    old_areas_full = np.concatenate([[lake_area], old_areas])

    new_labels_full = ["LAKE"] + simple_all_labels
    new_areas_full = np.concatenate([[lake_area], new_areas])

    # Convert to km² (1m × 1m pixels → 1e6 px = 1 km²)
    old_km2 = old_areas_full / 1e6
    new_km2 = new_areas_full / 1e6

    # Color schemes
    def old_color(label):
        if label == "LAKE":
            return "#3498db"
        return "#95a5a6"

    def new_color(label):
        if label == "LAKE":
            return "#3498db"
        if label.startswith("FZ"):
            return "#e74c3c"
        if label == "True B1":
            return "#0d6b2c"
        return "#95a5a6"

    old_colors = [old_color(label) for label in old_labels_full]
    new_colors = [new_color(label) for label in new_labels_full]

    fig, (ax_old, ax_new) = plt.subplots(2, 1, figsize=(16, 11))

    # Old (top)
    bars_old = ax_old.bar(
        range(len(old_labels_full)), old_km2, color=old_colors, edgecolor="black"
    )
    ax_old.set_xticks(range(len(old_labels_full)))
    ax_old.set_xticklabels(old_labels_full, rotation=45, ha="right", fontsize=9)
    ax_old.set_ylabel("Column area (km²)")
    ax_old.set_title(
        f"Current column structure (lake + 16 bins from pipeline HAND, {len(old_labels_full)} columns)"
    )
    ax_old.grid(True, alpha=0.3, axis="y")
    for bar, area in zip(bars_old, old_km2):
        if area > 0.1:
            ax_old.text(
                bar.get_x() + bar.get_width() / 2,
                area,
                f"{area:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # New (bottom)
    bars_new = ax_new.bar(
        range(len(new_labels_full)), new_km2, color=new_colors, edgecolor="black"
    )
    ax_new.set_xticks(range(len(new_labels_full)))
    ax_new.set_xticklabels(new_labels_full, rotation=45, ha="right", fontsize=9)
    ax_new.set_ylabel("Column area (km²)")
    ax_new.set_title(
        f"Proposed new column structure (lake + 1 flood zone + True B1 + B2-B16 = {len(new_labels_full)} columns)"
    )
    ax_new.grid(True, alpha=0.3, axis="y")
    for bar, area in zip(bars_new, new_km2):
        if area > 0.1:
            ax_new.text(
                bar.get_x() + bar.get_width() / 2,
                area,
                f"{area:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Same y-axis scale for direct comparison
    ymax = max(old_km2.max(), new_km2.max()) * 1.18
    ax_old.set_ylim(0, ymax)
    ax_new.set_ylim(0, ymax)

    # Legend on the new (bottom) axis
    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Lake (NWI water surface)"),
        Patch(
            facecolor="#e74c3c",
            edgecolor="black",
            label="Flood zone (raw HAND < 0, dry land below stream)",
        ),
        Patch(
            facecolor="#0d6b2c",
            edgecolor="black",
            label="True B1 (raw HAND in (0, 0.1m]) — what bin 1 should have been",
        ),
        Patch(
            facecolor="#95a5a6",
            edgecolor="black",
            label="Hillslope-direction bins (B2-B16)",
        ),
    ]
    ax_new.legend(handles=legend_elements, loc="upper left", fontsize=10)

    fig.tight_layout()
    out_path = OUT / "3_column_areas_old_vs_new.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # Stats summary printed
    total_old = int(old_areas_full.sum())
    total_new = int(new_areas_full.sum())
    print(
        f"  old totals: {total_old:,} pixels ({total_old / 1e6:.2f} km²) across {len(old_labels_full)} columns"
    )
    print(
        f"  new totals: {total_new:,} pixels ({total_new / 1e6:.2f} km²) across {len(new_labels_full)} columns"
    )
    assert total_old == total_new, (
        f"pixel count mismatch: old {total_old} vs new {total_new}"
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_spatial()
    plot_movers_histogram()
    plot_column_areas()
    print("\nAll three plots generated.")
