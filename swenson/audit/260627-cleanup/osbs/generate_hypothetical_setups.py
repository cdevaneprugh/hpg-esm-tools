"""Compute hypothetical hillslope parameters for two flood-zone setups.

Replicates the production pipeline's trap-fit-derived distance and width
calculations from the saved 2026-04-24 diagnostic arrays. Generates two
4-subplot figures (HAND, DTND, area, width) — one per setup — for visual
comparison.

Both setups differ only in the flood-zone (raw HAND < 0) bin partition.
Positive-HAND bins are identical: existing production edges applied to raw HAND.

Setups:
  1. Single FZ bin: raw HAND ≤ 0  → 18 columns total
  2. 5 FZ bins (original proposal) → 22 columns total

Setup 3 (8 equal-count bins) was generated previously but produced a
chain-monotonicity violation at the LAKE→FZ1 boundary (FZ1's trap-fit-derived
DTND came out below the PI-set lake DTND of 5m). Removed for now; logic
preserved in make_setup_edges() for possible future use after PI input.

Lake column (col 1 in chain order) uses PI-direction parameters:
  HAND  = −SPILLHEIGHT (−0.2 m)
  DTND  = ~stream width (~5 m, inert under routing-off config)
  Area  = sum of NWI water mask × 1 m²
  Width = 1/2 NWI total perimeter (computed from mask boundary pixels)

n_hillslopes is algebraically invariant in the trap-fit-derived distance/width
quadratic. Pass any value (here, 1) to get correct results from saved arrays
without needing the (unsaved) drainage_id.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion

# Allow imports from scripts/
REPO = Path("/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson")
sys.path.insert(0, str(REPO / "scripts"))

from hillslope_params import fit_trapezoidal_width, quadratic, tail_index  # noqa: E402

# ---------------------------------------------------------------------------
# Constants matching production pipeline
# ---------------------------------------------------------------------------

PIXEL_SIZE_M = 1.0
PIXEL_AREA_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M
SPILLHEIGHT_M = 0.2
LAKE_HAND_M = -SPILLHEIGHT_M  # NetCDF value before SourceMod runtime shift
LAKE_DTND_M = 5.0  # ~stream width; PI direction (mathematically inert)
SMALLEST_DTND_M = 1.0  # Swenson's fixed minimum (rh:699-700)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE = REPO / "output/osbs/2026-04-24_diagnostic/diagnostics"
OUT = REPO / "output/osbs/HAND-diagnostic-2026-04-25"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data and apply production-style filtering
# ---------------------------------------------------------------------------


def load_and_filter() -> dict:
    """Load saved arrays and apply production-pipeline filtering.

    Mirrors run_pipeline.py:1005-1049: tail removal, DTND minimum clip, water
    masking. Returns flat (1D) arrays of valid pixels for downstream binning.
    """
    print("Loading diagnostic arrays...")
    hand = np.load(SOURCE / "hand.npy")
    dtnd = np.load(SOURCE / "dtnd.npy")
    flooded_orig = np.load(SOURCE / "flooded_orig.npy")
    pit_filled = np.load(SOURCE / "pit_filled.npy")
    water_mask = np.load(SOURCE / "water_mask.npy")

    # Use saved bin edges from existing diagnostic (matches production)
    with open(SOURCE / "plots/summary.json") as f:
        summary = json.load(f)
    pos_edges = np.array(summary["bin_edges_m"])
    print(f"  loaded shape={hand.shape}, total={hand.size:,}")

    dep_fill = flooded_orig - pit_filled
    raw_hand = hand - dep_fill

    # Flatten for easier indexing (matches pipeline pattern)
    hand_flat = hand.flatten()
    raw_hand_flat = raw_hand.flatten()
    dtnd_flat = dtnd.flatten().copy()  # we'll modify with min-clip
    water_flat = water_mask.flatten()

    # Step 1: tail removal on (DTND, HAND) joint distribution. Matches
    # run_pipeline.py:1031-1037 exactly.
    land_finite = np.isfinite(hand_flat) & (water_flat == 0)
    print(f"  land+finite: {int(land_finite.sum()):,}")
    tail_ind = tail_index(dtnd_flat[land_finite], hand_flat[land_finite])
    land_indices = np.where(land_finite)[0]
    keep_tail = np.zeros(hand_flat.shape, dtype=bool)
    keep_tail[land_indices[tail_ind]] = True
    n_removed = int(np.sum(land_finite) - np.sum(keep_tail))
    print(f"  tail removed: {n_removed:,}")

    # Step 2: DTND minimum clip
    dtnd_flat[dtnd_flat < SMALLEST_DTND_M] = SMALLEST_DTND_M

    # Step 3: water masking
    valid = keep_tail & (water_flat == 0)
    print(f"  valid land pixels: {int(valid.sum()):,}")

    return {
        "hand_flat": hand_flat,
        "raw_hand_flat": raw_hand_flat,
        "dtnd_flat": dtnd_flat,
        "water_flat": water_flat,
        "valid": valid,
        "pos_edges": pos_edges,
        "water_mask_2d": water_mask,
    }


# ---------------------------------------------------------------------------
# Lake column parameters (computed once)
# ---------------------------------------------------------------------------


def compute_lake_column(water_mask_2d: np.ndarray) -> dict:
    """Compute lake column parameters from NWI water mask."""
    n_lake = int((water_mask_2d > 0).sum())
    area_m2 = n_lake * PIXEL_AREA_M2

    # Boundary pixels: pixels in mask whose erosion removes them.
    # Each boundary pixel contributes ~1m of perimeter at 1m resolution.
    eroded = binary_erosion(water_mask_2d > 0)
    boundary = (water_mask_2d > 0) & ~eroded
    n_boundary = int(boundary.sum())
    perimeter_m = float(n_boundary) * PIXEL_SIZE_M
    width_m = perimeter_m / 2.0

    print(
        f"Lake: n={n_lake:,}, area={area_m2 / 1e6:.3f} km², "
        f"boundary={n_boundary:,} px, perimeter≈{perimeter_m:.0f}m, "
        f"width=perim/2={width_m:.0f}m"
    )

    return {
        "label": "LAKE",
        "n_pixels": n_lake,
        "hand_m": LAKE_HAND_M,
        "dtnd_m": LAKE_DTND_M,
        "area_m2": area_m2,
        "width_m": width_m,
    }


# ---------------------------------------------------------------------------
# Setup definitions
# ---------------------------------------------------------------------------


def make_setup_edges(
    setup: int, pos_edges: np.ndarray, raw_hand_for_quantiles: np.ndarray = None
) -> tuple:
    """Return (all_edges, all_labels, n_neg) for the given setup.

    pos_edges is the existing production HAND-bin edge array (starts at 0).
    Final all_edges = neg_edges + pos_edges[1:] (skip duplicate 0).

    Setup 3 uses quantile-based equal-count flood-zone bins; pass the full
    raw HAND array via raw_hand_for_quantiles for that setup.
    """
    if setup == 1:
        neg_edges = np.array([-np.inf, 0.0])
        neg_labels = ["FZ"]
    elif setup == 2:
        neg_edges = np.array([-np.inf, -2.0, -1.0, -0.5, -0.2, 0.0])
        neg_labels = ["FZ_DEEP", "FZ_MOD", "FZ_SHALLOW", "FZ_RIM_NEAR", "FZ_MARGIN"]
    elif setup == 3:
        # 8 equal-count quantile bins across the flood zone (raw HAND ≤ 0).
        # Each bin gets ~1.95M pixels; matches existing production B5/B6 sizes.
        if raw_hand_for_quantiles is None:
            raise ValueError("Setup 3 requires raw_hand_for_quantiles")
        fz_values = raw_hand_for_quantiles[raw_hand_for_quantiles <= 0]
        n_fz_bins = 8
        quantiles = np.linspace(0, 1, n_fz_bins + 1)
        neg_edges = np.quantile(fz_values, quantiles)
        # Pin first/last edges to canonical values (-inf and 0)
        neg_edges[0] = -np.inf
        neg_edges[-1] = 0.0
        neg_labels = [f"FZ{i + 1}" for i in range(n_fz_bins)]
    else:
        raise ValueError(f"Unknown setup {setup}")
    pos_labels = ["True B1"] + [f"B{i + 1}" for i in range(1, len(pos_edges) - 1)]
    all_edges = np.concatenate([neg_edges, pos_edges[1:]])
    all_labels = neg_labels + pos_labels
    return all_edges, all_labels, len(neg_labels)


# ---------------------------------------------------------------------------
# Per-setup parameter computation (replicates run_pipeline.py:1141-1278)
# ---------------------------------------------------------------------------


def compute_setup_params(setup: int, data: dict, lake_col: dict) -> dict:
    """Compute hillslope parameters for one setup.

    Returns dict with:
      - "labels": list of column labels in chain order (lake first)
      - "hand_m":   per-column mean raw HAND
      - "dtnd_m":   per-column trap-fit distance (or fallback)
      - "area_m2":  per-column area
      - "width_m":  per-column trap-fit width
      - "n_pixels": per-column pixel count
      - "trap_fit": diagnostic info (slope, width, area, fallback?)
    """
    raw_hand_flat = data["raw_hand_flat"]
    dtnd_flat = data["dtnd_flat"]
    valid = data["valid"]
    pos_edges = data["pos_edges"]

    # Use valid-land raw HAND for Setup 3's quantile edges
    raw_hand_for_quantiles = raw_hand_flat[valid]
    all_edges, all_labels, n_neg = make_setup_edges(
        setup, pos_edges, raw_hand_for_quantiles
    )
    n_bins = len(all_labels)

    # Trap fit on all valid pixels (single aspect)
    dtnd_valid = dtnd_flat[valid]
    area_valid = np.full(int(valid.sum()), PIXEL_AREA_M2)
    n_hillslopes = 1  # invariant for distance/width
    trap = fit_trapezoidal_width(
        dtnd_valid, area_valid, n_hillslopes, min_dtnd=PIXEL_SIZE_M
    )
    trap_slope = trap["slope"]
    trap_width = trap["width"]
    trap_area = trap["area"]
    print(
        f"Setup {setup}: trap_slope={trap_slope:.4f}, "
        f"trap_width={trap_width:.2f}, trap_area={trap_area:.2f}"
    )
    fallback_used = trap_slope == 0
    if fallback_used:
        print(
            "  WARNING: trap_slope=0 (degenerate). Distances will fall back to median DTND."
        )

    # First pass: per-bin pixel counts and raw areas
    raw_hand_valid = raw_hand_flat[valid]
    bin_assignments = np.digitize(raw_hand_valid, all_edges) - 1

    bin_n = np.zeros(n_bins, dtype=int)
    bin_raw_area = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_assignments == b
        bin_n[b] = int(mask.sum())
        bin_raw_area[b] = bin_n[b] * PIXEL_AREA_M2

    total_raw = bin_raw_area.sum()
    if total_raw > 0:
        area_fractions = bin_raw_area / total_raw
    else:
        area_fractions = np.full(n_bins, 1.0 / n_bins)
    fitted_areas = trap_area * area_fractions

    # Second pass: per-bin parameters
    hand_per_bin = np.zeros(n_bins)
    dtnd_per_bin = np.zeros(n_bins)
    width_per_bin = np.zeros(n_bins)

    # Pre-extract per-pixel arrays for speed
    raw_hand_v = raw_hand_valid
    dtnd_v = dtnd_flat[valid]

    for b in range(n_bins):
        if bin_n[b] == 0:
            # Empty bin: use bin-edge midpoint if available, else 0
            lo = all_edges[b]
            hi = all_edges[b + 1]
            if np.isfinite(lo) and np.isfinite(hi):
                hand_per_bin[b] = (lo + hi) / 2
            elif np.isfinite(hi):
                hand_per_bin[b] = hi
            else:
                hand_per_bin[b] = lo
            dtnd_per_bin[b] = 0
            width_per_bin[b] = 0
            continue

        mask = bin_assignments == b
        hand_per_bin[b] = float(raw_hand_v[mask].mean())

        # Width at lower edge of bin
        da_width = float(np.sum(fitted_areas[:b])) if b > 0 else 0.0
        if trap_slope != 0:
            try:
                le = quadratic([trap_slope, trap_width, -da_width])
                width_per_bin[b] = max(trap_width + 2 * trap_slope * le, 1.0)
            except RuntimeError:
                width_per_bin[b] = max(trap_width * (1 - 0.15 * b), 1.0)
        else:
            width_per_bin[b] = max(trap_width, 1.0)

        # Distance at bin midpoint
        da_dist = float(np.sum(fitted_areas[: b + 1]) - fitted_areas[b] / 2)
        if trap_slope != 0:
            try:
                dtnd_per_bin[b] = float(quadratic([trap_slope, trap_width, -da_dist]))
            except RuntimeError:
                dtnd_per_bin[b] = float(np.median(dtnd_v[mask]))
        else:
            dtnd_per_bin[b] = float(np.median(dtnd_v[mask]))

    # Build chain-ordered column list: LAKE first, then bins in HAND order
    labels = [lake_col["label"]] + all_labels
    n_pixels = np.concatenate([[lake_col["n_pixels"]], bin_n])
    hand = np.concatenate([[lake_col["hand_m"]], hand_per_bin])
    dtnd = np.concatenate([[lake_col["dtnd_m"]], dtnd_per_bin])
    area = np.concatenate([[lake_col["area_m2"]], bin_raw_area])
    width = np.concatenate([[lake_col["width_m"]], width_per_bin])

    return {
        "labels": labels,
        "hand_m": hand,
        "dtnd_m": dtnd,
        "area_m2": area,
        "width_m": width,
        "n_pixels": n_pixels,
        "n_neg_bins": n_neg,
        "trap_slope": float(trap_slope),
        "trap_width": float(trap_width),
        "trap_area": float(trap_area),
        "fallback_used": fallback_used,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_setup(setup: int, params: dict, out_path: Path):
    """4-subplot figure: HAND, DTND, area, width per column."""
    labels = params["labels"]
    n = len(labels)
    x = np.arange(n)

    # Color: lake blue, FZ red, True B1 green, B-bins gray
    colors = ["#3498db"]  # lake
    for lbl in labels[1:]:
        if lbl.startswith("FZ"):
            colors.append("#e74c3c")
        elif lbl == "True B1":
            colors.append("#0d6b2c")
        else:
            colors.append("#95a5a6")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    (ax_hand, ax_dtnd), (ax_area, ax_width) = axes

    # HAND
    ax_hand.bar(x, params["hand_m"], color=colors, edgecolor="black")
    ax_hand.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax_hand.set_ylabel("HAND (m)")
    ax_hand.set_title("Element Height (mean raw HAND per column)")
    ax_hand.grid(True, alpha=0.3, axis="y")

    # DTND
    ax_dtnd.bar(x, params["dtnd_m"], color=colors, edgecolor="black")
    ax_dtnd.set_ylabel("DTND (m)")
    ax_dtnd.set_title("Element Distance (trap-fit-derived)")
    ax_dtnd.grid(True, alpha=0.3, axis="y")

    # Area
    area_km2 = params["area_m2"] / 1e6
    ax_area.bar(x, area_km2, color=colors, edgecolor="black")
    ax_area.set_ylabel("Area (km²)")
    ax_area.set_title("Element Area (from pixel counts)")
    ax_area.grid(True, alpha=0.3, axis="y")

    # Width
    ax_width.bar(x, params["width_m"], color=colors, edgecolor="black")
    ax_width.set_ylabel("Width (m)")
    ax_width.set_title("Element Width (trap-fit-derived)")
    ax_width.grid(True, alpha=0.3, axis="y")

    for ax in [ax_hand, ax_dtnd, ax_area, ax_width]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    legend_elements = [
        Patch(facecolor="#3498db", edgecolor="black", label="Lake (NWI)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Flood zone"),
        Patch(facecolor="#0d6b2c", edgecolor="black", label="True B1 (cleaned bin 1)"),
        Patch(facecolor="#95a5a6", edgecolor="black", label="Hillslope (B2-B16)"),
    ]
    ax_hand.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fb = (
        " [DEGENERATE TRAP FIT — using median DTND fallback]"
        if params["fallback_used"]
        else ""
    )
    fig.suptitle(
        f"Setup {setup}: Hypothetical Hillslope Parameters ({n} columns){fb}\n"
        f"trap_slope={params['trap_slope']:.4f}, "
        f"trap_width={params['trap_width']:.0f}m, "
        f"trap_area={params['trap_area']:.2e}m²",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    data = load_and_filter()

    # Lake column parameters (computed once, shared across setups)
    print()
    lake_col = compute_lake_column(data["water_mask_2d"])

    # Two setups (Setup 3 removed — chain-monotonicity issues with equal-count
    # bins and the small lake-DTND PI direction made it impractical for the
    # current PI conversation. The setup3 logic remains in make_setup_edges()
    # for future reference but is not executed.)
    summary = {"lake_column": lake_col, "setups": {}}
    for setup in [1, 2]:
        print()
        print(f"=== Setup {setup} ===")
        params = compute_setup_params(setup, data, lake_col)
        plot_path = OUT / f"{3 + setup}_hypothetical_params_setup{setup}.png"
        plot_setup(setup, params, plot_path)

        # Also print a summary table
        print(
            f"  {'#':>3} {'label':<14} {'n':>10} {'HAND (m)':>10} "
            f"{'DTND (m)':>10} {'area (km²)':>11} {'width (m)':>10}"
        )
        for i in range(len(params["labels"])):
            print(
                f"  {i:>3} {params['labels'][i]:<14} "
                f"{int(params['n_pixels'][i]):>10,} "
                f"{params['hand_m'][i]:>10.3f} "
                f"{params['dtnd_m'][i]:>10.1f} "
                f"{params['area_m2'][i] / 1e6:>11.3f} "
                f"{params['width_m'][i]:>10.1f}"
            )

        # Verify chain monotonicity
        d = params["dtnd_m"]
        viols = []
        for i in range(1, len(d)):
            if d[i] <= d[i - 1]:
                viols.append(
                    (
                        i,
                        params["labels"][i - 1],
                        params["labels"][i],
                        float(d[i - 1]),
                        float(d[i]),
                    )
                )
        if viols:
            print(f"  DTND monotonicity violations: {len(viols)}")
            for i, prev_lbl, lbl, p, c in viols:
                print(f"    pos {i}: {prev_lbl}({p:.1f}) → {lbl}({c:.1f})")
        else:
            print("  DTND chain monotonic ✓")

        # Save numeric summary
        summary["setups"][f"setup{setup}"] = {
            "labels": params["labels"],
            "n_pixels": [int(x) for x in params["n_pixels"]],
            "hand_m": [float(x) for x in params["hand_m"]],
            "dtnd_m": [float(x) for x in params["dtnd_m"]],
            "area_m2": [float(x) for x in params["area_m2"]],
            "width_m": [float(x) for x in params["width_m"]],
            "trap_slope": params["trap_slope"],
            "trap_width": params["trap_width"],
            "trap_area": params["trap_area"],
            "fallback_used": bool(params["fallback_used"]),
            "n_columns": len(params["labels"]),
        }

    # Write summary JSON
    summary_path = OUT / "hypothetical_setups_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved {summary_path}")


if __name__ == "__main__":
    main()
