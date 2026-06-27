"""Categorical spatial map of the proposed bin structure.

Layers (bottom to top):
1. Hillslope (raw HAND > 0.1m): gray background
2. Flood zone (raw HAND ≤ 0): OrRd gradient on depression_fill_depth
3. Near-stream (raw HAND ∈ (0, 0.1m]): solid green
4. NWI water (top, hard mask): solid blue

Renders at full resolution from the saved 2026-04-24 diagnostic arrays.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

REPO = Path("/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson")
SOURCE = REPO / "output/osbs/2026-04-24_diagnostic/diagnostics"
OUT = REPO / "output/osbs/HAND-diagnostic-2026-04-25"

print("Loading arrays...")
hand = np.load(SOURCE / "hand.npy")
flooded_orig = np.load(SOURCE / "flooded_orig.npy")
pit_filled = np.load(SOURCE / "pit_filled.npy")
water_mask = np.load(SOURCE / "water_mask.npy")
print(f"  shape={hand.shape}")

dep_fill = flooded_orig - pit_filled
raw_hand = hand - dep_fill

# Categorical masks
land = (water_mask == 0) & np.isfinite(hand) & (hand > 0)
hillslope_mask = land & (raw_hand > 0.1)
ns_mask = land & (raw_hand > 0) & (raw_hand <= 0.1)
fz_mask = land & (raw_hand <= 0)
nwi_mask = water_mask > 0

print(f"  hillslope: {int(hillslope_mask.sum()):>10,}")
print(f"  near-stream: {int(ns_mask.sum()):>10,}")
print(f"  flood zone: {int(fz_mask.sum()):>10,}")
print(f"  NWI water:  {int(nwi_mask.sum()):>10,}")

# Build per-layer overlays
ns_overlay = np.where(ns_mask, 1.0, np.nan)
nwi_overlay = np.where(nwi_mask, 1.0, np.nan)

# Hillslope uses raw HAND for gradient (light gray = low HAND, dark gray = ridge)
hillslope_values = np.where(hillslope_mask, raw_hand, np.nan)

# Flood zone uses depression_fill_depth for gradient
fz_values = np.where(fz_mask, dep_fill, np.nan)

# Truncated colormaps: skip the near-white ends so layers stand out on the map
greys_trunc = LinearSegmentedColormap.from_list(
    "Greys_trunc", mpl.colormaps["Greys"](np.linspace(0.2, 0.9, 256))
)
orrd_dark = LinearSegmentedColormap.from_list(
    "OrRd_dark", mpl.colormaps["OrRd"](np.linspace(0.4, 1.0, 256))
)

# Plot
print("Rendering...")
fig, ax = plt.subplots(figsize=(20, 18))

# Layer 1 (bottom): hillslope with gray gradient on raw HAND
hs_im = ax.imshow(
    hillslope_values,
    cmap=greys_trunc,
    vmin=0.1,
    vmax=15.0,
    alpha=1.0,
    interpolation="none",
    aspect="equal",
)
cbar_hs = fig.colorbar(
    hs_im, ax=ax, label="Hillslope HAND (m)", shrink=0.4, pad=0.02, location="left"
)
cbar_hs.ax.tick_params(labelsize=11)

# Layer 2: flood zone with darker OrRd gradient on depression_fill_depth
fz_im = ax.imshow(
    fz_values,
    cmap=orrd_dark,
    vmin=0.1,
    vmax=5.0,
    alpha=0.95,
    interpolation="none",
    aspect="equal",
)
cbar_fz = fig.colorbar(
    fz_im,
    ax=ax,
    label="Flood-zone depth (depression_fill_depth, m)",
    shrink=0.4,
    pad=0.02,
)
cbar_fz.ax.tick_params(labelsize=11)

# Layer 3: near-stream — darker, more saturated green
green_cmap = ListedColormap(["#0d6b2c"])
ax.imshow(
    ns_overlay,
    cmap=green_cmap,
    alpha=1.0,
    interpolation="none",
    aspect="equal",
)

# Layer 4 (TOP, hard mask): NWI water solid blue
blue_cmap = ListedColormap(["#1f77b4"])
ax.imshow(
    nwi_overlay,
    cmap=blue_cmap,
    alpha=1.0,
    interpolation="none",
    aspect="equal",
)

# Legend
legend_elements = [
    Patch(
        facecolor="#1f77b4",
        edgecolor="black",
        label=f"Lake (NWI water, top mask): {int(nwi_mask.sum()):,} px",
    ),
    Patch(
        facecolor="#0d6b2c",
        edgecolor="black",
        label=f"True B1 (raw HAND ∈ (0, 0.1m]): {int(ns_mask.sum()):,} px",
    ),
    Patch(
        facecolor="#990000",
        edgecolor="black",
        label=f"Flood zone (raw HAND ≤ 0, gradient by depth): {int(fz_mask.sum()):,} px",
    ),
    Patch(
        facecolor="#666666",
        edgecolor="black",
        label=f"Hillslope (raw HAND > 0.1m, gradient by elevation): {int(hillslope_mask.sum()):,} px",
    ),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95)

ax.set_xlabel("Pixel column (1m)")
ax.set_ylabel("Pixel row (1m)")

fig.tight_layout()
out_path = OUT / "7_categorical_spatial.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  saved {out_path}")
