#!/usr/bin/env python3
"""
plot_col_timeseries.py - Full-span column-level time series by hillslope group

PURPOSE:
    Plot a column-level variable over the full simulation span, grouped by
    Lake / Flood-zone / Upland. Designed for spinup analysis of the Phase E.5+
    hillslope file structure (lake column at chain index 1, flood-zone bins
    at HAND < 0, upland bins at HAND >= 0).

USAGE:
    python3 plot_col_timeseries.py <input_file> <output_file> <variable>

ARGUMENTS:
    input_file    Annual-binned h1 NetCDF file (e.g., h1a_annual.nc)
    output_file   Output PNG filename
    variable      Variable name to plot (e.g., ZWT, H2OSFC, GPP)

EXAMPLE:
    python3 plot_col_timeseries.py h1a_annual.nc plots/ZWT.png ZWT

OUTPUT:
    2-panel figure:
    - Top:    Group-weighted means (Lake / FZ / Upland) + gridcell weighted
              mean (dashed black). Weights from cols1d_wtgcell, renormalized
              within each group.
    - Bottom: Every hillslope column traced individually, colored by group.

GROUPING:
    Lake:    column index 0 (the prepended lake column).
    FZ:      hillslope_index > 0 (excluding lake) AND hillslope_elev < 0.
    Upland:  hillslope_index > 0 AND hillslope_elev >= 0.
    Bareground: hillslope_index = -9999 — excluded from groups AND from
                gridcell weighted mean.

COLOR SCHEME (matches Phase E.5 hillslope_params plot):
    Lake:   lightskyblue (#87CEFA)
    FZ:     darkred (#8B0000)
    Upland: seafoam green (#66C2A5, matplotlib Set2[0])
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Group colors (match Phase E.5 hillslope_params plot)
COLOR_LAKE = "lightskyblue"
COLOR_FZ = "darkred"
COLOR_UPLAND = "#66C2A5"  # matplotlib Set2[0]

GRIDCELL_LINE_KW = dict(color="black", linestyle="--", linewidth=2, alpha=0.7)


def classify_columns(ds):
    """Return dict of group -> array of column indices.

    Lake = column index 0 (the prepended lake column).
    FZ   = hillslope columns at HAND < 0 (excluding the lake at index 0).
    Upland = hillslope columns at HAND >= 0.

    Bareground (hillslope_index = -9999) is intentionally NOT returned;
    callers should handle it separately or ignore.
    """
    h_idx = ds["hillslope_index"].values
    elev = ds["hillslope_elev"].values

    lake = np.array([0])  # by convention: lake at chain index 0
    is_hillslope = (h_idx > 0) & (h_idx < 9000)
    fz = np.where(is_hillslope & (elev < 0))[0]
    fz = fz[fz != 0]  # exclude the lake
    upland = np.where(is_hillslope & (elev >= 0))[0]

    return {
        "Lake": lake,
        "Flood zone": fz,
        "Upland": upland,
    }


def weighted_mean(values, weights, col_indices):
    """Renormalized weighted mean of `values[:, col_indices]` using `weights`.

    Returns shape (n_time,). Weights are renormalized within the column subset.
    """
    if len(col_indices) == 0:
        return None
    w = weights[col_indices]
    wsum = w.sum()
    if wsum <= 0:
        return None
    renorm = w / wsum
    return (values[:, col_indices] * renorm).sum(axis=1)


def plot(input_file, output_file, variable):
    ds = xr.open_dataset(input_file, decode_times=False)

    if variable not in ds:
        print(f"ERROR: Variable '{variable}' not found in {input_file}")
        print(f"Available variables: {list(ds.data_vars)}")
        ds.close()
        sys.exit(1)

    var_data = ds[variable].values
    if var_data.ndim != 2:
        print(f"ERROR: Expected 2D (time, column), got shape {var_data.shape}")
        ds.close()
        sys.exit(1)

    units = ds[variable].attrs.get("units", "")
    elev = ds["hillslope_elev"].values
    weights = ds["cols1d_wtgcell"].values
    h_idx = ds["hillslope_index"].values

    # Year axis: derive from mcdate (// 10000 gives year for YYYYMMDD format).
    # The annual binning preserves a sensible mcdate per bin.
    years = (ds["mcdate"].values // 10000).astype(int)

    ds.close()

    groups = classify_columns_from_arrays(h_idx, elev)

    # Gridcell weighted mean over hillslope-only columns (exclude bareground).
    is_hillslope = (h_idx > 0) & (h_idx < 9000)
    hillslope_cols = np.where(is_hillslope)[0]
    gridcell_avg = weighted_mean(var_data, weights, hillslope_cols)

    group_color = {
        "Lake": COLOR_LAKE,
        "Flood zone": COLOR_FZ,
        "Upland": COLOR_UPLAND,
    }

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ylabel = f"{variable} ({units})" if units else variable

    # --- Top panel: group-weighted means + gridcell mean ---
    if gridcell_avg is not None:
        ax_top.plot(
            years, gridcell_avg, label="Hillslope-weighted mean", **GRIDCELL_LINE_KW
        )
    for name, cols in groups.items():
        gmean = weighted_mean(var_data, weights, cols)
        if gmean is None:
            continue
        ax_top.plot(
            years,
            gmean,
            color=group_color[name],
            linewidth=2.5,
            alpha=0.9,
            label=f"{name} (n={len(cols)})",
        )
    ax_top.set_ylabel(ylabel, fontsize=12)
    ax_top.set_title(
        f"{variable} — group-weighted means (annual)", fontsize=13, fontweight="bold"
    )
    ax_top.grid(True, alpha=0.3, linestyle="--")
    ax_top.legend(loc="best", fontsize=10)

    # --- Bottom panel: individual column traces colored by group ---
    if gridcell_avg is not None:
        ax_bot.plot(
            years, gridcell_avg, label="Hillslope-weighted mean", **GRIDCELL_LINE_KW
        )

    legend_seen = set()
    for name, cols in groups.items():
        c = group_color[name]
        for col_idx in cols:
            label = name if name not in legend_seen else None
            legend_seen.add(name)
            ax_bot.plot(
                years,
                var_data[:, col_idx],
                color=c,
                linewidth=0.9,
                alpha=0.6,
                label=label,
            )

    ax_bot.set_xlabel("Simulation year", fontsize=12)
    ax_bot.set_ylabel(ylabel, fontsize=12)
    ax_bot.set_title(
        "Individual columns (color = group; line per column)",
        fontsize=13,
        fontweight="bold",
    )
    ax_bot.grid(True, alpha=0.3, linestyle="--")
    ax_bot.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_lake = len(groups["Lake"])
    n_fz = len(groups["Flood zone"])
    n_upland = len(groups["Upland"])
    n_bare = int(np.sum(h_idx == -9999))
    print(f"Saved: {output_file}")
    print(
        f"  Columns: {n_lake} lake + {n_fz} FZ + {n_upland} upland "
        f"(+ {n_bare} bareground excluded)"
    )


def classify_columns_from_arrays(h_idx, elev):
    """Same as classify_columns(ds) but operates on raw arrays."""
    lake = np.array([0])
    is_hillslope = (h_idx > 0) & (h_idx < 9000)
    fz = np.where(is_hillslope & (elev < 0))[0]
    fz = fz[fz != 0]
    upland = np.where(is_hillslope & (elev >= 0))[0]
    return {"Lake": lake, "Flood zone": fz, "Upland": upland}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-span column-level time series, grouped by Lake/FZ/Upland",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s h1a_annual.nc ZWT.png ZWT
  %(prog)s h1a_annual.nc H2OSFC.png H2OSFC
  %(prog)s h1a_annual.nc GPP_by_col.png GPP
        """,
    )
    parser.add_argument("input_file", help="Annual-binned h1 NetCDF file")
    parser.add_argument("output_file", help="Output PNG filename")
    parser.add_argument(
        "variable", help="Variable name to plot (e.g., ZWT, H2OSFC, GPP)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Variable: {args.variable}")
    print()
    plot(args.input_file, args.output_file, args.variable)


if __name__ == "__main__":
    main()
