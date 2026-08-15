#!/usr/bin/env python3
"""
neon_forcing_qc.py -- production QC for the custom OSBS NEON->DATM forcing (Phase I / I5).

Spot-checks + sniff-tests the generated atm NetCDFs over the whole record, with no
external reference (so it covers the non-v4 months 2017 + 2025 too):

  1. Structural  -- per file: expected ntime (days*48), the 8 physical vars + <VAR>_fqc
                    present, NaN==0 (gap-fill fills everything), time axis strictly
                    monotonic within the month.
  2. Gap-fill    -- per var x month: fraction of gap-filled (fqc>0) timesteps; FLAG a
                    var-month whose gap-fill fraction exceeds GAPFILL_FLAG (instrument
                    outage). Reported as a warning, not a hard fail.
  3. Sanity      -- per var: min/max vs physical-plausibility bounds; a value outside the
                    hard bounds is a FAIL.
  4. Climatology -- whole-record monthly-mean series, a diurnal composite, and a gap-fill
                    heatmap (var x month) -- the eyeball "sniff test".

Complements neon_v4_regression.py (fqc-partitioned custom-vs-v4 on the 2018-2024 slice).
Output contract mirrors merit_regression.py / neon_v4_regression.py: results.json +
summary.txt + PASS/FAIL + sys.exit, plus PNGs.

Env: ctsm (python 3.12 / xarray / numpy / matplotlib).
Run:  python neon_forcing_qc.py [--atm-dir DIR] [--out-dir DIR] [--plot-dir DIR]
Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import calendar
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SWENSON = SCRIPT_DIR.parent.parent  # scripts/neon_forcing -> scripts -> swenson

PHYS_VARS = ["FLDS", "FSDS", "PRECTmms", "RH", "PSRF", "TBOT", "WIND"]
UNITS = {
    "FLDS": "W/m2",
    "FSDS": "W/m2",
    "PRECTmms": "mm/s",
    "RH": "%",
    "PSRF": "Pa",
    "TBOT": "K",
    "WIND": "m/s",
}
# physical-plausibility hard bounds (a value outside these is a FAIL)
RANGE = {
    "FLDS": (120, 600),
    "FSDS": (0, 1400),
    "PRECTmms": (0, 0.05),
    "RH": (0, 100.5),
    "PSRF": (94000, 106000),
    "TBOT": (245, 325),
    "WIND": (0, 40),
}
GAPFILL_FLAG = 0.60  # warn if a var-month is > this fraction gap-filled (fqc>0)


def load_record(atm_dir):
    """Open every OSBS_atm_*.nc; return files, month labels, per-file records, and
    whole-record concatenated value/fqc arrays + hour-of-day."""
    fs = sorted(glob.glob(os.path.join(atm_dir, "OSBS_atm_*.nc")))
    if not fs:
        sys.exit(f"ERROR: no OSBS_atm_*.nc in {atm_dir}")
    per = []
    val = {v: [] for v in PHYS_VARS}
    fqc = {v: [] for v in PHYS_VARS}
    hod_all = []
    for f in fs:
        ym = (
            os.path.basename(f).replace("OSBS_atm_", "").replace(".nc", "")
        )  # "YYYY-MM"
        yr, mo = (int(x) for x in ym.split("-"))
        d = xr.open_dataset(f, decode_times=True)
        n = int(d.time.size)
        t = d.time.values.astype("datetime64[s]")
        # hour-of-day (0..23.5) from decoded time
        hod = ((t - t.astype("datetime64[D]")) / np.timedelta64(30, "m")).astype(
            float
        ) / 2.0
        rec = {
            "ym": ym,
            "ntime": n,
            "ntime_expected": calendar.monthrange(yr, mo)[1] * 48,
            "time_monotonic": bool(np.all(np.diff(t.astype("int64")) > 0)),
        }
        for v in PHYS_VARS:
            x = np.asarray(d[v].values, dtype="float64").ravel()
            fqn = f"{v}_fqc"
            fq = (
                np.asarray(d[fqn].values, dtype="float64").ravel()
                if fqn in d
                else np.full(n, np.nan)
            )
            val[v].append(x)
            fqc[v].append(fq)
            rec[f"{v}_nan"] = int(np.isnan(x).sum())
            rec[f"{v}_gapfrac"] = round(float(np.mean(fq > 0)), 4) if fqn in d else None
            rec[f"{v}_min"] = (
                float(np.nanmin(x)) if np.any(np.isfinite(x)) else float("nan")
            )
            rec[f"{v}_max"] = (
                float(np.nanmax(x)) if np.any(np.isfinite(x)) else float("nan")
            )
            rec[f"{v}_fqc_present"] = fqn in d
        hod_all.append(hod)
        per.append(rec)
        d.close()
    months = [r["ym"] for r in per]
    arr = {v: np.concatenate(val[v]) for v in PHYS_VARS}
    fq = {v: np.concatenate(fqc[v]) for v in PHYS_VARS}
    return fs, months, per, arr, fq, np.concatenate(hod_all)


def structural_check(per):
    """Return (n_fail, list of failure strings)."""
    fails = []
    for r in per:
        if r["ntime"] != r["ntime_expected"]:
            fails.append(
                f"{r['ym']}: ntime {r['ntime']} != expected {r['ntime_expected']}"
            )
        if not r["time_monotonic"]:
            fails.append(f"{r['ym']}: time axis not strictly monotonic")
        for v in PHYS_VARS:
            if not r[f"{v}_fqc_present"]:
                fails.append(f"{r['ym']}: missing {v}_fqc")
            if r[f"{v}_nan"] > 0:
                fails.append(
                    f"{r['ym']}: {v} has {r[f'{v}_nan']} NaN (gap-fill incomplete)"
                )
    return len(fails), fails


def sanity_check(arr):
    """Per-var min/max vs hard bounds. Return (n_fail, list of failure strings, stats)."""
    fails = []
    stats = {}
    for v in PHYS_VARS:
        x = arr[v][np.isfinite(arr[v])]
        lo, hi = RANGE[v]
        vmin, vmax = float(x.min()), float(x.max())
        n_below = int(np.sum(x < lo))
        n_above = int(np.sum(x > hi))
        stats[v] = {
            "min": vmin,
            "max": vmax,
            "bound": [lo, hi],
            "n_below": n_below,
            "n_above": n_above,
            "mean": float(x.mean()),
        }
        if n_below or n_above:
            fails.append(
                f"{v}: {n_below} < {lo} and {n_above} > {hi} "
                f"(observed [{vmin:.4g}, {vmax:.4g}] {UNITS[v]})"
            )
    return len(fails), fails, stats


def gapfill_flags(per):
    """Var-months exceeding GAPFILL_FLAG (warnings, not fails)."""
    flags = []
    for r in per:
        for v in PHYS_VARS:
            gf = r[f"{v}_gapfrac"]
            if gf is not None and gf > GAPFILL_FLAG:
                flags.append(f"{r['ym']} {v}: {100 * gf:.0f}% gap-filled")
    return flags


def make_plots(months, per, arr, fq, hod, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    saved = []

    # Fig 1: gap-fill fraction heatmap (var x month)
    M = np.array([[(r[f"{v}_gapfrac"] or 0.0) for r in per] for v in PHYS_VARS])
    fig, ax = plt.subplots(figsize=(max(10, len(months) * 0.12), 4.5))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(PHYS_VARS)))
    ax.set_yticklabels(PHYS_VARS, fontsize=9)
    step = max(1, len(months) // 20)
    ax.set_xticks(range(0, len(months), step))
    ax.set_xticklabels(
        [months[i] for i in range(0, len(months), step)], rotation=90, fontsize=7
    )
    ax.set_title(
        "Gap-fill fraction (fqc>0) by variable and month",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="fraction gap-filled")
    plt.tight_layout()
    p = os.path.join(plot_dir, "gapfill_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Fig 2: whole-record monthly-mean series (7 panels)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    xs = np.arange(len(months))
    step = max(1, len(months) // 16)
    for ax, v in zip(axes.ravel(), PHYS_VARS):
        mm = [
            float(
                np.nanmean(
                    arr[v][
                        sum(r["ntime"] for r in per[:i]) : sum(
                            r["ntime"] for r in per[: i + 1]
                        )
                    ]
                )
            )
            for i in range(len(per))
        ]
        ax.plot(xs, mm, "-", color="#1f77b4", lw=1.5)
        ax.set_title(f"{v} monthly mean [{UNITS[v]}]", fontsize=11)
        ax.set_xticks(range(0, len(months), step))
        ax.set_xticklabels(
            [months[i] for i in range(0, len(months), step)], rotation=90, fontsize=7
        )
        ax.grid(True, alpha=0.3, linestyle="--")
    axes.ravel()[-1].axis("off")
    fig.suptitle(
        "OSBS custom forcing -- monthly means (whole record)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    p = os.path.join(plot_dir, "monthly_means.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Fig 3: diurnal composite (mean by hour-of-day, whole record)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    hbins = np.arange(0, 24, 0.5)
    for ax, v in zip(axes.ravel(), PHYS_VARS):
        comp = [float(np.nanmean(arr[v][np.isclose(hod, h)])) for h in hbins]
        ax.plot(hbins, comp, "-o", ms=3, color="#1f77b4", lw=1.5)
        ax.set_title(f"{v} diurnal composite [{UNITS[v]}]", fontsize=11)
        ax.set_xlabel("hour of day (UTC)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes.ravel()[-1].axis("off")
    fig.suptitle(
        "OSBS custom forcing -- diurnal composite (whole record)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    p = os.path.join(plot_dir, "diurnal_composite.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    return saved


def main():
    ap = argparse.ArgumentParser(
        description="Production QC for custom OSBS forcing (Phase I / I5)."
    )
    ap.add_argument(
        "--atm-dir", default=str(SWENSON / "data/datm/neon_OSBS/custom/OSBS/atm")
    )
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "output"))
    ap.add_argument(
        "--plot-dir", default=str(SWENSON / "output/osbs/2026-08-14_forcing_qc")
    )
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    fs, months, per, arr, fq, hod = load_record(a.atm_dir)
    n_struct, struct_fails = structural_check(per)
    n_sane, sane_fails, stats = sanity_check(arr)
    gflags = gapfill_flags(per)
    overall = "PASS" if (n_struct == 0 and n_sane == 0) else "FAIL"

    os.makedirs(a.out_dir, exist_ok=True)
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test": "neon_forcing_qc",
        "atm_dir": a.atm_dir,
        "n_files": len(fs),
        "months": [months[0], months[-1]],
        "structural_failures": struct_fails,
        "sanity_failures": sane_fails,
        "sanity_stats": stats,
        "gapfill_flags": gflags,
        "result": overall,
    }
    with open(os.path.join(a.out_dir, "qc_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    L = [
        "NEON custom forcing QC (Phase I / I5)",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Files: {len(fs)}  ({months[0]} .. {months[-1]})   dir: {a.atm_dir}",
        "",
        f"[1] Structural: {'OK' if n_struct == 0 else str(n_struct) + ' FAIL'}"
        "  (ntime=days*48, fqc present, NaN==0, time monotonic)",
    ]
    L += ["    " + s for s in struct_fails[:20]]
    L += [
        "",
        f"[2] Physical sanity: {'OK' if n_sane == 0 else str(n_sane) + ' FAIL'}",
        f"    {'var':<9s}{'min':>12s}{'max':>12s}{'mean':>12s}   bounds",
    ]
    for v in PHYS_VARS:
        s = stats[v]
        L.append(
            f"    {v:<9s}{s['min']:>12.4g}{s['max']:>12.4g}{s['mean']:>12.4g}   {s['bound']}"
        )
    L += [
        "",
        f"[3] Gap-fill flags (> {int(100 * GAPFILL_FLAG)}% gap-filled), {len(gflags)} var-months:",
    ]
    L += ["    " + s for s in gflags[:40]]
    if len(gflags) > 40:
        L.append(f"    ... +{len(gflags) - 40} more")
    L += ["", f"RESULT: {overall}"]
    text = "\n".join(L)
    with open(os.path.join(a.out_dir, "qc_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)

    if not a.no_plots:
        for p in make_plots(months, per, arr, fq, hod, a.plot_dir):
            print(f"Saved {p}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
