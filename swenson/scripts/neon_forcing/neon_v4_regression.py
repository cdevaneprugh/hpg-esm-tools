#!/usr/bin/env python3
"""
neon_v4_regression.py -- reproduce-v4 fidelity check for the OSBS NEON->DATM pipeline
(Phase I, task I4). Compares our offline-generated "custom" atm forcing against the
pre-built NCAR-NEON "v4" forcing for one calendar year (default 2018), per physical
variable: bulk RMS / max-delta / correlation / bias, plus the sanctioned fqc-partitioned
split (both-measured vs gap-filled). Writes results.json + summary.txt + a scatter plot,
all to one output dir (default scripts/neon_forcing/output/).

Framing: an exact match is NOT expected -- v4 is a late-2025 NEON release (provisional
on); we use RELEASE-2026, released-only. This is a *methodology-fidelity* check. The I2
v3-vs-v4 RMS-delta band is printed for context; the verdict uses generous physical-scale
thresholds on the both-measured RMS (a hard I2-band pass/fail would false-fail because
custom-vs-v4 carries our pipeline's difference on top of the release difference).

Model / conventions after scripts/merit_validation/merit_regression.py (results.json +
summary.txt + PASS/FAIL + sys.exit) and the repo matplotlib house style.

Env: ctsm (python 3.12 / xarray / numpy / matplotlib).
Run:  python neon_v4_regression.py [--year 2018]
Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
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

# Physical forcing variables (ZBOT is a constant tower height -> reported separately).
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
DERIVED = {"FLDS", "FSDS", "RH", "PSRF"}  # expect near-identical
MEASURED = {"TBOT", "WIND", "PRECTmms"}  # expect reprocessing scale

# I2 v3-vs-v4 RMS-delta reference band (context only, NOT the pass threshold).
REF_BAND = {
    "TBOT": 0.17,
    "PSRF": 9.0,
    "FSDS": 0.07,
    "FLDS": 0.09,
    "PRECTmms": 3e-4,
    "RH": 0.21,
    "WIND": 0.025,
}

# Generous physical-scale pass thresholds on the both-measured RMS (documented heuristic:
# ~2-10x the smoke preview; passes reprocessing-scale drift, fails a real pipeline bug).
PASS_RMS = {
    "FSDS": 1.0,
    "FLDS": 5.0,
    "RH": 2.0,
    "PSRF": 50.0,
    "TBOT": 1.0,
    "WIND": 0.5,
    "PRECTmms": 5e-3,
}
PRECIP_TOTAL_TOL_PCT = 5.0  # annual precip total must also agree within this


def _flat(ds, var):
    return np.asarray(ds[var].values, dtype="float64").ravel()


def load_year(custom_dir, v4_dir, year):
    """Concatenate custom & v4 atm across all 12 months; per-var value + fqc arrays."""
    months = [f"{year}-{m:02d}" for m in range(1, 13)]
    acc = {v: {"c": [], "v": [], "cf": [], "vf": []} for v in PHYS_VARS}
    ntime, zbot = {}, {}
    for ym in months:
        cp = os.path.join(custom_dir, f"OSBS_atm_{ym}.nc")
        vp = os.path.join(v4_dir, f"OSBS_atm_{ym}.nc")
        if not os.path.exists(cp) or not os.path.exists(vp):
            sys.exit(
                f"ERROR: missing {ym}: custom={os.path.exists(cp)} v4={os.path.exists(vp)}"
            )
        c = xr.open_dataset(cp, decode_times=False)
        v = xr.open_dataset(vp, decode_times=False)
        if c.time.size != v.time.size:
            sys.exit(
                f"ERROR {ym}: time size mismatch custom={c.time.size} v4={v.time.size}"
            )
        if not np.array_equal(np.asarray(c.time.values), np.asarray(v.time.values)):
            sys.exit(
                f"ERROR {ym}: time-axis values differ -- element-wise diff invalid"
            )
        ntime[ym] = int(c.time.size)
        if "ZBOT" in c and "ZBOT" in v:
            zbot[ym] = (float(_flat(c, "ZBOT")[0]), float(_flat(v, "ZBOT")[0]))
        for var in PHYS_VARS:
            acc[var]["c"].append(_flat(c, var))
            acc[var]["v"].append(_flat(v, var))
            fq = f"{var}_fqc"
            acc[var]["cf"].append(_flat(c, fq) if fq in c else np.zeros(c.time.size))
            acc[var]["vf"].append(_flat(v, fq) if fq in v else np.zeros(v.time.size))
        c.close()
        v.close()
    arrs = {
        var: {k: np.concatenate(acc[var][k]) for k in ("c", "v", "cf", "vf")}
        for var in PHYS_VARS
    }
    return arrs, ntime, zbot, months


def metrics(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return {
            "n": 0,
            "rms": float("nan"),
            "bias": float("nan"),
            "maxabs": float("nan"),
            "corr": float("nan"),
        }
    d = a - b
    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 0 and np.std(b) > 0
        else float("nan")
    )
    return {
        "n": int(a.size),
        "rms": float(np.sqrt(np.mean(d**2))),
        "bias": float(np.mean(d)),
        "maxabs": float(np.max(np.abs(d))),
        "corr": corr,
    }


def analyze(arrs):
    res = {}
    for var in PHYS_VARS:
        c, v = arrs[var]["c"], arrs[var]["v"]
        cf, vf = arrs[var]["cf"], arrs[var]["vf"]
        both = (cf == 0) & (vf == 0)
        e = {
            "units": UNITS[var],
            "class": "derived" if var in DERIVED else "measured",
            "ref_rms": REF_BAND.get(var),
            "frac_both_measured": round(float(both.mean()), 4),
            "bulk": metrics(c, v),
            "both_measured": metrics(c[both], v[both]),
            "gap_filled": metrics(c[~both], v[~both]),
        }
        if var == "PRECTmms":  # totals in mm: mm/s * 1800 s per 30-min step
            ct, vt = float(np.nansum(c) * 1800.0), float(np.nansum(v) * 1800.0)
            e["annual_total_mm"] = {
                "custom": round(ct, 1),
                "v4": round(vt, 1),
                "pct_diff": round((ct - vt) / vt * 100, 2) if vt else float("nan"),
            }
        res[var] = e
    return res


def verdict(res):
    any_fail = False
    for var, e in res.items():
        rms = e["both_measured"]["rms"]
        ok = bool(np.isfinite(rms) and rms <= PASS_RMS[var])
        if var == "PRECTmms":
            pd = abs(e.get("annual_total_mm", {}).get("pct_diff", 1e9))
            ok = ok and (pd <= PRECIP_TOTAL_TOL_PCT)
        e["status"] = "PASS" if ok else "FAIL"
        any_fail = any_fail or not ok
    return "FAIL" if any_fail else "PASS"


def write_results(res, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    overall = verdict(res)
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test": "neon_v4_regression",
        **meta,
        "variables": res,
        "result": overall,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    L = [
        "NEON custom-vs-v4 forcing regression (Phase I / I4)",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Year: {meta['year']}   months: {meta['n_months']}   "
        f"custom: {meta['custom_dir']}",
        "",
        "Reproduction of NCAR v4. Not byte-exact by design (RELEASE-2026 vs v4's "
        "late-2025 release);",
        "reprocessing-scale agreement = PASS. RMS in native units. ref = I2 v3-vs-v4 "
        "RMS band (context).",
        "",
        f"{'var':<9s}{'unit':<6s}{'class':<9s}{'corr':>9s}{'RMS.bulk':>10s}"
        f"{'RMS.meas':>10s}{'RMS.gap':>10s}{'ref':>9s}{'%meas':>7s}  status",
        "-" * 88,
    ]
    for var in PHYS_VARS:
        e = res[var]
        L.append(
            f"{var:<9s}{e['units']:<6s}{e['class']:<9s}"
            f"{e['bulk']['corr']:>9.5f}{e['bulk']['rms']:>10.4g}"
            f"{e['both_measured']['rms']:>10.4g}{e['gap_filled']['rms']:>10.4g}"
            f"{(e['ref_rms'] if e['ref_rms'] is not None else float('nan')):>9.4g}"
            f"{100 * e['frac_both_measured']:>6.1f}%  {e['status']}"
        )
    if "annual_total_mm" in res["PRECTmms"]:
        t = res["PRECTmms"]["annual_total_mm"]
        L += [
            "",
            f"Precip annual total: custom {t['custom']} mm | v4 {t['v4']} mm | "
            f"diff {t['pct_diff']}%",
        ]
    if meta.get("zbot"):
        zc, zv = next(iter(meta["zbot"].values()))
        L.append(f"ZBOT (constant tower height): custom {zc} m | v4 {zv} m")
    L += [
        "",
        "RMS.meas = both-measured (fqc==0 both) = cleanest fidelity metric; "
        "RMS.gap = gap-filled (drift expected).",
        f"RESULT: {overall}",
    ]
    text = "\n".join(L)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return overall


def make_plots(arrs, res, plot_dir, year):
    os.makedirs(plot_dir, exist_ok=True)
    # Scatter grid: all points colored measured (on 1:1) vs gap-filled (diverge).
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, var in zip(axes.ravel(), PHYS_VARS):
        c, v = arrs[var]["c"], arrs[var]["v"]
        both = (arrs[var]["cf"] == 0) & (arrs[var]["vf"] == 0)
        fin = np.isfinite(c) & np.isfinite(v)
        mm, gg = fin & both, fin & ~both
        ax.scatter(
            v[gg],
            c[gg],
            s=3,
            alpha=0.3,
            color="#E08A1E",
            rasterized=True,
            label="gap-filled",
        )
        ax.scatter(
            v[mm],
            c[mm],
            s=2,
            alpha=0.3,
            color="#1f77b4",
            rasterized=True,
            label="measured",
        )
        lo = float(np.nanmin([v[fin].min(), c[fin].min()]))
        hi = float(np.nanmax([v[fin].max(), c[fin].max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_title(
            f"{var}  (bulk r={res[var]['bulk']['corr']:.4f}, "
            f"meas RMS={res[var]['both_measured']['rms']:.2g})",
            fontsize=11,
        )
        ax.set_xlabel(f"v4 [{UNITS[var]}]", fontsize=9)
        ax.set_ylabel(f"custom [{UNITS[var]}]", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(markerscale=4, fontsize=9, loc="upper left")
    fig.suptitle(
        f"OSBS custom vs v4 atm forcing, {year} "
        "(measured points overlie 1:1; only gap-filled diverge)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    p1 = os.path.join(plot_dir, f"scatter_custom_vs_v4_{year}.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1}")


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce-v4 fidelity check (Phase I / I4)."
    )
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument(
        "--custom-dir", default=str(SWENSON / "data/datm/neon_OSBS/custom/OSBS/atm")
    )
    ap.add_argument("--v4-dir", default=str(SWENSON / "data/datm/neon_OSBS/v4/OSBS"))
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "output"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    arrs, ntime, zbot, months = load_year(a.custom_dir, a.v4_dir, a.year)
    res = analyze(arrs)
    meta = {
        "year": a.year,
        "n_months": len(months),
        "months": months,
        "custom_dir": a.custom_dir,
        "v4_dir": a.v4_dir,
        "ntime": ntime,
        "zbot": zbot,
    }
    overall = write_results(res, meta, a.out_dir)
    if not a.no_plots:
        make_plots(arrs, res, a.out_dir, a.year)
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
