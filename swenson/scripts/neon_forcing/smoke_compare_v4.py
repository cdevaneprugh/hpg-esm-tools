#!/usr/bin/env python3
"""smoke_compare_v4.py -- I6/I7 ingestion-smoke output check: custom-forcing vs v4.

Compares CLM h0a (monthly) output of the two NEON smoke cases over their 2018-2019
overlap:

  custom : $CASES/osbs.swenson.neon-custom-smoke   (custom 101-file stream, 2017-02..2025-06)
  v4     : $CASES/osbs.swenson.neon-v4-smoke        (pre-built v4 stream, 2018-2019 run)

For 2018-2024 the two forcing datasets are byte-identical on measured timesteps (see I4),
so forcing-driven fields (TBOT, FSDS, FLDS, RAIN, TSA) should match to ~machine precision --
the real proof the custom stream is ingested the same way v4 is. Prognostic fields (GPP,
FSH, EFLX_LH_TOT, TWS, H2OSOI, ELAI) differ because the runs used different cold-start dates
(custom 2017-02, v4 2018-01 -> ~11 mo extra spinup in custom, through the 2017 wet season);
they should be physically comparable (same seasonal cycle, offset), not identical.

Env: ctsm (python 3.12 / xarray / matplotlib). Writes a 6-panel PNG to
output/osbs/2026-08-15_neon-custom-smoke/. Archives are ephemeral (case run dirs);
the canonical verdict lives in phases/I-neon-forcing.md (2026-08-15 Log entry).
"""

import glob
import os
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

warnings.filterwarnings("ignore")

ARCH = "/blue/gerber/cdevaneprugh/earth_model_output/cime_output_root/archive"
CUSTOM = f"{ARCH}/osbs.swenson.neon-custom-smoke/lnd/hist"
V4 = f"{ARCH}/osbs.swenson.neon-v4-smoke/lnd/hist"
OUTDIR = "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/output/osbs/2026-08-15_neon-custom-smoke"

MONTHS = [f"{y}-{m:02d}" for y in (2018, 2019) for m in range(1, 13)]
# (var, class): F = forcing-driven (expect ~identical), P = prognostic (expect comparable)
VARS = [
    ("TBOT", "F"),
    ("FSDS", "F"),
    ("FLDS", "F"),
    ("RAIN", "F"),
    ("TSA", "F"),
    ("GPP", "P"),
    ("FSH", "P"),
    ("EFLX_LH_TOT", "P"),
    ("FIRA", "P"),
    ("FSA", "P"),
    ("TWS", "P"),
    ("H2OSOI", "P"),
    ("TSOI", "P"),
    ("ELAI", "P"),
]


def series(histdir):
    """Return {var: np.array(24)} of monthly means, opening each h0a file once."""
    out = {v: [] for v, _ in VARS}
    for ym in MONTHS:
        fs = glob.glob(f"{histdir}/*.clm2.h0a.{ym}.nc")
        if not fs:
            for v, _ in VARS:
                out[v].append(np.nan)
            continue
        ds = xr.open_dataset(fs[0], decode_times=False)
        for v, _ in VARS:
            out[v].append(
                float(ds[v].values.astype("float64").mean()) if v in ds else np.nan
            )
        ds.close()
    return {v: np.array(out[v]) for v, _ in VARS}


def main():
    s4, sc = series(V4), series(CUSTOM)
    print(
        f"{'var':13s}{'cls':4s}{'v4 mean':>12s}{'custom mean':>13s}"
        f"{'abs diff':>12s}{'rel %':>9s}{'max|Δmon|':>12s}"
    )
    print("-" * 75)
    for var, cls in VARS:
        a, b = s4[var], sc[var]
        if np.all(np.isnan(a)) or np.all(np.isnan(b)):
            print(f"{var:13s}{cls:4s}{'(absent)':>25s}")
            continue
        m4, mc = np.nanmean(a), np.nanmean(b)
        adiff = mc - m4
        rel = 100 * adiff / m4 if m4 else float("nan")
        print(
            f"{var:13s}{cls:4s}{m4:12.5g}{mc:13.5g}{adiff:12.4g}{rel:9.3f}"
            f"{np.nanmax(np.abs(b - a)):12.4g}"
        )
    return s4, sc


def plot(s4, sc):
    os.makedirs(OUTDIR, exist_ok=True)
    keys = [
        ("TBOT", "F"),
        ("FSDS", "F"),
        ("GPP", "P"),
        ("FSH", "P"),
        ("EFLX_LH_TOT", "P"),
        ("H2OSOI", "P"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(MONTHS))
    for ax, (var, cls) in zip(axes.flat, keys):
        ax.plot(x, s4[var], "o-", ms=3, label="v4", color="#4C78A8")
        ax.plot(x, sc[var], "s--", ms=3, label="custom", color="#E45756")
        ax.set_title(f"{var} ({'forcing-driven' if cls == 'F' else 'prognostic'})")
        ax.set_xlabel("month idx (2018-01 .. 2019-12)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "NEON custom-forcing smoke vs v4 smoke — 2018-2019 monthly means (OSBS)"
    )
    fig.tight_layout()
    out = f"{OUTDIR}/compare_v4_custom_2018-2019.png"
    fig.savefig(out, dpi=110)
    print(f"\nplot: {out}")


if __name__ == "__main__":
    a, b = main()
    plot(a, b)
