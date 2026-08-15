#!/usr/bin/env python3
"""
splice_2017_precip.py -- recover the 2017 precip gap from the secondary gauge (Phase I).

NEON's primary rain gauge at OSBS (weighing gauge, DP1.00044) was physically down for
2017 Jul-Dec (all rows NA, NEON finalQF=1), so the generated forcing carries missing
PRECTmms for those months (stored as the fill value -9999.0):

    2017-07  last 206 of 1488 steps   2017-08..11  entire months   2017-12  first 36 steps

The co-located secondary gauge (tipping bucket, DP1.00045) recorded that precipitation
with complete, QC-passed (finalQF=0) 30-minute data. When both gauges run they agree to
~2% on totals (r=0.96), so the tipping bucket is a validated stand-in. This is a POST-
PROCESSING patch -- it does NOT touch the generator (flow.api.clm.R); it overwrites only
the fill positions of PRECTmms (and their PRECTmms_fqc flag) in the six 2017 files, in
place, values-only.

Conventions (verified against the output files + raw tables):
  - Output PRECTmms is a rate in mm/s; tipping precipBulk is mm accumulated per 30-min
    interval, so PRECTmms = precipBulk / 1800.  (No /2 split -- that is only for the
    60-min weighing gauge.)
  - Both grids are UTC, 30-min, timestamp at the BEGINNING of the period; we match on the
    absolute UTC timestamp, not on array index.
  - Missing values are physically stored as -9999.0 (not literal NaN); we fill exactly
    those positions.
  - Spliced timesteps are flagged PRECTmms_fqc = 5, and the variable's method_gap-fill
    code-map attribute is extended with '5=secondary_gauge_substitution'.

Safety: pristine copies of the six files are saved under atm/pre_splice_backup/ before any
write (never overwritten on re-run). Idempotent: once spliced the -9999 positions are gone,
so a re-run is a no-op.

Env: ctsm (python 3.12 / numpy / netCDF4).
Run:  python splice_2017_precip.py            # DRY RUN: report what would change
      python splice_2017_precip.py --apply     # write the six files in place
Exits 0 on success, 1 if any fill position could not be filled.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import netCDF4
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SWENSON = SCRIPT_DIR.parent.parent
ATM_DIR = SWENSON / "data/datm/neon_OSBS/custom/OSBS/atm"
RAW_TIP = Path("/blue/gerber/earth_models/neon/raw/OSBS/filesToStack00045")

GAP_MONTHS = ["2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12"]
FILL = -9999.0
FQC_CODE = 5
FQC_MAP = (
    "0=no gap-filling, 1=regression, 2=ReddyProc_methA, 3=ReddyProc_methB, "
    "4=ReddyProc_methC, 5=secondary_gauge_substitution"
)
SEC_PER_STEP = 1800.0  # 30 min


def read_tipping(ym):
    """Return {utc_timestamp_str: rate_mm_s} for a gap month from the DP1.00045 TIPPRE
    30-min table. Only finalQF==0, numeric precipBulk are included."""
    zs = sorted(glob.glob(str(RAW_TIP / f"*DP1.00045*{ym}*.zip")))
    zs = [z for z in zs if "RELEASE" in z] or zs
    if not zs:
        sys.exit(f"ERROR: no tipping-bucket zip for {ym} in {RAW_TIP}")
    z = zipfile.ZipFile(zs[0])
    names = [n for n in z.namelist() if "TIPPRE_30min" in n and n.endswith(".csv")]
    if not names:
        sys.exit(f"ERROR: no TIPPRE_30min table in {os.path.basename(zs[0])}")
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(names[0]))))
    out, total_mm, nbad = {}, 0.0, 0
    for r in rows:
        p, qf = r["precipBulk"], r["finalQF"]
        key = r["startDateTime"].rstrip("Z")  # 'YYYY-MM-DDTHH:MM:SS'
        if p in ("", "NA") or qf not in ("0", "0.0"):
            nbad += 1
            continue
        mm = float(p)
        total_mm += mm
        out[key] = mm / SEC_PER_STEP
    return out, total_mm, len(rows), nbad


def splice_file(ym, apply, backup_dir):
    path = ATM_DIR / f"OSBS_atm_{ym}.nc"
    tip, tip_total_mm, nrows, nbad = read_tipping(ym)

    nc = netCDF4.Dataset(path, "r" if not apply else "r+")
    pv = nc.variables["PRECTmms"]
    pv.set_auto_mask(False)
    prect = np.asarray(pv[:], dtype="float64")  # (time,1,1)
    fqc = np.asarray(nc.variables["PRECTmms_fqc"][:], dtype="int32")
    tvar = nc.variables["time"]
    times = netCDF4.num2date(
        tvar[:], tvar.units, getattr(tvar, "calendar", "gregorian")
    )

    series = prect[:, 0, 0]
    fill_idx = np.where((series == FILL) | np.isnan(series))[0]
    filled, unfilled, spliced_mm = 0, [], 0.0
    for i in fill_idx:
        key = times[i].strftime("%Y-%m-%dT%H:%M:%S")
        rate = tip.get(key)
        if rate is None:
            unfilled.append(key)
            continue
        prect[i, 0, 0] = rate
        fqc[i, 0, 0] = FQC_CODE
        spliced_mm += rate * SEC_PER_STEP
        filled += 1

    if apply and filled and not unfilled:
        nc.close()
        nc = netCDF4.Dataset(path, "r+")
        nc.variables["PRECTmms"][:] = prect
        nc.variables["PRECTmms_fqc"][:] = fqc
        nc.variables["PRECTmms_fqc"].setncattr("method_gap-fill", FQC_MAP)
        nc.close()
    else:
        nc.close()

    return {
        "ym": ym,
        "fill_positions": len(fill_idx),
        "filled": filled,
        "unfilled": unfilled,
        "spliced_mm": spliced_mm,
        "tip_total_mm": tip_total_mm,
        "tip_rows": nrows,
        "tip_bad": nbad,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Splice 2017 tipping-bucket precip into the gap months."
    )
    ap.add_argument(
        "--apply", action="store_true", help="write the files (default: dry run)"
    )
    a = ap.parse_args()

    backup_dir = ATM_DIR / "pre_splice_backup"
    if a.apply:
        backup_dir.mkdir(exist_ok=True)
        for ym in GAP_MONTHS:
            src = ATM_DIR / f"OSBS_atm_{ym}.nc"
            dst = backup_dir / src.name
            if not dst.exists():  # never overwrite the pristine original
                shutil.copy2(src, dst)
        print(f"[backup] pristine originals in {backup_dir}\n")

    mode = "APPLY (writing files)" if a.apply else "DRY RUN (no writes)"
    print(f"=== splice_2017_precip.py -- {mode} ===")
    print(
        f"{'month':9s}{'fill pos':>9s}{'filled':>8s}{'unfilled':>9s}"
        f"{'spliced mm':>12s}{'tip total mm':>13s}"
    )
    tot_fill = tot_unfilled = 0
    results = []
    for ym in GAP_MONTHS:
        r = splice_file(ym, a.apply, backup_dir)
        results.append(r)
        tot_fill += r["filled"]
        tot_unfilled += len(r["unfilled"])
        print(
            f"{r['ym']:9s}{r['fill_positions']:>9d}{r['filled']:>8d}"
            f"{len(r['unfilled']):>9d}{r['spliced_mm']:>12.1f}{r['tip_total_mm']:>13.1f}"
        )
    print(f"\ntotal filled: {tot_fill}   unfilled: {tot_unfilled}")
    if tot_unfilled:
        print(
            "WARNING: some fill positions had no matching QC-good tipping value "
            "(left as -9999). See per-month 'unfilled'."
        )
    if not a.apply:
        print("\nDRY RUN complete -- re-run with --apply to write the six files.")
    else:
        print("\nAPPLIED. Verify with: python scripts/neon_forcing/neon_forcing_qc.py")
    sys.exit(1 if tot_unfilled else 0)


if __name__ == "__main__":
    main()
