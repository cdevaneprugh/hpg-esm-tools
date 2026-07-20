# NEON pre-built DATM forcing — OSBS

CTSM-ready, gap-filled NEON tower atmospheric forcing for OSBS, produced by the
external **NCAR-NEON** pipeline (`flow.api.clm.R`, David Durden; gap-fill via
ReddyProc) and downloaded from NEON's object store. Phase I task **I2**.

## Contents

```
neon_OSBS/
├── README.md          # this file (tracked in git)
└── v4/OSBS/           # OSBS_atm_YYYY-MM.nc — *.nc gitignored (not committed)
```

| Field | Value |
|---|---|
| Version | **v4** (newest published; server has v1–v4, no v5) |
| Coverage | **2018-01 → 2024-12**, 84 monthly files |
| Total size | 12.08 MB (~147 KB/file) |
| Generated (upstream) | 2025-11-09 (single reprocessing batch) |
| Data basis | RELEASE-2026 NEON data — all within the released window (ends 2025-06), so **released, not provisional** |
| Fetched | 2026-07-15 |
| Source | `https://storage.neonscience.org/neon-ncar/NEON/atm/cdeps/v4/OSBS/` |

## File format (per month)

Single-point `(time, lat=1, lon=1)`, **gregorian** half-hourly (1488 steps/month),
`_FillValue=-9999`. Eight physical variables (`double`): `FSDS` (W/m²), `FLDS`
(W/m²), `PRECTmms` (mm/s), `TBOT` (K), `RH` (%), `WIND` (m/s), `PSRF` (Pa),
`ZBOT` (m). Plus informational `<VAR>_fqc` gap-fill QC flags. CTSM/CDEPS reads
only the 8 physical vars (RH → specific humidity converted internally). Files
carry `created_with = "flow.api.clm.R"`.

## Provenance notes

- **v4 is a full reprocessing, not v3 + 6 months.** All 78 months that overlap the
  older on-disk v3 set differ (0/78 byte-identical) — every file was regenerated
  2025-11-09. The 6 genuinely new months are 2024-07 → 2024-12.
- **The differences from v3 are reprocessing-scale.** Verified 2026-07-15: RMS Δ
  over the 78 overlap months is negligible — TBOT 0.17 K, PSRF 9 Pa, FSDS
  0.07 W/m², FLDS 0.09 W/m², precip 3×10⁻⁴ mm/s, RH 0.21 %, WIND 0.025 m/s, ZBOT
  0. Larger max-Δ points (TBOT 14.6 K, RH 29 %) are isolated gap-filled intervals
  refilled differently — the bulk of the record is essentially unchanged. Integrity
  84/84 valid; new months physically plausible with no fill in physical vars.
- **2018 TBOT is fine** (NCAR-NEON Issue #34 — a v1-era C→K unit artifact — was
  fixed by reprocessing years ago; verified sane, 269–307 K).

## Role in Phase I

This is the **temporary starting forcing** and the **validation reference** for
the custom pipeline: task I4 reproduces v4 with our own `flow.api.clm.R` run and
diffs against these files before we produce the fuller 2016–2026 record (I5).
See `phases/I-neon-forcing.md`.

## Re-fetch

The `.nc` files are gitignored (12 MB, trivially re-downloadable). To restore:

```bash
TARGET=$SWENSON/data/datm/neon_OSBS/v4/OSBS
mkdir -p "$TARGET"
BASE=https://storage.neonscience.org/neon-ncar/NEON/atm/cdeps/v4/OSBS
for y in 2018 2019 2020 2021 2022 2023 2024; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  curl -fsSL --retry 3 "$BASE/OSBS_atm_${y}-${m}.nc" -o "$TARGET/OSBS_atm_${y}-${m}.nc"
done; done
```
