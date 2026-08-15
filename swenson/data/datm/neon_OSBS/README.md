# NEON DATM forcing — OSBS

CTSM-ready, gap-filled NEON tower atmospheric forcing for OSBS. Two sets live here:
**`v4/`** — the pre-built NCAR-NEON product (Phase I **I2**), the temporary starting
forcing and validation reference; **`custom/`** — **our own full-record dataset**
(Phase I **I5**, the deliverable), produced by running the same NCAR-NEON generator
(`flow.api.clm.R`, David Durden; gap-fill via ReddyProc) over the full raw archive on
HiPerGator. The sections below document the v4 reference; the **Custom dataset** section
(end) documents the deliverable.

## Contents

```
neon_OSBS/
├── README.md          # this file (tracked in git)
├── v4/OSBS/           # pre-built v4 reference — OSBS_atm_YYYY-MM.nc (*.nc gitignored)
└── custom/OSBS/       # OUR full dataset (Phase I I5) — atm/ (deliverable, 101 files) + eval/ (fluxes); *.nc gitignored
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

## Custom dataset (Phase I I5) — the deliverable

`custom/OSBS/atm/` — **our full-record forcing**, produced by the forked NCAR-NEON
generator run offline on HiPerGator against the shared raw archive
`/blue/gerber/earth_models/neon/raw/OSBS`. Same file format as v4 (above).

| Field | Value |
|---|---|
| Coverage | **2017-02 → 2025-06**, 101 monthly files (~15 MB) |
| Relation to v4 | **strict superset** — v4's 2018–2024 span + 11 mo earlier (2017) + 6 later (2025 H1) |
| Start bound | 2017-02, set by EC-flux availability (the generator anchors its output time grid to EC); reaching 2016-08 would need a declined source edit |
| End bound | 2025-06 — RELEASE-2026 released cut (no provisional, per PI) |
| Data basis | RELEASE-2026 throughout |
| Fidelity | reproduces v4 to machine precision on measured timesteps (I4 PASS) |
| QC | passes `scripts/neon_forcing/neon_forcing_qc.py` — 0 NaN, physical ranges OK |
| `eval/` sibling | flux NetCDFs (`OSBS_eval_*.nc`) — not forcing; CO₂ carries NaN at OSBS (no usable EC CO₂) |

### 2017 precipitation provenance (the one non-standard field)

`PRECTmms` for **2017-07 → 2017-12** comes from the **secondary tipping-bucket gauge
(DP1.00045)**, not the primary weighing gauge (DP1.00044) used everywhere else: the
primary was physically down that stretch (raw all-NA, NEON `finalQF=1`), so the generator
wrote `-9999`. The tipping bucket recorded the precipitation completely (`finalQF=0`,
incl. **Hurricane Irma, Sep 2017**). Spliced in post-hoc by
`scripts/neon_forcing/splice_2017_precip.py` (**no source edit**): `precipBulk`/1800 → mm/s,
only the fill positions overwritten, flagged **`PRECTmms_fqc = 5`**
(`5=secondary_gauge_substitution`, code-map attribute extended). Validated: the two gauges
agree **r=0.96 / ~2 % on totals** (35 mutually-complete months); spliced values bit-match
the raw at the same UTC timestamp; the heaviest Sep-2017 rain lands on 2017-09-10 (Irma's
landfall date). Pristine pre-splice originals: `custom/OSBS/atm/pre_splice_backup/`.

### Re-generate

Requires the raw archive (`scripts/neon_forcing/run_download.sh`) and the `neon-forcing`
env. Generate per year (**LOWMEM=FALSE, ~42 GB/yr on the burst QOS** — LOWMEM=TRUE is
broken):

```bash
for y in 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
  DATEBGN=${y}-01-01 DATEEND=${y}-12-31 LOWMEM=FALSE \
    sbatch --qos=gerber-b --mem=64gb scripts/neon_forcing/run_forcing.sh
done   # 2017 auto-starts 2017-02 (EC), 2025 auto-stops 2025-06 (archive)
python scripts/neon_forcing/splice_2017_precip.py --apply   # re-apply the 2017 precip splice
python scripts/neon_forcing/neon_forcing_qc.py              # verify: 0 NaN, PASS
```
