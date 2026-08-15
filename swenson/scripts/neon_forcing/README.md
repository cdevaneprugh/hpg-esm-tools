# NEON → DATM forcing pipeline (Phase I)

Builds site-specific NEON atmospheric forcing for OSBS by running NEON's
NCAR-NEON generator (`flow.api.clm.R`) on HiPerGator, to replace the coarse
CRUNCEPv7 reanalysis in the operative CTSM case. The raw NEON download **and** all
processing run **on** HiPerGator — a NEON API token lifts the `/data/` block from the
cluster (see `../../phases/I-neon-forcing.md` §12).

See `../../docs/neon-forcing-pipeline-hipergator.md` for the full plan and
`../../phases/I-neon-forcing.md` for phase tracking.

## Files

**`setup/`** — one-time conda-env build:

| File | Purpose |
|------|---------|
| `setup/environment.yml` | conda spec for the `neon-forcing` env (r-base **4.2** + conda-satisfiable packages + toolchain; source-only packages built by `install_source_pkgs.R`) |
| `setup/build_env.sh` | Build the env — conda solve → source-install → smoke test (audits compiler ABI) |
| `setup/install_source_pkgs.R` | Installs the R packages on no conda channel (`neonUtilities`, `REddyProc`, `eddy4R.base`/`qaqc` @898a72d, `NEON.gf`) from the v4-era CRAN snapshots |
| `setup/Makevars.conda` | Lenient C flags (`-std=gnu17` …) so old snapshot packages compile with conda's modern gcc; wired in via `R_MAKEVARS_USER` |

**Pipeline** — run repeatedly (in the `neon-forcing` env):

| File | Purpose |
|------|---------|
| `download_raw.R` + `run_download.sh` | **Step 1**: authenticated raw pull → shared archive `/blue/gerber/earth_models/neon/raw/OSBS` (idempotent; scope via `NEON_START`/`NEON_END`) |
| `run_forcing.sh` | **Step 2**: offline `flow.api.clm.R` (the fork) → DATM forcing NetCDFs under `data/datm/neon_OSBS/custom/` |
| `size_manifest.R` | full-pull size probe (metadata-only) |
| `neon_v4_regression.py` | validation vs pre-built v4 (fqc-partitioned; run in the `ctsm` env) |
| `neon_forcing_qc.py` | **I5** whole-record QC — structural / gap-fill / physical-sanity / climatology → `results.json` + `summary.txt` + PASS/FAIL + plots (`ctsm` env) |
| `splice_2017_precip.py` | **I5** 2017 precip recovery — splices the secondary tipping-bucket gauge (DP1.00045) into the six 2017 gap months when the primary was down (post-processing, no source edit; flags `PRECTmms_fqc=5`) |
| `output/` | comparison results — `results.json`, `summary.txt`, scatter PNG (gitignored) |

**Environment note:** this is a **reconstruction of NEON's v4-era stack** (2021), not
a modern build. `r-base` is pinned to **4.2** because `ffbase` (a hard `eddy4R.base`
dependency) is archived and will not build on newer R; the source layer resolves
against the four dated CRAN snapshots the v4 `renv.lock` used, and `Makevars.conda`
lets that old C compile under conda's gcc 15. Chosen for fidelity to the pre-built v4
forcing (the I4 comparison target). Full rationale in the plan doc. **Built + smoke-tested
2026-07-29** (all 16 packages load).

## Build the environment

```bash
# from an interactive dev session with internet egress + CPU headroom, e.g.
#   srun --partition=hpg-dev --cpus-per-task=4 --mem=16gb --time=02:00:00 --pty bash
cd $SWENSON
bash scripts/neon_forcing/setup/build_env.sh
```

Prerequisite: the NCAR-NEON repo cloned at `/blue/gerber/cdevaneprugh/ncar-neon`
(holds `gapFilling/pack/NEON.gf` and `TowerTools_ForcingData/flow.api.clm.R`).

The build audits that R's compiler resolves to conda's
(`x86_64-conda-linux-gnu-cc`, not lmod gcc) and smoke-tests that all 16 required
packages load.

## Related

| Resource | Location |
|----------|----------|
| Full pipeline plan | `../../docs/neon-forcing-pipeline-hipergator.md` |
| Phase tracking | `../../phases/I-neon-forcing.md` |
| Pre-built v4 (comparison reference) | `../../data/datm/neon_OSBS/v4/OSBS/` |
| NCAR-NEON generator | `/blue/gerber/cdevaneprugh/ncar-neon/TowerTools_ForcingData/flow.api.clm.R` |
