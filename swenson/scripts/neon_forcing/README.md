# NEON → DATM forcing pipeline (Phase I)

Builds site-specific NEON atmospheric forcing for OSBS by running NEON's
NCAR-NEON generator (`flow.api.clm.R`) on HiPerGator, to replace the coarse
CRUNCEPv7 reanalysis in the operative CTSM case. The raw NEON download runs
**off**-HiPerGator (NEON IP-blocks the `/data/` API from the cluster); the conda
environment and all processing run **here**.

See `../../docs/neon-forcing-pipeline-hipergator.md` for the full plan and
`../../phases/I-neon-forcing.md` for phase tracking.

## Files

| File | Purpose |
|------|---------|
| `environment.yml` | conda spec for the `neon-forcing` env (conda-satisfiable packages + toolchain; the 10 source-only packages are built by `install_source_pkgs.R`) |
| `build_env.sh` | Build the env — conda solve → source-install → smoke test |
| `install_source_pkgs.R` | Installs the R packages on no conda channel (REddyProc, eddy4R.base/qaqc @898a72d, metScanR, prism, NEON.gf) |
| `output/` | Results from the later `neon_v4_regression` comparison (gitignored) |

Planned, not yet added: `download_raw.R` (laptop-side `zipsByProduct`),
`run_pipeline.sh` (SLURM offline run), `neon_v4_regression.py` (fqc-partitioned
v4 comparison).

## Build the environment

```bash
# from an interactive dev session with internet egress + CPU headroom, e.g.
#   srun --partition=hpg-dev --cpus-per-task=4 --mem=16gb --time=02:00:00 --pty bash
cd $SWENSON
bash scripts/neon_forcing/build_env.sh
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
