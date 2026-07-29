#!/bin/bash
# Build the `neon-forcing` conda env for the NCAR-NEON CLM forcing pipeline:
# solve the env, source-install the 10 R packages on no conda channel, smoke-test.
# Run on a node WITH internet egress + CPU/time headroom (login-node process limits
# can kill long compiles). Prefer an interactive dev session:
#   srun --partition=hpg-dev --cpus-per-task=4 --mem=16gb --time=02:00:00 --pty bash
# Usage:  bash build_env.sh
set -euo pipefail

ENV_NAME="neon-forcing"
PROJ_DIR="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/scripts/neon_forcing"
ENV_YML="${PROJ_DIR}/environment.yml"
INSTALL_R="${PROJ_DIR}/install_source_pkgs.R"
export NCAR_NEON_DIR="${NCAR_NEON_DIR:-/blue/gerber/cdevaneprugh/ncar-neon}"

# clean module state: no lmod compiler/lib may leak into the solve or the compiles
module purge
module load conda 2>/dev/null

[[ -f "${ENV_YML}"   ]] || { echo "ERROR: missing ${ENV_YML}";   exit 1; }
[[ -f "${INSTALL_R}" ]] || { echo "ERROR: missing ${INSTALL_R}"; exit 1; }
[[ -d "${NCAR_NEON_DIR}/gapFilling/pack/NEON.gf" ]] || {
  echo "ERROR: NEON.gf not found under ${NCAR_NEON_DIR}; clone NCAR-NEON there first."; exit 1; }

echo "=== [1/3] Create/update env '${ENV_NAME}' (strict channel priority) ==="
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  CONDA_CHANNEL_PRIORITY=strict conda env update -n "${ENV_NAME}" -f "${ENV_YML}" --prune
else
  CONDA_CHANNEL_PRIORITY=strict conda env create -f "${ENV_YML}"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# supply our own Makevars with lenient C flags (-std=gnu17 etc.) so old v4-era
# snapshot packages (e.g. locfit) compile with conda's modern gcc; it does NOT set
# CC (conda's activated toolchain stays the compiler). Stray ~/.R config stays out.
export R_MAKEVARS_USER="${PROJ_DIR}/Makevars.conda"
export R_ENVIRON_USER=/dev/null
unset R_LIBS_USER R_LIBS 2>/dev/null || true

# audit -- MUST print x86_64-conda-linux-gnu-*, not /apps/... or /usr/bin/...
echo "R library : $(Rscript -e 'cat(.libPaths()[1])')"
echo "CC  : $(R CMD config CC)"; echo "CXX : $(R CMD config CXX)"; echo "FC  : $(R CMD config FC)"

echo "=== [2/3] Install source-only R packages ==="
Rscript "${INSTALL_R}"

echo "=== [3/3] Smoke test ==="
Rscript -e '
pkgs <- c("rhdf5","ncdf4","reshape2","ggplot2","gridExtra","knitr","naniar","Rfast",
          "neonUtilities","googleCloudStorageR","dplyr","tidyr","REddyProc",
          "eddy4R.base","eddy4R.qaqc","NEON.gf")
ok <- vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)
print(data.frame(package = pkgs, available = ok), row.names = FALSE)
if (any(!ok)) { cat("MISSING:", paste(pkgs[!ok], collapse=", "), "\n"); quit(status = 1L) }
cat("\nAll required packages load.\n")'
echo "=== DONE: env '${ENV_NAME}' ready ==="
