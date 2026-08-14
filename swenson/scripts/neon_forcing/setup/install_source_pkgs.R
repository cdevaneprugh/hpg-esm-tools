#!/usr/bin/env Rscript
# Install the R packages required by flow.api.clm.R that are on no conda channel.
# Run INSIDE the activated `neon-forcing` env (build_env.sh does this). conda already
# provides every compiled/heavy dep (incl. the precompiled ff+ffbase r42 binaries),
# so remotes (upgrade="never") builds ONLY pure-R/light sources and never rebuilds a
# conda package; missing deps still pull.
#
# repos = the four dated Posit CRAN snapshots the v4 renv.lock resolves against
# (2022-02-28 .. 2023-10-22). Current CRAN has ARCHIVED several eddy4R deps
# (e.g. DataCombine); these snapshots predate the archiving AND pin the source layer
# to v4-era versions. install.packages searches all repos and takes the newest
# version present across them.
options(repos = c(
          CRANNew = "https://packagemanager.rstudio.com/cran/2023-10-22",
          GCPFIX  = "https://packagemanager.rstudio.com/cran/2023-04-17",
          FIX     = "https://packagemanager.rstudio.com/cran/2022-11-30",
          CRAN    = "https://packagemanager.rstudio.com/cran/2022-02-28"),
        Ncpus = max(1L, parallel::detectCores()), timeout = 900, warn = 1)
message("Install library: ", .libPaths()[1])
message("repos:\n  ", paste(getOption("repos"), collapse = "\n  "))
stopifnot(dir.exists(.libPaths()[1]), requireNamespace("remotes", quietly = TRUE))

have <- function(p) requireNamespace(p, quietly = TRUE)

EDDY4R_REPO <- "NEONScience/eddy4R"
EDDY4R_REF  <- "898a72d3e658a9dbdf855fc086b32349dd1f6afb"   # v4 renv.lock pin
gh <- function(subdir) remotes::install_github(EDDY4R_REPO, subdir = subdir,
        ref = EDDY4R_REF, dependencies = TRUE, upgrade = "never")

# 0) neonUtilities: no r42 conda build -> source-install from the snapshots (~2.4.x,
#    matching the getProductInfo/loadByProduct API flow.api.clm.R uses). It is also
#    an eddy4R.base dependency, so install it before eddy4R.base.
if (!have("neonUtilities")) { message("== neonUtilities (snapshot ~2.4.x) =="); install.packages("neonUtilities") }
stopifnot(have("neonUtilities"))

# 1) REddyProc: v4 pin 1.3.2 (in the snapshots), else newest snapshot version.
#    Auto-pulls solartime + bigleaf from the snapshots.
if (!have("REddyProc")) {
  message("== REddyProc (try 1.3.2, else snapshot newest) ==")
  ok <- tryCatch({ remotes::install_version("REddyProc", "1.3.2",
          dependencies = TRUE, upgrade = "never"); have("REddyProc") },
       error = function(e) { message("  1.3.2 failed: ", conditionMessage(e)); FALSE })
  if (!isTRUE(ok)) { message("  -> snapshot REddyProc"); install.packages("REddyProc") }
}
stopifnot(have("REddyProc"))

# 2) eddy4R.base (subdir) -- archived dep DataCombine + others resolve from the
#    snapshots; ff/ffbase and other conda-provided deps left untouched (upgrade="never").
if (!have("eddy4R.base")) { message("== eddy4R.base @ 898a72d =="); gh("pack/eddy4R.base") }
stopifnot(have("eddy4R.base"))

# 3) eddy4R.qaqc (subdir) -- deps all conda-provided; does not need eddy4R.base
if (!have("eddy4R.qaqc")) { message("== eddy4R.qaqc @ 898a72d =="); gh("pack/eddy4R.qaqc") }
stopifnot(have("eddy4R.qaqc"))

# 4) NEON.gf -- local package from the clone (hard dep robustbase is conda-provided)
if (!have("NEON.gf")) {
  neon_gf <- file.path(Sys.getenv("NCAR_NEON_DIR", "/blue/gerber/cdevaneprugh/ncar-neon"),
                       "gapFilling", "pack", "NEON.gf")
  stopifnot(dir.exists(neon_gf))
  message("== NEON.gf (install_local) ==")
  remotes::install_local(neon_gf, dependencies = TRUE, upgrade = "never")
}
stopifnot(have("NEON.gf"))

# 5) mlegp -- optional (in the script's packReq but not in renv.lock; not load-critical)
if (!have("mlegp")) { message("== mlegp (optional) =="); try(install.packages("mlegp"), silent = TRUE) }

message("\nAll required source packages installed.")
