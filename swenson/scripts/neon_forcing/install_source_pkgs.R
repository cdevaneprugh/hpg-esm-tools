#!/usr/bin/env Rscript
# Install the R packages required by flow.api.clm.R that are on no conda channel.
# Run INSIDE the activated `neon-forcing` env (build_env.sh does this). conda already
# provides every compiled/heavy dep, so remotes (upgrade="never") builds ONLY
# pure-R/light sources and never rebuilds a conda package; missing CRAN deps still
# pull automatically.
options(repos = c(CRAN = "https://cloud.r-project.org"),   # conda R ships repos unset
        Ncpus = max(1L, parallel::detectCores()), timeout = 600, warn = 1)
message("Install library: ", .libPaths()[1])
stopifnot(dir.exists(.libPaths()[1]), requireNamespace("remotes", quietly = TRUE))

EDDY4R_REPO <- "NEONScience/eddy4R"
EDDY4R_REF  <- "898a72d3e658a9dbdf855fc086b32349dd1f6afb"   # v4 renv.lock pin
gh <- function(subdir) remotes::install_github(EDDY4R_REPO, subdir = subdir,
        ref = EDDY4R_REF, dependencies = TRUE, upgrade = "never")

# 1) REddyProc: try v4 pin 1.3.2, else current CRAN (auto-pulls solartime + bigleaf)
message("== REddyProc (try 1.3.2, else current) ==")
ok <- tryCatch({ remotes::install_version("REddyProc", "1.3.2",
        dependencies = TRUE, upgrade = "never"); requireNamespace("REddyProc", quietly = TRUE) },
     error = function(e) { message("  1.3.2 failed: ", conditionMessage(e)); FALSE })
if (!isTRUE(ok)) { message("  -> current CRAN REddyProc"); install.packages("REddyProc") }
stopifnot(requireNamespace("REddyProc", quietly = TRUE))

# 2) eddy4R.base (subdir) -- auto-pulls DataCombine/EMD/robfilter/ffbase from CRAN
message("== eddy4R.base @ 898a72d ==");  gh("pack/eddy4R.base")
stopifnot(requireNamespace("eddy4R.base", quietly = TRUE))

# 3) eddy4R.qaqc (subdir) -- deps all conda-provided; does not need eddy4R.base
message("== eddy4R.qaqc @ 898a72d ==");  gh("pack/eddy4R.qaqc")
stopifnot(requireNamespace("eddy4R.qaqc", quietly = TRUE))

# 4) standalone CRAN sources (deps raster/RCurl/geosphere/leaflet/matlab conda-provided)
message("== metScanR + prism ==");  install.packages(c("metScanR", "prism"))
stopifnot(requireNamespace("metScanR", quietly = TRUE), requireNamespace("prism", quietly = TRUE))

# 5) NEON.gf -- local package from the clone (hard dep robustbase is conda-provided)
neon_gf <- file.path(Sys.getenv("NCAR_NEON_DIR", "/blue/gerber/cdevaneprugh/ncar-neon"),
                     "gapFilling", "pack", "NEON.gf")
stopifnot(dir.exists(neon_gf))
message("== NEON.gf (install_local) ==")
remotes::install_local(neon_gf, dependencies = TRUE, upgrade = "never")
stopifnot(requireNamespace("NEON.gf", quietly = TRUE))

# 6) mlegp -- optional (in the script's loop but not in renv.lock / packReq)
message("== mlegp (optional) =="); try(install.packages("mlegp"), silent = TRUE)

message("\nAll required source packages installed.")
