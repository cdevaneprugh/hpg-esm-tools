#!/usr/bin/env Rscript
# =============================================================================
# download_raw.R -- off-HPG raw NEON tower download for the OSBS forcing pipeline
# (Phase I, Stage 1). Runs on a NON-HiPerGator machine (the user's laptop): NEON
# IP-blocks the /data/ API from HiPerGator, so the raw pull happens off-cluster
# and the resulting DirDnld/ is Globus-transferred to HPG for offline processing
# by flow.api.clm.R.
#
# License: MIT (hpg-esm-tools). Our own driver code -- it only *calls* NEON's
# neonUtilities package; it does not copy or modify NEON (AGPL) source.
#
# Requires: R + the neonUtilities package. NO conda, NO pip. See the runbook
#   swenson/docs/neon-raw-download-runbook.md for install steps + the no-root
#   human-action pause points.
#
# Usage (laptop, neonUtilities already installed):
#   Rscript download_raw.R
# Edit the CONFIG block first (at minimum DIRDNLD). Expect ~11 GB; ~20 GB free.
# =============================================================================

## ---- CONFIG (edit these) --------------------------------------------------
SITE     <- "OSBS"                                # NEON site (domain D03)
DIRDNLD  <- path.expand("~/neon_osbs_dirdnld")    # download root -> becomes DirDnld on HPG
START    <- "2016-08"                             # first month (precip-limited full record)
END      <- "2025-06"                             # last month (RELEASE-2026 released cut)
RELEASE  <- "RELEASE-2026"                        # released data only ...
PROVIS   <- FALSE                                 # ... exclude provisional (a moving target)
PACK     <- "basic"                               # matches flow.api.clm.R  Pack <- "basic"
TOKEN    <- Sys.getenv("NEON_TOKEN", "")          # optional free NEON API token -> higher
                                                  #   rate limits; export NEON_TOKEN to use it

# The products flow.api.clm.R consumes, CORRECTED for verified OSBS availability:
#   - DP1.00006.001 (the script's coded secondary precip) DROPPED: 0 months at OSBS.
#   - DP1.00045.001 (tipping bucket) is OSBS's actual secondary precip -> included.
#   - DP1.00023.001 listed ONCE (the stock script requests it twice: FLDS + Rg).
PRODUCTS <- c(
  "DP4.00200.001",  # eddy-covariance bundle: sonic WIND + T/PSRF/RH gap-fill donors + validation; 2017-02+
  "DP1.00003.001",  # triple aspirated air temp -> TBOT
  "DP1.00004.001",  # barometric pressure       -> PSRF
  "DP1.00001.001",  # 2D wind speed/direction   -> WIND (2D-sensor path)
  "DP1.00098.001",  # relative humidity         -> RH
  "DP1.00023.001",  # net radiometer SW + LW    -> FSDS + FLDS
  "DP1.00024.001",  # PAR                       -> gap-fill input
  "DP1.00014.001",  # direct/diffuse SW         -> gap-fill input
  "DP1.00044.001",  # precip, weighing gauge    -> PRECTmms (primary);   2016-09+
  "DP1.00045.001"   # precip, tipping bucket    -> PRECTmms (secondary); 2016-08+
)
## ---------------------------------------------------------------------------

if (!requireNamespace("neonUtilities", quietly = TRUE)) {
  stop("neonUtilities is not installed. See swenson/docs/neon-raw-download-runbook.md\n",
       "  Install into a USER library (no root needed via the Posit PPM binary repo).")
}

if (!dir.exists(DIRDNLD)) dir.create(DIRDNLD, recursive = TRUE)
message("neonUtilities ", as.character(utils::packageVersion("neonUtilities")))
message("DirDnld: ", DIRDNLD)
message("Site ", SITE, " | ", START, " .. ", END, " | ", RELEASE,
        " | provisional=", PROVIS, " | package=", PACK, "\n")

results <- data.frame(product = PRODUCTS, status = NA_character_,
                      zips = NA_integer_, stringsAsFactors = FALSE)

for (i in seq_along(PRODUCTS)) {
  dp  <- PRODUCTS[i]
  num <- strsplit(dp, "\\.")[[1]][2]              # "DP4.00200.001" -> "00200"
  message("== [", i, "/", length(PRODUCTS), "] ", dp, " ==")
  ok <- tryCatch({
    neonUtilities::zipsByProduct(
      dpID = dp, site = SITE, startdate = START, enddate = END,
      package = PACK, release = RELEASE, include.provisional = PROVIS,
      savepath = DIRDNLD, check.size = FALSE,
      token = if (nzchar(TOKEN)) TOKEN else NA_character_)
    TRUE
  }, error = function(e) { message("  SKIP ", dp, ": ", conditionMessage(e)); FALSE })

  stackdir <- file.path(DIRDNLD, paste0("filesToStack", num))
  nzip <- if (dir.exists(stackdir))
            length(list.files(stackdir, pattern = "\\.zip$", recursive = TRUE)) else 0L
  results$status[i] <- if (ok) "ok" else "skipped"
  results$zips[i]   <- nzip
  message("  -> ", results$status[i], "; ", nzip, " zip(s) in filesToStack", num, "\n")
}

message("==================== SUMMARY ====================")
print(results, row.names = FALSE)
sz <- tryCatch(system(paste("du -sh", shQuote(DIRDNLD), "| cut -f1"), intern = TRUE),
               error = function(e) "?")
message("\nTotal in ", DIRDNLD, ": ", sz, "  (expected ~11 GB)")
message("Note: 0 zips for a product means it is unavailable at ", SITE,
        " for this window/release -- expected only if a product was mis-listed.")
message("Next: Globus ", DIRDNLD, " to HiPerGator (see the runbook).")
