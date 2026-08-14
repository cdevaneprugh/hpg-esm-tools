#!/usr/bin/env Rscript
# =============================================================================
# download_raw.R -- raw NEON tower download for the OSBS forcing pipeline
# (Phase I, Stage 1). Runs ON HiPerGator: a NEON API token lifts the /data/ 403
# from HPG (phases/I-neon-forcing.md Research note 12), so the raw pull runs on a
# compute node -- no off-HPG laptop, no Globus. The resulting DirDnld/ is read in
# place by the offline flow.api.clm.R run.
#
# License: MIT (hpg-esm-tools). Our own driver code -- it only *calls* NEON's
# neonUtilities package; it does not copy or modify NEON (AGPL) source.
#
# Requires: the `neon-forcing` conda env (neonUtilities >= 2.4.0) + a NEON API
#   token in $NEON_TOKEN. Invoke from a SLURM wrapper that does
#   `module load conda && conda activate neon-forcing`.
#
# Before the full pull, run the phases/I-neon-forcing.md 12.4 test ladder
#   (connectivity gate -> real-tool smoke -> exact-size manifest -> EC probe ->
#   timing). The zipped download is ~11 GB (confirmed 2026-08-14; the 12.4 manifest's
#   22.6 GB is the UNCOMPRESSED /data/ sum, ~2x the zips). /blue has the space.
# =============================================================================

## ---- CONFIG (edit these) --------------------------------------------------
SITE     <- "OSBS"                                # NEON site (domain D03)
DIRDNLD  <- Sys.getenv("NEON_DIRDNLD",            # shared on-HPG archive (Option B); wrapper may override
  "/blue/gerber/earth_models/neon/raw/OSBS")      #   flow.api.clm.R reads filesToStack<num>/ in place
START    <- Sys.getenv("NEON_START", "2016-08")   # first month (precip-limited full record); wrapper may scope
END      <- Sys.getenv("NEON_END",   "2025-06")   # last month (RELEASE-2026 released cut); wrapper may scope
RELEASE  <- "RELEASE-2026"                        # released data only ...
PROVIS   <- FALSE                                 # ... exclude provisional (a moving target)
PACK     <- "basic"                               # matches flow.api.clm.R  Pack <- "basic"
TOKEN    <- Sys.getenv("NEON_TOKEN", "")          # NEON API token -- REQUIRED on HPG: lifts the
                                                  #   /data/ 403 (phases/I 12). export NEON_TOKEN.

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
  stop("neonUtilities not found. Activate the `neon-forcing` conda env first\n",
       "  (see scripts/neon_forcing/build_env.sh); neonUtilities 2.4.0 is provided there.")
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
message("Next: run flow.api.clm.R offline against ", DIRDNLD,
        " (Option B; see phases/I-neon-forcing.md §10).")
