#!/usr/bin/env Rscript
# =============================================================================
# size_manifest.R -- exact full-download-size manifest for the OSBS NEON products
# (Phase I, phases/I-neon-forcing.md sec 12.4 rung 2). METADATA ONLY -- queries the
# NEON /data/ API per product per in-window month and sums data.files[].size. No
# file bytes are downloaded, so it is safe to run before committing to the full pull.
#
# Mirrors download_raw.R (same products / window / release / package). NOTE: this sums
# UNCOMPRESSED individual /data/ files, which is ~2x the actual zipped download (measured
# 2026-08-14: 22.6 GB uncompressed here -> 11 GB of zips for the full 2016-08..2025-06 pull).
#
# Requires: the `neon-forcing` conda env (httr + jsonlite, pulled in by neonUtilities)
#   and a NEON API token in $NEON_TOKEN (lifts the /data/ 403 from HPG).
# =============================================================================

suppressMessages({ library(httr); library(jsonlite) })

SITE    <- "OSBS"
START   <- Sys.getenv("NEON_START", "2016-08")
END     <- Sys.getenv("NEON_END",   "2025-06")
RELEASE <- "RELEASE-2026"
PACK    <- "basic"
TOKEN   <- Sys.getenv("NEON_TOKEN", "")

PRODUCTS <- c(
  "DP4.00200.001",  # eddy-covariance bundle (HDF5; the bulk)
  "DP1.00003.001",  # triple aspirated air temp -> TBOT
  "DP1.00004.001",  # barometric pressure       -> PSRF
  "DP1.00001.001",  # 2D wind                   -> WIND
  "DP1.00098.001",  # relative humidity         -> RH
  "DP1.00023.001",  # net radiometer            -> FSDS + FLDS
  "DP1.00024.001",  # PAR
  "DP1.00014.001",  # direct/diffuse SW
  "DP1.00044.001",  # precip, weighing gauge    (primary)
  "DP1.00045.001"   # precip, tipping bucket    (secondary)
)

if (!nzchar(TOKEN)) stop("NEON_TOKEN is empty -- the /data/ query will 403 from HPG.")

months <- format(seq(as.Date(paste0(START, "-01")),
                     as.Date(paste0(END,   "-01")), by = "month"), "%Y-%m")
message("Manifest: ", length(PRODUCTS), " products x ", length(months),
        " months (", START, " .. ", END, ", ", RELEASE, "/", PACK, ")\n")

rows <- vector("list", length(PRODUCTS))
for (i in seq_along(PRODUCTS)) {
  dp <- PRODUCTS[i]
  tot <- 0; nfiles <- 0L; nmon <- 0L
  for (m in months) {
    url <- sprintf(
      "https://data.neonscience.org/api/v0/data/%s/%s/%s?package=%s&release=%s",
      dp, SITE, m, PACK, RELEASE)
    ok <- tryCatch({
      r <- GET(url, add_headers(`X-API-Token` = TOKEN), timeout(60))
      if (status_code(r) == 200) {
        j <- fromJSON(content(r, "text", encoding = "UTF-8"))
        f <- j$data$files
        if (!is.null(f) && length(f) && !is.null(f$size) && nrow(f) > 0) {
          tot <- tot + sum(as.numeric(f$size), na.rm = TRUE); nfiles <- nfiles + nrow(f); nmon <- nmon + 1L
        }
      }
      TRUE
    }, error = function(e) FALSE)
  }
  rows[[i]] <- data.frame(product = dp, months = nmon, files = nfiles,
                          GB = round(tot / 1e9, 3), stringsAsFactors = FALSE)
  message(sprintf("  [%2d/%2d] %-14s  %3d mo  %5d files  %7.3f GB",
                  i, length(PRODUCTS), dp, nmon, nfiles, tot / 1e9))
}

out <- do.call(rbind, rows)
cat("\n==================== SIZE MANIFEST ====================\n")
print(out, row.names = FALSE)
cat(sprintf("\nTOTAL: %.2f GB across %d files (RELEASE-2026 basic, %s .. %s)\n",
            sum(out$GB), sum(out$files), START, END))
cat("Note: this sums UNCOMPRESSED /data/ files; the actual zipped download is ~half",
    "(measured 22.6 GB here -> 11 GB of zips for the full 2016-08..2025-06 pull).\n")
