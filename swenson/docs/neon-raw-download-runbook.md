# Runbook: download raw NEON OSBS tower data (off-HiPerGator)

**Audience:** a Claude Code instance running on the user's **Linux laptop**.
**Goal:** download the raw NEON tower data the OSBS forcing pipeline needs, then
transfer it to HiPerGator via Globus. This is **Stage 1** of the NEON→DATM
pipeline (Phase I); everything after this runs on HiPerGator.

**Why this runs on the laptop, not HPG:** NEON IP-blocks its `/data/` download API
from HiPerGator (every `/data/` call returns HTTP 403). A residential/campus IP is
not blocked. So the raw pull happens here; the result is shipped to HPG.

---

## Ground rules for the Claude Code instance running this

1. **You do NOT have root.** Anything needing `sudo` / system package installs is a
   **human action** — stop and ask. These are marked below as:

   > **>>> PAUSE — HUMAN ACTION REQUIRED <<<**

2. **No conda, no pip.** Use the system R + an R **user library** (`~/R/library`),
   which needs no root.
3. Work non-destructively; print what you're doing. The download is ~**11 GB**
   (dominated by the eddy-covariance bundle, ~7 GB). Ensure **~20 GB free** first
   (`df -h ~`).

---

## Step 0 — Sanity

```bash
df -h "$HOME" | tail -1          # need ~20 GB free
. /etc/os-release; echo "$ID $VERSION_ID ${VERSION_CODENAME:-?}"   # note the distro codename
```

Record the distro **codename** (e.g. `jammy`, `noble`, `bookworm`) — Step 2 uses it.

---

## Step 1 — Is R installed?

```bash
command -v R && R --version | head -1
```

- **If R is present:** continue to Step 2.
- **If R is absent:**

  > **>>> PAUSE — HUMAN ACTION REQUIRED (root) <<<**
  > R is not installed and installing it needs admin rights. Ask the human to run
  > the appropriate command for their distro, then continue:
  > - Debian/Ubuntu: `sudo apt update && sudo apt install -y r-base`
  > - Fedora/RHEL:   `sudo dnf install -y R`
  > - Arch:          `sudo pacman -S r`
  > Do **not** attempt this yourself.

---

## Step 2 — Install `neonUtilities` into a user library (no root)

On Linux, plain CRAN serves *source* packages, which can trigger a compiler +
system-dev-library step (needs root). Avoid that by pointing R at **Posit Public
Package Manager (PPM) Linux binaries** — prebuilt, no compile, no root.

```bash
export R_LIBS_USER="$HOME/R/library"
mkdir -p "$R_LIBS_USER"

CODENAME="$(. /etc/os-release; echo "${VERSION_CODENAME:-}")"   # from Step 0
Rscript -e "
  .libPaths(Sys.getenv('R_LIBS_USER'))
  cn <- '${CODENAME}'
  repo <- if (nzchar(cn)) sprintf('https://packagemanager.posit.co/cran/__linux__/%s/latest', cn)
          else 'https://cloud.r-project.org'
  options(repos = c(CRAN = repo))
  message('repo: ', getOption('repos')[['CRAN']])
  install.packages('neonUtilities')
  cat('neonUtilities', as.character(packageVersion('neonUtilities')), 'OK\n')
"
```

- **If it prints `neonUtilities <version> OK`:** continue to Step 3.
- **If it fails compiling** (distro codename not on PPM, so R fell back to source
  and a system dev-library is missing, e.g. `libcurl`, `openssl`, `libxml-2.0`):

  > **>>> PAUSE — HUMAN ACTION REQUIRED (root) <<<**
  > A source build needs system dev libraries. Ask the human to install them, then
  > re-run the block above:
  > - Debian/Ubuntu: `sudo apt install -y libcurl4-openssl-dev libssl-dev libxml2-dev`
  > - Fedora/RHEL:   `sudo dnf install -y libcurl-devel openssl-devel libxml2-devel`

*(Optional, recommended for a download this size — a free NEON API token raises the
rate limit and avoids throttling. If the human has one, `export NEON_TOKEN=...`
before Step 3. Without it the download still works, just slower.)*

---

## Step 3 — Run the download

The download script is `download_raw.R` (committed at
`swenson/scripts/neon_forcing/download_raw.R`; the full text is reproduced at the
bottom of this runbook so this file is self-contained). It pulls **10 products**
for **OSBS, 2016-08 → 2025-06, RELEASE-2026, released-only, basic package** into
`~/neon_osbs_dirdnld/filesToStack<dpID>/`.

1. Put `download_raw.R` on the laptop (copy from the repo, or write the copy at the
   bottom of this file).
2. Edit its `DIRDNLD` if you want a different location (default `~/neon_osbs_dirdnld`).
3. Run it **with the same `R_LIBS_USER`** exported in Step 2:

```bash
export R_LIBS_USER="$HOME/R/library"
cd <dir containing download_raw.R>
Rscript download_raw.R
```

This prints per-product progress and a final summary table. It runs unattended
(no size prompt — `check.size=FALSE`). Expect it to take a while (the EC bundle is
~7 GB); NEON may throttle, and `neonUtilities` waits out rate limits automatically.

---

## Step 4 — Verify before transferring

```bash
DIRDNLD="$HOME/neon_osbs_dirdnld"
du -sh "$DIRDNLD"                                   # expect ~11 GB
ls -d "$DIRDNLD"/filesToStack*                      # one dir per downloaded product
for d in "$DIRDNLD"/filesToStack*; do
  printf "%-28s %s zips\n" "$(basename "$d")" "$(find "$d" -name '*.zip' | wc -l)"
done
```

Expected: **10 `filesToStack*` directories** (one per product), zip counts roughly
matching the month coverage:

| Product dir | ~zips | Note |
|---|---|---|
| filesToStack00200 | ~101 | EC bundle, 2017-02+ (`.h5` zips) |
| filesToStack00044 | ~106 | weighing precip, 2016-09+ |
| filesToStack00045 | ~107 | tipping precip, 2016-08+ |
| filesToStack00003/00004/00001/00098/00023/00024/00014 | ~107 each | within-window months |

**All ten products should download** (the corrected list already removed the one
that is absent at OSBS). If any product reports `skipped` / 0 zips in the script
summary, note it and report to the user — do not silently proceed.

---

## Step 5 — Transfer to HiPerGator via Globus

```bash
command -v globus && globus whoami          # is Globus CLI / Connect Personal set up?
```

- **If Globus Connect Personal is configured** (an endpoint exists and is running):
  transfer the whole `~/neon_osbs_dirdnld/` to the HiPerGator path
  **`/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson/data/neon/met/DirDnld/`**
  (create it on HPG if needed; it is gitignored). Preserve the `filesToStack*`
  directory structure exactly.
- **If Globus is not set up:**

  > **>>> PAUSE — HUMAN ACTION REQUIRED <<<**
  > Installing Globus Connect Personal and authenticating an endpoint is the human's
  > action (user-space install, but needs their Globus login). Ask them to set it up
  > (`docs.rc.ufl.edu/data_transfer/globus_transfer`), then do the transfer above.

HiPerGator's Globus endpoint is **`UFRC`**; the destination collection is the user's
`/blue/gerber/cdevaneprugh` space.

---

## Hand-back

When done, tell the user:
- total size transferred and the per-product zip counts (from Step 4);
- the HPG destination path;
- any product that came back empty/skipped.

The HiPerGator side (offline `flow.api.clm.R` run + reproduce-v4 validation) picks up
from there.

---

## Reference — what and why

- **Released-only** (`release="RELEASE-2026"`, `include.provisional=FALSE`): the PI
  wants frozen, citable data — not the provisional 2025-07→2026-06 tail. The stock
  script defaults to `include.provisional=TRUE` and no release; the download script
  overrides both.
- **Date range 2016-08 → 2025-06:** the maximum released record. Precipitation is the
  binding constraint (tipping bucket 2016-08, weighing gauge 2016-09); the other
  variables reach back to 2014–2015. The EC bundle only starts 2017-02 but is a
  supplementary source, so it does not cap the record.
- **Product-list corrections (verified against the NEON API):** the stock script's
  secondary-precip product **DP1.00006.001 does not exist at OSBS** (0 months, any
  release) — it is replaced here by the tipping bucket **DP1.00045.001**. DP1.00023.001
  (radiation) is pulled once though the script requests it twice.
- **`basic` package** matches the pipeline (`flow.api.clm.R Pack <- "basic"`).
  `expanded` would be far larger and is not used.

---

## Appendix — `download_raw.R` (self-contained copy)

The same `download_raw.R` as `swenson/scripts/neon_forcing/download_raw.R`
(condensed comments; functionally identical). Write it to a file and run it in
Step 3 if you don't have the repo on the laptop.

```r
#!/usr/bin/env Rscript
# download_raw.R -- off-HPG raw NEON tower download for the OSBS forcing pipeline.
# License: MIT (hpg-esm-tools). Calls neonUtilities; does not modify NEON source.
# Requires R + neonUtilities (no conda, no pip). Expect ~11 GB; ~20 GB free.

## ---- CONFIG (edit these) ----
SITE     <- "OSBS"
DIRDNLD  <- path.expand("~/neon_osbs_dirdnld")
START    <- "2016-08"
END      <- "2025-06"
RELEASE  <- "RELEASE-2026"
PROVIS   <- FALSE
PACK     <- "basic"
TOKEN    <- Sys.getenv("NEON_TOKEN", "")          # optional; export NEON_TOKEN to use

PRODUCTS <- c(
  "DP4.00200.001",  # EC bundle: sonic WIND + T/PSRF/RH donors + validation; 2017-02+
  "DP1.00003.001",  # triple aspirated air temp -> TBOT
  "DP1.00004.001",  # barometric pressure       -> PSRF
  "DP1.00001.001",  # 2D wind                   -> WIND
  "DP1.00098.001",  # relative humidity         -> RH
  "DP1.00023.001",  # net radiometer SW + LW    -> FSDS + FLDS
  "DP1.00024.001",  # PAR                       -> gap-fill input
  "DP1.00014.001",  # direct/diffuse SW         -> gap-fill input
  "DP1.00044.001",  # precip, weighing gauge    -> PRECTmms (primary);   2016-09+
  "DP1.00045.001"   # precip, tipping bucket    -> PRECTmms (secondary); 2016-08+
)
## ----------------------------

if (!requireNamespace("neonUtilities", quietly = TRUE))
  stop("neonUtilities not installed -- see the runbook (user-library install, no root).")

if (!dir.exists(DIRDNLD)) dir.create(DIRDNLD, recursive = TRUE)
message("neonUtilities ", as.character(utils::packageVersion("neonUtilities")))
message("DirDnld: ", DIRDNLD)
message("Site ", SITE, " | ", START, " .. ", END, " | ", RELEASE,
        " | provisional=", PROVIS, " | package=", PACK, "\n")

results <- data.frame(product = PRODUCTS, status = NA_character_,
                      zips = NA_integer_, stringsAsFactors = FALSE)
for (i in seq_along(PRODUCTS)) {
  dp  <- PRODUCTS[i]; num <- strsplit(dp, "\\.")[[1]][2]
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
  results$status[i] <- if (ok) "ok" else "skipped"; results$zips[i] <- nzip
  message("  -> ", results$status[i], "; ", nzip, " zip(s) in filesToStack", num, "\n")
}
message("==================== SUMMARY ====================")
print(results, row.names = FALSE)
sz <- tryCatch(system(paste("du -sh", shQuote(DIRDNLD), "| cut -f1"), intern = TRUE),
               error = function(e) "?")
message("\nTotal in ", DIRDNLD, ": ", sz, "  (expected ~11 GB)")
```
