# Docs audit — working to-do (2026-10-21)

Working directory for the second-pass docs review. Reports here:

- `hpg-esm-docs-audit-261020.md` — first pass (factual audit; MAJOR/MODERATE/MINOR fixes landed in hpg-esm-docs `67ec329..ef1fc44`)
- `hpg-esm-docs-prose-review-261020.md` — second pass (prose / story review)
- `lc-history-261021.md` — trace of the Lc determination and the plausibility-check history (item 2 / item 3 research)

**Rule:** no edit to `hpg-esm-docs` without explicit approval, one item at a time.

## Claude

- [x] Start this list; move the review docs from `~` into this directory (2026-10-21)
- [x] Confirm the pipeline computes Lc at run time rather than hard-coding it; report the current production value → `lc-history-261021.md` §1
- [x] Trace the plausibility script and its outputs back through commits and `audit/` → `lc-history-261021.md` §2
- [x] **Item 2.2 (Lc narrative)** — applied 2026-10-21 to `swenson/index.md:42` and `osbs-implementation.md` (Result subsection); not yet committed
- [x] **Item 2.2b (Lc "how")** — applied 2026-10-21 as an *Implementation* paragraph under *Solution: restricted-wavelength FFT*; not yet committed. Original spec: — add an *Implementation* paragraph under *Solution: restricted-wavelength FFT* on `osbs-implementation.md` that walks the chain in code order (`spatial_scale.py`): Laplacian of the mosaic → 2-D FFT → radial binning into 30 log-spaced wavelength bins → fit window [20 m, 2 × max hillslope length] (`_locate_peak` `lmin`/`lmax`; the cutoff is a fit-window bound, not a filter on the DEM or spectrum) → Gaussian and lognormal peak fits, selected by peak sharpness ≥ 1.5 then goodness-of-fit, with linear (t-score > 2) and flat fallbacks → Lc → A_thresh = int(0.5 Lc²). Note the 50 px edge blend/zero settings and state that 20 m is an empirical choice from the cutoff sweep. ~120 words; draft into `lc-history-261021.md` §3 for approval, then apply with 2.2
- [~] **Item 2.3 (plausibility PASS labels)** — PARKED 2026-10-21 by the user. User's recollection: the PI signed off in person (date unrecorded in the repos), and the checks were run on the full domain at some point but those results may no longer exist. Proposed wording in `lc-history-261021.md` §4 (variant (a), "on review with the PI") stands ready; §5 has the second-pass findings (paper's criterion is "typical ridge DTND ≈ Lc", not max(); the script's 0.90 figure is not in the paper; only surviving domain-wide DTND data point is Phase B at Lc = 300, P95/Lc ≈ 1.6)
- [~] Option: re-run both checks on the production domain at Lc = 355.98 m (~15 min job) — parked with 2.3
- [x] Item 2.4 (`src/` identical → Fortran identical) — applied 2026-10-21 to `lateral-flow-and-routing.md:16, :191`; not yet committed
- [~] Item 2.1 (origin story) — dropped per user: work predates these repos; user handles it
- [~] Items 2.5–2.8 (duplication, register, unit spacing, bold density) — deferred to the reorganization pass
- [x] Tools-side: Phase C summary annotated 2026-10-21 (dated annotation appended; original text left verbatim per the phase-file convention) — covers the 300→356 m outcome, the Check 1 FAIL→P95 history, and the wrong P95 entries (1.228 / 0.983) in the 02-17 table; not yet committed
- [x] Tools-side: STATUS locked-table Lc and A_thresh rows reframed as computed values dated 2026-02-26 (method locked 02-11); FFT-cutoff row unchanged; 2026-02-11 change-log line corrected — applied 2026-10-21, not yet committed

## For the user

- [ ] Lc plausibility checks (parked): PI signed off in person — pin down when, and whether any full-domain run results survive outside the repos (emails, old scratch dirs, SLURM accounting for jobs after 25164927)
- [ ] Origin story (pre-2026 work, condensed repos) — user handles separately
- [ ] After the to-do list is cleared: read the docs in full, assemble a rough methods draft, run it by the PI
- [ ] Independently verify the numbers
- [ ] Then: the compact rewrite of the pages (with Claude)
