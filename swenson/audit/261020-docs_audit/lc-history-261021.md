# Lc determination and the plausibility checks — history trace

**Date:** 2026-10-21. Read-only research for to-do items 2.2 and 2.3. Every statement below cites a commit, file, or output read this session.

---

## 1. Is Lc computed or hard-coded? (to-do item)

**Computed.** In the current pipeline the only fixed quantity is the FFT wavelength cutoff:

- `scripts/osbs/run_pipeline.py:85` — `MIN_WAVELENGTH = 20  # meters`
- `:925-941` — `fft_result = identify_spatial_scale_laplacian_dem(dem, …, min_wavelength=MIN_WAVELENGTH, blend_edges_n=50, zero_edges_n=50)`; then `lc_m = fft_result["spatialScale_m"]`, `lc_px = fft_result["spatialScale"]`, `accum_threshold = int(0.5 * lc_px**2)`
- `scripts/osbs/spatial_scale.py` — no floor, fallback, or override constant (grep for `MIN_LC|min_lc|floor|fallback` is empty)

There is no override path. Lc and A_thresh are whatever the lognormal/Gaussian peak fit returns on the input mosaic.

**The hard-coded era, precisely.** A forced minimum did exist:

| Commit | Date | What |
|---|---|---|
| `8963fc3` | 2026-01-26 | `MIN_LC_PIXELS = 100  # 100m minimum at 1m resolution` added; `if spatialScale < min_lc_pixels: … spatialScale = min_lc_pixels` |
| `eeff698` | 2026-02-10 | old script copy removed in the audit cleanup (the floor survived in `run_pipeline.py`) |
| `25e24f8` | 2026-03-16 | "Close Phase D" — `MIN_LC_PIXELS` and the floor logic deleted from `run_pipeline.py` |

The floor only fired when the FFT peak fell *below* 100 px, i.e. on the pre-cutoff 8 m artifact peak (Phase C's 2026-02-11 table lists "OSBS (full, 4x sub) | Forced minimum | 100m"). Once the 20 m cutoff was in place (2026-02-10) the peak was 356 px and the floor was inert; it was removed on 2026-03-16. **Every production run (2026-02-26 tier 3 onward) computed Lc with no floor active.**

**Current production value.** `output/osbs/2026-05-05_production/hillslope_params.json` → `metadata.spatial_scale_m = 355.983473511522`, `accum_threshold = 63362` (= `int(0.5 × 355.983²)`), `min_wavelength_m = 20.0`. `production_summary.txt` prints the rounded "356 m (356.0 px)". The NetCDF carries `accumulation_threshold = 63362` and `fft_min_wavelength_m = 20.0` as global attributes but not Lc itself. The same 355.983 m appears in the 2026-04-24 diagnostic run and the 2026-03-26 water-masking comparison, so the value is stable across pipeline revisions on the 90-tile mosaic.

**Lc depends on the domain.** Phase D tier runs on the rebuilt pipeline (`phases/D-rebuild-pipeline.md:80-84`, 2026-02-26): R6C10 single tile → 540.7 m; 5×5 block R6C7-R10C11 → 478.8 m; 90-tile production domain → 356.0 m. The plausibility checks (§2) were run on the 5×5 block — whose own FFT Lc is ~479 m — at the three candidate values 285/300/356 m taken from the 90-tile spectrum. This is not stated anywhere in the docs.

---

## 2. The plausibility checks: what happened, in order

| Date | Commit | Event |
|---|---|---|
| 2026-02-10 | `0721e51` | Restricted-wavelength FFT on the contiguous mosaic: 20 m cutoff → 356.0 m, 50 m → 356.0 m, 100 m → 285.4 m (Gaussian). Phase C log: "Lc candidate: ~300m (range 285-356m depending on cutoff and model)." |
| 2026-02-11 | `6ed92b1` | **PI accepts ~300 m as the working value** ("passes the sniff test"). Log explicitly reserves judgement: "If these checks fail, Lc will be revisited. The restricted-wavelength FFT result provides the starting point, not necessarily the final answer." Two checks named in advance: Lc vs max(DTND) ≈ 300 m; Lc² vs mean catchment area ≈ 90,000 m². |
| 2026-02-17 | `3af1c4d` | `scripts/smoke_tests/validate_lc_physical.py` written and run (job 25164927) on the 5×5 block at Lc = 285/300/356. Script criteria: `verdict(ratio)` on **max(DTND)/Lc** (PASS 0.5–2.0, MARGINAL 0.3–3.0, else FAIL) and mean(catch)/Lc². Output `validation_summary.txt`: **Result: FAIL**; Check 1 FAIL/FAIL/MARGINAL, Check 2 PASS×3. The script already computed P95 and P99 but used neither for a verdict. **The 02-17 Phase C entry as first committed** was headed "Physical validation — Check 2 passes, Check 1 fails on outlier max" and ended: "The Check 1 FAIL is driven by a single max(DTND) value and may warrant using P99 or another robust statistic instead. **This is a PI interpretation question.**" |
| 2026-02-19 | `cadff06` | Script moved to `phases/C-archive/` unchanged (verdict logic never modified; only two commits in its history). |
| 2026-02-23 | `b42b112` | "Close Phase C: Lc physical validation passes when resolution difference is accounted for." The 02-17 entry was **rewritten** to "both checks pass"; a new 02-23 "Closing analysis" argues max() is not resolution-fair (90 m MERIT ≈ 12 K smoothed pixels vs 25 M raw pixels at 1 m) and adopts **P95/Lc = 1.17 at 300 m** as the comparison; summary table lists max/Lc as "N/A — not comparable across resolutions". Conclusion: "**Lc = 300m is confirmed.** … PI acceptance, and both physical validation checks are consistent." The commit diff mentions the PI only for the 02-11 acceptance; **it does not record a PI decision on the P95 substitution.** |
| 2026-02-26 | Phase D tier runs | Rebuilt pipeline computes **356.0 m** on the production domain (no floor active). |
| 2026-03-16 | `25e24f8`, divergence audit | First record stating 356 m as *the* Lc: `audit/260310-osbs_pipeline_and_docs/osbs-pipeline-divergence-audit-260316.md:46` "Lc = 356m (range 285-356m)". Same day `MIN_LC_PIXELS` removed. |
| 2026-03-17 / 03-19 | `df61c7a` | The docs-update plan still says "~300m (range 285-356m)" at `:15` but quotes the P95 = 1.17 check at `:199`. |
| 2026-03-23 | `cb37aef` | STATUS §3 still reads "Lc = 300m (range 285-356m), A_thresh = 45,000 m²". |
| 2026-04-10 | `02e805a` | STATUS condensed; Phase C row now "Complete (Lc = 356m, A_thresh = 63,362)". No log entry explains the switch. |
| 2026-05-12 | `d062ec2` | STATUS locked-decision table created with "Lc = 356 m (production domain) \| 2026-02-11 \| Phase C" — the date is the 02-11 ~300 m acceptance. |
| 2026-10-20 | hpg-esm-docs `6d16d92` | First audit re-quoted the checks at Lc = 356 from the JSON (P95/Lc 1.09, mean/Lc² 0.877, max 2.6) with the 300 m values parenthetical. |

**What the trace shows.**

1. The *script's* verdict was never PASS. The reinterpretation lives only in prose (Phase C log, STATUS, the public page). The script and its JSON/summary still say FAIL/MARGINAL.
2. The 02-17 log called it "a PI interpretation question." The 02-23 rewrite resolved it but did not say who decided. That is consistent with the user's recollection (results given to the PI, PI overrode the scripted verdict) but the record does not prove it. **This is the gap the user's email search needs to close.**
3. Phase C closed on **300 m**, with the physical checks evaluated at 300 m. The operative value 356 m came from the pipeline a week later and entered the narrative without a logged decision; STATUS retroactively dated it to 02-11.
4. The checks were evaluated on a sub-domain whose own Lc is ~479 m, at candidate values from the full-domain spectrum.

None of this changes the numbers on the public page (all re-read from `validation_results.json` on 2026-10-20 and again today). It changes how the story should be told.

---

## 3. Draft revision for item 2.2 (Lc narrative) — APPLIED 2026-10-21 (with an added *Implementation* paragraph, item 2.2b; see the hpg-esm-docs working tree)

### `swenson/index.md:42` — current

> C established **Lc = 356 m** via a restricted-wavelength FFT (20 m cutoff), circumventing the k² Laplacian artifact that surfaces at high-resolution DEMs.

### Proposed

> C established the method for Lc: a restricted-wavelength FFT of the DEM Laplacian with a 20 m cutoff, circumventing the k² artifact that at 1 m puts the spectral peak at ~8 m. On the contiguous mosaic the cutoff sweep gave 356 m (20 and 50 m cutoffs, lognormal) to 285 m (100 m cutoff, Gaussian); the PI accepted ~300 m (range 285–356 m) as the working value on 2026-02-11, and the two physical-plausibility checks were run at 285, 300 and 356 m. Lc is not prescribed: the production pipeline computes it at run time from the FFT (`run_pipeline.py` fixes only the 20 m cutoff), and on the 90-tile production domain it returns **Lc = 355.98 m**, the upper end of the Phase C range. Lc is domain-dependent — the single-tile and 5 × 5 test domains give 541 and 479 m — so the production value is specific to R4C5–R12C14.

### `osbs-implementation.md` *Result* subsection (after the sweep table) — current

> **Lc = 356 m** (355.98 m; lognormal fit at 20 m cutoff), **A_thresh = 63,362 pixels** at 1m resolution.

### Proposed

> **Lc = 355.98 m** (lognormal fit at the 20 m cutoff), **A_thresh = int(0.5 × Lc²) = 63,362 pixels** at 1 m resolution. Lc is computed by the pipeline at run time on the production mosaic; the 20 m cutoff is the only prescribed quantity (`run_pipeline.py`, `MIN_WAVELENGTH = 20`). During Phase C the PI accepted ~300 m (range 285–356 m) as the working value, and the plausibility checks below were run at the three candidates; the rebuilt pipeline's first full-domain run (2026-02-26) returned 356 m, which has been the production value since. Lc is domain-dependent: the same method gives 541 m on the single tile R6C10 and 479 m on the 5 × 5 block used for the plausibility checks.

(The last sentence also pre-empts the "checks run on a different domain" question that item 2.3 raises.)

---

## 5. Further investigation (2026-10-21, second pass) — docs, scripts, audits, paper

Searched: the validation script's own text, every `audit/` directory, all phase files and STATUS versions from 2026-02-11 to 03-23, Phase A/B/D/E/F outputs, the hpg-esm-docs history of the plausibility passage, and the Swenson & Lawrence (2025) PDF (`docs/papers/2025-swenson-global-hillslope-dataset.pdf`, text extracted to the scratchpad).

### 5.1 The script encoded a stricter criterion than the paper states — the most important find

The script header and summary assert "Swenson Section 2.4 calibration (low-relief): max(DTND)/Lc = 0.90, mean(catch)/Lc² = 0.94". The paper (Section 2.4, pp. 8–9) actually says:

- High-relief gridcell (Lc = 1,830 m): "the resulting mean catchment area is 3.1 × 10⁶ m², which can be compared to … Lc² = 3.3 × 10⁶ m², confirming that the choice of accumulation threshold gives catchment areas that are consistent with Lc" → ratio 0.94. "It can be seen that the largest values of DTND (i.e., the ridge to channel distances) are of similar magnitude to the value of Lc."
- Low-relief gridcell (Lc = 805 m): "mean catchment area is 6.1 × 10⁵ m², which can be compared to … Lc² = 6.5 × 10⁵ m²" → ratio 0.94. "**Typical values of DTND along the ridges are about 800 m, in accordance with the inferred Lc value**" → 800/805 ≈ 0.99.

So: (a) the 0.94 catchment ratio is real and is the same for both of the paper's examples; (b) **there is no 0.90 max(DTND)/Lc figure in the paper**, and the paper's DTND statement is qualitative — "largest values … of similar magnitude" and "typical values along the ridges" — a statement about characteristic ridge distances, not a strict pixel maximum. The script's `check1 = max(DTND)/Lc` with PASS 0.5–2.0 was a tighter encoding than the paper supports. On that reading, replacing max() with a high percentile (P95 ≈ "typical ridge distance") is a *correction toward the paper's criterion*, not a softening of it. This is the strongest argument available for the P95 reinterpretation and it is not made anywhere in the record or on the public page (both argue only from resolution/sample-size).

### 5.2 The planned "PI discussion" was dropped from the record, not completed

STATUS at `3af1c4d` (2026-02-17) listed under Phase C "Remaining: 10. Revisit Check 1 interpretation after refactoring — max vs P99, domain size effects, PI discussion". At `b42b112` (02-23) that line was deleted and the deliverable rewritten to "Lc = 300m with scientific justification"; nothing marks the discussion as having happened. No phase file, STATUS version, or commit message between 02-12 and 03-23 mentions the PI other than the 02-11 acceptance. The only candidate record is off-repo (the user's email).

### 5.3 Three other DTND-check data points exist, and they point the same way

| Run | Domain | Lc used | DTND stats | Ratios | Source |
|---|---|---|---|---|---|
| R6C10 smoke, 2026-02-12 | 1 tile (1 km²) | 300 (hard-coded) | max 535 m | max/Lc 1.78 | `output/osbs/smoke_tests/r6c10_utm/r6c10_validation.json` ("DTND max ≤ 750 m" check) |
| Hasty 5×5 UTM smoke, 2026-02-18 (job 25254030) | 5×5 block | 300 | not recorded | "3 threshold calibration failures that were never analyzed"; logs deleted | `phases/A-pysheds-utm.md:352-363`; script `run_5x5_utm.py` (deleted `cadff06`) carried Check 1/2 with the same thresholds |
| Phase B scalability, 2026-02-23 | **90-tile production domain** | 300 (A_thresh 45,000) | mean 190, P95 471, max 1,989 m | **P95/Lc 1.57**, max/Lc 6.6 | `output/osbs/phase_b/scalability_64gb.json` |
| Phase C validation, 2026-02-17 | 5×5 block | 285/300/356 | P95 335/350/389, max 931 | P95/Lc 1.17/1.17/1.09; max/Lc 3.27/3.10/2.62 | `validation_results.json` |

The published figure (P95/Lc = 1.09) comes from the 5×5 block, which is the most favourable domain tested. On the full production domain the only measurement (at Lc = 300) gives P95/Lc ≈ 1.6 and max/Lc ≈ 6.6; the single-tile max/Lc of 1.78 "passes" only because a 1 km tile truncates flow paths. No DTND distribution was saved for a production run at Lc = 356 (the 2026-02-26 tier outputs are gone; the production directory holds only JSON/NetCDF/PNG; no SLURM log for job 31931926 survives in the repo), so the Check 1 ratio on the actual production domain at the actual production Lc is **unknown**. Check 2 on the production domain is also unknown: the production summary does not record the catchment count (the pipeline prints "Hillslopes: N" to stdout only); Phase E's "~508 hillslopes" (2026-03-27, on the 90-tile domain) would give mean catchment ≈ 90 km²/508 ≈ 1.8 × 10⁵ m² and a ratio of ≈ 1.4 against Lc² = 1.27 × 10⁵ m², but that count is informal and predates the final mask.

Recomputing both checks on the production domain at Lc = 355.98 m is a ~15-minute job (routing on the production mosaic is 67 s per the Phase C run scaled, ~6 min per Phase B); not run — read-only investigation.

### 5.4 Provenance of the public "PASS" wording

The PASS labels first appeared in hpg-esm-docs `fe52a92` (2026-03-17, "Complete rewrite of methodology page (methods-section tone with verified production data) … All claims audited against phase docs"), i.e. they were copied from the 02-23 Phase C closing analysis, and survived the 2026-10-12 and 10-20 factual passes because the numbers are correct. The 03-17 docs-update plan (`audit/260310-…/docs-update-plan-260317.md:199`) likewise lists "Physical validation: P95 DTND/Lc = 1.17" as a verified claim.

### 5.5 What this means for item 2.3

1. The P95 substitution has a better justification than the one on record: the paper's own criterion is "typical ridge DTND ≈ Lc", not max(). That argument should lead; the resolution/sample-size argument is supporting.
2. The PI's role is still unrecorded in the repos. The user's email search is the only way to close it.
3. The honest statement of what was *tested* is narrower than the page implies: both checks were evaluated on a 5×5 sub-domain at three candidate Lc values; on the production domain at the production Lc they have not been evaluated. Whether to run that (cheap) before the methods section is written is a user/PI decision — added to the to-do list as an option.

---

## 4. Proposed wording for item 2.3 — HELD until the PI conversation (revised 2026-10-21 after §5)

Replace the two bold **PASS** labels in `osbs-implementation.md` *Physical plausibility checks* with a disclosed reinterpretation. Two variants depending on what the user learns:

**(a) If the PI endorsed the P95 substitution** (expected, per the user's recollection):

> Swenson & Lawrence (2025, Section 2.4) give two consistency statements for Lc: the mean catchment area should be close to Lc² (0.94 × Lc² in both of their examples), and "typical values of DTND along the ridges" should be of similar magnitude to Lc. Both were evaluated on the 5 × 5 tile block R6C7–R10C11 at Lc = 285, 300 and 356 m. The catchment criterion holds at every Lc (mean area/Lc² = 0.85–0.88). For the DTND criterion the validation script originally used the pixel maximum, max(DTND)/Lc within 0.5–2.0, which returned 3.27, 3.10 and 2.62 (FAIL, FAIL, MARGINAL); on review with the PI that statistic was replaced by the 95th percentile, for two reasons. The paper's criterion is a typical ridge distance, not a maximum, and max() over 25 M raw 1 m pixels is an extreme-value statistic not comparable with a 90 m MERIT gridcell of ~12 K smoothed pixels. P95/Lc = 1.09 at 356 m (1.17 at 300 m). The 931 m maximum is the same ridge pixel at every Lc — real terrain, not an Lc mismatch. The checks have not been re-run on the full production domain at the production Lc; the one domain-wide DTND measurement (Phase B, Lc = 300 m) gives P95/Lc ≈ 1.6.

**(b) If the record cannot be confirmed:** same text with "on review with the PI" → "in the 2026-02-23 closing analysis".

Either way the verdict column becomes: Check 1 (P95/Lc) 1.09 — meets the paper's criterion as reinterpreted; Check 1 (max/Lc) 2.62 — scripted verdict MARGINAL, not used; Check 2 0.877 — PASS.
