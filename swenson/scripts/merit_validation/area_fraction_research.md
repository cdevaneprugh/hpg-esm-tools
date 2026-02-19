# Area Fraction Diagnostic Research

Living document tracking investigation of why the MERIT area fraction correlation
is 0.82 — the weakest of our 6 hillslope parameters.

**Diagnostic script:** `area_fraction_diagnostics.py`
**SLURM logs:** `../../logs/area_frac_diag_2528{1952,2406,2903}.log`
**pysheds branch:** `audit/pgrid-and-tests` (commit 61872df)

---

## 1. Problem Statement

The MERIT regression validates our pipeline against Swenson's published data for a
single 0.9x1.25 degree gridcell. Five of six parameters match at >0.95 correlation:

| Parameter | Correlation |
|-----------|-------------|
| Height (HAND) | 0.9999 |
| Aspect (circular) | 0.9999 |
| Distance (DTND) | 0.9992 |
| Slope | 0.9966 |
| Width | 0.9586 |
| **Area fraction** | **0.8197** |

Area fraction is the fraction of total gridcell area assigned to each of the
16 hillslope elements (4 aspects x 4 HAND bins). At 0.82 it is noticeably below
the other five, and the gap concentrates in specific elements rather than being
spread uniformly.

The question: is the gap caused by a missing filter, a bug, or an inherent
geometry/binning difference?

---

## 2. Test Gridcell

| Property | Value |
|----------|-------|
| Center | 92.5W, 32.5131N |
| Lon bounds | -93.125 to -91.875 |
| Lat bounds | 32.0419 to 32.9843 |
| Resolution | ~90m (MERIT DEM) |
| DEM file | `data/merit/n30w095_dem.tif` |
| Published reference | `data/reference/hillslopes_0.9x1.25_c240416.nc` |
| Pixel count | 1,695,369 total; 1,581,926 valid (HAND >= 0) |
| Negative-HAND pixels | 113,443 (6.69%) |
| HAND range | -20.09 to 72.58 m |
| DTND range | 0.00 to 3,945 m |
| Lc (FFT) | 8.2 px, A_thresh = 33 |
| HAND bin bounds | [0.0, 2.0, 5.57, 10.61, 1e6] |

The first bin boundary is fixed at 2.0m (the mandatory lowest-bin constraint from
Swenson's paper). Bins 2 and 3 are percentile-derived at 5.57m and 10.61m.

---

## 3. Baseline Characterization

### Per-aspect sub-correlations

| Aspect | Correlation | Our sum | Published sum |
|--------|-------------|---------|---------------|
| **North** | **0.5549** | 0.2253 | 0.2397 |
| East | 0.9865 | 0.2769 | 0.2527 |
| South | 0.9998 | 0.2443 | 0.2511 |
| West | 0.9964 | 0.2535 | 0.2565 |

North is the clear outlier at 0.55 — the other three are all above 0.98.

### Per-element breakdown

| Elem | Aspect | Bin | Ours | Published | Delta | Rank |
|------|--------|-----|------|-----------|-------|------|
| 0 | North | 0 | 0.0667 | 0.0612 | +0.0054 | 5 |
| 1 | North | 1 | 0.0510 | 0.0564 | -0.0053 | 6 |
| 2 | North | 2 | 0.0512 | 0.0577 | -0.0065 | 3 |
| 3 | North | 3 | 0.0564 | 0.0644 | -0.0080 | 2 |
| 4 | East | 0 | 0.0824 | 0.0696 | +0.0128 | 1 |
| 5 | East | 1 | 0.0641 | 0.0616 | +0.0024 | 9 |
| 6 | East | 2 | 0.0651 | 0.0605 | +0.0046 | 7 |
| 7 | East | 3 | 0.0654 | 0.0610 | +0.0044 | 8 |
| 8 | South | 0 | 0.0750 | 0.0806 | -0.0056 | 4 |
| 9 | South | 1 | 0.0579 | 0.0588 | -0.0009 | 13 |
| 10 | South | 2 | 0.0566 | 0.0574 | -0.0007 | 14 |
| 11 | South | 3 | 0.0547 | 0.0544 | +0.0003 | 16 |
| 12 | West | 0 | 0.0769 | 0.0782 | -0.0012 | 11 |
| 13 | West | 1 | 0.0577 | 0.0590 | -0.0013 | 10 |
| 14 | West | 2 | 0.0577 | 0.0571 | +0.0006 | 15 |
| 15 | West | 3 | 0.0611 | 0.0622 | -0.0011 | 12 |

The largest absolute error is element 4 (East bin-0) at +0.0128, followed by
element 3 (North bin-3) at -0.0080. Bin-0 (lowest HAND, near-stream) elements
consistently show the biggest deltas.

### Trapezoidal fit intermediates

| Aspect | Pixels | n_hillslopes | trap_area | trap_width | trap_slope |
|--------|--------|--------------|-----------|------------|------------|
| North | 351,929 | 24,211 | 105,201 | 260.86 | -0.1617 |
| East | 441,929 | 24,735 | 129,301 | 313.81 | -0.1904 |
| South | 382,919 | 24,292 | 114,039 | 269.50 | -0.1592 |
| West | 405,149 | 24,784 | 118,337 | 283.75 | -0.1701 |

North has the fewest pixels (351K vs 442K for East). All four aspects have
negative trap_slope (convergent plan form).

---

## 4. Hypotheses Tested

### Test A: DTND tail removal (TailIndex)

**What it tests:** Swenson's `TailIndex` fits an exponential to the DTND
distribution and removes pixels in the tail (above 5% of max PDF). These are
anomalously distant pixels, often on ridges or divides.

**Why it might matter:** Tail pixels could inflate bin-3 (ridge) areas,
distorting the distribution.

**Result:** Removed 31,338 pixels (1.85%). Correlation **decreased** from
0.8197 to 0.8097 (-0.0100). Width also dropped from 0.9586 to 0.9049.

**Interpretation:** The tail pixels are mostly where we expect them — removing
them worsens the fit. Our baseline is already closer to Swenson's answer without
this filter than with it. This may mean Swenson's published data was also
computed without tail removal, or that the effect on area fractions was neutral
in his pipeline.

### Test B: DTND minimum clipping

**What it tests:** Swenson clips DTND values below 1.0m to 1.0m
(`representative_hillslope.py:700-701`). These are pixels essentially on the
stream network but not classified as stream.

**Why it might matter:** Zero-distance pixels could distort the trapezoidal fit
at the origin.

**Result:** Clipped 179,162 pixels (10.57%) with DTND < 1.0m. Correlation
unchanged at 0.8197.

**Interpretation:** Minimum clipping doesn't affect area fractions because it
changes DTND values but not which bin a pixel lands in (binning is by HAND, not
DTND). It would only matter for distance and width computation.

### Test C: Flooded region handling

**What it tests:** Swenson identifies open water bodies via a slope-based mask
(`identify_open_water(slope, max_slope=1e-4)`), then marks low-HAND pixels in
flooded areas as HAND = -1, excluding them from binning.

**Why it might matter:** Flooded/open-water pixels have artificially low HAND and
should not be treated as hillslope elements.

**Critical bug discovered (Runs 1-2):** Our initial `fflood` was computed as
`np.array(grid.dem) - np.array(grid.flooded)` — the DEM-minus-conditioned-DEM
difference, a continuous field ranging 0-24m. Swenson's `fflood` is
`basin_mask` — a binary 0/1 field from `identify_open_water()` (confirmed at
`representative_hillslope.py:1702`).

With the wrong (continuous) fflood, the flood filter's threshold sweep over
`linspace(0, 20, 50)` behaved catastrophically: it marked 15.72% of pixels as
flooded (266,488 pixels), crashing the correlation to **-0.3165**.

**Result with correct fflood (Run 3):** `identify_open_water()` found zero
pixels with slope < 0.01% in this gridcell. The morphological erosion/dilation
removed any isolated low-slope candidates. Basin mask was entirely zero. The
flood filter is a **no-op** at 90m MERIT resolution for this gridcell.
Correlation unchanged at 0.8197.

**Interpretation:** The flood filter is designed for finer-resolution data where
open water bodies are resolved as contiguous flat regions. At 90m, even real
water bodies have enough slope variation from surrounding terrain to exceed the
threshold. This is consistent with Swenson's design intent — the filter catches
lakes/ponds that would otherwise corrupt the hillslope model.

See Section 6 for full bug analysis.

### Test D: Mean-HAND bin skip

**What it tests:** Swenson skips bins where `mean(HAND) <= 0`
(`representative_hillslope.py:819`), setting their area to zero.

**Why it might matter:** Bins with negative mean HAND represent areas below the
stream channel — physically unreasonable hillslope positions.

**Result:** No bins had mean HAND <= 0. Correlation unchanged at 0.8197.

**Interpretation:** With our bin bounds starting at 0.0 and the valid mask
excluding HAND < 0, no bin can have mean HAND <= 0. This filter would only
trigger if negative-HAND pixels were included in the binning (as in Swenson's
code where `isfinite` is the only mask).

### Test E: All filters combined (A+B+C+D)

Applied in Swenson's order: tail removal, flood filter, DTND clipping, with
bin skip enabled.

**Result with wrong fflood (Runs 1-2):** -0.3250 (dominated by bug C).

**Result with correct fflood (Run 3):** 0.8097 — same as Test A alone, since C
and D are no-ops and B doesn't affect area fractions.

### Test F: Swenson-style valid mask (include negative HAND)

**What it tests:** Swenson's aspect population includes all finite pixels, not
just HAND >= 0. This affects n_hillslopes (used in trapezoidal fit denominator),
the trapezoidal fit itself (more pixels → different area curve), and the area
fraction denominator.

**Why it might matter:** 6.69% of pixels have negative HAND. If they cluster in
one aspect, excluding them could skew the aspect population ratios.

**Result:** Correlation changed from 0.8197 to 0.8196 (-0.0001). Width improved
marginally (0.9586 to 0.9610).

**Interpretation:** Negative-HAND pixels are uniformly distributed across aspects
(6.25-7.23% per aspect — see diagnostics below). Including or excluding them
makes no practical difference.

**Negative-HAND distribution by aspect:**

| Aspect | HAND < 0 | HAND >= 0 | Total | % negative |
|--------|----------|-----------|-------|------------|
| North | 27,413 | 351,929 | 379,342 | 7.23% |
| East | 29,439 | 441,929 | 471,368 | 6.25% |
| South | 25,620 | 382,919 | 408,539 | 6.27% |
| West | 30,971 | 405,149 | 436,120 | 7.10% |

### Test G: Swenson mask + DTND tail removal

**What it tests:** Combining the Swenson-style valid mask (Test F) with tail
removal (Test A).

**Result:** 0.8101 (-0.0096). Slightly better than Test A alone (0.8097) due to
the mask interaction, but still worse than baseline.

### Test H: Full Swenson pipeline (corrected fflood)

**What it tests:** All of Swenson's filters in order — tail removal, flood filter
(correct binary mask), DTND clipping — plus Swenson-style valid mask and
mean-HAND bin skip. This is our best approximation of Swenson's actual
processing path.

**Result:** 0.8101 (-0.0096). Identical to Test G because the flood filter and
bin skip are both no-ops.

---

## 5. Sensitivity Analysis

For each of the 16 elements, we replace its area fraction with the published
value (renormalizing the remaining 15 proportionally) and measure the change in
overall correlation. This identifies which elements drive the gap.

| Elem | Aspect | Bin | Corr if fixed | Delta | Impact |
|------|--------|-----|---------------|-------|--------|
| 8 | South | 0 | 0.8648 | +0.0451 | *** |
| 4 | East | 0 | 0.8634 | +0.0436 | *** |
| 3 | North | 3 | 0.8475 | +0.0278 | *** |
| 0 | North | 0 | 0.8325 | +0.0128 | ** |
| 6 | East | 2 | 0.8302 | +0.0105 | ** |
| 7 | East | 3 | 0.8287 | +0.0090 | * |
| 12 | West | 0 | 0.8270 | +0.0073 | * |
| 2 | North | 2 | 0.8248 | +0.0051 | * |
| 5 | East | 1 | 0.8225 | +0.0028 | |
| 14 | West | 2 | 0.8210 | +0.0012 | |
| 11 | South | 3 | 0.8206 | +0.0009 | |
| 15 | West | 3 | 0.8200 | +0.0003 | |
| 1 | North | 1 | 0.8193 | -0.0004 | |
| 9 | South | 1 | 0.8189 | -0.0008 | |
| 13 | West | 1 | 0.8188 | -0.0009 | |
| 10 | South | 2 | 0.8186 | -0.0011 | |

**Key pattern: bin-0 elements dominate.** Three of the top four impactful
elements are bin-0 (lowest HAND, near-stream): South-0, East-0, and North-0.
The exception is North-3 (ridge), which reflects North's uniquely poor
sub-correlation (0.55).

If elements 8 and 4 (the two bin-0 outliers) were corrected, the overall
correlation would jump to ~0.86+. The gap is not spread uniformly — it
concentrates in the lowest HAND bin and in North-facing slopes.

---

## 6. Bug Found: fflood DEM-Difference vs Binary Mask

### The error

Runs 1-2 computed `fflood` as:
```python
fflood = np.array(grid.dem) - np.array(grid.flooded)  # WRONG: continuous 0-24m
```

Swenson's code sets (at `representative_hillslope.py:1702`):
```python
self.fflood = basin_mask  # CORRECT: binary 0/1 from identify_open_water()
```

### Why the difference is catastrophic

The flood filter (`representative_hillslope.py:678-696`) sweeps thresholds
through `linspace(0, 20, 50)` looking for the value where < 95% of flooded
low-HAND pixels remain:

```python
for ft in np.linspace(0, 20, 50):
    if (sum(|fflood[HAND < 2]| > ft) / num_flooded_pts) < 0.95:
        flood_thresh = ft
        break
```

With **binary input** (0/1): `num_flooded_pts` counts pixels where
`basin_mask == 1`. The first threshold value > 0 in the sweep (0.408) already
satisfies the < 95% condition, but since all values are 0 or 1, the threshold
is irrelevant — no pixel can exceed it. With no basin_mask pixels at 90m, the
entire filter is a no-op.

With **continuous input** (0-24m): `num_flooded_pts` counts any pixel where the
DEM-to-flooded difference is nonzero — 391,414 pixels (23% of the gridcell).
The threshold sweep finds a value in the middle of the continuous distribution,
marking 266,488 pixels (15.72%) as flooded. These are legitimate hillslope
pixels whose HAND gets set to -1, and area fraction crashes to -0.3165.

### The fix

Run 3 replaced the DEM-difference computation with Swenson's actual approach:
compute slope, run `identify_open_water(slope, max_slope=1e-4)`, and use the
resulting binary `basin_mask`. See `area_fraction_diagnostics.py` lines 71-148
and 324-328.

### Swenson also uses basin_mask to modify DEM before routing

At `representative_hillslope.py:1549`:
```python
flooded_dem[basin_mask > 0] -= 0.1
```

This depresses open-water areas by 0.1m so that flow routes through them rather
than around them. We don't do this, but it's moot since the mask is empty at 90m
MERIT resolution.

---

## 7. Current Understanding

**The 0.82 correlation appears to be a geometry/binning ceiling, not a filtering
issue.**

Evidence:
- None of the 8 hypothesis tests improved the correlation. The only tests with
  any effect were harmful (A: -0.01, G/H: -0.01).
- The gap concentrates in specific structural positions (bin-0, North bin-3)
  rather than being uniformly distributed, suggesting it arises from how pixels
  are assigned to bins rather than from a missing preprocessing step.
- Swenson's own full filter pipeline (Test H) produces 0.8101, slightly worse
  than our baseline.

**Possible sources of the remaining gap:**

1. **Trapezoidal fit sensitivity in bin-0.** The lowest HAND bin captures the
   near-stream zone where the hillslope-to-channel transition occurs. Small
   differences in the fitted trapezoidal model parameters (which set
   `fitted_area` for each bin) propagate most strongly into bin-0 because that's
   where the width function changes fastest (it's the base of the trapezoid).

2. **North aspect geometry.** North has the fewest pixels (352K vs 442K for
   East) and the worst sub-correlation (0.55). This could reflect a real
   asymmetry in catchment structure at this gridcell, or a numerical sensitivity
   in the trapezoidal fit when the population is smaller.

3. **Differences in stream network delineation.** If our stream network differs
   slightly from Swenson's (due to different pysheds versions, conditioning
   parameters, or the DEM depression step), the resulting HAND field would shift
   pixels between bins near the boundaries, especially in bin-0.

4. **Aspect boundary effects.** Aspect boundaries are hard cutoffs at 45/135/
   225/315 degrees. Pixels near these boundaries are assigned to one aspect or
   the other based on small angle differences. The diagnostics show the
   boundaries are near-symmetric (ratios 0.986-1.018), so this is likely a minor
   contributor at most.

---

## 8. Open Questions

1. **Is 0.82 acceptable?** The other 5 parameters are all above 0.96. Does a
   0.82 area fraction correlation materially affect CTSM simulations? The 16
   area fractions sum to 1.0 by construction, so errors in one element are
   offset by errors elsewhere. The physical impact depends on how CTSM uses area
   fractions in the lateral flow computation.

2. **Does Swenson's published data use tail removal?** If the published dataset
   was generated with TailIndex, our baseline (without tail removal) would be
   expected to differ. If it was generated without, our baseline is the right
   comparison point. The paper doesn't specify.

3. **Stream network comparison.** We could extract our stream network mask and
   Swenson's (if available) to compare directly. Differences in stream pixel
   classification would cascade through HAND, DTND, and all downstream
   parameters.

4. **Trapezoidal fit diagnostics.** The trapezoidal model fits A_cumsum(d) — the
   area with DTND >= d — for each aspect. Plotting the actual A_cumsum curves
   against the fitted quadratic could reveal where the model deviates, especially
   for North.

5. **Try alternative bin boundary computation.** Our `compute_hand_bins` matches
   Swenson's `SpecifyHandBounds`, but the 2.0m first-bin boundary was set for
   global applicability. A different first-bin boundary could redistribute pixels
   between bin-0 and bin-1, potentially improving the fit for this specific
   gridcell. (Not recommended — would break comparability.)

---

## Appendix: Run History

| Run | Log | Date | Changes from previous |
|-----|-----|------|-----------------------|
| 1 | `area_frac_diag_25281952.log` | 2026-02-19 12:01 | Initial: Tests A-E. fflood was DEM-difference (bug). |
| 2 | `area_frac_diag_25282406.log` | 2026-02-19 12:15 | Added Tests F-G, negative-HAND diagnostics. fflood still wrong. |
| 3 | `area_frac_diag_25282903.log` | 2026-02-19 12:25 | Corrected fflood to binary basin_mask. Added Test H. |

### Corrected summary table (Run 3 values)

| Test | Area frac | Delta | Verdict |
|------|-----------|-------|---------|
| Baseline | 0.8197 | -- | Starting point |
| A: DTND tail removal | 0.8097 | -0.0100 | Harmful |
| B: DTND min clipping | 0.8197 | +0.0000 | No effect |
| C: Flood filter (correct fflood) | 0.8197 | +0.0000 | No effect (no open water at 90m) |
| D: Mean-HAND bin skip | 0.8197 | +0.0000 | No effect |
| E: All combined | 0.8097 | -0.0100 | = Test A alone |
| F: Swenson valid mask | 0.8196 | -0.0001 | No effect |
| G: Swenson mask + tail | 0.8101 | -0.0096 | Slightly harmful |
| H: Full Swenson pipeline | 0.8101 | -0.0096 | = Test G |

### All-parameter correlation table (Run 3)

| Parameter | Baseline | All (E) | Swenson (F) | S+tail (G) | Full (H) |
|-----------|----------|---------|-------------|------------|----------|
| height | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| distance | 0.9992 | 0.9989 | 0.9996 | 0.9990 | 0.9990 |
| slope | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9966 |
| aspect | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| width | 0.9586 | 0.9049 | 0.9610 | 0.9142 | 0.9142 |
| area_frac | 0.8197 | 0.8097 | 0.8196 | 0.8101 | 0.8101 |
