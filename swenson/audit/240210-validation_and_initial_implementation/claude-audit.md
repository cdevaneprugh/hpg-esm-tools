# Audit of Claude's Work in the Swenson Implementation

## pysheds Test Suite
There are three pytest tests in the pysheds fork.
1. `conftest.py`
    - pytest fixture file
2. `test_grid.py`
    - Massive failures due to API mismatch after Swenson methods added
    - Functions tested here are prerecs to test_hillslope so not worth fixing
    - Notes/skips were added to provide context for future testing
3. `test_hillslope.py`
    - hangs on initial run due to Scipi import quirk
    - `test_river_network_length_slope_runs` was incorrectly written. now fixed
    - **33 deprecation warnings** carried over from Swensons older code
        - Need to address after audit of existing work

## Phase 3: Validating Against Swenson's Work
I moved Claude's test results to a new directory `claude_merit_validation`.
I will be rerunning the validation scripts again with output in `merit_validation`.
Apart from a few paths, everything in the tests will stay unchanged unless noted.

**All validation tests were run and matched Claude's claims except where bugs were fixed.**

### Stage 1: pgrid Validation

**Script Overview**
Validates that the pgrid methods work on a full tile
1. Loads the data
2. Preps data (fills pits and depressions) **LIKELY A PROBLEM FOR 1M DATA**
3. Calculate flow direction using fixed accumulation threshold (actual acc will be calculated via the FFT)
4. Creates the stream network "mask". A boolean array the same shape as the inputdata.
    - Pixels are classified as a stream (True/1) if its accumulation value exceeds the threshold value.
    - Gets passed to `create_channel_mask()` which assigns more granular information (banks, headwater, etc)
5. HAND/DTND are calculated by tracing the flow path using the `compute_hand()` method.
    - Starts at a non stream pixel, folows the D8 direction downstream until it hits a stream pixel
    - HAND = strarting pixel ele - stream pizel ele
    - DTND = flow path distance from starting to stream pixel (NOT as the crow flies)
6. Generate plots, save output

**Questions**
1. Define accumulation threshold and why you used a fixed treshold (in number of cells) for initial validation (Line 62).
    - Minimum number of upstream pixels required to classify a pixel as part of the stream.
    - Used a fixed value b/c this is a VALIDATION script only, not a scientific one.

2. Line 260: "Step 2". Why fill pits and depressions? Is this what Swenson did? Why did he do it?
    - Swenson did this to ensure every pixel has somewhere to route water to.
    - **Potentially a significant problem at a 1m resolution**. At 1m, some "pits" are likely real features.

3. Line 288: Why is the dirmap hardcoded? If it is a pysheds convention should it be implemented in the fork?
    - This is the convention Swenson used to map the D8 flow direction calculation. We're following his work as a guide.
    - **It's arguable this should be hardcoded into the fork.**

### Stage 2: Length Scale Analysis Using FFT

**Questions**
1. Line 54: How did it choose analysis parameters? MAX_HILLSLOPE_LENGTH, NLAMBDA
    - Both are from Swenson. A 10km max hillslope length sets an upper bound on what wavelengths the FFT looks for.
    - NLAMBDA is the num of bins

2. Effects of subsampling?
    - Line 262: subsampled to ~1500x1500
    - Full tile is 6000x6000 pixels --> subsampling x4 looks at every fourth pixel (90m to 360m resolution)
    - FFT only detects features at 2x360=720. Our Acc was 760m (close to the limit) so the FFT focused on larger scale features.
    - This gave us a acc of ~2700m for the subsampled full tile (incorrect)

3. The spatial_scale.py file is shared between the merit_validation and osbs script dirs.
    - Arguable some of these functions could be integrated into the pysheds fork,
    - The FFT specific methods should be left on their own though.

5. Line 228: Explain the paramater choices for the spatial scale analysis (detrend_elevation, land_threshold, etc). Why did you choose them?
    - All directly copied from Swenson.
    - Worth looking at changing

### Stage 3: Hillslope Params

**Questions**
1. We define 4 aspect bins and 4 HAND bins. OSBS is a low relief wetlandscape. We would like one hillslope to characterize the "average" hillslope.
    - How would we accomplish this? Calculate 4 hillslopes then average them? Can we just calculate one (with say 8 HAND bins) straight away?
    - What would this look like in practice? We use aspect bins to classify the N,E,S,W hills. How woud this work for one hill? Is there a risk of messing up our data analysis with one 0-360 aspect?
    - How does this affect the trapezoidal area calculation?
    - **Shouldn't be too much of a problem calculating one hillslope straight away**
  
  What CTSM supports: The hillslope configuration is fully flexible. CTSM reads nhillslope and nmaxhillcol as dimensions from the input file (surfrdMod.F90
   lines 1082-1096). There's no hardcoded requirement for 4 aspects or 4 elevation bins. You could have nhillslope=1 with nmaxhillcol=8 and CTSM would     
  handle it.
                                                                                                                                                           
  How to compute 1 hillslope directly: Skip aspect binning entirely. Treat all non-stream pixels as belonging to one hillslope, then bin by HAND into 8    
  elevation bins. The pipeline would:                                                                                                                      
  1. Compute HAND/DTND for all pixels (unchanged)                                                                                                          
  2. Compute slope/aspect per pixel (unchanged)
  3. Skip aspect classification — all pixels go into one pool
  4. Bin the single pool into 8 elevation bands by HAND
  5. Compute mean height, distance, slope, area per band
  6. Fit one trapezoidal width model using all pixels' A_sum(d) curve
  7. Compute width at each elevation band boundary

  The aspect problem: This is the main risk. Each column in CTSM still gets an hillslope_aspect value, which controls aspect-dependent insolation (solar
  radiation varies by slope orientation). With one hillslope, all 8 columns get the same aspect. If you compute a circular mean of all pixel aspects in a
  site where aspect is approximately uniformly distributed (24-26% per quadrant in the interior results), the resultant vector has near-zero magnitude —
  the mean direction is essentially undefined. You'd get an arbitrary number.

  This actually matters less than it sounds for OSBS specifically. Aspect-dependent insolation scales with slope * cos(aspect - solar_azimuth). When slope
  is near zero (OSBS slopes are 0.01-0.06 m/m), the insolation correction is tiny regardless of aspect. So an arbitrary aspect value on a nearly flat
  hillslope has negligible physical impact.

  The trapezoidal fit: Works the same way with one hillslope — you build A_sum(d) from all non-stream pixels rather than from one aspect quadrant's pixels.
   You'd have 4x more data points going into the polynomial fit, which should give a more robust fit. The resulting trapezoid describes the "average" plan
  form geometry of the site.

  Practical recommendation: Computing 1 hillslope with 8 elevation bins is straightforward and arguably more appropriate for OSBS than forcing a 4-aspect
  structure onto terrain that doesn't have meaningful aspect variation. The main tradeoff is losing aspect-dependent insolation, which is negligible at
  OSBS slopes. The benefit is double the elevation resolution (8 bins instead of 4), which matters more for capturing TAI dynamics — the whole point is
  resolving elevation-dependent water table behavior.

  That said — this is a modeling decision your PI should weigh in on. If there's any chance you'd want to compare your results to Swenson's 4x4 output or
  to the osbs2 baseline (which uses 4x4), keeping the same structure has value for apples-to-apples comparison. You could also generate both configurations
   and let the comparison speak for itself.

2. What is the expansion factor (line 95)? Is this directly from Swenson? Any need for us to change it for OSBS?
    - Only needed for matching the merit data to a gridcell boundary --> not used in our osbs pipeline

3. NEON provides precalculated OSBS slope/aspect values. Is it worth using them instead? What errors could we risk introducing? What might become more accurate in our pipeline as a result of using provided data? Presumably they have more robust answers and outlier handling for the specific site.

  NEON provides slope and aspect rasters as a separate data product (DP3.30025.001). The question is whether to use them instead of computing our own.

  What could improve:
  - NEON's processing pipeline is specifically calibrated for their LIDAR data
  - They likely have better handling of flat areas where slope is near zero and aspect becomes noisy — small gradient errors get amplified in aspect
  calculation when the gradient magnitude is tiny
  - Their QC pipeline may handle edge artifacts and nodata boundaries more robustly
  - For a low-relief site like OSBS, per-pixel slope quality matters because the values are so small that numerical noise is a larger fraction of the
  signal

  What could go wrong:
  - Grid alignment: The NEON slope/aspect rasters must be pixel-aligned with our DTM. If they're derived from the same DTM tiles, they should be, but this
  needs verification.
  - DEM conditioning inconsistency: Our pipeline fills pits and depressions before flow routing, which modifies the DEM surface. Slope/aspect computed from
   the original DEM (which is what both NEON and our current code do — the stage 8 bug fix explicitly uses the original DEM, not the conditioned one) would
   be consistent. But if we ever change to computing slope from the conditioned DEM, using NEON's values would create an inconsistency.
  - Subsampling: If we subsample the DEM for flow routing, we'd need to subsample or reaggregate the NEON slope/aspect identically. This is doable but adds
   a dependency.
  - Reproducibility: Introduces an external data dependency. If NEON reprocesses their LIDAR (new release, updated algorithms), our results change.

  What wouldn't change: The hillslope parameters use mean slope per elevation bin, averaged over thousands of pixels. At that level of aggregation,
  per-pixel differences between our gradient calculation and NEON's may wash out. The mean slope for a bin containing 100,000 pixels is robust to per-pixel
   noise.

### Stage 5/6

The bug fixes were implemented into the pipeline correctly, but the overall structure is not correct.
                                                                                                                                                         
  pysheds handles low-level hydrological grid operations: flow routing, accumulation, HAND, DTND, slope/aspect, channel masks. It takes a DEM and produces 
  per-pixel fields. That's where its responsibility ends.                                                                                                  
                                                                                                                                                           
  The width calculation (trapezoidal fitting, quadratic solver), aspect/elevation binning, and the 6-parameter computation are representative hillslope  
  analysis — higher-level statistical aggregation that operates on pysheds' output. In Swenson's codebase, this logic lives in representative_hillslope.py,
   which is a separate module from pysheds.

  We don't have that equivalent. Instead, the representative hillslope logic is duplicated as inline code in both stage3_hillslope_params.py and
  run_pipeline.py. That's where the maintenance risk is — not a missing pysheds fix, but missing our own shared module.

  The clean structure would be:
  ┌────────────────────────┬──────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┐
  │         Layer          │                        Responsibility                        │                        Where it lives                         │
  ├────────────────────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ pysheds (fork)         │ Flow routing, HAND, DTND, slope/aspect                       │ $PYSHEDS_FORK/pysheds/pgrid.py                                │
  ├────────────────────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ Hillslope analysis     │ Binning, trapezoidal fit, width calculation, 6-parameter     │ Should be a shared module (e.g., scripts/hillslope_params.py) │
  │ module                 │ computation                                                  │                                                               │
  ├────────────────────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ Pipeline scripts       │ Orchestration, I/O, plotting                                 │ scripts/osbs/run_pipeline.py,                                 │
  │                        │                                                              │ scripts/merit_validation/stage3_*.py                          │
  └────────────────────────┴──────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
  The middle layer is what's missing. The quadratic() function, the trapezoidal fitting, the HAND binning logic, and the width computation should be
  extracted into a shared module that both the MERIT validation and OSBS pipeline import. Right now they're copy-pasted.

### Stage 8
  Stage 8's purpose: It discovered that our custom np.gradient()-based slope/aspect calculation had a Y-axis sign inversion that systematically swapped    
  North and South aspects. The fix for the MERIT validation was to replace our custom code with pgrid's grid.slope_aspect() method, which uses the Horn  
  1981 8-neighbor stencil with correct coordinate conventions. This improved area fraction correlation from 0.64 to 0.82.                                  


  The OSBS pipeline did NOT get this fix. Look at the difference:

  - MERIT stage 3 (fixed): Uses grid.slope_aspect("dem") — pgrid's Horn 1981 method (line 800)
  - OSBS pipeline (not fixed): Uses np.gradient(dem_for_slope, pixel_size) — the same basic approach that stage 8 found to be buggy (line 1518)

  The OSBS pipeline has a comment on line 1514 saying "pysheds assumes geographic coords" as the reason for not using pgrid's method — which is true,
  pgrid's slope_aspect() uses haversine-based spacing and expects lat/lon, not UTM meters. So the OSBS pipeline couldn't use the pgrid fix directly.

  But the underlying bug — the Y-axis sign convention in np.gradient and how it maps to geographic north — may or may not be present in the OSBS code. Line
   1527 has aspect = np.degrees(np.arctan2(-dzdx, -dzdy)) with a comment about numpy's y-axis. Whether this is correct depends on how the UTM raster is
  oriented (northing increasing upward vs. row index increasing downward). It looks like an attempt to handle the convention, but it was never validated
  against a known-correct result the way the MERIT version was.

  Bottom line: The OSBS pipeline uses a different slope/aspect calculation than the fixed MERIT validation code, and that calculation has never been
  verified against the stage 8 findings. This needs to be tested — compare the OSBS np.gradient aspect classification against what pgrid would produce
  (after adapting pgrid for UTM, or by computing on a small geographic-coordinate version of the same area).

### Stage 9
Stage 9 tested whether varying the accumulation threshold could improve the area fraction correlation (which was stuck at ~0.82 after all other fixes).  
  It swept thresholds of 20, 34, 50, 100, and 200 cells on the MERIT tile and found:                                                                       
                                                                                                                                                           
  - Low sensitivity in the 20-50 cell range (correlation 0.80-0.83)
  - Rapid degradation above 100 cells (correlation collapses)                                                                                              
  - Conclusion: threshold isn't the cause of remaining discrepancy, proceed to OSBS                                                                      

  The stage 9 script itself is purely diagnostic — it didn't change any code. Its finding was "the methodology is validated, move on."

  What jumps out for OSBS: The threshold situation is fundamentally different and stage 9's conclusions don't transfer.

  For MERIT, the threshold was data-driven: FFT found Lc = 8.2 pixels = 763m, giving threshold = 0.5 * 8.2² = 34 cells. This worked because the FFT found a
   clean spectral peak.

  For OSBS, the FFT failed to find a meaningful peak at 1m resolution — it detected noise at 6m wavelength. So we imposed an arbitrary floor: min_lc_pixels
   = 100 (100m). The entire downstream chain depends on this forced value:
  ┌─────────────────┬──────────┬─────────────────────┬──────────────────────────┐
  │                 │  MERIT   │ OSBS (full, 4x sub) │ OSBS (interior, 4x sub)  │
  ├─────────────────┼──────────┼─────────────────────┼──────────────────────────┤
  │ Lc source       │ FFT peak │ Forced minimum      │ FFT peak (but at 4m res) │
  ├─────────────────┼──────────┼─────────────────────┼──────────────────────────┤
  │ Lc value        │ 763m     │ 100m                │ 166m                     │
  ├─────────────────┼──────────┼─────────────────────┼──────────────────────────┤
  │ Threshold       │ 34 cells │ 312 cells           │ 864 cells                │
  ├─────────────────┼──────────┼─────────────────────┼──────────────────────────┤
  │ Stream coverage │ 2.17%    │ 2.32%               │ 1.44%                    │
  └─────────────────┴──────────┴─────────────────────┴──────────────────────────┘
  The 100m Lc constraint is arbitrary. The 166m value from the interior tiles came from FFT on the 4x-subsampled grid — which we've already discussed is
  unreliable because subsampling can miss the real peak.

  So the real problem isn't stage 9's findings — it's that we don't have a trustworthy Lc for OSBS, and the accumulation threshold (which controls the
  entire stream network, HAND, DTND, and all downstream parameters) is built on either an arbitrary floor or an FFT run at degraded resolution. This ties
  directly back to the earlier discussion: running FFT at full 1m resolution on a large interior region is needed before we can trust any threshold value.
