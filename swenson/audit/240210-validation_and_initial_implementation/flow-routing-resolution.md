# Flow Routing Resolution: Subsampling Problem and Path Forward

Reference document for addressing the current 4x subsampling compromise in the OSBS hillslope pipeline.

---

## Problem Statement

The OSBS hillslope pipeline currently subsamples the 1m LIDAR DEM by 4x (to 4m) before flow routing. This discards 93.75% of the elevation data before the most scientifically critical steps: stream network delineation, HAND, and DTND computation. The entire justification for using 1m LIDAR instead of 90m MERIT is to capture fine-scale drainage structure in a low-relief wetlandscape — subsampling undermines that purpose.

The FFT spatial scale analysis also ran on the subsampled grid, but this is a separate issue with a simple fix: run FFT at full resolution (numpy handles 17000x19000 arrays trivially).

---

## What We Know

### The Bottleneck: `resolve_flats`

pysheds' DEM conditioning pipeline has three steps: `fill_pits`, `fill_depressions`, `resolve_flats`. The third step is the bottleneck. It imposes subtle gradients on perfectly flat areas so D8 routing can assign flow directions. For large flat regions (common in low-relief terrain like OSBS), the algorithm has poor scaling — potentially O(n^2) or worse.

### Test History

| Job | Pixels | Memory | Result |
|-----|--------|--------|--------|
| 23793378 | ~189M (full 1m) | 64GB | OOM during resolve_flats |
| 23801217 | ~18M (4x sub) | 128GB | Completed but max_accum=1 (edge bug) |
| 23807179 | ~18M (4x sub) | 128GB | Success, 6 min runtime |
| — | ~189M (full 1m) | 128GB+ | **Never tested** |
| — | ~47M (2x sub) | 128GB+ | **Never tested** |

### What We Don't Know

1. Whether full-resolution processing is feasible with sufficient memory (256GB+)
2. Whether it's feasible but just slow (hours)
3. Whether the `resolve_flats` algorithm fundamentally cannot handle this many flat pixels regardless of resources
4. What the actual memory profile looks like — is it a gradual climb or a sudden spike?
5. Whether 2x subsampling (preserving 25% of data at 2m resolution) is a viable middle ground

---

## Scientific Impact of Subsampling

### What 4x subsampling loses

At 4m resolution vs 1m:
- **Stream network detail:** Small drainage features (ditches, shallow channels, wetland margins) below 4m width become invisible. These are exactly the TAI features we're trying to capture.
- **HAND precision:** Elevation differences are averaged over 4x4 pixel blocks. In terrain with only 46m total relief, small elevation differences matter. A 0.5m wetland depression rim could be smoothed away.
- **DTND accuracy:** Flow paths computed at 4m follow coarser routes than at 1m. Distance estimates are less precise.
- **Aspect resolution:** Slope and aspect computed from a 4m grid capture broader terrain orientation but miss fine-scale variability.

### What would be different at full resolution

- Finer stream network that captures actual drainage patterns visible in LIDAR
- More accurate HAND values, especially near stream channels where small elevation differences drive TAI dynamics
- Lc from FFT at 1m may differ from Lc at 4m, cascading to different accumulation thresholds
- More reliable aspect classification in flat areas where small gradients determine direction

---

## Testing Plan

### Phase 1: Characterize the Problem

**Goal:** Determine whether full-resolution processing is feasible at all.

**Test A: Memory profiling at full resolution**
```bash
#SBATCH --mem=256gb
#SBATCH --time=24:00:00
#SBATCH --qos=gerber-b
```
Run the existing pipeline at full 1m resolution (subsample=1) on the interior mosaic with 256GB allocation. Add memory profiling (e.g., `tracemalloc` or `/proc/self/status` polling) at each DEM conditioning step. Three possible outcomes:
1. **Completes:** Problem solved. Note runtime and peak memory.
2. **OOM at 256GB:** The algorithm needs more memory than is practical. Proceed to optimization.
3. **Runs indefinitely:** The algorithm converges too slowly. Proceed to optimization.

**Test B: 2x subsampling as middle ground**
```bash
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
```
Run with subsample=2 (2m resolution, ~47M pixels). This preserves 4x more data than current 4x subsampling while keeping the problem size much smaller than full resolution. If this works easily, it may be an acceptable interim solution.

**Test C: Isolate resolve_flats**
Write a minimal test script that ONLY runs the three conditioning steps (fill_pits, fill_depressions, resolve_flats) on the full-res DEM. Skip everything else. This isolates the bottleneck and lets us measure its resource consumption independently. Time each step individually.

### Phase 2: Optimize if Needed

If full resolution fails, investigate these approaches in order:

#### Option 1: Alternative flat resolution algorithms

pysheds uses a specific algorithm for resolve_flats. Other implementations exist:
- **RICHDEM** library has multiple flat resolution algorithms with different scaling properties
- **WhiteboxTools** has optimized depression filling that handles large DEMs
- **TauDEM** is designed for large parallel DEM processing

Test whether any of these can handle the OSBS DEM at full resolution. If so, we could:
- Use the alternative tool ONLY for DEM conditioning (fill pits/depressions/flats)
- Export the conditioned DEM
- Feed it back to pysheds for flow direction, accumulation, HAND, DTND (which scale fine)

This avoids rewriting pysheds while bypassing its bottleneck.

#### Option 2: Tile-based processing

Process the DEM in overlapping tiles, each small enough for pysheds to handle:
1. Divide DEM into tiles with generous overlap (e.g., 2000-pixel overlap)
2. Condition each tile independently
3. Compute flow direction and accumulation per tile
4. Merge results, resolving overlaps
5. Compute HAND/DTND on the merged flow direction grid

**Complications:** Flow routing at tile boundaries is non-trivial. Flow must be consistent across tiles, which requires careful handling of the overlap zones. This is solvable but adds complexity.

#### Option 3: Selective resolution

Not all areas need 1m resolution equally:
- **Near streams and wetland margins:** 1m resolution matters most (TAI dynamics)
- **Ridge tops and uplands:** Coarser resolution is fine (terrain is uniform)

Could process at variable resolution: full 1m near drainage features, coarser elsewhere. This is complex to implement but scientifically justified.

#### Option 4: Pre-condition the DEM externally

Use GDAL or WhiteboxTools to fill depressions before passing to pysheds:
```bash
# WhiteboxTools depression filling (handles large DEMs efficiently)
whitebox_tools -r=FillDepressions -i=input.tif -o=filled.tif --fix_flats

# Then feed to pysheds, skipping its fill_pits/fill_depressions/resolve_flats
```
pysheds could then skip conditioning entirely and go straight to flow direction computation on the pre-conditioned DEM.

#### Option 5: Fix pysheds resolve_flats

If the issue is specifically in pgrid.py's flat resolution code, we could:
- Profile the exact function to find the scaling bottleneck
- Port an O(n) or O(n log n) flat resolution algorithm (e.g., Barnes et al. 2014)
- This is the most work but fixes the problem at the source

### Phase 3: Validate Resolution Impact

Once we can process at higher resolution, run the full pipeline at 1m, 2m, and 4m on the same region and compare:

| Metric | Compare across resolutions |
|--------|---------------------------|
| Lc (from FFT) | Should converge as resolution increases |
| Stream network density | More streams at finer resolution |
| HAND distribution | Finer HAND bins at higher resolution |
| Hillslope parameters | The 6 geomorphic params — how much do they change? |
| Runtime and memory | Cost-benefit |

If the hillslope parameters are nearly identical at 2m and 1m, then 2m is sufficient and we're over-engineering. If they differ meaningfully, 1m matters and we need to solve the bottleneck properly.

---

## Key Decision Points

1. **After Test A:** If full resolution works with enough memory/time, just use it. Done.
2. **After Test B:** If 2x works and full doesn't, compare 2m vs 4m hillslope params. If substantially better, use 2x.
3. **After Phase 2:** If pysheds can't handle it, choose the simplest alternative (likely Option 4: external pre-conditioning).
4. **After Phase 3:** Resolution comparison gives the scientific answer to "does it matter?"

---

## References

- Barnes, R., Lehman, C., & Mulla, D. (2014). An efficient assignment of drainage direction over flat surfaces in raster digital elevation models. Computers & Geosciences, 62, 128-135. (O(n) flat resolution algorithm)
- WhiteboxTools: https://www.whiteboxgeo.com/
- RichDEM: https://github.com/r-barnes/richdem
- TauDEM: https://hydrology.usu.edu/taudem/
