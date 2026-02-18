# Pysheds Test Suite Audit

Phase A added UTM CRS support to the pysheds fork. This document audits the 4 test files and 3 synthetic DEMs that validate the changes.

**Bottom line: the tests are sound.** No tautological tests, no missing assertions on critical paths, no testing-the-framework-instead-of-the-code. Every CRS-branching location in pgrid.py is exercised with analytically known expected values.

---

## Synthetic DEMs

Three purpose-built DEMs, each designed to catch a specific class of bug.

### V-Valley (`generate_synthetic_dem.py`)

Two planar hillslopes meeting at a central N-S channel. Every pixel's elevation is a closed-form function of its row and column: `elev = base + downstream_slope * row_dist + cross_slope * |col - channel|`. This means slope, aspect, HAND, and DTND all have exact analytical values. If any CRS math is wrong, values deviate systematically.

**Limitation:** Every pixel's geographically nearest stream IS its hydrologically nearest stream. An EDT-based DTND bug would pass undetected on this DEM alone.

### Split Valley (`generate_split_valley.py`)

Two V-valleys side by side. Channel B sits 3.01m higher than A, pushing the drainage divide ~10 columns past the geometric midpoint. This creates a "divergence zone" where pixels are geographically closer to B but hydrologically drain to A. The only DEM geometry that can distinguish flow-path DTND from EDT DTND.

### Depression Basin (`generate_depression_basin.py`)

A tilted plane with a parabolic bowl. Requires the full DEM conditioning chain (fill_depressions -> resolve_flats -> flowdir) that the other two DEMs skip entirely. Tests whether pysheds correctly fills depressions and resolves flats before routing.

### Design Choices

- **pixel_size = 5m** (not 1m) across all three DEMs. Prevents silently passing if the code treats UTM coordinates as unit-spaced instead of reading the actual pixel dimensions.
- **UTM EPSG:32617** matches the OSBS NEON data that motivated the fix.
- Each generator returns an `expectations` dict with analytically derived values, keeping the test assertions independent of the generator implementation.

---

## test_utm.py (22 tests)

The primary CRS validation file. Uses the V-valley to check every UTM code path against closed-form solutions.

### TestCRSDetection (2 tests)

| Test | What it checks |
|------|---------------|
| `test_utm_is_not_geographic` | `_crs_is_geographic()` returns False for EPSG:32617 |
| `test_geographic_dem_is_geographic` | `_crs_is_geographic()` returns True for the existing geographic DEM |

Gate tests. If CRS detection fails, every downstream if/else branch takes the wrong path.

### TestSlope (1 test)

| Test | What it checks |
|------|---------------|
| `test_slope_magnitude` | Interior pixel slopes match `sqrt(cross_slope^2 + downstream_slope^2)` within 0.001 m/m |

Validates `_gradient_horn_1981()` uses correct pixel spacing on UTM data. Excludes a 3-pixel band around the channel where the Horn stencil bridges the V-valley gradient discontinuity.

### TestAspect (2 tests)

| Test | What it checks |
|------|---------------|
| `test_aspect_west_side` | West-side pixels face east (~92 deg) within 0.5 deg |
| `test_aspect_east_side` | East-side pixels face west (~268 deg) within 0.5 deg |

Catches the N/S aspect swap bug that was the original motivation for the Phase A work. The expected values come from `arctan2(east_downhill, north_downhill)` -- if the sign convention on `dzdy` is wrong, east and west aspects reflect across the E-W axis.

### TestHAND (2 tests)

| Test | What it checks |
|------|---------------|
| `test_hand_values` | HAND matches `cross_slope * |col - channel| * pixel_size` row by row |
| `test_hand_non_negative` | No negative HAND values anywhere |

Pipeline validation. HAND is a pure elevation difference (no CRS distance math), so these confirm that D8 routing and the hndx drainage-pixel mapping work correctly on UTM grids.

### TestDTND (3 tests)

| Test | What it checks |
|------|---------------|
| `test_dtnd_values` | DTND matches `|col - channel| * pixel_size` row by row |
| `test_dtnd_channel_is_zero` | Channel pixels have DTND = 0 |
| `test_dtnd_not_haversine_garbage` | Max DTND < 2x expected maximum (catches haversine-on-meters producing millions-of-meters garbage) |

The core CRS tests. Haversine interpreting UTM meters as degrees would produce ~6,371,000m per degree instead of ~500m across the domain. `test_dtnd_values` validates exact distances; `test_dtnd_not_haversine_garbage` is the smoke detector that catches gross CRS failures even if other tests are somehow masked.

### TestAZND (3 tests)

| Test | What it checks |
|------|---------------|
| `test_aznd_west_side` | West-side pixels have azimuth ~90 deg (drainage is due east) |
| `test_aznd_east_side` | East-side pixels have azimuth ~270 deg (drainage is due west) |
| `test_aznd_channel_points_downstream` | Channel pixels have azimuth ~180 deg (self-referencing hndx, IEEE signed-zero) |

AZND uses the same distance formula as DTND but in a different code path (arctan2 of dx, dy). Tests that the azimuth computation also got the UTM treatment.

### TestRiverNetworkLengthSlope (3 tests)

| Test | What it checks |
|------|---------------|
| `test_channel_length` | Total reach length ~995m (199 segments * 5m) |
| `test_channel_slope` | Mean reach slope ~0.001 m/m |
| `test_length_not_haversine_garbage` | Length < 2km (haversine would give ~110,000 km) |

`river_network_length_and_slope()` has its own CRS distance computation. These confirm it reads pixel spacing correctly for UTM. The garbage-detector test is the most important -- 110,000 km vs 995m is an unmistakable failure.

### TestHandDtndRelationship (1 test)

| Test | What it checks |
|------|---------------|
| `test_hand_equals_cross_slope_times_dtnd` | HAND = cross_slope * DTND on all interior hillslope pixels |

Cross-validation. HAND and DTND use different code paths (elevation lookup vs distance computation) but reference the same hndx drainage-pixel mapping. If either is wrong, this ratio breaks. This is the strongest single test because it constrains both outputs simultaneously.

### TestHillslopeClassification (3 tests)

| Test | What it checks |
|------|---------------|
| `test_hillslope_types_present` | All 4 types present: headwater (1), right bank (2), left bank (3), channel (4) |
| `test_bank_separation` | West and east sides have consistent but different bank types |
| `test_channel_column_is_type_4` | Channel column pixels are classified as type 4 |

`compute_hillslope()` is purely topological (no CRS math), but exercising it on UTM data provides coverage. `test_bank_separation` is the key one -- it checks that the V-valley's clean left/right geometry survives the classification pipeline.

### TestEndToEndUTM (1 test)

| Test | What it checks |
|------|---------------|
| `test_full_pipeline` | Fresh Grid through `from_raster -> flowdir -> accumulation -> create_channel_mask -> compute_hand -> slope_aspect -> compute_hillslope`, validating key outputs at each stage |

Integration test. Individual tests share fixtures (module-scoped), so a state-passing bug between stages could be hidden by fixture reuse. This runs the full chain on a fresh Grid to catch that.

### TestGeographicRegression (2 tests)

| Test | What it checks |
|------|---------------|
| `test_slope_aspect_runs` | `slope_aspect()` still produces valid results on the existing geographic DEM |
| `test_compute_hand_runs` | `compute_hand()` still works on geographic data with non-negative HAND |

Regression guard. The UTM code changes added if/else branches in `slope_aspect()` and `compute_hand()`. These confirm the geographic branch still works.

---

## test_split_valley.py (16 tests)

Tests the one scenario the V-valley can't: flow-path distance != Euclidean distance.

### TestFlowRouting (4 tests)

| Test | What it checks |
|------|---------------|
| `test_flow_directions_are_cardinal` | Interior pixels route E or W (cross_slope >> downstream_slope) |
| `test_divide_location` | Drainage divide at the analytically predicted column, checked at 3 rows |
| `test_channel_a_has_high_accumulation` | Channel A outlet accumulation >= 90% of basin A area |
| `test_channel_b_has_high_accumulation` | Channel B outlet accumulation >= 90% of basin B area |

Foundation tests. If D8 routing gets the divide wrong, every downstream test fails for the wrong reason. `test_divide_location` is the sharpest -- it checks the exact column where flow direction flips from W to E, derived from the analytical intersection of the two valley surfaces.

### TestHAND (5 tests)

| Test | What it checks |
|------|---------------|
| `test_hand_non_negative` | No negative HAND values |
| `test_hand_channel_zero` | Both channels have HAND = 0 |
| `test_hand_basin_a` | Basin A interior HAND matches `cross_slope * |col - 200| * pixel_size` |
| `test_hand_basin_b` | Basin B interior HAND matches `cross_slope * |col - 700| * pixel_size` |
| `test_hand_divergence_zone` | Divergence-zone HAND follows channel A distance, not channel B |

`test_hand_divergence_zone` is the important one. At the divide column, HAND should reflect the distance to channel A (the hydrological target), not channel B (the geometric nearest). Checked at 5 rows to rule out single-row coincidence.

### TestDTND (6 tests)

| Test | What it checks |
|------|---------------|
| `test_dtnd_non_negative` | No negative DTND values |
| `test_dtnd_channel_zero` | Both channels have DTND = 0 |
| `test_dtnd_basin_a` | Basin A interior DTND = `|col - 200| * pixel_size` |
| `test_dtnd_basin_b` | Basin B interior DTND = `|col - 700| * pixel_size` |
| `test_dtnd_divergence_zone` | Divergence-zone DTND matches flow-path distance to A, not EDT distance to B |
| `test_dtnd_not_haversine_garbage` | Max DTND bounded by domain width |

`test_dtnd_divergence_zone` is the single most important test in the entire suite. It checks 3 sample columns where flow-path DTND and EDT DTND differ by hundreds of meters, asserting that the actual value matches flow-path and is farther from EDT. This is the test that would have caught the original `distance_transform_edt` bug in the OSBS pipeline.

### TestHandDtndRelationship (1 test)

| Test | What it checks |
|------|---------------|
| `test_hand_equals_cross_slope_times_dtnd` | HAND = cross_slope * DTND in both basins' interior pixels |

Same cross-validation as in test_utm.py, now applied to a two-basin geometry. Confirms the hndx mapping is consistent across both basins.

---

## test_depression_basin.py (17 tests)

Exercises the DEM conditioning chain that the other two DEMs skip.

### TestFillDepressions (4 tests)

| Test | What it checks |
|------|---------------|
| `test_fill_raises_to_spill_elevation` | All depression pixels below spill are raised to spill level |
| `test_fill_does_not_lower` | fill_depressions never lowers any pixel |
| `test_fill_preserves_outside` | Pixels outside the depression are unchanged |
| `test_filled_region_is_flat` | Filled region is uniformly at spill elevation |

Tests the fill_depressions algorithm's fundamental contracts: raise pits to spill level, don't touch anything else, don't lower anything.

### TestResolveFlats (2 tests)

| Test | What it checks |
|------|---------------|
| `test_no_interior_flats_remain` | No sampled pixel has all 8 neighbors at equal elevation after resolve_flats |
| `test_inflated_does_not_lower_filled` | resolve_flats only adds micro-gradients upward |

After filling creates a flat, Garbrecht-Martz must add micro-gradients so D8 routing can resolve flow across it. `test_no_interior_flats_remain` samples every 10th pixel to verify.

### TestFlowDirection (3 tests)

| Test | What it checks |
|------|---------------|
| `test_interior_pixels_have_valid_direction` | Every interior pixel has a valid D8 direction code |
| `test_background_flows_south` | Pixels far from the depression flow south (background tilt) |
| `test_no_pits_remain` | No interior pixel is a local minimum in the inflated DEM |

`test_no_pits_remain` is the strongest -- it vectorizes across the entire grid, checking that every interior pixel has at least one neighbor at equal or lower elevation. This is the definition of "conditioned."

### TestAccumulation (2 tests)

| Test | What it checks |
|------|---------------|
| `test_depression_outflow_accumulation` | South-edge max accumulation exceeds NROWS + depression area |
| `test_max_accumulation_at_south_edge` | Maximum accumulation is in the bottom row (all flow reaches the outlet) |

Verifies that the depression's ~2800 pixels route through the spill point and reach the south edge. If conditioning fails, flow gets trapped in the depression and accumulation is too low.

### TestHAND (3 tests)

| Test | What it checks |
|------|---------------|
| `test_hand_bounded_below` | HAND >= -(DEP_DEPTH + 0.1), since depression pixels have original elevation below their drainage point |
| `test_hand_channel_zero` | Channel pixels have HAND = 0 |
| `test_hand_bounded` | Max HAND within domain elevation range |

Note that `test_hand_bounded_below` correctly accounts for negative HAND inside the depression. `compute_hand()` uses the unfilled DEM for elevation, so depression-center pixels are legitimately below their drainage target. This is a subtlety that shows the test author understood the algorithm.

### TestDTND (3 tests)

| Test | What it checks |
|------|---------------|
| `test_dtnd_non_negative` | No negative DTND values |
| `test_dtnd_channel_zero` | Channel pixels have DTND = 0 |
| `test_dtnd_bounded` | Max DTND within domain diagonal |

Bound-checking after the conditioning chain. Less precise than the V-valley tests (no closed-form DTND solution for routed-through-depression flow), but confirms the pipeline doesn't produce garbage.

---

## test_hillslope.py (16 tests)

Smoke tests for Swenson's hillslope methods on the existing geographic DEM.

### TestSlopeAspect (4 tests)

| Test | What it checks |
|------|---------------|
| `test_slope_aspect_runs` | Method completes, attributes exist |
| `test_slope_aspect_output_shape` | Output arrays match DEM shape |
| `test_slope_non_negative` | Slope >= 0 everywhere |
| `test_aspect_range` | Aspect in [0, 360] everywhere |

### TestChannelMask (3 tests)

| Test | What it checks |
|------|---------------|
| `test_create_channel_mask_runs` | Method completes, 3 attributes exist |
| `test_channel_mask_is_binary` | channel_mask contains only 0 and 1 |
| `test_bank_mask_values` | bank_mask contains only -1, 0, +1 |

### TestComputeHand (2 tests)

| Test | What it checks |
|------|---------------|
| `test_compute_hand_runs` | Method completes, hand attribute exists |
| `test_hand_non_negative` | HAND >= 0 everywhere |

### TestComputeHillslope (2 tests)

| Test | What it checks |
|------|---------------|
| `test_compute_hillslope_runs` | Method completes, hillslope attribute exists |
| `test_hillslope_valid_values` | Values in {0, 1, 2, 3, 4}, at least some classified |

### TestExtractProfiles (2 tests)

| Test | What it checks |
|------|---------------|
| `test_extract_profiles_runs` | Returns (list, dict) tuple |
| `test_profiles_have_structure` | At least one profile exists, each is an ndarray |

### TestRiverNetworkLengthSlope (1 test)

| Test | What it checks |
|------|---------------|
| `test_river_network_length_slope_runs` | Returns dict with 'length' and 'slope' keys |

### TestIntegration (1 test)

| Test | What it checks |
|------|---------------|
| `test_full_workflow` | Chains slope_aspect -> channel_mask -> compute_hand -> compute_hillslope, checks attributes exist and channel type 4 is present |

### Honest Assessment

These are smoke-level tests only. They check "does it run" and "right shape/range" but cannot validate correctness because the geographic DEM has no analytical truth to compare against. This is by design: the original pre-Phase-A test file existed to catch regressions when porting Swenson's methods into pysheds, not to validate CRS math. The synthetic DEMs in the other 3 files handle correctness validation.

---

## Coverage Map

Which test file catches which bug class:

| Bug class | test_utm | test_split_valley | test_depression_basin | test_hillslope |
|-----------|:--------:|:-----------------:|:---------------------:|:--------------:|
| Haversine on UTM meters (DTND) | X | X | | |
| Haversine on UTM meters (AZND) | X | | | |
| Haversine on UTM meters (river length) | X | | | |
| Horn gradient wrong pixel spacing | X | | | |
| N/S aspect swap | X | | | |
| EDT used instead of flow-path DTND | | X | | |
| fill_depressions correctness | | | X | |
| resolve_flats correctness | | | X | |
| D8 routing on conditioned DEM | | | X | |
| Geographic CRS regression | X | | | X |
| Method API contracts (runs, shapes) | | | | X |

## What Is Not Tested

- **Performance/scaling.** No tests for large grids, memory usage, or resolve_flats scaling (the known bottleneck from STATUS.md problem #2). These are out of scope for unit tests.
- **Multi-band or nodata-heavy DEMs.** All synthetic DEMs are clean single-band with no nodata regions. Edge handling with nodata is tested implicitly by test_hillslope.py's geographic DEM.
- **Correctness on real geographic data.** test_hillslope.py validates that the geographic code path runs and produces reasonable ranges, but doesn't compare against known-correct values. The MERIT validation pipeline (stages 1-9, >0.95 correlation on 5/6 parameters) serves that role outside the test suite.
