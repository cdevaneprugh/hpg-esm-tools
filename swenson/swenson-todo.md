# Swenson Implementation TODO

Working task list for implementing representative hillslope methodology.

---

## Phase 1: Setup [COMPLETE]

- [x] Revert Swenson's pysheds fork pgrid.py
- [x] Create swenson/ directory structure
- [x] Create swenson/CLAUDE.md
- [x] Move SWENSON_IMPLEMENTATION.md to swenson/
- [x] Fork pysheds on GitHub (cdevaneprugh/pysheds)
- [x] Clone pysheds fork to $BLUE/pysheds_fork (branch: uf-hillslope)
- [x] Update environment.yml with DEM dependencies
- [x] Add PYSHEDS_FORK to ~/.bashrc
- [x] Verify baseline pysheds imports

---

## Phase 2: Recreate Swenson Results [COMPLETE]

### Methods Ported to `pysheds/hillslope.py`

- [x] `compute_hand_extended()` - Returns HAND + DTND + drainage_id
- [x] `compute_hillslope()` - Classify headwater/left/right bank/channel
- [x] `create_channel_mask()` - Channel/bank identification with segment IDs
- [x] `river_network_length_and_slope()` - Network geometry stats

### Implementation

- [x] Create `pysheds/hillslope.py` with HillslopeGrid subclass
- [x] Port methods from Swenson's pgrid.py
- [x] Add type hints and docstrings
- [x] Write unit tests (20 tests, all passing)

### Verification

- [x] Verify HAND is 0 on channel, positive off-channel
- [x] Verify hillslope classification covers all 4 categories
- [x] Verify river network statistics are reasonable
- [x] Verify HAND-DTND correlation is positive
- [x] Verify bank mask is roughly symmetric

### Test Data & Validation

- [x] Download MERIT DEM sample (n30w095 - Mississippi, free, no registration)
- [x] Download Swenson's published dataset from Zenodo
- [x] Run validation comparing our implementation to Swenson's
- [x] Validate HAND calculation (median within 10%)

#### Validation Results (2026-01-21)

Used **free MERIT sample data** (no registration required) to validate against
Swenson's published global dataset.

| Metric | Our Implementation | Swenson's Data | Ratio |
|--------|-------------------|----------------|-------|
| HAND median | 3.6 m | 4.0 m | 0.90 ✓ |
| HAND mean | 10.3 m | 6.7 m | 1.53 |

Differences in mean/max expected: we compare raw 90m pixels vs Swenson's
binned 16-element hillslope parameters at 0.9° resolution.

Scripts: `swenson/scripts/validate_against_swenson.py`
Results: `swenson/data/validation_comparison.png`

---

## Phase 3: OSBS Plan

- [ ] Check NEON LIDAR availability for OSBS
- [ ] Assess domain size and memory requirements
- [ ] Identify any pysheds modifications needed for high-res data
- [ ] Define processing pipeline
- [ ] Document parameter tuning for low-relief terrain

---

## Phase 4: Implement OSBS

- [ ] Download OSBS 1m LIDAR
- [ ] Apply processing pipeline
- [ ] Generate 16-column hillslope parameters
- [ ] Validate against global dataset
- [ ] Create CTSM-compatible surface dataset

---

## Verification Checklist

### Phase 1 Complete [x]
- [x] Swenson's fork pgrid.py reverted
- [x] swenson/ directory created with CLAUDE.md, todo, IMPLEMENTATION.md
- [x] Fresh pysheds fork cloned to $BLUE/pysheds_fork
- [x] environment.yml updated (clean version)
- [x] Baseline pysheds imports work

### Phase 2 Complete [x]
- [x] All 4 Swenson methods ported to our fork
- [x] 20 unit tests pass
- [x] Verification script confirms reasonable outputs
- [ ] MERIT DEM downloaded (pending registration)
- [ ] Test case on OSBS 90m data

### Phase 3 Complete
- [ ] OSBS data availability confirmed
- [ ] Processing pipeline defined
- [ ] Resource requirements estimated

### Phase 4 Complete
- [ ] OSBS hillslope dataset generated
- [ ] Output validated
- [ ] CTSM surface dataset created

---

## Git Commits (pysheds_fork)

| Hash | Description |
|------|-------------|
| 7f0d9e0 | Add hillslope.py with HillslopeGrid class |
| 7f55394 | Fix profile handling and add unit tests |
| 2ff1a68 | Add verification scripts |
