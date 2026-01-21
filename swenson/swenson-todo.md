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

### Test Data

- [ ] Download OSBS region at 90m (MERIT DEM) - **REQUIRES REGISTRATION**
- [ ] Run test case with ported methods on OSBS
- [ ] Compare characteristic length scale to global value

#### MERIT DEM Access

Registration required at: https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/

Tile needed: **n25w085** (covers N25-N30, W85-W80, includes OSBS)

Package: `dem_tif_n30w090.tar` contains Florida tiles

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
