# Fan et al. 2019: Hillslope Hydrology in Global Change Research and Earth System Modeling

**Citation:** Fan, Y., Clark, M., Lawrence, D. M., Swenson, S., et al. (2019). Hillslope hydrology in global change research and Earth system modeling. *Water Resources Research*, 55, 1737-1772.

**DOI:** https://doi.org/10.1029/2018WR023903

**Note:** This is a major community synthesis paper (36 pages, 45 authors) bringing together hydrologists, Critical Zone scientists, and ESM developers. Lawrence and Swenson (NCAR) are co-authors - the same developers behind CTSM hillslope hydrology.

---

## The Core Problem

Earth System Models (ESMs) cannot explicitly resolve hillslope-scale terrain structures that fundamentally organize water, energy, and biogeochemical stores and fluxes at subgrid scales.

**Current ESM Land Model Limitations:**
- **1-D (vertical) hydrology** - no lateral flow
- **2-3m shallow soil** - misses deep moisture storage
- **Free-draining** - instant drainage to rivers
- **No terrain structure** - flat slab with no ridges/valleys
- **Static PFTs** - vegetation not linked to local hydrology

**Consequences of 1-D free-draining approach:**
- Fast drainage loss during wet periods
- Reduced terrestrial water storage in dry periods
- Premature shutdown of ET and streamflow in dry season
- Contrary to flux tower and streamflow observations

---

## Two Key Organizing Processes

The paper identifies **two terrain-induced hydrologic structures** as first-order controls:

### 1. Down-Valley Drainage (Lateral Flow)

Gravity-driven convergence from ridges to valleys creates:
- Drier hills, wetter valleys
- Deep water tables under hills, shallow under valleys
- Efficient drainage in uplands, impeded drainage in valleys (wetlands)
- **Spatial carryover**: upland surplus subsidizes lowland deficit
- **Temporal carryover**: slow subsurface flow delays delivery to dry season

**Where it matters:**
| Climate | Relief | Impact |
|---------|--------|--------|
| Seasonally dry | High | Valley forests survive dry season |
| Ever-wet | Low | Elevated mounds improve drainage in waterlogged soils |
| Seasonal flooding | Low | Hills support forest islands above wetlands |

### 2. Slope Aspect Difference (Sunny vs Shady)

Topographic relief creates variations in solar insolation:
- **Water-limited regions**: Shady slopes support larger plants, higher biomass
- **Energy-limited regions**: Sunny slopes support larger plants, longer growing season

**Where it matters:**
| Latitude | Relief | Impact |
|----------|--------|--------|
| High | Any | Energy-limited; sunny slopes more productive |
| Mid | High | Can be water-limited; shady slopes retain moisture |
| Low | Any | Minimal aspect effect due to high sun angle |

---

## Critical Knowledge Gap: The Subsurface

> "The greatest knowledge gap is the subsurface structure"

The paper emphasizes that understanding the depth structure of the Critical Zone (CZ) is essential but poorly constrained:

**Key Findings from CZ Observatories:**
1. Porous/permeable layer is **far thicker** than 2-3m assumed in ESMs
2. Weathered/fractured bedrock is hydrologically active (roots found at 16m depth)
3. CZ thickness may correlate with topography
4. Aspect affects soil/regolith depth (thicker on shady slopes in water-limited regions)

**Recommendation:** Instead of a definitive depth, model the **exponential decrease** in porosity/permeability with depth, with a characteristic e-folding scale.

---

## The HAND Concept

**HAND (Height Above Nearest Drainage)** emerges as a powerful way to divide ESM grid cells:

```
Traditional DEM: elevation referenced to sea level
HAND: elevation referenced to nearest stream channel
```

**Advantages:**
- Removes regional topography, retains hillslope-scale structure
- Creates non-overlapping drainage zones with linear relationships
- All stream pixels have HAND=0 regardless of absolute elevation
- Correlates with water table depth
- Strong predictor of plant community composition

**Implementation:**
1. Divide grid cell into HAND zones
2. Apply 1-D soil hydrology on each zone
3. Route surplus from high to low HAND zones
4. Only lowest zone interacts with streams

---

## Implementation Approaches

### Implicit (TOPMODEL-based)
- Use Topographic Wetness Index (TWI) to partition grid cell
- Instantaneous redistribution (assumes equilibrium)
- Missing: explicit lateral flow dynamics, delayed delivery

### Explicit (Representative Hillslopes)
- Separate hillslope and channel processes
- Multiple columns at different elevations
- Route water from high to low columns
- Only lowest column feeds streams

### Key Physical Principles

**Use Darcy's Law** (not kinematic wave):
- Kinematic wave uses constant terrain slope
- Darcy's law captures negative feedbacks:
  - High water table → accelerated drainage
  - Low water table → decelerated drainage
- Preserves deep soil water storage
- Captures delayed hill-to-valley transfer

**Allow two-way surface-groundwater exchange:**
- Rivers can gain (from higher water table)
- Rivers can lose (to sediments/fractures)
- Dynamic connectivity (channel network expands/contracts)

---

## Testable Hypotheses

### H1: CZ Structure Hypotheses (for CZ scientists)

**H1a.** CZ depth and porosity-permeability vary systematically from ridges to valleys and from sunny to shady slopes.

**H1b.** In water-limited regions, greater valley storage supports larger plants and higher productivity, especially during dry season.

**H1c.** In waterlogged regions, better-drained hills support larger plants and higher productivity.

**H1d.** In water-limited regions, shady slopes support larger plants due to lower ET demand and deeper soils.

**H1e.** In energy-limited regions, sunny slopes support larger plants due to longer growing season and thicker thawed/drained depth.

### H2: ESM Implementation Hypotheses (for modelers)

**H2a.** Linking PFTs with drainage position and aspect will simulate higher productivity less sensitive to stress.

**H2b.** Implementing ridge-to-valley groundwater convergence will lengthen water residence times.

**H2c.** Using Darcy's law and allowing two-way exchange will give ESMs longer hydrologic memory.

**H2d.** Extending model soil depth to include weathered bedrock will simulate larger water storage capacity.

**H2e.** Differentiating sunny/shady slopes will alter energy fluxes through albedo changes.

**H2f.** Accounting for lateral flow will allow mechanistic prediction of wetland location and dynamics.

---

## Relevance to OSBS Work

### Direct Applicability

This paper provides the **scientific justification** for using hillslope hydrology at OSBS:

1. **Low-relief wetlandscape**: OSBS fits position 3 in Figure 6a - "ever wet, low relief" where lateral drainage improves soil aeration on slightly elevated mounds

2. **TAI dynamics**: The paper explicitly discusses how "the slightly elevated hills can improve local drainage and alleviate waterlogging" - exactly the TAI transitions we're trying to capture

3. **Vegetation-hydrology coupling**: The paper emphasizes that PFTs should be linked to local hydrologic conditions, which vary systematically from uplands to lowlands

### Key Quotes Relevant to OSBS

> "In humid and low-relief regions where water is in excess, lateral drainage is also important but for different reasons. Here regional drainage is impeded, resulting in waterlogged soils and oxygen stress for plants."

> "A positive feedback reinforces the terrain-vegetation association; denser vegetation and higher ET on higher terrain further lowers the water table, improving drainage and nutrient conditions."

### Methodological Guidance

The paper recommends:
1. Use HAND to partition the landscape
2. Apply Darcy's law for lateral flow (not kinematic wave)
3. Include deep regolith in soil column
4. Allow dynamic surface-groundwater exchange

---

## Key Figures

| Figure | Content | Relevance |
|--------|---------|-----------|
| 2d-f | Vegetation patterns in waterlogged landscapes | OSBS analog |
| 6a | Climate-terrain matrix for drainage importance | Position 3 = OSBS |
| 8 | HAND concept and implementation | Direct methodology |
| 9 | Groundwater-surface water exchange modes | TAI dynamics |

---

## Relationship to Swenson 2025

This paper provides the **scientific foundation** that Swenson 2025 implements:

| Fan 2019 | Swenson 2025 |
|----------|--------------|
| Identifies need for hillslope representation | Creates global hillslope dataset |
| Recommends HAND-based partitioning | Uses HAND/DTND for elevation bins |
| Calls for aspect differentiation | Provides 4 aspect bins |
| Emphasizes lateral flow | Enables lateral flow between columns |

---

## Summary

This synthesis paper establishes that:

1. **Hillslope hydrology matters** for ESM predictions where water or energy is limiting
2. **Two key processes** - down-valley drainage and aspect difference - are first-order organizers
3. **The subsurface** is the critical knowledge gap
4. **HAND** provides an efficient way to partition landscapes
5. **Physical formulations** (Darcy's law, two-way exchange) are essential for capturing feedbacks
6. **Low-relief wetlands** (like OSBS) are places where subtle topography drives ecosystem structure

The paper provides both the scientific justification and implementation guidance for the hillslope hydrology features now available in CTSM.
