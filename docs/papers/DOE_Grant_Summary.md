# DOE Grant Summary: Water and Carbon Dynamics of Coastal Plain Wetlandscapes

> **Note:** This document summarizes the DOE grant proposal for background context.
> See the end of this document for the **Actual Project Scope** section describing
> what we are currently working on, which differs from the grant's aspirational goals.

## Overview

**Funding Agency:** DOE-BER (Biological and Environmental Research)
**Science Area:** Terrestrial-Aquatic Interfaces
**Project Duration:** 2023-2026 (based on timeline)

## Principal Investigators

| Name | Institution | Role |
|------|-------------|------|
| Dr. Matthew Cohen | University of Florida | PI, Project Lead |
| Dr. Stefan Gerber | University of Florida | Co-I, **ESM Modeling** |
| Dr. James Jawitz | University of Florida | Co-I, Probabilistic Modeling |
| Dr. Amanda Subalusky | University of Florida | Co-I, Gas Evasion |
| Dr. David Lewis | University of South Florida | Co-I, Soil Carbon |
| Dr. Daniel McLaughlin | Virginia Tech | Co-I, Hydrology |
| Dr. Nicholas Ward | PNNL | Co-I, Carbon Measurements |

---

## Core Scientific Problem

**Wetlandscapes** are low-relief mosaics of wetlands embedded in terrestrial uplands. These landscapes are:
- Hot-spots for carbon (C) cycling and storage
- Highly productive ecosystems
- Poorly represented in Earth System Models

The **Terrestrial-Aquatic Interface (TAI)** is the dynamic boundary between wet and dry areas that shifts with water level changes. Current ESMs treat wetlands as static patches, ignoring:
- Dynamic TAI movement
- Threshold behaviors in connectivity
- Hot spots and hot moments of C flux

---

## Research Questions

### Q1: Point Scale
How do local variations in water table and OC quality interactively control C stocks and vertical fluxes?

### Q2: Wetland Scale
How does basin morphology (size, depth, shape) and TAI movement impact wetland-scale C stocks, vertical fluxes, and lateral exports?

### Q3: Wetlandscape Scale
How does variation in wetland density, surface connectivity, and basin morphology impact aggregate C stocks and fluxes?

---

## Study Sites (North Florida)

### Bradford Experimental Forest (BEF)
- **Ecosystem:** Pine flatwoods with cypress wetlands
- **Connectivity:** High surface connectivity, flashy blackwater streams
- **Wetlands monitored:** 48
- **Streams monitored:** 10
- **Wetland coverage:** ~25%

### Ordway-Swisher Biological Station (OSBS)
- **Ecosystem:** Sandhills with wetland depressions
- **Connectivity:** Primarily groundwater, rarely surface connected
- **Wetlands monitored:** 19
- **Streams monitored:** 4
- **Wetland coverage:** ~19%

Both sites have:
- High-resolution LIDAR topography
- Continuous water level recorders
- Similar climate (R ~1450 mm/yr, PET ~1300 mm/yr)

---

## Research Elements

### Element 1: Topography and Hydrology
- LIDAR-based wetland characterization
- Stage-area-perimeter relationships
- Connectivity thresholds
- **Lead:** McLaughlin, Jawitz, Cohen

### Element 2: Primary Production
- Landsat 8 LAI time series
- Spatial patterns across hydrotopographic gradients
- **Lead:** Cohen

### Element 3: Soil Organic Carbon
- Spatially extensive SOC surveys (15 points per wetland)
- Laboratory incubation experiments
- Hydroperiod manipulation studies
- **Lead:** Lewis, Ward

### Element 4: Gas Evasion
- CO2 and CH4 flux measurements
- Synoptic chamber measurements
- In-situ pCO2/pCH4 sensors
- **Lead:** Subalusky, Cohen

### Element 5: Lateral C Export
- High-frequency stream DOC/DIC measurements
- 12 stream monitoring locations
- fDOM proxy for DOC
- **Lead:** Cohen, Subalusky, McLaughlin

### Element 6: Model-Experiment Synthesis
- ELM (E3SM Land Model) implementation
- Reduced complexity probabilistic models
- Scale-performance tradeoff analysis
- **Lead:** Gerber, Jawitz

---

## Modeling Approach

### Three Levels of Landscape Representation in ELM

1. **Homogeneous**: Spatially averaged landscape (current ESM approach)
   - No TAI representation
   - Single "point" per wetlandscape

2. **Binary**: Two 1-D columns (wetland + upland)
   - Static TAI
   - Following CLM_SPRUCE approach (Shi et al. 2015)

3. **Patchy**: Multiple columns with varying properties
   - Dynamic TAI
   - Lateral flow between columns
   - Most realistic but computationally expensive

### Reduced Complexity Models
- Probabilistic descriptions of water depth PDFs
- DEM-based depth censoring for inundation dynamics
- Landscape-scale TAI length and connectivity predictions
- Developed by Bertassello et al. (2018, 2019, 2020)

---

## Stefan Gerber's Responsibilities

From Table 2 and Element 6:

| Task | Role |
|------|------|
| Spatial Representations of TAI in Numerical Models | Lead |
| Landscape Stock and Flux Predictions from Numerical Models | Lead |
| Lateral Export Predictions from Numerical Models | Lead (with Cohen) |
| Model-model comparisons | Co-Lead (with Jawitz) |

**Key Activities:**
- Implement ELM simulations with varying spatial complexity
- Compare homogeneous vs binary vs patchy representations
- Evaluate model performance against field measurements
- Develop scaling rules for wetlandscape representation

---

## Key Variables and Fluxes

### Carbon Stocks
- SOC (Soil Organic Carbon) quantity and quality
- Potentially mineralizable fractions
- Leachable/soluble fractions

### Vertical Fluxes (to atmosphere)
- CO2 evasion
- CH4 evasion
- Diffusive vs ebullitive pathways

### Lateral Fluxes (to drainage network)
- DOC (Dissolved Organic Carbon)
- DIC (Dissolved Inorganic Carbon)
- Flow-mediated export

---

## Relevance to CTSM Work

This project directly informs CTSM/ELM development for:

1. **Wetland representation**: How to represent dynamic TAI in land models
2. **Sub-grid heterogeneity**: Binary vs patchy landscape decomposition
3. **Hillslope hydrology**: Lateral water and C fluxes between landscape positions
4. **Carbon cycling**: Links between hydrology and C stocks/fluxes

The hillslope hydrology feature in CTSM is conceptually similar to the "patchy" landscape representation proposed here - both involve:
- Multiple elevation positions within a gridcell
- Lateral water flow between positions
- Different water table depths affecting biogeochemistry

---

## Key References

- Bertassello et al. 2019: Stochastic dynamics of wetlandscapes
- Klammler et al. 2020: Local storage dynamics predict wetlandscape discharge
- McLaughlin et al. 2019: Wetland connectivity thresholds
- Shi et al. 2015: Representing peatland microtopography in CLM
- Ward et al. 2020b: Representing TAI in Earth System Models

---

## Actual Project Scope (January 2025)

The grant's goal of simulating full wetlandscapes is not currently realistic. **Individual wetlands are not properly represented in Earth System Models** - this is the fundamental problem we're addressing.

### Current Focus

**Goal:** Improve individual wetland representation in CTSM using hillslope hydrology

**Test Site:** OSBS (Ordway-Swisher Biological Station)
- NEON/Ameriflux site with extensive datasets
- 1m resolution LIDAR topography available
- Collaboration with field teams on the DOE project
- Representative of Florida coastal plain wetlandscapes

### Why Wetlands Aren't "Properly Represented"

Current ESMs have no sense that wetlands are **dynamically changing entities**:
- The TAI (Terrestrial-Aquatic Interface) drives bulk of carbon exchange
- TAI position is time/season dependent
- Models treat wetland extent as static

### Technical Approach

Using CTSM **hillslope hydrology** to represent within-gridcell topography:
- Multiple columns per gridcell at different elevations
- Lateral water flow between columns
- Different water table depths affect biogeochemistry
- Even subtle Florida topography matters for TAI dynamics

### Current Status

**Phase:** Technical development (getting model working, not accuracy)

**Reference Cases:**
- `$CASES/osbs2.branch.spillheight/` - Spillheight mechanism testing
- `$CASES/osbs2.branch.v2/` - Development branch
- `$CASES/osbs2.branch.v3/` - Development branch

**Hillslope Data:**
- Currently using Swenson global dataset (placeholder)
- Goal: Create custom hillslope parameters from 1m OSBS LIDAR
- Swenson 2025 paper provides methodology guide

**Input Data Strategy:**
- Current: Globally subset data via `run_tower` script
- Goal: Maximize use of NEON-provided local data for production runs

### Team Roles

**Dr. Stefan Gerber:** Scientific direction, ESM expertise
**Technical Support (cdevaneprugh):**
- CTSM maintenance and operation on HiPerGator
- Feature implementation
- Documentation
- Output analysis and visualization
- Graduate student support
