# Swenson & Lawrence 2025: Global Representative Hillslope Data Set

**Citation:** Swenson, S. C., & Lawrence, D. M. (2025). Development of a global representative hillslope data set for use in Earth system models. *Journal of Advances in Modeling Earth Systems*, 17, e2024MS004410.

**DOI:** https://doi.org/10.1029/2024MS004410

---

## Summary

This paper describes the methodology for creating the global hillslope dataset currently used by CTSM. It provides a **blueprint for creating custom hillslope parameters from high-resolution LIDAR data** - exactly what we need for OSBS.

---

## The Six Geomorphic Parameters

Each hillslope element is defined by six parameters:

| Parameter | Symbol | Description | Source |
|-----------|--------|-------------|--------|
| Area | A | Horizontally projected surface area | DEM grid |
| Height | h | Mean height above stream channel | HAND |
| Distance | d | Mean distance from channel | DTND |
| Width | w | Width at downslope interface | Plan form model |
| Slope | α | Mean topographic slope | DEM gradient |
| Aspect | β | Azimuthal orientation (from North) | DEM gradient |

---

## Hillslope Structure in CLM/CTSM

```
Gridcell
├── North-facing hillslope
│   ├── Element 1 (lowest, near stream)
│   ├── Element 2
│   ├── Element 3
│   └── Element 4 (highest, ridge)
├── East-facing hillslope
│   └── (4 elements)
├── South-facing hillslope
│   └── (4 elements)
└── West-facing hillslope
    └── (4 elements)
```

**Total: 16 elements per gridcell, 96 parameters**

---

## Methodology Overview

### Step 1: Identify Characteristic Length Scale

Use **spectral analysis** of the Laplacian of elevation:
1. Apply Laplacian operator to DEM (highlights ridges/valleys)
2. Compute 2D Fourier transform
3. Find wavelength with maximum amplitude
4. This is the "characteristic length scale" (Lc)

**Why this matters:** The length scale determines how finely to resolve the stream network. A uniform scale doesn't work globally - mountains need larger scales than flatlands.

### Step 2: Delineate Catchments

Using **pysheds** toolkit:
1. Set accumulation threshold: `A_thresh = 0.5 * Lc²`
2. Identify stream network (pixels above threshold)
3. Delineate catchments draining to each stream reach
4. Classify pixels as left bank, right bank, or headwater

### Step 3: Calculate HAND and DTND

- **HAND** (Height Above Nearest Drainage): Elevation difference from nearest stream pixel
- **DTND** (Distance To Nearest Drainage): Horizontal distance to nearest stream pixel

These define the hillslope profile (height vs distance).

### Step 4: Discretize into Bins

**Aspect bins** (4):
| Bin | Range |
|-----|-------|
| North | ≥315° or <45° |
| East | ≥45°, <135° |
| South | ≥135°, <225° |
| West | ≥225°, <315° |

**Elevation bins** (4):
- Chosen so each bin has approximately equal area
- Constraint: Lowest bin upper bound ≤ 2m (for stream channel interaction)

### Step 5: Fit Plan Form Model

Use **trapezoidal shape** to estimate width vs distance:

```
w(d) = w_base + 2αd
```

Where:
- `w_base` = width at stream channel
- `α` = plan form divergence (+ = convergent, - = divergent)
- `d` = distance from channel

Fit by noting: `w(d) = -∂A_sum(d)/∂d`

### Step 6: Calculate Element Parameters

For each aspect-elevation bin combination:
- Average slope, aspect, area over all pixels in bin
- Calculate height from mean HAND
- Calculate distance from mean DTND
- Calculate width from trapezoidal model

---

## Key Equations

**Accumulation threshold from length scale:**
```
A_thresh = (1/2) * Lc²
```

**Slope from DEM:**
```
α = √[(∂z/∂x)² + (∂z/∂y)²]
```

**Aspect from DEM:**
```
β = -arctan(∂z/∂x / ∂z/∂y)
```

**Width from accumulated area:**
```
w(d) = -∂A_sum(d)/∂d
```

**Trapezoidal area:**
```
A(L) = αL² + w_base * L
```

---

## Tools and Data

### Software
| Tool | Purpose | URL |
|------|---------|-----|
| pysheds | Catchment delineation, HAND, DTND | https://github.com/mdbartos/pysheds |
| NumPy FFT | Spectral analysis | Built-in |
| Representative_Hillslopes | Full processing pipeline | https://github.com/swensosc/Representative_Hillslopes |

### Input Data
- **MERIT DEM**: ~90m resolution global DEM
- URL: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM

### Output Data
- Global hillslope parameters at ~1° resolution
- https://doi.org/10.5065/w01j-y441

---

## Relevance to OSBS Work

### Why Custom Data is Needed

The paper explicitly notes:
> "The MERIT DEM... may not be fine enough to capture topographic variations in areas of **very low topographic relief, such as wetlands**."

OSBS is exactly this case - a low-relief wetlandscape where:
- Global 90m data misses subtle topography
- 1m NEON LIDAR can capture actual hillslope structure
- TAI dynamics depend on fine-scale elevation differences

### Applying to OSBS

**Advantages of 1m LIDAR:**
- 90x finer resolution than global dataset
- Can capture wetland basin morphology
- Can identify actual stream/drainage networks
- Can resolve TAI transition zones

**Workflow for OSBS:**
1. Obtain 1m LIDAR DEM from NEON
2. Apply Laplacian spectral analysis to find Lc
3. Use pysheds for catchment delineation
4. Calculate HAND/DTND at high resolution
5. Discretize and average to create hillslope parameters
6. Format for CTSM surface dataset

### Expected Differences from Global Data

For OSBS, expect:
- **Smaller Lc**: Finer-scale drainage patterns
- **Lower HAND values**: Subtle elevation differences (meters, not hundreds of meters)
- **Different aspect distribution**: May not have 4 distinct hillslopes if relatively flat
- **More accurate TAI representation**: Can capture wetland-upland transitions

---

## Implementation Notes

### CLM Hillslope Processes

Three additional processes enabled by hillslope configuration:

1. **Lateral subsurface flow**: Water moves between columns based on hydraulic gradient
2. **Aspect-dependent insolation**: Solar radiation varies by slope/aspect
3. **Elevation downscaling**: Temperature and precipitation adjusted by elevation

### Height Constraint for Lowest Element

Important for wetland representation:
> "The upper bound of the lowest bin must be 2 m or less"

This ensures the soil column extends below stream channel elevation, allowing:
- Two-way water exchange between stream and soil
- Stream channel losses (water infiltrating from stream to groundwater)
- Realistic water table dynamics near streams

---

## Figures of Interest

- **Figure 1**: Hillslope processes (lateral flow, insolation, downscaling)
- **Figure 2**: Column geomorphic parameters definition
- **Figure 3**: Spectral analysis example (high vs low relief)
- **Figure 5-6**: HAND and DTND fields for contrasting terrain
- **Figure 9**: Flowchart of methodology (key reference)
- **Figure 10-11**: Representative hillslope plan view and profile examples

---

## Next Steps for OSBS

1. [ ] Clone Representative_Hillslopes repository
2. [ ] Obtain NEON 1m LIDAR for OSBS
3. [ ] Test pysheds on OSBS domain
4. [ ] Compare spectral length scale to global value
5. [ ] Generate custom hillslope parameters
6. [ ] Create CTSM-compatible surface dataset
7. [ ] Run comparison simulations (global vs custom hillslope data)
