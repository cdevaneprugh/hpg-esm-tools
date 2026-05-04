# Data Acquisition Dates

Date stamps and seasonal context for the three datasets that drive the OSBS
hillslope pipeline. Compiled 2026-04-30 from NEON metadata, the NWI shapefile
attribute table, and the Lee 2023 / McLaughlin 2019 papers. Used to interpret
basin-depth measurements and to flag where comparisons between datasets are
sensitive to vintage.

## NEON OSBS LIDAR (DP3.30024.001) — pipeline DTM source

**Pipeline data: OSBS 2023-05 collection, RELEASE-2026.** Confirmed in
`data/neon/README.md`. Used for `data/neon/dtm/`, `data/neon/slope/`, and
`data/neon/aspect/`.

All NEON OSBS DTM collections to date (per
`data.neonscience.org/api/v0/products/DP3.30024.001`):

| Year | Month | North-FL season | Used in our pipeline |
|------|-------|-----------------|----------------------|
| 2014 | May | late dry | — |
| 2016 | September | peak wet | — |
| 2017 | September | peak wet | — |
| 2018 | September | peak wet | — |
| 2019 | April | dry | — |
| 2021 | September | peak wet | — |
| **2023** | **May** | **late dry / transition** | **yes** |
| 2025 | May | (provisional release) | — |

NEON DP3.30024.001 is delivered as a bare-earth product — vegetation returns
have been filtered out by NEON's processing chain. No further canopy filter
applied in our pipeline.

**Seasonal interpretation.** May in north Florida is the end of the dry
season; water tables typically reach their seasonal minimum just before the
wet-season storms begin. The 2023-05 timing is favorable for capturing dry
beds in shallow depressions and on the rims of larger wetlands. Persistent
wetlands and cypress domes that hold water year-round will still show LIDAR
returns from the water surface, not the bed.

## NWI Lake Mask (HU8_03080103)

**Source imagery for OSBS production domain: 2017, true color, 1 m resolution.
"Lower St. John" mapping project.**

Determined by spatial intersection of
`HU8_03080103_Wetlands_Project_Metadata.shp` against the production domain
bounding box. Only one project polygon intersects:

| Field | Value |
|-------|-------|
| `PROJECT_NA` | Lower St. John |
| `IMAGE_YR` | 2017 |
| `IMAGE_DATE` | xx/17 (month not encoded) |
| `EMULSION` | TC (true color) |
| `IMAGE_SCAL` | 1 m |
| `SOURCE_TYP` | TC |
| `STATUS` | Digital |

The other project polygons in the metadata file (Eastport 1983, Mayport 1983,
Marietta 1983, Dinsmore 1983, Maxville 1983, Starke 1984, plus a residual
12 × 35 m Hawthorne 1984 fragment) sit in the northern coastal portion of the
HUC8 watershed near Jacksonville and do not overlap our domain. The 2015
"Atlantic CBRA" entry covers coastal barrier resource units only.

**Per-polygon attributes.** The wetland features themselves carry only
`ATTRIBUTE`, `WETLAND_TY`, `ACRES`, `NWI_ID`, `Shape_Leng`, `Shape_Area`. The
`NWI_ID` field has the form `202409CSw{<UUID>}_01`; the `202409` prefix is
the September 2024 database record version, not a per-polygon source-imagery
date. No finer-grain provenance is encoded per polygon.

**Database extract date: October 2024** (from `.shp.xml` `linkage`:
`Oct2024/Watershed_Extracts/...`).

**Seasonal interpretation.** Source imagery month is not encoded in
`IMAGE_DATE`. 2017 was a major hurricane year for Florida (Irma, September
2017); NWI source flights are typically scheduled before or well after
hurricane season, so the photography is most likely pre-Irma, but the
metadata does not confirm this. The 1 m TC resolution is much finer than the
1980s 1:58,000 CIR mapping that was the prior NWI baseline for this region.

The polygons align well with current Google Earth imagery (visual check), so
the dataset reflects approximately current OSBS wetland boundaries within
~9 years of the writing date.

## Lee 2023 OSBS LIDAR — ambiguous, follow-up needed

Lee, Epstein, Cohen 2023 (DOI: 10.1029/2023WR034553) cites "National Center
for Airborne Laser Mapping (NCALM)" for all four study sites' LIDAR (Section
2 / Figure 1 caption: *"LIDAR digital elevation models (DEMs; National Center
for Airborne Laser Mapping)"*). The paper states the DEMs were "obtained
during anomalously dry periods to minimize interference from ponded water,"
but does not give per-site dates.

**Three candidate datasets for the OSBS subset:**

| Candidate | Date | Season at flight time | Match to "anomalously dry"? |
|-----------|------|----------------------|------------------------------|
| NCALM Optech Gemini, OSBS Pathfinder | 2010-09-01 | peak wet | weak (2010 had moderate La Niña conditions but September is normally wet) |
| USGS 3DEP Florida Peninsular Putnam | late 2018–2019 | designed dry-conditions flight (per project spec) | strong |
| Custom NCALM mission (post-2010) | unknown | unknown | unknown |

McLaughlin 2019 (the methodology paper Lee cites for h_crit derivation) only
documents the BICY LIDAR explicitly: *"LiDAR data were collected during the
regional dry season (late May)"* (Section 2.2). McLaughlin did not study OSBS,
so OSBS-specific provenance is not in that paper either.

**Resolution path.** UF connection — Cohen (M. J. Cohen, senior author on Lee
2023) is at the same institution as the user, and per user note the
collaborating scientists on this project are reachable. **Direct
conversation is the fastest resolution.** Ask:

1. What LIDAR dataset and acquisition date was used for the OSBS subset of
   Lee 2023?
2. If multiple datasets, which one drove the h_crit measurements reported in
   Table 1 (mean OSBS spill depth 264.1 ± 95.0 cm)?

If a quick conversation isn't available, fall-back is to check the Wiley
supplement for Lee 2023 directly (the main fetch returned 403 to web tools).

## Seasonal context — north-central Florida (Putnam County)

From NWS Melbourne climate data, UF/IFAS rainfall sources, and SWFWMD
hydrologic conditions:

- **Dry season:** November–April. ~2–3 inches/month rainfall.
- **Wet season:** May–October. Daily afternoon thunderstorms; 5–8+
  inches/month at peak. Hurricane risk June–November.
- **Peak inundation:** late summer to early fall (August–September).
- **Peak drawdown:** late spring (April–May).
- **May:** transition month, typically still dry, before sustained wet-season
  storms begin.

## What our pipeline does, and doesn't do, with this provenance

**Currently accounted for:**
- NEON DTM bare-earth processing (canopy stripped by NEON; no additional
  vegetation filter applied in our pipeline).
- NWI water polygons rasterized to a binary mask; pixels marked as water are
  excluded from HAND statistics via the dual-mask approach
  (`run_pipeline.py:921` and surrounding logic).
- The `flooded_arr[water_mask > 0] -= 0.1` operation
  (`run_pipeline.py:831`) lowers water pixels by 10 cm purely for D8 routing
  direction, not as a physical correction.

**Not yet accounted for:**
- No physical correction for residual ponded water at NEON 2023-05 flight
  time. The May timing makes this a smaller concern than wet-season flights
  would, but persistent wetlands (cypress domes, large flatwoods depressions)
  may still hold water in May. Pixels in those will have LIDAR returns from
  the water surface, contaminating `depression_fill_depth` interpretations
  for those locations.
- No cross-check between NEON 2023 DTM vintage and NWI 2017 mask vintage. Six
  years of potential boundary drift; assumed minimal but not measured.
- No reconciliation of our 3.33 m basin-spill-depth measurement
  (lake-column-ctsm-audit.md Section 6.7.3) against Lee's 2.64 m once Lee's
  vintage is known. If Lee used the 2010 wet-season NCALM, their measurement
  may underestimate true depths in inundated wetlands; if Lee used the 2018
  USGS dry-season data, the comparison is direct.

## Action items

- [ ] Ask Cohen (or whoever has Lee 2023 internals) what LIDAR dataset and
      acquisition date the OSBS subset uses. Capture the answer in this doc
      and update `docs/lake-column-ctsm-audit.md` Section 6.7.3 with the
      vintage-aware comparison.
- [ ] If Lee 2023's OSBS LIDAR turns out to be the 2010 wet-season NCALM,
      flag the comparability caveat in the audit and consider whether our
      3.33 m number should be the primary reference rather than Lee's 2.64 m.
- [ ] Optional: pull the May 2023 north Florida rainfall record (NWS
      Gainesville or SWFWMD) to confirm 2023-05 was a typical or anomalously
      dry month, strengthening the favorable-timing claim for our DTM.

## References

- NEON DP3.30024.001 product page:
  https://data.neonscience.org/data-products/DP3.30024.001
- NEON OSBS field site:
  https://www.neonscience.org/field-sites/osbs
- USFWS NWI Wetlands Mapper / data download:
  https://www.fws.gov/program/national-wetlands-inventory/data-download
- Lee, E., Epstein, J. M., & Cohen, M. J. (2023). Patterns of Wetland
  Hydrologic Connectivity Across Coastal-Plain Wetlandscapes. *Water
  Resources Research*, 59(8), e2023WR034553.
  https://doi.org/10.1029/2023WR034553
- McLaughlin, D. L., Diamond, J. S., Quintero, C., Heffernan, J., & Cohen, M.
  J. (2019). Wetland Connectivity Thresholds and Flow Dynamics from Stage
  Measurements. *Water Resources Research*, 55(7), 6018–6032.
  https://doi.org/10.1029/2018WR024652
- USGS 3DEP FL Peninsular Putnam 2018 dataset:
  https://portal.opentopography.org/usgsDataset?dsid=FL_Peninsular_Putnam_2018
- OpenTopography NCALM 2003-2012 archive:
  https://opentopography.org/news/opentopography-completes-ingestion-all-ncalm-lidar-data-collected-between-2003-2012
- NWS Melbourne — Onset of the Wet and Dry Seasons in East Central Florida:
  https://www.weather.gov/media/mlb/climate/wetdryseason.pdf
- UF/IFAS — Florida Rainfall Data Sources and Types:
  https://edis.ifas.ufl.edu/publication/AE517
