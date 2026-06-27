# HU8 03080103 Watershed (NWI source data)

NWI (National Wetlands Inventory) shapefile bundle for HUC8 watershed
**03080103 — Lower St. John**. Source of the NWI water mask used by the
OSBS production hillslope pipeline.

## Provenance

| Field | Value |
|---|---|
| Source | USFWS National Wetlands Inventory |
| HUC8 watershed | 03080103 (Lower St. John) |
| Imagery year | 2017 (true color, 1 m resolution) |
| Mapping project | "Lower St. John" |
| Database extract | October 2024 |
| `NWI_ID` prefix | `202409CSw{...}` (September 2024 DB version) |

Full vintage discussion (including how the project polygon was identified
and why other HU8-03080103 projects don't overlap the OSBS domain) is in
`swenson/docs/data-acquisition-dates.md` — "NWI Lake Mask (HU8_03080103)"
section.

## What's in this directory

```
HU8_03080103.gdb/          ESRI File Geodatabase (binary; primary source)
HU8_03080103_Wetlands.shp  Wetland polygons (shapefile copy)
HU8_03080103_Wetlands_Project_Metadata.shp  Per-project metadata polygons
*.shx / *.dbf / *.prj / *.xml  Standard shapefile sidecars
```

Wetland polygon attributes used in the pipeline:
- `ATTRIBUTE` — Cowardin code (e.g. `L1UBHh`, `PUBHh`)
- `WETLAND_TY` — Plain-English type
- `Shape_Leng`, `Shape_Area` — Geometry metadata

## How this data is used in this repo

- **`swenson/scripts/visualization/export_nwi_water_kml.py`** — Reads
  `HU8_03080103_Wetlands.shp`, filters to Lacustrine (`L*`) and Palustrine
  Unconsolidated Bottom (`PUB*`) prefixes, clips to the production domain
  (UTM 17N), and exports as KML for Google Earth.
- **`swenson/data/mosaics/production/water_mask.tif`** — Generated upstream
  from the same shapefile via the archived
  `audit/260512-cleanup/osbs/generate_water_mask.py` (rasterization onto
  the production DTM grid with Phase E.6 hole-fill). The pipeline reads
  this binary mask, NOT the shapefile directly.
- **NWI dual-mask logic in `scripts/osbs/run_pipeline.py`** — See module
  docstring + Step 1 / Step 3c / Step 4.

## Acquisition

USFWS NWI Wetlands Mapper / data download:
https://www.fws.gov/program/national-wetlands-inventory/data-download

Select HUC8 watershed 03080103 — Lower St. John, then download the
shapefile bundle.

## Size

~115 MB total (gitignored; the `data/.gitignore` excludes large data
files). Not committed to the repo. If you need to rebuild
`water_mask.tif` from scratch, you'll need to re-acquire this directory.
