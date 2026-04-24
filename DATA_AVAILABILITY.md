# Data availability

This repository is the data-and-code availability package for the GBR disturbance-sequence study.

## Included directly

All CSV data used by the analysis are included in this repository:

- raw and lookup CSV files under `data/`
- daily thermal-stress CSV files under `data/daily_climate_full/`
- ecological raw CSV files under `data/reef_raw/`
- derived analysis matrices under `output/data/`
- statistical result tables under `output/tables/`
- audit and consistency-check tables under `output/audits/`

The included derived outputs are sufficient to inspect the exact values reported in the manuscript. The included raw and lookup CSV files support rerunning the analysis pipeline.

## External source links retained for provenance

Raw ecological monitoring data source:

- Australian Institute of Marine Science Long-Term Monitoring Program: https://apps.aims.gov.au/reef-monitoring/reefs
- AIMS metadata portal: http://apps.aims.gov.au/metadata/

Thermal-stress data source:

- NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov/

Tropical cyclone data source:

- Australian Bureau of Meteorology Tropical Cyclone Database: http://www.bom.gov.au/cyclone/history/database/
- Cyclone history portal: http://www.bom.gov.au/cyclone/history/index.shtml

Spatial boundary data for map rendering:

- Great Barrier Reef Marine Park Authority: https://www.gbrmpa.gov.au/

## Not included

The repository excludes non-CSV spatial boundary files and other large binary/map assets, including shapefiles under `data/Great_Barrier_Reef_Features/` and `data/NRM_Terrestrial_and_Marine_Regions_GBR_GDA20/`. These files are only needed to regenerate the study-area map and can be obtained from the public spatial-data providers above.

Manuscript drafts, cover letters, and journal-preparation notes are intentionally excluded because this repository is only for data and code availability.

## Repository URL

Repository link:

- GitHub repository: https://github.com/Wangmingzhuang1/gbr-disturbance-sequence-data-code
