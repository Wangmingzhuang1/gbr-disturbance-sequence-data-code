# Data availability and provenance

This repository contains the CSV inputs, derived matrices, analysis scripts and machine-readable result tables needed to reproduce the manuscript analyses.

## Included directly

The following CSV files are included because they are compact and required for reproducibility:

- `data/eco_response_master_matrix_merged.csv`
- `data/master_disturbance_matrix.csv`
- `data/sites_lon_lat.csv`
- `output/legacy_load_analysis_matrix.csv`
- `output/legacy_load_enriched_eco.csv`
- `output/tables/table_s1_variable_definitions.csv`
- `output/tables/table_s2_sample_composition.csv`
- `output/tables/table_s3_recurrence_model_ols_gee.csv`
- `output/tables/table_s4_sensitivity_models.csv`
- `output/tables/table_s5_window_sensitivity.csv`
- `output/tables/table_s6_baseline_loss_subsets.csv`
- `output/tables/table_s7_upper_quantile_boundary_check.csv`
- `output/tables/table_s8_reviewer_risk_sensitivity.csv`
- `output/tables/table_s9_ecological_sensitivity.csv`
- `output/tables/table_s10_vif_and_sample_diagnostics.csv`

## Public source data

Raw ecological monitoring data:

- Australian Institute of Marine Science Long-Term Monitoring Program: https://apps.aims.gov.au/reef-monitoring/reefs
- AIMS metadata portal: http://apps.aims.gov.au/metadata/

Thermal-stress data:

- NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov/

Tropical cyclone and wind data:

- Australian Bureau of Meteorology Tropical Cyclone Database: http://www.bom.gov.au/cyclone/history/database/
- Australian Bureau of Meteorology cyclone history portal: http://www.bom.gov.au/cyclone/history/index.shtml

Spatial boundary and reef feature data used for map rendering:

- Great Barrier Reef Marine Park Authority spatial data: https://www.gbrmpa.gov.au/
- Australian Government spatial and regional boundary data portals, as applicable for NRM and GBR boundary layers.

## Excluded files

The repository excludes large or non-essential binary assets, including:

- shapefile bundles under `data/Great_Barrier_Reef_Features/`
- shapefile bundles under `data/NRM_Terrestrial_and_Marine_Regions_GBR_GDA20/`
- generated PDF/JPG figure files
- reference PDFs
- manuscript drafts, cover letters and submission-management notes

The map script can run without the optional shapefiles; when those files are absent, it renders the reef points and base geography using the included CSV coordinates and public map features.
