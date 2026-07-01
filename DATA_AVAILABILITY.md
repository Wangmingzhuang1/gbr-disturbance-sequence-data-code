# Data availability and provenance

This repository contains the processed CSV inputs, derived analysis matrices, analysis scripts and machine-readable result tables needed to reproduce the manuscript analyses.

## Included directly

The following files are included because they are compact and required for reproducibility:

- `data/eco_response_master_matrix_merged.csv`
- `data/master_disturbance_matrix.csv`
- `data/sites_lon_lat.csv`
- `data/coral_species_data_all.csv`
- `output/legacy_load_analysis_matrix.csv`
- `output/legacy_load_enriched_eco.csv`
- `output/composition_event_analysis_matrix.csv`
- `output/tables/composition_reef_mapping.csv`
- `output/tables/composition_hard_coral_validation.csv`
- `output/tables/table_s1_variable_definitions.csv`
- `output/tables/table_s2_sample_composition.csv`
- `output/tables/table_s3_recurrence_model_ols_gee.csv`
- `output/tables/table_s4_sensitivity_models.csv`
- `output/tables/table_s5_window_sensitivity.csv`
- `output/tables/table_s6_baseline_loss_subsets.csv`
- `output/tables/table_s7_upper_quantile_boundary_check.csv`
- `output/tables/table_s8_response_metric_dependence_sensitivity.csv`
- `output/tables/table_s9_ecological_sensitivity.csv`
- `output/tables/table_s10_vif_and_sample_diagnostics.csv`
- `output/tables/table_s11_composition_data_diagnostics.csv`
- `output/tables/table_s12_composition_grouped_summary.csv`
- `output/tables/table_s13_composition_reorganization_models.csv`
- `output/tables/table_s14_composition_response_models.csv`
- `output/tables/table_s15_composition_denominator_sensitivity.csv`
- `output/tables/table_s16_all_category_reorganization_models.csv`
- `output/tables/table_s17_category_to_loss_sensitivity.csv`
- `output/tables/table_s18_category_availability.csv`
- `output/tables/table_s19_storm_category_reorganization_models.csv`
- `output/tables/table_s20_dhw_threshold_sensitivity.csv`
- `output/tables/table_s21_spatial_and_retention_equivalence.csv`
- `analysis/baseline_cover_threshold_validation/results/*.csv`

The `analysis/baseline_cover_threshold_validation` directory also includes the
script and text outputs needed to reproduce the baseline interpretation-zone,
threshold-sensitivity, and nonlinear spline-heterogeneity diagnostics.

## Submitted supplementary-table mapping

The repository retains source tables S1-S21 as separate machine-readable outputs. The submitted Supplementary Information uses the following consolidated S1-S18 mapping:

| Submitted table | Machine-readable source table(s) |
|---|---|
| S1-S8 | source S1-S8 |
| S9 | source S9 + S10 |
| S10 | source S11 |
| S11 | source S12 + S13 |
| S12 | source S14 |
| S13 | source S15 |
| S14 | source S16 + S18 |
| S15 | source S17 |
| S16 | source S19 |
| S17 | source S20 |
| S18 | source S21 |

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
- Australian Government spatial and regional boundary data portals for NRM and GBR boundary layers.

## Excluded files

The repository excludes large or non-essential binary assets, including:

- shapefile bundles under `data/Great_Barrier_Reef_Features/`
- shapefile bundles under `data/NRM_Terrestrial_and_Marine_Regions_GBR_GDA20/`
- per-reef benthic-composition CSV bundles under `data/coral_species_data/` when the merged file `data/coral_species_data_all.csv` is deposited instead
- generated PDF/JPG figure files
- reference PDFs
- manuscript drafts, cover letters and submission-management notes

The map script can run without the optional shapefiles; when those files are absent, it renders the reef points and base geography using the included CSV coordinates and public map features. The composition workflow can run from the merged file `data/coral_species_data_all.csv` without needing the original per-reef directory layout. Generated figure binaries can be reproduced from the deposited processed data and scripts.
