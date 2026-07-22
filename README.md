# Data and code for the GBR recurrent disturbance study

This repository provides the processed data and analysis code needed to reproduce the manuscript analyses for recurrent heat exposure, baseline-dependent coral-loss signals, and composition-linked supplementary diagnostics on the Great Barrier Reef.

The deposited workflow begins with integrated reef-year ecological and environmental matrices. It does not rerun the upstream conversion of raw tropical-cyclone tracks into reef-level annual maximum wind speed; `max_wind_ms` is treated as an input variable in the analyses reported here.

The repository is for data and code availability. It does not include manuscript drafts, cover letters, reference PDFs, journal-preparation notes, or generated submission packages.

## Repository structure

- `data/`: processed ecological, disturbance, reef-coordinate, and composition inputs used by the reproducibility pipeline.
- `data/coral_species_data_all.csv`: merged benthic-composition input used by the composition-sensitivity workflow.
- `output/legacy_load_analysis_matrix.csv`: model-ready disturbance-response matrix used by the main recurrence models.
- `output/legacy_load_enriched_eco.csv`: retained processed snapshot of the ecological-covariate data; the current workflow reads the corresponding covariates from `output/legacy_load_analysis_matrix.csv`.
- `output/composition_event_analysis_matrix.csv`: composition-matched event matrix used by the composition analyses.
- `output/tables/`: machine-readable source tables for Supplementary Tables S1-S20 and composition diagnostics.
- `output/figures/`: generated figure output directory. Figure binaries are not versioned and are recreated by the scripts.
- `scripts/`: canonical data-processing, statistical-analysis, audit, and figure-generation scripts.
- `DATA_AVAILABILITY.md`: data provenance, included files, and public source-data links for excluded large assets.

## Environment

Use Python 3.10 or later. Install the required packages with:

```bash
pip install -r requirements.txt
```

## Reproduce the analysis

Run the full reproducibility workflow from the repository root:

```bash
python scripts/04_build_legacy_load_features.py
python scripts/05_run_lmm_legacy_model.py
python analysis/baseline_cover_threshold_validation/run_threshold_validation.py
python scripts/07_generate_map_figure.py
python scripts/06_generate_figures.py
python scripts/08_composition_sensitivity.py
python analysis/sector_hard_coral_supplement/run_sector_hard_coral_supplement.py
```

## Key reproducibility notes

- `scripts/04_build_legacy_load_features.py` reconstructs `output/legacy_load_analysis_matrix.csv`, including the ecological covariates used by the current sensitivity analyses.
- `scripts/05_run_lmm_legacy_model.py` reproduces the OLS, GEE, window, threshold, sector-stratified, episode-level, TOST, return-interval-encoding, and ecological-covariate sensitivity analyses, and exports Supplementary Tables S1-S9, S17, and S18.
- `scripts/07_generate_map_figure.py` reproduces the map panels for Figure 1 and Fig. S1. Non-CSV spatial boundary files are optional and are not included here; source links are listed in `DATA_AVAILABILITY.md`.
- `scripts/06_generate_figures.py` reproduces the core non-map figures and residual diagnostics from the recurrence analyses.
- `scripts/08_composition_sensitivity.py` reproduces the composition-matched event matrix, composition diagnostics, all-category scan, denominator sensitivity, category-to-loss sensitivity, and the machine-readable source tables for Supplementary Tables S10-S16. The script reads `data/coral_species_data_all.csv` when present, and otherwise falls back to per-reef CSVs under `data/coral_species_data/`.
- `analysis/baseline_cover_threshold_validation/run_threshold_validation.py` reproduces the baseline-cutoff and spline sensitivity outputs used in Supplementary Tables S19-S20 and Fig. S4.
- `analysis/sector_hard_coral_supplement/run_sector_hard_coral_supplement.py` reproduces Supplementary Text S5, including the four-sector and six-management-area hard-coral category scans, spatial interaction models, and baseline-cover >10% checks. If the NRM boundary shapefile is unavailable, the script uses the included reef-to-management-area mapping.
- `scripts/verify_contamination.py` and `scripts/verify_text_quality.py` are audit helpers used during manuscript checking.

## Supplementary-table numbering

The current scripts write the manuscript-facing S1-S20 numbering directly. Some tables have more than one machine-readable CSV when the Supplementary Information combines related panels or diagnostics under one table number (for example, S9, S11, and S14).

## Large or external assets

Large non-CSV spatial files, reference PDFs, and journal-preparation materials are excluded. Public source websites for ecological monitoring, climate products, cyclone records, and spatial boundaries are provided in `DATA_AVAILABILITY.md`. Generated figure binaries can be recreated from the deposited processed data and scripts.
