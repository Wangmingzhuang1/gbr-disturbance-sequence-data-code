# Data and code for the GBR recurrent disturbance study

This repository provides the processed data and analysis code needed to reproduce the manuscript analyses for recurrent heat exposure, baseline-dependent coral-loss signals, and composition-linked supplementary diagnostics on the Great Barrier Reef.

The repository is for data and code availability. It does not include manuscript drafts, cover letters, reference PDFs, journal-preparation notes, or generated submission packages.

## Repository structure

- `data/`: processed ecological, disturbance, reef-coordinate, and composition inputs used by the reproducibility pipeline.
- `data/coral_species_data_all.csv`: merged benthic-composition input used by the composition-sensitivity workflow.
- `output/legacy_load_analysis_matrix.csv`: model-ready disturbance-response matrix used by the main recurrence models.
- `output/legacy_load_enriched_eco.csv`: ecological-covariate matrix used for cumulative-load sensitivity checks.
- `output/composition_event_analysis_matrix.csv`: composition-matched event matrix used by the composition analyses.
- `output/tables/`: machine-readable source tables for Supplementary Tables S1-S21 and composition diagnostics.
- `output/figures/`: generated figure output directory. Figure binaries are not versioned and are recreated by the scripts.
- `analysis/baseline_cover_threshold_validation/`: baseline-threshold and spline-heterogeneity analysis used for interpretation-zone checks and Fig. S6.
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
```

## Key reproducibility notes

- `scripts/04_build_legacy_load_features.py` reconstructs `output/legacy_load_analysis_matrix.csv` and `output/legacy_load_enriched_eco.csv`.
- `scripts/05_run_lmm_legacy_model.py` reproduces the OLS, GEE, window, threshold, sector-stratified, episode-level, TOST, and ecological-covariate sensitivity analyses, and exports Supplementary Tables S1-S10 plus the threshold and spatial-equivalence tables used in the final manuscript package.
- `analysis/baseline_cover_threshold_validation/run_threshold_validation.py` reproduces the baseline interpretation-zone, threshold scan, and nonlinear spline-heterogeneity outputs used by Fig. S6 and the baseline-limited interpretation checks.
- `scripts/07_generate_map_figure.py` reproduces the map panels for Figure 1 and Fig. S1. Non-CSV spatial boundary files are optional and are not included here; source links are listed in `DATA_AVAILABILITY.md`.
- `scripts/06_generate_figures.py` reproduces the core non-map figures and residual diagnostics from the recurrence analyses.
- `scripts/08_composition_sensitivity.py` reproduces the composition-matched event matrix, composition diagnostics, all-category scan, denominator sensitivity, category-to-loss sensitivity, and the machine-readable source tables for Supplementary Tables S11-S19. The script reads `data/coral_species_data_all.csv` when present, and otherwise falls back to per-reef CSVs under `data/coral_species_data/`.
- `scripts/verify_contamination.py` and `scripts/verify_text_quality.py` are audit helpers used during manuscript checking.

## Submitted supplementary-table mapping

The repository retains the original machine-readable source tables S1-S21. The submitted Supplementary Information consolidates them into S1-S18 without renaming or deleting analysis outputs.

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

## Large or external assets

Large non-CSV spatial files, reference PDFs, and journal-preparation materials are excluded. Public source websites for ecological monitoring, climate products, cyclone records, and spatial boundaries are provided in `DATA_AVAILABILITY.md`. Generated figure binaries can be recreated from the deposited processed data and scripts.
