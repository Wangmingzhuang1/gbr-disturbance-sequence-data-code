# Reproducibility pipeline

Run all commands from the repository root.

## 1. Build the disturbance-response matrix

```bash
python scripts/04_build_legacy_load_features.py
```

Inputs:

- `data/eco_response_master_matrix_merged.csv`
- `data/master_disturbance_matrix.csv`

Output:

- `output/legacy_load_analysis_matrix.csv`

## 2. Reproduce model results and supplementary tables

```bash
python scripts/05_run_lmm_legacy_model.py
```

Main outputs:

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

This step reproduces the reef-cluster robust OLS models, GEE validation, response-metric sensitivity, fixed-effect checks, episode-level analysis, VIF diagnostics, quantile regression boundary check and ecological-covariate sensitivity analysis.

## 3. Reproduce non-map figures

```bash
python scripts/06_generate_figures.py
```

Outputs:

- `output/figures/figure_02.*`
- `output/figures/figure_03.*`
- `output/figures/figure_04.*`
- `output/figures/figure_05.*`
- `output/figures/figure_06.*`

## 4. Reproduce map figures

```bash
python scripts/07_generate_map_figure.py
```

Outputs:

- `output/figures/figure_01.*`
- `output/figures/si_figure_01.*`

Optional spatial shapefiles are not included in this repository. Source links are provided in `DATA_AVAILABILITY.md`.

## 5. Optional audit helpers

```bash
python scripts/verify_contamination.py
python scripts/verify_text_quality.py
```

These scripts were used as manuscript-audit helpers and are not required for the main reproduction workflow.
