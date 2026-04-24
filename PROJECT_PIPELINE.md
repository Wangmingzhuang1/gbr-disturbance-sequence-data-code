# CEE Project

This repository now keeps a single canonical analysis pipeline for the GBR disturbance-sequence study.

## Main directories

- `scripts/`: analysis and modeling code
- `scripts/visualizations/`: manuscript figure scripts
- `data/`: input datasets
- `output/`: current generated tables and figures
- `manuscript/`: manuscript, supplement, and cover letter
- `docs/`: only retained reference note for event classification

## Canonical pipeline

Run the core analysis in this order:

1. `scripts/02_build_matrix.py`
2. `scripts/03_merge_eco_dist.py`
3. `scripts/05_analyze_succession.py`
4. `scripts/08_feature_engineering.py`
5. `scripts/06_run_gee_model_main.py`
6. `scripts/07_physical_cooling_proof.py`
7. `scripts/11_spatial_autocorrelation.py`
8. `scripts/13_bootstrap_ci.py`
9. `scripts/12_sensitivity_analysis.py`
10. `scripts/17_targeted_robustness.py`
11. `scripts/18_fate_divergence.py`
12. `scripts/14_auxiliary_ecology.py`
13. `scripts/15_auxiliary_juveniles.py`
14. `scripts/16_model_diagnostics.py`
15. `scripts/19_extended_sequence_inference.py`

Summary readout:

- `scripts/09_analysis_combined.py`

Figure generation:

- `scripts/visualizations/figure0_map.py`
- `scripts/visualizations/plot_fig1_hierarchy.py`
- `scripts/visualizations/plot_fig2_adjusted_means.py`
- `scripts/visualizations/plot_fig3_fate_divergence.py`
- `scripts/visualizations/plot_fig4_robustness.py`
- `scripts/visualizations/plot_fig5_supporting_evidence.py`
- `scripts/visualizations/plot_figS1_gap_memory.py`
- `scripts/visualizations/plot_figS2_juvenile_evidence.py`
- `scripts/visualizations/plot_figS3_ols_diagnostics.py`
- `scripts/visualizations/plot_figS4_baseline_state_gradients.py`

## Key outputs

- `output/data/topic_b_features.csv`
- `output/data/eco_response_master_matrix_merged.csv`
- `output/data/extracted_sequences.csv`
- `output/tables/regression_results_main.csv`
- `output/tables/adjusted_marginal_means.csv`
- `output/tables/bootstrap_hierarchy_ci.csv`
- `output/tables/psm_sequence_effect.csv`
- `output/tables/permutation_sequence_effect.csv`
- `output/tables/spatiotemporal_trend_summary.csv`
- `output/tables/gap_memory_main_results.csv`
- `output/tables/robustness_summary.csv`
- `output/tables/tukey_negative_tail_*.csv`
- `output/tables/fate_divergence_*.csv`
- `output/tables/fate_supplement_*.csv`
- `output/audits/pipeline_parameter_audit.csv`
- `output/audits/sequence_consistency_audit.csv`
- `output/audits/eco_duplicate_aggregation_audit.csv`
- `output/audits/negative_rel_loss_audit*.csv`
- `output/figures/fig*.png`

## Non-canonical utility scripts

These are diagnostic helpers, not part of the main pipeline:

- `audit_*.py`
- `check_*.py`
