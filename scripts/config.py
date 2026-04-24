# -*- coding: utf-8 -*-
"""
Centralized configuration for the finalized disturbance-sequence pipeline.
"""

import os

# Sequence extraction
BASELINE_WINDOW = range(2, 6)
NADIR_WINDOW = range(0, 4)
MIN_BASELINE_SURVEYS = 2
MIN_BASELINE_HC = 5.0
LOOKAHEAD_WINDOW = 4
RECOVERY_LOOKAHEAD = 4
EXTRACTION_MODE = 'strict'
RARE_CLASS_MIN_N = 5
YEAR_START = 1985
YEAR_END = 2027
DOWNSTREAM_EXCLUDED_SEQS = ('Concurrent', 'H_to_S')

# Statistics
BOOTSTRAP_REPS = 2000
PERMUTATION_REPS = 5000
RANDOM_SEED = 42
IMPACT_THRESHOLD = 5.0

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
OUTPUT_TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
OUTPUT_AUDITS_DIR = os.path.join(OUTPUT_DIR, 'audits')
OUTPUT_FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

for path in [OUTPUT_DIR, OUTPUT_DATA_DIR, OUTPUT_TABLES_DIR, OUTPUT_AUDITS_DIR, OUTPUT_FIGURES_DIR]:
    os.makedirs(path, exist_ok=True)


def _data_path(filename):
    return os.path.join(OUTPUT_DATA_DIR, filename)


def _table_path(filename):
    return os.path.join(OUTPUT_TABLES_DIR, filename)


def _audit_path(filename):
    return os.path.join(OUTPUT_AUDITS_DIR, filename)


def _figure_path(filename):
    return os.path.join(OUTPUT_FIGURES_DIR, filename)


MASTER_MATRIX_PHYSICAL_PATH = _data_path('eco_response_master_matrix.csv')
MASTER_MATRIX_PATH = _data_path('eco_response_master_matrix_merged.csv')
EXTRACTED_SEQS_PATH = _data_path('extracted_sequences.csv')
REEF_YEAR_SEQUENCE_SUMMARY_PATH = _data_path('reef_year_sequence_summary.csv')
FINAL_FEATURES_PATH = _data_path('topic_b_features.csv')
FATE_SUPPLEMENT_SEQS_PATH = _data_path('fate_supplement_sequences.csv')
MAIN_RESIDUALS_PATH = _data_path('main_model_residuals.csv')

ATTRITION_AUDIT_PATH = _audit_path('attrition_audit_stages.csv')
ATTRITION_BY_REEF_PATH = _audit_path('attrition_audit_by_reef.csv')
MODEL_ATTRITION_PATH = _audit_path('model_attrition_audit.csv')
SPATIAL_DIAGNOSTICS_PATH = _audit_path('spatial_residual_diagnostics.csv')
NEG_REL_LOSS_AUDIT_PATH = _audit_path('negative_rel_loss_audit.csv')
NEG_REL_LOSS_BASELINE_BINS_PATH = _audit_path('negative_rel_loss_audit_baseline_bins.csv')
REEF_MERGE_AUDIT_PATH = _audit_path('reef_merge_audit.csv')
STORM_IDENTIFICATION_AUDIT_PATH = _audit_path('storm_identification_audit.csv')
SITE_COORDINATE_AUDIT_PATH = _audit_path('site_coordinate_audit.csv')
REEF_YEAR_SEQUENCE_CONFLICTS_PATH = _audit_path('reef_year_sequence_conflicts.csv')
ECO_DUPLICATE_AGG_AUDIT_PATH = _audit_path('eco_duplicate_aggregation_audit.csv')
ECO_YEAR_ALIGNMENT_AUDIT_PATH = _audit_path('eco_year_alignment_audit.csv')
SEQUENCE_CONSISTENCY_AUDIT_PATH = _audit_path('sequence_consistency_audit.csv')
PIPELINE_PARAMETER_AUDIT_PATH = _audit_path('pipeline_parameter_audit.csv')

REGIME_SHIFT_PATH = _table_path('regime_shift_analysis.csv')
DESCRIPTIVE_STATS_PATH = _table_path('topic_b_descriptive_stats.csv')
FATE_SUMMARY_PATH = _table_path('fate_divergence_summary.csv')
FATE_MODEL_PATH = _table_path('fate_divergence_recovery_model.csv')
FATE_STRATIFIED_PATH = _table_path('fate_divergence_stratified.csv')
FATE_SUPPLEMENT_SUMMARY_PATH = _table_path('fate_supplement_summary.csv')
FATE_SUPPLEMENT_MODEL_PATH = _table_path('fate_supplement_recovery_model.csv')
FATE_SUPPLEMENT_STRATIFIED_PATH = _table_path('fate_supplement_stratified.csv')

REGRESSION_MAIN_PATH = _table_path('regression_results_main.csv')
REGRESSION_ECO_PATH = _table_path('regression_results_ecological.csv')  # rel_loss model + COTS covariate
REGRESSION_BIO_PATH = _table_path('regression_results_biological.csv')  # rel_loss model + COTS/herbivore covariates
REGRESSION_RARE_PATH = _table_path('regression_results_rare_class.csv')
ADJUSTED_MEANS_PATH = _table_path('adjusted_marginal_means.csv')
BOOTSTRAP_HIERARCHY_PATH = _table_path('bootstrap_hierarchy_ci.csv')
PERMUTATION_ASYMMETRY_PATH = _table_path('permutation_asymmetry.csv')
PSM_SEQUENCE_EFFECT_PATH = _table_path('psm_sequence_effect.csv')
PSM_BALANCE_TABLE_PATH = _table_path('psm_balance_table.csv')
PSM_MATCHED_PAIRS_PATH = _table_path('psm_matched_pairs.csv')
PERMUTATION_SEQUENCE_EFFECT_PATH = _table_path('permutation_sequence_effect.csv')
PERMUTATION_SEQUENCE_DIST_PATH = _table_path('permutation_sequence_effect_distribution.csv')
SPATIOTEMPORAL_TREND_SUMMARY_PATH = _table_path('spatiotemporal_trend_summary.csv')
SPATIOTEMPORAL_COUNTS_PATH = _table_path('spatiotemporal_sequence_counts.csv')
GAP_MEMORY_MAIN_RESULTS_PATH = _table_path('gap_memory_main_results.csv')
GAP_MEMORY_CURVES_PATH = _table_path('gap_memory_curve_points.csv')
ROBUSTNESS_SUMMARY_PATH = _table_path('robustness_summary.csv')
ECO_AUX_COVERAGE_PATH = _table_path('auxiliary_signal_coverage.csv')
ECO_AUX_SUMMARY_PATH = _table_path('auxiliary_signal_summary.csv')
ECO_AUX_CORR_PATH = _table_path('auxiliary_signal_correlations.csv')
ECO_AUX_ALGAE_MODEL_PATH = _table_path('regression_results_algae_aux.csv')
JUV_AUX_COVERAGE_PATH = _table_path('juvenile_signal_coverage.csv')
JUV_AUX_SUMMARY_PATH = _table_path('juvenile_signal_summary.csv')
JUV_AUX_CORR_PATH = _table_path('juvenile_signal_correlations.csv')
JUV_CHANGE_MODEL_PATH = _table_path('regression_results_juvenile_change.csv')
JUV_LOSS_MODEL_PATH = _table_path('regression_results_juvenile_loss.csv')
JUV_RECOVERY_MODEL_PATH = _table_path('regression_results_juvenile_recovery.csv')
TUKEY_NEGTAIL_SUMMARY_PATH = _table_path('tukey_negative_tail_summary.csv')
TUKEY_NEGTAIL_HIERARCHY_PATH = _table_path('tukey_negative_tail_hierarchy.csv')
TUKEY_NEGTAIL_MODEL_PATH = _table_path('tukey_negative_tail_model_results.csv')
TUKEY_NEGTAIL_ADJUSTED_PATH = _table_path('tukey_negative_tail_adjusted_means.csv')
REFERENCE_CATEGORY_SENSITIVITY_PATH = _table_path('reference_category_sensitivity.csv')
REFERENCE_CATEGORY_ADJUSTED_MEANS_PATH = _table_path('reference_category_adjusted_means.csv')
ABSOLUTE_LOSS_MODEL_PATH = _table_path('absolute_loss_model_results.csv')
ABSOLUTE_LOSS_ADJUSTED_PATH = _table_path('absolute_loss_adjusted_means.csv')
DUAL_TRACK_COMPARISON_PATH = _table_path('dual_track_comparison.csv')
COOLING_EVIDENCE_RESULTS_PATH = _table_path('cooling_evidence_results.csv')
CONCURRENT_DAILY_PROOF_PATH = _table_path('concurrent_daily_proof_cleaned.csv')

FIG0_PATH = _figure_path('fig0_study_area_map.png')
FIG1_PATH = _figure_path('fig1_hierarchy_ranking.png')
FIG2_PATH = _figure_path('fig2_adjusted_sequence_inference.png')
FATE_FIG_PATH = _figure_path('fig3_fate_divergence.png')
ROBUSTNESS_FIG_PATH = _figure_path('fig4_robustness_matrix.png')
SUPPORTING_EVIDENCE_FIG_PATH = _figure_path('fig5_supporting_evidence.png')
FIGS1_PATH = _figure_path('figS1_gap_memory.png')
FIGS2_PATH = _figure_path('figS2_juvenile_evidence.png')
OLS_DIAGNOSTIC_FIG_PATH = _figure_path('figS3_ols_diagnostics.png')
FIGS4_PATH = _figure_path('figS4_baseline_state_gradients.png')
FIGS5_PATH = _figure_path('figS6_spatial_autocorrelation.png')
OLS_COEFFICIENT_FIG_PATH = _figure_path('figS5_ols_coefficients.png')
FIGS6_PATH = _figure_path('figS7_extraction_flowchart.png')
