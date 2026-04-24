# -*- coding: utf-8 -*-
"""
Consolidated readout for the finalized disturbance-sequence pipeline.

The fixed execution order is:
02_build_matrix -> 03_merge_eco_dist -> 05_analyze_succession ->
08_feature_engineering -> 06_run_gee_model_main -> 07_physical_cooling_proof ->
11_spatial_autocorrelation -> 13_bootstrap_ci -> 12_sensitivity_analysis ->
17_targeted_robustness -> 18_fate_divergence -> 14_auxiliary_ecology ->
15_auxiliary_juveniles -> 16_model_diagnostics ->
19_extended_sequence_inference -> 09_analysis_combined

`16_model_diagnostics` is required for the negative-relative-loss audit
used in the manuscript and for Figure S3.
"""
import os

import pandas as pd

import config


def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    print("=" * 60)
    print("FINAL CONSOLIDATED RESULTS: GBR Disturbance Sequence Pipeline")
    print("=" * 60)
    print("Fixed pipeline order: 02 -> 03 -> 05 -> 08 -> 06 -> 07 -> 11 -> 13 -> 12 -> 17 -> 18 -> 14 -> 15 -> 16 -> 19 -> 09")

    attrition = load_csv(config.ATTRITION_AUDIT_PATH)
    if attrition is not None:
        print("\n[Audit] Stage-wise attrition:")
        print(attrition.to_string(index=False))

    features = load_csv(config.FINAL_FEATURES_PATH)
    if features is not None:
        print(f"\n[Data] Retained sample: {len(features)} sequences from {features['reef_name'].nunique()} reefs")
        print("\nValue counts by category:")
        print(features['seq_category_main'].value_counts().to_string())

    hierarchy = load_csv(config.BOOTSTRAP_HIERARCHY_PATH)
    if hierarchy is not None:
        print("\n[Result] Bootstrap hierarchy:")
        print(hierarchy[['seq_category', 'n', 'mean_rel_loss', 'ci_lo', 'ci_hi']].to_string(index=False))

    adjusted = load_csv(config.ADJUSTED_MEANS_PATH)
    if adjusted is not None:
        print("\n[Result] Adjusted marginal means:")
        print(adjusted[['seq_category', 'adjusted_mean', 'ci_lo', 'ci_hi']].to_string(index=False))

    model_attrition = load_csv(config.MODEL_ATTRITION_PATH)
    if model_attrition is not None:
        print("\n[Audit] Model-layer attrition:")
        print(model_attrition.to_string(index=False))

    spatial = load_csv(config.SPATIAL_DIAGNOSTICS_PATH)
    if spatial is not None:
        print("\n[Audit] Spatial residual independence:")
        print(spatial.to_string(index=False))

    neg_audit = load_csv(config.NEG_REL_LOSS_AUDIT_PATH)
    if neg_audit is not None:
        print("\n[Audit] Negative relative-loss summary:")
        show = neg_audit[['rel_loss_sign', 'n_events', 'mean_nadir_lag', 'mean_baseline_hc']]
        print(show.to_string(index=False))

    psm = load_csv(config.PSM_SEQUENCE_EFFECT_PATH)
    if psm is not None:
        print("\n[Extended] PSM sequence effects:")
        print(psm.to_string(index=False))

    perm = load_csv(config.PERMUTATION_SEQUENCE_EFFECT_PATH)
    if perm is not None:
        print("\n[Extended] Permutation sequence effects:")
        print(perm.to_string(index=False))

    trend = load_csv(config.SPATIOTEMPORAL_TREND_SUMMARY_PATH)
    if trend is not None:
        print("\n[Extended] Spatiotemporal trend summary:")
        print(trend.to_string(index=False))

    gap = load_csv(config.GAP_MEMORY_MAIN_RESULTS_PATH)
    if gap is not None:
        print("\n[Extended] Gap-memory summary:")
        print(gap.to_string(index=False))

    fate_summary = load_csv(config.FATE_SUMMARY_PATH)
    if fate_summary is not None:
        show = fate_summary[
            (fate_summary['summary_type'] == 'stage') &
            (fate_summary['group_type'] == 'overall')
        ]
        print("\n[Result] Fate divergence stages:")
        print(show[['group_value', 'n_events', 'n_reefs']].to_string(index=False))

    fate_supplement = load_csv(config.FATE_SUPPLEMENT_SUMMARY_PATH)
    if fate_supplement is not None:
        show = fate_supplement[
            (fate_supplement['summary_type'] == 'stage') &
            (fate_supplement['group_type'] == 'overall')
        ]
        print("\n[Result] Fate supplementary stages:")
        print(show[['sample_scope', 'group_value', 'n_events', 'n_reefs']].to_string(index=False))


if __name__ == '__main__':
    main()
