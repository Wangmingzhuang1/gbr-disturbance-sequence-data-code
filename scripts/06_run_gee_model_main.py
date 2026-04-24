# -*- coding: utf-8 -*-
import os

import pandas as pd

import config
from modeling_core import (
    adjusted_marginal_means,
    build_attrition_audit,
    build_base_formula,
    downstream_sample_audit,
    filter_downstream_analysis_sample,
    fit_cluster_robust_model,
    get_rare_categories,
    prepare_model_data,
    should_include_rare_classes,
)


def save_summary(summary_df, path):
    summary_df.to_csv(path)
    print(f"Saved {path}")


def run_models():
    print("Running cluster-robust regression models...")
    if not os.path.exists(config.FINAL_FEATURES_PATH):
        print(f"Error: {config.FINAL_FEATURES_PATH} not found.")
        return

    df = pd.read_csv(config.FINAL_FEATURES_PATH)
    analysis_df = filter_downstream_analysis_sample(df)
    sample_audit = downstream_sample_audit(df)
    rare_categories = get_rare_categories(analysis_df)
    include_rare_classes_main = should_include_rare_classes(analysis_df)

    main_df = prepare_model_data(df, include_cots=False, include_herbivore=False, include_rare_classes=include_rare_classes_main)
    eco_df = prepare_model_data(df, include_cots=True, include_herbivore=False, include_rare_classes=include_rare_classes_main)
    bio_df = prepare_model_data(df, include_cots=True, include_herbivore=True, include_rare_classes=include_rare_classes_main)

    def formula_for(df_model, include_cots=False, include_herbivore=False, include_rare_classes=False):
        categories = sorted(df_model['seq_category_model'].dropna().unique().tolist())
        if not categories:
            return None
        reference = 'Isolated_Heatwave' if 'Isolated_Heatwave' in categories else categories[0]
        return build_base_formula(
            include_cots=include_cots,
            include_herbivore=include_herbivore,
            include_rare_classes=include_rare_classes,
            reference_category=reference,
        )

    specs = [
        ('main_model', main_df, formula_for(main_df, include_cots=False, include_herbivore=False, include_rare_classes=include_rare_classes_main), config.REGRESSION_MAIN_PATH),
        ('ecological_pressure', eco_df, formula_for(eco_df, include_cots=True, include_herbivore=False, include_rare_classes=include_rare_classes_main), config.REGRESSION_ECO_PATH),
        ('biological_covariates', bio_df, formula_for(bio_df, include_cots=True, include_herbivore=True, include_rare_classes=include_rare_classes_main), config.REGRESSION_BIO_PATH),
    ]

    fitted = {}
    for label, df_model, formula, path in specs:
        if df_model.empty or formula is None:
            print(f"Skipping {label}: no eligible rows after filtering.")
            pd.DataFrame(
                columns=['Coefficient', 'StdErr', 'P-value', 'CI_lower', 'CI_upper', 'P-value_FDR', 'Significant_FDR', 'model_name']
            ).to_csv(path)
            continue
        print(f"Fitting {label} (n={len(df_model)})...")
        result, summary = fit_cluster_robust_model(df_model, formula, label)
        save_summary(summary, path)
        fitted[label] = (result, df_model)

    if 'main_model' not in fitted:
        raise RuntimeError("main_model failed to fit; cannot generate adjusted means or residuals.")

    adjusted = adjusted_marginal_means(fitted['main_model'][0], fitted['main_model'][1])
    adjusted.to_csv(config.ADJUSTED_MEANS_PATH, index=False)

    residuals = fitted['main_model'][1][['reef_name', 'start_year']].copy()
    residuals['residual'] = fitted['main_model'][0].resid
    residuals.to_csv(config.MAIN_RESIDUALS_PATH, index=False)

    model_audit = build_attrition_audit(
        df,
        {
            'main_model': main_df,
            'ecological_pressure': eco_df,
            'biological_covariates': bio_df,
        },
    )
    model_audit.to_csv(config.MODEL_ATTRITION_PATH, index=False)
    pd.DataFrame(
        columns=['Coefficient', 'StdErr', 'P-value', 'CI_lower', 'CI_upper', 'P-value_FDR', 'Significant_FDR', 'model_name']
    ).to_csv(config.REGRESSION_RARE_PATH, index=False)

    print("\n--- Model sample sizes ---")
    for label, df_model in [('main_model', main_df), ('ecological_pressure', eco_df), ('biological_covariates', bio_df)]:
        print(f"{label:24s}: {len(df_model)} events / {df_model['reef_name'].nunique()} reefs")

    print(f"\nDownstream excluded categories: {sorted(config.DOWNSTREAM_EXCLUDED_SEQS)}")
    print(f"Strict retained sample: {sample_audit['raw_n']} events")
    print(f"Downstream analysis sample: {sample_audit['analysis_n']} events")
    print(f"Excluded by fixed rule: {sample_audit['excluded_counts']}")
    print(f"Dynamic rare categories under n<{config.RARE_CLASS_MIN_N} after fixed exclusion: {rare_categories if rare_categories else 'None'}")
    print(f"Main-model includes all categories: {include_rare_classes_main}")

    print("\n--- Main adjusted marginal means ---")
    print(adjusted.to_string(index=False))


if __name__ == '__main__':
    run_models()
