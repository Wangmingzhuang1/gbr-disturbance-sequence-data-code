# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
from modeling_core import adjusted_marginal_means, filter_downstream_analysis_sample, fit_cluster_robust_model, prepare_model_data


def _build_formula(response, reference_category):
    category_term = f"C(seq_category_model, Treatment(reference='{reference_category}'))"
    terms = [
        category_term,
        'storm_exposure_seq_c',
        'heat_exposure_seq_c',
        'baseline_hc_c',
        'start_year_c',
        'region_lat_c',
    ]
    return f"{response} ~ " + " + ".join(terms)


def _save(df, filename):
    mapping = {
        'reference_category_sensitivity.csv': config.REFERENCE_CATEGORY_SENSITIVITY_PATH,
        'reference_category_adjusted_means.csv': config.REFERENCE_CATEGORY_ADJUSTED_MEANS_PATH,
        'absolute_loss_model_results.csv': config.ABSOLUTE_LOSS_MODEL_PATH,
        'absolute_loss_adjusted_means.csv': config.ABSOLUTE_LOSS_ADJUSTED_PATH,
    }
    path = mapping.get(filename, os.path.join(config.OUTPUT_TABLES_DIR, filename))
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def _fit_custom_response(df_model, response, reference_category, label):
    work = df_model.dropna(
        subset=[
            response,
            'reef_name',
            'seq_category_model',
            'storm_exposure_seq_c',
            'heat_exposure_seq_c',
            'baseline_hc_c',
            'start_year_c',
            'region_lat_c',
        ]
    ).copy()
    formula = _build_formula(response, reference_category)
    model = smf.ols(formula=formula, data=work)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': work['reef_name']})

    summary = pd.concat([result.params, result.bse, result.pvalues, result.conf_int()], axis=1)
    summary.columns = ['Coefficient', 'StdErr', 'P-value', 'CI_lower', 'CI_upper']
    summary.insert(0, 'term', summary.index)
    summary.insert(0, 'reference_category', reference_category)
    summary.insert(0, 'response', response)
    summary.insert(0, 'model_name', label)

    adjusted = adjusted_marginal_means(result, work)
    adjusted.insert(0, 'reference_category', reference_category)
    adjusted.insert(0, 'response', response)
    adjusted.insert(0, 'model_name', label)
    return summary.reset_index(drop=True), adjusted


def main():
    df = filter_downstream_analysis_sample(pd.read_csv(config.FINAL_FEATURES_PATH))
    main_df = prepare_model_data(df, include_cots=False, include_herbivore=False, include_rare_classes=None)

    ref_summaries = []
    ref_adjusted = []
    for reference in ['Isolated_Heatwave', 'Isolated_Storm', 'H_to_H']:
        formula = _build_formula('rel_loss', reference)
        result, summary = fit_cluster_robust_model(main_df, formula, f"rel_loss_ref_{reference}")
        summary = summary.reset_index().rename(columns={'index': 'term'})
        summary.insert(0, 'reference_category', reference)
        ref_summaries.append(summary)

        adjusted = adjusted_marginal_means(result, main_df)
        adjusted.insert(0, 'reference_category', reference)
        ref_adjusted.append(adjusted)

    _save(pd.concat(ref_summaries, ignore_index=True), 'reference_category_sensitivity.csv')
    _save(pd.concat(ref_adjusted, ignore_index=True), 'reference_category_adjusted_means.csv')

    df['absolute_loss'] = df['baseline_hc'] - df['nadir_hc']
    abs_df = prepare_model_data(df, include_cots=False, include_herbivore=False, include_rare_classes=None)
    abs_summary, abs_adjusted = _fit_custom_response(
        abs_df,
        response='absolute_loss',
        reference_category='Isolated_Heatwave',
        label='absolute_loss_main',
    )
    _save(abs_summary, 'absolute_loss_model_results.csv')
    _save(abs_adjusted, 'absolute_loss_adjusted_means.csv')

    q1 = df['rel_loss'].quantile(0.25)
    q3 = df['rel_loss'].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    tukey_df = df[df['rel_loss'] >= lower_fence].copy()
    tukey_model_df = prepare_model_data(tukey_df, include_cots=False, include_herbivore=False, include_rare_classes=None)
    tukey_summary, tukey_adjusted = _fit_custom_response(
        tukey_model_df,
        response='rel_loss',
        reference_category='Isolated_Heatwave',
        label='tukey_negative_tail_filtered',
    )
    tukey_hierarchy = (
        tukey_df.groupby('seq_category_main', as_index=False)
        .agg(
            n_events=('rel_loss', 'size'),
            mean_rel_loss=('rel_loss', 'mean'),
        )
        .sort_values('mean_rel_loss', ascending=False)
    )
    tukey_meta = pd.DataFrame([{
        'sample_name': 'tukey_negative_tail_filtered',
        'lower_fence': lower_fence,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'n_removed': int((df['rel_loss'] < lower_fence).sum()),
        'n_retained_total': int(len(tukey_df)),
        'n_model_events': int(len(tukey_model_df)),
        'n_model_reefs': int(tukey_model_df['reef_name'].nunique()),
    }])
    tukey_meta.to_csv(config.TUKEY_NEGTAIL_SUMMARY_PATH, index=False)
    tukey_hierarchy.to_csv(config.TUKEY_NEGTAIL_HIERARCHY_PATH, index=False)
    tukey_summary.to_csv(config.TUKEY_NEGTAIL_MODEL_PATH, index=False)
    tukey_adjusted.to_csv(config.TUKEY_NEGTAIL_ADJUSTED_PATH, index=False)
    print(f"Saved {config.TUKEY_NEGTAIL_SUMMARY_PATH}")
    print(f"Saved {config.TUKEY_NEGTAIL_HIERARCHY_PATH}")
    print(f"Saved {config.TUKEY_NEGTAIL_MODEL_PATH}")
    print(f"Saved {config.TUKEY_NEGTAIL_ADJUSTED_PATH}")

    print("\nReference-category sensitivity complete.")
    print("Absolute-loss robustness complete.")
    print("Tukey negative-tail robustness complete.")


if __name__ == '__main__':
    main()
