# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

import config
from modeling_core import adjusted_marginal_means, build_base_formula, filter_downstream_analysis_sample, fit_cluster_robust_model, prepare_model_data
from sequence_analysis_core import SEQ_ORDER, extract_sequences, load_master_matrix


def append_hierarchy_rows(rows, scenario_group, scenario_name, df_seq):
    summary = df_seq.groupby('seq_category_main')['rel_loss'].agg(['count', 'mean']).reindex(SEQ_ORDER)
    summary['rank'] = summary['mean'].rank(ascending=False, method='min')
    for category, row in summary.iterrows():
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'hierarchy', 'metric_name': 'count', 'seq_category': category, 'value': row['count'], 'ci_lo': np.nan, 'ci_hi': np.nan})
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'hierarchy', 'metric_name': 'mean_rel_loss', 'seq_category': category, 'value': row['mean'], 'ci_lo': np.nan, 'ci_hi': np.nan})
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'hierarchy', 'metric_name': 'rank', 'seq_category': category, 'value': row['rank'], 'ci_lo': np.nan, 'ci_hi': np.nan})


def append_contrast_rows(rows, scenario_group, scenario_name, df_seq):
    means = df_seq.groupby('seq_category_main')['rel_loss'].mean()
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'contrast', 'metric_name': 'S_to_H_minus_Isolated_Heatwave', 'seq_category': '', 'value': means.get('S_to_H', np.nan) - means.get('Isolated_Heatwave', np.nan), 'ci_lo': np.nan, 'ci_hi': np.nan})
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'contrast', 'metric_name': 'S_to_H_minus_H_to_H', 'seq_category': '', 'value': means.get('S_to_H', np.nan) - means.get('H_to_H', np.nan), 'ci_lo': np.nan, 'ci_hi': np.nan})


def append_sample_rows(rows, scenario_group, scenario_name, df_seq):
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'sample', 'metric_name': 'retained_sequences', 'seq_category': '', 'value': float(len(df_seq)), 'ci_lo': np.nan, 'ci_hi': np.nan})
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'sample', 'metric_name': 'retained_reefs', 'seq_category': '', 'value': float(df_seq['reef_name'].nunique()), 'ci_lo': np.nan, 'ci_hi': np.nan})


def append_model_rows(rows, scenario_group, scenario_name, df_seq, include_rare_classes=None, reference_category=None):
    df_model = prepare_model_data(df_seq, include_cots=False, include_herbivore=False, include_rare_classes=include_rare_classes)
    if df_model.empty:
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'model_status', 'metric_name': 'model_ran', 'seq_category': '', 'value': 0.0, 'ci_lo': np.nan, 'ci_hi': np.nan})
        return

    category_column = 'seq_category_model'
    categories = df_model[category_column].dropna().unique().tolist()
    if reference_category is None:
        reference_category = 'Isolated_Heatwave' if 'Isolated_Heatwave' in categories else sorted(categories)[0]
    if reference_category not in categories:
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'model_status', 'metric_name': 'model_ran', 'seq_category': '', 'value': 0.0, 'ci_lo': np.nan, 'ci_hi': np.nan})
        return

    formula = build_base_formula(include_cots=False, include_herbivore=False, include_rare_classes=include_rare_classes, reference_category=reference_category)
    result, summary = fit_cluster_robust_model(df_model, formula, scenario_name)

    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'model_status', 'metric_name': 'model_ran', 'seq_category': '', 'value': 1.0, 'ci_lo': np.nan, 'ci_hi': np.nan})
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'sample', 'metric_name': 'model_events', 'seq_category': '', 'value': float(len(df_model)), 'ci_lo': np.nan, 'ci_hi': np.nan})
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'sample', 'metric_name': 'model_reefs', 'seq_category': '', 'value': float(df_model['reef_name'].nunique()), 'ci_lo': np.nan, 'ci_hi': np.nan})
    rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'model_status', 'metric_name': 'reference_category', 'seq_category': reference_category, 'value': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan})

    adjusted = adjusted_marginal_means(result, df_model, category_column=category_column)
    for _, row in adjusted.iterrows():
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'adjusted_mean', 'metric_name': 'adjusted_mean', 'seq_category': row['seq_category'], 'value': row['adjusted_mean'], 'ci_lo': row['ci_lo'], 'ci_hi': row['ci_hi']})

    for term in ['storm_exposure_seq_c', 'heat_exposure_seq_c', 'baseline_hc_c']:
        if term not in summary.index:
            continue
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'model_coef', 'metric_name': term, 'seq_category': '', 'value': summary.loc[term, 'Coefficient'], 'ci_lo': summary.loc[term, 'CI_lower'], 'ci_hi': summary.loc[term, 'CI_upper']})

    seq_terms = summary[summary.index.str.contains('seq_category_model', regex=False)].copy()
    for term, row in seq_terms.iterrows():
        rows.append({'scenario_group': scenario_group, 'scenario_name': scenario_name, 'metric_group': 'sequence_coef', 'metric_name': term, 'seq_category': '', 'value': row['Coefficient'], 'ci_lo': row['CI_lower'], 'ci_hi': row['CI_upper']})


def _extract_sequences_with_min_baseline(master, min_baseline_hc):
    original = config.MIN_BASELINE_HC
    try:
        config.MIN_BASELINE_HC = float(min_baseline_hc)
        return extract_sequences(master, lookahead_years=4, rule='first_event', mode='strict')
    finally:
        config.MIN_BASELINE_HC = original


def run_sensitivity():
    master = load_master_matrix()
    rows = []

    scenarios = [
        ('rule', 'primary_first_event', extract_sequences(master, lookahead_years=4, rule='first_event', mode='strict'), None),
        ('rule', 'storm_priority', extract_sequences(master, lookahead_years=4, rule='storm_heat_priority', mode='strict'), None),
        ('window', 'window_1_3', extract_sequences(master, lookahead_years=3, rule='first_event', mode='strict'), None),
        ('window', 'window_1_4', extract_sequences(master, lookahead_years=4, rule='first_event', mode='strict'), None),
        ('window', 'window_1_5', extract_sequences(master, lookahead_years=5, rule='first_event', mode='strict'), None),
        ('baseline_threshold', 'baseline_hc_10pct', _extract_sequences_with_min_baseline(master, 10.0), None),
    ]

    for group_name, scenario_name, df_seq, include_rare_classes in scenarios:
        df_seq = filter_downstream_analysis_sample(df_seq)
        append_sample_rows(rows, group_name, scenario_name, df_seq)
        append_hierarchy_rows(rows, group_name, scenario_name, df_seq)
        append_contrast_rows(rows, group_name, scenario_name, df_seq)
        append_model_rows(rows, group_name, scenario_name, df_seq, include_rare_classes=include_rare_classes)

    primary_df = next(filter_downstream_analysis_sample(df_seq) for group_name, scenario_name, df_seq, _ in scenarios if scenario_name == 'primary_first_event')
    for reference_category in ['Isolated_Heatwave', 'Isolated_Storm', 'H_to_H']:
        append_model_rows(
            rows,
            'reference_category',
            f'reference_{reference_category}',
            primary_df,
            include_rare_classes=False,
            reference_category=reference_category,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(config.ROBUSTNESS_SUMMARY_PATH, index=False)
    print(summary.head().to_string(index=False))


if __name__ == '__main__':
    run_sensitivity()
