# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests

import config


REFERENCE_CATEGORY = 'Isolated_Heatwave'


def category_counts(df, category_col='seq_category_main'):
    if category_col not in df.columns:
        return pd.Series(dtype='int64')
    return df[category_col].value_counts(dropna=False)


def get_rare_categories(df, min_n=config.RARE_CLASS_MIN_N, category_col='seq_category_main'):
    counts = category_counts(df, category_col=category_col)
    if counts.empty:
        return []
    return sorted([str(category) for category, count in counts.items() if pd.notna(category) and int(count) < min_n])


def downstream_excluded_sequences():
    return set(config.DOWNSTREAM_EXCLUDED_SEQS)


def filter_downstream_analysis_sample(df, seq_column='seq_category_main', copy=True):
    work = df.copy() if copy else df
    if seq_column not in work.columns:
        return work
    return work[~work[seq_column].isin(downstream_excluded_sequences())].copy()


def downstream_sample_audit(df, seq_column='seq_category_main'):
    if seq_column not in df.columns:
        return {
            'raw_n': int(len(df)),
            'analysis_n': int(len(df)),
            'excluded_n': 0,
            'excluded_counts': {},
        }
    counts = (
        df.loc[df[seq_column].isin(downstream_excluded_sequences()), seq_column]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    analysis_n = int((~df[seq_column].isin(downstream_excluded_sequences())).sum())
    return {
        'raw_n': int(len(df)),
        'analysis_n': analysis_n,
        'excluded_n': int(len(df) - analysis_n),
        'excluded_counts': {str(k): int(v) for k, v in counts.items()},
    }


def count_rare_class(df, category=None, min_n=config.RARE_CLASS_MIN_N, category_col='seq_category_main'):
    if category is not None:
        if category_col not in df.columns:
            return 0
        return int((df[category_col] == category).sum())
    rare_categories = get_rare_categories(df, min_n=min_n, category_col=category_col)
    if not rare_categories or category_col not in df.columns:
        return 0
    return int(df[df[category_col].isin(rare_categories)].shape[0])


def should_include_rare_classes(df, min_n=config.RARE_CLASS_MIN_N, category_col='seq_category_main'):
    return len(get_rare_categories(df, min_n=min_n, category_col=category_col)) == 0


def build_base_formula(include_cots=False, include_herbivore=False, include_rare_classes=True, reference_category=REFERENCE_CATEGORY):
    category_term = f"C(seq_category_model, Treatment(reference='{reference_category}'))"
    terms = [
        category_term,
        'storm_exposure_seq_c',
        'heat_exposure_seq_c',
        'baseline_hc_c',
        'start_year_c',
        'region_lat_c',
    ]
    if include_cots:
        terms.append('cots_lag_3yr_c')
    if include_herbivore:
        terms.append('initial_herbivore_c')
    return 'rel_loss ~ ' + ' + '.join(terms)


def prepare_model_data(df, include_cots=False, include_herbivore=False, include_rare_classes=None):
    work = filter_downstream_analysis_sample(df)
    work['sector'] = work['sector'].fillna('Unknown').astype(str)
    sector_map = {
        'North-Central': 'Northern',
        'North': 'Northern',
        'Northern': 'Northern',
        'Central': 'Central',
        'Southern': 'Southern',
        'South': 'Southern',
    }
    work['sector'] = work['sector'].map(lambda value: sector_map.get(value, value))

    if work['region_lat'].notna().any():
        work['region_lat'] = work['region_lat'].fillna(work['region_lat'].mean())

    rare_categories = get_rare_categories(work)
    if include_rare_classes is None:
        include_rare_classes = should_include_rare_classes(work)

    if include_rare_classes:
        work['seq_category_model'] = work['seq_category_main']
    else:
        work = work[~work['seq_category_main'].isin(rare_categories)].copy()
        work['seq_category_model'] = work['seq_category_main']

    centered = [
        'storm_exposure_seq',
        'heat_exposure_seq',
        'baseline_hc',
        'start_year',
        'region_lat',
        'cots_lag_3yr',
        'initial_herbivore',
    ]
    for column in centered:
        if f'{column}_c' not in work.columns and column in work.columns:
            series = pd.to_numeric(work[column], errors='coerce')
            work[f'{column}_c'] = series - series.mean() if series.notna().any() else np.nan

    required = [
        'rel_loss',
        'reef_name',
        'seq_category_model',
        'storm_exposure_seq_c',
        'heat_exposure_seq_c',
        'baseline_hc_c',
        'start_year_c',
        'region_lat_c',
    ]
    if include_cots:
        required.append('cots_lag_3yr_c')
    if include_herbivore:
        required.append('initial_herbivore_c')

    work = work.dropna(subset=required).copy()
    return work


def fit_cluster_robust_model(df_model, formula, label):
    model = smf.ols(formula=formula, data=df_model)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df_model['reef_name']})

    summary = pd.concat([result.params, result.bse, result.pvalues, result.conf_int()], axis=1)
    summary.columns = ['Coefficient', 'StdErr', 'P-value', 'CI_lower', 'CI_upper']
    non_intercept = summary.index != 'Intercept'
    reject, pvals_fdr, _, _ = multipletests(summary.loc[non_intercept, 'P-value'].values, method='fdr_bh')
    summary.loc[non_intercept, 'P-value_FDR'] = pvals_fdr
    summary.loc[non_intercept, 'Significant_FDR'] = reject
    summary['model_name'] = label
    return result, summary


def adjusted_marginal_means(result, df_model, category_column='seq_category_model', n_draw=2000, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    design_info = result.model.data.design_info
    params = result.params.values
    cov = result.cov_params().to_numpy()

    try:
        draws = rng.multivariate_normal(params, cov, size=n_draw)
    except Exception:
        draws = rng.normal(params, np.sqrt(np.abs(np.diag(cov))), size=(n_draw, len(params)))

    rows = []
    categories = sorted(df_model[category_column].dropna().unique())
    for category in categories:
        new_df = df_model.copy()
        new_df[category_column] = category
        design = np.asarray(build_design_matrices([design_info], new_df)[0])
        point_pred = result.predict(new_df).mean()
        draw_means = (design @ draws.T).mean(axis=0)
        rows.append(
            {
                'seq_category': category,
                'adjusted_mean': float(point_pred),
                'ci_lo': float(np.percentile(draw_means, 2.5)),
                'ci_hi': float(np.percentile(draw_means, 97.5)),
                'n_events': int((df_model[category_column] == category).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_model_attrition(all_df, model_specs):
    rows = [
        {
            'model_name': 'all_retained_sequences',
            'n_events': int(len(all_df)),
            'n_reefs': int(all_df['reef_name'].nunique()),
            'dropped_vs_retained': 0,
            'drop_share_vs_retained': 0.0,
        }
    ]
    total_events = len(all_df)
    for name, df_model in model_specs.items():
        dropped = total_events - len(df_model)
        rows.append(
            {
                'model_name': name,
                'n_events': int(len(df_model)),
                'n_reefs': int(df_model['reef_name'].nunique()),
                'dropped_vs_retained': int(dropped),
                'drop_share_vs_retained': (dropped / total_events) if total_events else np.nan,
            }
        )
    return pd.DataFrame(rows)
def build_attrition_audit(all_df, model_specs):
    return build_model_attrition(all_df, model_specs)
