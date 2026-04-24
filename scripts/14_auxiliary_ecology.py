# -*- coding: utf-8 -*-
"""
Auxiliary ecological evidence screening for the finalized disturbance-sequence paper.

This script does not alter the main analysis chain. It evaluates whether
secondary ecological dimensions are strong enough to support the hard-coral
risk-hierarchy story.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
from modeling_core import filter_downstream_analysis_sample


AUX_VARS = [
    'cots_lag_3yr',
    'initial_cots',
    'baseline_algae',
    'nadir_algae',
    'final_algae',
    'initial_herbivore',
]


def load_features():
    df = pd.read_csv(config.FINAL_FEATURES_PATH)
    df = filter_downstream_analysis_sample(df)
    df['algae_change_nadir'] = df['nadir_algae'] - df['baseline_algae']
    df['algae_change_final'] = df['final_algae'] - df['baseline_algae']
    return df


def build_coverage(df):
    records = []
    for var in AUX_VARS + ['algae_change_nadir', 'algae_change_final']:
        overall = {
            'variable': var,
            'scope': 'overall',
            'seq_category_main': 'ALL',
            'non_missing_rows': int(df[var].notna().sum()),
            'non_missing_reefs': int(df.loc[df[var].notna(), 'reef_name'].nunique()),
            'mean': float(df[var].mean()) if df[var].notna().any() else np.nan,
            'median': float(df[var].median()) if df[var].notna().any() else np.nan,
        }
        records.append(overall)
        for seq, sub in df.groupby('seq_category_main'):
            s = sub[var]
            records.append({
                'variable': var,
                'scope': 'by_sequence',
                'seq_category_main': seq,
                'non_missing_rows': int(s.notna().sum()),
                'non_missing_reefs': int(sub.loc[s.notna(), 'reef_name'].nunique()),
                'mean': float(s.mean()) if s.notna().any() else np.nan,
                'median': float(s.median()) if s.notna().any() else np.nan,
            })
    return pd.DataFrame(records)


def build_correlations(df):
    records = []
    targets = ['rel_loss', 'recovery_pct']
    variables = [
        'cots_lag_3yr',
        'initial_cots',
        'initial_herbivore',
        'baseline_algae',
        'algae_change_nadir',
        'algae_change_final',
    ]
    for var in variables:
        for target in targets:
            sub = df[[var, target]].dropna()
            if len(sub) < 5:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = sub[var].corr(sub[target], method='pearson')
                spearman = sub[var].corr(sub[target], method='spearman')
            records.append({
                'variable': var,
                'target': target,
                'n': int(len(sub)),
                'pearson_r': pearson,
                'spearman_rho': spearman,
            })
    return pd.DataFrame(records)


def fit_algae_model(df):
    sub = df.copy()
    required = [
        'algae_change_final',
        'storm_exposure_seq_c',
        'heat_exposure_seq_c',
        'baseline_hc_c',
        'start_year_c',
        'region_lat_c',
        'reef_name',
        'seq_category_model',
    ]
    sub = sub.dropna(subset=required)
    if sub.empty:
        return pd.DataFrame()

    model = smf.ols(
        "algae_change_final ~ C(seq_category_model, Treatment(reference='Isolated_Heatwave')) + "
        "storm_exposure_seq_c + heat_exposure_seq_c + baseline_hc_c + start_year_c + region_lat_c",
        data=sub,
    ).fit(cov_type='cluster', cov_kwds={'groups': sub['reef_name']})

    out = pd.DataFrame({
        'term': model.params.index,
        'Coefficient': model.params.values,
        'StdErr': model.bse.values,
        'P_value': model.pvalues.values,
        'CI_lower': model.conf_int()[0].values,
        'CI_upper': model.conf_int()[1].values,
    })
    out['model_name'] = 'algae_change_final_aux'
    out['n_events'] = int(len(sub))
    out['n_reefs'] = int(sub['reef_name'].nunique())
    return out


def build_summary(df):
    grouped = df.groupby('seq_category_main').agg(
        n_events=('seq_category_main', 'size'),
        mean_rel_loss=('rel_loss', 'mean'),
        mean_recovery_pct=('recovery_pct', 'mean'),
        mean_cots_lag_3yr=('cots_lag_3yr', 'mean'),
        mean_initial_cots=('initial_cots', 'mean'),
        mean_initial_herbivore=('initial_herbivore', 'mean'),
        mean_baseline_algae=('baseline_algae', 'mean'),
        mean_algae_change_nadir=('algae_change_nadir', 'mean'),
        mean_algae_change_final=('algae_change_final', 'mean'),
        non_missing_final_algae=('final_algae', lambda x: int(x.notna().sum())),
    ).reset_index()
    return grouped


def main():
    df = load_features()

    coverage = build_coverage(df)
    coverage.to_csv(config.ECO_AUX_COVERAGE_PATH, index=False)

    corr = build_correlations(df)
    corr.to_csv(config.ECO_AUX_CORR_PATH, index=False)

    summary = build_summary(df)
    summary.to_csv(config.ECO_AUX_SUMMARY_PATH, index=False)

    algae_model = fit_algae_model(df)
    algae_model.to_csv(config.ECO_AUX_ALGAE_MODEL_PATH, index=False)

    print("Auxiliary ecology screening complete.")
    print(f"Coverage: {config.ECO_AUX_COVERAGE_PATH}")
    print(f"Summary: {config.ECO_AUX_SUMMARY_PATH}")
    print(f"Correlations: {config.ECO_AUX_CORR_PATH}")
    print(f"Algae model: {config.ECO_AUX_ALGAE_MODEL_PATH}")


if __name__ == '__main__':
    main()
