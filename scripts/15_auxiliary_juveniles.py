# -*- coding: utf-8 -*-
"""
Juvenile-response screening for the finalized disturbance-sequence paper.

This script preserves the fixed sequence sample and evaluates whether juvenile
signals are coherent enough to support the hard-coral risk hierarchy.

Primary juvenile response:
    juv_change_nadir
        signed net change from baseline to nadir abundance

Supplementary decline screen:
    juv_loss
        one-sided decline metric; truncated at zero when nadir abundance
        exceeds baseline abundance
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import config
from modeling_core import filter_downstream_analysis_sample


JUVENILE_VARS = [
    'baseline_juv',
    'nadir_juv',
    'final_juv',
    'juv_loss',
    'juv_change_nadir',
    'juv_recovery_rate',
]


def load_features():
    df = pd.read_csv(config.FINAL_FEATURES_PATH)
    return filter_downstream_analysis_sample(df)


def build_coverage(df):
    records = []
    for var in JUVENILE_VARS:
        records.append({
            'variable': var,
            'scope': 'overall',
            'seq_category_main': 'ALL',
            'non_missing_rows': int(df[var].notna().sum()),
            'non_missing_reefs': int(df.loc[df[var].notna(), 'reef_name'].nunique()),
            'mean': float(df[var].mean()) if df[var].notna().any() else np.nan,
            'median': float(df[var].median()) if df[var].notna().any() else np.nan,
        })
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


def build_summary(df):
    grouped = df.groupby('seq_category_main').agg(
        n_events=('seq_category_main', 'size'),
        non_missing_juv_loss=('juv_loss', lambda x: int(x.notna().sum())),
        non_missing_juv_change=('juv_change_nadir', lambda x: int(x.notna().sum())),
        non_missing_juv_recovery=('juv_recovery_rate', lambda x: int(x.notna().sum())),
        mean_juv_loss=('juv_loss', 'mean'),
        mean_juv_change_nadir=('juv_change_nadir', 'mean'),
        mean_juv_recovery_rate=('juv_recovery_rate', 'mean'),
        mean_heat_exposure_seq=('heat_exposure_seq', 'mean'),
        mean_storm_exposure_seq=('storm_exposure_seq', 'mean'),
        mean_rel_loss=('rel_loss', 'mean'),
    ).reset_index()
    return grouped


def build_correlations(df):
    records = []
    pairs = [
        ('juv_loss', 'rel_loss'),
        ('juv_change_nadir', 'rel_loss'),
        ('juv_loss', 'heat_exposure_seq'),
        ('juv_change_nadir', 'heat_exposure_seq'),
        ('juv_loss', 'storm_exposure_seq'),
        ('juv_change_nadir', 'storm_exposure_seq'),
        ('juv_recovery_rate', 'rel_loss'),
        ('juv_recovery_rate', 'heat_exposure_seq'),
        ('juv_recovery_rate', 'storm_exposure_seq'),
    ]
    for var, target in pairs:
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


def _fit_model(df, response):
    required = [
        response,
        'reef_name',
        'seq_category_model',
        'storm_exposure_seq_c',
        'heat_exposure_seq_c',
        'baseline_hc_c',
        'start_year_c',
        'region_lat_c',
    ]
    sub = df.dropna(subset=required).copy()
    if sub.empty:
        return pd.DataFrame()

    formula = (
        f"{response} ~ C(seq_category_model, Treatment(reference='Isolated_Heatwave')) + "
        "storm_exposure_seq_c + heat_exposure_seq_c + baseline_hc_c + start_year_c + "
        "region_lat_c"
    )
    model = smf.ols(formula=formula, data=sub).fit(
        cov_type='cluster',
        cov_kwds={'groups': sub['reef_name']},
    )

    out = pd.DataFrame({
        'term': model.params.index,
        'Coefficient': model.params.values,
        'StdErr': model.bse.values,
        'P_value': model.pvalues.values,
        'CI_lower': model.conf_int()[0].values,
        'CI_upper': model.conf_int()[1].values,
    })
    non_intercept = out['term'] != 'Intercept'
    if non_intercept.any():
        mt = multipletests(out.loc[non_intercept, 'P_value'].values, method='fdr_bh')
        out.loc[non_intercept, 'P_value_FDR'] = mt[1]
        out.loc[non_intercept, 'Significant_FDR'] = mt[0].astype(bool)
    out['model_name'] = response
    out['n_events'] = int(len(sub))
    out['n_reefs'] = int(sub['reef_name'].nunique())
    return out


def main():
    df = load_features()

    build_coverage(df).to_csv(config.JUV_AUX_COVERAGE_PATH, index=False)
    build_summary(df).to_csv(config.JUV_AUX_SUMMARY_PATH, index=False)
    build_correlations(df).to_csv(config.JUV_AUX_CORR_PATH, index=False)
    _fit_model(df, 'juv_change_nadir').to_csv(config.JUV_CHANGE_MODEL_PATH, index=False)
    _fit_model(df, 'juv_loss').to_csv(config.JUV_LOSS_MODEL_PATH, index=False)
    _fit_model(df, 'juv_recovery_rate').to_csv(config.JUV_RECOVERY_MODEL_PATH, index=False)

    print("Juvenile auxiliary screening complete.")
    print(f"Coverage: {config.JUV_AUX_COVERAGE_PATH}")
    print(f"Summary: {config.JUV_AUX_SUMMARY_PATH}")
    print(f"Correlations: {config.JUV_AUX_CORR_PATH}")
    print(f"Change model: {config.JUV_CHANGE_MODEL_PATH}")
    print(f"Loss model: {config.JUV_LOSS_MODEL_PATH}")
    print(f"Recovery model: {config.JUV_RECOVERY_MODEL_PATH}")


if __name__ == '__main__':
    main()
