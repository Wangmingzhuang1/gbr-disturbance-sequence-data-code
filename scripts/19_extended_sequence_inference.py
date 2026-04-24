# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, permutation_test, spearmanr, ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

import config
from modeling_core import filter_downstream_analysis_sample


MAIN_COVARIATES = [
    'storm_exposure_seq',
    'heat_exposure_seq',
    'baseline_hc',
    'start_year',
    'region_lat',
]

MAIN_OUTCOMES = ['rel_loss', 'drop_abs']
GAP_GROUPS = ['S_to_H', 'S_to_S', 'H_to_H']
LOWESS_FRAC = 0.75
SHORT_GAP_THRESHOLD = 2
PERIOD_BINS = [1985, 1995, 2005, 2015, 2026]
PERIOD_LABELS = ['1985-1994', '1995-2004', '2005-2014', '2015-2025']


def _load_features():
    if not os.path.exists(config.FINAL_FEATURES_PATH):
        raise FileNotFoundError(f"Missing required input: {config.FINAL_FEATURES_PATH}")
    df = filter_downstream_analysis_sample(pd.read_csv(config.FINAL_FEATURES_PATH))
    df['is_s_to_h'] = (df['seq_category_main'] == 'S_to_H').astype(int)
    return df


def _analysis_frame(df):
    keep = df.copy()
    keep = keep.dropna(subset=MAIN_COVARIATES + ['rel_loss', 'drop_abs'])
    keep = keep[keep['seq_category_main'].notna()].copy()
    return keep


def _fit_propensity(df):
    X = df[MAIN_COVARIATES].copy()
    y = df['is_s_to_h'].astype(int)
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('logit', LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED)),
        ]
    )
    pipe.fit(X, y)
    df = df.copy()
    df['propensity_score'] = pipe.predict_proba(X)[:, 1]
    return df


def _standardized_mean_diff(treated, control, column):
    t = treated[column].astype(float).dropna().to_numpy()
    c = control[column].astype(float).dropna().to_numpy()
    if len(t) == 0 or len(c) == 0:
        return np.nan
    pooled = np.sqrt((np.var(t, ddof=1) + np.var(c, ddof=1)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return (np.mean(t) - np.mean(c)) / pooled


def _match_without_replacement(treated, control):
    cost = np.abs(treated['propensity_score'].to_numpy()[:, None] - control['propensity_score'].to_numpy()[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)
    matched_treated = treated.iloc[row_idx].copy().reset_index(drop=True)
    matched_control = control.iloc[col_idx].copy().reset_index(drop=True)
    matched_treated['match_id'] = np.arange(1, len(matched_treated) + 1)
    matched_control['match_id'] = np.arange(1, len(matched_control) + 1)
    matched_treated['match_role'] = 'treated'
    matched_control['match_role'] = 'control'
    return matched_treated, matched_control, cost[row_idx, col_idx]


def _signflip_paired_p(diff, n_resamples=20000, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
    observed = diff.mean()
    null_means = np.empty(n_resamples)
    for i in range(n_resamples):
        signs = rng.choice([-1, 1], size=len(diff))
        null_means[i] = np.mean(diff * signs)
    p = (np.sum(np.abs(null_means) >= abs(observed)) + 1) / (n_resamples + 1)
    return observed, p, null_means


def run_psm(df):
    work = _analysis_frame(df)
    work = _fit_propensity(work)
    treated = work[work['is_s_to_h'] == 1].copy().sort_values('propensity_score').reset_index(drop=True)
    control = work[work['is_s_to_h'] == 0].copy().sort_values('propensity_score').reset_index(drop=True)
    matched_treated, matched_control, ps_distance = _match_without_replacement(treated, control)

    balance_rows = []
    for cov in MAIN_COVARIATES:
        balance_rows.append(
            {
                'covariate': cov,
                'treated_mean_before': treated[cov].mean(),
                'control_mean_before': control[cov].mean(),
                'smd_before': _standardized_mean_diff(treated, control, cov),
                'treated_mean_after': matched_treated[cov].mean(),
                'control_mean_after': matched_control[cov].mean(),
                'smd_after': _standardized_mean_diff(matched_treated, matched_control, cov),
            }
        )

    effect_rows = []
    null_dist_rows = []
    for outcome in MAIN_OUTCOMES:
        diff = matched_treated[outcome].to_numpy() - matched_control[outcome].to_numpy()
        diff = diff[np.isfinite(diff)]
        paired_t = ttest_rel(matched_treated[outcome], matched_control[outcome], nan_policy='omit')
        try:
            wilc = wilcoxon(diff) if len(diff) > 0 else None
            wilc_p = wilc.pvalue if wilc is not None else np.nan
        except ValueError:
            wilc_p = np.nan
        observed, signflip_p, null_means = _signflip_paired_p(diff)
        for value in null_means:
            null_dist_rows.append({'analysis': 'psm_signflip', 'outcome': outcome, 'null_mean_diff': value})
        effect_rows.append(
            {
                'analysis': 'psm',
                'outcome': outcome,
                'n_treated_pool': len(treated),
                'n_control_pool': len(control),
                'n_matched_pairs': len(diff),
                'treated_mean_matched': matched_treated[outcome].mean(),
                'matched_control_mean': matched_control[outcome].mean(),
                'att': observed,
                'paired_t_p': paired_t.pvalue,
                'wilcoxon_p': wilc_p,
                'signflip_p': signflip_p,
                'avg_abs_ps_gap': float(np.mean(ps_distance)),
            }
        )

    matched = pd.concat(
        [
            matched_treated[
                ['match_id', 'match_role', 'reef_name', 'seq_category_main', 'start_year', 'sector', 'propensity_score'] + MAIN_COVARIATES + MAIN_OUTCOMES
            ],
            matched_control[
                ['match_id', 'match_role', 'reef_name', 'seq_category_main', 'start_year', 'sector', 'propensity_score'] + MAIN_COVARIATES + MAIN_OUTCOMES
            ],
        ],
        ignore_index=True,
    ).sort_values(['match_id', 'match_role'])

    pd.DataFrame(balance_rows).to_csv(config.PSM_BALANCE_TABLE_PATH, index=False)
    pd.DataFrame(effect_rows).to_csv(config.PSM_SEQUENCE_EFFECT_PATH, index=False)
    matched.to_csv(config.PSM_MATCHED_PAIRS_PATH, index=False)
    return pd.DataFrame(effect_rows), pd.DataFrame(balance_rows)


def run_permutation(df):
    work = _analysis_frame(df)
    treated = work[work['is_s_to_h'] == 1].copy()
    control = work[work['is_s_to_h'] == 0].copy()
    rows = []
    dist_rows = []
    for outcome in MAIN_OUTCOMES:
        t = treated[outcome].to_numpy()
        c = control[outcome].to_numpy()
        result = permutation_test(
            (t, c),
            statistic=lambda a, b: np.mean(a) - np.mean(b),
            n_resamples=50000,
            alternative='greater',
            random_state=config.RANDOM_SEED,
        )
        rows.append(
            {
                'analysis': 'raw_permutation',
                'outcome': outcome,
                'treated_n': len(t),
                'control_n': len(c),
                'treated_mean': np.mean(t),
                'control_mean': np.mean(c),
                'observed_diff': np.mean(t) - np.mean(c),
                'permutation_p': result.pvalue,
                'n_resamples': 50000,
            }
        )
        if getattr(result, 'null_distribution', None) is not None:
            for value in result.null_distribution:
                dist_rows.append({'analysis': 'raw_permutation', 'outcome': outcome, 'null_mean_diff': value})

    pd.DataFrame(rows).to_csv(config.PERMUTATION_SEQUENCE_EFFECT_PATH, index=False)
    pd.DataFrame(dist_rows).to_csv(config.PERMUTATION_SEQUENCE_DIST_PATH, index=False)
    return pd.DataFrame(rows)


def run_spatiotemporal(df):
    work = df.copy()
    work['period'] = pd.cut(work['start_year'], bins=PERIOD_BINS, labels=PERIOD_LABELS, right=False, include_lowest=True)
    seq_by_period = (
    work.groupby(['period', 'seq_category_main'], observed=False)
        .size()
        .reset_index(name='n_events')
    )
    seq_by_sector = (
        work.groupby(['sector', 'seq_category_main'])
        .size()
        .reset_index(name='n_events')
    )
    s_to_h_by_period = (
    work.groupby('period', observed=False)
        .agg(total_events=('reef_name', 'size'), s_to_h_events=('is_s_to_h', 'sum'))
        .reset_index()
    )
    s_to_h_by_period['s_to_h_fraction'] = s_to_h_by_period['s_to_h_events'] / s_to_h_by_period['total_events']
    s_to_h_by_sector = (
        work.groupby('sector')
        .agg(total_events=('reef_name', 'size'), s_to_h_events=('is_s_to_h', 'sum'))
        .reset_index()
    )
    s_to_h_by_sector['s_to_h_fraction'] = s_to_h_by_sector['s_to_h_events'] / s_to_h_by_sector['total_events']

    period_table = pd.crosstab(work['period'], work['is_s_to_h'])
    sector_table = pd.crosstab(work['sector'], work['is_s_to_h'])
    seq_period_table = pd.crosstab(work['period'], work['seq_category_main'])

    period_chi2 = chi2_contingency(period_table)
    sector_chi2 = chi2_contingency(sector_table)
    seq_period_chi2 = chi2_contingency(seq_period_table)
    year_rho, year_p = spearmanr(work['start_year'], work['is_s_to_h'])

    summary = pd.DataFrame(
        [
            {
                'analysis': 's_to_h_by_period',
                'statistic': period_chi2[0],
                'p_value': period_chi2[1],
                'degrees_freedom': period_chi2[2],
                'note': 'Chi-square test of S_to_H frequency across fixed periods',
            },
            {
                'analysis': 's_to_h_by_sector',
                'statistic': sector_chi2[0],
                'p_value': sector_chi2[1],
                'degrees_freedom': sector_chi2[2],
                'note': 'Chi-square test of S_to_H frequency across sectors',
            },
            {
                'analysis': 'sequence_composition_by_period',
                'statistic': seq_period_chi2[0],
                'p_value': seq_period_chi2[1],
                'degrees_freedom': seq_period_chi2[2],
                'note': 'Chi-square test of full sequence composition across periods',
            },
            {
                'analysis': 's_to_h_start_year_spearman',
                'statistic': year_rho,
                'p_value': year_p,
                'degrees_freedom': np.nan,
                'note': 'Spearman correlation between start year and S_to_H indicator',
            },
        ]
    )
    counts = pd.concat(
        [
            seq_by_period.assign(group_type='period_sequence'),
            seq_by_sector.assign(group_type='sector_sequence'),
            s_to_h_by_period.rename(columns={'period': 'group_value'}).assign(group_type='period_s_to_h'),
            s_to_h_by_sector.rename(columns={'sector': 'group_value'}).assign(group_type='sector_s_to_h'),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(config.SPATIOTEMPORAL_TREND_SUMMARY_PATH, index=False)
    counts.to_csv(config.SPATIOTEMPORAL_COUNTS_PATH, index=False)
    return summary, counts


def run_gap_memory(df):
    rows = []
    curve_rows = []
    for seq in GAP_GROUPS:
        sub = df[df['seq_category_main'] == seq].copy()
        sub = sub.dropna(subset=['time_gap_years', 'rel_loss'])
        n_obs = len(sub)
        unique_gaps = sub['time_gap_years'].nunique()
        rho, rho_p = (np.nan, np.nan)
        if n_obs >= 3:
            rho, rho_p = spearmanr(sub['time_gap_years'], sub['rel_loss'])
        short = sub[sub['time_gap_years'] <= SHORT_GAP_THRESHOLD]['rel_loss'].to_numpy()
        long = sub[sub['time_gap_years'] > SHORT_GAP_THRESHOLD]['rel_loss'].to_numpy()
        if len(short) > 0 and len(long) > 0:
            perm = permutation_test(
                (short, long),
                statistic=lambda a, b: np.mean(a) - np.mean(b),
                n_resamples=20000,
                alternative='greater',
                random_state=config.RANDOM_SEED,
            )
            short_long_p = perm.pvalue
            diff = np.mean(short) - np.mean(long)
        else:
            short_long_p = np.nan
            diff = np.nan

        if n_obs >= 4 and unique_gaps >= 3:
            lowess_fit = lowess(sub['rel_loss'], sub['time_gap_years'], frac=LOWESS_FRAC, return_sorted=True)
            for x, y in lowess_fit:
                curve_rows.append({'seq_category_main': seq, 'time_gap_years': x, 'smooth_rel_loss': y})

        rows.append(
            {
                'seq_category_main': seq,
                'n_events': n_obs,
                'unique_gap_years': unique_gaps,
                'spearman_rho': rho,
                'spearman_p': rho_p,
                'short_gap_threshold': SHORT_GAP_THRESHOLD,
                'short_gap_n': len(short),
                'long_gap_n': len(long),
                'short_gap_mean_rel_loss': np.mean(short) if len(short) else np.nan,
                'long_gap_mean_rel_loss': np.mean(long) if len(long) else np.nan,
                'short_minus_long_diff': diff,
                'short_long_permutation_p': short_long_p,
            }
        )

    result = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    result.to_csv(config.GAP_MEMORY_MAIN_RESULTS_PATH, index=False)
    curves.to_csv(config.GAP_MEMORY_CURVES_PATH, index=False)
    return result, curves


def main():
    print("=" * 60)
    print("Extended sequence inference: PSM / permutation / spatiotemporal / gap-memory")
    print("=" * 60)
    df = _load_features()
    psm, balance = run_psm(df)
    permutation = run_permutation(df)
    trend, _counts = run_spatiotemporal(df)
    gap, _curves = run_gap_memory(df)

    print("\n[PSM]")
    print(psm.to_string(index=False))
    print("\n[PSM balance]")
    print(balance.to_string(index=False))
    print("\n[Permutation]")
    print(permutation.to_string(index=False))
    print("\n[Spatiotemporal]")
    print(trend.to_string(index=False))
    print("\n[Gap-memory]")
    print(gap.to_string(index=False))


if __name__ == '__main__':
    main()
