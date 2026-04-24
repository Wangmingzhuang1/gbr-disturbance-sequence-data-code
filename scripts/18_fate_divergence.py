# -*- coding: utf-8 -*-
"""
Fate-divergence screening for the finalized disturbance-sequence paper.

This script preserves the fixed core chain and adds a recovery/failed split
only after substantial coral-cover impact has already occurred.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config
import sequence_analysis_core as sac
from modeling_core import filter_downstream_analysis_sample


FATE_ORDER = ['non_impact', 'recovery_unobserved', 'failed', 'recovered']
DISTURBANCE_ORDER = ['Heat-linked', 'Storm-linked']
BASELINE_GROUP_ORDER = ['Low baseline HC', 'High baseline HC']


def load_features():
    return pd.read_csv(config.EXTRACTED_SEQS_PATH)


def _add_centered_columns(df, columns):
    work = df.copy()
    for column in columns:
        if column not in work.columns:
            continue
        series = pd.to_numeric(work[column], errors='coerce')
        work[f'{column}_c'] = series - series.mean() if series.notna().any() else np.nan
    return work


def prepare_fate_features(df):
    work = sac.add_fate_fields(df.copy())
    work = sac.add_disturbance_group(work)
    work = sac.add_baseline_hc_group(work)
    work = _add_centered_columns(work, ['baseline_hc'])
    return filter_downstream_analysis_sample(work)


def _strict_start_years_by_reef(df):
    grouped = df.groupby('reef_name')['start_year']
    return {reef_name: set(sub.astype(int).tolist()) for reef_name, sub in grouped}


def build_supplement_sequences(strict_df):
    master = sac.load_master_matrix()
    strict_start_years = _strict_start_years_by_reef(strict_df)
    eligible_start_years_by_reef = {}

    for reef_name, df_reef in master.groupby('reef_name'):
        event_years = set(
            df_reef.loc[
                (df_reef['has_storm'] == 1) | (df_reef['has_heatwave'] == 1),
                'year',
            ].dropna().astype(int).tolist()
        )
        eligible = event_years - strict_start_years.get(reef_name, set())
        if eligible:
            eligible_start_years_by_reef[reef_name] = eligible

    supplement = sac.extract_sequences(
        master,
        lookahead_years=config.LOOKAHEAD_WINDOW,
        rule='first_event',
        mode=config.EXTRACTION_MODE,
        allow_baseline_conflicts=True,
        eligible_start_years_by_reef=eligible_start_years_by_reef,
    )
    if supplement.empty:
        return supplement

    strict_keys = set(zip(strict_df['reef_name'], strict_df['start_year'].astype(int)))
    keep_mask = [
        (reef_name, int(start_year)) not in strict_keys
        for reef_name, start_year in zip(supplement['reef_name'], supplement['start_year'])
    ]
    return supplement.loc[keep_mask].reset_index(drop=True)


def build_summary(df, sample_scope):
    impacted = df[df['impact_flag'] == 1].copy()
    observed = impacted[impacted['recovery_observed_flag'] == 1].copy()

    rows = [
        {'summary_type': 'stage', 'group_type': 'overall', 'group_value': 'retained', 'n_events': int(len(df)), 'n_reefs': int(df['reef_name'].nunique())},
        {'summary_type': 'stage', 'group_type': 'overall', 'group_value': 'impacted', 'n_events': int(len(impacted)), 'n_reefs': int(impacted['reef_name'].nunique())},
        {'summary_type': 'stage', 'group_type': 'overall', 'group_value': 'recovery_observed', 'n_events': int(len(observed)), 'n_reefs': int(observed['reef_name'].nunique())},
        {'summary_type': 'stage', 'group_type': 'overall', 'group_value': 'recovery_unobserved', 'n_events': int((impacted['recovery_observed_flag'] == 0).sum()), 'n_reefs': int(impacted.loc[impacted['recovery_observed_flag'] == 0, 'reef_name'].nunique())},
        {'summary_type': 'stage', 'group_type': 'overall', 'group_value': 'recovered', 'n_events': int((observed['recovered_flag'] == 1).sum()), 'n_reefs': int(observed.loc[observed['recovered_flag'] == 1, 'reef_name'].nunique())},
        {'summary_type': 'stage', 'group_type': 'overall', 'group_value': 'failed', 'n_events': int((observed['recovered_flag'] == 0).sum()), 'n_reefs': int(observed.loc[observed['recovered_flag'] == 0, 'reef_name'].nunique())},
    ]

    for fate in FATE_ORDER:
        sub = df[df['fate_status'] == fate]
        rows.append({
            'summary_type': 'fate_status',
            'group_type': 'overall',
            'group_value': fate,
            'n_events': int(len(sub)),
            'n_reefs': int(sub['reef_name'].nunique()),
        })

    for group_type in ['baseline_hc_group', 'disturbance_group']:
        for group_value, sub in df.groupby(group_type):
            rows.append({
                'summary_type': 'retained_by_group',
                'group_type': group_type,
                'group_value': group_value,
                'n_events': int(len(sub)),
                'n_reefs': int(sub['reef_name'].nunique()),
            })

            impacted_sub = sub[sub['impact_flag'] == 1]
            observed_sub = impacted_sub[impacted_sub['recovery_observed_flag'] == 1]
            rows.append({
                'summary_type': 'impacted_by_group',
                'group_type': group_type,
                'group_value': group_value,
                'n_events': int(len(impacted_sub)),
                'n_reefs': int(impacted_sub['reef_name'].nunique()),
                'recovery_observed_n': int(len(observed_sub)),
                'recovered_n': int((observed_sub['recovered_flag'] == 1).sum()),
                'failed_n': int((observed_sub['recovered_flag'] == 0).sum()),
                'recovered_fraction': float(observed_sub['recovered_flag'].mean()) if not observed_sub.empty else np.nan,
            })

    out = pd.DataFrame(rows)
    out.insert(0, 'sample_scope', sample_scope)
    return out


def build_stratified(df, sample_scope):
    sub = df[
        (df['impact_flag'] == 1) &
        (df['recovery_observed_flag'] == 1)
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for baseline_group in BASELINE_GROUP_ORDER:
        for disturbance_group in DISTURBANCE_ORDER:
            part = sub[
                (sub['baseline_hc_group'] == baseline_group) &
                (sub['disturbance_group'] == disturbance_group)
            ].copy()
            rows.append({
                'baseline_hc_group': baseline_group,
                'disturbance_group': disturbance_group,
                'n_events': int(len(part)),
                'n_reefs': int(part['reef_name'].nunique()),
                'recovered_n': int((part['recovered_flag'] == 1).sum()),
                'failed_n': int((part['recovered_flag'] == 0).sum()),
                'recovered_fraction': float(part['recovered_flag'].mean()) if not part.empty else np.nan,
                'mean_recovery_frac_loss': float(part['recovery_frac_loss'].mean()) if not part.empty else np.nan,
                'median_recovery_frac_loss': float(part['recovery_frac_loss'].median()) if not part.empty else np.nan,
            })
    out = pd.DataFrame(rows)
    out.insert(0, 'sample_scope', sample_scope)
    return out


def fit_recovery_model(df, sample_scope):
    sub = df[
        (df['impact_flag'] == 1) &
        (df['recovery_observed_flag'] == 1)
    ].copy()
    required = ['recovered_flag', 'baseline_hc_c', 'disturbance_group', 'reef_name']
    sub = sub.dropna(subset=required)
    if sub.empty:
        return pd.DataFrame()

    formula = (
        "recovered_flag ~ "
        "C(disturbance_group, Treatment(reference='Heat-linked')) + baseline_hc_c"
    )
    model = smf.glm(
        formula=formula,
        data=sub,
        family=sm.families.Binomial(),
    ).fit(cov_type='cluster', cov_kwds={'groups': sub['reef_name']})

    conf = model.conf_int()
    out = pd.DataFrame({
        'term': model.params.index,
        'Coefficient': model.params.values,
        'StdErr': model.bse.values,
        'P_value': model.pvalues.values,
        'CI_lower': conf[0].values,
        'CI_upper': conf[1].values,
        'OddsRatio': np.exp(model.params.values),
        'OR_CI_lower': np.exp(conf[0].values),
        'OR_CI_upper': np.exp(conf[1].values),
    })
    out['model_name'] = 'fate_recovery_glm'
    out['sample_scope'] = sample_scope
    out['n_events'] = int(len(sub))
    out['n_reefs'] = int(sub['reef_name'].nunique())
    return out


def main():
    raw_strict_df = load_features()
    strict_df = prepare_fate_features(raw_strict_df)
    supplement_seq = build_supplement_sequences(raw_strict_df)
    supplement_df = prepare_fate_features(supplement_seq) if not supplement_seq.empty else pd.DataFrame()
    combined_df = (
        pd.concat([strict_df, supplement_df], ignore_index=True)
        if not supplement_df.empty else strict_df.copy()
    )

    build_summary(strict_df, 'strict').to_csv(config.FATE_SUMMARY_PATH, index=False)
    build_stratified(strict_df, 'strict').to_csv(config.FATE_STRATIFIED_PATH, index=False)
    fit_recovery_model(strict_df, 'strict').to_csv(config.FATE_MODEL_PATH, index=False)

    supplement_outputs = [
        build_summary(supplement_df, 'supplement_only') if not supplement_df.empty else pd.DataFrame(),
        build_summary(combined_df, 'strict_plus_supplement'),
    ]
    stratified_outputs = [
        build_stratified(supplement_df, 'supplement_only') if not supplement_df.empty else pd.DataFrame(),
        build_stratified(combined_df, 'strict_plus_supplement'),
    ]
    model_outputs = [
        fit_recovery_model(supplement_df, 'supplement_only') if not supplement_df.empty else pd.DataFrame(),
        fit_recovery_model(combined_df, 'strict_plus_supplement'),
    ]

    pd.concat([df for df in supplement_outputs if not df.empty], ignore_index=True).to_csv(
        config.FATE_SUPPLEMENT_SUMMARY_PATH,
        index=False,
    )
    pd.concat([df for df in stratified_outputs if not df.empty], ignore_index=True).to_csv(
        config.FATE_SUPPLEMENT_STRATIFIED_PATH,
        index=False,
    )
    pd.concat([df for df in model_outputs if not df.empty], ignore_index=True).to_csv(
        config.FATE_SUPPLEMENT_MODEL_PATH,
        index=False,
    )

    if not supplement_df.empty:
        supplement_df.to_csv(config.FATE_SUPPLEMENT_SEQS_PATH, index=False)
    else:
        pd.DataFrame().to_csv(config.FATE_SUPPLEMENT_SEQS_PATH, index=False)

    print("Fate divergence screening complete.")
    print(f"Summary: {config.FATE_SUMMARY_PATH}")
    print(f"Stratified: {config.FATE_STRATIFIED_PATH}")
    print(f"Recovery model: {config.FATE_MODEL_PATH}")
    print(f"Supplement sequences: {config.FATE_SUPPLEMENT_SEQS_PATH}")
    print(f"Supplement summary: {config.FATE_SUPPLEMENT_SUMMARY_PATH}")
    print(f"Supplement stratified: {config.FATE_SUPPLEMENT_STRATIFIED_PATH}")
    print(f"Supplement recovery model: {config.FATE_SUPPLEMENT_MODEL_PATH}")


if __name__ == '__main__':
    main()
