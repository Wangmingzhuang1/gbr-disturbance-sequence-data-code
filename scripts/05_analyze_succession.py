# -*- coding: utf-8 -*-
import importlib.util
import os

import numpy as np
import pandas as pd

import config
import sequence_analysis_core as sac


def _build_stage_audit(total_reef_years, event_years, attrition_df, final_df):
    dropped_consumed = int(attrition_df['dropped_consumed'].sum())
    dropped_not_clean = int(attrition_df['dropped_baseline_not_clean'].sum())
    dropped_insufficient = int(attrition_df['dropped_baseline_insufficient_points'].sum())
    dropped_threshold = int(attrition_df['dropped_baseline_below_threshold'].sum())
    dropped_nadir = int(attrition_df['dropped_nadir_missing_data'].sum())
    dropped_label = int(attrition_df['dropped_label_failure'].sum())

    remaining = int(event_years)
    rows = [
        {'stage_code': 'S0', 'stage_label': 'Total reef-years', 'count': int(total_reef_years), 'dropped': 0, 'remaining': int(total_reef_years)},
        {'stage_code': 'S1', 'stage_label': 'Potential starts (event-years)', 'count': int(event_years), 'dropped': 0, 'remaining': int(event_years)},
    ]

    for code, label, dropped in [
        ('S2', 'Disjoint continuity filter', dropped_consumed),
        ('S3', 'Baseline not clean', dropped_not_clean),
        ('S4', 'Baseline insufficient surveys', dropped_insufficient),
        ('S5', 'Baseline below HC threshold', dropped_threshold),
        ('S6', 'Missing nadir response', dropped_nadir),
        ('S7', 'Label failure', dropped_label),
    ]:
        remaining -= dropped
        rows.append({'stage_code': code, 'stage_label': label, 'count': None, 'dropped': int(dropped), 'remaining': int(remaining)})

    rows.append({
        'stage_code': 'S8',
        'stage_label': 'Final retained sequences',
        'count': int(len(final_df)),
        'dropped': 0,
        'remaining': int(len(final_df)),
    })
    rows.append({
        'stage_code': 'S9',
        'stage_label': 'Final retained reefs',
        'count': int(final_df['reef_name'].nunique()),
        'dropped': 0,
        'remaining': int(final_df['reef_name'].nunique()),
    })
    return pd.DataFrame(rows)


def _build_regime_summary(final_df):
    regime = final_df[
        [
            'reef_name',
            'start_year',
            'seq_category_main',
            'baseline_hc',
            'nadir_hc',
            'final_hc',
            'recovery_rate',
            'recovery_pct',
            'nadir_year',
            'final_year',
        ]
    ].copy()
    regime = sac.add_fate_fields(regime)
    regime['regime_shift_status'] = regime['fate_status'] == 'failed'
    regime['recovered_above_baseline'] = regime['final_hc'] >= regime['baseline_hc']
    return regime


def _export_reef_year_sequence_summary():
    script_path = os.path.join(os.path.dirname(__file__), '04_export_reef_year_sequence_summary.py')
    spec = importlib.util.spec_from_file_location('reef_year_sequence_summary_exporter', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.export_reef_year_sequence_summary()


def _build_parameter_audit():
    return pd.DataFrame([
        {'parameter': 'YEAR_START', 'value': config.YEAR_START},
        {'parameter': 'YEAR_END', 'value': config.YEAR_END},
        {'parameter': 'BASELINE_WINDOW', 'value': list(config.BASELINE_WINDOW)},
        {'parameter': 'NADIR_WINDOW', 'value': list(config.NADIR_WINDOW)},
        {'parameter': 'MIN_BASELINE_SURVEYS', 'value': config.MIN_BASELINE_SURVEYS},
        {'parameter': 'MIN_BASELINE_HC', 'value': config.MIN_BASELINE_HC},
        {'parameter': 'LOOKAHEAD_WINDOW', 'value': config.LOOKAHEAD_WINDOW},
        {'parameter': 'RECOVERY_LOOKAHEAD', 'value': config.RECOVERY_LOOKAHEAD},
        {'parameter': 'EXTRACTION_MODE', 'value': config.EXTRACTION_MODE},
    ])


def _build_sequence_consistency_audit(final_df, df_master):
    rows = []
    if final_df.empty:
        return pd.DataFrame(columns=[
            'reef_name', 'start_year', 'target_year', 'anchor_year', 'nadir_year',
            'nadir_hc', 'window_min_hc', 'window_min_year', 'nadir_consistent',
            'rel_loss', 'rel_loss_recomputed', 'rel_loss_consistent',
        ])

    for _, row in final_df.iterrows():
        reef_name = row['reef_name']
        anchor_year = int(row['target_year']) if pd.notna(row['target_year']) else int(row['start_year'])
        valid_years = [anchor_year + offset for offset in config.NADIR_WINDOW]
        matches = df_master[
            (df_master['reef_name'] == reef_name) &
            (df_master['year'].isin(valid_years))
        ].dropna(subset=['HC_cover']).copy()
        if matches.empty:
            window_min_hc = np.nan
            window_min_year = np.nan
            nadir_consistent = False
        else:
            min_row = matches.loc[matches['HC_cover'].idxmin()]
            window_min_hc = float(min_row['HC_cover'])
            window_min_year = int(min_row['year'])
            nadir_consistent = np.isclose(float(row['nadir_hc']), window_min_hc, atol=1e-12) and int(row['nadir_year']) == window_min_year

        if pd.notna(row['baseline_hc']) and float(row['baseline_hc']) > 0:
            rel_loss_recomputed = float(np.clip((float(row['baseline_hc']) - float(row['nadir_hc'])) / float(row['baseline_hc']), -1.0, 1.0))
            rel_loss_consistent = np.isclose(float(row['rel_loss']), rel_loss_recomputed, atol=1e-12)
        else:
            rel_loss_recomputed = np.nan
            rel_loss_consistent = pd.isna(row['rel_loss'])

        rows.append({
            'reef_name': reef_name,
            'start_year': int(row['start_year']),
            'target_year': int(row['target_year']) if pd.notna(row['target_year']) else np.nan,
            'anchor_year': anchor_year,
            'nadir_year': int(row['nadir_year']),
            'nadir_hc': float(row['nadir_hc']),
            'window_min_hc': window_min_hc,
            'window_min_year': window_min_year,
            'nadir_consistent': bool(nadir_consistent),
            'rel_loss': float(row['rel_loss']),
            'rel_loss_recomputed': rel_loss_recomputed,
            'rel_loss_consistent': bool(rel_loss_consistent),
        })
    return pd.DataFrame(rows)


def main():
    print(f"PIPELINE: DISTURBANCE SEQUENCE EXTRACTION ({config.EXTRACTION_MODE})")
    print(
        f"Baseline: {min(config.BASELINE_WINDOW)}-{max(config.BASELINE_WINDOW)} years, "
        f"N>={config.MIN_BASELINE_SURVEYS}, HC>={config.MIN_BASELINE_HC:g}"
    )
    print(
        f"Nadir window: {list(config.NADIR_WINDOW)}, "
        f"Lookahead: {config.LOOKAHEAD_WINDOW}, Recovery lookahead: {config.RECOVERY_LOOKAHEAD}"
    )

    if not os.path.exists(config.MASTER_MATRIX_PATH):
        print(f"Error: {config.MASTER_MATRIX_PATH} not found. Run 03_merge_eco_dist.py first.")
        return

    df_master = pd.read_csv(config.MASTER_MATRIX_PATH)
    total_reef_years = len(df_master)
    event_years = int(((df_master['has_storm'] == 1) | (df_master['has_heatwave'] == 1)).sum())
    print(
        f"Master matrix: {len(df_master)} rows, {df_master['reef_name'].nunique()} reefs, "
        f"years {int(df_master['year'].min())}-{int(df_master['year'].max())}"
    )

    extracted_frames = []
    attrition_rows = []
    for reef_name, df_reef in df_master.groupby('reef_name'):
        df_seq, tally = sac.extract_reef_sequences(
            df_reef.sort_values('year'),
            rule='first_event',
            lookahead_years=config.LOOKAHEAD_WINDOW,
            mode=config.EXTRACTION_MODE,
        )
        tally['reef_name'] = reef_name
        attrition_rows.append(tally)
        if not df_seq.empty:
            extracted_frames.append(df_seq)

    final_df = pd.concat(extracted_frames, ignore_index=True) if extracted_frames else pd.DataFrame(columns=['reef_name'])
    if not final_df.empty:
        final_df = sac.add_fate_fields(final_df)
    attrition_df = pd.DataFrame(attrition_rows)
    audit_df = _build_stage_audit(total_reef_years, event_years, attrition_df, final_df)
    regime_df = _build_regime_summary(final_df) if not final_df.empty else pd.DataFrame()
    parameter_audit_df = _build_parameter_audit()
    consistency_df = _build_sequence_consistency_audit(final_df, df_master)

    inconsistent = consistency_df[
        (~consistency_df['nadir_consistent']) | (~consistency_df['rel_loss_consistent'])
    ].copy()

    final_df.to_csv(config.EXTRACTED_SEQS_PATH, index=False)
    audit_df.to_csv(config.ATTRITION_AUDIT_PATH, index=False)
    attrition_df.to_csv(config.ATTRITION_BY_REEF_PATH, index=False)
    regime_df.to_csv(config.REGIME_SHIFT_PATH, index=False)
    parameter_audit_df.to_csv(config.PIPELINE_PARAMETER_AUDIT_PATH, index=False)
    consistency_df.to_csv(config.SEQUENCE_CONSISTENCY_AUDIT_PATH, index=False)
    _export_reef_year_sequence_summary()

    if not inconsistent.empty:
        raise RuntimeError(
            f"Sequence consistency check failed for {len(inconsistent)} rows. "
            f"See {config.SEQUENCE_CONSISTENCY_AUDIT_PATH}"
        )

    print(f"Retained {len(final_df)} sequences from {final_df['reef_name'].nunique() if not final_df.empty else 0} reefs.")
    print("\n--- Stage-wise Attrition ---")
    print(audit_df[['stage_code', 'stage_label', 'dropped', 'remaining']].to_string(index=False))
    print(f"\nSaved parameter audit to {config.PIPELINE_PARAMETER_AUDIT_PATH}")
    print(f"Saved sequence consistency audit to {config.SEQUENCE_CONSISTENCY_AUDIT_PATH}")


if __name__ == '__main__':
    main()
