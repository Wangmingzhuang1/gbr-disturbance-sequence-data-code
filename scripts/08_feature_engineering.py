# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import config
import sequence_analysis_core as sac


def add_centered_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors='coerce')
        if series.notna().any():
            df[f'{column}_c'] = series - series.mean()
        else:
            df[f'{column}_c'] = np.nan


def build_features():
    print("=" * 60)
    print("Feature engineering for the finalized disturbance-sequence pipeline")
    print("=" * 60)

    df = pd.read_csv(config.EXTRACTED_SEQS_PATH)
    df = sac.add_fate_fields(df)
    df = sac.add_disturbance_group(df)
    df = sac.add_baseline_hc_group(df)

    juv_valid = df['baseline_juv'] > 0
    raw_juv_loss = np.where(
        juv_valid,
        (df['baseline_juv'] - df['nadir_juv']) / df['baseline_juv'],
        np.nan,
    )
    df['juv_change_nadir'] = np.where(
        juv_valid,
        ((df['nadir_juv'] - df['baseline_juv']) / df['baseline_juv']).clip(-1.0, 5.0),
        np.nan,
    )
    df['juv_loss'] = np.where(
        juv_valid,
        np.clip(raw_juv_loss, 0.0, 1.0),
        np.nan,
    )
    df['juv_recovery_rate'] = np.where(
        juv_valid,
        ((df['final_juv'] - df['nadir_juv']) / df['baseline_juv']).clip(-1.0, 5.0),
        np.nan,
    )
    df['baseline_span_years'] = df['start_year'] - df['baseline_start_year']
    df['baseline_recent_gap_years'] = df['start_year'] - df['baseline_end_year']
    df['baseline_search_window_min_years'] = min(config.BASELINE_WINDOW)
    df['baseline_search_window_max_years'] = max(config.BASELINE_WINDOW)

    df['seq_category_model'] = df['seq_category_main']
    df['is_s_to_h'] = (df['seq_category_main'] == 'S_to_H').astype(int)
    df['is_h_to_s'] = (df['seq_category_main'] == 'H_to_S').astype(int)
    df['is_isolated_heatwave'] = (df['seq_category_main'] == 'Isolated_Heatwave').astype(int)
    df['log_gap'] = np.log1p(df['gap_years'].fillna(0))
    df['gap_squared'] = df['gap_years'] ** 2

    df['sector'] = df['sector'].fillna('Unknown').astype(str)
    if df['region_lat'].notna().any():
        df['region_lat'] = df['region_lat'].fillna(df['region_lat'].mean())
    else:
        df['region_lat'] = -18.0

    add_centered_columns(
        df,
        [
            'storm_exposure_seq',
            'heat_exposure_seq',
            'baseline_hc',
            'gap_years',
            'start_year',
            'region_lat',
            'cots_lag_3yr',
            'initial_cots',
            'initial_herbivore',
        ],
    )

    df.to_csv(config.FINAL_FEATURES_PATH, index=False)

    stats_df = df.groupby('seq_category_main').agg(
        rel_loss_mean=('rel_loss', 'mean'),
        rel_loss_std=('rel_loss', 'std'),
        rel_loss_count=('rel_loss', 'count'),
        baseline_hc_mean=('baseline_hc', 'mean'),
        storm_exposure_seq_mean=('storm_exposure_seq', 'mean'),
        heat_exposure_seq_mean=('heat_exposure_seq', 'mean'),
        gap_years_mean=('gap_years', 'mean'),
    ).round(3)
    stats_df.to_csv(config.DESCRIPTIVE_STATS_PATH)

    print(f"Loaded {len(df)} sequences from {df['reef_name'].nunique()} reefs")
    print(df['seq_category_main'].value_counts().to_string())
    print(f"\nSaved engineered features to {config.FINAL_FEATURES_PATH}")


if __name__ == '__main__':
    build_features()
