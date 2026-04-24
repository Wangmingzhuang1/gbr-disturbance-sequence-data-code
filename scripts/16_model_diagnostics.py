# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd

import config

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
if VIS_DIR not in sys.path:
    sys.path.append(VIS_DIR)

from plot_figS3_ols_diagnostics import plot_figS3


def build_negative_rel_loss_audit():
    df = pd.read_csv(config.EXTRACTED_SEQS_PATH)

    anchor_year = df['target_year'].where(df['target_year'].notna(), df['start_year'])
    df['nadir_lag'] = df['nadir_year'] - anchor_year
    df['rel_loss_sign'] = np.where(df['rel_loss'] < 0, 'negative', 'non_negative')

    rows = []
    for sign, sub in df.groupby('rel_loss_sign', dropna=False):
        lag_counts = sub['nadir_lag'].value_counts().to_dict()
        rows.append(
            {
                'rel_loss_sign': sign,
                'n_events': int(len(sub)),
                'mean_rel_loss': float(sub['rel_loss'].mean()),
                'mean_nadir_lag': float(sub['nadir_lag'].mean()),
                'median_nadir_lag': float(sub['nadir_lag'].median()),
                'lag_1_n': int(lag_counts.get(1.0, 0)),
                'lag_2_n': int(lag_counts.get(2.0, 0)),
                'lag_3_n': int(lag_counts.get(3.0, 0)),
                'mean_baseline_hc': float(sub['baseline_hc'].mean()),
                'median_baseline_hc': float(sub['baseline_hc'].median()),
            }
        )

    out = pd.DataFrame(rows).sort_values('rel_loss_sign')
    out.to_csv(config.NEG_REL_LOSS_AUDIT_PATH, index=False)
    print(f"Saved {config.NEG_REL_LOSS_AUDIT_PATH}")
    return out


def build_negative_rel_loss_baseline_bins():
    df = pd.read_csv(config.EXTRACTED_SEQS_PATH)
    df = df[df['baseline_hc'] >= config.MIN_BASELINE_HC].copy()

    bins = [config.MIN_BASELINE_HC, 10, 20, 30, np.inf]
    labels = [f'{int(config.MIN_BASELINE_HC)}-<10', '10-<20', '20-<30', '30+']
    df['baseline_bin'] = pd.cut(df['baseline_hc'], bins=bins, labels=labels, right=False, include_lowest=True)

    out = (
        df.groupby('baseline_bin', observed=False)
        .agg(
            n_events=('rel_loss', 'size'),
            negative_n=('rel_loss', lambda x: int((x < 0).sum())),
            negative_share=('rel_loss', lambda x: float((x < 0).mean())),
            mean_rel_loss=('rel_loss', 'mean'),
            median_rel_loss=('rel_loss', 'median'),
        )
        .reset_index()
    )

    baseline_path = config.NEG_REL_LOSS_BASELINE_BINS_PATH
    out.to_csv(baseline_path, index=False)
    print(f"Saved {baseline_path}")
    return out


def run():
    if not os.path.exists(config.EXTRACTED_SEQS_PATH):
        raise FileNotFoundError(config.EXTRACTED_SEQS_PATH)
    if not os.path.exists(config.FINAL_FEATURES_PATH):
        raise FileNotFoundError(config.FINAL_FEATURES_PATH)

    audit = build_negative_rel_loss_audit()
    baseline_bins = build_negative_rel_loss_baseline_bins()
    print(audit.to_string(index=False))
    print(baseline_bins.to_string(index=False))
    plot_figS3()


if __name__ == '__main__':
    run()
