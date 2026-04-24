# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from style_config import (
    FS_ANNOT,
    FS_TICK,
    SEQ_LABELS,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    ordered_sequences,
    save_publication_figure,
    seq_color,
    set_percent_axis,
    set_square_panel,
)


def plot_figS5():
    apply_pub_style()
    df_margin = pd.read_csv(config.ADJUSTED_MEANS_PATH)
    df_labels = ordered_sequences(df_margin['seq_category'].values)
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 4.02), gridspec_kw={'width_ratios': [1.1, 0.9]})

    ax1 = axes[0]
    for i, cat in enumerate(df_labels):
        row = df_margin[df_margin['seq_category'] == cat].iloc[0]
        y_pos = len(df_labels) - 1 - i
        ax1.plot([row['ci_lo'], row['ci_hi']], [y_pos, y_pos], color=seq_color(cat), linewidth=4.4, alpha=0.62, solid_capstyle='round')
        ax1.scatter(row['adjusted_mean'], y_pos, color='#212529', s=56, zorder=3, edgecolor='white', linewidth=0.8)

    ax1.axvline(0, color='#4a4e54', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_yticks(range(len(df_labels)))
    ax1.set_yticklabels([SEQ_LABELS[c] for c in df_labels[::-1]], fontsize=FS_TICK)
    ax1.set_xlabel('Adjusted mean coral loss')
    set_percent_axis(ax1, 'x', 1.0)
    clean_axis(ax1)
    set_square_panel(ax1)
    add_panel_label(ax1, 'A')

    ax2 = axes[1]
    res_df = pd.read_csv(config.REGRESSION_MAIN_PATH)
    term_col = 'Unnamed: 0' if 'Unnamed: 0' in res_df.columns else res_df.columns[0]
    res_df = res_df.rename(columns={term_col: 'term'})
    covars = ['start_year_c', 'baseline_hc_c', 'storm_exposure_seq_c', 'heat_exposure_seq_c']
    cov_labels = ['Start year', 'Baseline cover', 'Storm exp.', 'Heat exp.']
    plot_df = res_df[res_df['term'].isin(covars)].copy()
    plot_df['term'] = pd.Categorical(plot_df['term'], categories=covars, ordered=True)
    plot_df = plot_df.sort_values('term')

    for i, covar in enumerate(covars):
        if covar not in plot_df['term'].values:
            continue
        row = plot_df[plot_df['term'] == covar].iloc[0]
        y_pos = len(covars) - 1 - i
        bar_color = '#d62728' if row['Coefficient'] >= 0 else '#1f77b4'
        ax2.barh(y_pos, row['Coefficient'], color=bar_color, edgecolor='white', alpha=0.82, height=0.5, linewidth=0.8)
        ax2.errorbar(
            row['Coefficient'],
            y_pos,
            xerr=[[row['Coefficient'] - row['CI_lower']], [row['CI_upper'] - row['Coefficient']]],
            fmt='none',
            color='#212529',
            capsize=2,
            linewidth=1.0,
        )
        pval = row['P-value_FDR']
        sig = '**' if pval < 0.01 else ('*' if pval < 0.05 else '')
        if sig:
            ci_span = abs(row['CI_upper'] - row['CI_lower'])
            offset = max(ci_span * 0.18, 0.015)
            offset = offset if row['Coefficient'] >= 0 else -offset
            align = 'left' if row['Coefficient'] >= 0 else 'right'
            x_text = max(row['CI_upper'], row['Coefficient']) + offset if row['Coefficient'] >= 0 else min(row['CI_lower'], row['Coefficient']) + offset
            ax2.text(x_text, y_pos, sig, va='center', ha=align, fontsize=FS_ANNOT, fontweight='bold', color='#4a4e54')

    ax2.axvline(0, color='#4a4e54', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_yticks(range(len(covars)))
    ax2.set_yticklabels(cov_labels[::-1], fontsize=FS_TICK)
    ax2.set_xlabel('Regression coefficient')
    clean_axis(ax2)
    set_square_panel(ax2)
    add_panel_label(ax2, 'B', x=-0.10)

    apply_layout(fig, 'supplement_landscape', left=0.17, right=0.985, top=0.93, bottom=0.18)
    fig.subplots_adjust(wspace=0.24)
    save_publication_figure(fig, config.OLS_COEFFICIENT_FIG_PATH)
    plt.close()
    print(f"Figure S5 saved: {config.OLS_COEFFICIENT_FIG_PATH}")


if __name__ == '__main__':
    plot_figS5()
