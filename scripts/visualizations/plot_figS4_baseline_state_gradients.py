# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modeling_core import filter_downstream_analysis_sample

from style_config import (
    FS_LEGEND,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    save_publication_figure,
    set_square_panel,
)


FEATURES = config.FINAL_FEATURES_PATH
OUTPUT = config.FIGS4_PATH
DISTURBANCE_ORDER = ['Heat-linked', 'Storm-linked']
DISTURBANCE_COLORS = {
    'Heat-linked': '#ff7f0e',
    'Storm-linked': '#1f77b4',
}
DISTURBANCE_MARKERS = {
    'Heat-linked': 'o',
    'Storm-linked': 's',
}


def _scatter_by_group(ax, data, x_col, y_col):
    for group in DISTURBANCE_ORDER:
        part = data[data['disturbance_group'] == group]
        if part.empty:
            continue
        ax.scatter(
            part[x_col],
            part[y_col],
            s=42,
            marker=DISTURBANCE_MARKERS[group],
            c=DISTURBANCE_COLORS[group],
            edgecolors='white',
            linewidths=0.55,
            alpha=0.72,
            label=group,
            zorder=3,
        )


def _panel_a(ax, df):
    plot_df = df.dropna(subset=['baseline_hc', 'drop_abs']).copy()
    _scatter_by_group(ax, plot_df, 'baseline_hc', 'drop_abs')
    sns.regplot(data=plot_df, x='baseline_hc', y='drop_abs', lowess=True, scatter=False, ci=None, color='#2f2f2f', line_kws={'linewidth': 1.8}, ax=ax)
    ax.axhline(5.0, color='#9f1d35', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Baseline hard coral cover (%)')
    ax.set_ylabel('Acute coral-cover loss at nadir (%)')
    clean_axis(ax)
    set_square_panel(ax)


def _panel_b(ax, df):
    plot_df = df[(df['impact_flag'] == 1) & (df['recovery_observed_flag'] == 1)].dropna(subset=['baseline_hc', 'recovery_frac_loss']).copy()
    _scatter_by_group(ax, plot_df, 'baseline_hc', 'recovery_frac_loss')
    sns.regplot(data=plot_df, x='baseline_hc', y='recovery_frac_loss', lowess=True, scatter=False, ci=None, color='#2f2f2f', line_kws={'linewidth': 1.8}, ax=ax)
    ax.axhline(0.5, color='#9f1d35', linestyle='--', linewidth=1.0)
    ax.axhline(0.0, color='#7b8085', linestyle=':', linewidth=0.9)
    ax.set_xlabel('Baseline hard coral cover (%)')
    ax.set_ylabel('Recovery fraction of loss')
    clean_axis(ax)
    set_square_panel(ax)


def plot_figS4():
    apply_pub_style()
    df = filter_downstream_analysis_sample(pd.read_csv(FEATURES))
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.78))
    _panel_a(axes[0], df)
    _panel_b(axes[1], df)
    add_panel_label(axes[0], 'A')
    add_panel_label(axes[1], 'B')

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[1].legend(by_label.values(), by_label.keys(), title='', frameon=False, loc='upper left', fontsize=FS_LEGEND, bbox_to_anchor=(0.02, 0.98))

    apply_layout(fig, 'supplement_landscape', left=0.10, right=0.985, top=0.93, bottom=0.18)
    fig.subplots_adjust(wspace=0.24)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_figS4()
