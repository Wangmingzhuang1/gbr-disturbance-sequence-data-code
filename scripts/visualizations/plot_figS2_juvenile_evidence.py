# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modeling_core import filter_downstream_analysis_sample

from style_config import (
    FS_ANNOT,
    FS_LEGEND,
    FS_TICK,
    SEQ_LABELS,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    ordered_sequences,
    save_publication_figure,
    seq_color,
    set_square_panel,
)


FEATURES = config.FINAL_FEATURES_PATH
OUTPUT = config.FIGS2_PATH
X_LABELS = {
    'S_to_H': 'S->H',
    'S_to_S': 'S->S',
    'Isolated_Storm': 'Isolated\nstorm',
    'Concurrent': 'Concurrent',
    'Isolated_Heatwave': 'Isolated\nheatwave',
    'H_to_H': 'H->H',
}


def _prep():
    return filter_downstream_analysis_sample(pd.read_csv(FEATURES))


def _panel_a(ax, df):
    sub = df.dropna(subset=['juv_change_nadir']).copy()
    order = ordered_sequences(sub['seq_category_main'].unique())
    sub = sub[sub['seq_category_main'].isin(order)].copy()
    palette = {seq: seq_color(seq) for seq in order}

    sns.boxplot(
        data=sub,
        x='seq_category_main',
        y='juv_change_nadir',
        order=order,
        palette=palette,
        hue='seq_category_main',
        dodge=False,
        legend=False,
        width=0.52,
        fliersize=0,
        linewidth=0.8,
        ax=ax,
        boxprops={'alpha': 0.78},
    )
    sns.stripplot(
        data=sub,
        x='seq_category_main',
        y='juv_change_nadir',
        order=order,
        color='#2f2f2f',
        alpha=0.52,
        size=3.1,
        jitter=0.16,
        ax=ax,
    )
    ax.axhline(0, color='#4a4e54', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Juvenile net change')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([X_LABELS[s] for s in order], fontsize=FS_TICK)
    clean_axis(ax)
    set_square_panel(ax)


def _panel_b(ax, df):
    sub = df.dropna(subset=['juv_change_nadir', 'rel_loss']).copy()
    order = ordered_sequences(sub['seq_category_main'].unique())
    sub = sub[sub['seq_category_main'].isin(order)].copy()

    for seq in order:
        part = sub[sub['seq_category_main'] == seq]
        ax.scatter(
            part['rel_loss'],
            part['juv_change_nadir'],
            s=42,
            color=seq_color(seq),
            alpha=0.75,
            edgecolor='white',
            linewidth=0.4,
            label=SEQ_LABELS[seq],
        )

    if len(sub) >= 5:
        slope, intercept = np.polyfit(sub['rel_loss'], sub['juv_change_nadir'], 1)
        xs = np.linspace(sub['rel_loss'].min(), sub['rel_loss'].max(), 100)
        ax.plot(xs, intercept + slope * xs, color='#222222', linewidth=1.7, zorder=5)

    ax.axhline(0, color='#9b9ea2', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.axvline(0, color='#9b9ea2', linestyle=':', linewidth=0.9, alpha=0.5)
    ax.set_xlabel('Hard coral relative loss')
    ax.set_ylabel('Juvenile net change')
    ax.legend(loc='lower left', fontsize=FS_LEGEND, frameon=False, ncol=2, handletextpad=0.18, columnspacing=0.45, labelspacing=0.18)
    clean_axis(ax)
    set_square_panel(ax)


def plot_figS2():
    apply_pub_style()
    df = _prep()
    if df['juv_change_nadir'].notna().sum() == 0:
        print('WARNING: No valid juvenile change data. S2 not generated.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.92), gridspec_kw={'width_ratios': [1.0, 1.0]})
    _panel_a(axes[0], df)
    _panel_b(axes[1], df)
    for idx, ax in enumerate(axes):
        add_panel_label(ax, chr(ord('A') + idx))

    apply_layout(fig, 'supplement_landscape', left=0.10, right=0.985, top=0.93, bottom=0.18)
    fig.subplots_adjust(wspace=0.20)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_figS2()
