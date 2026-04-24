# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modeling_core import filter_downstream_analysis_sample

from style_config import (
    FS_ANNOT,
    FS_TICK,
    SEQ_LABELS,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    create_figure,
    make_gridspec,
    ordered_sequences,
    save_publication_figure,
    seq_color,
    set_percent_axis,
)


FEATURES = config.FINAL_FEATURES_PATH
GAP = config.GAP_MEMORY_MAIN_RESULTS_PATH
CURVES = config.GAP_MEMORY_CURVES_PATH
OUTPUT = config.FIG2_PATH
SHORT_GAP_THRESHOLD = 2
MEMORY_ORDER = ['S_to_H', 'S_to_S', 'H_to_H']


def _panel_a(ax, features, curves):
    sub = features[features['seq_category_main'] == 'S_to_H'].dropna(subset=['time_gap_years', 'rel_loss']).copy()
    ax.scatter(
        sub['time_gap_years'],
        sub['rel_loss'],
        s=60,
        color=seq_color('S_to_H'),
        edgecolor='white',
        linewidth=0.5,
        alpha=0.7,
        zorder=3,
    )
    curve = curves[curves['seq_category_main'] == 'S_to_H'].copy()
    if not curve.empty:
        ax.plot(curve['time_gap_years'], curve['smooth_rel_loss'], color='white', linewidth=4.0, zorder=4, alpha=0.8)
        ax.plot(curve['time_gap_years'], curve['smooth_rel_loss'], color='#212529', linewidth=1.8, zorder=5)

    ax.axvline(SHORT_GAP_THRESHOLD, color='#9b9ea2', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.axhline(0, color='#9b9ea2', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.text(
        SHORT_GAP_THRESHOLD + 0.1,
        ax.get_ylim()[1] * 0.9,
        '2-year split',
        fontsize=FS_ANNOT,
        color='#4a4e54',
        ha='left',
        va='top',
        fontweight='bold'
    )
    ax.set_xlabel('Gap between storm and heatwave (years)')
    ax.set_ylabel('Relative coral loss')
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(-0.05, 0.85)
    set_percent_axis(ax, 'y', 1.0)
    clean_axis(ax)


def _panel_b(ax, gap):
    plot_df = gap[gap['seq_category_main'].isin(MEMORY_ORDER)].copy()
    plot_df = plot_df.set_index('seq_category_main').loc[MEMORY_ORDER].reset_index()
    x = np.arange(len(plot_df))

    bars = ax.bar(
        x,
        plot_df['short_minus_long_diff'] * 100.0,
        color=[seq_color(seq) for seq in plot_df['seq_category_main']],
        edgecolor='white',
        alpha=0.85,
        width=0.6,
        linewidth=0.8
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + (2 if height > 0 else -6),
            f'{height:.1f}%',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=FS_ANNOT, fontweight='bold', color='#333a40'
        )

    ax.axhline(0, color='#4a4e54', linestyle='-', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SEQ_LABELS[s] for s in plot_df['seq_category_main']])
    ax.set_ylabel('Short - long gap difference (%)')
    ax.set_ylim(min(0, plot_df['short_minus_long_diff'].min()*100 - 15),
                max(0, plot_df['short_minus_long_diff'].max()*100 + 15))
    clean_axis(ax)


def _panel_c(ax, features):
    category_order = ordered_sequences(MEMORY_ORDER + ['Isolated_Storm', 'Isolated_Heatwave'])
    plot_df = (
        features.groupby('seq_category_main')[['storm_exposure_seq', 'heat_exposure_seq']]
        .mean()
        .reset_index()
    )
    plot_df = plot_df[plot_df['seq_category_main'].isin(category_order)].copy()
    plot_df['seq_category_main'] = pd.Categorical(plot_df['seq_category_main'], categories=category_order, ordered=True)
    plot_df = plot_df.sort_values('seq_category_main')

    for _, row in plot_df.iterrows():
        seq = row['seq_category_main']
        color = seq_color(seq)
        ax.scatter(
            row['storm_exposure_seq'],
            row['heat_exposure_seq'],
            s=90 if seq == 'S_to_H' else 65,
            color=color,
            edgecolor='white',
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
            label=SEQ_LABELS[seq]
        )

        if seq == 'S_to_H':
            ax.text(
                row['storm_exposure_seq'] + 0.8,
                row['heat_exposure_seq'] - 0.05,
                SEQ_LABELS[seq],
                fontsize=FS_ANNOT,
                ha='left',
                va='top',
                color='#333a40',
                fontweight='bold'
            )

    ax.set_xlabel('Mean storm exposure')
    ax.set_ylabel('Mean heat exposure')

    ax.set_xlim(plot_df['storm_exposure_seq'].min() - 5, plot_df['storm_exposure_seq'].max() + 15)
    ax.set_ylim(plot_df['heat_exposure_seq'].min() - 0.5, plot_df['heat_exposure_seq'].max() + 1.2)

    # Remove legend as requested to keep it clean like Fig 0
    clean_axis(ax)


def plot_fig2():
    apply_pub_style()
    features = filter_downstream_analysis_sample(pd.read_csv(FEATURES))
    gap = pd.read_csv(GAP)
    curves = pd.read_csv(CURVES)

    fig, _ = create_figure('main_landscape', override_height_mm=102)
    gs = make_gridspec(fig, 1, 3, preset='main_landscape', wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _panel_a(ax_a, features, curves)
    _panel_b(ax_b, gap)
    _panel_c(ax_c, features)

    from style_config import set_square_panel
    for ax in [ax_a, ax_b, ax_c]:
        set_square_panel(ax)

    add_panel_label(ax_a, 'A')
    add_panel_label(ax_b, 'B', x=-0.15)
    add_panel_label(ax_c, 'C', x=-0.15)

    apply_layout(fig, 'main_landscape', left=0.07, bottom=0.16, top=0.93)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_fig2()
