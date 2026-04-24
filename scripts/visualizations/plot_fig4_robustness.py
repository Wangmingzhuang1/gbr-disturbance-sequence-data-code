# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from style_config import (
    SEQ_LABELS,
    SEQ_ORDER,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    create_figure,
    make_gridspec,
    save_publication_figure,
    seq_color,
    set_square_panel,
)


ROBUSTNESS = config.ROBUSTNESS_SUMMARY_PATH
OUTPUT = config.ROBUSTNESS_FIG_PATH

SCENARIO_LABELS = {
    'primary_first_event': 'Primary rule\n(first event)',
    'storm_priority': 'Storm-priority\nrule',
    'window_1_3': 'Lookahead\n1-3 yr',
    'window_1_4': 'Lookahead\n1-4 yr',
    'window_1_5': 'Lookahead\n1-5 yr',
    'baseline_hc_10pct': 'Baseline HC\n>= 10%',
}

KEY_CATS = [seq for seq in SEQ_ORDER if seq not in {'Concurrent', 'H_to_S'}]
SHORT_LABELS = {
    'S_to_H': 'S->H',
    'S_to_S': 'S->S',
    'Isolated_Storm': 'Storm',
    'Isolated_Heatwave': 'Heatwave',
    'H_to_H': 'H->H',
}


def _panel_a(ax, df_rob):
    hier_rows = df_rob[df_rob['metric_name'] == 'mean_rel_loss'].copy()
    scenarios = [name for name in SCENARIO_LABELS if name in hier_rows['scenario_name'].unique()]
    matrix = np.full((len(scenarios), len(KEY_CATS)), np.nan)

    for i, scenario in enumerate(scenarios):
        sub = hier_rows[hier_rows['scenario_name'] == scenario]
        lookup = sub.set_index('seq_category')['value'].to_dict()
        for j, seq in enumerate(KEY_CATS):
            matrix[i, j] = lookup.get(seq, np.nan)

    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', vmin=-0.6, vmax=0.6)
    ax.set_xticks(np.arange(len(KEY_CATS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(scenarios)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xticks(range(len(KEY_CATS)))
    ax.set_xticklabels([SHORT_LABELS[seq] for seq in KEY_CATS], fontsize=8.2)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=8.1)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = 'white' if abs(value) > 0.35 else '#212529'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7.9, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.set_title('Rel.\nloss', fontsize=8.3, pad=7, fontweight='bold')
    cbar.ax.tick_params(labelsize=7.8)

    clean_axis(ax)
    ax.tick_params(axis='both', length=0)


def _panel_b(ax, df_rob):
    rank_rows = df_rob[
        (df_rob['metric_name'] == 'rank') &
        (df_rob['scenario_name'] == 'primary_first_event')
    ].copy()
    rank_map = {
        row['seq_category']: row['value']
        for _, row in rank_rows.iterrows()
        if row['seq_category'] in KEY_CATS
    }
    # Filter to KEY_CATS and re-rank from 1 to N
    filtered_ranks = {
        seq: rank_map[seq]
        for seq in KEY_CATS
        if seq in rank_map
    }
    # Sort by original rank and assign new ranks 1, 2, 3...
    sorted_seqs = sorted(filtered_ranks.keys(), key=lambda s: filtered_ranks[s])
    new_rank_map = {seq: i + 1 for i, seq in enumerate(sorted_seqs)}

    display_order = sorted_seqs
    y = np.arange(len(display_order))[::-1]

    for y_pos, seq in zip(y, display_order):
        rank = new_rank_map.get(seq, np.nan)
        if np.isnan(rank):
            continue
        ax.plot([1, rank], [y_pos, y_pos], color='#d8dee4', linewidth=1.3, zorder=1)
        ax.scatter(rank, y_pos, s=130, color=seq_color(seq), edgecolor='white', linewidth=1.0, zorder=3)
        ax.text(rank + 0.18, y_pos, f'#{int(rank)}', va='center', ha='left', fontsize=8.4, color='#4a4e54', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([SEQ_LABELS[seq] for seq in display_order])
    ax.set_xlim(0.8, len(display_order) + 0.8)
    ax.set_xticks(range(1, len(display_order) + 1))
    ax.set_xlabel('Main-analysis rank (1st = highest loss)')
    clean_axis(ax)


def plot_fig4():
    apply_pub_style()
    df_rob = pd.read_csv(ROBUSTNESS) if os.path.exists(ROBUSTNESS) else None

    fig, _ = create_figure('main_landscape', override_height_mm=102)
    gs = make_gridspec(fig, 1, 2, preset='main_landscape', width_ratios=[1.22, 0.78], wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    if df_rob is not None and not df_rob.empty:
        _panel_a(ax_a, df_rob)
        _panel_b(ax_b, df_rob)
    else:
        ax_a.text(0.5, 0.5, 'Robustness data not available', ha='center', va='center', transform=ax_a.transAxes)
        ax_b.text(0.5, 0.5, 'Rank data not available', ha='center', va='center', transform=ax_b.transAxes)

    add_panel_label(ax_a, 'A')
    add_panel_label(ax_b, 'B', x=-0.10)

    apply_layout(fig, 'main_landscape', left=0.17, right=0.985, bottom=0.14, top=0.94)
    save_publication_figure(fig, OUTPUT)
    plt.close()
    print(f'Figure 4 saved: {OUTPUT}')


if __name__ == '__main__':
    plot_fig4()
