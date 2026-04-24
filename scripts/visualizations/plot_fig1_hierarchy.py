# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
    create_figure,
    make_gridspec,
    ordered_sequences,
    save_publication_figure,
    seq_color,
    seq_fill,
    set_percent_axis,
)


BOOTSTRAP = config.BOOTSTRAP_HIERARCHY_PATH
PSM = config.PSM_SEQUENCE_EFFECT_PATH
BALANCE = config.PSM_BALANCE_TABLE_PATH
PERM = config.PERMUTATION_SEQUENCE_EFFECT_PATH
PERM_DIST = config.PERMUTATION_SEQUENCE_DIST_PATH
OUTPUT = config.FIG1_PATH


def _panel_a(ax, hierarchy):
    order = ordered_sequences(hierarchy['seq_category'].tolist())
    plot_df = hierarchy.set_index('seq_category').loc[order].reset_index()
    y = np.arange(len(plot_df))[::-1]

    yticklabels = []
    for _, row in plot_df.iterrows():
        label = SEQ_LABELS[row['seq_category']]
        yticklabels.append(label)

    for y_pos, (_, row) in zip(y, plot_df.iterrows()):
        color = seq_color(row['seq_category'])
        fill_color = seq_fill(row['seq_category'], 0.8)

        ax.plot(
            [row['ci_lo'], row['ci_hi']],
            [y_pos, y_pos],
            color=fill_color,
            linewidth=3.5,
            solid_capstyle='round',
            zorder=2,
        )

        ax.scatter(
            row['mean_rel_loss'],
            y_pos,
            s=82,
            facecolor='white',
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(row['mean_rel_loss'], y_pos, s=24, color=color, zorder=5)

    ax.axvline(0, color='#9b9ea2', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Raw mean relative coral loss')
    ax.set_xlim(min(-0.45, plot_df['ci_lo'].min() - 0.08), max(0.95, plot_df['ci_hi'].max() + 0.15))
    set_percent_axis(ax, 'x', 1.0)
    clean_axis(ax)


def _panel_c(ax, balance):
    plot_df = balance.copy().sort_values('smd_before', key=np.abs, ascending=True)
    y = np.arange(len(plot_df))
    nice_labels = {
        'storm_exposure_seq': 'Storm exposure',
        'heat_exposure_seq': 'Heat exposure',
        'baseline_hc': 'Baseline HC',
        'start_year': 'Start year',
        'region_lat': 'Latitude',
    }

    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.plot([row['smd_before'], row['smd_after']], [idx, idx], color='#d1d7dd', linewidth=1.2, zorder=1)

    ax.scatter(
        plot_df['smd_before'],
        y,
        color='#d62728',
        s=48,
        label='Before matching',
        zorder=3,
        edgecolor='white',
        linewidth=0.6,
        alpha=0.85,
    )
    ax.scatter(
        plot_df['smd_after'],
        y,
        color='#1f77b4',
        s=48,
        label='After matching',
        zorder=4,
        edgecolor='white',
        linewidth=0.6,
        alpha=0.95,
    )
    ax.axvline(0, color='#4a4e54', linestyle='-', linewidth=0.8)
    ax.axvline(0.1, color='#e7ba52', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.axvline(-0.1, color='#e7ba52', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([nice_labels.get(c, c.replace('_', ' ').title()) for c in plot_df['covariate']])
    ax.set_xlabel('Standardized mean difference')
    ax.legend(loc='upper left', frameon=False, fontsize=8.0, handletextpad=0.2, labelspacing=0.3)
    ax.set_xlim(
        min(-0.85, plot_df[['smd_before', 'smd_after']].min().min() - 0.1),
        max(0.65, plot_df[['smd_before', 'smd_after']].max().max() + 0.1),
    )
    clean_axis(ax)


def _panel_b(ax, psm):
    plot_df = psm[psm['outcome'].isin(['rel_loss', 'drop_abs'])].copy()
    label_map = {'rel_loss': 'Relative loss', 'drop_abs': 'Absolute loss'}
    plot_df['label'] = plot_df['outcome'].map(label_map)
    plot_df['att_display'] = np.where(
        plot_df['outcome'] == 'rel_loss',
        plot_df['att'] * 100.0,
        plot_df['att'],
    )
    plot_df['unit'] = np.where(plot_df['outcome'] == 'rel_loss', '%', 'pp')
    plot_df['text'] = plot_df.apply(
        lambda row: f"{row['att_display']:.1f}{row['unit']}" + ("*" if row['signflip_p'] < 0.05 else ""),
        axis=1,
    )

    y = np.arange(len(plot_df))[::-1]
    colors = [seq_color('S_to_H'), '#e7ba52']
    ax.barh(
        y,
        plot_df['att_display'],
        color=colors[:len(plot_df)],
        alpha=0.85,
        edgecolor='white',
        linewidth=0.8,
        height=0.5,
    )

    for y_pos, (_, row) in zip(y, plot_df.iterrows()):
        ax.text(
            row['att_display'] + max(plot_df['att_display']) * 0.05,
            y_pos,
            row['text'],
            va='center',
            ha='left',
            fontsize=8.5,
            color='#333a40',
            fontweight='bold' if row['signflip_p'] < 0.05 else 'normal',
        )

    ax.axvline(0, color='#4a4e54', linestyle='-', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['label'])
    ax.set_xlabel('Matched effect size')
    ax.set_xlim(0, max(plot_df['att_display']) * 1.65)
    clean_axis(ax)


def _panel_d(ax, perm, perm_dist):
    rel_row = perm[perm['outcome'] == 'rel_loss'].iloc[0]
    dist = perm_dist[perm_dist['outcome'] == 'rel_loss']['null_mean_diff'].to_numpy() * 100.0
    observed = rel_row['observed_diff'] * 100.0

    ax.hist(dist, bins=25, color='#a8dadc', edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(observed, color='#d62728', linewidth=1.8, linestyle='--')

    ax.text(
        0.97,
        0.92,
        f'Observed = {observed:.1f}%\np = {rel_row["permutation_p"]:.3f}',
        color='#d62728',
        fontsize=FS_ANNOT,
        ha='right',
        va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d1d7dd', alpha=0.92),
        fontweight='bold',
    )
    ax.set_xlabel('Null mean contrast (%)')
    ax.set_ylabel('Frequency')
    clean_axis(ax)


def plot_fig1():
    apply_pub_style()
    hierarchy = pd.read_csv(BOOTSTRAP)
    psm = pd.read_csv(PSM)
    balance = pd.read_csv(BALANCE)
    perm = pd.read_csv(PERM)
    perm_dist = pd.read_csv(PERM_DIST)

    # Increase height to accommodate 2x2 square-ish layout
    # Double column width is ~180mm, so 2 panels need ~180mm height for squares
    fig, _ = create_figure('main_dense', override_height_mm=165)
    gs = make_gridspec(
        fig,
        2,
        2,
        wspace=0.28,
        hspace=0.35,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a, hierarchy)
    _panel_b(ax_b, psm)
    _panel_c(ax_c, balance)
    _panel_d(ax_d, perm, perm_dist)

    from style_config import set_square_panel

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        set_square_panel(ax)

    add_panel_label(ax_a, 'A')
    add_panel_label(ax_b, 'B', x=-0.12)
    add_panel_label(ax_c, 'C', x=-0.12)
    add_panel_label(ax_d, 'D', x=-0.12)

    apply_layout(fig, 'main_dense', left=0.15, bottom=0.10, right=0.98, top=0.95)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_fig1()
