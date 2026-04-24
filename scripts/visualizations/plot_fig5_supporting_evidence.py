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
    set_square_panel,
)


FEATURES = config.FINAL_FEATURES_PATH
OUTPUT = config.SUPPORTING_EVIDENCE_FIG_PATH
SIGNAL_LABELS = {
    'COTS lag (3 yr)': 'COTS lag\n(3 yr)',
    'Initial COTS': 'Initial\nCOTS',
    'Initial herbivore': 'Initial\nherbivore',
    'Final algae change': 'Final algae\nchange',
}
X_LABELS = {
    'S_to_H': 'S->H',
    'S_to_S': 'S->S',
    'Isolated_Storm': 'Isol.\nstorm',
    'Concurrent': 'Concurrent',
    'Isolated_Heatwave': 'Isol.\nheatwave',
    'H_to_H': 'H->H',
    'H_to_S': 'H->S',
}


def _prep():
    df = filter_downstream_analysis_sample(pd.read_csv(FEATURES))
    df['algae_change_final'] = df['final_algae'] - df['baseline_algae']
    return df


def _panel_a(ax, df):
    sub = df.dropna(subset=['algae_change_final']).copy()
    order = ordered_sequences(sub['seq_category_main'].unique())
    palette = {seq: seq_color(seq) for seq in order}

    sns.boxplot(
        data=sub,
        x='seq_category_main',
        y='algae_change_final',
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
        y='algae_change_final',
        order=order,
        color='#4a4e54',
        alpha=0.42,
        size=2.8,
        jitter=0.14,
        ax=ax,
    )
    ax.axhline(0, color='#4a4e54', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Final algae change')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([X_LABELS[s] for s in order], fontsize=6.9, rotation=0, ha='center')
    ax.tick_params(axis='x', pad=2.2)
    clean_axis(ax)


def _panel_b(ax, df):
    sub = df.dropna(subset=['algae_change_final', 'rel_loss']).copy()
    order = ordered_sequences(sub['seq_category_main'].unique())

    for seq in order:
        part = sub[sub['seq_category_main'] == seq]
        ax.scatter(
            part['rel_loss'],
            part['algae_change_final'],
            s=40 if seq == 'S_to_H' else 30,
            color=seq_color(seq),
            alpha=0.72,
            edgecolor='white',
            linewidth=0.4,
            label=SEQ_LABELS[seq],
        )

    if len(sub) >= 5:
        slope, intercept = np.polyfit(sub['rel_loss'], sub['algae_change_final'], 1)
        xs = np.linspace(sub['rel_loss'].min(), sub['rel_loss'].max(), 160)
        ax.plot(xs, intercept + slope * xs, color='#212529', linewidth=1.45, zorder=4)

    rho = sub['algae_change_final'].corr(sub['rel_loss'], method='spearman') if len(sub) >= 5 else np.nan
    ax.text(
        0.95, 0.93,
        f'$\\rho$ = {rho:.2f}',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=7.9,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#dde3e8', alpha=0.92),
    )
    ax.axhline(0, color='#9b9ea2', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='#9b9ea2', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Relative coral loss')
    ax.set_ylabel('Final algae change')
    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.0, 1.025, 1.0, 0.14),
        mode='expand',
        fontsize=7.2,
        frameon=False,
        ncol=3,
        handletextpad=0.12,
        columnspacing=0.30,
        labelspacing=0.10,
        borderaxespad=0.0,
    )
    clean_axis(ax)


def _panel_c(ax, df):
    rows = []
    for var, label in [
        ('cots_lag_3yr', 'COTS lag (3 yr)'),
        ('initial_cots', 'Initial COTS'),
        ('initial_herbivore', 'Initial herbivore'),
        ('algae_change_final', 'Final algae change'),
    ]:
        sub = df[[var, 'rel_loss']].dropna()
        rho = sub[var].corr(sub['rel_loss'], method='spearman') if len(sub) >= 5 else np.nan
        rows.append({'signal': label, 'rho': rho})

    summary = pd.DataFrame(rows).sort_values('rho', ascending=True).reset_index(drop=True)
    summary['signal_display'] = summary['signal'].map(SIGNAL_LABELS)
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#729ece']

    ax.barh(
        summary['signal_display'],
        summary['rho'],
        color=colors,
        edgecolor='white',
        alpha=0.86,
        height=0.58,
        linewidth=0.8,
    )
    ax.axvline(0, color='#4a4e54', linestyle='-', linewidth=0.8, alpha=0.55)
    ax.set_xlabel('Spearman association ($\\rho$)')
    ax.set_ylabel('')
    ax.set_xlim(min(0, summary['rho'].min() - 0.1), max(0, summary['rho'].max() + 0.1))
    clean_axis(ax)


def plot_fig5():
    apply_pub_style()
    df = _prep()

    fig, _ = create_figure('main_landscape', override_height_mm=95)
    gs = make_gridspec(fig, 1, 3, preset='main_landscape', wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _panel_a(ax_a, df)
    _panel_b(ax_b, df)
    _panel_c(ax_c, df)

    for ax in [ax_a, ax_b, ax_c]:
        set_square_panel(ax)

    add_panel_label(ax_a, 'A')
    add_panel_label(ax_b, 'B', x=-0.10)
    add_panel_label(ax_c, 'C', x=-0.10)

    apply_layout(fig, 'main_landscape', left=0.078, right=0.985, bottom=0.18, top=0.88)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_fig5()
