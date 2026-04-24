# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modeling_core import filter_downstream_analysis_sample

from style_config import (
    FAILURE_RED,
    FS_ANNOT,
    FS_LEGEND,
    FS_TICK,
    HEATWAVE_COLOR,
    RECOVERY_GREEN,
    STORM_COLOR,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    create_figure,
    make_gridspec,
    save_publication_figure,
)


FEATURES = config.FINAL_FEATURES_PATH
SUMMARY = config.FATE_SUMMARY_PATH
STRATIFIED = config.FATE_STRATIFIED_PATH
OUTPUT = config.FATE_FIG_PATH

DISTURBANCE_ORDER = ['Heat-linked', 'Storm-linked']
BASELINE_GROUP_ORDER = ['Low baseline HC', 'High baseline HC']
DISTURBANCE_COLORS = {
    'Heat-linked': HEATWAVE_COLOR,
    'Storm-linked': STORM_COLOR,
}
BASELINE_COLORS = {
    'Low baseline HC': '#7F8C8D',  # Muted Gray
    'High baseline HC': '#C0392B',  # Deep Red
}
DISTURBANCE_MARKERS = {
    'Heat-linked': 'o',
    'Storm-linked': 's',
}


def _load():
    features = filter_downstream_analysis_sample(pd.read_csv(FEATURES))
    summary = pd.read_csv(SUMMARY)
    stratified = pd.read_csv(STRATIFIED)
    sub = features[
        (features['impact_flag'] == 1) &
        (features['recovery_observed_flag'] == 1)
    ].copy()
    return features, summary, stratified, sub


def _panel_a(ax, stratified):
    """Refactored from original Panel B"""
    plot_df = stratified[stratified['n_events'] > 0].copy()
    plot_df = plot_df[plot_df['disturbance_group'].isin(DISTURBANCE_ORDER)].copy()
    plot_df['baseline_hc_group'] = pd.Categorical(plot_df['baseline_hc_group'], categories=BASELINE_GROUP_ORDER, ordered=True)
    plot_df['disturbance_group'] = pd.Categorical(plot_df['disturbance_group'], categories=DISTURBANCE_ORDER, ordered=True)

    sns.barplot(
        data=plot_df,
        x='baseline_hc_group',
        y='recovered_fraction',
        hue='disturbance_group',
        order=BASELINE_GROUP_ORDER,
        hue_order=DISTURBANCE_ORDER,
        palette=DISTURBANCE_COLORS,
        edgecolor='white',
        linewidth=0.8,
        ax=ax,
        alpha=0.85
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('')
    ax.set_ylabel('Recovered fraction')
    ax.legend(title='', frameon=False, loc='upper right', fontsize=FS_LEGEND, handletextpad=0.2, labelspacing=0.25)

    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=FS_ANNOT,
                fontweight='bold',
                color='#333a40',
            )

    clean_axis(ax)


def _panel_b(ax, sub):
    """Refactored from original Panel C, now with marginal-like logic or cleaned density"""
    plot_df = sub[sub['disturbance_group'].isin(DISTURBANCE_ORDER)].copy()
    plot_df['baseline_hc_group'] = pd.Categorical(plot_df['baseline_hc_group'], categories=BASELINE_GROUP_ORDER, ordered=True)
    plot_df['disturbance_group'] = pd.Categorical(plot_df['disturbance_group'], categories=DISTURBANCE_ORDER, ordered=True)

    x_split = float(plot_df['drop_abs'].median())
    y_split = 0.5
    x_max = max(20.0, float(plot_df['drop_abs'].max()) + 4.0)
    y_min = min(-0.4, float(plot_df['recovery_frac_loss'].min()) - 0.2)
    y_max = max(2.2, float(plot_df['recovery_frac_loss'].max()) + 0.2)

    # Background zones
    ax.axhspan(y_split, y_max, color='#e8f5e9', alpha=0.5, zorder=0)
    ax.axhspan(y_min, y_split, color='#ffebee', alpha=0.5, zorder=0)
    ax.axvline(x_split, color='#4a4e54', linestyle='--', linewidth=0.9, zorder=1)
    ax.axhline(y_split, color='#d62728', linestyle='--', linewidth=0.9, zorder=1)
    ax.axhline(0.0, color='#9b9ea2', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    # Scatters
    rng = np.random.default_rng(42)
    for group in DISTURBANCE_ORDER:
        for baseline_group in BASELINE_GROUP_ORDER:
            part = plot_df[
                (plot_df['disturbance_group'] == group) &
                (plot_df['baseline_hc_group'] == baseline_group)
            ]
            if part.empty:
                continue

            ax.scatter(
                part['drop_abs'] + rng.normal(0, 0.25, len(part)),
                part['recovery_frac_loss'] + rng.normal(0, 0.03, len(part)),
                s=65,
                marker=DISTURBANCE_MARKERS[group],
                c=BASELINE_COLORS[baseline_group],
                edgecolors='white',
                linewidths=0.5,
                alpha=0.8,
                zorder=3,
            )

    # Labels
    ax.text(x_split + 1.2, y_max - 0.15, 'Recovery zone', ha='left', va='top', fontsize=FS_ANNOT, color=RECOVERY_GREEN, fontweight='bold')
    ax.text(x_split + 1.2, y_split - 0.15, 'Failure zone', ha='left', va='top', fontsize=FS_ANNOT, color=FAILURE_RED, fontweight='bold')

    ax.set_xlim(4.0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Absolute coral loss (pp)')
    ax.set_ylabel('Recovery fraction of loss')
    clean_axis(ax)

    # Legend
    baseline_handles = [
        Line2D([0], [0], marker='o', color='none', label=label, markerfacecolor=color,
               markeredgecolor='white', markeredgewidth=0.7, markersize=7.2)
        for label, color in BASELINE_COLORS.items()
    ]
    disturbance_handles = [
        Line2D([0], [0], marker=DISTURBANCE_MARKERS[label], color='#50545a', label=label,
               markerfacecolor='#d9d9d9', markeredgecolor='white', markeredgewidth=0.7,
               linestyle='None', markersize=7.2)
        for label in DISTURBANCE_ORDER
    ]
    handles = [*baseline_handles, Line2D([], [], linestyle='None', label=''), *disturbance_handles]
    ax.legend(
        handles=handles,
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(0.995, 0.995),
        fontsize=FS_LEGEND - 0.5,
        ncol=1,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def plot_fig3():
    apply_pub_style()
    _, _, stratified, sub = _load()

    # Change to a cleaner 1x2 landscape layout
    fig, _ = create_figure('main_landscape', override_height_mm=92)
    gs = make_gridspec(
        fig,
        1,
        2,
        preset='main_landscape',
        width_ratios=[1.0, 1.1],
        wspace=0.25,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _panel_a(ax_a, stratified)
    _panel_b(ax_b, sub)

    from style_config import set_square_panel
    for ax in [ax_a, ax_b]:
        set_square_panel(ax)

    add_panel_label(ax_a, 'A')
    add_panel_label(ax_b, 'B', x=-0.12)

    apply_layout(fig, 'main_landscape', left=0.08, right=0.985, bottom=0.16, top=0.92)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_fig3()
