# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from style_config import (
    FS_LABEL,
    SEQ_LABELS,
    add_panel_label,
    apply_layout,
    apply_pub_style,
    clean_axis,
    save_publication_figure,
    seq_color,
    set_percent_axis,
    set_square_panel,
)


FEATURES = config.FINAL_FEATURES_PATH
OUTPUT = config.FIGS1_PATH
MEMORY_ORDER = ['S_to_H', 'S_to_S', 'H_to_H']


def plot_memory_panel(ax, subset, seq, y_limits):
    x = subset['time_gap_years']
    y = subset['rel_loss']
    ax.scatter(x, y, s=40, color=seq_color(seq), alpha=0.75, edgecolor='white', linewidth=0.45)
    if len(subset) >= 4 and subset['time_gap_years'].nunique() >= 3:
        curve = lowess(y, x, frac=0.75, return_sorted=True)
        ax.plot(curve[:, 0], curve[:, 1], color='#2f2f2f', linewidth=1.7)
    ax.axhline(0, color='#7b8085', linestyle='--', linewidth=0.9)
    ax.set_xlabel('Gap between disturbances (years)')
    ax.set_ylabel('Relative coral loss')
    ax.set_xlim(0.85, 4.15)
    ax.set_ylim(*y_limits)
    set_percent_axis(ax, 'y', 1.0)
    clean_axis(ax)
    set_square_panel(ax)


def plot_figS1():
    apply_pub_style()
    df = pd.read_csv(FEATURES)
    subsets = [
        df[df['seq_category_main'] == seq].dropna(subset=['time_gap_years', 'rel_loss']).copy()
        for seq in MEMORY_ORDER
    ]
    y_min = min(-1.05, min((sub['rel_loss'].min() for sub in subsets if not sub.empty), default=-0.6)) - 0.08
    y_max = max(0.95, max((sub['rel_loss'].max() for sub in subsets if not sub.empty), default=0.6)) + 0.08

    fig, axes = plt.subplots(1, 3, figsize=(6.69, 3.74))
    for idx, seq in enumerate(MEMORY_ORDER):
        plot_memory_panel(axes[idx], subsets[idx], seq, (y_min, y_max))
        add_panel_label(axes[idx], chr(ord('A') + idx))
        axes[idx].set_title(SEQ_LABELS[seq], fontsize=FS_LABEL, fontweight='bold', pad=6)

    apply_layout(fig, 'supplement_landscape', left=0.08, right=0.985, top=0.90, bottom=0.19)
    fig.subplots_adjust(wspace=0.25)
    save_publication_figure(fig, OUTPUT)
    plt.close()


if __name__ == '__main__':
    plot_figS1()
