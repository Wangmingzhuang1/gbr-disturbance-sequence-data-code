# -*- coding: utf-8 -*-
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from style_config import (
    FS_ANNOT,
    FS_LABEL,
    FS_TICK,
    apply_pub_style,
    create_figure,
    save_publication_figure,
)


ATTRITION = config.ATTRITION_AUDIT_PATH
FEATURES = config.FINAL_FEATURES_PATH
PERM = config.PERMUTATION_SEQUENCE_EFFECT_PATH
FATE = config.FATE_SUMMARY_PATH
FATE_SUPP = config.FATE_SUPPLEMENT_SUMMARY_PATH
OUTPUT = config.FIGS6_PATH


def _stage_value(attrition, code, field):
    row = attrition.loc[attrition['stage_code'] == code].iloc[0]
    value = row[field]
    return int(value) if pd.notna(value) else 0


def _box(ax, xy, width, height, text, facecolor, edgecolor='#36454f', fontsize=FS_TICK):
    # Remove n=... if present in text
    clean_text = re.sub(r'\n?n\s*=\s*\d+.*$', '', text, flags=re.MULTILINE).strip()
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle='round,pad=0.012,rounding_size=0.018',
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, clean_text, ha='center', va='center', fontsize=fontsize, color='#1f2933', linespacing=1.22)
    return patch


def _arrow(ax, start, end, color='#4a4e54'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9, linewidth=1.05, color=color, shrinkA=2, shrinkB=2))


def plot_figs6():
    apply_pub_style()
    attrition = pd.read_csv(ATTRITION)
    features = pd.read_csv(FEATURES)
    perm = pd.read_csv(PERM)
    fate = pd.read_csv(FATE)
    fate_supp = pd.read_csv(FATE_SUPP)

    total_candidates = _stage_value(attrition, 'S1', 'count')
    dropped_consumed = _stage_value(attrition, 'S2', 'dropped')
    dropped_unclean = _stage_value(attrition, 'S3', 'dropped')
    dropped_surveys = _stage_value(attrition, 'S4', 'dropped')
    dropped_baseline = _stage_value(attrition, 'S5', 'dropped')
    dropped_nadir = _stage_value(attrition, 'S6', 'dropped')
    retained = _stage_value(attrition, 'S8', 'count')
    reefs = _stage_value(attrition, 'S9', 'count')

    model_events = int(features.loc[features['seq_category_model'].notna()].shape[0])
    model_reefs = int(features.loc[features['seq_category_model'].notna(), 'reef_name'].nunique())

    perm_row = perm.loc[perm['outcome'] == 'rel_loss'].iloc[0]
    s_to_h_n = int(perm_row['treated_n'])
    controls_n = int(perm_row['control_n'])

    strict_stage = fate[(fate['sample_scope'] == 'strict') & (fate['summary_type'] == 'stage')]
    impacted = int(strict_stage.loc[strict_stage['group_value'] == 'impacted', 'n_events'].iloc[0])
    observed = int(strict_stage.loc[strict_stage['group_value'] == 'recovery_observed', 'n_events'].iloc[0])
    recovered = int(strict_stage.loc[strict_stage['group_value'] == 'recovered', 'n_events'].iloc[0])
    failed = int(strict_stage.loc[strict_stage['group_value'] == 'failed', 'n_events'].iloc[0])

    supp_stage = fate_supp[(fate_supp['sample_scope'] == 'supplement_only') & (fate_supp['summary_type'] == 'stage')]
    supp_retained = int(supp_stage.loc[supp_stage['group_value'] == 'retained', 'n_events'].iloc[0])
    supp_impacted = int(supp_stage.loc[supp_stage['group_value'] == 'impacted', 'n_events'].iloc[0])
    supp_observed = int(supp_stage.loc[supp_stage['group_value'] == 'recovery_observed', 'n_events'].iloc[0])

    fig, preset = create_figure('flowchart')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    main_x, branch_x, drop_x = 0.35, 0.73, 0.08
    w_main, h = 0.26, 0.083
    w_side, w_drop = 0.23, 0.20
    y_candidates, y_retained, y_model, y_fate = 0.83, 0.60, 0.41, 0.20

    _box(ax, (main_x, y_candidates), w_main, h, f'Candidate disturbance\nreef-years\nn = {total_candidates}', '#EBF5FB')
    _box(ax, (main_x, y_retained), w_main, h, f'Strict retained sequences\nn = {retained} from {reefs} reefs', '#E8F6F3')
    _box(ax, (main_x, y_model), w_main, h, f'Main OLS sample\nn = {model_events} from {model_reefs} reefs\n(rare classes excluded)', '#FEF9E7')
    _box(ax, (main_x, y_fate), w_main, h, f'Strict fate branch\nImpacted n = {impacted}\nObserved fate n = {observed}', '#FBEEE6')

    _box(ax, (drop_x, 0.78), w_drop, 0.068, f'Consumed targets\nremoved n = {dropped_consumed}', '#f3f4f6', fontsize=FS_ANNOT)
    _box(ax, (drop_x, 0.69), w_drop, 0.068, f'Baseline not clean\nremoved n = {dropped_unclean}', '#f3f4f6', fontsize=FS_ANNOT)
    _box(ax, (drop_x, 0.60), w_drop, 0.068, f'Insufficient baseline\nsurveys n = {dropped_surveys}', '#f3f4f6', fontsize=FS_ANNOT)
    _box(ax, (drop_x, 0.51), w_drop, 0.068, f'Baseline < 5%\nremoved n = {dropped_baseline}', '#f3f4f6', fontsize=FS_ANNOT)
    _box(ax, (drop_x, 0.42), w_drop, 0.068, f'Missing nadir\nremoved n = {dropped_nadir}', '#f3f4f6', fontsize=FS_ANNOT)

    _box(ax, (branch_x, y_model), w_side, h, f'Non-parametric screen\nS->H n = {s_to_h_n}\nControls n = {controls_n}', '#FDEDEC')
    _box(ax, (branch_x, y_fate + 0.10), w_side, h, f'Observed final state\nRecovered n = {recovered}\nFailed n = {failed}', '#F4ECF7')
    _box(ax, (branch_x, 0.10), w_side, h, f'Supplement-only fate expansion\nRetained n = {supp_retained}\nImpacted n = {supp_impacted}; observed n = {supp_observed}', '#FEF5E7', fontsize=FS_ANNOT)

    _arrow(ax, (main_x + w_main / 2, y_candidates), (main_x + w_main / 2, y_retained + h))
    _arrow(ax, (main_x + w_main / 2, y_retained), (main_x + w_main / 2, y_model + h))
    _arrow(ax, (main_x + w_main / 2, y_model), (main_x + w_main / 2, y_fate + h))
    for y_center in [0.814, 0.724, 0.634, 0.544, 0.454]:
        _arrow(ax, (main_x, y_center), (drop_x + w_drop, y_center))
    _arrow(ax, (main_x + w_main, y_model + h / 2), (branch_x, y_model + h / 2))
    _arrow(ax, (main_x + w_main, y_fate + h / 2), (branch_x, y_fate + 0.10 + h / 2))
    _arrow(ax, (main_x + w_main / 2, y_fate), (branch_x + w_side / 2, 0.10 + h))

    ax.text(0.5, 0.965, 'Strict extraction and analysis flowchart', ha='center', va='top', fontsize=FS_LABEL, fontweight='bold', color='#1f2933')

    fig.subplots_adjust(**preset['margins'])
    save_publication_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == '__main__':
    plot_figs6()
