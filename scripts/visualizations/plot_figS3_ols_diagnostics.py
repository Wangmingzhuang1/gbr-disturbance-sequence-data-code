# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modeling_core import build_base_formula, fit_cluster_robust_model, prepare_model_data
from style_config import add_panel_label, apply_layout, apply_pub_style, clean_axis, save_publication_figure, set_square_panel


def _fit_main_model():
    df = pd.read_csv(config.FINAL_FEATURES_PATH)
    df_model = prepare_model_data(df, include_cots=False, include_herbivore=False)
    formula = build_base_formula(include_cots=False, include_herbivore=False)
    result, _ = fit_cluster_robust_model(df_model, formula, 'main_model')
    return result


def _panel_residuals(ax, result):
    fitted = result.fittedvalues.to_numpy()
    resid = result.resid.to_numpy()
    ax.scatter(fitted, resid, s=32, alpha=0.72, color='#4d7a97', edgecolor='white', linewidth=0.35)
    ax.axhline(0, color='#7b8085', linestyle='--', linewidth=0.9)
    if len(fitted) >= 5:
        slope, intercept = np.polyfit(fitted, resid, 1)
        xs = np.linspace(fitted.min(), fitted.max(), 100)
        ax.plot(xs, intercept + slope * xs, color='#222222', linewidth=1.5)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    clean_axis(ax)
    set_square_panel(ax)


def _panel_qq(ax, result):
    infl = result.get_influence()
    std_resid = np.asarray(infl.resid_studentized_internal, dtype=float)
    std_resid = std_resid[np.isfinite(std_resid)]
    std_resid.sort()
    n = len(std_resid)
    theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.scatter(theoretical, std_resid, s=32, alpha=0.72, color='#9f1d35', edgecolor='white', linewidth=0.35)
    lim_min = min(theoretical.min(), std_resid.min())
    lim_max = max(theoretical.max(), std_resid.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color='#222222', linewidth=1.4)
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Studentized residuals')
    clean_axis(ax, grid_axis='both')
    set_square_panel(ax)


def plot_figS3():
    apply_pub_style()
    result = _fit_main_model()
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 3.78))
    _panel_residuals(axes[0], result)
    _panel_qq(axes[1], result)
    for idx, ax in enumerate(axes):
        add_panel_label(ax, chr(ord('A') + idx))
    apply_layout(fig, 'supplement_landscape', left=0.095, right=0.985, top=0.93, bottom=0.18)
    fig.subplots_adjust(wspace=0.24)
    save_publication_figure(fig, config.OLS_DIAGNOSTIC_FIG_PATH)
    plt.close()
    print(f"Figure S3 saved: {config.OLS_DIAGNOSTIC_FIG_PATH}")


if __name__ == '__main__':
    plot_figS3()
