import os
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output", "figures")
MATRIX_PATH = os.path.join(BASE, "output", "legacy_load_analysis_matrix.csv")
SPLINE_EFFECTS_PATH = os.path.join(
    BASE,
    "analysis",
    "baseline_cover_threshold_validation",
    "results",
    "nonlinear_marginal_effects.csv",
)
SPLINE_TESTS_PATH = os.path.join(
    BASE,
    "analysis",
    "baseline_cover_threshold_validation",
    "results",
    "nonlinear_spline_tests.csv",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOWS = (3, 5, 7, 8)
THERMAL = "#b44d3a"
STORM = "#3f6f9f"
NEUTRAL = "#6f6f6f"
GREEN = "#4b8f73"
GOLD = "#c49a3a"
LIGHT = "#eeeeee"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def label_event(row):
    if row["has_storm"] == 1 and row["has_heatwave"] == 1:
        return "Concurrent"
    if row["has_storm"] == 1 and row["has_heatwave"] == 0:
        return "Storm only"
    if row["has_storm"] == 0 and row["has_heatwave"] == 1:
        return "Heatwave only"
    return "No event"


def max_run(values):
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * np.nan
    return (series - series.mean()) / std


def prepare_matrix(df):
    df = df.copy()
    df["positive_loss"] = df["loss_abs"].clip(lower=0)
    df["rel_loss_clipped"] = (df["loss_abs"] / df["baseline_hc"]).clip(-1, 1)
    df["retention"] = df["nadir_hc"] / df["baseline_hc"]
    for col in [
        "baseline_hc",
        "recent_max_dhw",
        "recent_max_wind",
        "yrs_since_last_dist",
        "cumulative_dhw_5yr",
        "cumulative_wind_5yr",
        "heatwave_years_5yr",
        "storm_years_5yr",
        "max_consecutive_heatwave_5yr",
        "max_consecutive_storm_5yr",
    ]:
        df[f"{col}_z"] = zscore(df[col])
    return df


def save_figure(fig, number):
    stem = os.path.join(OUTPUT_DIR, f"figure_{number:02d}")
    fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.jpg", format="jpg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_si_figure(fig, number):
    stem = os.path.join(OUTPUT_DIR, f"si_figure_{number:02d}")
    fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.jpg", format="jpg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.12, 1.08, str(label).upper(), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def robust_ols(data, formula):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["reef_name"]})


def robust_gee(data, formula):
    return smf.gee(
        formula,
        groups="reef_name",
        data=data,
        cov_struct=sm.cov_struct.Exchangeable(),
        family=sm.families.Gaussian(),
    ).fit()


def adjusted_effect(data, outcome, exposure, method="ols"):
    z_col = f"{exposure}_z"
    work = data.copy()
    if z_col not in work:
        work[z_col] = zscore(work[exposure])
    formula = (
        f"{outcome} ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        f"+ {z_col} + yrs_since_last_dist_z + C(event_type)"
    )
    model = robust_gee(work, formula) if method == "gee" else robust_ols(work, formula)
    ci = model.conf_int().loc[z_col].tolist()
    return {
        "coef": model.params[z_col],
        "low": ci[0],
        "high": ci[1],
        "p": model.pvalues[z_col],
        "term": exposure,
        "method": method,
    }


def effect_table(data, outcome, exposures, method="ols"):
    return pd.DataFrame([adjusted_effect(data, outcome, exposure, method) for exposure in exposures])


def response_metric_effects(data):
    specs = [
        (
            "Absolute loss",
            "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)",
            THERMAL,
        ),
        (
            "Nadir cover",
            "nadir_hc ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)",
            GREEN,
        ),
        (
            "Proportional retention",
            "retention ~ recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)",
            GOLD,
        ),
    ]
    rows = []
    for label, formula, color in specs:
        outcome = formula.split("~", 1)[0].strip()
        work = data.dropna(subset=[outcome]).copy()
        standardized_outcome = f"{outcome}_standardized"
        work[standardized_outcome] = zscore(work[outcome])
        standardized_formula = formula.replace(
            f"{outcome} ~", f"{standardized_outcome} ~", 1
        )
        model = robust_ols(work, standardized_formula)
        conf = model.conf_int().loc["heatwave_years_5yr_z"].tolist()
        rows.append(
            {
                "coef": model.params["heatwave_years_5yr_z"],
                "low": conf[0],
                "high": conf[1],
                "p": model.pvalues["heatwave_years_5yr_z"],
                "label": label,
                "color": color,
            }
        )
    return pd.DataFrame(rows)


def model_terms(data, formula, terms, method="ols"):
    model = robust_gee(data, formula) if method == "gee" else robust_ols(data, formula)
    conf = model.conf_int()
    rows = []
    for term, label, color in terms:
        rows.append(
            {
                "coef": model.params[term],
                "low": conf.loc[term, 0],
                "high": conf.loc[term, 1],
                "p": model.pvalues[term],
                "term": label,
                "color": color,
            }
        )
    return pd.DataFrame(rows)


def plot_forest(ax, table, labels, colors, xlabel, xlim=None):
    y = np.arange(len(table))
    for idx, row in table.reset_index(drop=True).iterrows():
        ax.errorbar(
            row["coef"],
            idx,
            xerr=[[row["coef"] - row["low"]], [row["high"] - row["coef"]]],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=1.1,
            capsize=2.5,
            markersize=4,
        )
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim)


def figure_02(eco, legacy):
    eco = eco.copy()
    eco["event_class"] = eco.apply(label_event, axis=1)
    yearly = eco.groupby("year").agg(
        reefs=("reef_name", "nunique"),
        heatwave_reefs=("has_heatwave", "sum"),
        storm_reefs=("has_storm", "sum"),
    )
    yearly["heatwave_pct"] = yearly["heatwave_reefs"] / yearly["reefs"] * 100
    yearly["storm_pct"] = yearly["storm_reefs"] / yearly["reefs"] * 100

    annual = legacy.groupby("event_year").agg(
        heatwave_years=("heatwave_years_5yr", "mean"),
        heatwave_run=("max_consecutive_heatwave_5yr", "mean"),
        cumulative_dhw=("cumulative_dhw_5yr", "mean"),
        storm_years=("storm_years_5yr", "mean"),
        storm_run=("max_consecutive_storm_5yr", "mean"),
    )
    mixed = legacy.assign(
        mixed_history_5yr=(
            legacy["heatwave_years_5yr"].gt(0) & legacy["storm_years_5yr"].gt(0)
        ).astype(float)
    )
    annual["mixed_frequency"] = mixed.groupby("event_year")["mixed_history_5yr"].mean() * 100

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.55))
    axes = axes.ravel()

    axes[0].plot(yearly.index, yearly["heatwave_pct"], color=THERMAL, lw=1.6, label="Heatwave")
    axes[0].plot(yearly.index, yearly["storm_pct"], color=STORM, lw=1.2, label="Storm")
    axes[0].fill_between(yearly.index, yearly["heatwave_pct"], color=THERMAL, alpha=0.12)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Exposed reefs (%)")
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "a")

    axes[1].plot(annual.index, annual["heatwave_years"], color=THERMAL, lw=1.6)
    axes[1].scatter(annual.index, annual["heatwave_years"], s=10, color=THERMAL, alpha=0.65)
    axes[1].set_xlabel("Target year")
    axes[1].set_ylabel("Mean heatwave years\nin previous 5 years")
    panel_label(axes[1], "b")

    axes[2].plot(annual.index, annual["heatwave_run"], color=GOLD, lw=1.6)
    axes[2].scatter(annual.index, annual["heatwave_run"], s=10, color=GOLD, alpha=0.7)
    axes[2].set_xlabel("Target year")
    axes[2].set_ylabel("Mean maximum consecutive\nheatwave years")
    panel_label(axes[2], "c")

    axes[3].plot(annual.index, annual["storm_years"], color=STORM, lw=1.6)
    axes[3].scatter(annual.index, annual["storm_years"], s=10, color=STORM, alpha=0.65)
    axes[3].set_xlabel("Target year")
    axes[3].set_ylabel("Mean storm years\nin previous 5 years")
    panel_label(axes[3], "d")

    axes[4].plot(annual.index, annual["storm_run"], color="#7aa0c4", lw=1.6)
    axes[4].scatter(annual.index, annual["storm_run"], s=10, color="#7aa0c4", alpha=0.7)
    axes[4].set_xlabel("Target year")
    axes[4].set_ylabel("Mean maximum consecutive\nstorm years")
    panel_label(axes[4], "e")

    axes[5].plot(annual.index, annual["mixed_frequency"], color=GOLD, lw=1.6)
    axes[5].scatter(annual.index, annual["mixed_frequency"], s=10, color=GOLD, alpha=0.7)
    axes[5].set_xlabel("Target year")
    axes[5].set_ylabel("Mixed-history frequency (%)")
    axes[5].set_ylim(bottom=0)
    panel_label(axes[5], "f")

    fig.tight_layout(w_pad=1.5, h_pad=1.35)
    save_figure(fig, 2)


def figure_03(legacy):
    fig = plt.figure(figsize=(7.1, 5.35))
    gs = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    axes[0].scatter(legacy["baseline_hc"], legacy["loss_abs"], s=9, color=NEUTRAL, alpha=0.38)
    sns.regplot(
        data=legacy,
        x="baseline_hc",
        y="loss_abs",
        ax=axes[0],
        scatter=False,
        line_kws={"color": "black", "lw": 1.2, "ls": "--"},
    )
    axes[0].axhline(0, color="#aaaaaa", lw=0.8)
    axes[0].set_xlabel("Baseline hard-coral cover (%)")
    axes[0].set_ylabel("Absolute hard-coral loss (%)")
    
    max_hc = legacy['baseline_hc'].max()
    axes[0].plot([0, max_hc], [0, max_hc], color='#7F8C8D', linestyle=':', linewidth=0.8, label='1:1 Limit')
    
    panel_label(axes[0], "a")

    order = sorted(legacy["heatwave_years_5yr"].dropna().unique())
    sns.boxplot(
        data=legacy,
        x="heatwave_years_5yr",
        y="loss_abs",
        order=order,
        ax=axes[1],
        color="#d8b267",
        width=0.58,
        fliersize=1.2,
        linewidth=0.8,
    )
    sns.stripplot(
        data=legacy,
        x="heatwave_years_5yr",
        y="loss_abs",
        order=order,
        ax=axes[1],
        color="#424242",
        alpha=0.18,
        size=2,
        jitter=0.22,
    )
    axes[1].axhline(0, color="#999999", lw=0.8)
    axes[1].set_xlabel("Heatwave years in previous 5 years")
    axes[1].set_ylabel("Absolute hard-coral loss (%)")
    panel_label(axes[1], "b")

    rel_loss_pct = legacy["rel_loss_clipped"] * 100
    scatter = axes[2].scatter(
        rel_loss_pct,
        legacy["loss_abs"],
        c=legacy["baseline_hc"],
        cmap="cividis_r",
        s=10,
        alpha=0.72,
        edgecolors="none",
    )
    axes[2].axvline(0, color="#999999", lw=0.8)
    axes[2].axhline(0, color="#999999", lw=0.8)
    axes[2].set_xlim(-105, 105)
    axes[2].set_xlabel("Clipped relative loss (%)")
    axes[2].set_ylabel("Absolute hard-coral loss (%)")
    cbar = fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.035)
    cbar.set_label("Baseline hard-coral cover (%)")
    panel_label(axes[2], "c")

    table = response_metric_effects(legacy)
    plot_forest(
        axes[3],
        table,
        table["label"].tolist(),
        table["color"].tolist(),
        "Standardized response coefficient",
    )
    panel_label(axes[3], "d")

    fig.tight_layout(w_pad=1.25, h_pad=1.15)
    save_figure(fig, 3)


def figure_04(legacy):
    formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)"
    )
    terms = [
        ("baseline_hc_z", "Baseline coral cover", GREEN),
        ("recent_max_dhw_z", "Recent thermal stress", THERMAL),
        ("recent_max_wind_z", "Recent storm stress", STORM),
        ("heatwave_years_5yr_z", "Heatwave years", THERMAL),
        ("storm_years_5yr_z", "Storm years", STORM),
        ("yrs_since_last_dist_z", "Return interval", NEUTRAL),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.45))
    variables = [term[0] for term in terms]
    labels = [term[1] for term in terms]
    colors = [term[2] for term in terms]
    
    # Panel a: OLS coefficients
    model_ols = robust_ols(legacy, formula)
    ci_ols = model_ols.conf_int()
    ols_rows = []
    for var, label in zip(variables, labels):
        ols_rows.append({
            "coef": model_ols.params[var],
            "low": ci_ols.loc[var, 0],
            "high": ci_ols.loc[var, 1],
            "p": model_ols.pvalues[var],
            "term": label
        })
    ols_table = pd.DataFrame(ols_rows)
    plot_forest(axes[0, 0], ols_table, labels, colors, "Cluster-robust OLS coefficient")
    panel_label(axes[0, 0], "a")

    # Panel b: GEE coefficients
    model_gee = robust_gee(legacy, formula)
    ci_gee = model_gee.conf_int()
    gee_rows = []
    for var, label in zip(variables, labels):
        gee_rows.append({
            "coef": model_gee.params[var],
            "low": ci_gee.loc[var, 0],
            "high": ci_gee.loc[var, 1],
            "p": model_gee.pvalues[var],
            "term": label
        })
    gee_table = pd.DataFrame(gee_rows)
    plot_forest(axes[0, 1], gee_table, labels, colors, "GEE coefficient")
    panel_label(axes[0, 1], "b")

    # Panel c: pointplot of history class
    category = legacy.copy()
    category["history_class"] = np.select(
        [
            category["heatwave_years_5yr"].eq(0) & category["storm_years_5yr"].eq(0),
            category["heatwave_years_5yr"].gt(0) & category["storm_years_5yr"].eq(0),
            category["heatwave_years_5yr"].eq(0) & category["storm_years_5yr"].gt(0),
            category["heatwave_years_5yr"].gt(0) & category["storm_years_5yr"].gt(0),
        ],
        ["No prior", "Thermal only", "Storm only", "Mixed"],
        default="Other",
    )
    order = ["No prior", "Thermal only", "Storm only", "Mixed"]
    palette = ["#c7c7c7", THERMAL, STORM, GOLD]
    sns.pointplot(
        data=category,
        x="history_class",
        y="loss_abs",
        order=order,
        ax=axes[1, 0],
        color="black",
        errorbar=("ci", 95),
        join=False,
        markers="o",
        scale=0.7,
    )
    sns.stripplot(
        data=category,
        x="history_class",
        y="loss_abs",
        order=order,
        ax=axes[1, 0],
        palette=palette,
        alpha=0.2,
        size=2,
        jitter=0.25,
    )
    axes[1, 0].axhline(0, color="#999999", lw=0.8)
    axes[1, 0].set_xlabel("Previous 5-year history")
    axes[1, 0].set_ylabel("Absolute hard-coral loss (%)")
    axes[1, 0].tick_params(axis="x", rotation=25)
    panel_label(axes[1, 0], "c")

    # Panel d: Dose-response curves
    legacy_copy = legacy.copy()
    legacy_copy["hw_group"] = pd.cut(legacy_copy["heatwave_years_5yr"], bins=[-1, 0, 1, 5], labels=["0 yr", "1 yr", "2+ yr"])
    
    interaction_formula = (
        "loss_abs ~ baseline_hc_z + recent_max_wind_z + yrs_since_last_dist_z "
        "+ storm_years_5yr_z + C(event_type) + recent_max_dhw_z * C(hw_group)"
    )
    model_inter = robust_ols(legacy_copy, interaction_formula)
    
    mean_dhw = legacy_copy["recent_max_dhw"].mean()
    std_dhw = legacy_copy["recent_max_dhw"].std()
    
    group_colors = {"0 yr": "#1f77b4", "1 yr": "#ff7f0e", "2+ yr": "#2ca02c"}
    
    ax_d = axes[1, 1]
    for group in ["0 yr", "1 yr", "2+ yr"]:
        group_data = legacy_copy[legacy_copy["hw_group"].eq(group)]
        dhw_min = group_data["recent_max_dhw"].quantile(0.05)
        dhw_max = group_data["recent_max_dhw"].quantile(0.95)
        dhw_vals = np.linspace(dhw_min, dhw_max, 100)
        z_vals = (dhw_vals - mean_dhw) / std_dhw
        pred_df = pd.DataFrame({
            "recent_max_dhw_z": z_vals,
            "hw_group": [group] * len(z_vals),
            "baseline_hc_z": [0.0] * len(z_vals),
            "recent_max_wind_z": [0.0] * len(z_vals),
            "yrs_since_last_dist_z": [0.0] * len(z_vals),
            "storm_years_5yr_z": [0.0] * len(z_vals),
            "event_type": ["Heatwave_Only"] * len(z_vals)
        })
        
        pred_res = model_inter.get_prediction(pred_df).summary_frame()
        preds = pred_res["mean"]
        ci_lows = pred_res["mean_ci_lower"]
        ci_highs = pred_res["mean_ci_upper"]
        
        label = f"{group} heatwave history (n={len(group_data)})"
        ax_d.plot(dhw_vals, preds, color=group_colors[group], label=label, linewidth=1.4, zorder=3)
        ax_d.fill_between(dhw_vals, ci_lows, ci_highs, color=group_colors[group], alpha=0.15, zorder=2)
        
    ax_d.set_xlabel("Target-year DHW (degree C-weeks)")
    ax_d.set_ylabel("Fitted absolute cover loss (p.p.)")
    ax_d.set_xlim(0, 12)
    ax_d.set_ylim(-5, 22)
    ax_d.axhline(0, color="#999999", linestyle="--", linewidth=0.6)
    ax_d.legend(loc="upper left", frameon=False, fontsize=5.8)
    panel_label(ax_d, "d")

    fig.tight_layout(h_pad=1.35, w_pad=1.5)
    save_figure(fig, 4)


def figure_05(legacy):
    exposures = [
        ("cumulative_dhw", "Cumulative DHW"),
        ("heatwave_years", "Heatwave years"),
        ("max_consecutive_heatwave", "Consecutive heatwave years"),
        ("cumulative_wind", "Cumulative wind"),
        ("storm_years", "Storm years"),
        ("max_consecutive_storm", "Consecutive storm years"),
    ]
    rows = []
    for window in WINDOWS:
        for stem, label in exposures:
            exposure = f"{stem}_{window}yr"
            result = adjusted_effect(legacy, "loss_abs", exposure, "ols")
            result["window"] = window
            result["label"] = label
            rows.append(result)
    table = pd.DataFrame(rows)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]},
    )
    cax = inset_axes(
        axes[0],
        width="3.5%",
        height="74%",
        loc="center left",
        bbox_to_anchor=(1.015, 0, 1, 1),
        bbox_transform=axes[0].transAxes,
        borderpad=0,
    )

    pivot_coef = table.pivot(index="label", columns="window", values="coef").loc[[x[1] for x in exposures]]
    sns.heatmap(
        pivot_coef,
        ax=axes[0],
        cmap="vlag",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_ax=cax,
        cbar_kws={},
    )
    plt.setp(axes[0].get_yticklabels(), rotation=0, ha="right", fontsize=6.2)
    cax.set_title("Std.\ncoef.", fontsize=6.0, pad=3)
    axes[0].set_xlabel("Historical window (years)")
    axes[0].set_ylabel("")
    panel_label(axes[0], "a")

    thermal = table[table["label"].isin(["Cumulative DHW", "Heatwave years", "Consecutive heatwave years"])]
    for label, color, marker in [
        ("Cumulative DHW", "#8f8f8f", "o"),
        ("Heatwave years", THERMAL, "s"),
        ("Consecutive heatwave years", GOLD, "^"),
    ]:
        sub = thermal[thermal["label"].eq(label)]
        axes[1].errorbar(
            sub["window"],
            sub["coef"],
            yerr=[sub["coef"] - sub["low"], sub["high"] - sub["coef"]],
            color=color,
            marker=marker,
            lw=1.2,
            capsize=2.5,
            label=label,
        )
    axes[1].axhline(0, color="#888888", linestyle="--", lw=0.8)
    axes[1].set_xticks(list(WINDOWS))
    axes[1].set_xlabel("Historical window (years)")
    axes[1].set_ylabel("Standardized coefficient", labelpad=5)
    axes[1].set_xlim(2.5, 8.5)
    axes[1].set_ylim(-3.8, 0.8)
    axes[1].legend(loc="upper right", fontsize=6.0, frameon=True, facecolor='white', edgecolor='none', framealpha=0.8)
    panel_label(axes[1], "b")

    storm = table[table["label"].isin(["Cumulative wind", "Storm years", "Consecutive storm years"])]
    for label, color, marker in [
        ("Cumulative wind", "#8f8f8f", "o"),
        ("Storm years", STORM, "s"),
        ("Consecutive storm years", "#7aa0c4", "^"),
    ]:
        sub = storm[storm["label"].eq(label)]
        axes[2].errorbar(
            sub["window"],
            sub["coef"],
            yerr=[sub["coef"] - sub["low"], sub["high"] - sub["coef"]],
            color=color,
            marker=marker,
            lw=1.2,
            capsize=2.5,
            label=label,
        )
    axes[2].axhline(0, color="#888888", linestyle="--", lw=0.8)
    axes[2].set_xticks(list(WINDOWS))
    axes[2].set_xlabel("Historical window (years)")
    axes[2].set_ylabel("")
    axes[2].set_xlim(2.5, 8.5)
    axes[2].set_ylim(-1.8, 2.2)
    axes[2].legend(loc="upper right", fontsize=6.0, frameon=True, facecolor='white', edgecolor='none', framealpha=0.8)
    panel_label(axes[2], "c")

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.90, wspace=0.64)
    save_figure(fig, 5)


def add_box(ax, xy, text, color, width=0.24, height=0.16):
    x, y = xy
    rect = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="none", alpha=0.95)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color="white", fontsize=7)
    return rect


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555", shrinkA=2, shrinkB=2),
    )


def si_figure_03_residual_diagnostics(legacy):
    formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)"
    )
    model = robust_ols(legacy, formula)
    diagnostic = legacy.copy()
    diagnostic["fitted_loss"] = model.fittedvalues
    diagnostic["residual"] = model.resid

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.05))

    axes[0].scatter(diagnostic["fitted_loss"], diagnostic["residual"], s=10, color=NEUTRAL, alpha=0.42)
    sns.regplot(
        data=diagnostic,
        x="fitted_loss",
        y="residual",
        scatter=False,
        lowess=True,
        color=THERMAL,
        line_kws={"lw": 1.2},
        ax=axes[0],
    )
    axes[0].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[0].set_xlabel("Fitted absolute loss")
    axes[0].set_ylabel("Model residual")
    axes[0].set_title("Residuals vs fitted")

    axes[1].scatter(diagnostic["baseline_hc"], diagnostic["residual"], s=10, color=NEUTRAL, alpha=0.42)
    sns.regplot(
        data=diagnostic,
        x="baseline_hc",
        y="residual",
        scatter=False,
        lowess=True,
        color=STORM,
        line_kws={"lw": 1.2},
        ax=axes[1],
    )
    axes[1].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[1].set_xlabel("Baseline hard-coral cover (%)")
    axes[1].set_ylabel("Model residual")
    axes[1].set_title("Residuals vs baseline")

    bins = pd.qcut(diagnostic["baseline_hc"], q=5, duplicates="drop")
    binned = (
        diagnostic.assign(baseline_bin=bins)
        .groupby("baseline_bin", observed=True)
        .agg(
            baseline_mid=("baseline_hc", "median"),
            residual_sd=("residual", "std"),
            n=("residual", "size"),
        )
        .reset_index(drop=True)
    )
    axes[2].plot(binned["baseline_mid"], binned["residual_sd"], marker="o", color=GREEN, lw=1.2, ms=4)
    for _, row in binned.iterrows():
        axes[2].text(row["baseline_mid"], row["residual_sd"], f"n={int(row['n'])}", fontsize=6, ha="center", va="bottom")
    axes[2].set_xlabel("Baseline hard-coral cover (%)")
    axes[2].set_ylabel("Residual SD")
    axes[2].set_title("Residual spread by baseline")

    for label, ax in zip(["A", "B", "C"], axes):
        panel_label(ax, label)

    fig.tight_layout(w_pad=1.2)
    save_si_figure(fig, 3)


def si_figure_02_metric_framework():
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, (0.04, 0.66), "Prior recurrent\nheat exposure", THERMAL, width=0.20)
    add_box(ax, (0.04, 0.28), "Current DHW and\nwind exposure", STORM, width=0.20)

    add_box(ax, (0.34, 0.73), "Lower observable\nabsolute loss\nSTRONG", GOLD, width=0.25, height=0.18)
    add_box(ax, (0.34, 0.47), "Altered algal and\nmacroalgal baselines\nMODERATE", GREEN, width=0.25, height=0.18)
    add_box(ax, (0.34, 0.21), "Full-matrix proportional\nretention\nNO INCREASE", "#777777", width=0.25, height=0.18)

    add_box(ax, (0.70, 0.73), "Baseline stock constrains\nlater absolute loss\nSUPPORTED", "#4b6f8f", width=0.26, height=0.18)
    add_box(ax, (0.70, 0.47), "Community filtering or\ndisturbance legacy\nPLAUSIBLE, NOT ISOLATED", "#777777", width=0.26, height=0.18)
    add_box(ax, (0.70, 0.21), "Increased physiological\ntolerance\nNOT TESTED", "#999999", width=0.26, height=0.18)

    arrow(ax, (0.24, 0.74), (0.34, 0.82))
    arrow(ax, (0.24, 0.74), (0.34, 0.56))
    arrow(ax, (0.24, 0.74), (0.34, 0.30))
    ax.annotate(
        "",
        xy=(0.34, 0.77),
        xytext=(0.24, 0.40),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.2,
            color="#555555",
            connectionstyle="arc3,rad=-0.28",
            shrinkA=2,
            shrinkB=2,
        ),
    )
    arrow(ax, (0.59, 0.82), (0.70, 0.82))
    arrow(ax, (0.59, 0.56), (0.70, 0.56))
    arrow(ax, (0.59, 0.30), (0.70, 0.30))

    ax.text(0.04, 0.94, "Exposure history", fontsize=8, fontweight="bold", color="#222222")
    ax.text(0.34, 0.94, "Evidence in this study", fontsize=8, fontweight="bold", color="#222222")
    ax.text(0.70, 0.94, "Interpretive status", fontsize=8, fontweight="bold", color="#222222")
    ax.text(
        0.04,
        0.05,
        "Arrows organize competing interpretations; they do not represent an identified causal pathway.",
        fontsize=6.5,
        color="#333333",
    )
    save_si_figure(fig, 2)


def si_figure_06_smooth_heterogeneity(legacy):
    effects = pd.read_csv(SPLINE_EFFECTS_PATH)
    tests = pd.read_csv(SPLINE_TESTS_PATH).set_index("response")
    response_specs = [
        ("loss_abs", "Absolute hard-coral cover loss", "P < 0.001"),
        ("retention", "Raw proportional retention", "P = 0.149"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    for ax, (response, title, p_label) in zip(axes, response_specs):
        sub = effects.loc[effects["response"].eq(response)].sort_values("baseline_hc")
        if len(sub) != 80 or int(tests.loc[response, "n_observations"]) != 525:
            raise ValueError(f"Unexpected spline source data for {response}")

        ax.fill_between(
            sub["baseline_hc"],
            sub["ci_low"],
            sub["ci_high"],
            color=THERMAL,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            sub["baseline_hc"],
            sub["recurrence_effect_per_1sd"],
            color=THERMAL,
            linewidth=1.6,
        )
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)

        x_min = float(sub["baseline_hc"].min())
        x_max = float(sub["baseline_hc"].max())
        rug = legacy.loc[legacy["baseline_hc"].between(x_min, x_max), "baseline_hc"]
        ax.plot(
            rug,
            np.full(len(rug), 0.018),
            "|",
            transform=ax.get_xaxis_transform(),
            color=NEUTRAL,
            alpha=0.18,
            markersize=3.4,
            markeredgewidth=0.55,
            clip_on=True,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Baseline hard-coral cover (%)")
        ax.set_title(title, fontweight="bold", pad=5)
        ax.text(
            0.98,
            0.96,
            f"Spline interaction\n{p_label}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#444444",
        )
        panel_label(ax, "A" if response == "loss_abs" else "B")

    fig.supylabel(
        "Marginal association per SD increase\nin prior heatwave years",
        x=0.005,
        fontsize=8,
    )
    fig.tight_layout(w_pad=1.7, rect=(0.035, 0, 1, 1))
    save_si_figure(fig, 6)


def main():
    print("Loading data...")
    legacy = prepare_matrix(pd.read_csv(MATRIX_PATH))
    eco = pd.read_csv(os.path.join(DATA_DIR, "eco_response_master_matrix_merged.csv"))

    print("Generating Figure 2...")
    figure_02(eco, legacy)
    print("Generating Figure 3...")
    figure_03(legacy)
    print("Generating Figure 4...")
    figure_04(legacy)
    print("Generating Figure 5...")
    figure_05(legacy)
    print("Generating SI Figure 2...")
    si_figure_02_metric_framework()
    print("Generating SI Figure 3...")
    si_figure_03_residual_diagnostics(legacy)
    print("Generating SI Figure 6...")
    si_figure_06_smooth_heterogeneity(legacy)
    print("Figures 2-5 and SI Figures 2-3 and 6 generated.")


if __name__ == "__main__":
    main()
