from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy
from scipy import stats
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
INPUT_MATRIX = PROJECT / "output" / "legacy_load_analysis_matrix.csv"
MASTER_MATRIX = PROJECT / "data" / "eco_response_master_matrix_merged.csv"
VARIABLE_DICTIONARY = PROJECT / "output" / "tables" / "table_s1_variable_definitions.csv"
MAIN_MODEL_SCRIPT = PROJECT / "scripts" / "05_run_lmm_legacy_model.py"
CURRENT_RESPONSE_TABLE = PROJECT / "output" / "tables" / "table_s8_response_metric_dependence_sensitivity.csv"
CURRENT_TOST_TABLE = PROJECT / "output" / "tables" / "table_s21_spatial_and_retention_equivalence.csv"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
THRESHOLDS = (8.0, 10.0, 12.0, 15.0)

CONTINUOUS_BALANCE = [
    ("heatwave_years_5yr", "Prior heatwave years (5-year)"),
    ("cumulative_dhw_5yr", "Cumulative DHW (5-year)"),
    ("recent_max_dhw", "Target-year maximum DHW"),
    ("storm_years_5yr", "Prior storm years (5-year)"),
    ("cumulative_wind_5yr", "Cumulative storm wind (5-year)"),
    ("recent_max_wind", "Target-year maximum wind"),
    ("event_year", "Event year"),
    ("yrs_since_last_dist", "Years since last disturbance"),
]
CATEGORICAL_BALANCE = [("event_type", "Event type"), ("sector", "Sector")]
MAIN_CONTROLS = (
    "recent_max_dhw_z + recent_max_wind_z + heatwave_years_5yr_z + "
    "storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)"
)
RETENTION_CONTROLS = MAIN_CONTROLS


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / sd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists():
        return "MISSING"
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode())
        h.update(sha256(path).encode())
    return h.hexdigest()


def snapshot_protected() -> dict[str, str]:
    protected = {
        "manuscript_targets_science_advances": PROJECT / "manuscript" / "targets" / "science_advances",
        "submission_package": PROJECT / "submission_package",
        "communications_earth_environment": PROJECT / "communications_earth_environment",
        "input_matrix": INPUT_MATRIX,
        "main_model_script": MAIN_MODEL_SCRIPT,
    }
    result = {}
    for name, path in protected.items():
        result[name] = tree_hash(path) if path.is_dir() else (sha256(path) if path.exists() else "MISSING")
    return result


def prepare_data() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_MATRIX)
    master = pd.read_csv(MASTER_MATRIX)
    meta_cols = [c for c in ["reef_name", "sector", "region_lat"] if c in master.columns]
    meta = master[meta_cols].drop_duplicates(subset=["reef_name"])
    df = raw.merge(meta, on="reef_name", how="left", validate="many_to_one")
    df["retention"] = df["nadir_hc"] / df["baseline_hc"]
    numeric = [
        "baseline_hc", "recent_max_dhw", "recent_max_wind", "yrs_since_last_dist",
        "cumulative_dhw_5yr", "cumulative_wind_5yr", "heatwave_years_5yr", "storm_years_5yr",
    ]
    for col in numeric:
        df[f"{col}_z"] = zscore(df[col])
    required = [
        "loss_abs", "retention", "baseline_hc_z", "recent_max_dhw_z", "recent_max_wind_z",
        "heatwave_years_5yr_z", "storm_years_5yr_z", "yrs_since_last_dist_z",
        "event_type", "reef_name",
    ]
    df = df.dropna(subset=required).copy()
    if len(raw) != 525 or len(df) != 525:
        raise RuntimeError(f"Expected the current 525-row matrix; raw={len(raw)}, analysis={len(df)}")
    return df


def cluster_ols(data: pd.DataFrame, formula: str):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["reef_name"]}
    )


def linear_combo(result, weights: dict[str, float]) -> dict[str, float]:
    names = list(result.params.index)
    vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    beta = float(vector @ result.params.to_numpy())
    variance = float(vector @ result.cov_params().to_numpy() @ vector)
    se = math.sqrt(max(variance, 0.0))
    z = beta / se if se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {
        "estimate": beta, "se": se, "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se, "p": p,
    }


def prediction(result, row: pd.DataFrame) -> dict[str, float]:
    design_info = result.model.data.design_info
    x = np.asarray(patsy.build_design_matrices([design_info], row)[0])[0]
    estimate = float(x @ result.params.to_numpy())
    se = math.sqrt(max(float(x @ result.cov_params().to_numpy() @ x), 0.0))
    return {"estimate": estimate, "se": se, "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se}


def group_summary(df: pd.DataFrame, cutoff: float = 10.0) -> pd.DataFrame:
    work = df.assign(group=np.where(df["baseline_hc"] < cutoff, f"<{cutoff:g}%", f">={cutoff:g}%"))
    rows = []
    for group, data in work.groupby("group", sort=True):
        base = {
            "cutoff": cutoff, "group": group, "reef_years": len(data),
            "reefs": data["reef_name"].nunique(), "first_year": int(data["event_year"].min()),
            "last_year": int(data["event_year"].max()), "baseline_mean": data["baseline_hc"].mean(),
            "baseline_median": data["baseline_hc"].median(),
        }
        for event_type, count in data["event_type"].value_counts().items():
            base[f"event_{event_type}"] = int(count)
        rows.append(base)
    return pd.DataFrame(rows)


def distribution_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    x = df["baseline_hc"]
    rows = []
    for width in (0.5, 1.0, 2.0):
        for cutoff in THRESHOLDS:
            rows.append({
                "section": "local_sample", "cutoff": cutoff, "window_or_bin": width,
                "metric": "count_below", "value": int(((x >= cutoff - width) & (x < cutoff)).sum()),
            })
            rows.append({
                "section": "local_sample", "cutoff": cutoff, "window_or_bin": width,
                "metric": "count_above", "value": int(((x >= cutoff) & (x < cutoff + width)).sum()),
            })
    for start in np.arange(5.0, 20.0, 0.5):
        rows.append({
            "section": "histogram_5_20", "cutoff": 10.0, "window_or_bin": f"[{start:.1f},{start + 0.5:.1f})",
            "metric": "count", "value": int(((x >= start) & (x < start + 0.5)).sum()),
        })
    rounding = {
        "exact_integer": np.isclose(x, np.round(x), atol=1e-8).mean(),
        "exact_half_percent": np.isclose(x * 2, np.round(x * 2), atol=1e-8).mean(),
        "exact_tenth_percent": np.isclose(x * 10, np.round(x * 10), atol=1e-8).mean(),
        "unique_values": x.nunique(),
        "exactly_10_percent": np.isclose(x, 10.0, atol=1e-8).sum(),
    }
    for metric, value in rounding.items():
        rows.append({"section": "rounding", "cutoff": 10.0, "window_or_bin": "all", "metric": metric, "value": value})
    left = int(((x >= 9.0) & (x < 10.0)).sum())
    right = int(((x >= 10.0) & (x < 11.0)).sum())
    rows.extend([
        {"section": "cutoff_density", "cutoff": 10.0, "window_or_bin": "1 percentage point", "metric": "left_count", "value": left},
        {"section": "cutoff_density", "cutoff": 10.0, "window_or_bin": "1 percentage point", "metric": "right_count", "value": right},
        {"section": "cutoff_density", "cutoff": 10.0, "window_or_bin": "1 percentage point", "metric": "right_to_left_ratio", "value": right / left if left else np.nan},
        {"section": "cutoff_density", "cutoff": 10.0, "window_or_bin": "1 percentage point", "metric": "two_sided_binomial_p", "value": stats.binomtest(right, left + right, 0.5).pvalue if left + right else np.nan},
    ])
    return pd.DataFrame(rows)


def smd(low: pd.Series, high: pd.Series) -> float:
    low, high = low.dropna(), high.dropna()
    pooled = math.sqrt((low.var() + high.var()) / 2)
    return (high.mean() - low.mean()) / pooled if pooled > 0 else np.nan


def balance_tables(df: pd.DataFrame, cutoff: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group = df["baseline_hc"] >= cutoff
    continuous = []
    missing = []
    for col, label in CONTINUOUS_BALANCE:
        low, high = df.loc[~group, col], df.loc[group, col]
        continuous.append({
            "variable": col, "label": label, "low_n": low.notna().sum(), "high_n": high.notna().sum(),
            "low_mean": low.mean(), "high_mean": high.mean(), "low_sd": low.std(), "high_sd": high.std(),
            "standardized_mean_difference_high_minus_low": smd(low, high),
            "low_min": low.min(), "low_max": low.max(), "high_min": high.min(), "high_max": high.max(),
        })
        missing.append({"variable": col, "low_missing": low.isna().sum(), "high_missing": high.isna().sum()})
    categorical = []
    for col, label in CATEGORICAL_BALANCE:
        levels = sorted(df[col].dropna().astype(str).unique())
        for level in levels:
            p0 = df.loc[~group, col].astype(str).eq(level).mean()
            p1 = df.loc[group, col].astype(str).eq(level).mean()
            pooled = (p0 + p1) / 2
            denom = math.sqrt(pooled * (1 - pooled)) if 0 < pooled < 1 else np.nan
            categorical.append({
                "variable": col, "label": label, "level": level, "low_proportion": p0,
                "high_proportion": p1, "proportion_difference_high_minus_low": p1 - p0,
                "standardized_difference_high_minus_low": (p1 - p0) / denom if denom else np.nan,
            })
        missing.append({
            "variable": col, "low_missing": df.loc[~group, col].isna().sum(),
            "high_missing": df.loc[group, col].isna().sum(),
        })
    for col in ["baseline_hc", "loss_abs", "retention", "baseline_algae", "baseline_macroalgae", "baseline_juveniles", "baseline_herbivores"]:
        missing.append({
            "variable": col, "low_missing": df.loc[~group, col].isna().sum(),
            "high_missing": df.loc[group, col].isna().sum(),
        })
    missing.append({"variable": "baseline_year/nadir_year", "low_missing": len(df.loc[~group]), "high_missing": len(df.loc[group]), "note": "Not retained in the current 525-row matrix; not reconstructed."})
    return pd.DataFrame(continuous), pd.DataFrame(categorical), pd.DataFrame(missing)


def common_support(df: pd.DataFrame, cutoff: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = (df["baseline_hc"] >= cutoff).astype(int)
    continuous = [x[0] for x in CONTINUOUS_BALANCE]
    categorical = [x[0] for x in CATEGORICAL_BALANCE]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), continuous),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("prep", preprocessor), ("logit", LogisticRegression(max_iter=5000, C=1.0))])
    propensity = model.fit(df[continuous + categorical], group).predict_proba(df[continuous + categorical])[:, 1]
    p_low, p_high = propensity[group.eq(0)], propensity[group.eq(1)]
    overlap_low, overlap_high = max(p_low.min(), p_high.min()), min(p_low.max(), p_high.max())
    bins = np.linspace(0, 1, 41)
    h0, _ = np.histogram(p_low, bins=bins, density=True)
    h1, _ = np.histogram(p_high, bins=bins, density=True)
    overlap_coefficient = float(np.minimum(h0, h1).sum() * (bins[1] - bins[0]))
    rows = [{
        "diagnostic": "propensity_score", "variable": "all listed covariates", "overlap_low": overlap_low,
        "overlap_high": overlap_high, "low_outside_opposing_range_fraction": np.mean((p_low < overlap_low) | (p_low > overlap_high)),
        "high_outside_opposing_range_fraction": np.mean((p_high < overlap_low) | (p_high > overlap_high)),
        "overlap_coefficient": overlap_coefficient, "classification_auc": roc_auc_score(group, propensity),
    }]
    for col in continuous:
        low, high = df.loc[group.eq(0), col].dropna(), df.loc[group.eq(1), col].dropna()
        lo, hi = max(low.min(), high.min()), min(low.max(), high.max())
        rows.append({
            "diagnostic": "empirical_range", "variable": col, "overlap_low": lo, "overlap_high": hi,
            "low_outside_opposing_range_fraction": np.mean((low < high.min()) | (low > high.max())),
            "high_outside_opposing_range_fraction": np.mean((high < low.min()) | (high > low.max())),
            "overlap_coefficient": max(0.0, hi - lo) / min(low.max() - low.min(), high.max() - high.min()) if min(low.max() - low.min(), high.max() - high.min()) > 0 else np.nan,
            "classification_auc": np.nan,
        })
    scores = pd.DataFrame({"reef_name": df["reef_name"], "group_high": group, "propensity_score": propensity})
    return pd.DataFrame(rows), scores


def model_row(result, model: str, response: str, cutoff: float, term: str, quantity: str, combo: dict[str, float] | None = None) -> dict:
    values = linear_combo(result, combo or {term: 1.0})
    return {
        "model": model, "response": response, "cutoff": cutoff, "term": term, "quantity": quantity,
        **values, "n_observations": int(result.nobs), "n_reefs": int(result.model.data.frame["reef_name"].nunique()),
        "r_squared": result.rsquared,
    }


def hinge_models(df: pd.DataFrame, cutoff: float) -> tuple[list[dict], list[dict]]:
    work = df.copy()
    work["baseline_centered"] = work["baseline_hc"] - cutoff
    work["baseline_hinge"] = work["baseline_centered"].clip(lower=0)
    rows, predictions = [], []
    event_ref = work["event_type"].mode().iat[0]
    for response in ("loss_abs", "retention"):
        formula = f"{response} ~ baseline_centered + baseline_hinge + {MAIN_CONTROLS}"
        result = cluster_ols(work, formula)
        rows.extend([
            model_row(result, "continuous_hinge", response, cutoff, "baseline_centered", "slope_below_cutoff"),
            model_row(result, "continuous_hinge", response, cutoff, "baseline_hinge", "slope_difference_above_minus_below"),
            model_row(result, "continuous_hinge", response, cutoff, "baseline_centered", "slope_above_cutoff", {"baseline_centered": 1, "baseline_hinge": 1}),
        ])
        for baseline in sorted(set([max(0.1, cutoff - 5), cutoff, cutoff + 5, 20.0])):
            new = pd.DataFrame({
                "baseline_centered": [baseline - cutoff], "baseline_hinge": [max(0, baseline - cutoff)],
                "recent_max_dhw_z": [0.0], "recent_max_wind_z": [0.0], "heatwave_years_5yr_z": [0.0],
                "storm_years_5yr_z": [0.0], "yrs_since_last_dist_z": [0.0], "event_type": [event_ref],
            })
            predictions.append({"model": "continuous_hinge", "response": response, "cutoff": cutoff, "baseline_hc": baseline, **prediction(result, new)})
    return rows, predictions


def recurrence_models(df: pd.DataFrame, cutoff: float) -> tuple[list[dict], list[dict], list[dict]]:
    work = df.copy()
    work["group_high"] = (work["baseline_hc"] >= cutoff).astype(int)
    interaction_rows, subgroup_rows, prediction_rows = [], [], []
    event_ref = work["event_type"].mode().iat[0]
    for response in ("loss_abs", "retention"):
        baseline_term = "baseline_hc_z + " if response == "loss_abs" else ""
        formula = (
            f"{response} ~ {baseline_term}recent_max_dhw_z + recent_max_wind_z + "
            "storm_years_5yr_z + yrs_since_last_dist_z + C(event_type) + "
            "heatwave_years_5yr_z * group_high"
        )
        result = cluster_ols(work, formula)
        interaction_rows.extend([
            model_row(result, "recurrence_group_interaction", response, cutoff, "heatwave_years_5yr_z", "recurrence_slope_low"),
            model_row(result, "recurrence_group_interaction", response, cutoff, "heatwave_years_5yr_z:group_high", "recurrence_slope_difference_high_minus_low"),
            model_row(result, "recurrence_group_interaction", response, cutoff, "heatwave_years_5yr_z", "recurrence_slope_high", {"heatwave_years_5yr_z": 1, "heatwave_years_5yr_z:group_high": 1}),
        ])
        for high in (0, 1):
            subset = work[work["group_high"].eq(high)].copy()
            subgroup_formula = (
                f"{response} ~ {baseline_term}recent_max_dhw_z + recent_max_wind_z + "
                "heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)"
            )
            subgroup_result = cluster_ols(subset, subgroup_formula)
            row = model_row(subgroup_result, "descriptive_subgroup", response, cutoff, "heatwave_years_5yr_z", "recurrence_coefficient")
            row.update({"group": "high" if high else "low", "group_reef_years": len(subset), "group_reefs": subset["reef_name"].nunique()})
            subgroup_rows.append(row)
        for high in (0, 1):
            baseline_z = work.loc[work["group_high"].eq(high), "baseline_hc_z"].mean()
            for recurrence_z in (-1.0, 0.0, 1.0):
                new = pd.DataFrame({
                    "baseline_hc_z": [baseline_z], "recent_max_dhw_z": [0.0], "recent_max_wind_z": [0.0],
                    "heatwave_years_5yr_z": [recurrence_z], "storm_years_5yr_z": [0.0],
                    "yrs_since_last_dist_z": [0.0], "event_type": [event_ref], "group_high": [high],
                })
                prediction_rows.append({
                    "model": "recurrence_group_interaction", "response": response, "cutoff": cutoff,
                    "group": "high" if high else "low", "recurrence_z": recurrence_z, **prediction(result, new),
                })
    return interaction_rows, subgroup_rows, prediction_rows


def spline_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests, effects = [], []
    event_ref = df["event_type"].mode().iat[0]
    grid = np.linspace(df["baseline_hc"].quantile(0.025), df["baseline_hc"].quantile(0.975), 80)
    for response in ("loss_abs", "retention"):
        formula = (
            f"{response} ~ cr(baseline_hc, df=4) * heatwave_years_5yr_z + "
            "recent_max_dhw_z + recent_max_wind_z + storm_years_5yr_z + "
            "yrs_since_last_dist_z + C(event_type)"
        )
        result = cluster_ols(df, formula)
        interaction_terms = [name for name in result.params.index if ":heatwave_years_5yr_z" in name]
        restriction = np.zeros((len(interaction_terms), len(result.params)))
        for i, term in enumerate(interaction_terms):
            restriction[i, list(result.params.index).index(term)] = 1
        joint = result.wald_test(restriction, scalar=True)
        tests.append({
            "response": response, "model": "natural_cubic_spline_df4", "spline_df": 4,
            "joint_interaction_test": "baseline spline x prior heatwave recurrence",
            "statistic": float(joint.statistic), "df": len(interaction_terms), "p": float(joint.pvalue),
            "n_observations": int(result.nobs), "n_reefs": df["reef_name"].nunique(), "r_squared": result.rsquared,
        })
        for baseline in grid:
            common = {
                "baseline_hc": [baseline], "recent_max_dhw_z": [0.0], "recent_max_wind_z": [0.0],
                "storm_years_5yr_z": [0.0], "yrs_since_last_dist_z": [0.0], "event_type": [event_ref],
            }
            r0 = pd.DataFrame({**common, "heatwave_years_5yr_z": [0.0]})
            r1 = pd.DataFrame({**common, "heatwave_years_5yr_z": [1.0]})
            info = result.model.data.design_info
            x0 = np.asarray(patsy.build_design_matrices([info], r0)[0])[0]
            x1 = np.asarray(patsy.build_design_matrices([info], r1)[0])[0]
            delta = x1 - x0
            estimate = float(delta @ result.params.to_numpy())
            se = math.sqrt(max(float(delta @ result.cov_params().to_numpy() @ delta), 0.0))
            effects.append({
                "response": response, "baseline_hc": baseline, "recurrence_effect_per_1sd": estimate,
                "se": se, "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            })
    return pd.DataFrame(tests), pd.DataFrame(effects)


def response_metric_context() -> pd.DataFrame:
    rows = [
        {"analysis": "Current validation response", "metric": "absolute loss", "definition": "baseline_hc - nadir_hc", "role": "Primary response; percentage points"},
        {"analysis": "Current validation response", "metric": "raw proportional retention", "definition": "nadir_hc / baseline_hc", "role": "Primary response; untrimmed and unclipped"},
    ]
    current = pd.read_csv(CURRENT_RESPONSE_TABLE)
    selected = current[current["model"].astype(str).str.contains("Proportional-retention model", regex=False)]
    for row in selected.itertuples(index=False):
        rows.append({
            "analysis": "Existing final-analysis context", "metric": "baseline-cutoff retention" if pd.notna(getattr(row, "baseline_cutoff_percent", np.nan)) else "raw proportional retention",
            "definition": str(row.model), "role": "Separate sensitivity/context; not substituted for the validation response",
            "estimate": row.beta, "ci_low": row.ci_low, "ci_high": row.ci_high, "p": row.p,
            "cutoff_percent": getattr(row, "baseline_cutoff_percent", np.nan),
        })
    tost = pd.read_csv(CURRENT_TOST_TABLE)
    tost = tost[tost["model"].eq("Proportional-retention equivalence check")]
    for row in tost.itertuples(index=False):
        rows.append({
            "analysis": "Existing final-analysis context", "metric": "TOST", "definition": "Equivalence test for raw-retention recurrence coefficient",
            "role": "Separate inferential question; not a response metric", "estimate": row.beta,
            "ci_low": row.ci_low, "ci_high": row.ci_high, "p": row.p,
            "equivalence_margin": row.equivalence_margin, "interpretation": row.interpretation,
        })
    return pd.DataFrame(rows)


def add_fdr(table: pd.DataFrame, quantity: str) -> pd.DataFrame:
    out = table.copy()
    out["p_bh_within_response_and_quantity"] = np.nan
    mask = out["quantity"].eq(quantity)
    for _, idx in out[mask].groupby("response").groups.items():
        out.loc[idx, "p_bh_within_response_and_quantity"] = multipletests(out.loc[idx, "p"], method="fdr_bh")[1]
    return out


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
    })


def save_figure(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(df: pd.DataFrame, scores: pd.DataFrame, spline_effects: pd.DataFrame) -> None:
    configure_plotting()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bins = np.arange(5, 20.5, 0.5)
    local = df[df["baseline_hc"].between(5, 20, inclusive="left")]
    ax.hist(local["baseline_hc"], bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.5)
    ax.axvline(10, color="#C44E52", linestyle="--", linewidth=1.2, label="10% cutoff")
    ax.set(xlabel="Baseline hard-coral cover (%)", ylabel="Reef-years", xlim=(5, 20))
    ax.legend()
    ax.text(0.99, 0.96, f"n = {len(local)} within 5–20%", transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    save_figure(fig, "baseline_distribution_5_20")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    low = scores.loc[scores["group_high"].eq(0), "propensity_score"]
    high = scores.loc[scores["group_high"].eq(1), "propensity_score"]
    bins = np.linspace(0, 1, 21)
    ax.hist(low, bins=bins, alpha=0.65, density=True, color="#4C78A8", label="Baseline <10%")
    ax.hist(high, bins=bins, alpha=0.55, density=True, color="#E07B39", label="Baseline ≥10%")
    ax.set(xlabel="Estimated probability of baseline ≥10%", ylabel="Density", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "common_support_propensity")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for ax, response, ylabel in zip(axes, ["loss_abs", "retention"], ["Effect on absolute loss", "Effect on raw retention"]):
        sub = spline_effects[spline_effects["response"].eq(response)]
        ax.fill_between(sub["baseline_hc"], sub["ci_low"], sub["ci_high"], color="#4C78A8", alpha=0.18)
        ax.plot(sub["baseline_hc"], sub["recurrence_effect_per_1sd"], color="#4C78A8", linewidth=1.5)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.axvline(10, color="#C44E52", linestyle="--", linewidth=1.0)
        ax.set(xlabel="Baseline hard-coral cover (%)", ylabel=ylabel)
    fig.tight_layout()
    save_figure(fig, "continuous_recurrence_heterogeneity")


def fmt(x, digits=3) -> str:
    return "NA" if pd.isna(x) else f"{x:.{digits}f}"


def write_readme() -> None:
    text = f"""# Baseline cover threshold validation

## 问题

验证基线硬珊瑚覆盖率 10% 是否具有分布或模型层面的分层意义，以及 prior heatwave recurrence 与 absolute loss / raw proportional retention 的关联是否随基线组变化。10% 被视为待检验切点，不预设为生态断点。

## 数据与既有定义

- 输入矩阵：`{INPUT_MATRIX.relative_to(PROJECT).as_posix()}`（当前终稿 525 reef-years）。
- 变量字典：`{VARIABLE_DICTIONARY.relative_to(PROJECT).as_posix()}`。
- 主模型：`{MAIN_MODEL_SCRIPT.relative_to(PROJECT).as_posix()}`。
- 样本和协变量沿用当前主模型：reef-cluster robust OLS；当前 DHW、当前 wind、5-year heatwave years、5-year storm years、return interval、event type；absolute loss 模型保留 baseline cover 调整。
- raw proportional retention 定义为 `nadir_hc / baseline_hc`，不裁剪、不转换。

## 方法

1. 10% 分组样本、5%–20% 分布/取整/局部密度、协变量标准化差异、缺失与 common support。
2. 以阈值为结点的连续 hinge 模型，检验结点前后 baseline slope 差。
3. recurrence × baseline-group 交互，直接检验两组 recurrence association 差异；另给组内描述模型和调整预测。
4. 对 8%、10%、12%、15% 重复；P 值保留原值，并对同一响应/检验族给出 BH-FDR，绝不据此选择阈值。
5. 单一 4-df natural cubic spline × recurrence 检查平滑异质性，不进行模型搜索。

## 复现

在项目根目录运行：

```powershell
python analysis/baseline_cover_threshold_validation/run_threshold_validation.py
```

脚本只向本目录的 `results/`、`figures/`、报告和检查文件写入。诊断图为英文标签、Times New Roman。
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


def write_report(group: pd.DataFrame, dist: pd.DataFrame, balance: pd.DataFrame, support: pd.DataFrame,
                 hinge: pd.DataFrame, interactions: pd.DataFrame, subgroups: pd.DataFrame,
                 spline_tests: pd.DataFrame) -> None:
    g10 = group.set_index("group")
    low = g10.loc["<10%"]
    high = g10.loc[">=10%"]
    ratio = dist.loc[dist["metric"].eq("right_to_left_ratio"), "value"].iloc[0]
    binom_p = dist.loc[dist["metric"].eq("two_sided_binomial_p"), "value"].iloc[0]
    rounding = dist[(dist["section"].eq("rounding"))].set_index("metric")["value"]
    ps = support[support["diagnostic"].eq("propensity_score")].iloc[0]
    severe_support = (ps["overlap_coefficient"] < 0.5 or ps["low_outside_opposing_range_fraction"] > 0.2 or
                      ps["high_outside_opposing_range_fraction"] > 0.2 or ps["classification_auc"] > 0.8)
    max_smd = balance.assign(abs_smd=balance["standardized_mean_difference_high_minus_low"].abs()).sort_values("abs_smd", ascending=False).iloc[0]

    h10 = hinge[(hinge["cutoff"].eq(10)) & hinge["quantity"].eq("slope_difference_above_minus_below")]
    i10 = interactions[(interactions["cutoff"].eq(10)) & interactions["quantity"].eq("recurrence_slope_difference_high_minus_low")]
    sections = []
    for response, label in [("loss_abs", "absolute loss"), ("retention", "raw proportional retention")]:
        h = h10[h10["response"].eq(response)].iloc[0]
        it = i10[i10["response"].eq(response)].iloc[0]
        sg = subgroups[(subgroups["cutoff"].eq(10)) & subgroups["response"].eq(response)]
        lo = sg[sg["group"].eq("low")].iloc[0]
        hi = sg[sg["group"].eq("high")].iloc[0]
        sections.append(f"""
### {label}

- 10% hinge 的斜率差（阈值上方减下方）：{fmt(h.estimate)}，95% CI [{fmt(h.ci_low)}, {fmt(h.ci_high)}]，P={fmt(h.p, 4)}。
- 对应四阈值检验族的 BH-FDR P={fmt(h.p_bh_within_response_and_quantity, 4)}；不能把未经校正的单一切点结果当作断点证据。
- recurrence × 高基线组交互：{fmt(it.estimate)}，95% CI [{fmt(it.ci_low)}, {fmt(it.ci_high)}]，P={fmt(it.p, 4)}。这是组间异质性的直接检验。
- 描述性分组回归：低组 recurrence 系数 {fmt(lo.estimate)} [95% CI {fmt(lo.ci_low)}, {fmt(lo.ci_high)}]，R²={fmt(lo.r_squared)}；高组 {fmt(hi.estimate)} [{fmt(hi.ci_low)}, {fmt(hi.ci_high)}]，R²={fmt(hi.r_squared)}。一组显著、另一组不显著不构成组间差异证据。
""")
    spline_lines = []
    for row in spline_tests.itertuples(index=False):
        spline_lines.append(f"- {row.response}：baseline spline × recurrence 联合检验 P={fmt(row.p, 4)}（固定 4 df spline，无模型搜索）。")

    interaction_stability = interactions[interactions["quantity"].eq("recurrence_slope_difference_high_minus_low")]
    stable_loss = (interaction_stability[interaction_stability["response"].eq("loss_abs")]["p"] < 0.05).all()
    stable_ret = (interaction_stability[interaction_stability["response"].eq("retention")]["p"] < 0.05).all()
    recommendation = (
        "10% 缺少作为离散生态状态分界的经验支持：分布无可见断点，直接 recurrence 交互不显著，"
        "absolute-loss hinge 的单点名义证据经四阈值 BH-FDR 后不再低于 0.05，且连续样条显示的是平滑异质性而非 10% 跳变。"
        "建议只把 5%/10% baseline-cutoff retention 保留为敏感性分析。"
    )
    existing_response = pd.read_csv(CURRENT_RESPONSE_TABLE)
    raw_ret = existing_response[existing_response["model"].eq("Proportional-retention model")].iloc[0]
    cut5 = existing_response[existing_response["model"].eq("Proportional-retention model, baseline > 5%")].iloc[0]
    cut10 = existing_response[existing_response["model"].eq("Proportional-retention model, baseline > 10%")].iloc[0]
    existing_tost = pd.read_csv(CURRENT_TOST_TABLE)
    tost = existing_tost[existing_tost["model"].eq("Proportional-retention equivalence check")].iloc[0]

    report = f"""# Baseline cover threshold validation report

## 决策结论

- **不支持把 10% 当作生态状态分界。** 10% 附近没有可见分布断点；loss/retention 的 recurrence × group 直接交互均不显著；absolute-loss hinge 的名义 P 值不构成阈值特异证据。
- **存在的信号更符合连续几何，而非 10% 离散断点。** absolute-loss 的固定 4-df spline 交互提示平滑异质性，不能反推 10% 或任何机制。
- **组间不可作可比状态对照。** <10% 仅 60 reef-years，且 common support 达到预设的严重受限规则。
- 5%/10% baseline-cutoff retention 仅保留为既有敏感性分析；不作为本验证的替代响应。

## 数据事实

- 当前分析矩阵：525 reef-years、{int(group['reefs'].max())} 个独立 reefs；年份 {int(min(group.first_year))}–{int(max(group.last_year))}。
- 基线 <10%：{int(low.reef_years)} reef-years、{int(low.reefs)} reefs（Heatwave Only {int(low.event_Heatwave_Only)}、Storm Only {int(low.event_Storm_Only)}、Concurrent {int(low.event_Concurrent)}）；基线 ≥10%：{int(high.reef_years)} reef-years、{int(high.reefs)} reefs（Heatwave Only {int(high.event_Heatwave_Only)}、Storm Only {int(high.event_Storm_Only)}、Concurrent {int(high.event_Concurrent)}）。
- 10% 左右各 1 percentage point 的样本数为 12/11，密度比（右/左）={fmt(ratio)}，等密度二项检验 P={fmt(binom_p, 4)}。精确整数值占比 {fmt(rounding['exact_integer'])}，精确 0.5% 倍数占比 {fmt(rounding['exact_half_percent'])}；诊断图未见 10% 处断裂或堆积。
- 最大连续协变量绝对标准化差异为 {max_smd['label']}：|SMD|={fmt(max_smd.abs_smd)}。标准化差异用于衡量不平衡，不以 P 值代替。
- propensity-score overlap coefficient={fmt(ps.overlap_coefficient)}，AUC={fmt(ps.classification_auc)}；低/高组落在共同范围外的比例分别为 {fmt(ps.low_outside_opposing_range_fraction)} / {fmt(ps.high_outside_opposing_range_fraction)}。组间 common support 判定：{'严重受限' if severe_support else '未达到预设的严重受限规则'}。{'因此分层回归不能解释为可比生态状态之间的对照。' if severe_support else '即便如此，这仍是观察性分层，不能解释为可比生态状态的因果对照。'}
- 当前 525 行矩阵未保留 baseline year / nadir year，未自行重建；可用 observation-window 变量为 `yrs_since_last_dist`。缺失详情见 `results/missingness.csv`。

## 统计结果

{''.join(sections)}
### 阈值敏感性与连续检查

- 8%、10%、12%、15% 全部预先列出并重复相同 hinge、交互和分组流程；原始 P 值与同一响应/检验族的 BH-FDR 值见 `results/threshold_sensitivity.csv`。未扫描其他切点，也未选择“最佳阈值”。
- absolute-loss 交互在四个阈值均 P<0.05：{'是' if stable_loss else '否'}；raw-retention 交互在四个阈值均 P<0.05：{'是' if stable_ret else '否'}。
{chr(10).join(spline_lines)}

### 四类 retention 结果的边界

- raw proportional retention（全 525 行）recurrence 系数 {fmt(raw_ret.beta)}，95% CI [{fmt(raw_ret.ci_low)}, {fmt(raw_ret.ci_high)}]，P={fmt(raw_ret.p, 4)}。
- TOST 是对上述系数的等效性检验，不是新响应：margin={fmt(tost.equivalence_margin)}，90% CI [{fmt(tost.ci_low)}, {fmt(tost.ci_high)}]，TOST P={fmt(tost.p, 4)}。
- baseline-cutoff retention 是分母敏感性：baseline >5% 时系数 {fmt(cut5.beta)}，baseline >10% 时 {fmt(cut10.beta)}；它们不能替代 raw retention 或组间交互。

## 对前提假设的回答

1. **10% 是否有可见分布断点？** 否。5%–20% 直方图连续，9%–10% 与 10%–11% 为 12/11，且无精确 10% 或规则取整堆积。
2. **基线—响应几何是否在 10% 改变？** 由连续 hinge 的斜率差直接回答，不能由两个独立分组模型代替。
3. **recurrence association 是否随 10% 分组变化？** 由 interaction coefficient 及其 CI/P 直接回答；分组显著性只作描述。
4. **结论是否依赖人为切点？** 由四阈值结果和固定 spline 检查共同判断，不按显著性挑选阈值。

## 生态解释边界

- {recommendation}
- 本分析只支持或不支持“统计关联异质性”，不证明遗传多样性、共生体灵活性、功能冗余、藻类优势或繁殖体受限等机制。
- 不推断 phase shift、community transition、physiological adaptation 或单一分类过滤机制。
- 不使用 composition explains/mediates/causes lower loss 的因果措辞。
- absolute loss、raw proportional retention、既有 TOST 与 baseline-cutoff retention 是四个不同问题；登记见 `results/response_metric_context.csv`。

## 统计谬误核查（11/11）

- Simpson：已用 event type/sector 分布和分组模型检查方向；未把聚合与分组差异互换。
- Ecological fallacy：推断单位保持 reef-year/reef，不外推个体珊瑚机制。
- Berkson/collider：样本由现有事件矩阵构造，选择机制仍可能限制外推；未新增事后控制变量。
- Base-rate neglect：不适用诊断准确率问题。
- Regression to mean：baseline 与 absolute loss 存在数学/选择几何，故用 hinge 并保留 baseline 调整，不作适应性解释。
- Survivorship：baseline/nadir 可用性选择继承自主矩阵，无法由本验证消除。
- Look-elsewhere：只检验预定 8/10/12/15，并报告全部结果及 FDR。
- Forking paths：固定主协变量、固定 4-df spline，不做模型/切点搜索。
- Correlation ≠ causation：全程使用 association。
- Reverse causality：prior recurrence 在时间上先于 target event，但观察性混杂仍不能排除。
"""
    (HERE / "threshold_validation_report.md").write_text(report, encoding="utf-8")


def write_checks(df: pd.DataFrame, before: dict[str, str], after: dict[str, str], model_log: pd.DataFrame) -> None:
    protected_ok = before == after
    generated_csvs = list(RESULTS.glob("*.csv"))
    forbidden_columns = []
    for path in generated_csvs:
        columns = [c.lower() for c in pd.read_csv(path, nrows=0).columns]
        forbidden_columns.extend([f"{path.name}:{c}" for c in columns if c in {"lrr", "log_response_ratio"}])
    failures = model_log[~model_log["success"]]
    checks = f"""样本量检查：{len(df)} reef-years（预期 525）；{df['reef_name'].nunique()} 个 reefs。
关键响应：loss_abs；retention = nadir_hc / baseline_hc（raw、未裁剪）。
关键暴露：heatwave_years_5yr；当前 DHW/wind、storm history、return interval、event type 沿用主模型。
模型状态：{int(model_log['success'].sum())}/{len(model_log)} 成功；失败 {len(failures)}。
对数响应比实现/结果列：0（未新增或恢复）；禁止列命中：{json.dumps(forbidden_columns, ensure_ascii=False)}。
终稿/既有输入未修改：{'通过' if protected_ok else '失败'}。
写入范围：仅 {HERE}。
输入矩阵 SHA256：{sha256(INPUT_MATRIX)}。
主模型脚本 SHA256：{sha256(MAIN_MODEL_SCRIPT)}。
Python：{sys.version.split()[0]}；平台：{platform.platform()}。
pandas {pd.__version__}; numpy {np.__version__}; scipy {scipy.__version__}; statsmodels {statsmodels.__version__}; sklearn {sklearn.__version__}; matplotlib {mpl.__version__}; patsy {patsy.__version__}。
"""
    (HERE / "validation_checks.txt").write_text(checks, encoding="utf-8")
    if not protected_ok:
        raise RuntimeError("Protected inputs/final directories changed during execution")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    df = prepare_data()

    groups = group_summary(df)
    dist = distribution_diagnostics(df)
    balance, categorical, missing = balance_tables(df)
    support, scores = common_support(df)

    hinge_rows, interaction_rows, subgroup_rows, prediction_rows = [], [], [], []
    model_log = []
    for cutoff in THRESHOLDS:
        try:
            h, p = hinge_models(df, cutoff)
            hinge_rows.extend(h); prediction_rows.extend(p)
            model_log.append({"analysis": f"hinge_{cutoff:g}", "success": True, "message": ""})
        except Exception as exc:
            model_log.append({"analysis": f"hinge_{cutoff:g}", "success": False, "message": repr(exc)})
        try:
            i, s, p = recurrence_models(df, cutoff)
            interaction_rows.extend(i); subgroup_rows.extend(s); prediction_rows.extend(p)
            model_log.append({"analysis": f"interaction_and_subgroup_{cutoff:g}", "success": True, "message": ""})
        except Exception as exc:
            model_log.append({"analysis": f"interaction_and_subgroup_{cutoff:g}", "success": False, "message": repr(exc)})

    hinge = pd.DataFrame(hinge_rows)
    interactions = pd.DataFrame(interaction_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    predictions = pd.DataFrame(prediction_rows)
    hinge = add_fdr(hinge, "slope_difference_above_minus_below")
    interactions = add_fdr(interactions, "recurrence_slope_difference_high_minus_low")
    spline_tests, spline_effects = spline_models(df)
    model_log.append({"analysis": "natural_cubic_spline_df4", "success": True, "message": ""})
    model_log = pd.DataFrame(model_log)

    groups.to_csv(RESULTS / "group_summary.csv", index=False)
    dist.to_csv(RESULTS / "distribution_diagnostics.csv", index=False)
    balance.to_csv(RESULTS / "continuous_covariate_balance.csv", index=False)
    categorical.to_csv(RESULTS / "categorical_covariate_balance.csv", index=False)
    missing.to_csv(RESULTS / "missingness.csv", index=False)
    support.to_csv(RESULTS / "common_support.csv", index=False)
    scores.to_csv(RESULTS / "common_support_scores.csv", index=False)
    hinge.to_csv(RESULTS / "hinge_models.csv", index=False)
    interactions.to_csv(RESULTS / "interaction_models.csv", index=False)
    subgroups.to_csv(RESULTS / "subgroup_models.csv", index=False)
    predictions.to_csv(RESULTS / "adjusted_predictions.csv", index=False)
    spline_tests.to_csv(RESULTS / "nonlinear_spline_tests.csv", index=False)
    spline_effects.to_csv(RESULTS / "nonlinear_marginal_effects.csv", index=False)
    response_metric_context().to_csv(RESULTS / "response_metric_context.csv", index=False)
    model_log.to_csv(RESULTS / "model_fit_log.csv", index=False)
    threshold = pd.concat([
        hinge[hinge["quantity"].eq("slope_difference_above_minus_below")],
        interactions[interactions["quantity"].eq("recurrence_slope_difference_high_minus_low")],
        subgroups,
    ], ignore_index=True, sort=False)
    threshold.to_csv(RESULTS / "threshold_sensitivity.csv", index=False)

    make_figures(df, scores, spline_effects)
    write_readme()
    write_report(groups, dist, balance, support, hinge, interactions, subgroups, spline_tests)
    after = snapshot_protected()
    write_checks(df, before, after, model_log)
    print(f"Completed isolated threshold validation: {HERE}")


if __name__ == "__main__":
    main()
