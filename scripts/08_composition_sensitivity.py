import os
import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

LEGACY_MATRIX_PATH = os.path.join(OUTPUT_DIR, "legacy_load_analysis_matrix.csv")
MASTER_MATRIX_PATH = os.path.join(DATA_DIR, "eco_response_master_matrix_merged.csv")
COMPOSITION_DIR = os.path.join(DATA_DIR, "coral_species_data")
COMPOSITION_MERGED_PATH = os.path.join(DATA_DIR, "coral_species_data_all.csv")

BASE_LOSS_FORMULA = (
    "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
    "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z "
    "+ C(event_type)"
)

RETENTION_FORMULA = (
    "retention ~ recent_max_dhw_z + recent_max_wind_z + baseline_hc_z "
    "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z "
    "+ C(event_type)"
)

MAIN_CONTROL_FORMULA = "heatwave_years_5yr_z + event_year_z + baseline_hc_z + C(sector)"
STORM_CONTROL_FORMULA = "storm_years_5yr_z + event_year_z + baseline_hc_z + C(sector)"

TARGET_HARD_CORAL_GROUPS = {
    "Acropora": "acropora",
    "Porites": "porites",
    "Pocilloporidae": "pocilloporidae",
    "Montipora": "montipora",
    "Merulinidae": "merulinidae",
}

COMPOSITION_TERMS = [
    ("acropora_prop_baseline", "Acropora proportion"),
    ("pocilloporidae_prop_baseline", "Pocilloporidae proportion"),
    ("porites_prop_baseline", "Porites proportion"),
    ("macroalgae_sum_baseline", "Macroalgae cover"),
    ("acropora_cover_baseline", "Acropora cover"),
    ("pocilloporidae_cover_baseline", "Pocilloporidae cover"),
    ("porites_cover_baseline", "Porites cover"),
]

VARIABLE_ORDER = ["ALGAE", "MACROALGAE", "HARD CORAL", "SOFT CORAL"]

FIGURE6_ALGAL_CATEGORIES = [
    ("MACROALGAE", "Other Brown Macroalgae"),
    ("ALGAE", "Other Brown Macroalgae"),
    ("ALGAE", "Red Macroalgae"),
    ("ALGAE", "Turf algae"),
    ("ALGAE", "Green Macroalgae"),
    ("MACROALGAE", "Red Macroalgae"),
    ("MACROALGAE", "Lobophora"),
]

FIGURE6_HARD_CORAL_CATEGORIES = [
    "Acropora",
    "Pocilloporidae",
    "Porites",
    "Montipora",
    "Merulinidae",
    "Other",
]


def q_to_stars(q_value):
    if pd.isna(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def normalize_reef_name(value):
    text = str(value).lower().replace("&", "and")
    text = re.sub(r"no\.?\s*([0-9]+)", r"no \1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_category_name(variable, category):
    text = f"{variable}__{category}".lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return f"cat_{text}"


def zscore(series):
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * np.nan
    return (series - series.mean()) / std


def cluster_ols(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["reef_name"]},
    )


def load_inputs():
    if os.path.exists(COMPOSITION_MERGED_PATH):
        composition = pd.read_csv(COMPOSITION_MERGED_PATH)
        if "source_file" not in composition.columns:
            composition["source_file"] = "coral_species_data_all.csv"
    else:
        files = sorted(
            file_name
            for file_name in os.listdir(COMPOSITION_DIR)
            if file_name.lower().endswith(".csv")
        )
        frames = []
        for file_name in files:
            path = os.path.join(COMPOSITION_DIR, file_name)
            frames.append(pd.read_csv(path).assign(source_file=file_name))
        composition = pd.concat(frames, ignore_index=True)
    composition["reef_key"] = composition["domain_name"].map(normalize_reef_name)
    composition["report_year"] = composition["report_year"].astype(int)

    events = pd.read_csv(LEGACY_MATRIX_PATH)
    master = pd.read_csv(MASTER_MATRIX_PATH)
    events["reef_key"] = events["reef_name"].map(normalize_reef_name)
    master["reef_key"] = master["reef_name"].map(normalize_reef_name)
    meta = master[["reef_key", "sector", "region_lat"]].drop_duplicates("reef_key")
    events = events.merge(meta, on="reef_key", how="left")
    events["retention"] = events["nadir_hc"] / events["baseline_hc"]
    return composition, events, master


def infer_response_years(events, master):
    master_by_reef = {reef: group.sort_values("year") for reef, group in master.groupby("reef_key")}
    baseline_years = []
    nadir_years = []
    for row in events.itertuples(index=False):
        reef_data = master_by_reef.get(row.reef_key)
        event_year = int(row.event_year)
        baseline_year = np.nan
        nadir_year = np.nan
        if reef_data is not None:
            baseline = reef_data[
                (reef_data["year"] >= event_year - 3)
                & (reef_data["year"] < event_year)
            ].dropna(subset=["HC_cover"])
            response = reef_data[
                (reef_data["year"] >= event_year)
                & (reef_data["year"] <= event_year + 3)
            ].dropna(subset=["HC_cover"])
            if not baseline.empty:
                baseline_year = int(baseline.iloc[-1]["year"])
            if not response.empty:
                nadir_year = int(response.loc[response["HC_cover"].idxmin(), "year"])
        baseline_years.append(baseline_year)
        nadir_years.append(nadir_year)
    events = events.copy()
    events["baseline_year"] = baseline_years
    events["nadir_year"] = nadir_years
    return events


def make_mapping_table(composition, master):
    source_reefs = (
        composition[["domain_name", "reef_key"]]
        .drop_duplicates()
        .rename(columns={"domain_name": "composition_reef_name"})
    )
    master_reefs = (
        master[["reef_name", "reef_key"]]
        .drop_duplicates()
        .rename(columns={"reef_name": "master_reef_name"})
    )
    mapping = master_reefs.merge(source_reefs, on="reef_key", how="outer")
    mapping["status"] = np.select(
        [
            mapping["master_reef_name"].notna() & mapping["composition_reef_name"].notna(),
            mapping["master_reef_name"].notna() & mapping["composition_reef_name"].isna(),
            mapping["master_reef_name"].isna() & mapping["composition_reef_name"].notna(),
        ],
        ["matched", "missing_from_composition", "not_used_in_master_matrix"],
        default="unknown",
    )
    return mapping.sort_values(["status", "reef_key"])


def validate_hard_coral_sum(composition, master):
    hard = (
        composition[composition["variable"].eq("HARD CORAL")]
        .groupby(["reef_key", "domain_name", "report_year"], as_index=False)["median"]
        .sum()
        .rename(columns={"report_year": "year", "median": "hard_coral_category_sum"})
    )
    comparison = (
        master[["reef_name", "reef_key", "year", "HC_cover"]]
        .dropna(subset=["HC_cover"])
        .merge(hard[["reef_key", "domain_name", "year", "hard_coral_category_sum"]], on=["reef_key", "year"], how="inner")
    )
    comparison["difference"] = comparison["hard_coral_category_sum"] - comparison["HC_cover"]
    comparison["abs_difference"] = comparison["difference"].abs()
    comparison["relative_abs_difference_pct"] = 100 * comparison["abs_difference"] / comparison["HC_cover"].replace(0, np.nan)
    return comparison


def build_composition_metrics(composition):
    hard = composition[composition["variable"].eq("HARD CORAL")].copy()
    hard_sum = (
        hard.groupby(["reef_key", "report_year"])["median"]
        .sum()
        .reset_index(name="modelled_hard_coral_sum")
    )
    metrics = hard_sum.copy()

    for category, prefix in TARGET_HARD_CORAL_GROUPS.items():
        group = hard[hard["reefpage_category"].eq(category)][["reef_key", "report_year", "median"]]
        group = group.rename(columns={"median": f"{prefix}_cover"})
        metrics = metrics.merge(group, on=["reef_key", "report_year"], how="left")
        metrics[f"{prefix}_prop"] = metrics[f"{prefix}_cover"] / metrics["modelled_hard_coral_sum"].replace(0, np.nan)

    macroalgae = (
        composition[composition["variable"].eq("MACROALGAE")]
        .groupby(["reef_key", "report_year"])["median"]
        .sum()
        .reset_index(name="macroalgae_sum")
    )
    metrics = metrics.merge(macroalgae, on=["reef_key", "report_year"], how="left")
    return metrics


def build_all_category_metrics(composition):
    metrics = None
    category_map = []
    for variable in VARIABLE_ORDER:
        subset = composition[composition["variable"].eq(variable)].copy()
        for category, group in subset.groupby("reefpage_category"):
            col = safe_category_name(variable, category)
            values = group[["reef_key", "report_year", "median"]].rename(columns={"median": col})
            metrics = values if metrics is None else metrics.merge(values, on=["reef_key", "report_year"], how="outer")
            category_map.append(
                {
                    "variable": variable,
                    "category": category,
                    "metric": col,
                    "label": category,
                    "figure_label": f"{category} ({variable.title()})",
                }
            )
    return metrics, pd.DataFrame(category_map)


def attach_composition(events, metrics):
    baseline_metrics = metrics.rename(columns={"report_year": "baseline_year"})
    nadir_metrics = metrics.rename(columns={"report_year": "nadir_year"})

    baseline_cols = ["reef_key", "baseline_year"] + [
        col for col in baseline_metrics.columns if col not in {"reef_key", "baseline_year"}
    ]
    event = events.merge(baseline_metrics[baseline_cols], on=["reef_key", "baseline_year"], how="left")
    rename = {
        col: f"{col}_baseline"
        for col in baseline_cols
        if col not in {"reef_key", "baseline_year"}
    }
    event = event.rename(columns=rename)

    nadir_cols = ["reef_key", "nadir_year"] + [
        col for col in nadir_metrics.columns if col not in {"reef_key", "nadir_year"}
    ]
    event = event.merge(nadir_metrics[nadir_cols], on=["reef_key", "nadir_year"], how="left")
    rename = {
        col: f"{col}_nadir"
        for col in nadir_cols
        if col not in {"reef_key", "nadir_year"}
    }
    event = event.rename(columns=rename)

    event["composition_baseline_available"] = event["modelled_hard_coral_sum_baseline"].notna()
    event["composition_nadir_available"] = event["modelled_hard_coral_sum_nadir"].notna()
    event["composition_both_available"] = (
        event["composition_baseline_available"] & event["composition_nadir_available"]
    )
    for prefix in TARGET_HARD_CORAL_GROUPS.values():
        event[f"{prefix}_prop_hcdenom_baseline"] = event[f"{prefix}_cover_baseline"] / event["baseline_hc"].replace(0, np.nan)
    return event


def attach_all_category_metrics(events, category_metrics):
    baseline_metrics = category_metrics.rename(columns={"report_year": "baseline_year"})
    baseline_cols = ["reef_key", "baseline_year"] + [
        col for col in baseline_metrics.columns if col not in {"reef_key", "baseline_year"}
    ]
    event = events.merge(baseline_metrics[baseline_cols], on=["reef_key", "baseline_year"], how="left")
    rename = {
        col: f"{col}_baseline"
        for col in baseline_cols
        if col not in {"reef_key", "baseline_year"}
    }
    return event.rename(columns=rename)


def add_standardized_controls(event):
    event = event.copy()
    for col in [
        "baseline_hc",
        "recent_max_dhw",
        "recent_max_wind",
        "heatwave_years_5yr",
        "storm_years_5yr",
        "yrs_since_last_dist",
        "event_year",
    ]:
        event[f"{col}_z"] = zscore(event[col])
    return event


def diagnostics_table(composition, mapping, hard_validation, event):
    category_counts = (
        composition.groupby(["domain_name", "report_year", "variable"])["reefpage_category"]
        .nunique()
        .reset_index(name="category_count")
        .groupby("variable")["category_count"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={"variable": "term"})
    )
    category_counts.insert(0, "section", "reef-year category count")
    category_counts["note"] = "Number of reported reefpage categories per reef-year-variable."

    rows = [
        {
            "section": "reef matching",
            "term": "matched reefs",
            "count": int(mapping["status"].eq("matched").sum()),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "note": "Composition reefs matched to master ecological matrix after reef-name normalization.",
        },
        {
            "section": "event-level availability",
            "term": "baseline composition available",
            "count": int(event["composition_baseline_available"].sum()),
            "mean": event["composition_baseline_available"].mean(),
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "note": "Target events with matched pre-disturbance composition.",
        },
        {
            "section": "event-level availability",
            "term": "nadir composition available",
            "count": int(event["composition_nadir_available"].sum()),
            "mean": event["composition_nadir_available"].mean(),
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "note": "Target events with matched nadir-year composition.",
        },
        {
            "section": "event-level availability",
            "term": "baseline and nadir composition available",
            "count": int(event["composition_both_available"].sum()),
            "mean": event["composition_both_available"].mean(),
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "note": "Target events with both pre-disturbance and nadir-year composition.",
        },
        {
            "section": "hard-coral validation",
            "term": "common reef-years",
            "count": len(hard_validation),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "note": "Reef-years with both HC_cover and summed modelled hard-coral categories.",
        },
        {
            "section": "hard-coral validation",
            "term": "category sum minus HC_cover",
            "count": len(hard_validation),
            "mean": hard_validation["difference"].mean(),
            "std": hard_validation["difference"].std(),
            "min": hard_validation["difference"].min(),
            "median": hard_validation["difference"].median(),
            "max": hard_validation["difference"].max(),
            "note": "Difference in percentage points.",
        },
        {
            "section": "hard-coral validation",
            "term": "absolute category-sum difference",
            "count": len(hard_validation),
            "mean": hard_validation["abs_difference"].mean(),
            "std": hard_validation["abs_difference"].std(),
            "min": hard_validation["abs_difference"].min(),
            "median": hard_validation["abs_difference"].median(),
            "max": hard_validation["abs_difference"].max(),
            "note": "Absolute difference in percentage points.",
        },
        {
            "section": "hard-coral validation",
            "term": "correlation with HC_cover",
            "count": len(hard_validation),
            "mean": hard_validation[["HC_cover", "hard_coral_category_sum"]].corr().iloc[0, 1],
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
            "note": "Pearson correlation between total HC_cover and summed modelled hard-coral categories.",
        },
    ]
    return pd.concat([category_counts, pd.DataFrame(rows)], ignore_index=True, sort=False)


def grouped_composition_table(event):
    df = event[event["composition_baseline_available"]].copy()
    df["heatwave_group"] = np.select(
        [df["heatwave_years_5yr"].eq(0), df["heatwave_years_5yr"].eq(1)],
        ["0 prior heatwave years", "1 prior heatwave year"],
        default="2+ prior heatwave years",
    )
    metrics = [
        ("baseline_hc", "Baseline hard-coral cover"),
        ("loss_abs", "Absolute hard-coral cover loss"),
        ("modelled_hard_coral_sum_baseline", "Modelled hard-coral sum"),
        ("macroalgae_sum_baseline", "Macroalgae cover"),
        ("acropora_cover_baseline", "Acropora cover"),
        ("acropora_prop_baseline", "Acropora proportion"),
        ("pocilloporidae_cover_baseline", "Pocilloporidae cover"),
        ("pocilloporidae_prop_baseline", "Pocilloporidae proportion"),
        ("porites_cover_baseline", "Porites cover"),
        ("porites_prop_baseline", "Porites proportion"),
    ]
    rows = []
    for group_name, group in df.groupby("heatwave_group"):
        for col, label in metrics:
            values = group[col].dropna()
            rows.append(
                {
                    "table": "Table S11",
                    "heatwave_group": group_name,
                    "metric": col,
                    "label": label,
                    "n_observations": len(values),
                    "n_reefs": group.loc[group[col].notna(), "reef_name"].nunique(),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def model_row(table, model, family, response, result, term, label, n_observations, n_reefs, extra=None):
    ci_low, ci_high = result.conf_int().loc[term].tolist()
    row = {
        "table": table,
        "model": model,
        "model_family": family,
        "response": response,
        "term": term,
        "label": label,
        "beta": result.params[term],
        "z": result.tvalues[term],
        "p": result.pvalues[term],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_observations": n_observations,
        "n_reefs": n_reefs,
        "r_squared": getattr(result, "rsquared", np.nan),
    }
    if extra:
        row.update(extra)
    return row


def reorganization_models(event):
    rows = []
    for outcome, label in COMPOSITION_TERMS:
        df = event.dropna(
            subset=[
                outcome,
                "heatwave_years_5yr_z",
                "event_year_z",
                "baseline_hc_z",
                "sector",
                "reef_name",
            ]
        ).copy()
        if len(df) < 80:
            continue
        result = cluster_ols(f"{outcome} ~ {MAIN_CONTROL_FORMULA}", df)
        rows.append(
            model_row(
                "Table S11",
                "Prior heatwave recurrence predicting baseline composition",
                "reef-cluster robust OLS",
                outcome,
                result,
                "heatwave_years_5yr_z",
                "5-year heatwave years",
                len(df),
                df["reef_name"].nunique(),
                {"composition_metric": label},
            )
        )
    return pd.DataFrame(rows)


def response_models(event):
    rows = []
    metric_sets = [
        ("Reference recurrence model on baseline-composition subset", [], True),
        ("Loss model + Acropora proportion", ["acropora_prop_baseline"], False),
        ("Loss model + Acropora, Pocilloporidae and Porites proportions", ["acropora_prop_baseline", "pocilloporidae_prop_baseline", "porites_prop_baseline"], False),
        ("Loss model + coral proportions and macroalgae", ["acropora_prop_baseline", "pocilloporidae_prop_baseline", "porites_prop_baseline", "macroalgae_sum_baseline"], False),
    ]
    for model_name, metrics, require_baseline_composition in metric_sets:
        subset = [
            "loss_abs",
            "baseline_hc_z",
            "recent_max_dhw_z",
            "recent_max_wind_z",
            "heatwave_years_5yr_z",
            "storm_years_5yr_z",
            "yrs_since_last_dist_z",
            "event_type",
            "reef_name",
        ] + metrics
        if require_baseline_composition:
            subset.append("modelled_hard_coral_sum_baseline")
        df = event.dropna(subset=subset).copy()
        for metric in metrics:
            df[f"{metric}_z"] = zscore(df[metric])
        formula = BASE_LOSS_FORMULA + "".join([f" + {metric}_z" for metric in metrics])
        result = cluster_ols(formula, df)
        terms = [("heatwave_years_5yr_z", "5-year heatwave years"), ("storm_years_5yr_z", "5-year storm years")]
        terms.extend([(f"{metric}_z", metric.replace("_baseline", "").replace("_", " ").title()) for metric in metrics])
        for term, label in terms:
            rows.append(
                model_row(
                    "Table S12",
                    model_name,
                    "reef-cluster robust OLS",
                    "loss_abs",
                    result,
                    term,
                    label,
                    len(df),
                    df["reef_name"].nunique(),
                )
            )

    retention_sets = [
        ("Retention model + Acropora and Porites proportions", ["acropora_prop_baseline", "porites_prop_baseline"]),
        ("Retention model + coral proportions", ["acropora_prop_baseline", "pocilloporidae_prop_baseline", "porites_prop_baseline"]),
        ("Retention model + coral proportions and macroalgae", ["acropora_prop_baseline", "pocilloporidae_prop_baseline", "porites_prop_baseline", "macroalgae_sum_baseline"]),
    ]
    for model_name, metrics in retention_sets:
        subset = [
            "retention",
            "baseline_hc",
            "baseline_hc_z",
            "recent_max_dhw_z",
            "recent_max_wind_z",
            "heatwave_years_5yr_z",
            "storm_years_5yr_z",
            "yrs_since_last_dist_z",
            "event_type",
            "reef_name",
        ] + metrics
        df = event[event["baseline_hc"] > 10].dropna(subset=subset).copy()
        for metric in metrics:
            df[f"{metric}_z"] = zscore(df[metric])
        formula = RETENTION_FORMULA + "".join([f" + {metric}_z" for metric in metrics])
        result = cluster_ols(formula, df)
        for metric in metrics:
            term = f"{metric}_z"
            rows.append(
                model_row(
                    "Table S12",
                    model_name,
                    "reef-cluster robust OLS",
                    "retention",
                    result,
                    term,
                    metric.replace("_baseline", "").replace("_", " ").title(),
                    len(df),
                    df["reef_name"].nunique(),
                    {"baseline_filter": "baseline_hc > 10"},
                )
            )
    return pd.DataFrame(rows)


def denominator_sensitivity(event):
    rows = []
    for prefix, label in [
        ("acropora", "Acropora"),
        ("pocilloporidae", "Pocilloporidae"),
        ("porites", "Porites"),
    ]:
        for denom_suffix, denom_label in [
            ("prop_baseline", "modelled hard-coral sum"),
            ("prop_hcdenom_baseline", "baseline_hc"),
        ]:
            outcome = f"{prefix}_{denom_suffix}"
            df = event.dropna(
                subset=[
                    outcome,
                    "heatwave_years_5yr_z",
                    "event_year_z",
                    "baseline_hc_z",
                    "sector",
                    "reef_name",
                ]
            ).copy()
            result = cluster_ols(f"{outcome} ~ {MAIN_CONTROL_FORMULA}", df)
            rows.append(
                model_row(
                    "Table S13",
                    f"Denominator sensitivity: {label} proportion",
                    "reef-cluster robust OLS",
                    outcome,
                    result,
                    "heatwave_years_5yr_z",
                    "5-year heatwave years",
                    len(df),
                    df["reef_name"].nunique(),
                    {"denominator": denom_label},
                )
            )
    return pd.DataFrame(rows)


def add_fdr_columns(table, p_col="p", group_col="variable"):
    table = table.copy()
    if table.empty:
        table["q_all"] = np.nan
        table["q_within_variable"] = np.nan
        return table
    table["q_all"] = multipletests(table[p_col].fillna(1), method="fdr_bh")[1]
    table["q_within_variable"] = np.nan
    for _, idx in table.groupby(group_col).groups.items():
        table.loc[idx, "q_within_variable"] = multipletests(table.loc[idx, p_col].fillna(1), method="fdr_bh")[1]
    return table


def all_category_reorganization_models(
    event,
    category_map,
    predictor="heatwave_years_5yr_z",
    predictor_label="Prior heatwave recurrence",
    table_name="Table S14",
    formula_controls=MAIN_CONTROL_FORMULA,
):
    rows = []
    for item in category_map.itertuples(index=False):
        outcome = f"{item.metric}_baseline"
        df = event.dropna(
            subset=[
                outcome,
                predictor,
                "event_year_z",
                "baseline_hc_z",
                "sector",
                "reef_name",
            ]
        ).copy()
        if len(df) < 150 or df["reef_name"].nunique() < 30:
            continue
        result = cluster_ols(f"{outcome} ~ {formula_controls}", df)
        term = predictor
        ci_low, ci_high = result.conf_int().loc[term].tolist()
        rows.append(
            {
                "table": table_name,
                "model": f"{predictor_label} predicting baseline category cover",
                "model_family": "reef-cluster robust OLS",
                "variable": item.variable,
                "category": item.category,
                "metric": outcome,
                "term": term,
                "beta": result.params[term],
                "z": result.tvalues[term],
                "p": result.pvalues[term],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_observations": len(df),
                "n_reefs": df["reef_name"].nunique(),
                "r_squared": result.rsquared,
            }
        )
    table = pd.DataFrame(rows)
    return add_fdr_columns(table).sort_values(["q_all", "p", "variable", "category"])


def category_to_loss_sensitivity(event, category_map):
    rows = []
    for item in category_map.itertuples(index=False):
        metric = f"{item.metric}_baseline"
        df = event.dropna(
            subset=[
                metric,
                "loss_abs",
                "baseline_hc_z",
                "recent_max_dhw_z",
                "recent_max_wind_z",
                "heatwave_years_5yr_z",
                "storm_years_5yr_z",
                "yrs_since_last_dist_z",
                "event_type",
                "reef_name",
            ]
        ).copy()
        if len(df) < 150 or df["reef_name"].nunique() < 30:
            continue
        z_metric = f"{metric}_z"
        df[z_metric] = zscore(df[metric])
        result = cluster_ols(BASE_LOSS_FORMULA + f" + {z_metric}", df)
        for term, label in [(z_metric, "Baseline category cover"), ("heatwave_years_5yr_z", "5-year heatwave years")]:
            ci_low, ci_high = result.conf_int().loc[term].tolist()
            rows.append(
                {
                    "table": "Table S15",
                    "model": "Category-informed absolute-loss sensitivity",
                    "model_family": "reef-cluster robust OLS",
                    "variable": item.variable,
                    "category": item.category,
                    "metric": metric,
                    "response": "loss_abs",
                    "term": term,
                    "label": label,
                    "beta": result.params[term],
                    "z": result.tvalues[term],
                    "p": result.pvalues[term],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_observations": len(df),
                    "n_reefs": df["reef_name"].nunique(),
                    "r_squared": result.rsquared,
                }
            )
    table = pd.DataFrame(rows)
    category_rows = table[table["label"].eq("Baseline category cover")].copy()
    category_rows = add_fdr_columns(category_rows)
    recurrence_rows = table[table["label"].eq("5-year heatwave years")].copy()
    recurrence_rows["q_all"] = np.nan
    recurrence_rows["q_within_variable"] = np.nan
    return pd.concat([category_rows, recurrence_rows], ignore_index=True).sort_values(["label", "q_all", "p", "variable", "category"])


def category_availability_table(event, category_map):
    rows = []
    for item in category_map.itertuples(index=False):
        metric = f"{item.metric}_baseline"
        if metric not in event.columns:
            continue
        available = event[metric].notna()
        rows.append(
            {
                "table": "Table S14",
                "variable": item.variable,
                "category": item.category,
                "metric": metric,
                "baseline_available_n": int(available.sum()),
                "baseline_available_fraction": available.mean(),
                "baseline_available_reefs": event.loc[available, "reef_name"].nunique(),
                "baseline_cover_mean": event.loc[available, metric].mean(),
                "baseline_cover_median": event.loc[available, metric].median(),
                "baseline_cover_q25": event.loc[available, metric].quantile(0.25),
                "baseline_cover_q75": event.loc[available, metric].quantile(0.75),
                "note": "Missing values were retained as missing and were not recoded to zero.",
            }
        )
    return pd.DataFrame(rows).sort_values(["variable", "category"])


def make_si_figure(event, response_table):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    df = event[event["composition_baseline_available"]].copy()
    df["Heatwave history"] = np.select(
        [df["heatwave_years_5yr"].eq(0), df["heatwave_years_5yr"].eq(1)],
        ["0", "1"],
        default="2+",
    )
    order = ["0", "1", "2+"]
    colors = {"0": "#6f7f8f", "1": "#d6a64f", "2+": "#b85c4a"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))

    ax = axes[0, 0]
    data = [df.loc[df["Heatwave history"].eq(group), "macroalgae_sum_baseline"].dropna() for group in order]
    bp = ax.boxplot(data, patch_artist=True, labels=order, showfliers=False)
    for patch, group in zip(bp["boxes"], order):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.75)
    ax.set_xlabel("Prior 5-year heatwave years")
    ax.set_ylabel("Baseline macroalgae cover (%)")
    ax.set_title("A", loc="left", fontweight="bold")

    ax = axes[0, 1]
    x = np.arange(len(order))
    width = 0.24
    metric_cols = [
        ("acropora_prop_baseline", "Acropora", "#4c78a8"),
        ("pocilloporidae_prop_baseline", "Pocilloporidae", "#59a14f"),
        ("porites_prop_baseline", "Porites", "#9c755f"),
    ]
    for i, (col, label, color) in enumerate(metric_cols):
        means = [df.loc[df["Heatwave history"].eq(group), col].mean() for group in order]
        sems = [df.loc[df["Heatwave history"].eq(group), col].sem() for group in order]
        ax.bar(x + (i - 1) * width, means, width, yerr=sems, color=color, alpha=0.82, label=label, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Prior 5-year heatwave years")
    ax.set_ylabel("Proportion of hard-coral cover")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("B", loc="left", fontweight="bold")

    ax = axes[1, 0]
    coef_rows = response_table[
        response_table["response"].eq("loss_abs")
        & response_table["term"].eq("heatwave_years_5yr_z")
    ].copy()
    coef_rows = coef_rows.sort_values("n_observations", ascending=False)
    labels = [
        "Reference",
        "Acropora",
        "Coral groups",
        "Groups + macroalgae",
    ][: len(coef_rows)]
    y = np.arange(len(coef_rows))
    ax.axvline(0, color="0.7", lw=0.8)
    ax.errorbar(
        coef_rows["beta"],
        y,
        xerr=[coef_rows["beta"] - coef_rows["ci_low"], coef_rows["ci_high"] - coef_rows["beta"]],
        fmt="o",
        color="#3b5f7f",
        ecolor="#3b5f7f",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized heatwave-recurrence coefficient")
    ax.set_title("C", loc="left", fontweight="bold")

    ax = axes[1, 1]
    scatter = df.dropna(subset=["acropora_prop_baseline", "loss_abs"])
    ax.scatter(scatter["acropora_prop_baseline"], scatter["loss_abs"], s=14, alpha=0.45, color="#4c78a8", edgecolor="none")
    if len(scatter) > 5:
        xfit = np.linspace(scatter["acropora_prop_baseline"].min(), scatter["acropora_prop_baseline"].max(), 100)
        fit = np.polyfit(scatter["acropora_prop_baseline"], scatter["loss_abs"], deg=1)
        ax.plot(xfit, fit[0] * xfit + fit[1], color="#1f3349", lw=1.2)
    ax.set_xlabel("Baseline Acropora proportion")
    ax.set_ylabel("Absolute hard-coral cover loss")
    ax.set_title("D", loc="left", fontweight="bold")

    fig.tight_layout()
    out_base = os.path.join(FIGURE_DIR, "si_figure_04")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)


def coefficient_panel(ax, table, rows, title, xlabel="Heatwave-recurrence coefficient", include_variable=False):
    plot_rows = []
    for variable, category in rows:
        hit = table[(table["variable"].eq(variable)) & (table["category"].eq(category))]
        if not hit.empty:
            plot_rows.append(hit.iloc[0].to_dict())
    if not plot_rows:
        ax.axis("off")
        return
    plot = pd.DataFrame(plot_rows)
    labels = []
    for row in plot.itertuples(index=False):
        name = f"{row.category} ({row.variable.title()})" if include_variable else row.category
        labels.append(name)
    y = np.arange(len(plot))
    colors = np.where(plot["ci_low"].gt(0) | plot["ci_high"].lt(0), "#a65e4e", "#7a8793")
    ax.axvline(0, color="0.72", lw=0.9, zorder=0)
    x_min = float(plot["ci_low"].min())
    x_max = float(plot["ci_high"].max())
    x_span = x_max - x_min if x_max > x_min else 1.0
    for i, row in enumerate(plot.itertuples(index=False)):
        color = colors[i]
        ax.errorbar(
            row.beta,
            i,
            xerr=[[row.beta - row.ci_low], [row.ci_high - row.beta]],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
            zorder=1,
        )
        ax.scatter(row.beta, i, s=24, color=color, zorder=2)
        stars = q_to_stars(row.q_all)
        if stars:
            ax.text(row.ci_high + 0.06 * x_span, i, stars, ha="left", va="center", fontsize=7.5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(x_min - 0.12 * x_span, x_max + 0.28 * x_span)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")


def make_figure6(event, all_reorg, response_table):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    df = event[event["composition_baseline_available"]].copy()
    df["Heatwave history"] = np.select(
        [df["heatwave_years_5yr"].eq(0), df["heatwave_years_5yr"].eq(1)],
        ["0", "1"],
        default="2+",
    )
    order = ["0", "1", "2+"]
    colors = {"0": "#7d8790", "1": "#d1a04d", "2+": "#b56556"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), gridspec_kw={"width_ratios": [0.95, 1.15]})

    ax = axes[0, 0]
    data = [df.loc[df["Heatwave history"].eq(group), "macroalgae_sum_baseline"].dropna() for group in order]
    bp = ax.boxplot(data, patch_artist=True, labels=order, showfliers=False, widths=0.55)
    for patch, group in zip(bp["boxes"], order):
        patch.set_facecolor(colors[group])
        patch.set_edgecolor("0.25")
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("0.2")
            item.set_linewidth(1.0)
    ax.set_xlabel("Prior 5-year heatwave years")
    ax.set_ylabel("Baseline macroalgae cover (%)")
    ax.set_title("A", loc="left", fontweight="bold")

    ax = axes[0, 1]
    coefficient_panel(
        ax,
        all_reorg,
        FIGURE6_ALGAL_CATEGORIES,
        "B  Algal/macroalgal reporting layers",
        include_variable=True,
    )

    ax = axes[1, 0]
    hard_rows = [("HARD CORAL", category) for category in FIGURE6_HARD_CORAL_CATEGORIES]
    coefficient_panel(
        ax,
        all_reorg,
        hard_rows,
        "C  Hard-coral categories",
    )

    ax = axes[1, 1]
    coef_rows = response_table[
        response_table["response"].eq("loss_abs")
        & response_table["term"].eq("heatwave_years_5yr_z")
    ].copy()
    coef_rows = coef_rows.sort_values("n_observations", ascending=False)
    labels = ["Reference", "Acropora", "Coral groups", "Groups + macroalgae"][: len(coef_rows)]
    y = np.arange(len(coef_rows))
    ax.axvline(0, color="0.72", lw=0.9)
    ax.errorbar(
        coef_rows["beta"],
        y,
        xerr=[coef_rows["beta"] - coef_rows["ci_low"], coef_rows["ci_high"] - coef_rows["beta"]],
        fmt="o",
        color="#3b5f7f",
        ecolor="#3b5f7f",
        capsize=2.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Heatwave-recurrence coefficient for absolute loss")
    ax.set_title("D  Composition-control sensitivity", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=1.4, h_pad=1.4)
    out_base = os.path.join(FIGURE_DIR, "figure_05")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_outputs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    composition, events, master = load_inputs()
    events = infer_response_years(events, master)
    metrics = build_composition_metrics(composition)
    category_metrics, category_map = build_all_category_metrics(composition)
    event = attach_composition(events, metrics)
    event = attach_all_category_metrics(event, category_metrics)
    event = add_standardized_controls(event)

    mapping = make_mapping_table(composition, master)
    hard_validation = validate_hard_coral_sum(composition, master)

    mapping.to_csv(os.path.join(TABLE_DIR, "composition_reef_mapping.csv"), index=False)
    hard_validation.to_csv(os.path.join(TABLE_DIR, "composition_hard_coral_validation.csv"), index=False)
    event.to_csv(os.path.join(OUTPUT_DIR, "composition_event_analysis_matrix.csv"), index=False)

    diagnostics = diagnostics_table(composition, mapping, hard_validation, event)
    grouped = grouped_composition_table(event)
    reorg = reorganization_models(event)
    response = response_models(event)
    denom = denominator_sensitivity(event)
    all_reorg = all_category_reorganization_models(event, category_map)
    storm_reorg = all_category_reorganization_models(
        event,
        category_map,
        predictor="storm_years_5yr_z",
        predictor_label="Prior storm recurrence",
        table_name="Table S16",
        formula_controls=STORM_CONTROL_FORMULA,
    )
    category_loss = category_to_loss_sensitivity(event, category_map)
    category_availability = category_availability_table(event, category_map)

    diagnostics.to_csv(os.path.join(TABLE_DIR, "table_s10_composition_data_diagnostics.csv"), index=False)
    grouped.to_csv(os.path.join(TABLE_DIR, "table_s11_composition_grouped_summary.csv"), index=False)
    reorg.to_csv(os.path.join(TABLE_DIR, "table_s11_composition_association_models.csv"), index=False)
    response.to_csv(os.path.join(TABLE_DIR, "table_s12_composition_response_models.csv"), index=False)
    denom.to_csv(os.path.join(TABLE_DIR, "table_s13_composition_denominator_sensitivity.csv"), index=False)
    all_reorg.to_csv(os.path.join(TABLE_DIR, "table_s14_all_category_heatwave_scan.csv"), index=False)
    category_loss.to_csv(os.path.join(TABLE_DIR, "table_s15_category_to_loss_sensitivity.csv"), index=False)
    category_availability.to_csv(os.path.join(TABLE_DIR, "table_s14_category_availability.csv"), index=False)
    storm_reorg.to_csv(os.path.join(TABLE_DIR, "table_s16_storm_category_reorganization_models.csv"), index=False)
    make_si_figure(event, response)
    make_figure6(event, all_reorg, response)

    print("=" * 72)
    print("Composition analysis complete")
    print("=" * 72)
    print(f"Composition files: {composition['source_file'].nunique()}")
    print(f"Matched reefs: {mapping['status'].eq('matched').sum()}/{master['reef_key'].nunique()}")
    print(f"Baseline composition available: {event['composition_baseline_available'].sum()}/{len(event)}")
    print(f"Nadir composition available: {event['composition_nadir_available'].sum()}/{len(event)}")
    print(f"Both available: {event['composition_both_available'].sum()}/{len(event)}")
    print(f"Hard-coral sum correlation: {hard_validation[['HC_cover', 'hard_coral_category_sum']].corr().iloc[0, 1]:.6f}")
    print(f"Hard-coral median absolute difference: {hard_validation['abs_difference'].median():.6f}")
    print("Outputs written under output/tables and output/figures.")


if __name__ == "__main__":
    write_outputs()
