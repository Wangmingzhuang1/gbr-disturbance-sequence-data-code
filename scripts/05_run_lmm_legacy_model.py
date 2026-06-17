import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH = os.path.join(BASE, "output", "legacy_load_analysis_matrix.csv")
MASTER_MATRIX_PATH = os.path.join(BASE, "data", "eco_response_master_matrix_merged.csv")
TABLE_DIR = os.path.join(BASE, "output", "tables")
WINDOWS = (3, 5, 7, 8)

TERM_LABELS = {
    "Intercept": "Intercept",
    "baseline_hc_z": "Baseline hard-coral cover",
    "recent_max_dhw_z": "Recent maximum DHW",
    "recent_max_wind_z": "Recent maximum wind",
    "yrs_since_last_dist_z": "Return interval",
    "cumulative_dhw_5yr_z": "5-year cumulative DHW",
    "cumulative_wind_5yr_z": "5-year cumulative wind",
    "heatwave_years_5yr_z": "5-year heatwave years",
    "storm_years_5yr_z": "5-year storm years",
    "max_consecutive_heatwave_5yr_z": "Maximum consecutive heatwave years",
    "max_consecutive_storm_5yr_z": "Maximum consecutive storm years",
    "C(event_period)[T.2016-2020]": "Period: 2016-2020",
    "C(event_period)[T.2021-2025]": "Period: 2021-2025",
    "baseline_algae_z": "Baseline total algae cover",
    "baseline_macroalgae_z": "Baseline macroalgae cover",
    "baseline_juveniles_z": "Baseline juvenile density",
    "baseline_herbivores_z": "Baseline herbivorous fish density",
}


def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * np.nan
    return (series - series.mean()) / std


def dms_to_decimal(parts, direction):
    degrees = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0.0
    seconds = float(parts[2]) if len(parts) > 2 else 0.0
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if direction in {"S", "W"}:
        value *= -1
    return value


def parse_site_coordinates():
    path = os.path.join(BASE, "data", "sites_lon_lat.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["reef_name", "site_lat", "site_lon"])

    rows = []
    raw = pd.read_csv(path, header=None, names=["reef_name", "coord"], encoding="latin1")
    for row in raw.itertuples(index=False):
        coord = str(row.coord)
        numbers = re.findall(r"\d+(?:\.\d+)?", coord)
        directions = re.findall(r"[NSEW]", coord)
        if len(numbers) < 4 or len(directions) < 2:
            continue
        lat = dms_to_decimal(numbers[:3], directions[0])
        lon = dms_to_decimal(numbers[3:6], directions[1])
        rows.append({"reef_name": row.reef_name, "site_lat": lat, "site_lon": lon})
    return pd.DataFrame(rows).drop_duplicates(subset=["reef_name"], keep="first")


def reef_metadata():
    if not os.path.exists(MASTER_MATRIX_PATH):
        return pd.DataFrame(columns=["reef_name", "sector", "region_lat", "site_lat", "site_lon"])

    source = pd.read_csv(MASTER_MATRIX_PATH)
    cols = [col for col in ["reef_name", "sector", "region_lat"] if col in source.columns]
    meta = source[cols].drop_duplicates(subset=["reef_name"]).copy()
    coords = parse_site_coordinates()
    if not coords.empty:
        meta = meta.merge(coords, on="reef_name", how="left")
    return meta


def assign_period(year):
    if year < 2016:
        return "pre-2016"
    if year <= 2020:
        return "2016-2020"
    return "2021-2025"


def prepare_data():
    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError(
            f"Matrix file not found at: {MATRIX_PATH}. "
            "Run scripts/04_build_legacy_load_features.py first."
        )

    df = pd.read_csv(MATRIX_PATH)
    meta = reef_metadata()
    if not meta.empty:
        df = df.merge(meta, on="reef_name", how="left")
    df["positive_loss"] = df["loss_abs"].clip(lower=0)
    df["rel_loss_clipped"] = (df["loss_abs"] / df["baseline_hc"]).clip(-1, 1)
    df["retention"] = df["nadir_hc"] / df["baseline_hc"]
    df["event_period"] = df["event_year"].map(assign_period)

    numeric_cols = [
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
        "baseline_algae",
        "baseline_macroalgae",
        "baseline_juveniles",
        "baseline_herbivores",
    ]
    for window in WINDOWS:
        numeric_cols.extend(
            [
                f"cumulative_dhw_{window}yr",
                f"cumulative_wind_{window}yr",
                f"heatwave_years_{window}yr",
                f"storm_years_{window}yr",
                f"max_consecutive_heatwave_{window}yr",
                f"max_consecutive_storm_{window}yr",
            ]
        )

    for col in sorted(set(numeric_cols)):
        if col in df.columns:
            df[f"{col}_z"] = zscore(df[col])

    required = [
        "loss_abs",
        "positive_loss",
        "rel_loss_clipped",
        "baseline_hc_z",
        "recent_max_dhw_z",
        "recent_max_wind_z",
        "yrs_since_last_dist_z",
        "event_type",
        "reef_name",
    ]
    return df.dropna(subset=required).copy()


def robust_ols(df, formula):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["reef_name"]},
    )


def gee(df, formula):
    return smf.gee(
        formula,
        groups="reef_name",
        data=df,
        cov_struct=sm.cov_struct.Exchangeable(),
        family=sm.families.Gaussian(),
    ).fit()


def quantile_regression(df, formula, q=0.9):
    return smf.quantreg(formula, data=df).fit(q=q, max_iter=10000)


def print_terms(title, result, terms):
    print(f"\n{title}")
    print("-" * len(title))
    for term, label in terms:
        if term not in result.params.index:
            continue
        ci = result.conf_int().loc[term].tolist()
        print(
            f"{label:34s} beta={result.params[term]:7.3f}  "
            f"z={result.tvalues[term]:7.3f}  p={result.pvalues[term]:.4g}  "
            f"95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]"
        )


def model_rows(
    result,
    table,
    model,
    model_family,
    response,
    terms=None,
    n_reefs=None,
    extra=None,
):
    rows = []
    conf = result.conf_int()
    selected_terms = list(result.params.index) if terms is None else [term for term, _ in terms]
    for term in selected_terms:
        if term not in result.params.index:
            continue
        ci_low, ci_high = conf.loc[term].tolist()
        row = {
            "table": table,
            "model": model,
            "model_family": model_family,
            "response": response,
            "term": term,
            "label": TERM_LABELS.get(term, term),
            "beta": result.params[term],
            "z": result.tvalues[term],
            "p": result.pvalues[term],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_observations": int(result.nobs) if hasattr(result, "nobs") else np.nan,
            "n_reefs": n_reefs,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def build_episode_dataset(df):
    records = []
    for reef, reef_df in df.sort_values(["reef_name", "event_year"]).groupby("reef_name"):
        current = []
        episode_index = 0
        for row in reef_df.itertuples(index=False):
            if not current or int(row.event_year) <= int(current[-1].event_year) + 3:
                current.append(row)
            else:
                records.append(episode_record(reef, episode_index, current))
                episode_index += 1
                current = [row]
        if current:
            records.append(episode_record(reef, episode_index, current))

    episodes = pd.DataFrame(records)
    if episodes.empty:
        return episodes

    numeric_cols = [
        "baseline_hc",
        "nadir_hc",
        "recent_max_dhw",
        "recent_max_wind",
        "yrs_since_last_dist",
        "heatwave_years_5yr",
        "storm_years_5yr",
    ]
    for col in numeric_cols:
        episodes[f"{col}_z"] = zscore(episodes[col])
    return episodes.dropna(
        subset=[
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


def episode_record(reef, episode_index, rows):
    first = rows[0]
    years = [int(row.event_year) for row in rows]
    event_types = {row.event_type for row in rows}
    if "Concurrent" in event_types or (
        "Heatwave_Only" in event_types and "Storm_Only" in event_types
    ):
        event_type = "Concurrent"
    elif "Heatwave_Only" in event_types:
        event_type = "Heatwave_Only"
    else:
        event_type = "Storm_Only"

    nadir = min(float(row.nadir_hc) for row in rows)
    baseline = float(first.baseline_hc)
    record = {
        "reef_name": reef,
        "episode_id": f"{reef}_{episode_index}",
        "episode_start": min(years),
        "episode_end": max(years),
        "episode_span_years": max(years) - min(years) + 1,
        "n_events": len(rows),
        "event_type": event_type,
        "recent_max_dhw": max(float(row.recent_max_dhw) for row in rows),
        "recent_max_wind": max(float(row.recent_max_wind) for row in rows),
        "baseline_hc": baseline,
        "nadir_hc": nadir,
        "loss_abs": baseline - nadir,
        "retention": nadir / baseline if baseline > 0 else np.nan,
        "yrs_since_last_dist": float(first.yrs_since_last_dist),
        "heatwave_years_5yr": float(first.heatwave_years_5yr),
        "storm_years_5yr": float(first.storm_years_5yr),
    }
    for attr in ["sector", "region_lat", "site_lat", "site_lon", "event_period"]:
        if hasattr(first, attr):
            record[attr] = getattr(first, attr)
    return record


def haversine_matrix(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def morans_i(values, lat, lon, permutations=999, seed=42):
    valid = ~(pd.isna(values) | pd.isna(lat) | pd.isna(lon))
    values = np.asarray(values[valid], dtype=float)
    lat = np.asarray(lat[valid], dtype=float)
    lon = np.asarray(lon[valid], dtype=float)
    n = len(values)
    if n < 5:
        return {"n_reefs": n, "morans_i": np.nan, "p": np.nan}

    distances = haversine_matrix(lat, lon)
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances
    np.fill_diagonal(weights, 0.0)
    weights[~np.isfinite(weights)] = 0.0
    row_sums = weights.sum(axis=1)
    weights = np.divide(weights, row_sums[:, None], out=np.zeros_like(weights), where=row_sums[:, None] > 0)
    centered = values - values.mean()
    denominator = np.sum(centered**2)
    s0 = weights.sum()
    observed = (n / s0) * np.sum(weights * np.outer(centered, centered)) / denominator

    rng = np.random.default_rng(seed)
    permuted = []
    for _ in range(permutations):
        shuffled = rng.permutation(centered)
        permuted.append((n / s0) * np.sum(weights * np.outer(shuffled, shuffled)) / denominator)
    permuted = np.asarray(permuted)
    p_value = (np.sum(np.abs(permuted) >= abs(observed)) + 1) / (permutations + 1)
    return {"n_reefs": n, "morans_i": observed, "p": p_value}


def label_event(row):
    if row["has_storm"] == 1 and row["has_heatwave"] == 1:
        return "Concurrent"
    if row["has_storm"] == 1 and row["has_heatwave"] == 0:
        return "Storm_Only"
    if row["has_storm"] == 0 and row["has_heatwave"] == 1:
        return "Heatwave_Only"
    return "None"


def clean_event_summary():
    if not os.path.exists(MASTER_MATRIX_PATH):
        return None
    source = pd.read_csv(MASTER_MATRIX_PATH)
    source = source.sort_values(["reef_name", "year"]).reset_index(drop=True)
    source["event_type"] = source.apply(label_event, axis=1)
    events = source[source["event_type"].isin(["Concurrent", "Storm_Only", "Heatwave_Only"])]

    rows = []
    for _, row in events.iterrows():
        reef = row["reef_name"]
        event_year = row["year"]
        reef_data = source[source["reef_name"] == reef]
        baseline = reef_data[
            (reef_data["year"] >= event_year - 3) & (reef_data["year"] < event_year)
        ].dropna(subset=["HC_cover"])
        nadir = reef_data[
            (reef_data["year"] >= event_year) & (reef_data["year"] <= event_year + 3)
        ].dropna(subset=["HC_cover"])
        if baseline.empty or nadir.empty:
            continue

        prior = reef_data[
            (reef_data["year"] >= event_year - 2) & (reef_data["year"] < event_year)
        ]
        subsequent = reef_data[
            (reef_data["year"] > event_year) & (reef_data["year"] <= event_year + 2)
        ]
        prior_disturbance = prior["has_storm"].eq(1).any() or prior["has_heatwave"].eq(1).any()
        subsequent_disturbance = (
            subsequent["has_storm"].eq(1).any() or subsequent["has_heatwave"].eq(1).any()
        )
        rows.append(
            {
                "reef_name": reef,
                "event_year": event_year,
                "event_type": row["event_type"],
                "is_clean": not prior_disturbance and not subsequent_disturbance,
            }
        )
    return pd.DataFrame(rows)


def sample_composition(df):
    rows = [
        {
            "sample": "Full event-year matrix",
            "definition": "Target reef-years with baseline and nadir hard-coral cover available",
            "n_observations": len(df),
            "n_reefs": df["reef_name"].nunique(),
            "first_year": int(df["event_year"].min()),
            "last_year": int(df["event_year"].max()),
        },
        {
            "sample": "Positive-loss subset",
            "definition": "Full matrix restricted to loss_abs >= 0",
            "n_observations": int((df["loss_abs"] >= 0).sum()),
            "n_reefs": df.loc[df["loss_abs"] >= 0, "reef_name"].nunique(),
            "first_year": int(df.loc[df["loss_abs"] >= 0, "event_year"].min()),
            "last_year": int(df.loc[df["loss_abs"] >= 0, "event_year"].max()),
        },
    ]

    clean = clean_event_summary()
    if clean is not None:
        clean_subset = clean[clean["is_clean"]]
        rows.append(
            {
                "sample": "Clean-event subset",
                "definition": "Target reef-years with no additional storm or heatwave in the 2 years before or after the target event",
                "n_observations": len(clean_subset),
                "n_reefs": clean_subset["reef_name"].nunique(),
                "first_year": int(clean_subset["event_year"].min()),
                "last_year": int(clean_subset["event_year"].max()),
            }
        )

    for event_type, subset in df.groupby("event_type"):
        rows.append(
            {
                "sample": f"Event type: {event_type}",
                "definition": "Target-year event classification",
                "n_observations": len(subset),
                "n_reefs": subset["reef_name"].nunique(),
                "first_year": int(subset["event_year"].min()),
                "last_year": int(subset["event_year"].max()),
            }
        )

    history_class = np.select(
        [
            (df["heatwave_years_5yr"] > 0) & (df["storm_years_5yr"] > 0),
            (df["heatwave_years_5yr"] > 0) & (df["storm_years_5yr"] == 0),
            (df["heatwave_years_5yr"] == 0) & (df["storm_years_5yr"] > 0),
        ],
        ["Mixed history", "Thermal-only history", "Storm-only history"],
        default="No prior disturbance",
    )
    history_df = df.assign(history_class_5yr=history_class)
    for history, subset in history_df.groupby("history_class_5yr"):
        rows.append(
            {
                "sample": f"5-year history class: {history}",
                "definition": "Previous 5-year disturbance-history class",
                "n_observations": len(subset),
                "n_reefs": subset["reef_name"].nunique(),
                "first_year": int(subset["event_year"].min()),
                "last_year": int(subset["event_year"].max()),
            }
        )
    return pd.DataFrame(rows)


def variable_dictionary():
    return pd.DataFrame(
        [
            ["reef_name", "Reef identifier", "Categorical", "AIMS LTMP", "Used for reef-cluster robust standard errors and GEE grouping"],
            ["event_year", "Target disturbance year", "Year", "AIMS-aligned environmental records", "Years crossing heatwave, storm or concurrent thresholds"],
            ["event_type", "Target event class", "Categorical", "Derived", "Heatwave_Only, Storm_Only or Concurrent"],
            ["baseline_hc", "Pre-disturbance hard-coral cover", "% cover", "AIMS LTMP", "Most recent non-missing observation within 3 years before target year"],
            ["nadir_hc", "Post-disturbance nadir hard-coral cover", "% cover", "AIMS LTMP", "Minimum non-missing observation from target year through 3 years after target year"],
            ["loss_abs", "Absolute hard-coral cover loss", "percentage points", "Derived", "baseline_hc - nadir_hc"],
            ["positive_loss", "Positive-only absolute loss", "percentage points", "Derived", "max(loss_abs, 0)"],
            ["rel_loss_clipped", "Clipped relative loss", "Proportion", "Derived", "clip(loss_abs / baseline_hc, -1, 1)"],
            ["retention", "Post-event proportional retention", "Proportion", "Derived", "nadir_hc / baseline_hc"],
            ["recent_max_dhw", "Current thermal exposure", "Degree C-weeks", "NOAA Coral Reef Watch", "Maximum DHW in the target reef-year"],
            ["recent_max_wind", "Current wind exposure", "m s-1", "Australian Bureau of Meteorology", "Maximum wind speed in the target reef-year"],
            ["cumulative_dhw_wyr", "Historical cumulative DHW", "Degree C-weeks", "Derived", "Sum of annual maximum DHW over the previous w years"],
            ["cumulative_wind_wyr", "Historical cumulative storm-wind load", "m s-1", "Derived", "Sum of annual maximum wind only for storm years over the previous w years"],
            ["heatwave_years_wyr", "Historical heatwave-year count", "Years", "Derived", "Number of previous w years with maximum DHW >= 4 degree C-weeks"],
            ["storm_years_wyr", "Historical storm-year count", "Years", "Derived", "Number of previous w years with maximum wind >= 17.5 m s-1"],
            ["max_consecutive_heatwave_wyr", "Maximum consecutive heatwave run", "Years", "Derived", "Longest run of heatwave years within the previous w years"],
            ["max_consecutive_storm_wyr", "Maximum consecutive storm run", "Years", "Derived", "Longest run of storm years within the previous w years"],
            ["yrs_since_last_dist", "Return interval", "Years", "Derived", "Years since most recent prior heatwave or storm, capped at 10 years"],
            ["event_period", "Target-event period", "Categorical", "Derived", "pre-2016, 2016-2020, or 2021-2025"],
            ["sector", "Great Barrier Reef monitoring sector", "Categorical", "AIMS LTMP", "Used for spatial-sector fixed-effect sensitivity analysis"],
        ],
        columns=["variable", "definition", "unit", "source", "construction_rule"],
    )


def baseline_relationship_summary(df):
    rows = []
    subset_specs = [
        (
            "Full event-year matrix",
            "Target reef-years with baseline and nadir hard-coral cover available",
            df,
        ),
        (
            "Positive-loss subset",
            "Full matrix restricted to absolute loss >= 0",
            df[df["loss_abs"] >= 0],
        ),
    ]
    clean = clean_event_summary()
    if clean is not None:
        clean_keys = clean.loc[clean["is_clean"], ["reef_name", "event_year"]]
        clean_df = df.merge(clean_keys, on=["reef_name", "event_year"], how="inner")
        subset_specs.append(
            (
                "Clean-event subset",
                "Target reef-years with no additional storm or heatwave in the 2 years before or after the target event",
                clean_df,
            )
        )

    for subset, definition, data in subset_specs:
        result = smf.ols("loss_abs ~ baseline_hc", data=data).fit()
        rows.append(
            {
                "subset": subset,
                "definition": definition,
                "n_observations": len(data),
                "n_reefs": data["reef_name"].nunique(),
                "r_squared": result.rsquared,
                "baseline_beta": result.params["baseline_hc"],
                "p": result.pvalues["baseline_hc"],
            }
        )
    return pd.DataFrame(rows)


def vif_rows(df, formula, model_name):
    model = smf.ols(formula, data=df).fit()
    exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    rows = []
    for index, term in enumerate(exog.columns):
        if term == "Intercept":
            continue
        rows.append(
            {
                "section": "VIF",
                "diagnostic": model_name,
                "term": term,
                "label": TERM_LABELS.get(term, term),
                "value": variance_inflation_factor(exog.values, index),
                "n_observations": int(model.nobs),
                "n_reefs": df["reef_name"].nunique(),
            }
        )
    return rows


def standardized_difference(full, subset, column):
    full_values = full[column].dropna()
    subset_values = subset[column].dropna()
    pooled_sd = np.sqrt((full_values.var() + subset_values.var()) / 2)
    if pooled_sd == 0 or pd.isna(pooled_sd):
        return np.nan
    return (subset_values.mean() - full_values.mean()) / pooled_sd


def continuous_sample_rows(full, subset, subset_name, columns):
    rows = []
    for column, label in columns:
        rows.append(
            {
                "section": "Ecological-sample representativeness",
                "diagnostic": subset_name,
                "term": column,
                "label": label,
                "full_value": full[column].mean(),
                "subset_value": subset[column].mean(),
                "value": standardized_difference(full, subset, column),
                "n_observations": len(subset),
                "n_reefs": subset["reef_name"].nunique(),
                "note": "Value is standardized mean difference; positive values indicate higher mean in the subset.",
            }
        )
    return rows


def categorical_sample_rows(full, subset, subset_name, column, label):
    rows = []
    levels = sorted(set(full[column].dropna()).union(set(subset[column].dropna())))
    for level in levels:
        full_prop = full[column].eq(level).mean()
        subset_prop = subset[column].eq(level).mean()
        rows.append(
            {
                "section": "Ecological-sample representativeness",
                "diagnostic": subset_name,
                "term": f"{column}={level}",
                "label": f"{label}: {level}",
                "full_value": full_prop,
                "subset_value": subset_prop,
                "value": subset_prop - full_prop,
                "n_observations": len(subset),
                "n_reefs": subset["reef_name"].nunique(),
                "note": "Value is subset proportion minus full-sample proportion.",
            }
        )
    return rows


def diagnostic_summary(df, recurrence_formula):
    rows = vif_rows(df, recurrence_formula, "Main recurrence model")

    ecological_subsets = [
        ("Baseline total algae available", ["baseline_algae"]),
        ("Baseline juvenile density available", ["baseline_juveniles"]),
        ("Baseline macroalgae available", ["baseline_macroalgae"]),
        ("Baseline herbivorous fish density available", ["baseline_herbivores"]),
        (
            "All ecological covariates available",
            ["baseline_algae", "baseline_juveniles", "baseline_macroalgae", "baseline_herbivores"],
        ),
    ]
    continuous_columns = [
        ("baseline_hc", "Baseline hard-coral cover"),
        ("loss_abs", "Absolute hard-coral cover loss"),
        ("recent_max_dhw", "Recent maximum DHW"),
        ("recent_max_wind", "Recent maximum wind"),
        ("heatwave_years_5yr", "5-year heatwave years"),
        ("storm_years_5yr", "5-year storm years"),
        ("cumulative_dhw_5yr", "5-year cumulative DHW"),
        ("cumulative_wind_5yr", "5-year cumulative wind"),
        ("event_year", "Target event year"),
    ]
    for subset_name, required_cols in ecological_subsets:
        subset = df.dropna(subset=required_cols).copy()
        rows.append(
            {
                "section": "Ecological-sample representativeness",
                "diagnostic": subset_name,
                "term": "sample_size",
                "label": "Sample size retained",
                "value": len(subset) / len(df),
                "n_observations": len(subset),
                "n_reefs": subset["reef_name"].nunique(),
                "note": "Value is retained fraction of full event-year matrix.",
            }
        )
        rows.extend(continuous_sample_rows(df, subset, subset_name, continuous_columns))
        rows.extend(categorical_sample_rows(df, subset, subset_name, "event_type", "Target event type"))
        if "sector" in df.columns:
            rows.extend(categorical_sample_rows(df, subset, subset_name, "sector", "GBR sector"))
    return pd.DataFrame(rows)


def main():
    df = prepare_data()
    os.makedirs(TABLE_DIR, exist_ok=True)
    print("=" * 78)
    print(f"Valid observations: {len(df)}; reefs: {df['reef_name'].nunique()}")
    print("=" * 78)

    load_formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ cumulative_dhw_5yr_z + cumulative_wind_5yr_z "
        "+ yrs_since_last_dist_z + C(event_type)"
    )
    recurrence_formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ heatwave_years_5yr_z + storm_years_5yr_z "
        "+ yrs_since_last_dist_z + C(event_type)"
    )
    run_formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ max_consecutive_heatwave_5yr_z + max_consecutive_storm_5yr_z "
        "+ yrs_since_last_dist_z + C(event_type)"
    )

    load_ols = robust_ols(df, load_formula)
    load_gee = gee(df, load_formula)
    recurrence_ols = robust_ols(df, recurrence_formula)
    recurrence_gee = gee(df, recurrence_formula)
    run_ols = robust_ols(df, run_formula)
    run_gee = gee(df, run_formula)

    common_terms = [
        ("baseline_hc_z", "Baseline hard-coral cover"),
        ("recent_max_dhw_z", "Recent DHW"),
        ("recent_max_wind_z", "Recent wind"),
        ("yrs_since_last_dist_z", "Return interval"),
    ]
    load_terms = common_terms + [
        ("cumulative_dhw_5yr_z", "5-year cumulative DHW"),
        ("cumulative_wind_5yr_z", "5-year cumulative wind"),
    ]
    recurrence_terms = common_terms + [
        ("heatwave_years_5yr_z", "5-year heatwave years"),
        ("storm_years_5yr_z", "5-year storm years"),
    ]
    run_terms = common_terms + [
        ("max_consecutive_heatwave_5yr_z", "Max consecutive heatwave years"),
        ("max_consecutive_storm_5yr_z", "Max consecutive storm years"),
    ]

    print_terms("Model A: cumulative-load model, reef-cluster robust OLS", load_ols, load_terms)
    print_terms("Model A validation: cumulative-load model, GEE", load_gee, load_terms)
    print_terms("Model B: thermal recurrence model, reef-cluster robust OLS", recurrence_ols, recurrence_terms)
    print_terms("Model B validation: thermal recurrence model, GEE", recurrence_gee, recurrence_terms)
    print_terms("Model C: consecutive-run model, reef-cluster robust OLS", run_ols, run_terms)
    print_terms("Model C validation: consecutive-run model, GEE", run_gee, run_terms)

    variable_dictionary().to_csv(os.path.join(TABLE_DIR, "table_s1_variable_definitions.csv"), index=False)
    sample_composition(df).to_csv(os.path.join(TABLE_DIR, "table_s2_sample_composition.csv"), index=False)
    baseline_relationship_summary(df).to_csv(
        os.path.join(TABLE_DIR, "table_s6_baseline_loss_subsets.csv"),
        index=False,
    )

    recurrence_rows = []
    recurrence_rows.extend(
        model_rows(
            recurrence_ols,
            "Table S3",
            "Thermal recurrence model",
            "reef-cluster robust OLS",
            "loss_abs",
            terms=None,
            n_reefs=df["reef_name"].nunique(),
        )
    )
    recurrence_rows.extend(
        model_rows(
            recurrence_gee,
            "Table S3",
            "Thermal recurrence model",
            "GEE",
            "loss_abs",
            terms=None,
            n_reefs=df["reef_name"].nunique(),
        )
    )
    pd.DataFrame(recurrence_rows).to_csv(
        os.path.join(TABLE_DIR, "table_s3_recurrence_model_ols_gee.csv"),
        index=False,
    )
    diagnostic_summary(df, recurrence_formula).to_csv(
        os.path.join(TABLE_DIR, "table_s10_vif_and_sample_diagnostics.csv"),
        index=False,
    )

    print("\nResponse-metric sensitivity for heatwave years")
    print("----------------------------------------------")
    sensitivity_rows = []
    sensitivity_rows.extend(
        model_rows(
            load_ols,
            "Table S4",
            "Cumulative-load comparison",
            "reef-cluster robust OLS",
            "loss_abs",
            terms=load_terms,
            n_reefs=df["reef_name"].nunique(),
        )
    )
    sensitivity_rows.extend(
        model_rows(
            load_gee,
            "Table S4",
            "Cumulative-load comparison",
            "GEE",
            "loss_abs",
            terms=load_terms,
            n_reefs=df["reef_name"].nunique(),
        )
    )
    sensitivity_rows.extend(
        model_rows(
            run_ols,
            "Table S4",
            "Consecutive-run comparison",
            "reef-cluster robust OLS",
            "loss_abs",
            terms=run_terms,
            n_reefs=df["reef_name"].nunique(),
        )
    )
    sensitivity_rows.extend(
        model_rows(
            run_gee,
            "Table S4",
            "Consecutive-run comparison",
            "GEE",
            "loss_abs",
            terms=run_terms,
            n_reefs=df["reef_name"].nunique(),
        )
    )
    for outcome, label in [
        ("loss_abs", "Absolute loss"),
        ("positive_loss", "Positive loss"),
        ("rel_loss_clipped", "Clipped relative loss"),
    ]:
        formula = (
            f"{outcome} ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z "
            "+ yrs_since_last_dist_z + C(event_type)"
        )
        result = robust_ols(df, formula)
        sensitivity_rows.extend(
            model_rows(
                result,
                "Table S4",
                "Response-metric sensitivity",
                "reef-cluster robust OLS",
                outcome,
                terms=[("heatwave_years_5yr_z", "5-year heatwave years")],
                n_reefs=df["reef_name"].nunique(),
                extra={"response_label": label},
            )
        )
        ci = result.conf_int().loc["heatwave_years_5yr_z"].tolist()
        print(
            f"{label:24s} beta={result.params['heatwave_years_5yr_z']:7.3f}  "
            f"z={result.tvalues['heatwave_years_5yr_z']:7.3f}  "
            f"p={result.pvalues['heatwave_years_5yr_z']:.4g}  "
            f"95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]"
        )

    print("\nWindow sensitivity: reef-cluster robust OLS")
    print("------------------------------------------")
    window_rows = []
    for window in WINDOWS:
        for term, label in [
            (f"cumulative_dhw_{window}yr_z", "cumulative DHW"),
            (f"cumulative_wind_{window}yr_z", "cumulative wind"),
            (f"heatwave_years_{window}yr_z", "heatwave years"),
            (f"storm_years_{window}yr_z", "storm years"),
            (f"max_consecutive_heatwave_{window}yr_z", "consecutive heatwave years"),
            (f"max_consecutive_storm_{window}yr_z", "consecutive storm years"),
        ]:
            formula = (
                "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
                f"+ {term} + yrs_since_last_dist_z + C(event_type)"
            )
            result = robust_ols(df, formula)
            window_rows.extend(
                model_rows(
                    result,
                    "Table S5",
                    f"{window}-year window sensitivity",
                    "reef-cluster robust OLS",
                    "loss_abs",
                    terms=[(term, label)],
                    n_reefs=df["reef_name"].nunique(),
                    extra={"window_years": window, "metric": label},
                )
            )
            print(
                f"{window}-year {label:28s} beta={result.params[term]:7.3f}  "
                f"p={result.pvalues[term]:.4g}"
            )
    pd.DataFrame(window_rows).to_csv(
        os.path.join(TABLE_DIR, "table_s5_window_sensitivity.csv"),
        index=False,
    )

    print("\nLMM sensitivity with baseline control")
    print("-------------------------------------")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lmm = smf.mixedlm(load_formula, df, groups=df["reef_name"]).fit()
        print_terms("LMM cumulative-load sensitivity", lmm, load_terms)
        group_var = lmm.cov_re.iloc[0, 0] if lmm.cov_re.size else np.nan
        flagged = any(
            "singular" in str(w.message).lower() or "boundary" in str(w.message).lower()
            for w in caught
        )
        sensitivity_rows.extend(
            model_rows(
                lmm,
                "Table S4",
                "Cumulative-load LMM sensitivity",
                "linear mixed-effects model",
                "loss_abs",
                terms=load_terms,
                n_reefs=df["reef_name"].nunique(),
                extra={
                    "random_intercept_variance": group_var,
                    "singular_or_boundary_warning": flagged,
                },
            )
        )
        print(f"Random intercept variance: {group_var:.6f}")
        print(f"Singular or boundary warning captured: {flagged}")
    except Exception as exc:
        print(f"LMM fitting failed: {exc}")
        sensitivity_rows.append(
            {
                "table": "Table S4",
                "model": "Cumulative-load LMM sensitivity",
                "model_family": "linear mixed-effects model",
                "response": "loss_abs",
                "term": "model_fit",
                "label": "LMM fitting failed",
                "note": str(exc),
            }
        )

    pd.DataFrame(sensitivity_rows).to_csv(
        os.path.join(TABLE_DIR, "table_s4_sensitivity_models.csv"),
        index=False,
    )

    print("\nUpper-quantile sensitivity for loss ceiling")
    print("-------------------------------------------")
    quantile_formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ heatwave_years_5yr_z + storm_years_5yr_z "
        "+ yrs_since_last_dist_z + C(event_type)"
    )
    quantile_result = quantile_regression(df, quantile_formula, q=0.9)
    quantile_terms = common_terms + [
        ("heatwave_years_5yr_z", "5-year heatwave years"),
        ("storm_years_5yr_z", "5-year storm years"),
    ]
    print_terms("0.90 quantile recurrence model", quantile_result, quantile_terms)
    quantile_rows = model_rows(
        quantile_result,
        "Table S7",
        "0.90 quantile recurrence model",
        "quantile regression",
        "loss_abs",
        terms=None,
        n_reefs=df["reef_name"].nunique(),
        extra={"quantile": 0.9},
    )
    pd.DataFrame(quantile_rows).to_csv(
        os.path.join(TABLE_DIR, "table_s7_upper_quantile_boundary_check.csv"),
        index=False,
    )

    print("\nReviewer-risk sensitivity models")
    print("--------------------------------")
    diagnostic_rows = []

    alternative_specs = [
        (
            "Nadir-cover model",
            "nadir_hc",
            "nadir_hc ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)",
            [("baseline_hc_z", "Baseline hard-coral cover"), ("heatwave_years_5yr_z", "5-year heatwave years")],
        ),
        (
            "Proportional-retention model",
            "retention",
            "retention ~ recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z + C(event_type)",
            [("heatwave_years_5yr_z", "5-year heatwave years")],
        ),
    ]
    for model_name, response, formula, terms in alternative_specs:
        result = robust_ols(df, formula)
        diagnostic_rows.extend(
            model_rows(
                result,
                "Table S8",
                model_name,
                "reef-cluster robust OLS",
                response,
                terms=terms,
                n_reefs=df["reef_name"].nunique(),
            )
        )
        print_terms(model_name, result, terms)

    no_event_type_formula = (
        "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
        "+ heatwave_years_5yr_z + storm_years_5yr_z + yrs_since_last_dist_z"
    )
    no_event_type_result = robust_ols(df, no_event_type_formula)
    diagnostic_rows.extend(
        model_rows(
            no_event_type_result,
            "Table S8",
            "Recurrence model without target-event type",
            "reef-cluster robust OLS",
            "loss_abs",
            terms=[("heatwave_years_5yr_z", "5-year heatwave years")],
            n_reefs=df["reef_name"].nunique(),
        )
    )
    print_terms(
        "Recurrence model without target-event type",
        no_event_type_result,
        [("heatwave_years_5yr_z", "5-year heatwave years")],
    )

    period_formula = recurrence_formula + " + C(event_period)"
    year_formula = recurrence_formula + " + C(event_year)"
    sector_formula = recurrence_formula + " + C(sector)"
    reef_fe_formula = recurrence_formula + " + C(reef_name)"
    fixed_effect_specs = [
        ("Period fixed-effects recurrence model", period_formula),
        ("Event-year fixed-effects recurrence model", year_formula),
        ("GBR-sector fixed-effects recurrence model", sector_formula),
        ("Within-reef fixed-effects recurrence model", reef_fe_formula),
    ]
    for model_name, formula in fixed_effect_specs:
        data = df.dropna(subset=["sector"]) if "C(sector)" in formula else df
        result = robust_ols(data, formula)
        diagnostic_rows.extend(
            model_rows(
                result,
                "Table S8",
                model_name,
                "reef-cluster robust OLS",
                "loss_abs",
                terms=[("heatwave_years_5yr_z", "5-year heatwave years")],
                n_reefs=data["reef_name"].nunique(),
            )
        )
        print_terms(model_name, result, [("heatwave_years_5yr_z", "5-year heatwave years")])

    episodes = build_episode_dataset(df)
    if not episodes.empty:
        episode_formula = (
            "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            "+ heatwave_years_5yr_z + storm_years_5yr_z "
            "+ yrs_since_last_dist_z + C(event_type)"
        )
        episode_result = robust_ols(episodes, episode_formula)
        diagnostic_rows.extend(
            model_rows(
                episode_result,
                "Table S8",
                "Episode-level recurrence model",
                "reef-cluster robust OLS",
                "loss_abs",
                terms=[("heatwave_years_5yr_z", "5-year heatwave years")],
                n_reefs=episodes["reef_name"].nunique(),
                extra={
                    "n_episodes": len(episodes),
                    "episode_rule": "Target events on the same reef were merged when event years were <=3 years apart.",
                },
            )
        )
        print_terms(
            "Episode-level recurrence model",
            episode_result,
            [("heatwave_years_5yr_z", "5-year heatwave years")],
        )

    if {"site_lat", "site_lon"}.issubset(df.columns):
        residuals = df.assign(main_model_residual=recurrence_ols.resid)
        reef_residuals = (
            residuals.groupby("reef_name", as_index=False)
            .agg(
                main_model_residual=("main_model_residual", "mean"),
                site_lat=("site_lat", "first"),
                site_lon=("site_lon", "first"),
            )
            .dropna(subset=["site_lat", "site_lon"])
        )
        moran = morans_i(
            reef_residuals["main_model_residual"],
            reef_residuals["site_lat"],
            reef_residuals["site_lon"],
        )
        diagnostic_rows.append(
            {
                "table": "Table S8",
                "model": "Spatial residual check",
                "model_family": "Moran's I on reef-mean residuals",
                "response": "main recurrence-model residual",
                "term": "Morans_I",
                "label": "Moran's I",
                "beta": moran["morans_i"],
                "z": np.nan,
                "p": moran["p"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_observations": len(reef_residuals),
                "n_reefs": moran["n_reefs"],
                "permutations": 999,
            }
        )
        print(
            f"Moran's I on reef-mean residuals: I={moran['morans_i']:.3f}, "
            f"p={moran['p']:.4g}, reefs={moran['n_reefs']}"
        )

    pd.DataFrame(diagnostic_rows).to_csv(
        os.path.join(TABLE_DIR, "table_s8_reviewer_risk_sensitivity.csv"),
        index=False,
    )

    # ===================================================================
    # Ecological covariate sensitivity analysis (Table S9)
    # ===================================================================
    print("\nEcological covariate sensitivity analysis")
    print("------------------------------------------")
    eco_rows = []

    eco_vars = [
        ("baseline_algae_z", "Baseline total algae cover"),
        ("baseline_juveniles_z", "Baseline juvenile density"),
        ("baseline_macroalgae_z", "Baseline macroalgae cover"),
        ("baseline_herbivores_z", "Baseline herbivorous fish density"),
    ]

    # Model A reference: original cumulative-load model on full data (already fitted as load_ols)
    eco_rows.extend(
        model_rows(
            load_ols,
            "Table S9",
            "Reference: cumulative-load model (full sample)",
            "reef-cluster robust OLS",
            "loss_abs",
            terms=[
                ("cumulative_dhw_5yr_z", "5-year cumulative DHW"),
                ("cumulative_wind_5yr_z", "5-year cumulative wind"),
            ],
            n_reefs=df["reef_name"].nunique(),
            extra={"r_squared": load_ols.rsquared},
        )
    )
    print(
        f"  Reference model (n={int(load_ols.nobs)}): "
        f"cumDHW beta={load_ols.params['cumulative_dhw_5yr_z']:.4f}, "
        f"p={load_ols.pvalues['cumulative_dhw_5yr_z']:.4g}, "
        f"R2={load_ols.rsquared:.4f}"
    )

    # Models B-E: add one ecological covariate at a time
    for eco_term, eco_label in eco_vars:
        eco_subset = df.dropna(subset=[eco_term.replace("_z", "")])
        if len(eco_subset) < 30:
            continue
        eco_formula = (
            f"loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            f"+ cumulative_dhw_5yr_z + cumulative_wind_5yr_z "
            f"+ yrs_since_last_dist_z + {eco_term} + C(event_type)"
        )
        eco_result = robust_ols(eco_subset, eco_formula)
        eco_rows.extend(
            model_rows(
                eco_result,
                "Table S9",
                f"Cumulative-load + {eco_label}",
                "reef-cluster robust OLS",
                "loss_abs",
                terms=[
                    ("cumulative_dhw_5yr_z", "5-year cumulative DHW"),
                    ("cumulative_wind_5yr_z", "5-year cumulative wind"),
                    (eco_term, eco_label),
                ],
                n_reefs=eco_subset["reef_name"].nunique(),
                extra={"r_squared": eco_result.rsquared},
            )
        )
        print(
            f"  + {eco_label} (n={int(eco_result.nobs)}): "
            f"cumDHW beta={eco_result.params['cumulative_dhw_5yr_z']:.4f}, "
            f"p={eco_result.pvalues['cumulative_dhw_5yr_z']:.4g}, "
            f"{eco_term} beta={eco_result.params[eco_term]:.4f}, "
            f"p={eco_result.pvalues[eco_term]:.4g}, "
            f"R2={eco_result.rsquared:.4f}"
        )

    # Model F: full ecological model (all four covariates)
    eco_all_cols = [v.replace("_z", "") for v, _ in eco_vars]
    eco_full_subset = df.dropna(subset=eco_all_cols)
    if len(eco_full_subset) >= 30:
        eco_full_formula = (
            "loss_abs ~ baseline_hc_z + recent_max_dhw_z + recent_max_wind_z "
            "+ cumulative_dhw_5yr_z + cumulative_wind_5yr_z "
            "+ yrs_since_last_dist_z "
            "+ baseline_algae_z + baseline_juveniles_z "
            "+ baseline_macroalgae_z + baseline_herbivores_z "
            "+ C(event_type)"
        )
        eco_full_result = robust_ols(eco_full_subset, eco_full_formula)
        eco_full_terms = [
            ("cumulative_dhw_5yr_z", "5-year cumulative DHW"),
            ("cumulative_wind_5yr_z", "5-year cumulative wind"),
        ] + eco_vars
        eco_rows.extend(
            model_rows(
                eco_full_result,
                "Table S9",
                "Cumulative-load + all ecological covariates",
                "reef-cluster robust OLS",
                "loss_abs",
                terms=eco_full_terms,
                n_reefs=eco_full_subset["reef_name"].nunique(),
                extra={"r_squared": eco_full_result.rsquared},
            )
        )
        print(
            f"  Full eco model (n={int(eco_full_result.nobs)}): "
            f"cumDHW beta={eco_full_result.params['cumulative_dhw_5yr_z']:.4f}, "
            f"p={eco_full_result.pvalues['cumulative_dhw_5yr_z']:.4g}, "
            f"R2={eco_full_result.rsquared:.4f}"
        )

    pd.DataFrame(eco_rows).to_csv(
        os.path.join(TABLE_DIR, "table_s9_ecological_sensitivity.csv"),
        index=False,
    )

    print(f"\nExported supplementary tables to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
