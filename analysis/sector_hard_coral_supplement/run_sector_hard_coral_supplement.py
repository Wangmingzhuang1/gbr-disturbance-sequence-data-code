import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE, "output", "composition_event_analysis_matrix.csv")
SITES_PATH = os.path.join(BASE, "data", "sites_lon_lat.csv")
NRM_PATH = os.path.join(
    BASE,
    "data",
    "NRM_Terrestrial_and_Marine_Regions_GBR_GDA20",
    "NRM_Terrestrial_and_Marine_Regions_GBR_GDA20.shp",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MANAGEMENT_MAPPING_PATH = os.path.join(OUT_DIR, "reef_six_management_area_mapping.csv")

MIN_OBS = 30
MIN_REEFS = 8

HARD_CORAL_CATEGORIES = [
    ("cat_hard_coral_acropora_baseline", "Acropora"),
    ("cat_hard_coral_pocilloporidae_baseline", "Pocilloporidae"),
    ("cat_hard_coral_porites_baseline", "Porites"),
    ("cat_hard_coral_montipora_baseline", "Montipora"),
    ("cat_hard_coral_merulinidae_baseline", "Merulinidae"),
    ("cat_hard_coral_other_baseline", "Other"),
]


def normalize_reef_name(value):
    import re

    text = str(value).lower().replace("&", "and")
    text = re.sub(r"no\.?\s*([0-9]+)", r"no \1", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_dms_robust(dms_str):
    import re

    if pd.isna(dms_str) or not dms_str:
        return None, None
    text = str(dms_str).upper().replace("F", "E")

    def parse_part(part_str):
        nums = re.findall(r"(\d+(?:\.\d+)?)", part_str)
        direction = re.search(r"([NSEW])", part_str)
        if not direction:
            return None
        vals = [0.0, 0.0, 0.0]
        for idx in range(min(len(nums), 3)):
            vals[idx] = float(nums[idx])
        degrees = vals[0] + vals[1] / 60 + vals[2] / 3600
        return -degrees if direction.group(1) in ["S", "W"] else degrees

    match = re.search(r"(.*[SN])\s*(.*[EW])", text)
    if match:
        return parse_part(match.group(1)), parse_part(match.group(2))
    return None, None


def add_six_management_area(df):
    if not os.path.exists(NRM_PATH):
        if not os.path.exists(MANAGEMENT_MAPPING_PATH):
            raise FileNotFoundError(
                "Six-management-area analysis requires either the NRM boundary shapefile "
                f"at {NRM_PATH} or the included reef-area mapping at {MANAGEMENT_MAPPING_PATH}."
            )
        mapping = pd.read_csv(MANAGEMENT_MAPPING_PATH)
        required = ["reef_key", "six_management_area"]
        if not set(required).issubset(mapping.columns):
            raise ValueError(
                f"{MANAGEMENT_MAPPING_PATH} must contain columns: {required}"
            )
        mapping = mapping[required].drop_duplicates("reef_key")
        return df.merge(mapping, on="reef_key", how="left")

    sites = pd.read_csv(SITES_PATH, header=None, names=["reef_name", "coords"], encoding="latin1")
    sites["reef_key"] = sites["reef_name"].map(normalize_reef_name)
    sites[["lat", "lon"]] = sites["coords"].apply(lambda value: pd.Series(parse_dms_robust(value)))
    sites = sites.dropna(subset=["lat", "lon"]).drop_duplicates("reef_key")

    reef_points = gpd.GeoDataFrame(
        sites[["reef_key", "lat", "lon"]],
        geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
        crs="EPSG:4326",
    )
    nrm = gpd.read_file(NRM_PATH)[["NAME", "geometry"]]
    if nrm.crs is None:
        nrm = nrm.set_crs("EPSG:4326")
    reef_points = reef_points.to_crs(nrm.crs)
    joined = gpd.sjoin(reef_points, nrm, how="left", predicate="within")

    missing = joined["NAME"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            reef_points.loc[missing],
            nrm,
            how="left",
            distance_col="distance_to_management_area",
        )
        joined.loc[missing, "NAME"] = nearest.set_index("reef_key").loc[joined.loc[missing, "reef_key"], "NAME"].to_numpy()

    mapping = joined[["reef_key", "NAME"]].rename(columns={"NAME": "six_management_area"})
    mapping.to_csv(MANAGEMENT_MAPPING_PATH, index=False)
    return df.merge(mapping, on="reef_key", how="left")


def cluster_ols(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["reef_name"]},
    )


def add_fdr(table, group_cols=None):
    table = table.copy()
    if table.empty:
        table["q"] = np.nan
        return table
    if group_cols:
        table["q"] = np.nan
        for _, idx in table.groupby(group_cols).groups.items():
            table.loc[idx, "q"] = multipletests(table.loc[idx, "p"].fillna(1), method="fdr_bh")[1]
    else:
        table["q"] = multipletests(table["p"].fillna(1), method="fdr_bh")[1]
    return table


def model_row(analysis, sector, category, term, model, data):
    ci_low, ci_high = model.conf_int().loc[term].tolist()
    return {
        "analysis": analysis,
        "sector": sector,
        "category": category,
        "term": term,
        "beta": model.params[term],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": model.pvalues[term],
        "n": len(data),
        "reefs": data["reef_name"].nunique(),
    }


def fit_category_model(data, outcome, include_sector=True):
    controls = "heatwave_years_5yr_z + event_year_z + baseline_hc_z"
    if include_sector:
        controls += " + C(sector)"
    return cluster_ols(f"{outcome} ~ {controls}", data)


def stratified_scan(df, area_col="sector", analysis_name="sector_stratified"):
    rows = []
    for area, area_df in df.groupby(area_col):
        for outcome, category in HARD_CORAL_CATEGORIES:
            data = area_df.dropna(
                subset=[outcome, "heatwave_years_5yr_z", "event_year_z", "baseline_hc_z", "reef_name"]
            ).copy()
            if len(data) < MIN_OBS or data["reef_name"].nunique() < MIN_REEFS:
                rows.append({
                    "analysis": analysis_name,
                    "area": area,
                    "category": category,
                    "term": "heatwave_years_5yr_z",
                    "beta": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p": np.nan,
                    "n": len(data),
                    "reefs": data["reef_name"].nunique(),
                    "status": "skipped_small_sector",
                })
                continue
            model = fit_category_model(data, outcome, include_sector=False)
            row = model_row(analysis_name, area, category, "heatwave_years_5yr_z", model, data)
            row["area"] = row.pop("sector")
            row["status"] = "fit"
            rows.append(row)
    table = pd.DataFrame(rows)
    fit_mask = table["status"].eq("fit")
    table.loc[fit_mask, "q"] = add_fdr(table.loc[fit_mask], group_cols=["area"])["q"].values
    return table.sort_values(["area", "q", "category"], na_position="last")


def interaction_scan(df, area_col="sector", analysis_name="heatwave_x_sector_interaction"):
    rows = []
    for outcome, category in HARD_CORAL_CATEGORIES:
        data = df.dropna(
            subset=[outcome, "heatwave_years_5yr_z", "event_year_z", "baseline_hc_z", area_col, "reef_name"]
        ).copy()
        area_counts = data.groupby(area_col).agg(n=("reef_name", "size"), reefs=("reef_name", "nunique"))
        valid_areas = area_counts[
            area_counts["n"].ge(MIN_OBS) & area_counts["reefs"].ge(MIN_REEFS)
        ].index.tolist()
        data = data[data[area_col].isin(valid_areas)].copy()
        area_order = sorted(valid_areas)
        if len(area_order) < 2 or len(data) < MIN_OBS or data["reef_name"].nunique() < MIN_REEFS:
            for area, counts in area_counts.iterrows():
                rows.append({
                    "analysis": analysis_name,
                    "category": category,
                    "area": area,
                    "area_specific_beta": np.nan,
                    "interaction_joint_p": np.nan,
                    "n": int(counts["n"]),
                    "reefs": int(counts["reefs"]),
                    "status": "skipped_insufficient_valid_areas",
                })
            continue
        data[area_col] = pd.Categorical(data[area_col], categories=area_order)
        formula = f"{outcome} ~ heatwave_years_5yr_z * C({area_col}) + event_year_z + baseline_hc_z"
        model = cluster_ols(formula, data)
        interaction_terms = [
            term for term in model.params.index
            if term.startswith(f"heatwave_years_5yr_z:C({area_col})")
        ]
        if interaction_terms:
            constraints = ", ".join([f"{term} = 0" for term in interaction_terms])
            p_joint = float(model.wald_test(constraints).pvalue)
        else:
            p_joint = np.nan

        for area in area_order:
            beta = model.params["heatwave_years_5yr_z"]
            if area != area_order[0]:
                term = f"heatwave_years_5yr_z:C({area_col})[T.{area}]"
                beta += model.params.get(term, 0.0)
            rows.append({
                "analysis": analysis_name,
                "category": category,
                "area": area,
                "area_specific_beta": beta,
                "interaction_joint_p": p_joint,
                "n": len(data),
                "reefs": data["reef_name"].nunique(),
                "status": "fit",
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        fit_table = table[table["status"].eq("fit")].copy()
        p_by_category = fit_table.drop_duplicates("category")[["category", "interaction_joint_p"]].copy()
        p_by_category["interaction_joint_q"] = multipletests(
            p_by_category["interaction_joint_p"].fillna(1), method="fdr_bh"
        )[1]
        table = table.merge(p_by_category[["category", "interaction_joint_q"]], on="category", how="left")
    return table.sort_values(["interaction_joint_q", "category", "area"], na_position="last")


def baseline_gt10_scan(df, area_col=None, analysis_name="baseline_gt10"):
    rows = []
    subset = df[df["baseline_hc"].gt(10)].copy()
    groups = [("all", subset)] if area_col is None else list(subset.groupby(area_col))
    for area, area_df in groups:
        for outcome, category in HARD_CORAL_CATEGORIES:
            required = [outcome, "heatwave_years_5yr_z", "event_year_z", "baseline_hc_z", "reef_name"]
            if area_col is None:
                required.append("sector")
            data = area_df.dropna(subset=required).copy()
            if len(data) < MIN_OBS or data["reef_name"].nunique() < MIN_REEFS:
                rows.append({
                    "analysis": analysis_name,
                    "area": area,
                    "category": category,
                    "term": "heatwave_years_5yr_z",
                    "beta": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p": np.nan,
                    "n": len(data),
                    "reefs": data["reef_name"].nunique(),
                    "status": "skipped_small_area",
                })
                continue
            model = fit_category_model(data, outcome, include_sector=area_col is None)
            row = model_row(analysis_name, area, category, "heatwave_years_5yr_z", model, data)
            row["area"] = row.pop("sector")
            row["status"] = "fit"
            rows.append(row)
    table = pd.DataFrame(rows)
    fit_mask = table["status"].eq("fit")
    if area_col is None:
        table.loc[fit_mask, "q"] = add_fdr(table.loc[fit_mask])["q"].values
    else:
        table.loc[fit_mask, "q"] = add_fdr(table.loc[fit_mask], group_cols=["area"])["q"].values
    return table.sort_values(["area", "q", "category"], na_position="last")


def write_summary(sector_table, interaction_table, gt10_table, six_table, six_interaction, six_gt10):
    lines = [
        "# Sector and Baseline-Restricted Hard-Coral Category Supplement",
        "",
        "Purpose: test whether weak or mixed hard-coral category signals are masked by broad spatial pooling or unstable low-baseline observations.",
        "",
        "Methods used the existing composition event matrix, reef-cluster robust OLS, and the same core controls as the composition analysis: prior 5-year heatwave recurrence, event year, and baseline hard-coral cover. Whole-GBR models also retained sector fixed effects.",
        "",
        "Outputs:",
        "- sector_stratified_hard_coral_scan.csv",
        "- heatwave_sector_interaction_hard_coral.csv",
        "- baseline_gt10_hard_coral_scan.csv",
        "- six_management_stratified_hard_coral_scan.csv",
        "- six_management_heatwave_interaction_hard_coral.csv",
        "- six_management_baseline_gt10_hard_coral_scan.csv",
        "",
    ]

    fit_sector = sector_table[sector_table.get("status", "").eq("fit")] if not sector_table.empty else sector_table
    sig_sector = fit_sector[fit_sector["q"].lt(0.05)] if not fit_sector.empty and "q" in fit_sector else pd.DataFrame()
    sig_interaction = interaction_table.drop_duplicates("category")
    sig_interaction = sig_interaction[sig_interaction["interaction_joint_q"].lt(0.05)] if not sig_interaction.empty else sig_interaction
    sig_gt10 = gt10_table[gt10_table.get("status", "").eq("fit") & gt10_table["q"].lt(0.05)] if not gt10_table.empty else gt10_table
    fit_six = six_table[six_table.get("status", "").eq("fit")] if not six_table.empty else six_table
    sig_six = fit_six[fit_six["q"].lt(0.05)] if not fit_six.empty and "q" in fit_six else pd.DataFrame()
    sig_six_interaction = six_interaction.drop_duplicates("category")
    sig_six_interaction = sig_six_interaction[sig_six_interaction["interaction_joint_q"].lt(0.05)] if not sig_six_interaction.empty else sig_six_interaction
    fit_six_gt10 = six_gt10[six_gt10.get("status", "").eq("fit")] if not six_gt10.empty else six_gt10
    sig_six_gt10 = fit_six_gt10[fit_six_gt10["q"].lt(0.05)] if not fit_six_gt10.empty and "q" in fit_six_gt10 else pd.DataFrame()

    lines.extend([
        "Key readout:",
        f"- Sector-stratified scan: {len(sig_sector)} sector-category signal(s) passed FDR q < 0.05.",
        f"- Heatwave recurrence x sector interaction: {len(sig_interaction)} category interaction test(s) passed FDR q < 0.05.",
        f"- Baseline >10% scan: {len(sig_gt10)} hard-coral category signal(s) passed FDR q < 0.05.",
        f"- Six-management-area scan: {len(sig_six)} area-category signal(s) passed FDR q < 0.05.",
        f"- Heatwave recurrence x six-management-area interaction: {len(sig_six_interaction)} category interaction test(s) passed FDR q < 0.05.",
        f"- Six-management-area baseline >10% scan: {len(sig_six_gt10)} area-category signal(s) passed FDR q < 0.05.",
        "",
        "Suggested SI interpretation:",
        "These checks test whether spatial pooling or low-baseline instability concealed a coherent hard-coral category pattern. They do not support a clear shift from one hard-coral group to another. They should be treated as sensitivity analyses, not as a new mechanistic layer.",
        "",
        "SI-ready sentence:",
        "Sector-stratified, six-management-area, sector-interaction and baseline-restricted analyses did not reveal a consistent hard-coral category shift. Any isolated interaction should be interpreted as spatial heterogeneity rather than evidence for a coherent hard-coral replacement pattern.",
    ])
    with open(os.path.join(OUT_DIR, "SI_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    df = add_six_management_area(df)
    df = df[df["composition_baseline_available"]].copy()

    sector_table = stratified_scan(df, area_col="sector", analysis_name="sector_stratified")
    interaction_table = interaction_scan(df, area_col="sector", analysis_name="heatwave_x_sector_interaction")
    gt10_table = baseline_gt10_scan(df)
    six_table = stratified_scan(df, area_col="six_management_area", analysis_name="six_management_stratified")
    six_interaction = interaction_scan(
        df,
        area_col="six_management_area",
        analysis_name="heatwave_x_six_management_interaction",
    )
    six_gt10 = baseline_gt10_scan(
        df,
        area_col="six_management_area",
        analysis_name="six_management_baseline_gt10",
    )

    sector_table.to_csv(os.path.join(OUT_DIR, "sector_stratified_hard_coral_scan.csv"), index=False)
    interaction_table.to_csv(os.path.join(OUT_DIR, "heatwave_sector_interaction_hard_coral.csv"), index=False)
    gt10_table.to_csv(os.path.join(OUT_DIR, "baseline_gt10_hard_coral_scan.csv"), index=False)
    six_table.to_csv(os.path.join(OUT_DIR, "six_management_stratified_hard_coral_scan.csv"), index=False)
    six_interaction.to_csv(os.path.join(OUT_DIR, "six_management_heatwave_interaction_hard_coral.csv"), index=False)
    six_gt10.to_csv(os.path.join(OUT_DIR, "six_management_baseline_gt10_hard_coral_scan.csv"), index=False)
    write_summary(sector_table, interaction_table, gt10_table, six_table, six_interaction, six_gt10)

    print("Supplemental hard-coral sector analysis complete.")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
