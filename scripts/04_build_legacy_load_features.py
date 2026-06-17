import os

import numpy as np
import pandas as pd


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output")
WINDOWS = (3, 5, 7, 8)


def label_event(row):
    if row["has_storm"] == 1 and row["has_heatwave"] == 1:
        return "Concurrent"
    if row["has_storm"] == 1 and row["has_heatwave"] == 0:
        return "Storm_Only"
    if row["has_storm"] == 0 and row["has_heatwave"] == 1:
        return "Heatwave_Only"
    return "None"


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


def history_features(dist_history, event_year, window):
    past = dist_history[
        (dist_history["year"] >= event_year - window)
        & (dist_history["year"] < event_year)
    ].sort_values("year")

    heat_mask = past["has_heatwave"].eq(1)
    storm_mask = past["has_storm"].eq(1)
    concurrent_mask = heat_mask & storm_mask

    return {
        f"cumulative_dhw_{window}yr": past["max_dhw"].sum(skipna=True),
        f"cumulative_wind_{window}yr": past.loc[storm_mask, "max_wind_ms"].sum(skipna=True),
        f"heatwave_years_{window}yr": int(heat_mask.sum()),
        f"storm_years_{window}yr": int(storm_mask.sum()),
        f"concurrent_years_{window}yr": int(concurrent_mask.sum()),
        f"max_consecutive_heatwave_{window}yr": max_run(heat_mask.tolist()),
        f"max_consecutive_storm_{window}yr": max_run(storm_mask.tolist()),
    }


def build_matrix():
    merged_path = os.path.join(DATA_DIR, "eco_response_master_matrix_merged.csv")
    dist_path = os.path.join(DATA_DIR, "master_disturbance_matrix.csv")

    df_merged = pd.read_csv(merged_path).sort_values(["reef_name", "year"]).reset_index(drop=True)
    df_dist = pd.read_csv(dist_path).sort_values(["reef_name", "year"]).reset_index(drop=True)
    df_merged["event_type"] = df_merged.apply(label_event, axis=1)

    events = df_merged[df_merged["event_type"].isin(["Concurrent", "Storm_Only", "Heatwave_Only"])].copy()
    dist_by_reef = {reef: group for reef, group in df_dist.groupby("reef_name")}
    eco_by_reef = {reef: group for reef, group in df_merged.groupby("reef_name")}

    records = []
    for row in events.itertuples(index=False):
        reef = row.reef_name
        event_year = int(row.year)
        reef_data = eco_by_reef.get(reef)
        if reef_data is None:
            continue

        baseline_data = reef_data[
            (reef_data["year"] >= event_year - 3) & (reef_data["year"] < event_year)
        ].dropna(subset=["HC_cover"])
        if baseline_data.empty:
            continue

        response_data = reef_data[
            (reef_data["year"] >= event_year) & (reef_data["year"] <= event_year + 3)
        ].dropna(subset=["HC_cover"])
        if response_data.empty:
            continue

        baseline_row = baseline_data.iloc[-1]
        nadir_row = response_data.loc[response_data["HC_cover"].idxmin()]
        baseline_hc = baseline_row["HC_cover"]
        nadir_hc = nadir_row["HC_cover"]
        loss_abs = baseline_hc - nadir_hc

        dist_history = dist_by_reef.get(reef, pd.DataFrame(columns=df_dist.columns))
        past_all = dist_history[dist_history["year"] < event_year].sort_values("year", ascending=False)
        disturbances = past_all[past_all["has_storm"].eq(1) | past_all["has_heatwave"].eq(1)]
        yrs_since_last = event_year - int(disturbances.iloc[0]["year"]) if not disturbances.empty else 10

        record = {
            "reef_name": reef,
            "event_year": event_year,
            "event_type": row.event_type,
            "recent_max_dhw": row.max_dhw,
            "recent_max_wind": row.max_wind_ms,
            "baseline_hc": baseline_hc,
            "nadir_hc": nadir_hc,
            "loss_abs": loss_abs,
            "positive_loss": max(loss_abs, 0),
            "rel_loss": loss_abs / baseline_hc if baseline_hc > 0 else np.nan,
            "rel_loss_clipped": np.clip(loss_abs / baseline_hc, -1, 1) if baseline_hc > 0 else np.nan,
            "yrs_since_last_dist": yrs_since_last,
        }

        # Baseline ecological covariates (most recent non-missing value in pre-event window)
        for eco_col, rec_key in [
            ("ALGAE_cover", "baseline_algae"),
            ("MACROALGAE_cover", "baseline_macroalgae"),
            ("Juveniles", "baseline_juveniles"),
            ("Fish_Herbivores", "baseline_herbivores"),
        ]:
            eco_vals = baseline_data.dropna(subset=[eco_col])
            record[rec_key] = eco_vals.iloc[-1][eco_col] if not eco_vals.empty else np.nan
        for window in WINDOWS:
            record.update(history_features(dist_history, event_year, window))

        # Backward-compatible aliases used by older scripts and notes.
        record["cumulative_dhw_5yr"] = record["cumulative_dhw_5yr"]
        record["cumulative_wind_5yr"] = record["cumulative_wind_5yr"]

        records.append(record)

    legacy_df = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "legacy_load_analysis_matrix.csv")
    legacy_df.to_csv(output_path, index=False)

    print("=" * 70)
    print(f"Legacy-load matrix written to: {output_path}")
    print(f"Valid disturbance-response episodes: {len(legacy_df)}")
    print("=" * 70)
    print(legacy_df["event_type"].value_counts().to_string())
    print("\nThermal recurrence summary by target event type:")
    summary = legacy_df.groupby("event_type")[
        ["cumulative_dhw_5yr", "heatwave_years_5yr", "max_consecutive_heatwave_5yr",
         "cumulative_wind_5yr", "storm_years_5yr", "loss_abs"]
    ].mean()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    build_matrix()
