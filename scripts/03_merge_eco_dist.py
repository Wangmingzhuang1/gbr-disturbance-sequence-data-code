# -*- coding: utf-8 -*-
"""
Merge ecological survey data with the physical disturbance matrix.
Output: one row per reef-year.
"""
import os

import numpy as np
import pandas as pd

import config


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH = config.MASTER_MATRIX_PHYSICAL_PATH
REEF_DIR = os.path.join(BASE_DIR, 'data', 'reef_raw')
OUTPUT_FILE = config.MASTER_MATRIX_PATH
MERGE_AUDIT_FILE = config.REEF_MERGE_AUDIT_PATH
DUPLICATE_AUDIT_FILE = config.ECO_DUPLICATE_AGG_AUDIT_PATH
YEAR_ALIGNMENT_AUDIT_FILE = config.ECO_YEAR_ALIGNMENT_AUDIT_PATH

FILE_MAP = {
    'manta_HC_visual_estimate': ('HC_cover', 'mean', 'report_year', 'backup'),
    'photo_transect_HARD_CORAL': ('HC_cover', 'mean', 'report_year', 'primary'),
    'photo_transect_ALGAE': ('ALGAE_cover', 'mean', 'report_year', 'primary'),
    'photo_transect_SOFT_CORAL': ('SOFT_CORAL_cover', 'mean', 'report_year', 'primary'),
    'photo_transect_MACROALGAE': ('MACROALGAE_cover', 'mean', 'report_year', 'primary'),
    'juvenile_ABUNDANCE': ('Juveniles', 'mean', 'report_year', 'primary'),
    'fish_Herbivores': ('Fish_Herbivores', 'mean', 'report_year', 'primary'),
    'manta_cots_density': ('COTS_density', 'cotsptow', 'year', 'primary'),
}

DEPTH_VALIDATION_KEYS = {
    'photo_transect_HARD_CORAL',
    'photo_transect_ALGAE',
    'photo_transect_SOFT_CORAL',
    'photo_transect_MACROALGAE',
    'juvenile_ABUNDANCE',
}
REQUIRED_SINGLE_DEPTH = 9.0

CORE_COLUMN_ORDER = [
    'reef_name',
    'year',
    'region_lat',
    'sector',
    'max_dhw',
    'dhw_start_date',
    'dhw_peak_date',
    'dhw_start_aims_month',
    'dhw_peak_aims_month',
    'has_heatwave',
    'has_hw_moderate',
    'has_hw_severe',
    'min_dist',
    'max_wind_ms',
    'storm_name',
    'storm_id',
    'storm_track_id',
    'storm_track_key_type',
    'storm_peak_date',
    'storm_aims_month',
    'has_storm',
    'Disturbance_Type',
    'HC_cover',
    'SOFT_CORAL_cover',
    'ALGAE_cover',
    'MACROALGAE_cover',
    'Juveniles',
    'Fish_Herbivores',
    'COTS_density',
    'cots_lag_3yr',
    'decimal_date',
]


def _validate_depth(df, reef_folder, fname, key):
    if key not in DEPTH_VALIDATION_KEYS or 'depth' not in df.columns:
        return

    depth_values = pd.to_numeric(df['depth'], errors='coerce').dropna().unique().tolist()
    depth_values = sorted(float(value) for value in depth_values)
    if not depth_values:
        return
    if len(depth_values) > 1:
        raise ValueError(
            f"{reef_folder}/{fname} contains multiple depths {depth_values}. "
            "Master matrix requires a single depth per file."
        )
    if not np.isclose(depth_values[0], REQUIRED_SINGLE_DEPTH):
        raise ValueError(
            f"{reef_folder}/{fname} depth={depth_values[0]} != {REQUIRED_SINGLE_DEPTH}. "
            "Raw ecology files must be cleaned to 9 m before merge."
        )


def _decimal_year_to_aims_year(decimal_year):
    if pd.isna(decimal_year):
        return pd.NA
    try:
        value = float(decimal_year)
    except (TypeError, ValueError):
        return pd.NA

    year = int(np.floor(value))
    fraction = value - year
    day_of_year = max(1, int(round(fraction * 365.25)))
    try:
        timestamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year - 1)
    except Exception:
        return pd.NA
    return int(timestamp.year + (timestamp.month >= 7))


def _resolve_ecological_year(df, yr_col):
    work = df.copy()
    raw_year = pd.to_numeric(work[yr_col], errors='coerce')

    time_source = pd.Series('year_column', index=work.index, dtype='object')
    decimal_date = pd.Series(np.nan, index=work.index, dtype='float64')

    if 'survey_date' in work.columns:
        survey_date = pd.to_numeric(work['survey_date'], errors='coerce')
        mask = survey_date.notna()
        decimal_date.loc[mask] = survey_date.loc[mask]
        time_source.loc[mask] = 'survey_date'

    if 'date' in work.columns:
        date_decimal = pd.to_numeric(work['date'], errors='coerce')
        mask = decimal_date.isna() & date_decimal.notna()
        decimal_date.loc[mask] = date_decimal.loc[mask]
        time_source.loc[mask] = 'date'

    aims_year = decimal_date.apply(_decimal_year_to_aims_year)
    missing_aims = aims_year.isna() & raw_year.notna()
    aims_year.loc[missing_aims] = raw_year.loc[missing_aims].round().astype('Int64')

    alignment = pd.DataFrame({
        'raw_year': raw_year.round().astype('Int64'),
        'decimal_date': decimal_date,
        'year_source': time_source,
        'aims_year': pd.array(aims_year, dtype='Int64'),
    }, index=work.index)
    return alignment


def _trim_to_hc_window(merged):
    hc_rows = merged.dropna(subset=['HC_cover']).copy()
    if hc_rows.empty:
        raise ValueError("No HC_cover records found after merge. Cannot define survey windows.")

    survey_stats = (
        hc_rows.groupby('reef_name')['year']
        .agg(first_hc_year='min', last_hc_year='max')
        .reset_index()
    )
    survey_stats['window_start_year'] = survey_stats['first_hc_year'] - 3
    survey_stats['window_end_year'] = survey_stats['last_hc_year']

    trimmed = merged.merge(survey_stats, on='reef_name', how='inner')
    trimmed = trimmed[
        (trimmed['year'] >= trimmed['window_start_year']) &
        (trimmed['year'] <= trimmed['window_end_year'])
    ].copy()
    trimmed = trimmed.drop(columns=['first_hc_year', 'last_hc_year', 'window_start_year', 'window_end_year'])
    return trimmed


def _add_history_fields(df):
    df = df.sort_values(['reef_name', 'year']).copy()
    if 'COTS_density' in df.columns:
        cots_for_lag = pd.to_numeric(df['COTS_density'], errors='coerce').fillna(0.0)
        df['cots_lag_3yr'] = (
            cots_for_lag.groupby(df['reef_name'])
            .transform(lambda series: series.shift(1).rolling(3, min_periods=1).max())
        )
    else:
        df['cots_lag_3yr'] = np.nan
    return df


def _reorder_columns(df):
    ordered = [col for col in CORE_COLUMN_ORDER if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]

print("Loading physical disturbance matrix...")
df_dist = pd.read_csv(MATRIX_PATH)
print(f"  {len(df_dist)} rows, {df_dist['reef_name'].nunique()} reefs")
print(f"  Year range: {int(df_dist['year'].min())}-{int(df_dist['year'].max())}")

reef_folders = sorted(name for name in os.listdir(REEF_DIR) if os.path.isdir(os.path.join(REEF_DIR, name)))
dist_reefs = sorted(df_dist['reef_name'].dropna().astype(str).unique().tolist())
folder_set = set(reef_folders)
dist_set = set(dist_reefs)
merge_audit = pd.DataFrame({'reef_name': sorted(folder_set | dist_set)})
merge_audit['in_disturbance_matrix'] = merge_audit['reef_name'].isin(dist_set)
merge_audit['in_reef_raw'] = merge_audit['reef_name'].isin(folder_set)
merge_audit['match_status'] = np.where(
    merge_audit['in_disturbance_matrix'] & merge_audit['in_reef_raw'],
    'matched_exact',
    np.where(merge_audit['in_disturbance_matrix'], 'missing_reef_raw', 'missing_disturbance_matrix'),
)

print("Loading ecological survey data...")
eco_records = []
year_alignment_records = []
for reef_folder in reef_folders:
    reef_path = os.path.join(REEF_DIR, reef_folder)
    for fname in os.listdir(reef_path):
        if not fname.endswith('.csv'):
            continue
        key = fname[:-4]
        if key not in FILE_MAP:
            continue

        target_col, val_col, yr_col, priority = FILE_MAP[key]
        fpath = os.path.join(reef_path, fname)
        try:
            df = pd.read_csv(fpath, encoding='latin1')
            df.columns = df.columns.str.strip()
            if yr_col not in df.columns or val_col not in df.columns:
                print(f"  Warning: expected cols '{yr_col}'/'{val_col}' not in {fname}, skipping")
                continue

            _validate_depth(df, reef_folder, fname, key)

            df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
            year_alignment = _resolve_ecological_year(df, yr_col)
            df = df.join(year_alignment)
            df = df.dropna(subset=['aims_year', val_col]).copy()
            if df['decimal_date'].isna().any():
                df.loc[df['decimal_date'].isna(), 'decimal_date'] = df.loc[df['decimal_date'].isna(), 'raw_year'].astype(float)

            audit_chunk = df[['raw_year', 'decimal_date', 'year_source', 'aims_year']].copy()
            audit_chunk['reef_name'] = reef_folder
            audit_chunk['source_file'] = fname
            audit_chunk['column'] = target_col
            audit_chunk['year_changed'] = audit_chunk['raw_year'] != audit_chunk['aims_year']
            year_alignment_records.extend(audit_chunk.to_dict('records'))

            for _, row in df.iterrows():
                eco_records.append({
                    'reef_name': reef_folder,
                    'year': int(row['aims_year']),
                    'column': target_col,
                    'value': float(row[val_col]),
                    'decimal_date': row['decimal_date'],
                    'priority': priority,
                    'source_file': fname,
                })
        except Exception as e:
            print(f"  Warning: skipped {fname}: {e}")

df_eco_long = pd.DataFrame(eco_records)
print(f"  {len(df_eco_long)} eco records loaded across {df_eco_long['reef_name'].nunique()} reefs")

df_eco_long['priority_num'] = df_eco_long['priority'].map({'primary': 1, 'backup': 0})
df_eco_long['max_priority_num'] = df_eco_long.groupby(['reef_name', 'year', 'column'])['priority_num'].transform('max')
selected = df_eco_long[df_eco_long['priority_num'] == df_eco_long['max_priority_num']].copy()

duplicate_audit = (
    selected.groupby(['reef_name', 'year', 'column'], as_index=False)
    .agg(
        raw_record_n=('value', 'size'),
        retained_priority_num=('priority_num', 'max'),
        retained_priority=('priority', 'first'),
        value_min=('value', 'min'),
        value_max=('value', 'max'),
        value_mean=('value', 'mean'),
        decimal_date_min=('decimal_date', 'min'),
        decimal_date_max=('decimal_date', 'max'),
        decimal_date_mean=('decimal_date', 'mean'),
        source_file_n=('source_file', 'nunique'),
    )
)
duplicate_audit['aggregation_applied'] = duplicate_audit['raw_record_n'] > 1
duplicate_audit['is_hc_cover'] = duplicate_audit['column'] == 'HC_cover'
year_alignment_audit = pd.DataFrame(year_alignment_records)

df_eco_agg = (
    selected.groupby(['reef_name', 'year', 'column'], as_index=False)
    .agg(value=('value', 'mean'), decimal_date=('decimal_date', 'mean'))
)

df_eco_wide = df_eco_agg.pivot_table(
    index=['reef_name', 'year'],
    columns='column',
    values='value',
    aggfunc='mean',
).reset_index()
df_eco_wide.columns.name = None

df_date_pref = df_eco_agg[df_eco_agg['column'].isin(['HC_cover', 'Juveniles'])].copy()
date_priority = {'HC_cover': 0, 'Juveniles': 1}
df_date_pref['date_priority'] = df_date_pref['column'].map(date_priority)
df_date_pref = (
    df_date_pref
    .sort_values(['reef_name', 'year', 'date_priority'])
    .drop_duplicates(subset=['reef_name', 'year'], keep='first')
    [['reef_name', 'year', 'decimal_date']]
    .rename(columns={'decimal_date': 'decimal_date_priority'})
)
df_dec_fallback = (
    selected.groupby(['reef_name', 'year'], as_index=False)['decimal_date']
    .median()
    .rename(columns={'decimal_date': 'decimal_date_median_all'})
)
df_eco_wide = df_eco_wide.merge(df_date_pref, on=['reef_name', 'year'], how='left')
df_eco_wide = df_eco_wide.merge(df_dec_fallback, on=['reef_name', 'year'], how='left')
df_eco_wide['decimal_date'] = df_eco_wide['decimal_date_priority'].combine_first(df_eco_wide['decimal_date_median_all'])
df_eco_wide = df_eco_wide.drop(columns=['decimal_date_priority', 'decimal_date_median_all'])

print(f"  Wide eco table: {len(df_eco_wide)} reef-year rows")

print("Merging with physical disturbance matrix...")
merged = pd.merge(df_dist, df_eco_wide, on=['reef_name', 'year'], how='left')
merged['Disturbance_Type'] = merged['Disturbance_Type'].fillna('Safe')
if 'COTS_density' in merged.columns:
    merged['COTS_density'] = pd.to_numeric(merged['COTS_density'], errors='coerce').fillna(0.0)
merged = _trim_to_hc_window(merged)
merged = _add_history_fields(merged)
merged = _reorder_columns(merged.sort_values(['reef_name', 'year']).copy())

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False)
merge_audit.to_csv(MERGE_AUDIT_FILE, index=False)
duplicate_audit.to_csv(DUPLICATE_AUDIT_FILE, index=False)
year_alignment_audit.to_csv(YEAR_ALIGNMENT_AUDIT_FILE, index=False)

print(f"\nMerge complete. {len(merged)} records saved to {OUTPUT_FILE}")
print(f"Reefs: {merged['reef_name'].nunique()}")
print(f"Year range: {int(merged['year'].min())}-{int(merged['year'].max())}")
print(f"Rows with HC_cover data: {merged['HC_cover'].notna().sum()}")
print(f"Exact reef-name matches audited: {(merge_audit['match_status'] == 'matched_exact').sum()} / {len(merge_audit)}")
print(f"Duplicate groups aggregated: {int(duplicate_audit['aggregation_applied'].sum())}")
print(f"HC_cover duplicate groups aggregated: {int((duplicate_audit['aggregation_applied'] & duplicate_audit['is_hc_cover']).sum())}")
if not year_alignment_audit.empty:
    print(f"Eco rows remapped to AIMS year: {int(year_alignment_audit['year_changed'].sum())} / {len(year_alignment_audit)}")
print("\nDisturbance_Type breakdown:")
print(merged['Disturbance_Type'].value_counts())
