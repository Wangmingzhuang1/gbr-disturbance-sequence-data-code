import glob
import os
import re
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

import config


def sanitize_name(name):
    s = re.sub(r'[^a-zA-Z0-9]', '_', str(name))
    return re.sub(r'_+', '_', s).strip('_')


def normalize_dms_text(text):
    if pd.isna(text):
        return ''
    normalized = str(text)
    replacements = {
        '\xa1\xe3': '°',
        '\xa1\xe4': "'",
        '\xa1\xe5': '"',
        'º': '°',
        '˚': '°',
        '′': "'",
        '’': "'",
        '‵': "'",
        '″': '"',
        '“': '"',
        '”': '"',
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized.strip()


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371


def get_point_line_dist(px, py, x1, y1, x2, y2):
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_len_sq == 0:
        return haversine(px, py, x1, y1)
    u = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    return haversine(px, py, x1 + u * (x2 - x1), y1 + u * (y2 - y1))


def dms_to_decimal(dms_str):
    try:
        cleaned = normalize_dms_text(dms_str)
        matches = re.findall(r"([0-9\.\d°'\"]+)\s*([a-zA-Z])", cleaned)
        if len(matches) < 2:
            return None, None

        def extract_val(val_str, direction):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            if not nums:
                return 0
            d = float(nums[0])
            m = float(nums[1]) if len(nums) > 1 else 0
            s = float(nums[2]) if len(nums) > 2 else 0
            dec = d + m / 60.0 + s / 3600.0
            if direction.upper() in ['S', 'W']:
                dec = -dec
            return dec

        return extract_val(matches[0][0], matches[0][1]), extract_val(matches[1][0], matches[1][1])
    except (ValueError, IndexError):
        return None, None


def classify_sector(lat):
    if pd.isna(lat):
        return np.nan
    if lat >= -14.0:
        return 'North'
    if lat >= -16.0:
        return 'North-Central'
    if lat >= -20.0:
        return 'Central'
    return 'South'


KNOTS_TO_MS = 0.51444
RMAX_DEFAULT = 45.0
DECAY_EXPONENT = 0.8
BUFF_LIMIT = 100
PHYSICAL_THRESHOLD = 17.5

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES_PATH = os.path.join(BASE, 'data', 'sites_lon_lat.csv')
CYCLONE_PATH = os.path.join(BASE, 'data', 'IDCKMSTM0S.csv')
DAILY_CLIMATE_DIR = os.path.join(BASE, 'data', 'daily_climate_full')
OUTPUT_PATH = config.MASTER_MATRIX_PHYSICAL_PATH
AUDIT_PATH = config.STORM_IDENTIFICATION_AUDIT_PATH
SITE_AUDIT_PATH = config.SITE_COORDINATE_AUDIT_PATH

print("Building physical disturbance matrix (MRV wind decay model)...")
sites_df = pd.read_csv(SITES_PATH, header=None, names=['reef_name', 'coords'], encoding='latin1')
sites_df['reef_name'] = sites_df['reef_name'].apply(sanitize_name)
sites_df['coords_normalized'] = sites_df['coords'].apply(normalize_dms_text)
sites_df[['lat', 'lon']] = sites_df['coords'].apply(lambda x: pd.Series(dms_to_decimal(x)))
site_counts = sites_df['reef_name'].value_counts()
sites_df['duplicate_name_count'] = sites_df['reef_name'].map(site_counts)

site_audit = sites_df[['reef_name', 'coords', 'coords_normalized', 'lat', 'lon', 'duplicate_name_count']].copy()
site_audit['site_selection'] = 'kept_unique'
site_audit.loc[site_audit['lat'].isna() | site_audit['lon'].isna(), 'site_selection'] = 'dropped_parsing_failed'
site_audit.loc[
    (site_audit['duplicate_name_count'] > 1) &
    (site_audit['site_selection'] != 'dropped_parsing_failed'),
    'site_selection'
] = 'dropped_duplicate'

sites_df = sites_df.dropna(subset=['lat', 'lon']).copy()
if (site_audit['duplicate_name_count'] > 1).any():
    sites_df = sites_df.drop_duplicates(subset=['reef_name'], keep='last').copy()
    kept_names = set(sites_df['reef_name'])
    site_audit.loc[
        (site_audit['duplicate_name_count'] > 1) &
        site_audit['reef_name'].isin(kept_names) &
        (site_audit['site_selection'] != 'dropped_parsing_failed'),
        'site_selection'
    ] = 'kept_after_dedup'
    site_audit.loc[
        (site_audit['duplicate_name_count'] > 1) &
        ~site_audit['reef_name'].isin(kept_names) &
        (site_audit['site_selection'] != 'dropped_parsing_failed'),
        'site_selection'
    ] = 'dropped_duplicate'

sites_df['region_lat'] = sites_df['lat']
sites_df['sector'] = sites_df['lat'].apply(classify_sector)

print("Processing cyclone archives...")
raw_df = pd.read_csv(CYCLONE_PATH, skiprows=4, encoding='latin1')
raw_df = raw_df.dropna(subset=['TM'])
raw_df['NAME'] = raw_df['NAME'].str.strip().str.upper()
raw_df['DISTURBANCE_ID'] = raw_df['DISTURBANCE_ID'].fillna('').astype(str).str.strip()
v_cols = ['MAX_WIND_SPD', 'MAX_WIND_GUST']
available_cols = [c for c in v_cols if c in raw_df.columns]
for col in available_cols:
    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)
raw_df['v_center_ms'] = raw_df[available_cols].max(axis=1) * KNOTS_TO_MS
cyclone_df = raw_df[['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON', 'v_center_ms']].dropna(subset=['TM', 'LAT', 'LON'])
dates_tm = pd.to_datetime(cyclone_df['TM'], errors='coerce')
cyclone_df = cyclone_df.copy()
cyclone_df['year'] = dates_tm.dt.year + (dates_tm.dt.month >= 7).astype(int)
cyclone_df = cyclone_df.dropna(subset=['year']).query('year >= 1980')
cyclone_df['storm_track_id'] = cyclone_df['DISTURBANCE_ID'].where(
    cyclone_df['DISTURBANCE_ID'].str.len() > 0,
    cyclone_df['NAME'] + '_' + cyclone_df['year'].astype(int).astype(str),
)

valid_track_ids = cyclone_df['storm_track_id'].dropna().unique()

storm_matrix = []
storm_audit = []
for _, reef in sites_df.iterrows():
    year_winds = {}
    for track_id in valid_track_ids:
        group = cyclone_df[cyclone_df['storm_track_id'] == track_id].sort_values('TM')
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            p1 = group.iloc[i]
            p2 = group.iloc[i + 1]
            d_min = get_point_line_dist(reef['lon'], reef['lat'], p1['LON'], p1['LAT'], p2['LON'], p2['LAT'])
            if d_min > BUFF_LIMIT:
                continue
            d_safe = max(d_min, 0.001)
            v_center = max(p1['v_center_ms'], p2['v_center_ms'])
            v_local = v_center if d_safe <= RMAX_DEFAULT else v_center * (RMAX_DEFAULT / d_safe) ** DECAY_EXPONENT
            if v_local >= PHYSICAL_THRESHOLD:
                seg_year = int(p1['year'])
                year_winds.setdefault(seg_year, []).append({
                    'dist_km': d_safe,
                    'wind_ms': v_local,
                    'storm_name': p1['NAME'],
                    'storm_id': p1['DISTURBANCE_ID'],
                    'storm_track_id': track_id,
                    'tm': p1['TM'],
                    'track_key_type': 'DISTURBANCE_ID' if str(p1['DISTURBANCE_ID']).strip() else 'NAME_YEAR_FALLBACK',
                })
    for year, decayed_winds in year_winds.items():
        top = sorted(decayed_winds, key=lambda x: x['wind_ms'], reverse=True)[0]
        tm_dt = pd.to_datetime(top['tm'])
        storm_aims_month = (tm_dt.month - 7) % 12
        storm_matrix.append({
            'reef_name': reef['reef_name'],
            'year': year,
            'min_dist': top['dist_km'],
            'max_wind_ms': top['wind_ms'],
            'storm_name': top['storm_name'],
            'storm_id': top['storm_id'],
            'storm_track_id': top['storm_track_id'],
            'storm_track_key_type': top['track_key_type'],
            'storm_peak_date': tm_dt.year + (tm_dt.dayofyear / 365.25),
            'storm_aims_month': storm_aims_month,
        })
        storm_audit.append({
            'reef_name': reef['reef_name'],
            'year': year,
            'storm_track_id': top['storm_track_id'],
            'storm_id': top['storm_id'],
            'storm_name': top['storm_name'],
            'storm_track_key_type': top['track_key_type'],
            'max_wind_ms': top['wind_ms'],
            'min_dist_km': top['dist_km'],
            'buff_limit_km': BUFF_LIMIT,
            'rmax_default_km': RMAX_DEFAULT,
            'decay_exponent': DECAY_EXPONENT,
            'physical_threshold_ms': PHYSICAL_THRESHOLD,
        })

print("Merging thermal stress (daily DHW, AIMS ecological year Jul-Jun)...")
dhw_records = []
for reef_name in sites_df['reef_name'].unique():
    reef_file_prefix = sanitize_name(reef_name)
    search_pattern = os.path.join(DAILY_CLIMATE_DIR, f"{reef_file_prefix}_*_daily.csv")
    all_daily_data = []
    for daily_file in sorted(glob.glob(search_pattern)):
        try:
            df_c = pd.read_csv(daily_file)
            df_c.columns = df_c.columns.str.strip()
            d_col = next((c for c in df_c.columns if c.upper() in ['DHW', 'DEGREE_HEATING_WEEK']), None)
            if not d_col:
                continue
            df_c['time'] = pd.to_datetime(df_c['time'], errors='coerce')
            df_c = df_c.dropna(subset=['time'])
            df_c[d_col] = pd.to_numeric(df_c[d_col], errors='coerce')
            df_c['y_aims'] = df_c['time'].dt.year + (df_c['time'].dt.month >= 7).astype(int)
            df_c['decimal_date'] = df_c['time'].dt.year + (df_c['time'].dt.dayofyear / 365.25)
            all_daily_data.append(df_c[['y_aims', 'decimal_date', d_col]])
        except Exception as e:
            print(f"Warning: skipped {daily_file}: {e}")

    if all_daily_data:
        df_long = pd.concat(all_daily_data, ignore_index=True)
        d_col = next((c for c in df_long.columns if c.upper() in ['DHW', 'DEGREE_HEATING_WEEK']), None)
        yearly_max = df_long.dropna(subset=['y_aims', d_col]).groupby('y_aims')[d_col].max()
        for y_aims, max_val in yearly_max.items():
            year_data = df_long[df_long['y_aims'] == y_aims].dropna(subset=[d_col])
            peak_row = year_data[year_data[d_col] == max_val].iloc[0]
            onset_rows = year_data[year_data[d_col] > 4.0]
            dhw_start_date = onset_rows['decimal_date'].min() if not onset_rows.empty else np.nan

            def decimal_to_aims_month(dec):
                if pd.isna(dec):
                    return np.nan
                cal_year = int(dec)
                doy = int((dec - cal_year) * 365.25) + 1
                try:
                    dt = pd.Timestamp(year=cal_year, month=1, day=1) + pd.Timedelta(days=doy - 1)
                    return (dt.month - 7) % 12
                except Exception:
                    return np.nan

            dhw_records.append({
                'reef_name': reef_name,
                'year': int(y_aims),
                'max_dhw': max_val,
                'dhw_peak_date': peak_row['decimal_date'],
                'dhw_start_date': dhw_start_date,
                'dhw_peak_aims_month': decimal_to_aims_month(peak_row['decimal_date']),
                'dhw_start_aims_month': decimal_to_aims_month(dhw_start_date),
            })

dhw_df = pd.DataFrame(dhw_records)
master_matrix = pd.MultiIndex.from_product(
    [sites_df['reef_name'].unique(), range(config.YEAR_START, config.YEAR_END + 1)],
    names=['reef_name', 'year'],
).to_frame(index=False)
site_static = sites_df[['reef_name', 'region_lat', 'sector']].drop_duplicates(subset=['reef_name']).copy()
master_matrix = (
    master_matrix
    .merge(site_static, on='reef_name', how='left')
    .merge(dhw_df, on=['reef_name', 'year'], how='left')
    .merge(pd.DataFrame(storm_matrix), on=['reef_name', 'year'], how='left')
)
master_matrix['has_storm'] = (master_matrix['max_wind_ms'] >= PHYSICAL_THRESHOLD).astype(int)
master_matrix['has_heatwave'] = (master_matrix['max_dhw'] >= 4.0).astype(int)
master_matrix['has_hw_moderate'] = ((master_matrix['max_dhw'] >= 4.0) & (master_matrix['max_dhw'] < 8.0)).astype(int)
master_matrix['has_hw_severe'] = (master_matrix['max_dhw'] >= 8.0).astype(int)
master_matrix.fillna({'max_dhw': 0, 'max_wind_ms': 0}, inplace=True)


def _classify_disturbance(row):
    s = row['has_storm'] == 1
    sev = row['has_hw_severe'] == 1
    mod = row['has_hw_moderate'] == 1
    h = sev or mod

    if not (s and h):
        if s:
            return 'Isolated Storm'
        if sev:
            return 'Isolated Heatwave (Severe)'
        if mod:
            return 'Isolated Heatwave (Moderate)'
        return 'Safe'

    storm_month = row.get('storm_aims_month', np.nan)
    dhw_start_month = row.get('dhw_start_aims_month', np.nan)
    dhw_peak_month = row.get('dhw_peak_aims_month', np.nan)

    if pd.isna(storm_month) or pd.isna(dhw_start_month) or pd.isna(dhw_peak_month):
        return 'Synergistic Concurrent (Severe)' if sev else 'Synergistic Concurrent (Moderate)'

    sm = int(storm_month)
    ds = int(dhw_start_month)
    dp = int(dhw_peak_month)

    if dp < ds:
        return 'Synergistic Concurrent (Severe)' if sev else 'Synergistic Concurrent (Moderate)'
    if sm < ds:
        return 'Sequential S->H (Same Year)'
    if ds <= sm <= dp:
        return 'Synergistic Concurrent (Severe)' if sev else 'Synergistic Concurrent (Moderate)'
    return 'Sequential H->S (Same Year)'


master_matrix['Disturbance_Type'] = master_matrix.apply(_classify_disturbance, axis=1)

master_matrix.to_csv(OUTPUT_PATH, index=False)
pd.DataFrame(storm_audit).to_csv(AUDIT_PATH, index=False)
site_audit.to_csv(SITE_AUDIT_PATH, index=False)
print(f"Done. {len(master_matrix)} records.")
print(f"Year range: {int(master_matrix['year'].min())}-{int(master_matrix['year'].max())}")
print(f"Retained reefs: {master_matrix['reef_name'].nunique()}")
print(master_matrix['Disturbance_Type'].value_counts())
