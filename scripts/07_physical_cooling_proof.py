# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt
import re
import config

# ==============================================================================
# 物理冷却效应验证 (Physical Cooling Evidence Analysis)
# 
# 目的: 针对 Concurrent (Storm + Heatwave) 事件，提取真实的 SST 降温幅度
# 逻辑: 
# 1. 精确定位气旋最接近该礁石的日期 (T_0)
# 2. 从下载的高频日度数据中提取 T-3~T-1 的均温，以及 T+1~T+7 的最低温
# 3. 计算 \Delta SST，若 > 1°C 则视为强物理缓冲
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_FILE = config.MASTER_MATRIX_PHYSICAL_PATH
CYCLONE_FILE = os.path.join(BASE_DIR, 'data', 'IDCKMSTM0S.csv')
COORDS_FILE = os.path.join(BASE_DIR, 'data', 'sites_lon_lat.csv')
DAILY_DATA_DIR = os.path.join(BASE_DIR, 'data', 'daily_climate_full')
OUTPUT_FILE = config.COOLING_EVIDENCE_RESULTS_PATH
CLEAN_FILE = config.CONCURRENT_DAILY_PROOF_PATH


def sanitize_name(name):
    s = re.sub(r'[^a-zA-Z0-9]', '_', str(name))
    return re.sub(r'_+', '_', s).strip('_')

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1)/2)**2
    return 6371 * 2 * asin(sqrt(a)) 

def get_coords():
    import re
    def dms_to_decimal(dms_str):
        try:
            parts = re.split(r'([NSEW])', str(dms_str))
            lat_val, lat_dir, lon_val, lon_dir = parts[0], parts[1], parts[2], parts[3]
            nums_lat = re.findall(r"[-+]?\d*\.\d+|\d+", lat_val)
            lat = float(nums_lat[0]) + (float(nums_lat[1])/60.0 if len(nums_lat)>1 else 0)
            if lat_dir in ['S']: lat = -lat
            nums_lon = re.findall(r"[-+]?\d*\.\d+|\d+", lon_val)
            lon = float(nums_lon[0]) + (float(nums_lon[1])/60.0 if len(nums_lon)>1 else 0)
            if lon_dir in ['W']: lon = -lon
            return lat, lon
        except: return None, None
    df = pd.read_csv(COORDS_FILE, header=None, names=['reef_name', 'coords'], encoding='latin1')
    df['reef_name'] = df['reef_name'].apply(sanitize_name)
    res = {}
    for _, r in df.iterrows():
        lat, lon = dms_to_decimal(r['coords'])
        if lat is not None: res[r['reef_name']] = (lat, lon)
    return res

print("Loading datasets...")
master = pd.read_csv(MASTER_FILE)
compounds = master[(master['has_storm']==1) & (master['has_heatwave']==1)].copy()

cyclones = pd.read_csv(CYCLONE_FILE, skiprows=4, encoding='latin1')
cyclones['NAME'] = cyclones['NAME'].fillna('').astype(str).str.strip().str.upper()
cyclones['DISTURBANCE_ID'] = cyclones['DISTURBANCE_ID'].fillna('').astype(str).str.strip()
cyclones = cyclones[['NAME', 'DISTURBANCE_ID', 'TM', 'LAT', 'LON']].dropna(subset=['TM', 'LAT', 'LON'])
cyclones['date'] = pd.to_datetime(cyclones['TM'], errors='coerce')
cyclones = cyclones.dropna(subset=['date'])

coords = get_coords()

results_raw = []
results_clean = []

print("Analyzing compound events for physical cooling and timing...")
for idx, row in compounds.iterrows():
    reef = row['reef_name']
    year = int(row['year'])
    storm_name = str(row.get('storm_name', '')).strip().upper()
    storm_id = str(row.get('storm_id', '')).strip()
    
    if reef not in coords: continue
    r_lat, r_lon = coords[reef]
    
    # Match cyclone track consistently with 02_build_matrix.py: prefer DISTURBANCE_ID, then storm name within the same season window.
    c_subset = pd.DataFrame()
    match_method = ''
    if storm_id:
        c_subset = cyclones[
            (cyclones['DISTURBANCE_ID'] == storm_id) &
            (cyclones['date'].dt.year >= year-1) &
            (cyclones['date'].dt.year <= year)
        ].copy()
        if not c_subset.empty:
            match_method = 'storm_id'

    if c_subset.empty and storm_name:
        c_subset = cyclones[
            (cyclones['NAME'] == storm_name) &
            (cyclones['date'].dt.year >= year-1) &
            (cyclones['date'].dt.year <= year)
        ].copy()
        if not c_subset.empty:
            match_method = 'storm_name'

    if c_subset.empty:
        results_raw.append({
            'reef_name': reef, 'year': year, 'storm_name': storm_name, 'storm_id': storm_id,
            'storm_date': '', 'pre_storm_sst': np.nan, 'post_storm_min_sst': np.nan, 'sst_cooling': np.nan,
            'dhw_at_storm': np.nan, 'max_dhw_year': np.nan, 'is_antagonistic': False,
            'is_effective_cooling': False, 'match_method': 'unmatched', 'matched_track_points': 0
        })
        continue
    
    c_subset['dist'] = c_subset.apply(lambda x: haversine(r_lon, r_lat, x['LON'], x['LAT']), axis=1)
    closest_point = c_subset.loc[c_subset['dist'].idxmin()]
    storm_date = closest_point['date'].replace(hour=0, minute=0, second=0)
    
    daily_file = os.path.join(DAILY_DATA_DIR, f"{sanitize_name(reef)}_{year}_daily.csv")
    if not os.path.exists(daily_file):
        results_raw.append({
            'reef_name': reef, 'year': year, 'storm_name': storm_name, 'storm_id': storm_id,
            'storm_date': storm_date.strftime('%Y-%m-%d'), 'pre_storm_sst': np.nan, 'post_storm_min_sst': np.nan,
            'sst_cooling': np.nan, 'dhw_at_storm': np.nan, 'max_dhw_year': np.nan, 'is_antagonistic': False,
            'is_effective_cooling': False, 'match_method': f'{match_method}_daily_missing', 'matched_track_points': len(c_subset)
        })
        continue
    
    daily = pd.read_csv(daily_file)
    daily.columns = [c.strip() for c in daily.columns]
    daily['SST'] = pd.to_numeric(daily['SST'], errors='coerce')
    daily['DHW'] = pd.to_numeric(daily['DHW'], errors='coerce')
    daily = daily.dropna(subset=['time', 'SST', 'DHW'])
    daily['time'] = pd.to_datetime(daily['time']).dt.tz_localize(None).dt.normalize()
    
    # Timing analysis using absolute DateTimes (avoids Southern Hemisphere year-end wrap bugs)
    # Use DHW > 4.0 threshold consistent with 02_build_matrix.py (bleaching threshold)
    dhw_start = daily[daily['DHW'] > 4.0]
    dhw_start_date = dhw_start['time'].min() if not dhw_start.empty else pd.NaT
    
    peak_row = daily.loc[daily['DHW'].idxmax()]
    dhw_peak_date = peak_row['time']
    dhw_peak = peak_row['DHW']
    
    # Physical cooling proof
    pre_mask = (daily['time'] >= storm_date - pd.Timedelta(days=4)) & (daily['time'] < storm_date)
    post_mask = (daily['time'] > storm_date) & (daily['time'] <= storm_date + pd.Timedelta(days=7))
    pre_sst = daily.loc[pre_mask, 'SST'].mean()
    post_sst = daily.loc[post_mask, 'SST'].min()
    delta_sst = pre_sst - post_sst if not pd.isna(pre_sst) and not pd.isna(post_sst) else np.nan
    
    # Core logic: Antagonistic if it falls during heat accumulation phase (using safe Dates)
    is_antagonistic = pd.notna(dhw_start_date) and (storm_date >= dhw_start_date) and (storm_date <= dhw_peak_date)
    
    dhw_at_storm_val = round(daily.loc[daily['time'] == storm_date, 'DHW'].mean(), 2) if not daily.loc[daily['time'] == storm_date, 'DHW'].empty else 0
    
    results_raw.append({
        'reef_name': reef, 'year': year, 'storm_name': storm_name, 'storm_id': storm_id, 'storm_date': storm_date.strftime('%Y-%m-%d'),
        'pre_storm_sst': round(pre_sst, 2), 'post_storm_min_sst': round(post_sst, 2), 'sst_cooling': round(delta_sst, 2),
        'dhw_at_storm': dhw_at_storm_val,
        'max_dhw_year': round(dhw_peak, 2),
        'is_antagonistic': is_antagonistic,
        'is_effective_cooling': delta_sst >= 1.0 if not pd.isna(delta_sst) else False,
        'match_method': match_method,
        'matched_track_points': len(c_subset),
    })
    
    results_clean.append({
        'reef_name': reef, 'year': year, 'storm': storm_name, 'storm_id': storm_id,
        'storm_month': storm_date.month,
        'dhw_start_month': dhw_start_date.month if pd.notna(dhw_start_date) else np.nan, 
        'dhw_peak_month': dhw_peak_date.month if pd.notna(dhw_peak_date) else np.nan,
        'dhw_peak': dhw_peak, 'dhw_at_storm': dhw_at_storm_val,
        'is_antagonistic_cooling': is_antagonistic,
        'match_method': match_method,
    })

pd.DataFrame(results_raw).to_csv(OUTPUT_FILE, index=False)
pd.DataFrame(results_clean).to_csv(CLEAN_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE} and {CLEAN_FILE}")
print(f"\nExtracted physical cooling evidence for {len(results_raw)} valid compound events.")
print(f"Results saved to {OUTPUT_FILE}")

# Calculate quick statistics
results_df = pd.DataFrame(results_raw)
if not results_df.empty:
    avg_cooling_all = results_df['sst_cooling'].mean()
    effective_count = results_df['is_effective_cooling'].sum()
    print(f"\n--- Cooling Validation Stats (ALL {len(results_df)} events) ---")
    print(f"Average SST Drop (all events): {avg_cooling_all:.2f} °C")
    print(f"Events yielding >1°C cooling: {effective_count} out of {len(results_df)} ({(effective_count/len(results_df))*100:.1f}%)")
    
    valid_cooling = results_df[results_df['dhw_at_storm'] > 0].copy()
    print(f"\n--- CONSERVATIVE Analysis (storms during active DHW, n={len(valid_cooling)}) ---")
    if not valid_cooling.empty:
        avg_cooling_valid = valid_cooling['sst_cooling'].mean()
        print(f"Average SST Drop (DHW>0 subset): {avg_cooling_valid:.2f} °C")
        
        interrupted = valid_cooling[valid_cooling['dhw_at_storm'] < results_df['max_dhw_year'].loc[valid_cooling.index]]
        print(f"Storms that interrupted DHW climbing phase: {len(interrupted)} out of {len(valid_cooling)} ({(len(interrupted)/len(valid_cooling))*100:.1f}%)")
        
        strong_cooling = valid_cooling[valid_cooling['sst_cooling'] >= 1.0]
        print(f"Events with strong cooling (>1°C): {len(strong_cooling)} out of {len(valid_cooling)} ({(len(strong_cooling)/len(valid_cooling))*100:.1f}%)")
    
    pre_dhw_storms = results_df[results_df['dhw_at_storm'] == 0]
    print(f"\nNote: {len(pre_dhw_storms)} storms arrived before DHW accumulation began (dhw_at_storm=0)")

