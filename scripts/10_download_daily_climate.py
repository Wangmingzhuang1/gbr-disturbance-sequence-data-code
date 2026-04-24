# -*- coding: utf-8 -*-
"""
Download daily SST and DHW for reef points from NOAA ERDDAP.

Default behavior:
- read reef coordinates from data/sites_lon_lat.csv
- infer each reef's continuous survey window from local ecological files
- download every year from (survey_start_year - 3) through survey_end_year
- write only missing daily files

Optional:
- pass reef names as CLI args to restrict downloads to specific reefs
"""

import argparse
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config


ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap/griddap"
STRIDE = 1
MAX_RETRIES = 3
TIMEOUT = 60
DELAY = 1.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORDS_FILE = os.path.join(BASE_DIR, 'data', 'sites_lon_lat.csv')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'daily_climate_full')
REEF_RAW_DIR = os.path.join(BASE_DIR, 'data', 'reef_raw')
os.makedirs(OUT_DIR, exist_ok=True)


def sanitize_name(name):
    s = re.sub(r'[^a-zA-Z0-9]', '_', str(name))
    return re.sub(r'_+', '_', s).strip('_')


def normalize_dms_text(text):
    if pd.isna(text):
        return ''
    normalized = str(text)
    replacements = {
        '��': '°',
        '掳': '°',
        '潞': '°',
        '藲': '°',
        '′': "'",
        '’': "'",
        '`': "'",
        '″': '"',
        '“': '"',
        '”': '"',
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized.strip()


def dms_to_decimal(dms_str):
    try:
        cleaned = normalize_dms_text(dms_str)
        parts = re.split(r'([NSEW])', cleaned)
        lat_val, lat_dir, lon_val, lon_dir = parts[0], parts[1], parts[2], parts[3]

        def convert(val, direction):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
            d = float(nums[0])
            m = float(nums[1]) if len(nums) > 1 else 0
            s = float(nums[2]) if len(nums) > 2 else 0
            dec = d + m / 60.0 + s / 3600.0
            if direction in ['S', 'W']:
                dec = -dec
            return dec

        return convert(lat_val, lat_dir), convert(lon_val, lon_dir)
    except Exception:
        return None, None


def load_coordinates():
    df = pd.read_csv(COORDS_FILE, header=None, names=['reef_name', 'coords'], encoding='latin1')
    df['reef_name'] = df['reef_name'].astype(str).str.strip().apply(sanitize_name)
    df[['lat', 'lon']] = df['coords'].apply(lambda x: pd.Series(dms_to_decimal(x)))
    df = df.dropna(subset=['lat', 'lon']).copy()
    df = df.drop_duplicates(subset=['reef_name'], keep='last')
    return {row['reef_name']: (row['lat'], row['lon']) for _, row in df.iterrows()}


def fetch_chunk(dataset_id, variable, lat, lon, year):
    t_start = f"{year - 1}-11-01T12:00:00Z"
    t_end = f"{year}-10-31T12:00:00Z"
    url = (
        f"{ERDDAP_BASE}/{dataset_id}.csv?"
        f"{variable}[({t_start}):{STRIDE}:({t_end})]"
        f"[({lat}):1:({lat})]"
        f"[({lon}):1:({lon})]"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CEE-Daily-Climate-Downloader)"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                content = response.read().decode("utf-8")
                lines = content.strip().split("\n")
                return lines[2:] if len(lines) > 2 else []
        except Exception:
            time.sleep(2 * attempt)
    return None


def parse_rows(rows, column_names):
    return pd.DataFrame([row.split(",")[:len(column_names)] for row in rows], columns=column_names)


def download_daily(reef_name, lat, lon, year):
    out_file = os.path.join(OUT_DIR, f"{sanitize_name(reef_name)}_{year}_daily.csv")
    if os.path.exists(out_file):
        return "SKIP", 0

    sst_rows = fetch_chunk("noaacrwsstDaily", "analysed_sst", lat, lon, year)
    time.sleep(DELAY)
    dhw_rows = fetch_chunk("noaacrwdhwDaily", "degree_heating_week", lat, lon, year)
    time.sleep(DELAY)

    if sst_rows is None or dhw_rows is None:
        return "FAILED", 0
    if not sst_rows or not dhw_rows:
        return "EMPTY", 0

    sst_df = parse_rows(sst_rows, ["time", "latitude", "longitude", "SST"])
    dhw_df = parse_rows(dhw_rows, ["time", "latitude", "longitude", "DHW"])
    merged = pd.merge(sst_df, dhw_df[["time", "DHW"]], on="time", how="inner")
    if merged.empty:
        return "EMPTY", 0

    merged.to_csv(out_file, index=False)
    return "OK", len(merged)


def infer_survey_year_range(reef_name):
    reef_dir = os.path.join(REEF_RAW_DIR, reef_name)
    if not os.path.isdir(reef_dir):
        return None

    year_values = []
    for filename in os.listdir(reef_dir):
        if not filename.endswith('.csv'):
            continue
        path = os.path.join(reef_dir, filename)
        try:
            df = pd.read_csv(path, nrows=5000, encoding='latin1')
        except Exception:
            continue

        candidate_cols = [col for col in ['report_year', 'year'] if col in df.columns]
        for col in candidate_cols:
            years = pd.to_numeric(df[col], errors='coerce').dropna()
            if not years.empty:
                year_values.extend(years.round().astype(int).tolist())

    if not year_values:
        return None

    survey_start = min(year_values)
    survey_end = max(year_values)
    return max(config.YEAR_START, survey_start - 3), min(config.YEAR_END, survey_end)


def build_tasks(coords, selected_reefs=None, year_start=None, year_end=None):
    reef_names = sorted(coords.keys())
    if selected_reefs:
        wanted = {sanitize_name(name): name for name in selected_reefs}
        reef_names = [reef for reef in reef_names if reef in wanted or sanitize_name(reef) in wanted]

    tasks = []
    for reef_name in reef_names:
        lat, lon = coords[reef_name]
        if year_start is not None or year_end is not None:
            start_year = config.YEAR_START if year_start is None else int(year_start)
            end_year = config.YEAR_END if year_end is None else int(year_end)
        else:
            inferred = infer_survey_year_range(reef_name)
            if inferred is None:
                continue
            start_year, end_year = inferred

        for year in range(start_year, end_year + 1):
            tasks.append((reef_name, lat, lon, year))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reefs', nargs='*')
    parser.add_argument('--year-start', type=int, default=None)
    parser.add_argument('--year-end', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    coords = load_coordinates()
    tasks = build_tasks(coords, args.reefs, args.year_start, args.year_end)
    if not tasks:
        print("No download tasks found.")
        return

    print(f"Starting daily climate download for {len(tasks)} reef-years...")
    stats = {"OK": 0, "SKIP": 0, "FAILED": 0, "EMPTY": 0}
    if args.workers <= 1:
        for idx, (reef_name, lat, lon, year) in enumerate(tasks, 1):
            print(f"[{idx:4d}/{len(tasks)}] {reef_name:30s} {year} ... ", end="", flush=True)
            status, _ = download_daily(reef_name, lat, lon, year)
            stats[status] = stats.get(status, 0) + 1
            print(status)
            if idx % 25 == 0:
                print(f"Current stats: {stats}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(download_daily, reef_name, lat, lon, year): (reef_name, year)
                for reef_name, lat, lon, year in tasks
            }
            for idx, future in enumerate(as_completed(future_map), 1):
                reef_name, year = future_map[future]
                try:
                    status, _ = future.result()
                except Exception:
                    status = "FAILED"
                stats[status] = stats.get(status, 0) + 1
                print(f"[{idx:4d}/{len(tasks)}] {reef_name:30s} {year} ... {status}")
                if idx % 25 == 0:
                    print(f"Current stats: {stats}")

    print(f"Done. Final stats: {stats}")


if __name__ == "__main__":
    main()
