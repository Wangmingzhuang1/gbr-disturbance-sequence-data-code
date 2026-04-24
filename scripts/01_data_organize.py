# -*- coding: utf-8 -*-
import glob
import os
import re
import shutil

import pandas as pd


def sanitize_name(name):
    if not isinstance(name, str) or pd.isna(name):
        return "Unknown"
    s = re.sub(r'[^a-zA-Z0-9]', '_', str(name))
    return re.sub(r'_+', '_', s).strip('_')


def normalize_coord_text(text):
    if not isinstance(text, str) or pd.isna(text):
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
        '""': '"',
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def update_sites_file(base_dir):
    sites_path = os.path.join(os.path.dirname(base_dir), 'sites_lon_lat.csv')
    if not os.path.exists(sites_path):
        print(f"sites_lon_lat.csv not found at {sites_path}, skipping")
        return

    sites_df = pd.read_csv(sites_path, header=None, names=['reef_name', 'coords'], encoding='latin1')
    sites_df['reef_name'] = sites_df['reef_name'].apply(sanitize_name)
    sites_df['coords'] = sites_df['coords'].apply(normalize_coord_text)
    try:
        sites_df.to_csv(sites_path, index=False, header=False, encoding='latin1')
        print(f"Updated reef naming and coordinate text in {sites_path}")
    except PermissionError:
        print(f"Warning: could not rewrite {sites_path} because the file is locked. "
              "The pipeline will still work if reader scripts sanitize reef names on load.")


def organize_aims_data(base_dir):
    print(f"Scanning directory: {base_dir}")
    all_csvs = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True)

    for filepath in all_csvs:
        try:
            df_head = pd.read_csv(filepath, nrows=1)
            if df_head.empty:
                continue

            row = df_head.iloc[0]

            if not all(col in df_head.columns for col in ['domain_name', 'variable']):
                print(f"Skipping {filepath}: missing required columns")
                continue

            domain_name = row['domain_name']

            if 'data_type' in df_head.columns:
                data_type = row['data_type']
            elif 'tows' in df_head.columns or 'cots' in str(row['variable']).lower():
                data_type = 'manta'
            else:
                data_type = 'unknown'

            variable = row['variable']

            sanitized_reef = sanitize_name(domain_name)
            target_dir = os.path.join(base_dir, sanitized_reef)
            os.makedirs(target_dir, exist_ok=True)

            if data_type.lower() == 'manta' and 'cots' in str(variable).lower():
                new_filename = "manta_cots_density.csv"
            elif data_type.lower() == 'manta' and str(variable).upper() == 'HC':
                new_filename = "manta_HC_visual_estimate.csv"
            else:
                new_filename = f"{sanitize_name(data_type)}_{sanitize_name(variable)}.csv"

            target_path = os.path.join(target_dir, new_filename)

            if os.path.abspath(filepath) != os.path.abspath(target_path):
                if os.path.exists(target_path):
                    print(f"Skipping {filepath}: target already exists at {target_path}")
                    continue
                shutil.move(filepath, target_path)
                print(f"Moved: {os.path.basename(filepath)} -> {sanitized_reef}/{new_filename}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    # Remove empty directories left behind after moves
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")

    update_sites_file(base_dir)


if __name__ == "__main__":
    organize_aims_data('data/reef_raw')
