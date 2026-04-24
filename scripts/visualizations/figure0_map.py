# -*- coding: utf-8 -*-
"""
Figure 0: GBR study area map and disturbance timeline.
Adopting aesthetics and layout from reference script.
"""
import os
import re
import sys
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modeling_core import filter_downstream_analysis_sample

from style_config import (
    FS_ANNOT,
    FS_LABEL,
    FS_LEGEND,
    FS_LEGEND_TITLE,
    FS_PANEL,
    FS_TICK,
    SEQ_COLORS,
    add_panel_label,
    apply_pub_style,
    save_publication_figure,
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, 'data')


def sanitize_name(name):
    s = re.sub(r'[^a-zA-Z0-9]', '_', str(name))
    return re.sub(r'_+', '_', s).strip('_')


def parse_dms_robust(dms_str):
    if pd.isna(dms_str) or not dms_str:
        return None, None
    s = str(dms_str).upper().replace("'", "'").replace('"', '"').replace('F', 'E')

    def parse_part(part_str):
        nums = re.findall(r"(\d+(?:\.\d+)?)", part_str)
        direction = re.search(r"([NSEW])", part_str)
        if not direction:
            return None
        direction = direction.group(1)
        vals = [0.0, 0.0, 0.0]
        for i in range(min(len(nums), 3)):
            vals[i] = float(nums[i])
        deg = vals[0] + vals[1] / 60 + vals[2] / 3600
        return -deg if direction in ['S', 'W'] else deg

    match = re.search(r"(.*[SN])\s*(.*[EW])", s)
    if match:
        return parse_part(match.group(1)), parse_part(match.group(2))
    return None, None


def _site_table():
    sites = pd.read_csv(
        os.path.join(DATA_DIR, 'sites_lon_lat.csv'),
        header=None,
        names=['reef_name', 'coords'],
        encoding='latin1',
    )
    sites['reef_name'] = sites['reef_name'].apply(sanitize_name)
    sites[['lat', 'lon']] = sites['coords'].apply(lambda x: pd.Series(parse_dms_robust(x)))
    sites = sites.dropna(subset=['lat', 'lon'])

    master_path = config.MASTER_MATRIX_PATH
    if os.path.exists(master_path):
        master_reefs = set(pd.read_csv(master_path, usecols=['reef_name'])['reef_name'].dropna().unique())
        sites = sites[sites['reef_name'].isin(master_reefs)].copy()

    analyzed_reefs = set()
    feat_path = config.FINAL_FEATURES_PATH
    if os.path.exists(feat_path):
        analyzed_reefs = set(filter_downstream_analysis_sample(pd.read_csv(feat_path))['reef_name'].dropna().unique())

    seq_path = config.EXTRACTED_SEQS_PATH
    reef_best_type = {}
    if os.path.exists(seq_path):
        seq_df = filter_downstream_analysis_sample(pd.read_csv(seq_path))
        # Match colors and priorities
        priority_map = {name: idx for idx, name in enumerate([
            'S -> H', 'S -> S', 'Isolated Storm', 'Isolated Heatwave', 'H -> H'
        ])}
        for reef, group in seq_df.groupby('reef_name'):
            labels = group['sequence_type'].dropna().unique().tolist()
            if labels:
                reef_best_type[reef] = sorted(labels, key=lambda x: priority_map.get(x, 99))[0]

    sites['display_cat'] = sites['reef_name'].apply(
        lambda reef: reef_best_type.get(reef, 'Monitoring site (No major disturbance)')
        if reef in analyzed_reefs else 'Excluded due to data gaps'
    )
    return sites


def _timeline_table():
    master_path = config.MASTER_MATRIX_PATH
    if not os.path.exists(master_path):
        return None

    df = pd.read_csv(master_path, usecols=['year', 'has_storm', 'has_heatwave']).copy()
    for column in ['has_storm', 'has_heatwave']:
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)

    timeline = (
        df.groupby('year', as_index=True)[['has_storm', 'has_heatwave']]
        .sum()
        .rename(columns={'has_storm': 'Storm', 'has_heatwave': 'Heatwave'})
    )
    return timeline[(timeline['Storm'] > 0) | (timeline['Heatwave'] > 0)]


def plot_fig0():
    apply_pub_style()
    sites = _site_table()
    dist = _timeline_table()

    # Premium Color Palette (Synced with global style)
    type_colors = {
        'S -> H': SEQ_COLORS['S_to_H'],
        'S -> S': SEQ_COLORS['S_to_S'],
        'Isolated Storm': SEQ_COLORS['Isolated_Storm'],
        'Isolated Heatwave': SEQ_COLORS['Isolated_Heatwave'],
        'H -> H': SEQ_COLORS['H_to_H'],
        'Monitoring site (No major disturbance)': '#BDC3C7',
        'Excluded due to data gaps': '#FFFFFF',
    }
    plot_sequence = [
        'Excluded due to data gaps',
        'Monitoring site (No major disturbance)',
        'H -> H',
        'Isolated Heatwave',
        'Isolated Storm',
        'S -> S',
        'S -> H',
    ]

    # Vertical layout
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.15)
    
    # Panel A: Map
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    extent = [142, 154, -25, -10]
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax_map.add_feature(cfeature.OCEAN, facecolor='#eef6fa', alpha=1, zorder=0)
    ax_map.add_feature(cfeature.LAND, facecolor='#faf0d9', edgecolor='#444444', linewidth=0.7, zorder=5)

    nrm_shp = os.path.join(DATA_DIR, 'NRM_Terrestrial_and_Marine_Regions_GBR_GDA20', 'NRM_Terrestrial_and_Marine_Regions_GBR_GDA20.shp')
    if os.path.exists(nrm_shp):
        nrm_gdf = gpd.read_file(nrm_shp)
        nrm_gdf.boundary.plot(ax=ax_map, color='#7f8c8d', linestyle=':', linewidth=0.8, zorder=1, alpha=0.5, transform=ccrs.PlateCarree())
        for _, row in nrm_gdf.iterrows():
            if hasattr(row.geometry, 'centroid'):
                cent = row.geometry.centroid
                ax_map.text(cent.x, cent.y, row.get('REGION',''), color='#7f8c8d', fontsize=FS_TICK-1, 
                        style='italic', ha='center', va='center', alpha=0.4, zorder=1, transform=ccrs.PlateCarree())

    reef_shp = os.path.join(DATA_DIR, 'Great_Barrier_Reef_Features', 'Great_Barrier_Reef_Features.shp')
    if os.path.exists(reef_shp):
        reefs_gdf = gpd.read_file(reef_shp)
        reefs_gdf.plot(ax=ax_map, facecolor='#95a5a6', edgecolor='none', alpha=0.6, zorder=2, transform=ccrs.PlateCarree())

    for cat in plot_sequence:
        sub = sites[sites['display_cat'] == cat]
        if sub.empty:
            continue
        kwargs = dict(
            c=type_colors[cat],
            s=35,
            edgecolors='k',
            linewidths=0.2,
            alpha=1,
            zorder=10 + plot_sequence.index(cat),
            transform=ccrs.PlateCarree()
        )
        label = 'Excluded sites' if cat == 'Excluded due to data gaps' else cat
        ax_map.scatter(sub['lon'], sub['lat'], label=label, **kwargs)

    ax_map.set_xticks(np.arange(142, 156, 4), crs=ccrs.PlateCarree())
    ax_map.set_yticks(np.arange(-25, -9, 5), crs=ccrs.PlateCarree())
    ax_map.xaxis.set_major_formatter(LongitudeFormatter())
    ax_map.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax_map.text(143, -11.0, 'N', fontsize=12, fontweight='bold', ha='center', transform=ccrs.PlateCarree())
    ax_map.annotate('', xy=(143, -11.0), xytext=(143, -12.0), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), transform=ccrs.PlateCarree())

    add_panel_label(ax_map, 'a', x=0.03, y=0.97)
    
    ax_map.legend(
        loc='upper right',
        title='Site classification',
        fontsize=FS_LEGEND,
        title_fontsize=FS_LEGEND_TITLE,
        frameon=True,
        framealpha=0.9
    )
    ax_map.set_aspect('equal', adjustable='box')

    # Panel B: Timeline
    ax_time = fig.add_subplot(gs[1, 0])
    map_pos = ax_map.get_position()
    time_pos = ax_time.get_position()
    ax_time.set_position([map_pos.x0, time_pos.y0, map_pos.width, time_pos.height])
    for side in ['left', 'right', 'top', 'bottom']:
        ax_time.spines[side].set_visible(True)
        ax_time.spines[side].set_linewidth(0.8)
        ax_time.spines[side].set_color('black')
    
    if dist is not None and not dist.empty:
        t_order = ['Storm', 'Heatwave']
        t_colors = {'Storm': SEQ_COLORS['S_to_S'], 'Heatwave': SEQ_COLORS['Isolated_Heatwave']}
        years = dist.index.to_numpy()
        width = 0.5
        offsets = {'Storm': -width / 2, 'Heatwave': width / 2}
        for key in t_order:
            if key not in dist.columns:
                continue
            values = dist[key].to_numpy()
            ax_time.bar(
                years + offsets[key],
                values,
                width=width,
                color=t_colors[key],
                edgecolor='black',
                linewidth=0.1,
                label=key,
            )

        ax_time.set_xlabel('Year', fontsize=FS_LABEL, fontweight='bold')
        ax_time.set_ylabel('Affected Reef Count', fontsize=FS_LABEL, fontweight='bold')
        ax_time.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax_time.set_xticks(np.arange(1985, 2026, 5))
        ax_time.tick_params(axis='both', labelsize=FS_TICK)
        ax_time.legend(loc='upper left', fontsize=FS_LEGEND, frameon=False, ncol=2)
        add_panel_label(ax_time, 'b', x=0.03, y=0.92)
        
        ax_time.annotate('Annual monitored reefs exposed to storm or heatwave in the full master matrix',
                         xy=(0.01, -0.25), xycoords='axes fraction', fontsize=FS_ANNOT,
                         color='#555555', style='italic')
    else:
        ax_time.text(0.5, 0.5, 'Disturbance timeline unavailable', transform=ax_time.transAxes, ha='center', va='center')

    save_publication_figure(fig, config.FIG0_PATH)
    plt.close()
    print(f'Figure 0 saved (Vertical A4): {config.FIG0_PATH}')


if __name__ == '__main__':
    plot_fig0()
