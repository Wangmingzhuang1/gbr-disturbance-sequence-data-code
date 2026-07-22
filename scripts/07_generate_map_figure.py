# -*- coding: utf-8 -*-
"""
Figure 1 & SI Figure 1: GBR study area maps showing monitored reefs, long-term thermal
recurrence frequency, storm frequency, and concurrent disturbance frequency.
"""
import os
import re
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def sanitize_name(name):
    value = re.sub(r"[^a-zA-Z0-9]", "_", str(name))
    return re.sub(r"_+", "_", value).strip("_")


def parse_dms_robust(dms_str):
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


def get_reef_data():
    sites_path = os.path.join(DATA_DIR, "sites_lon_lat.csv")
    sites = pd.read_csv(sites_path, header=None, names=["reef_name", "coords"], encoding="latin1")
    sites["reef_name"] = sites["reef_name"].apply(sanitize_name)
    sites[["lat", "lon"]] = sites["coords"].apply(lambda value: pd.Series(parse_dms_robust(value)))
    sites = sites.dropna(subset=["lat", "lon"])

    dist_path = os.path.join(DATA_DIR, "master_disturbance_matrix.csv")
    dist = pd.read_csv(dist_path)
    
    # 计算热浪、风暴和并发的频次
    dist["thermal_recurrence_frequency"] = dist["has_heatwave"].eq(1).astype(int)
    dist["storm_recurrence_frequency"] = dist["has_storm"].eq(1).astype(int)
    dist["concurrent_recurrence_frequency"] = (dist["has_heatwave"].eq(1) & dist["has_storm"].eq(1)).astype(int)
    
    counts = (
        dist.groupby("reef_name")[["thermal_recurrence_frequency", "storm_recurrence_frequency", "concurrent_recurrence_frequency"]]
        .sum()
        .reset_index()
    )
    counts["reef_name"] = counts["reef_name"].apply(sanitize_name)
    return pd.merge(sites, counts, on="reef_name", how="inner")


def draw_map_panel(ax, sites, color_col, cmap, cbar_label, panel_letter):
    ax.set_extent([142.5, 153.5, -24.5, -10.5], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#edf4f7", alpha=1, zorder=0)

    nrm_shp = os.path.join(
        DATA_DIR,
        "NRM_Terrestrial_and_Marine_Regions_GBR_GDA20",
        "NRM_Terrestrial_and_Marine_Regions_GBR_GDA20.shp",
    )
    if os.path.exists(nrm_shp):
        try:
            gpd.read_file(nrm_shp).boundary.plot(
                ax=ax,
                color="#949494",
                linestyle=":",
                linewidth=0.45,
                zorder=2,
                alpha=0.52,
                transform=ccrs.PlateCarree(),
            )
        except Exception as e:
            print(f"Warning: Failed to plot NRM boundary shp: {e}")

    reef_shp = os.path.join(DATA_DIR, "Great_Barrier_Reef_Features", "Great_Barrier_Reef_Features.shp")
    if os.path.exists(reef_shp):
        try:
            gpd.read_file(reef_shp).plot(
                ax=ax,
                facecolor="#74c7bb",
                edgecolor="none",
                alpha=0.38,
                zorder=1,
                transform=ccrs.PlateCarree(),
            )
        except Exception as e:
            print(f"Warning: Failed to plot Reef features shp: {e}")

    # 将陆地图层 zorder 置顶（设为 4），遮挡任何渗入陆地内部的 NRM 边界与大堡礁 shp 特征
    ax.add_feature(cfeature.LAND, facecolor="#eef2ea", edgecolor="#3d3d3d", linewidth=0.5, zorder=4)

    sc = ax.scatter(
        sites["lon"],
        sites["lat"],
        c=sites[color_col],
        cmap=cmap,
        s=12,
        edgecolors="#1f1f1f",
        linewidths=0.22,
        alpha=0.9,
        zorder=10,
        transform=ccrs.PlateCarree(),
    )

    ax.set_xticks(np.arange(144, 154, 4), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-24, -10, 4), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(length=2.5, width=0.8)

    ax.text(144.15, -11.05, "N", fontsize=7, fontweight="bold", ha="center", transform=ccrs.PlateCarree())
    ax.annotate(
        "",
        xy=(144.15, -11.35),
        xytext=(144.15, -12.12),
        arrowprops=dict(arrowstyle="->", lw=0.85, color="black"),
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, shrink=0.82)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)

    if panel_letter:
        ax.text(0.025, 0.975, str(panel_letter).upper(), transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax.set_aspect("equal", adjustable="box")


def plot_figure_1(sites):
    fig = plt.figure(figsize=(3.5, 4.8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    draw_map_panel(
        ax, 
        sites, 
        "thermal_recurrence_frequency", 
        "inferno_r",
        "Thermal recurrence frequency (1985-2025)", 
        ""
    )
    fig.savefig(os.path.join(OUTPUT_DIR, "figure_01.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "figure_01.jpg"), format="jpg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Figure 1 generated.")


def plot_si_figure_1(sites):
    fig = plt.figure(figsize=(7.0, 4.8))
    
    # Left Panel: Storm Recurrence
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    draw_map_panel(
        ax1, 
        sites, 
        "storm_recurrence_frequency", 
        "viridis", 
        "Storm recurrence frequency (1985-2025)", 
        "A"
    )
    
    # Right Panel: Concurrent Recurrence
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    draw_map_panel(
        ax2, 
        sites, 
        "concurrent_recurrence_frequency", 
        "magma", 
        "Concurrent recurrence frequency (1985-2025)", 
        "B"
    )
    
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUTPUT_DIR, "si_figure_01.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "si_figure_01.jpg"), format="jpg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("SI Figure 1 generated.")


def main():
    print("Loading data for maps...")
    sites = get_reef_data()
    plot_figure_1(sites)
    plot_si_figure_1(sites)


if __name__ == "__main__":
    main()
