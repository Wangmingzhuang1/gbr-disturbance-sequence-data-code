# -*- coding: utf-8 -*-
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import config

matplotlib.use('Agg')


COORDS = os.path.join(config.DATA_DIR, 'sites_decimal.csv')
DIST_THRESHOLDS_KM = [100, 200, 500]


def haversine_matrix(lats, lons):
    radius = 6371.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def morans_i(z, weights):
    n = len(z)
    z_centered = z - z.mean()
    s0 = weights.sum()
    statistic = (n / s0) * (z_centered @ weights @ z_centered) / (z_centered @ z_centered)
    expected = -1.0 / (n - 1)

    s1 = 0.5 * np.sum((weights + weights.T) ** 2)
    s2 = np.sum((weights.sum(axis=1) + weights.sum(axis=0)) ** 2)
    n_sq = n * n
    a_term = n * ((n_sq - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
    b_term = (z_centered ** 4).sum() / (z_centered ** 2).sum() ** 2 * ((n_sq - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
    c_term = (n - 1) * (n - 2) * (n - 3) * s0 ** 2
    variance = max((a_term - b_term) / c_term - expected ** 2, 1e-12)
    z_score = (statistic - expected) / np.sqrt(variance)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return statistic, expected, z_score, p_value


def run_spatial_diagnostics():
    residuals = pd.read_csv(config.MAIN_RESIDUALS_PATH)
    coords = pd.read_csv(COORDS)[['reef_name', 'latitude', 'longitude']]

    reef_residuals = residuals.groupby('reef_name')['residual'].mean().reset_index()
    reef_residuals = reef_residuals.merge(coords, on='reef_name', how='left').dropna(subset=['latitude', 'longitude'])

    distances = haversine_matrix(reef_residuals['latitude'].to_numpy(), reef_residuals['longitude'].to_numpy())
    np.fill_diagonal(distances, np.inf)
    residual_values = reef_residuals['residual'].to_numpy()

    rows = []
    for threshold in DIST_THRESHOLDS_KM:
        weights_binary = (distances <= threshold).astype(float)
        row_sums = weights_binary.sum(axis=1, keepdims=True)
        isolated = row_sums.flatten() == 0
        row_sums[row_sums == 0] = 1
        weights_std = weights_binary / row_sums
        mask = ~isolated
        statistic, expected, z_score, p_value = morans_i(residual_values[mask], weights_std[np.ix_(mask, mask)])
        rows.append(
            {
                'threshold_km': threshold,
                'n_reefs': int(mask.sum()),
                'n_pairs': int(weights_binary[np.ix_(mask, mask)].sum() / 2),
                'morans_i': round(statistic, 4),
                'expected_i': round(expected, 4),
                'z_score': round(z_score, 3),
                'p_value': round(p_value, 4),
                'significant': bool(p_value < 0.05),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(config.SPATIAL_DIAGNOSTICS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ['#9f1d35' if sig else '#8f99a3' for sig in results['significant']]
    ax.bar(results['threshold_km'].astype(str) + ' km', results['morans_i'], color=colors, edgecolor='#333333', width=0.62)
    ax.axhline(0, color='#7b8085', linestyle='--', linewidth=1)
    ax.set_xlabel('Distance threshold')
    ax.set_ylabel("Moran's I")
    ax.set_title('Supplementary Figure S6. Spatial autocorrelation of main-model residuals', loc='left', fontsize=12, fontweight='bold')
    for idx, row in results.iterrows():
        ax.text(idx, row['morans_i'] + 0.01, f"p={row['p_value']:.3f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGS5_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(results.to_string(index=False))


if __name__ == '__main__':
    run_spatial_diagnostics()
