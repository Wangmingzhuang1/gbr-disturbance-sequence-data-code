# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import config
from sequence_analysis_core import SEQ_ORDER
from modeling_core import filter_downstream_analysis_sample

# --- PARAMETERS FROM CONFIG ---
INPUT = config.EXTRACTED_SEQS_PATH
N_BOOT = config.BOOTSTRAP_REPS
N_PERM = config.PERMUTATION_REPS
SEED = config.RANDOM_SEED
ANALYSIS_SEQ_ORDER = [seq for seq in SEQ_ORDER if seq not in config.DOWNSTREAM_EXCLUDED_SEQS]

def cluster_bootstrap(df, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    reefs = df['reef_name'].unique()
    boot_rows = {category: [] for category in ANALYSIS_SEQ_ORDER}
    for _ in range(n_boot):
        sampled = rng.choice(reefs, size=len(reefs), replace=True)
        sample_df = pd.concat([df[df['reef_name'] == reef] for reef in sampled], ignore_index=True)
        for category in ANALYSIS_SEQ_ORDER:
            values = sample_df.loc[sample_df['seq_category_main'] == category, 'rel_loss'].dropna()
            boot_rows[category].append(values.mean() if len(values) else np.nan)
    return boot_rows

def summarize_bootstrap(df, boot_rows):
    rows = []
    # Reference for Effect Size calculation
    reference_rows = df.loc[df['seq_category_main'] == 'Isolated_Heatwave', 'rel_loss'].dropna()
    for category in ANALYSIS_SEQ_ORDER:
        obs = df.loc[df['seq_category_main'] == category, 'rel_loss'].dropna()
        values = np.array([val for val in boot_rows[category] if not np.isnan(val)])
        pooled_sd = np.sqrt((obs.std() ** 2 + reference_rows.std() ** 2) / 2) if len(obs) > 1 and len(reference_rows) > 1 else np.nan
        rows.append({
            'seq_category': category,
            'n': len(obs),
            'mean_rel_loss': obs.mean(),
            'ci_lo': np.percentile(values, 2.5) if len(values) >= 50 else np.nan,
            'ci_hi': np.percentile(values, 97.5) if len(values) >= 50 else np.nan,
            'cohens_d_vs_IsoHW': (obs.mean() - reference_rows.mean()) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan,
        })
    return pd.DataFrame(rows).sort_values('mean_rel_loss', ascending=False)

def permutation_asymmetry(df, n_perm=N_PERM, seed=SEED + 81):
    return pd.DataFrame([{
        'comparison': 'S_to_H_vs_H_to_S',
        'available': False,
        'reason': 'H_to_S is a fixed downstream-excluded rare class',
    }])

def run_bootstrap():
    if not os.path.exists(INPUT):
        print(f"Error: {INPUT} not found.")
        return
        
    raw_df = pd.read_csv(INPUT)
    df = filter_downstream_analysis_sample(raw_df)
    print(f"Running Bootstrap (Reps={N_BOOT}) on {len(df)} downstream-analysis sequences...")
    
    boot_rows = cluster_bootstrap(df)
    hierarchy = summarize_bootstrap(df, boot_rows)
    hierarchy.to_csv(config.BOOTSTRAP_HIERARCHY_PATH, index=False)

    asym = permutation_asymmetry(df)
    asym.to_csv(config.PERMUTATION_ASYMMETRY_PATH, index=False)

    print("\n--- FINAL BOOTSTRAP HIERARCHY ---")
    print(hierarchy.to_string(index=False))
    print("\n--- PERMUTATION ASYMMETRY (S->H vs H->S) ---")
    print(asym.to_string(index=False))

if __name__ == '__main__':
    run_bootstrap()
