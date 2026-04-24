# -*- coding: utf-8 -*-
"""
Auditing Sensitivity of Concurrent vs S->H classification.
Checks if buffering the DHW window (ds to dp) significantly changes event counts.
"""
import pandas as pd
import numpy as np
import os

# Base paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE, 'output', 'topic_b_features.csv')

def run_sensitivity():
    # Note: topic_b_features.csv might not have ds/dp raw months if they weren't saved.
    # I'll check the 'extracted_sequences' file or rerun the logic on merged records.
    merged_file = os.path.join(BASE, 'output', 'eco_response_master_matrix_merged.csv')
    if not os.path.exists(merged_file):
        print("Merged matrix not found. Run 03_merge_eco_dist.py first.")
        return

    df = pd.read_csv(merged_file)
    
    # Filter for cases with both storm and heatwave in same AIMS year
    s_h_year = df[(df['has_storm'] == 1) & (df['has_heatwave'] == 1)].copy()
    
    print(f"Total Same-Year (S+H) pairs: {len(s_h_year)}")
    
    results = []
    # Buffer in months (approximate)
    for buffer in [0, 0.5, 1.0, 2.0]:
        counts = {'S->H': 0, 'Concurrent': 0, 'H->S': 0}
        for _, row in s_h_year.iterrows():
            sm = row['storm_aims_month']
            ds = row['dhw_start_aims_month']
            dp = row['dhw_peak_aims_month']
            
            if pd.isna(sm) or pd.isna(ds) or pd.isna(dp): continue
            
            # Adjusted window
            ds_adj = ds - buffer
            dp_adj = dp + buffer
            
            if sm < ds_adj:
                counts['S->H'] += 1
            elif ds_adj <= sm <= dp_adj:
                counts['Concurrent'] += 1
            else:
                counts['H->S'] += 1
        
        results.append({'buffer_months': buffer, **counts})
    
    print("\nSensitivity Table (Impact of increasing Concurrent window):")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Check if the 'S_to_H' sequences (cross-year) are affected
    # (They aren't, by definition, as they happen in different years).

if __name__ == "__main__":
    run_sensitivity()
