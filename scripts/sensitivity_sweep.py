# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sequence_analysis_core as sac
import config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_PATH = config.MASTER_MATRIX_PATH

def run_dual_track_sweep():
    df_master = pd.read_csv(MASTER_PATH)
    
    # Using the 2-5 year window as agreed
    scenarios = [
        {'name': 'Strict Independent (Disjoint)', 'mode': 'strict'},
        {'name': 'Rolling Origin (Chains)',      'mode': 'rolling'},
    ]
    
    results = []
    
    for scen in scenarios:
        extracted_all = []
        for reef_name in df_master['reef_name'].unique():
            df_reef = df_master[df_master['reef_name'] == reef_name].sort_values('year')
            res = sac.extract_reef_sequences(df_reef, mode=scen['mode'])
            if not res.empty:
                extracted_all.append(res)
        
        if extracted_all:
            final_df = pd.concat(extracted_all)
            n_seq = len(final_df)
            n_reef = final_df['reef_name'].nunique()
            sh = final_df[final_df['seq_category_main'] == 'S_to_H']
            hs = final_df[final_df['seq_category_main'] == 'H_to_S']
            sh_mean = sh['rel_loss'].mean()
            hs_mean = hs['rel_loss'].mean()
            
            results.append({
                'Design_Mode': scen['name'],
                'N_Sequences': n_seq,
                'N_Reefs': n_reef,
                'S_to_H_N': len(sh),
                'S_to_H_Mean': f"{sh_mean:.2%}",
                'H_to_S_N': len(hs),
                'H_to_S_Mean': f"{hs_mean:.2%}",
                'Asymmetry': f"{(sh_mean - hs_mean):.2%}"
            })
            
    summary = pd.DataFrame(results)
    print("\n--- DUAL-TRACK DESIGN COMPARISON (2-5yr Window) ---")
    print(summary.to_string(index=False))
    summary.to_csv(config.DUAL_TRACK_COMPARISON_PATH, index=False)

if __name__ == '__main__':
    run_dual_track_sweep()
