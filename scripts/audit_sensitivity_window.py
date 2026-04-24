# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE, 'output', 'eco_response_master_matrix_merged.csv')

def test_lookahead_sensitivity(window_years):
    master = pd.read_csv(INPUT_FILE).sort_values(['reef_name', 'year'])
    sequences = []
    
    for reef, group in master.groupby('reef_name'):
        group = group.sort_values('year')
        consumed = set()
        for i in range(len(group)):
            row = group.iloc[i]
            yr = row['year']
            if not (row['has_storm'] == 1 or row['has_heatwave'] == 1): continue
            
            # Simplified S->H check
            if row['Disturbance_Type'] == 'Isolated Storm':
                lookahead = group[(group['year'] > yr) & (group['year'] <= yr + window_years) & (~group['year'].isin(consumed))]
                h_events = lookahead[lookahead['has_heatwave'] == 1]
                if not h_events.empty:
                    sequences.append({'type': 'S_to_H', 'gap': h_events.iloc[0]['year'] - yr})
                    consumed.add(h_events.iloc[0]['year'])
                    
    return len([s for s in sequences if s['type'] == 'S_to_H'])

if __name__ == "__main__":
    print("Testing Sensitivity of S->H Sequence Counts to Memory Window...")
    results = {}
    for w in [2, 3, 4, 5, 6]:
        count = test_lookahead_sensitivity(w)
        results[w] = count
        print(f"  Window: {w} years -> Found {count} S->H sequences")
    
    # Check if growth is linear (bad) or saturates (good, implies natural threshold)
    # If it saturates at 4-5 years, it confirms the 'Ecological Memory' limit.
