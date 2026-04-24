#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full pipeline audit: verify data integrity, baseline definition consistency,
and manuscript-critical sample counts."""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'scripts'))
import config
from modeling_core import get_rare_categories

print("=" * 70)
print("PIPELINE AUDIT")
print("=" * 70)

# 1. Config vs Manuscript
print("\n[1] CONFIG PARAMETERS")
print(f"  BASELINE_WINDOW       = {list(config.BASELINE_WINDOW)}  (search window)")
print(f"  NADIR_WINDOW          = {list(config.NADIR_WINDOW)}")
print(f"  MIN_BASELINE_SURVEYS  = {config.MIN_BASELINE_SURVEYS}")
print(f"  MIN_BASELINE_HC       = {config.MIN_BASELINE_HC}")
print(f"  LOOKAHEAD_WINDOW      = {config.LOOKAHEAD_WINDOW}")
print(f"  RECOVERY_LOOKAHEAD    = {config.RECOVERY_LOOKAHEAD}")
print(f"  EXTRACTION_MODE       = {config.EXTRACTION_MODE}")

# 2. Master Matrix Integrity
print("\n[2] MASTER MATRIX INTEGRITY")
phys = pd.read_csv(config.MASTER_MATRIX_PHYSICAL_PATH)
merged = pd.read_csv(config.MASTER_MATRIX_PATH)
phys_overlap = phys[phys['reef_name'].isin(merged['reef_name'])].copy()
print(f"  Physical matrix:  {len(phys)} rows, {phys['reef_name'].nunique()} reefs")
print(f"  Merged matrix:    {len(merged)} rows, {merged['reef_name'].nunique()} reefs")
print(f"  Reefs without merged ecology window: {phys['reef_name'].nunique() - merged['reef_name'].nunique()}")
print(f"  Rows removed outside HC survey window: {len(phys_overlap) - len(merged)}")
print(f"  Year range:       {merged['year'].min()} - {merged['year'].max()}")

hc_coverage = merged['HC_cover'].notna().sum()
print(f"  HC_cover non-null: {hc_coverage} / {len(merged)} ({100 * hc_coverage / len(merged):.1f}%)")

dt_counts = merged['Disturbance_Type'].value_counts()
print(f"  Disturbance types: {len(dt_counts)}")
for dt, n in dt_counts.head(10).items():
    print(f"    {dt}: {n}")

# 3. Ecological Data Completeness
print("\n[3] ECOLOGICAL DATA COMPLETENESS")
eco_cols = ['HC_cover', 'ALGAE_cover', 'MACROALGAE_cover', 'Juveniles', 'Fish_Herbivores', 'COTS_density']
for col in eco_cols:
    if col in merged.columns:
        n = merged[col].notna().sum()
        reefs = merged.loc[merged[col].notna(), 'reef_name'].nunique()
        print(f"  {col:25s}: {n:5d} rows, {reefs:3d} reefs")
    else:
        print(f"  {col:25s}: MISSING COLUMN")

# 4. Extracted Sequences
print("\n[4] EXTRACTED SEQUENCES")
seq = pd.read_csv(config.EXTRACTED_SEQS_PATH)
print(f"  Total sequences: {len(seq)}")
print(f"  Total reefs:     {seq['reef_name'].nunique()}")
print("  Categories:")
for cat, n in seq['seq_category_main'].value_counts().items():
    print(f"    {cat:22s}: {n}")

# 5. Baseline Validity Checks
print("\n[5] BASELINE VALIDITY")
print(f"  Min baseline HC:     {seq['baseline_hc'].min():.3f} (should be >= {config.MIN_BASELINE_HC})")
print(f"  Min baseline surveys:{seq['baseline_survey_n'].min()} (should be >= {config.MIN_BASELINE_SURVEYS})")
seq['bl_span'] = seq['start_year'] - seq['baseline_start_year']
seq['bl_recent_gap'] = seq['start_year'] - seq['baseline_end_year']
print(f"  Config search window: {min(config.BASELINE_WINDOW)}-{max(config.BASELINE_WINDOW)} years pre-disturbance")
print(f"  Actual retained baseline span: {seq['bl_span'].min()}-{seq['bl_span'].max()} years")
print(f"  Most recent baseline survey gap: {seq['bl_recent_gap'].min()}-{seq['bl_recent_gap'].max()} years")

overlap_issues = 0
for _, row in seq.iterrows():
    bl_years = set(range(int(row['baseline_start_year']), int(row['baseline_end_year']) + 1))
    events_in_bl = merged[
        (merged['reef_name'] == row['reef_name']) &
        (merged['year'].isin(bl_years)) &
        ((merged['has_storm'] == 1) | (merged['has_heatwave'] == 1))
    ]
    if not events_in_bl.empty:
        overlap_issues += 1
print(f"  Baselines overlapping with events: {overlap_issues} / {len(seq)}")

# 6. Nadir Validity
print("\n[6] NADIR VALIDITY")
seq['nadir_offset'] = seq['nadir_year'] - seq.apply(
    lambda r: int(r['target_year']) if pd.notna(r['target_year']) else int(r['start_year']),
    axis=1,
)
print(f"  Nadir offset range: {seq['nadir_offset'].min()}-{seq['nadir_offset'].max()} (should be {min(config.NADIR_WINDOW)}-{max(config.NADIR_WINDOW)})")
out_of_range = seq[(seq['nadir_offset'] < min(config.NADIR_WINDOW)) | (seq['nadir_offset'] > max(config.NADIR_WINDOW))]
print(f"  Nadir out of window: {len(out_of_range)} / {len(seq)}")

# 7. rel_loss Consistency
print("\n[7] REL_LOSS CONSISTENCY")
seq['computed_rel_loss'] = ((seq['baseline_hc'] - seq['nadir_hc']) / seq['baseline_hc']).clip(-1, 1)
diff = (seq['rel_loss'] - seq['computed_rel_loss']).abs()
print(f"  Max discrepancy: {diff.max():.10f}")
print(f"  Mean discrepancy: {diff.mean():.10f}")

# 8. Attrition Audit
print("\n[8] ATTRITION AUDIT")
attrition = pd.read_csv(config.ATTRITION_AUDIT_PATH)
print(attrition.to_string(index=False))

# 9. Algae Prioritization Check
print("\n[9] ALGAE PRIORITIZATION")
print(f"  Sequences with baseline algae data: {seq['baseline_algae'].notna().sum()} / {len(seq)}")
print(f"  MACROALGAE_cover in merged matrix: {'MACROALGAE_cover' in merged.columns}")
print(f"  ALGAE_cover in merged matrix: {'ALGAE_cover' in merged.columns}")

# 10. Feature Engineering Check
print("\n[10] FEATURE ENGINEERING")
feat = pd.read_csv(config.FINAL_FEATURES_PATH)
print(f"  Features: {len(feat)} rows, {feat['reef_name'].nunique()} reefs")
print(f"  juv_loss non-null: {feat['juv_loss'].notna().sum()}")
print(f"  juv_change_nadir non-null: {feat['juv_change_nadir'].notna().sum()}")
print(f"  Centered columns present: {[c for c in feat.columns if c.endswith('_c')]}")

# 11. Juvenile direction check for S_to_H
print("\n[11] S_TO_H JUVENILE DIRECTION CHECK")
sth = feat[feat['seq_category_main'] == 'S_to_H']
sth_juv = sth[
    ['reef_name', 'start_year', 'baseline_juv', 'nadir_juv', 'juv_loss', 'juv_change_nadir']
].dropna(subset=['juv_change_nadir'])
for _, r in sth_juv.iterrows():
    direction = "DECLINE" if r['juv_loss'] > 0 else "NO DECLINE / NADIR HIGHER"
    print(
        f"  {r['reef_name']} (yr={int(r['start_year'])}): "
        f"baseline_juv={r['baseline_juv']:.2f}, nadir_juv={r['nadir_juv']:.2f}, "
        f"juv_loss={r['juv_loss']:.3f}, juv_change_nadir={r['juv_change_nadir']:.3f} -> {direction}"
    )

# 12. Regression sample check
print("\n[12] REGRESSION SAMPLE CHECK")
rare_categories = get_rare_categories(feat)
feat_model = feat[~feat['seq_category_main'].isin(rare_categories)] if rare_categories else feat.copy()
print(f"  Rare categories excluded from main model: {rare_categories if rare_categories else 'None'}")
print(f"  Main model n (excl rare classes): {len(feat_model)}")
for category in rare_categories:
    print(f"  {category} count: {(feat['seq_category_main'] == category).sum()}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
