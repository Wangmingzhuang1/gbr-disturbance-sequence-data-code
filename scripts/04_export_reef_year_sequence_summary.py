# -*- coding: utf-8 -*-
import os

import pandas as pd

import config


ROLE_START = 'start'
ROLE_TARGET = 'target'
ROLE_START_AND_TARGET = 'start_and_target'
ROLE_NONE = 'none'

SINGLE_ROW_SEQUENCE_CODES = {'Concurrent'}


def _sequence_id(row):
    return f"{row['reef_name']}|{int(row['start_year'])}|{row['sequence_type_primary']}"


def _is_single_row_sequence(row):
    sequence_code = str(row['sequence_type_primary'])
    gap_years = row.get('gap_years', pd.NA)
    return (
        sequence_code in SINGLE_ROW_SEQUENCE_CODES
        or str(sequence_code).startswith('Isolated_')
        or (pd.notna(gap_years) and float(gap_years) == 0.0)
    )


def _build_membership_rows(sequences_df):
    membership_rows = []

    for _, row in sequences_df.iterrows():
        sequence_code = str(row['sequence_type_primary'])
        start_year = int(row['start_year'])
        target_year = row['target_year']
        has_target = pd.notna(target_year)
        if has_target:
            target_year = int(target_year)

        common = {
            'reef_name': row['reef_name'],
            'sequence_id': _sequence_id(row),
            'in_sequence': 1,
            'sequence_type_primary': sequence_code,
            'sequence_type': row['sequence_type'],
            'sequence_start_year': start_year,
            'sequence_target_year': target_year if has_target else pd.NA,
            'sequence_end_year': int(row['end_year']) if pd.notna(row['end_year']) else pd.NA,
            'sequence_gap_years': row['gap_years'] if pd.notna(row['gap_years']) else pd.NA,
        }

        if _is_single_row_sequence(row) or not has_target:
            membership_rows.append({
                **common,
                'year': start_year,
                'sequence_role': ROLE_START_AND_TARGET,
                'paired_year': pd.NA,
            })
            continue

        membership_rows.append({
            **common,
            'year': start_year,
            'sequence_role': ROLE_START,
            'paired_year': target_year,
        })
        membership_rows.append({
            **common,
            'year': target_year,
            'sequence_role': ROLE_TARGET,
            'paired_year': start_year,
        })

    return pd.DataFrame(membership_rows)


def _validate_memberships(membership_df):
    dup_mask = membership_df.duplicated(subset=['reef_name', 'year'], keep=False)
    if not dup_mask.any():
        return

    conflict_df = (membership_df.loc[dup_mask, ['reef_name', 'year', 'sequence_id', 'sequence_type_primary']]
        .sort_values(['reef_name', 'year', 'sequence_id'])
        .copy())
    conflict_path = config.REEF_YEAR_SEQUENCE_CONFLICTS_PATH
    conflict_df.to_csv(conflict_path, index=False)
    raise ValueError(
        f"Detected conflicting sequence assignments for the same reef-year. "
        f"Conflict audit written to {conflict_path}"
    )


def _print_summary(master_df, summary_df):
    in_sequence_count = int((summary_df['in_sequence'] == 1).sum())
    print(f"Master rows: {len(master_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Rows with in_sequence=1: {in_sequence_count}")

    assigned = summary_df[summary_df['in_sequence'] == 1].copy()
    if assigned.empty:
        print("No retained sequence memberships found.")
        return

    counts = (assigned.groupby(['sequence_type_primary', 'sequence_role'])
        .size()
        .unstack(fill_value=0))
    for role in [ROLE_START, ROLE_TARGET, ROLE_START_AND_TARGET]:
        if role not in counts.columns:
            counts[role] = 0
    counts = counts[[ROLE_START, ROLE_TARGET, ROLE_START_AND_TARGET]].sort_index()
    print("\nSequence role counts by sequence_type_primary:")
    print(counts.to_string())


def export_reef_year_sequence_summary():
    if not os.path.exists(config.MASTER_MATRIX_PATH):
        raise FileNotFoundError(
            f"{config.MASTER_MATRIX_PATH} not found. Run 03_merge_eco_dist.py first."
        )
    if not os.path.exists(config.EXTRACTED_SEQS_PATH):
        raise FileNotFoundError(
            f"{config.EXTRACTED_SEQS_PATH} not found. Run 05_analyze_succession.py first."
        )

    master_df = pd.read_csv(config.MASTER_MATRIX_PATH)
    sequences_df = pd.read_csv(config.EXTRACTED_SEQS_PATH)

    membership_df = _build_membership_rows(sequences_df)
    if membership_df.empty:
        membership_df = pd.DataFrame(columns=[
            'reef_name', 'year', 'sequence_id', 'in_sequence',
            'sequence_type_primary', 'sequence_type', 'sequence_role',
            'paired_year', 'sequence_start_year', 'sequence_target_year',
            'sequence_end_year', 'sequence_gap_years',
        ])
    _validate_memberships(membership_df)

    summary_df = master_df.merge(membership_df, on=['reef_name', 'year'], how='left')
    summary_df['in_sequence'] = summary_df['in_sequence'].fillna(0).astype(int)
    summary_df['sequence_role'] = summary_df['sequence_role'].fillna(ROLE_NONE)

    nullable_int_cols = ['paired_year', 'sequence_start_year', 'sequence_target_year', 'sequence_end_year']
    for col in nullable_int_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.array(summary_df[col], dtype='Int64')

    os.makedirs(os.path.dirname(config.REEF_YEAR_SEQUENCE_SUMMARY_PATH), exist_ok=True)
    summary_df.to_csv(config.REEF_YEAR_SEQUENCE_SUMMARY_PATH, index=False)

    if len(summary_df) != len(master_df):
        raise ValueError(
            f"Row count mismatch after summary export: master={len(master_df)}, summary={len(summary_df)}"
        )

    _print_summary(master_df, summary_df)
    print(f"\nSaved reef-year sequence summary to {config.REEF_YEAR_SEQUENCE_SUMMARY_PATH}")


if __name__ == '__main__':
    export_reef_year_sequence_summary()
