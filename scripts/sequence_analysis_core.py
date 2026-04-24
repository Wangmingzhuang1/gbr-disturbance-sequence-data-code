# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

import config


SEQ_ORDER = [
    'Isolated_Storm',
    'Isolated_Heatwave',
    'Concurrent',
    'S_to_H',
    'S_to_S',
    'H_to_H',
    'H_to_S',
]

STORM_LINKED_SEQS = {'Isolated_Storm', 'S_to_H', 'S_to_S'}
HEAT_LINKED_SEQS = {'Isolated_Heatwave', 'H_to_H', 'H_to_S'}

SAME_YEAR_S_TO_H = 'Sequential S->H (Same Year)'
SAME_YEAR_H_TO_S = 'Sequential H->S (Same Year)'
CONCURRENT_PREFIX = 'Synergistic Concurrent'


def load_master_matrix(base_dir=None):
    path = config.MASTER_MATRIX_PATH
    if base_dir is not None:
        candidates = [
            os.path.join(base_dir, 'output', 'data', 'eco_response_master_matrix_merged.csv'),
            os.path.join(base_dir, 'output', 'eco_response_master_matrix_merged.csv'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break
    return pd.read_csv(path)


def _preferred_algae_column(df_reef):
    if 'MACROALGAE_cover' in df_reef.columns:
        return 'MACROALGAE_cover'
    if 'ALGAE_cover' in df_reef.columns:
        return 'ALGAE_cover'
    return None


def _event_rows(df_reef):
    mask = (df_reef['has_storm'] == 1) | (df_reef['has_heatwave'] == 1)
    return df_reef.loc[mask].sort_values('year').copy()


def _disturbance_type(row):
    return str(row.get('Disturbance_Type', '') or '')


def _embedded_same_year_sequence(row):
    disturbance = _disturbance_type(row)
    if SAME_YEAR_S_TO_H in disturbance:
        return ('S -> H', 'S_to_H', 'Storm', 'Heatwave')
    if SAME_YEAR_H_TO_S in disturbance:
        return ('H -> S', 'H_to_S', 'Heatwave', 'Storm')
    return None


def _event_signature(row):
    disturbance = _disturbance_type(row)
    has_storm = int(row.get('has_storm', 0)) == 1
    has_heatwave = int(row.get('has_heatwave', 0)) == 1

    if CONCURRENT_PREFIX in disturbance:
        return 'Concurrent'
    if SAME_YEAR_S_TO_H in disturbance:
        return 'Storm'
    if SAME_YEAR_H_TO_S in disturbance:
        return 'Heatwave'
    if 'Concurrent' in disturbance or (has_storm and has_heatwave):
        return 'Concurrent'
    if has_storm:
        return 'Storm'
    if has_heatwave:
        return 'Heatwave'
    return None


def _label_sequence(start_type, target_type):
    if target_type is None:
        mapping = {
            'Storm': ('Isolated Storm', 'Isolated_Storm', 'Storm', None),
            'Heatwave': ('Isolated Heatwave', 'Isolated_Heatwave', 'Heatwave', None),
            'Concurrent': ('Concurrent', 'Concurrent', 'Concurrent', None),
        }
        return mapping.get(start_type, (None, None, None, None))

    pair_map = {
        ('Storm', 'Heatwave'): ('S -> H', 'S_to_H', 'Storm', 'Heatwave'),
        ('Heatwave', 'Storm'): ('H -> S', 'H_to_S', 'Heatwave', 'Storm'),
        ('Storm', 'Storm'): ('S -> S', 'S_to_S', 'Storm', 'Storm'),
        ('Heatwave', 'Heatwave'): ('H -> H', 'H_to_H', 'Heatwave', 'Heatwave'),
    }
    return pair_map.get((start_type, target_type), ('Concurrent', 'Concurrent', start_type, target_type))


def find_clean_baseline(df_reef, start_year, all_event_years, allow_conflicts=False):
    window_years = [start_year - offset for offset in config.BASELINE_WINDOW]
    lower = min(window_years)
    conflicts = [year for year in all_event_years if lower <= year < start_year]
    if conflicts and not allow_conflicts:
        return None, 'baseline_not_clean'

    matches = df_reef[df_reef['year'].isin(window_years)].dropna(subset=['HC_cover']).copy()
    if len(matches) < config.MIN_BASELINE_SURVEYS:
        return None, 'baseline_insufficient_points'

    algae_col = _preferred_algae_column(matches)
    baseline = {
        'years': sorted(matches['year'].astype(int).tolist()),
        'n_surveys': int(len(matches)),
        'hc': float(matches['HC_cover'].mean()),
        'juv': float(matches['Juveniles'].mean()) if 'Juveniles' in matches.columns and matches['Juveniles'].notna().any() else np.nan,
        'algae': float(matches[algae_col].mean()) if algae_col and matches[algae_col].notna().any() else np.nan,
    }
    if baseline['hc'] < config.MIN_BASELINE_HC:
        return baseline, 'baseline_below_threshold'
    return baseline, 'success'


def find_nadir_state(df_reef, anchor_year):
    years = [anchor_year + offset for offset in config.NADIR_WINDOW]
    matches = df_reef[df_reef['year'].isin(years)].dropna(subset=['HC_cover']).copy()
    if matches.empty:
        return None, 'nadir_missing_data'

    algae_col = _preferred_algae_column(matches)
    nadir_row = matches.loc[matches['HC_cover'].idxmin()]
    return {
        'year': int(nadir_row['year']),
        'hc': float(nadir_row['HC_cover']),
        'juv': float(nadir_row['Juveniles']) if 'Juveniles' in matches.columns and pd.notna(nadir_row.get('Juveniles')) else np.nan,
        'algae': float(nadir_row[algae_col]) if algae_col and pd.notna(nadir_row.get(algae_col)) else np.nan,
    }, 'success'


def find_final_state(df_reef, nadir_year):
    target_year = nadir_year + config.RECOVERY_LOOKAHEAD
    candidates = [target_year, target_year - 1, target_year + 1]
    matches = df_reef[df_reef['year'].isin(candidates)].dropna(subset=['HC_cover']).copy()
    if matches.empty:
        return None

    matches['distance_to_target'] = (matches['year'] - target_year).abs()
    row = matches.sort_values(['distance_to_target', 'year']).iloc[0]
    algae_col = _preferred_algae_column(matches)
    return {
        'year': int(row['year']),
        'hc': float(row['HC_cover']),
        'juv': float(row['Juveniles']) if 'Juveniles' in matches.columns and pd.notna(row.get('Juveniles')) else np.nan,
        'algae': float(row[algae_col]) if algae_col and pd.notna(row.get(algae_col)) else np.nan,
    }


def _mean_prior_cots(df_reef, start_year):
    if 'COTS_density' not in df_reef.columns:
        return np.nan
    years = [start_year - 1, start_year - 2, start_year - 3]
    matches = df_reef[df_reef['year'].isin(years)]['COTS_density'].dropna()
    return float(matches.mean()) if not matches.empty else np.nan


def choose_target_row(df_reef, start_row, consumed_targets, lookahead_years, rule='first_event'):
    candidates = df_reef[
        (df_reef['year'] > start_row['year']) &
        (df_reef['year'] <= start_row['year'] + lookahead_years) &
        ((df_reef['has_storm'] == 1) | (df_reef['has_heatwave'] == 1))
    ].copy()
    if candidates.empty:
        return None

    candidates = candidates[candidates.apply(_embedded_same_year_sequence, axis=1).isna()]
    if candidates.empty:
        return None

    if consumed_targets:
        candidates = candidates[~candidates['year'].isin(consumed_targets)]
    if candidates.empty:
        return None

    candidates['event_signature'] = candidates.apply(_event_signature, axis=1)
    candidates = candidates.sort_values('year')

    if rule == 'storm_heat_priority':
        heat_follow = candidates[candidates['event_signature'] == 'Heatwave']
        if _event_signature(start_row) == 'Storm' and not heat_follow.empty:
            return heat_follow.iloc[0]
    return candidates.iloc[0]


def _same_year_exposure_fields(start_row, sequence_code):
    wind = float(start_row.get('max_wind_ms', 0) or 0)
    dhw = float(start_row.get('max_dhw', 0) or 0)
    if sequence_code == 'S_to_H':
        return {
            'start_max_wind': wind,
            'start_max_dhw': 0.0,
            'target_max_wind': 0.0,
            'target_max_dhw': dhw,
            'storm_exposure_seq': wind,
            'heat_exposure_seq': dhw,
        }
    if sequence_code == 'H_to_S':
        return {
            'start_max_wind': 0.0,
            'start_max_dhw': dhw,
            'target_max_wind': wind,
            'target_max_dhw': 0.0,
            'storm_exposure_seq': wind,
            'heat_exposure_seq': dhw,
        }
    return {
        'start_max_wind': wind,
        'start_max_dhw': dhw,
        'target_max_wind': 0.0,
        'target_max_dhw': 0.0,
        'storm_exposure_seq': wind,
        'heat_exposure_seq': dhw,
    }


def extract_reef_sequences(
    df_reef,
    rule='first_event',
    lookahead_years=None,
    mode=None,
    allow_baseline_conflicts=False,
    eligible_start_years=None,
):
    lookahead_years = config.LOOKAHEAD_WINDOW if lookahead_years is None else lookahead_years
    mode = config.EXTRACTION_MODE if mode is None else mode

    reef_name = df_reef.iloc[0]['reef_name']
    df_reef = df_reef.sort_values('year').copy()
    events = _event_rows(df_reef)
    all_event_years = set(events['year'].astype(int).tolist())
    consumed_targets = set()
    rows = []

    attrition = {
        'stage1_potential_starts': int(len(events)),
        'dropped_consumed': 0,
        'dropped_baseline_not_clean': 0,
        'dropped_baseline_insufficient_points': 0,
        'dropped_baseline_below_threshold': 0,
        'dropped_nadir_missing_data': 0,
        'dropped_label_failure': 0,
        'retained_sequences': 0,
    }

    for _, start_row in events.iterrows():
        start_year = int(start_row['year'])
        if eligible_start_years is not None and start_year not in eligible_start_years:
            continue
        if mode == 'strict' and start_year in consumed_targets:
            attrition['dropped_consumed'] += 1
            continue

        baseline, baseline_status = find_clean_baseline(
            df_reef,
            start_year,
            all_event_years,
            allow_conflicts=allow_baseline_conflicts,
        )
        if baseline_status != 'success':
            attrition[f'dropped_{baseline_status}'] += 1
            continue

        embedded_sequence = _embedded_same_year_sequence(start_row)
        if embedded_sequence is not None:
            target_row = None
            sequence_type, sequence_code, start_event_type, target_event_type = embedded_sequence
            target_year = start_year
            gap_years = 0
            anchor_year = start_year
        else:
            target_row = choose_target_row(df_reef, start_row, consumed_targets, lookahead_years, rule=rule)
            start_type = _event_signature(start_row)
            target_type = _event_signature(target_row) if target_row is not None else None
            sequence_type, sequence_code, start_event_type, target_event_type = _label_sequence(start_type, target_type)
            if sequence_code is None:
                attrition['dropped_label_failure'] += 1
                continue
            target_year = int(target_row['year']) if target_row is not None else np.nan
            gap_years = (int(target_row['year']) - start_year) if target_row is not None else np.nan
            anchor_year = int(target_row['year']) if target_row is not None else start_year

        nadir_state, nadir_status = find_nadir_state(df_reef, anchor_year)
        if nadir_status != 'success':
            attrition[f'dropped_{nadir_status}'] += 1
            continue

        final_state = find_final_state(df_reef, nadir_state['year'])
        recovery_rate = np.nan
        recovery_pct = np.nan
        if final_state is not None and baseline['hc'] > 0:
            recovery_years = final_state['year'] - nadir_state['year']
            if recovery_years > 0:
                recovery_rate = (final_state['hc'] - nadir_state['hc']) / recovery_years
            recovery_pct = (final_state['hc'] - nadir_state['hc']) / baseline['hc']

        if embedded_sequence is not None:
            exposure_fields = _same_year_exposure_fields(start_row, sequence_code)
            target_disturbance_type = _disturbance_type(start_row)
        else:
            start_wind = float(start_row.get('max_wind_ms', 0) or 0)
            start_dhw = float(start_row.get('max_dhw', 0) or 0)
            target_wind = float(target_row.get('max_wind_ms', 0) or 0) if target_row is not None else 0.0
            target_dhw = float(target_row.get('max_dhw', 0) or 0) if target_row is not None else 0.0
            exposure_fields = {
                'start_max_wind': start_wind,
                'start_max_dhw': start_dhw,
                'target_max_wind': target_wind,
                'target_max_dhw': target_dhw,
                'storm_exposure_seq': start_wind + target_wind,
                'heat_exposure_seq': start_dhw + target_dhw,
            }
            target_disturbance_type = _disturbance_type(target_row) if target_row is not None else ''

        result = {
            'reef_name': reef_name,
            'region_lat': start_row.get('region_lat', np.nan),
            'sector': start_row.get('sector', 'Unknown'),
            'start_year': start_year,
            'event_year': start_year,
            'target_year': target_year,
            'end_year': nadir_state['year'],
            'nadir_year': nadir_state['year'],
            'final_year': final_state['year'] if final_state is not None else np.nan,
            'gap_years': gap_years,
            'time_gap_years': gap_years,
            'sequence_type': sequence_type,
            'sequence_type_primary': sequence_code,
            'seq_category_main': sequence_code,
            'seq_category': sequence_code,
            'start_event_type': start_event_type,
            'target_event_type': target_event_type,
            'start_disturbance_type': _disturbance_type(start_row),
            'target_disturbance_type': target_disturbance_type,
            'baseline_start_year': min(baseline['years']),
            'baseline_end_year': max(baseline['years']),
            'baseline_survey_n': baseline['n_surveys'],
            'baseline_hc': baseline['hc'],
            'baseline_juv': baseline['juv'],
            'baseline_algae': baseline['algae'],
            'nadir_hc': nadir_state['hc'],
            'nadir_juv': nadir_state['juv'],
            'nadir_algae': nadir_state['algae'],
            'final_hc': final_state['hc'] if final_state is not None else np.nan,
            'final_juv': final_state['juv'] if final_state is not None else np.nan,
            'final_algae': final_state['algae'] if final_state is not None else np.nan,
            'rel_loss': np.clip((baseline['hc'] - nadir_state['hc']) / baseline['hc'], -1.0, 1.0) if baseline['hc'] > 0 else np.nan,
            'recovery_rate': recovery_rate,
            'recovery_pct': recovery_pct,
            'cots_lag_3yr': _mean_prior_cots(df_reef, start_year),
            'initial_cots': float(start_row.get('COTS_density', np.nan)),
            'initial_herbivore': float(start_row.get('Fish_Herbivores', np.nan)),
        }
        result.update(exposure_fields)
        rows.append(result)
        attrition['retained_sequences'] += 1
        if embedded_sequence is not None:
            consumed_targets.add(start_year)
        elif target_row is not None:
            consumed_targets.add(int(target_row['year']))

    return pd.DataFrame(rows), attrition


def extract_sequences(
    master_df,
    lookahead_years=None,
    rule='first_event',
    mode=None,
    allow_baseline_conflicts=False,
    eligible_start_years_by_reef=None,
):
    frames = []
    for reef_name, df_reef in master_df.groupby('reef_name'):
        _ = reef_name
        df_seq, _attrition = extract_reef_sequences(
            df_reef,
            rule=rule,
            lookahead_years=lookahead_years,
            mode=mode,
            allow_baseline_conflicts=allow_baseline_conflicts,
            eligible_start_years=None if eligible_start_years_by_reef is None else eligible_start_years_by_reef.get(reef_name),
        )
        if not df_seq.empty:
            frames.append(df_seq)
    if not frames:
        return pd.DataFrame(columns=['reef_name', 'seq_category_main', 'rel_loss'])
    return pd.concat(frames, ignore_index=True)


def add_fate_fields(df):
    work = df.copy()
    if work.empty:
        return work

    work['drop_abs'] = work['baseline_hc'] - work['nadir_hc']
    work['impact_flag'] = np.where(work['drop_abs'] >= config.IMPACT_THRESHOLD, 1, 0)
    work['recovery_gain_abs'] = work['final_hc'] - work['nadir_hc']
    work['recovery_observed_flag'] = np.where(work['final_hc'].notna(), 1, 0)

    eligible = (
        (work['impact_flag'] == 1) &
        (work['recovery_observed_flag'] == 1) &
        work['drop_abs'].notna() &
        (work['drop_abs'] > 0)
    )
    work['recovery_frac_loss'] = np.nan
    work.loc[eligible, 'recovery_frac_loss'] = (
        work.loc[eligible, 'recovery_gain_abs'] / work.loc[eligible, 'drop_abs']
    )

    work['recovered_flag'] = np.nan
    recovered = eligible & (work['recovery_frac_loss'] >= 0.5)
    failed = eligible & (work['recovery_frac_loss'] < 0.5)
    work.loc[recovered, 'recovered_flag'] = 1
    work.loc[failed, 'recovered_flag'] = 0

    work['fate_status'] = np.where(
        work['impact_flag'] == 0,
        'non_impact',
        np.where(
            work['recovery_observed_flag'] == 0,
            'recovery_unobserved',
            np.where(work['recovered_flag'] == 1, 'recovered', 'failed')
        ),
    )

    return work


def add_disturbance_group(df, seq_column='seq_category_main'):
    work = df.copy()
    if work.empty:
        return work

    work['disturbance_group'] = pd.Series(index=work.index, dtype='object')
    work.loc[work[seq_column].isin(STORM_LINKED_SEQS), 'disturbance_group'] = 'Storm-linked'
    work.loc[work[seq_column].isin(HEAT_LINKED_SEQS), 'disturbance_group'] = 'Heat-linked'
    work.loc[work[seq_column] == 'Concurrent', 'disturbance_group'] = 'Concurrent'
    return work


def add_baseline_hc_group(df, baseline_column='baseline_hc'):
    work = df.copy()
    if work.empty:
        return work

    median_hc = work[baseline_column].median()
    work['baseline_hc_median'] = median_hc
    work['baseline_hc_group'] = np.where(
        work[baseline_column] <= median_hc,
        'Low baseline HC',
        'High baseline HC',
    )
    return work
