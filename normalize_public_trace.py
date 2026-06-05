from __future__ import annotations
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
DATASET_DIR = Path('public_trace_data')
TRACE_FILE = DATASET_DIR / 'cloud_edge_workload_trace.csv'
PARAM_FILE = DATASET_DIR / 'scenario_parameters.csv'
TARGET_DURATION_S = 600.0
DEBATCHING_FACTOR = 0.0

def backup_file(path: Path) -> Path:
    backup_path = path.with_name(path.stem + '_raw_timing_backup' + path.suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f'Backup created: {backup_path}')
    else:
        print(f'Backup already exists: {backup_path}')
    return backup_path

def normalize_arrival_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['arrival_time_s', 'task_id'], ignore_index=True)
    raw_min = float(df['arrival_time_s'].min())
    raw_max = float(df['arrival_time_s'].max())
    raw_span = raw_max - raw_min
    n = len(df)
    if raw_span <= 0:
        normalized = np.linspace(0, TARGET_DURATION_S, n)
    else:
        raw_relative = (df['arrival_time_s'] - raw_min) / raw_span
        rank_relative = np.linspace(0, 1, n)
        blended_relative = (1.0 - DEBATCHING_FACTOR) * raw_relative + DEBATCHING_FACTOR * rank_relative
        normalized = blended_relative * TARGET_DURATION_S
    df['arrival_time_s'] = np.round(normalized, 6)
    df['inter_arrival_time_s'] = df['arrival_time_s'].diff().fillna(df['arrival_time_s'])
    df['inter_arrival_time_s'] = df['inter_arrival_time_s'].clip(lower=0).round(6)
    arrival_second = np.floor(df['arrival_time_s']).astype(int)
    counts = arrival_second.map(arrival_second.value_counts()).astype(int)
    df['lambda_at_arrival_tps'] = counts
    df['scenario'] = 'public_trace_validation'
    df['run_id'] = 1
    return df

def update_scenario_parameters() -> None:
    if not PARAM_FILE.exists():
        print(f'Warning: {PARAM_FILE} not found. Skipping parameter update.')
        return
    params = pd.read_csv(PARAM_FILE)
    if 'scenario' in params.columns:
        mask = params['scenario'] == 'public_trace_validation'
        if mask.any():
            params.loc[mask, 'duration_s'] = TARGET_DURATION_S
            params.loc[mask, 'description'] = 'Time-normalized Google ClusterData-derived validation workload'
        else:
            print('Warning: public_trace_validation row not found in scenario parameters.')
    params.to_csv(PARAM_FILE, index=False)
    print(f'Updated: {PARAM_FILE}')

def write_report(original_df: pd.DataFrame, normalized_df: pd.DataFrame) -> None:
    raw_span = max(float(original_df['arrival_time_s'].max() - original_df['arrival_time_s'].min()), 1e-09)
    norm_span = max(float(normalized_df['arrival_time_s'].max() - normalized_df['arrival_time_s'].min()), 1e-09)
    raw_rate = len(original_df) / raw_span
    norm_rate = len(normalized_df) / norm_span
    report = f'\nPublic Trace Time Normalization Report\n======================================\n\nRecords:\n{len(normalized_df):,}\n\nOriginal trace time span:\n{raw_span:.6f} seconds\n\nOriginal approximate arrival rate:\n{raw_rate:.3f} tasks/s\n\nNormalized validation time span:\n{norm_span:.6f} seconds\n\nNormalized approximate arrival rate:\n{norm_rate:.3f} tasks/s\n\nTarget duration:\n{TARGET_DURATION_S:.3f} seconds\n\nDe-batching factor:\n{DEBATCHING_FACTOR:.3f}\n\nReason for normalization:\nThe public trace subset is derived from a large production cluster and may\ncontain a high density of task submissions over a short interval. Directly\nreplaying the raw timestamp scale against the smaller simulated cloud-edge\nconfiguration causes near-total queue saturation and tests only overload\ncollapse. Time normalization preserves the task ordering and relative temporal\npattern while matching the validation workload scale to the simulated\ncloud-edge environment.\n\nNote:\nTask resource fields such as CPU demand, memory demand, task size, and priority\nare not modified by this script. Only arrival timing and derived inter-arrival\nfields are updated.\n'.strip()
    report_path = DATASET_DIR / 'normalization_report.txt'
    report_path.write_text(report, encoding='utf-8')
    print(f'Report written: {report_path}')

def write_updated_public_trace_summary(normalized_df: pd.DataFrame) -> None:
    """Refresh public_trace_summary.csv after timing normalization.

    This prevents a stale summary file from continuing to report the raw
    pre-normalization duration and arrival rate.
    """
    duration = max(float(normalized_df['arrival_time_s'].max() - normalized_df['arrival_time_s'].min()), 1e-09)
    summary = pd.DataFrame([{
        'scenario': 'public_trace_validation',
        'records': int(len(normalized_df)),
        'duration_s': round(duration, 3),
        'observed_arrival_rate_tps': round(len(normalized_df) / duration, 3),
        'avg_task_size_kb': round(float(normalized_df['task_size_kb'].mean()), 3),
        'avg_cpu_demand_mi': round(float(normalized_df['cpu_demand_mi'].mean()), 3),
        'avg_memory_demand_mb': round(float(normalized_df['memory_demand_mb'].mean()), 3),
        'avg_deadline_ms': round(float(normalized_df['deadline_ms'].mean()), 3),
        'avg_network_delay_ms': round(float(normalized_df['edge_cloud_network_delay_ms'].mean()), 3),
        'time_normalized': True,
        'target_duration_s': TARGET_DURATION_S,
        'debatching_factor': DEBATCHING_FACTOR,
    }])
    summary_path = DATASET_DIR / 'public_trace_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'Updated normalized summary: {summary_path}')

def update_metadata() -> None:
    metadata_path = DATASET_DIR / 'metadata.json'
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception:
        metadata = {}
    metadata['time_normalization'] = {'applied': True, 'target_duration_s': TARGET_DURATION_S, 'debatching_factor': DEBATCHING_FACTOR, 'reason': 'Raw public trace timing was normalized to match the scale of the simulated cloud-edge validation environment while preserving task order and relative arrival pattern.'}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Updated metadata: {metadata_path}')

def main() -> None:
    if not TRACE_FILE.exists():
        raise FileNotFoundError(f'Missing file: {TRACE_FILE}\nRun build_public_trace_data.py first.')
    print('Loading public trace validation dataset...')
    original = pd.read_csv(TRACE_FILE)
    backup_file(TRACE_FILE)
    if PARAM_FILE.exists():
        backup_file(PARAM_FILE)
    print('Normalizing arrival times...')
    normalized = normalize_arrival_times(original)
    normalized.to_csv(TRACE_FILE, index=False)
    print(f'Updated normalized dataset: {TRACE_FILE}')
    update_scenario_parameters()
    write_updated_public_trace_summary(normalized)
    write_report(original, normalized)
    update_metadata()
    print()
    print('Time normalization completed successfully.')
    print()
    print('Now re-run:')
    print('python run_public_trace_validation.py')
    print()
    print('Or open run_public_trace_validation.py in IDLE and press F5.')
if __name__ == '__main__':
    main()
