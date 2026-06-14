from __future__ import annotations
import gzip
import json
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
TARGET_RECORDS = 20000
RANDOM_SEED = 42
BASE_URL = 'https://commondatastorage.googleapis.com/clusterdata-2011-2/task_events'
PART_FILES = ['part-00000-of-00500.csv.gz', 'part-00001-of-00500.csv.gz', 'part-00002-of-00500.csv.gz', 'part-00003-of-00500.csv.gz', 'part-00004-of-00500.csv.gz']
DOWNLOAD_DIR = Path('trace_downloads')
OUTPUT_DIR = Path('public_trace_data')
TASK_EVENT_COLUMNS = ['time', 'missing_info', 'job_id', 'task_index', 'machine_id', 'event_type', 'user', 'scheduling_class', 'priority', 'resource_request_cpu', 'resource_request_memory', 'resource_request_disk', 'different_machine_constraint']
TASK_TYPES = ['batch_job', 'service_task', 'analytics_task', 'compute_task', 'background_task']

def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f'Already downloaded: {output_path}')
        return
    print(f'Downloading: {url}')
    print(f'Saving to: {output_path}')
    urllib.request.urlretrieve(url, output_path)
    print('Download complete.')

def ensure_downloads() -> list[Path]:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []
    for filename in PART_FILES:
        url = f'{BASE_URL}/{filename}'
        output_path = DOWNLOAD_DIR / filename
        download_file(url, output_path)
        downloaded_paths.append(output_path)
    return downloaded_paths

def read_submit_events_from_part(path: Path, remaining: int) -> pd.DataFrame:
    print(f'Reading submit events from: {path}')
    chunks = []
    rows_collected = 0
    reader = pd.read_csv(path, compression='gzip', header=None, names=TASK_EVENT_COLUMNS, chunksize=200000, low_memory=False)
    for chunk in reader:
        chunk['event_type'] = pd.to_numeric(chunk['event_type'], errors='coerce')
        submit = chunk[chunk['event_type'] == 0].copy()
        if submit.empty:
            continue
        submit = submit[['time', 'job_id', 'task_index', 'scheduling_class', 'priority', 'resource_request_cpu', 'resource_request_memory', 'resource_request_disk']]
        chunks.append(submit)
        rows_collected += len(submit)
        print(f'Collected submit rows so far from this file: {rows_collected}')
        if rows_collected >= remaining:
            break
    if not chunks:
        return pd.DataFrame(columns=['time', 'job_id', 'task_index', 'scheduling_class', 'priority', 'resource_request_cpu', 'resource_request_memory', 'resource_request_disk'])
    result = pd.concat(chunks, ignore_index=True)
    return result.head(remaining)

def collect_submit_events(paths: list[Path], target_records: int) -> pd.DataFrame:
    collected = []
    total = 0
    for path in paths:
        remaining = target_records - total
        if remaining <= 0:
            break
        part_df = read_submit_events_from_part(path, remaining)
        if len(part_df) > 0:
            collected.append(part_df)
            total += len(part_df)
            print(f'Total collected submit events: {total}')
    if not collected:
        raise RuntimeError('No submit events were collected. Check download files and internet connection.')
    df = pd.concat(collected, ignore_index=True)
    df = df.sort_values('time').head(target_records).reset_index(drop=True)
    return df

def clean_numeric(series: pd.Series, default: float=0.0) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.fillna(default)
    return s

def map_to_cloud_edge_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df['time'] = clean_numeric(df['time'], 0)
    df['job_id'] = clean_numeric(df['job_id'], 0).astype('int64')
    df['task_index'] = clean_numeric(df['task_index'], 0).astype('int64')
    df['priority'] = clean_numeric(df['priority'], 0)
    df['scheduling_class'] = clean_numeric(df['scheduling_class'], 0)
    df['resource_request_cpu'] = clean_numeric(df['resource_request_cpu'], 0)
    df['resource_request_memory'] = clean_numeric(df['resource_request_memory'], 0)
    df['resource_request_disk'] = clean_numeric(df['resource_request_disk'], 0)
    df = df[(df['resource_request_cpu'] >= 0) & (df['resource_request_memory'] >= 0) & (df['time'] >= 0)].copy()
    df = df.sort_values('time').head(TARGET_RECORDS).reset_index(drop=True)
    first_time = float(df['time'].iloc[0])
    df['arrival_time_s'] = (df['time'] - first_time) / 1000000.0
    df['arrival_time_s'] = df['arrival_time_s'].clip(lower=0)
    df['inter_arrival_time_s'] = df['arrival_time_s'].diff().fillna(df['arrival_time_s'])
    if df['arrival_time_s'].max() <= 0:
        df['arrival_time_s'] = np.arange(len(df)) * 0.001
        df['inter_arrival_time_s'] = df['arrival_time_s'].diff().fillna(0)
    df['task_type'] = df['scheduling_class'].astype(int).map({0: 'background_task', 1: 'batch_job', 2: 'analytics_task', 3: 'service_task'}).fillna('compute_task')
   priority_bucket = pd.cut(
    df['priority'].clip(lower=0, upper=100),
    bins=[-1, 1, 4, 8, 100],
    labels=[1, 2, 3, 4],
)

df['priority_mapped'] = priority_bucket.astype(float).fillna(1).astype(int)
    cpu = df['resource_request_cpu'].clip(lower=0)
    mem = df['resource_request_memory'].clip(lower=0)
    disk = df['resource_request_disk'].clip(lower=0)
    df['cpu_demand_mi'] = (50 + cpu * 600 + mem * 200).round(3).clip(lower=1)
    df['memory_demand_mb'] = (64 + mem * 4096).round(3).clip(lower=8)
    df['task_size_kb'] = (64 + cpu * 2048 + mem * 1024 + disk * 512).round(3).clip(lower=1)
    df['deadline_ms'] = (250 + (4 - df['priority_mapped']) * 150 + (3 - df['scheduling_class'].clip(0, 3)) * 100).round().astype(int).clip(lower=100)
    zones = np.array(['edge_zone_A', 'edge_zone_B', 'edge_zone_C', 'edge_zone_D'])
    zone_idx = (df['job_id'].astype(int) % 4).to_numpy()
    df['source_zone'] = zones[zone_idx]
    df['edge_node_candidate'] = ['edge_' + str(i + 1) for i in zone_idx]
    df['edge_cloud_network_delay_ms'] = (45 + (df['task_size_kb'] / df['task_size_kb'].quantile(0.95)).clip(0, 1) * 25).round(3)
    out = pd.DataFrame({'task_id': [f'GCT_{int(j)}_{int(t)}_{idx:05d}' for idx, (j, t) in enumerate(zip(df['job_id'], df['task_index']))], 'scenario': 'public_trace_validation', 'run_id': 1, 'arrival_time_s': df['arrival_time_s'].round(6), 'inter_arrival_time_s': df['inter_arrival_time_s'].round(6), 'lambda_at_arrival_tps': 0, 'burst_flag': 0, 'source_zone': df['source_zone'], 'edge_node_candidate': df['edge_node_candidate'], 'task_type': df['task_type'], 'priority': df['priority_mapped'], 'task_size_kb': df['task_size_kb'], 'cpu_demand_mi': df['cpu_demand_mi'], 'memory_demand_mb': df['memory_demand_mb'], 'deadline_ms': df['deadline_ms'], 'edge_cloud_network_delay_ms': df['edge_cloud_network_delay_ms']})
    out['arrival_second'] = np.floor(out['arrival_time_s']).astype(int)
    counts = out.groupby('arrival_second')['task_id'].transform('count')
    out['lambda_at_arrival_tps'] = counts.astype(int)
    out.drop(columns=['arrival_second'], inplace=True)
    return out

def write_scenario_parameters(output_dir: Path) -> None:
    params = pd.DataFrame([{'scenario': 'public_trace_validation', 'description': 'Google ClusterData 2011 task_events-derived validation workload', 'duration_s': 0, 'base_lambda_tps': 0, 'burst_start_s': None, 'burst_end_s': None, 'burst_lambda_tps': None, 'edge_nodes': 4, 'edge_service_rate_tps_each': 7, 'cloud_servers': 3, 'cloud_service_rate_tps_each': 18, 'edge_queue_capacity': 120, 'cloud_queue_capacity': 250, 'mean_network_delay_ms': 55, 'std_network_delay_ms': 0, 'total_edge_capacity_tps': 28, 'total_cloud_capacity_tps': 54, 'total_system_capacity_tps': 82}])
    params.to_csv(output_dir / 'scenario_parameters.csv', index=False)

def write_summary_and_metadata(mapped: pd.DataFrame, output_dir: Path) -> None:
    duration = max(mapped['arrival_time_s'].max() - mapped['arrival_time_s'].min(), 1)
    summary = pd.DataFrame([{'scenario': 'public_trace_validation', 'records': len(mapped), 'duration_s': round(duration, 3), 'observed_arrival_rate_tps': round(len(mapped) / duration, 3), 'avg_task_size_kb': round(mapped['task_size_kb'].mean(), 3), 'avg_cpu_demand_mi': round(mapped['cpu_demand_mi'].mean(), 3), 'avg_memory_demand_mb': round(mapped['memory_demand_mb'].mean(), 3), 'avg_network_delay_ms': round(mapped['edge_cloud_network_delay_ms'].mean(), 3)}])
    summary.to_csv(output_dir / 'public_trace_summary.csv', index=False)
    metadata = {'dataset_name': 'Public Trace-Derived Cloud-Edge Validation Dataset', 'source_dataset': 'Google ClusterData 2011-2 task_events', 'source_url': 'https://commondatastorage.googleapis.com/clusterdata-2011-2/', 'selected_records': int(len(mapped)), 'selection_method': 'Time-ordered submit events from task_events partitions', 'mapping_note': 'Google cluster trace provides production task arrival and resource request patterns. Cloud-edge specific fields such as network delay and queue parameters are supplied by the proposed model because the source trace is not a native cloud-edge trace.'}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    readme = f'\nPublic Trace-Derived Validation Dataset\n=======================================\n\nSource\n------\nGoogle ClusterData 2011-2 task_events table.\n\nPurpose\n-------\nThis dataset maps real production cloud task submission events into the\ncloud-edge simulator format used in the journal paper.\n\nImportant Note\n--------------\nThe Google trace is a cloud/datacenter trace, not a native cloud-edge trace.\nTherefore, it is used to provide realistic task arrival order and resource\ndemand patterns. Edge/cloud queue capacity, network delay, and redirection\nrules are supplied by the proposed cloud-edge model.\n\nRecords\n-------\n{len(mapped)} task records.\n\nFiles\n-----\n1. cloud_edge_workload_trace.csv\n2. scenario_parameters.csv\n3. public_trace_summary.csv\n4. metadata.json\n5. README.txt\n'.strip()
    (output_dir / 'README.txt').write_text(readme, encoding='utf-8')

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure_downloads()
    raw = collect_submit_events(paths, TARGET_RECORDS)
    print('Mapping public trace records to cloud-edge simulator format...')
    mapped = map_to_cloud_edge_format(raw)
    mapped.to_csv(OUTPUT_DIR / 'cloud_edge_workload_trace.csv', index=False)
    write_scenario_parameters(OUTPUT_DIR)
    write_summary_and_metadata(mapped, OUTPUT_DIR)
    print()
    print('Public trace-derived validation dataset created successfully.')
    print(f'Output folder: {OUTPUT_DIR.resolve()}')
    print(f'Records: {len(mapped):,}')
    print()
    print('Summary:')
    print(pd.read_csv(OUTPUT_DIR / 'public_trace_summary.csv').to_string(index=False))
if __name__ == '__main__':
    main()
