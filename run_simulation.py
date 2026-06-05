from __future__ import annotations
import argparse
import json
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None
DEFAULT_INPUT_DIR = Path('generated_workload')
DEFAULT_OUTPUT_DIR = Path('simulation_results')
TASK_FILE_NAME = 'cloud_edge_workload_trace.csv'
PARAM_FILE_NAME = 'scenario_parameters.csv'
PROCESSING_MODES = ['edge_only', 'cloud_only', 'hybrid_cloud_edge', 'deadline_aware_hybrid']
SCENARIO_ORDER = ['low_load', 'medium_load', 'high_load', 'burst_load', 'extended_workload', 'public_trace_validation']
HYBRID_EDGE_DELAY_THRESHOLD_MS = 220.0
MIN_SERVICE_TIME_MS = 5.0
NOMINAL_CPU_DEMAND_MI = 100.0
EDGE_CLOUD_BANDWIDTH_MBPS = 25.0
RESULT_RETURN_DELAY_FACTOR = 0.65
MIN_RETURN_DELAY_MS = 5.0

# Cost/energy proxy coefficients are normalized simulation units, not currency or Joules.
# They allow relative comparison across routing modes while keeping the model reproducible.
EDGE_COST_PER_SERVICE_MS = 0.00004
CLOUD_COST_PER_SERVICE_MS = 0.00025
NETWORK_COST_PER_MB = 0.00002
EDGE_ENERGY_PER_SERVICE_MS = 0.004
CLOUD_ENERGY_PER_SERVICE_MS = 0.018
NETWORK_ENERGY_PER_MB = 0.002

THRESHOLD_VALUES_MS = [25, 50, 100, 150, 220, 300]
EDGE_SERVER_VALUES = [2, 4, 6, 8]
CLOUD_SERVER_VALUES = [1, 2, 3, 4]
EDGE_QUEUE_VALUES = [60, 120, 180, 240]
NETWORK_DELAY_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
SENSITIVITY_SCENARIOS = ['high_load', 'burst_load']

@dataclass
class LayerState:
    server_available_times: List[float]
    waiting_start_heap: List[float] = field(default_factory=list)
    queue_events: List[Tuple[float, int]] = field(default_factory=list)
    total_busy_time_ms: float = 0.0

def required_columns() -> set[str]:
    return {'task_id', 'scenario', 'run_id', 'arrival_time_s', 'task_size_kb', 'cpu_demand_mi', 'deadline_ms', 'edge_cloud_network_delay_ms'}

def load_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task_file = input_dir / TASK_FILE_NAME
    param_file = input_dir / PARAM_FILE_NAME
    if not task_file.exists():
        raise FileNotFoundError(f'Missing workload trace: {task_file}')
    if not param_file.exists():
        raise FileNotFoundError(f'Missing scenario parameter file: {param_file}')
    tasks_df = pd.read_csv(task_file)
    params_df = pd.read_csv(param_file)
    missing = required_columns().difference(tasks_df.columns)
    if missing:
        raise ValueError(f'Missing required columns in workload trace: {sorted(missing)}')
    tasks_df['run_id'] = tasks_df['run_id'].astype(int)
    if 'seed' not in tasks_df.columns:
        tasks_df['seed'] = np.nan
    return (tasks_df, params_df)

def get_scenario_params(params_df: pd.DataFrame, scenario: str) -> Dict:
    row = params_df[params_df['scenario'] == scenario]
    if row.empty:
        raise ValueError(f'No parameters found for scenario: {scenario}')
    return row.iloc[0].to_dict()

def scenario_sort_key(scenario: str) -> int:
    scenario = str(scenario)
    if scenario in SCENARIO_ORDER:
        return SCENARIO_ORDER.index(scenario)
    return len(SCENARIO_ORDER)

def mode_sort_key(mode: str) -> int:
    mode = str(mode)
    if mode in PROCESSING_MODES:
        return PROCESSING_MODES.index(mode)
    return len(PROCESSING_MODES)

def sort_result_frame(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Sort output frames without converting unknown labels to NaN."""
    if df.empty:
        return df
    out = df.copy()
    sort_cols = []
    if 'scenario' in group_cols and 'scenario' in out.columns:
        out['_scenario_order'] = out['scenario'].astype(str).map(scenario_sort_key)
        sort_cols.append('_scenario_order')
    if 'mode' in group_cols and 'mode' in out.columns:
        out['_mode_order'] = out['mode'].astype(str).map(mode_sort_key)
        sort_cols.append('_mode_order')
    for col in group_cols:
        if col in out.columns and col not in {'scenario', 'mode'}:
            sort_cols.append(col)
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
        out = out.drop(columns=[c for c in ['_scenario_order', '_mode_order'] if c in out.columns])
    return out

def service_time_ms(cpu_demand_mi: float, service_rate_tps_each: float) -> float:
    base_service_time_ms = 1000.0 / max(float(service_rate_tps_each), 0.001)
    demand_multiplier = max(float(cpu_demand_mi), 1.0) / NOMINAL_CPU_DEMAND_MI
    return max(MIN_SERVICE_TIME_MS, base_service_time_ms * demand_multiplier)

def transfer_delay_ms(task_size_kb: float, bandwidth_mbps: float) -> float:
    kb_per_ms = max(float(bandwidth_mbps) * 0.125, 0.001)
    return max(0.0, float(task_size_kb) / kb_per_ms)

def return_delay_ms(network_delay_ms: float) -> float:
    return max(MIN_RETURN_DELAY_MS, float(network_delay_ms) * RESULT_RETURN_DELAY_FACTOR)

def purge_started_waiting_tasks(state: LayerState, time_ms: float) -> None:
    while state.waiting_start_heap and state.waiting_start_heap[0] <= time_ms:
        heapq.heappop(state.waiting_start_heap)

def waiting_queue_length(state: LayerState, time_ms: float) -> int:
    purge_started_waiting_tasks(state, time_ms)
    return len(state.waiting_start_heap)

def choose_earliest_server(server_available_times: List[float]) -> int:
    return int(np.argmin(server_available_times))

def estimate_waiting_delay_ms(state: LayerState, arrival_time_ms: float) -> float:
    return max(0.0, min(state.server_available_times) - arrival_time_ms)

def add_queue_event(state: LayerState, time_ms: float, delta: int) -> None:
    state.queue_events.append((float(time_ms), int(delta)))

def compute_average_queue_length(queue_events: List[Tuple[float, int]], simulation_end_ms: float) -> float:
    if not queue_events or simulation_end_ms <= 0:
        return 0.0
    events = sorted(queue_events, key=lambda x: x[0])
    current_q = 0
    previous_time = 0.0
    area = 0.0
    for event_time, delta in events:
        event_time = max(0.0, min(float(event_time), simulation_end_ms))
        area += current_q * max(0.0, event_time - previous_time)
        current_q = max(0, current_q + int(delta))
        previous_time = event_time
    area += current_q * max(0.0, simulation_end_ms - previous_time)
    return float(area / simulation_end_ms)

def make_empty_layer_state(server_count: int) -> LayerState:
    return LayerState(server_available_times=[0.0 for _ in range(max(int(server_count), 1))])

def schedule_on_layer(state: LayerState, arrival_to_layer_ms: float, service_ms: float, queue_capacity: int) -> Dict:
    queue_at_arrival = waiting_queue_length(state, arrival_to_layer_ms)
    if queue_at_arrival >= int(queue_capacity):
        return {'accepted': False, 'start_time_ms': None, 'service_finish_time_ms': None, 'waiting_time_ms': None, 'queue_length_at_arrival': queue_at_arrival, 'queue_length_after_assignment': queue_at_arrival}
    server_idx = choose_earliest_server(state.server_available_times)
    start_time_ms = max(float(arrival_to_layer_ms), state.server_available_times[server_idx])
    waiting_time_ms = start_time_ms - float(arrival_to_layer_ms)
    service_finish_time_ms = start_time_ms + float(service_ms)
    state.server_available_times[server_idx] = service_finish_time_ms
    state.total_busy_time_ms += float(service_ms)
    if waiting_time_ms > 0:
        heapq.heappush(state.waiting_start_heap, start_time_ms)
        add_queue_event(state, arrival_to_layer_ms, +1)
        add_queue_event(state, start_time_ms, -1)
    queue_after = waiting_queue_length(state, arrival_to_layer_ms)
    return {'accepted': True, 'start_time_ms': start_time_ms, 'service_finish_time_ms': service_finish_time_ms, 'waiting_time_ms': waiting_time_ms, 'queue_length_at_arrival': queue_at_arrival, 'queue_length_after_assignment': queue_after}

def task_size_mb(task_size_kb: float) -> float:
    return max(0.0, float(task_size_kb) / 1024.0)

def cost_energy_proxy(processing_location: str, service_time_ms: Optional[float], task_size_kb: float) -> Tuple[float, float]:
    """Return normalized cost and energy proxy values for a completed task."""
    if service_time_ms is None:
        return (0.0, 0.0)
    service_ms = max(0.0, float(service_time_ms))
    size_mb = task_size_mb(task_size_kb)
    if processing_location == 'edge':
        return (EDGE_COST_PER_SERVICE_MS * service_ms, EDGE_ENERGY_PER_SERVICE_MS * service_ms)
    if processing_location == 'cloud':
        cost = CLOUD_COST_PER_SERVICE_MS * service_ms + NETWORK_COST_PER_MB * size_mb
        energy = CLOUD_ENERGY_PER_SERVICE_MS * service_ms + NETWORK_ENERGY_PER_MB * size_mb
        return (cost, energy)
    return (0.0, 0.0)

def build_result_row(task: Dict, mode: str, processing_location: str, status: str, start_time_ms: Optional[float], finish_time_ms: Optional[float], service_time: Optional[float], waiting_time: Optional[float], response_time: Optional[float], edge_queue_estimate: int, cloud_queue_estimate: int, cloud_transfer_delay: float, cloud_return_delay: float) -> Dict:
    deadline_ms = float(task.get('deadline_ms', np.nan))
    sla_violation = bool(response_time is not None and np.isfinite(deadline_ms) and (response_time > deadline_ms))
    deadline_slack_ms = deadline_ms - float(response_time) if response_time is not None and np.isfinite(deadline_ms) else np.nan
    cost_proxy, energy_proxy = cost_energy_proxy(processing_location, service_time, float(task.get('task_size_kb', 0.0))) if status == 'completed' else (0.0, 0.0)
    return {'task_id': task['task_id'], 'scenario': task['scenario'], 'run_id': int(task['run_id']), 'seed': task.get('seed', np.nan), 'mode': mode, 'arrival_time_s': float(task['arrival_time_s']), 'processing_location': processing_location, 'status': status, 'deadline_ms': round(deadline_ms, 3) if np.isfinite(deadline_ms) else None, 'deadline_slack_ms': round(deadline_slack_ms, 3) if np.isfinite(deadline_slack_ms) else None, 'start_time_ms': round(start_time_ms, 3) if start_time_ms is not None else None, 'finish_time_ms': round(finish_time_ms, 3) if finish_time_ms is not None else None, 'service_time_ms': round(service_time, 3) if service_time is not None else None, 'waiting_time_ms': round(waiting_time, 3) if waiting_time is not None else None, 'response_time_ms': round(response_time, 3) if response_time is not None else None, 'sla_violation': int(sla_violation), 'edge_queue_estimate': int(edge_queue_estimate), 'cloud_queue_estimate': int(cloud_queue_estimate), 'cloud_transfer_delay_ms': round(float(cloud_transfer_delay), 3), 'cloud_return_delay_ms': round(float(cloud_return_delay), 3), 'cost_proxy': round(float(cost_proxy), 6), 'energy_proxy': round(float(energy_proxy), 6)}

def summarize_results(detailed_rows: List[Dict], edge_state: LayerState, cloud_state: LayerState, subset: pd.DataFrame, mode: str) -> Dict:
    results_df = pd.DataFrame(detailed_rows)
    total_tasks = int(len(results_df))
    completed_df = results_df[results_df['status'] == 'completed']
    blocked_df = results_df[results_df['status'] == 'blocked']
    completed_count = int(len(completed_df))
    blocked_count = int(len(blocked_df))
    max_finish = float(completed_df['finish_time_ms'].max()) if completed_count else 0.0
    last_arrival = float(subset['arrival_time_s'].max()) * 1000.0 if len(subset) else 0.0
    simulation_end_ms = max(max_finish, last_arrival, 1.0)
    duration_s = simulation_end_ms / 1000.0
    edge_capacity_time = len(edge_state.server_available_times) * simulation_end_ms
    cloud_capacity_time = len(cloud_state.server_available_times) * simulation_end_ms
    edge_utilization = edge_state.total_busy_time_ms / edge_capacity_time if edge_capacity_time > 0 else 0.0
    cloud_utilization = cloud_state.total_busy_time_ms / cloud_capacity_time if cloud_capacity_time > 0 else 0.0
    avg_edge_queue_length = compute_average_queue_length(edge_state.queue_events, simulation_end_ms)
    avg_cloud_queue_length = compute_average_queue_length(cloud_state.queue_events, simulation_end_ms)
    cloud_completed = completed_df[completed_df['processing_location'] == 'cloud']
    offloaded_count = int(len(cloud_completed))
    sla_violation_count = int(completed_df['sla_violation'].sum()) if completed_count else 0
    total_cost_proxy = float(completed_df['cost_proxy'].sum()) if completed_count and 'cost_proxy' in completed_df.columns else 0.0
    total_energy_proxy = float(completed_df['energy_proxy'].sum()) if completed_count and 'energy_proxy' in completed_df.columns else 0.0
    return {'scenario': str(subset['scenario'].iloc[0]), 'run_id': int(subset['run_id'].iloc[0]), 'seed': subset['seed'].iloc[0] if 'seed' in subset.columns else np.nan, 'mode': mode, 'total_tasks': total_tasks, 'completed_tasks': completed_count, 'blocked_tasks': blocked_count, 'sla_violations': sla_violation_count, 'throughput_tasks_per_s': completed_count / duration_s if duration_s > 0 else 0.0, 'avg_response_time_ms': float(completed_df['response_time_ms'].mean()) if completed_count else np.nan, 'avg_waiting_time_ms': float(completed_df['waiting_time_ms'].mean()) if completed_count else np.nan, 'edge_utilization': float(edge_utilization), 'cloud_utilization': float(cloud_utilization), 'avg_edge_queue_length': float(avg_edge_queue_length), 'avg_cloud_queue_length': float(avg_cloud_queue_length), 'blocking_probability': blocked_count / total_tasks if total_tasks else 0.0, 'cloud_offloading_ratio': offloaded_count / completed_count if completed_count else 0.0, 'sla_violation_rate': sla_violation_count / completed_count if completed_count else 0.0, 'avg_cost_proxy': total_cost_proxy / completed_count if completed_count else np.nan, 'total_cost_proxy': total_cost_proxy, 'avg_energy_proxy': total_energy_proxy / completed_count if completed_count else np.nan, 'total_energy_proxy': total_energy_proxy}

def simulate_edge_only(subset: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, Dict]:
    mode = 'edge_only'
    edge_state = make_empty_layer_state(int(params['edge_nodes']))
    cloud_state = make_empty_layer_state(int(params['cloud_servers']))
    edge_queue_capacity = int(params['edge_queue_capacity'])
    edge_service_rate = float(params['edge_service_rate_tps_each'])
    detailed_rows: List[Dict] = []
    subset = subset.sort_values('arrival_time_s').reset_index(drop=True)
    for task in subset.to_dict('records'):
        arrival_ms = float(task['arrival_time_s']) * 1000.0
        service_ms = service_time_ms(float(task['cpu_demand_mi']), edge_service_rate)
        sched = schedule_on_layer(edge_state, arrival_ms, service_ms, edge_queue_capacity)
        if not sched['accepted']:
            row = build_result_row(task, mode, 'blocked', 'blocked', None, None, None, None, None, sched['queue_length_at_arrival'], 0, 0.0, 0.0)
        else:
            finish_ms = sched['service_finish_time_ms']
            response_ms = finish_ms - arrival_ms
            row = build_result_row(task, mode, 'edge', 'completed', sched['start_time_ms'], finish_ms, service_ms, sched['waiting_time_ms'], response_ms, sched['queue_length_after_assignment'], 0, 0.0, 0.0)
        detailed_rows.append(row)
    summary = summarize_results(detailed_rows, edge_state, cloud_state, subset, mode)
    return (pd.DataFrame(detailed_rows), summary)

def simulate_cloud_only(subset: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, Dict]:
    mode = 'cloud_only'
    edge_state = make_empty_layer_state(int(params['edge_nodes']))
    cloud_state = make_empty_layer_state(int(params['cloud_servers']))
    cloud_queue_capacity = int(params['cloud_queue_capacity'])
    cloud_service_rate = float(params['cloud_service_rate_tps_each'])
    cloud_arrivals = []
    subset = subset.sort_values('arrival_time_s').reset_index(drop=True)
    for task in subset.to_dict('records'):
        source_arrival_ms = float(task['arrival_time_s']) * 1000.0
        network_ms = float(task['edge_cloud_network_delay_ms'])
        upload_ms = transfer_delay_ms(float(task['task_size_kb']), EDGE_CLOUD_BANDWIDTH_MBPS)
        cloud_arrival_ms = source_arrival_ms + network_ms + upload_ms
        cloud_arrivals.append({'task': task, 'source_arrival_ms': source_arrival_ms, 'cloud_arrival_ms': cloud_arrival_ms, 'upload_ms': upload_ms, 'return_ms': return_delay_ms(network_ms)})
    cloud_arrivals.sort(key=lambda x: x['cloud_arrival_ms'])
    detailed_rows: List[Dict] = []
    for item in cloud_arrivals:
        task = item['task']
        cloud_service_ms = service_time_ms(float(task['cpu_demand_mi']), cloud_service_rate)
        sched = schedule_on_layer(cloud_state, item['cloud_arrival_ms'], cloud_service_ms, cloud_queue_capacity)
        if not sched['accepted']:
            row = build_result_row(task, mode, 'blocked', 'blocked', None, None, None, None, None, 0, sched['queue_length_at_arrival'], item['upload_ms'], item['return_ms'])
        else:
            final_finish_ms = sched['service_finish_time_ms'] + item['return_ms']
            response_ms = final_finish_ms - item['source_arrival_ms']
            row = build_result_row(task, mode, 'cloud', 'completed', sched['start_time_ms'], final_finish_ms, cloud_service_ms, sched['waiting_time_ms'], response_ms, 0, sched['queue_length_after_assignment'], item['upload_ms'], item['return_ms'])
        detailed_rows.append(row)
    detailed_rows.sort(key=lambda r: r['arrival_time_s'])
    summary = summarize_results(detailed_rows, edge_state, cloud_state, subset, mode)
    return (pd.DataFrame(detailed_rows), summary)

def simulate_hybrid(subset: pd.DataFrame, params: Dict, threshold_ms: float=HYBRID_EDGE_DELAY_THRESHOLD_MS) -> Tuple[pd.DataFrame, Dict]:
    mode = 'hybrid_cloud_edge'
    edge_state = make_empty_layer_state(int(params['edge_nodes']))
    cloud_state = make_empty_layer_state(int(params['cloud_servers']))
    edge_queue_capacity = int(params['edge_queue_capacity'])
    cloud_queue_capacity = int(params['cloud_queue_capacity'])
    edge_service_rate = float(params['edge_service_rate_tps_each'])
    cloud_service_rate = float(params['cloud_service_rate_tps_each'])
    detailed_rows: List[Dict] = []
    cloud_candidates: List[Dict] = []
    subset = subset.sort_values('arrival_time_s').reset_index(drop=True)
    for task in subset.to_dict('records'):
        source_arrival_ms = float(task['arrival_time_s']) * 1000.0
        edge_queue_now = waiting_queue_length(edge_state, source_arrival_ms)
        estimated_edge_wait = estimate_waiting_delay_ms(edge_state, source_arrival_ms)
        edge_service_ms = service_time_ms(float(task['cpu_demand_mi']), edge_service_rate)
        estimated_edge_response = estimated_edge_wait + edge_service_ms
        network_ms = float(task['edge_cloud_network_delay_ms'])
        upload_ms = transfer_delay_ms(float(task['task_size_kb']), EDGE_CLOUD_BANDWIDTH_MBPS)
        return_ms = return_delay_ms(network_ms)
        cloud_arrival_ms = source_arrival_ms + network_ms + upload_ms
        cloud_service_ms = service_time_ms(float(task['cpu_demand_mi']), cloud_service_rate)
        estimated_cloud_wait = estimate_waiting_delay_ms(cloud_state, cloud_arrival_ms)
        estimated_cloud_response = network_ms + upload_ms + estimated_cloud_wait + cloud_service_ms + return_ms
        use_edge = edge_queue_now < edge_queue_capacity and estimated_edge_wait <= float(threshold_ms) and (estimated_edge_response <= estimated_cloud_response)
        if use_edge:
            sched = schedule_on_layer(edge_state, source_arrival_ms, edge_service_ms, edge_queue_capacity)
            if sched['accepted']:
                finish_ms = sched['service_finish_time_ms']
                response_ms = finish_ms - source_arrival_ms
                row = build_result_row(task, mode, 'edge', 'completed', sched['start_time_ms'], finish_ms, edge_service_ms, sched['waiting_time_ms'], response_ms, sched['queue_length_after_assignment'], 0, 0.0, 0.0)
                detailed_rows.append(row)
                continue
        cloud_candidates.append({'task': task, 'source_arrival_ms': source_arrival_ms, 'cloud_arrival_ms': cloud_arrival_ms, 'cloud_service_ms': cloud_service_ms, 'upload_ms': upload_ms, 'return_ms': return_ms, 'edge_queue_at_decision': edge_queue_now})
    cloud_candidates.sort(key=lambda x: x['cloud_arrival_ms'])
    for item in cloud_candidates:
        task = item['task']
        sched = schedule_on_layer(cloud_state, item['cloud_arrival_ms'], item['cloud_service_ms'], cloud_queue_capacity)
        if not sched['accepted']:
            row = build_result_row(task, mode, 'blocked', 'blocked', None, None, None, None, None, item['edge_queue_at_decision'], sched['queue_length_at_arrival'], item['upload_ms'], item['return_ms'])
        else:
            final_finish_ms = sched['service_finish_time_ms'] + item['return_ms']
            response_ms = final_finish_ms - item['source_arrival_ms']
            row = build_result_row(task, mode, 'cloud', 'completed', sched['start_time_ms'], final_finish_ms, item['cloud_service_ms'], sched['waiting_time_ms'], response_ms, item['edge_queue_at_decision'], sched['queue_length_after_assignment'], item['upload_ms'], item['return_ms'])
        detailed_rows.append(row)
    detailed_rows.sort(key=lambda r: r['arrival_time_s'])
    summary = summarize_results(detailed_rows, edge_state, cloud_state, subset, mode)
    return (pd.DataFrame(detailed_rows), summary)

def simulate_deadline_aware_hybrid(subset: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Baseline hybrid routing with deadline-aware override and cost/energy tie-break."""
    mode = 'deadline_aware_hybrid'
    edge_state = make_empty_layer_state(int(params['edge_nodes']))
    cloud_state = make_empty_layer_state(int(params['cloud_servers']))
    edge_queue_capacity = int(params['edge_queue_capacity'])
    cloud_queue_capacity = int(params['cloud_queue_capacity'])
    edge_service_rate = float(params['edge_service_rate_tps_each'])
    cloud_service_rate = float(params['cloud_service_rate_tps_each'])
    detailed_rows: List[Dict] = []
    cloud_candidates: List[Dict] = []
    subset = subset.sort_values('arrival_time_s').reset_index(drop=True)

    for task in subset.to_dict('records'):
        source_arrival_ms = float(task['arrival_time_s']) * 1000.0
        deadline_ms = float(task.get('deadline_ms', np.inf))

        edge_queue_now = waiting_queue_length(edge_state, source_arrival_ms)
        estimated_edge_wait = estimate_waiting_delay_ms(edge_state, source_arrival_ms)
        edge_service_ms = service_time_ms(float(task['cpu_demand_mi']), edge_service_rate)
        estimated_edge_response = estimated_edge_wait + edge_service_ms
        edge_available = edge_queue_now < edge_queue_capacity
        edge_feasible = edge_available and estimated_edge_response <= deadline_ms

        network_ms = float(task['edge_cloud_network_delay_ms'])
        upload_ms = transfer_delay_ms(float(task['task_size_kb']), EDGE_CLOUD_BANDWIDTH_MBPS)
        return_ms = return_delay_ms(network_ms)
        cloud_arrival_ms = source_arrival_ms + network_ms + upload_ms
        cloud_queue_now = waiting_queue_length(cloud_state, cloud_arrival_ms)
        estimated_cloud_wait = estimate_waiting_delay_ms(cloud_state, cloud_arrival_ms)
        cloud_service_ms = service_time_ms(float(task['cpu_demand_mi']), cloud_service_rate)
        estimated_cloud_response = network_ms + upload_ms + estimated_cloud_wait + cloud_service_ms + return_ms
        cloud_available = cloud_queue_now < cloud_queue_capacity
        cloud_feasible = cloud_available and estimated_cloud_response <= deadline_ms

        # Start from the original hybrid rule so that the enhanced mode remains
        # directly comparable with the baseline hybrid policy.
        baseline_use_edge = (
            edge_available
            and estimated_edge_wait <= HYBRID_EDGE_DELAY_THRESHOLD_MS
            and estimated_edge_response <= estimated_cloud_response
        )
        if baseline_use_edge:
            selected = 'edge'
        elif cloud_available:
            selected = 'cloud'
        elif edge_available:
            selected = 'edge'
        else:
            selected = 'blocked'

        # Deadline-aware override: switch only when the selected layer is
        # predicted to miss the deadline and the alternate layer is predicted to
        # meet it.
        if selected == 'edge' and (not edge_feasible) and cloud_feasible:
            selected = 'cloud'
        elif selected == 'cloud' and (not cloud_feasible) and edge_feasible:
            selected = 'edge'
        elif edge_feasible and cloud_feasible:
            # Cost/energy is a tie-break, not the main objective. Apply it only
            # when both estimates are safely within deadline and nearly equal.
            response_gap_ms = abs(estimated_edge_response - estimated_cloud_response)
            best_response = min(estimated_edge_response, estimated_cloud_response)
            deadline_slack = deadline_ms - best_response
            if response_gap_ms <= 25.0 and deadline_slack >= 100.0:
                edge_cost, edge_energy = cost_energy_proxy('edge', edge_service_ms, float(task['task_size_kb']))
                cloud_cost, cloud_energy = cost_energy_proxy('cloud', cloud_service_ms, float(task['task_size_kb']))
                selected = 'edge' if (edge_cost + edge_energy) <= (cloud_cost + cloud_energy) else 'cloud'
        elif not edge_feasible and not cloud_feasible and selected != 'blocked':
            # If both layers are predicted to miss the deadline, choose the
            # smaller predicted lateness when both are available.
            edge_lateness = max(0.0, estimated_edge_response - deadline_ms) if edge_available else np.inf
            cloud_lateness = max(0.0, estimated_cloud_response - deadline_ms) if cloud_available else np.inf
            if cloud_lateness < edge_lateness:
                selected = 'cloud'
            elif edge_lateness < cloud_lateness:
                selected = 'edge'

        if selected == 'edge' and edge_available:
            sched = schedule_on_layer(edge_state, source_arrival_ms, edge_service_ms, edge_queue_capacity)
            if sched['accepted']:
                finish_ms = sched['service_finish_time_ms']
                response_ms = finish_ms - source_arrival_ms
                detailed_rows.append(build_result_row(task, mode, 'edge', 'completed', sched['start_time_ms'], finish_ms, edge_service_ms, sched['waiting_time_ms'], response_ms, sched['queue_length_after_assignment'], cloud_queue_now, 0.0, 0.0))
                continue

        if selected == 'cloud' and cloud_available:
            cloud_candidates.append({'task': task, 'source_arrival_ms': source_arrival_ms, 'cloud_arrival_ms': cloud_arrival_ms, 'cloud_service_ms': cloud_service_ms, 'upload_ms': upload_ms, 'return_ms': return_ms, 'edge_queue_at_decision': edge_queue_now})
        elif edge_available:
            # Cloud is not available, so fall back to edge if possible.
            sched = schedule_on_layer(edge_state, source_arrival_ms, edge_service_ms, edge_queue_capacity)
            if sched['accepted']:
                finish_ms = sched['service_finish_time_ms']
                response_ms = finish_ms - source_arrival_ms
                detailed_rows.append(build_result_row(task, mode, 'edge', 'completed', sched['start_time_ms'], finish_ms, edge_service_ms, sched['waiting_time_ms'], response_ms, sched['queue_length_after_assignment'], cloud_queue_now, 0.0, 0.0))
            else:
                detailed_rows.append(build_result_row(task, mode, 'blocked', 'blocked', None, None, None, None, None, sched['queue_length_at_arrival'], cloud_queue_now, 0.0, 0.0))
        else:
            detailed_rows.append(build_result_row(task, mode, 'blocked', 'blocked', None, None, None, None, None, edge_queue_now, cloud_queue_now, upload_ms, return_ms))

    cloud_candidates.sort(key=lambda x: x['cloud_arrival_ms'])
    for item in cloud_candidates:
        task = item['task']
        sched = schedule_on_layer(cloud_state, item['cloud_arrival_ms'], item['cloud_service_ms'], cloud_queue_capacity)
        if not sched['accepted']:
            row = build_result_row(task, mode, 'blocked', 'blocked', None, None, None, None, None, item['edge_queue_at_decision'], sched['queue_length_at_arrival'], item['upload_ms'], item['return_ms'])
        else:
            final_finish_ms = sched['service_finish_time_ms'] + item['return_ms']
            response_ms = final_finish_ms - item['source_arrival_ms']
            row = build_result_row(task, mode, 'cloud', 'completed', sched['start_time_ms'], final_finish_ms, item['cloud_service_ms'], sched['waiting_time_ms'], response_ms, item['edge_queue_at_decision'], sched['queue_length_after_assignment'], item['upload_ms'], item['return_ms'])
        detailed_rows.append(row)

    detailed_rows.sort(key=lambda r: r['arrival_time_s'])
    summary = summarize_results(detailed_rows, edge_state, cloud_state, subset, mode)
    return (pd.DataFrame(detailed_rows), summary)

def simulate_subset(subset: pd.DataFrame, params: Dict, mode: str, threshold_ms: float=HYBRID_EDGE_DELAY_THRESHOLD_MS) -> Tuple[pd.DataFrame, Dict]:
    if mode == 'edge_only':
        return simulate_edge_only(subset, params)
    if mode == 'cloud_only':
        return simulate_cloud_only(subset, params)
    if mode == 'hybrid_cloud_edge':
        return simulate_hybrid(subset, params, threshold_ms=threshold_ms)
    if mode == 'deadline_aware_hybrid':
        return simulate_deadline_aware_hybrid(subset, params)
    raise ValueError(f'Invalid processing mode: {mode}')

def run_controlled_simulations(tasks_df: pd.DataFrame, params_df: pd.DataFrame, include_detailed: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_detailed_results = []
    all_summaries = []
    scenarios = sorted(tasks_df['scenario'].unique(), key=scenario_sort_key)
    for scenario in scenarios:
        params = get_scenario_params(params_df, scenario)
        scenario_df = tasks_df[tasks_df['scenario'] == scenario]
        for run_id in sorted(scenario_df['run_id'].unique()):
            subset = scenario_df[scenario_df['run_id'] == run_id]
            for mode in PROCESSING_MODES:
                print(f'Controlled: scenario={scenario}, run={run_id}, mode={mode}')
                detailed, summary = simulate_subset(subset, params, mode)
                if include_detailed:
                    all_detailed_results.append(detailed)
                all_summaries.append(summary)
    detailed_df = pd.concat(all_detailed_results, ignore_index=True) if all_detailed_results else pd.DataFrame()
    summary_df = pd.DataFrame(all_summaries)
    return (detailed_df, summary_df)

def t_critical_975(df: int) -> float:
    if df <= 0:
        return np.nan
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(0.975, df))
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.16, 14: 2.145, 15: 2.131, 16: 2.12, 17: 2.11, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.08, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.06, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    if df in table:
        return table[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.0
    return 1.96

def aggregate_with_ci(summary_df: pd.DataFrame, group_cols: List[str], metric_cols: Optional[List[str]]=None) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = ['throughput_tasks_per_s', 'avg_response_time_ms', 'avg_waiting_time_ms', 'edge_utilization', 'cloud_utilization', 'avg_edge_queue_length', 'avg_cloud_queue_length', 'blocking_probability', 'cloud_offloading_ratio', 'sla_violation_rate', 'avg_cost_proxy', 'avg_energy_proxy']
    rows = []
    for group_values, group in summary_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_cols, group_values))
        row['runs'] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            n = int(len(values))
            mean = float(values.mean()) if n else np.nan
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            tcrit = t_critical_975(n - 1) if n > 1 else np.nan
            ci = float(tcrit * std / np.sqrt(n)) if n > 1 else 0.0
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_ci95'] = ci
            row[f'{metric}_ci95_low'] = mean - ci if np.isfinite(mean) else np.nan
            row[f'{metric}_ci95_high'] = mean + ci if np.isfinite(mean) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return sort_result_frame(out, group_cols)

def fmt_mean_ci(mean: float, ci: float, multiplier: float=1.0, decimals: int=2) -> str:
    if not np.isfinite(mean):
        return 'NA'
    value = mean * multiplier
    ci_value = ci * multiplier
    return f'{value:.{decimals}f} ± {ci_value:.{decimals}f}'

def build_paper_results_table_ci(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in agg_df.iterrows():
        rows.append({'scenario': str(row['scenario']), 'mode': str(row['mode']), 'runs': int(row['runs']), 'response_ms_mean_ci': fmt_mean_ci(row['avg_response_time_ms_mean'], row['avg_response_time_ms_ci95']), 'wait_ms_mean_ci': fmt_mean_ci(row['avg_waiting_time_ms_mean'], row['avg_waiting_time_ms_ci95']), 'throughput_mean_ci': fmt_mean_ci(row['throughput_tasks_per_s_mean'], row['throughput_tasks_per_s_ci95']), 'edge_util_%_mean_ci': fmt_mean_ci(row['edge_utilization_mean'], row['edge_utilization_ci95'], multiplier=100), 'cloud_util_%_mean_ci': fmt_mean_ci(row['cloud_utilization_mean'], row['cloud_utilization_ci95'], multiplier=100), 'edge_queue_mean_ci': fmt_mean_ci(row['avg_edge_queue_length_mean'], row['avg_edge_queue_length_ci95']), 'cloud_queue_mean_ci': fmt_mean_ci(row['avg_cloud_queue_length_mean'], row['avg_cloud_queue_length_ci95']), 'block_%_mean_ci': fmt_mean_ci(row['blocking_probability_mean'], row['blocking_probability_ci95'], multiplier=100), 'cloud_offload_%_mean_ci': fmt_mean_ci(row['cloud_offloading_ratio_mean'], row['cloud_offloading_ratio_ci95'], multiplier=100), 'sla_violation_%_mean_ci': fmt_mean_ci(row['sla_violation_rate_mean'], row['sla_violation_rate_ci95'], multiplier=100), 'cost_proxy_mean_ci': fmt_mean_ci(row['avg_cost_proxy_mean'], row['avg_cost_proxy_ci95'], decimals=5), 'energy_proxy_mean_ci': fmt_mean_ci(row['avg_energy_proxy_mean'], row['avg_energy_proxy_ci95'], decimals=4)})
    return pd.DataFrame(rows)

def run_statistical_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ['avg_response_time_ms', 'avg_waiting_time_ms', 'throughput_tasks_per_s', 'blocking_probability', 'cloud_offloading_ratio', 'sla_violation_rate', 'avg_cost_proxy', 'avg_energy_proxy']
    rows = []
    if scipy_stats is None:
        for scenario in sorted(summary_df['scenario'].unique(), key=scenario_sort_key):
            for metric in metrics:
                rows.append({'scenario': scenario, 'metric': metric, 'test': 'not_run', 'statistic': np.nan, 'p_value': np.nan, 'note': 'scipy is not installed; install scipy to run ANOVA/Kruskal-Wallis.'})
        return pd.DataFrame(rows)
    for scenario in sorted(summary_df['scenario'].unique(), key=scenario_sort_key):
        scenario_df = summary_df[summary_df['scenario'] == scenario]
        for metric in metrics:
            groups = []
            labels = []
            for mode in PROCESSING_MODES:
                vals = pd.to_numeric(scenario_df[scenario_df['mode'] == mode][metric], errors='coerce').dropna().values
                if len(vals) >= 2:
                    groups.append(vals)
                    labels.append(mode)
            if len(groups) < 2:
                continue
            try:
                anova_stat, anova_p = scipy_stats.f_oneway(*groups)
            except Exception:
                anova_stat, anova_p = (np.nan, np.nan)
            try:
                kw_stat, kw_p = scipy_stats.kruskal(*groups)
            except Exception:
                kw_stat, kw_p = (np.nan, np.nan)
            rows.append({'scenario': scenario, 'metric': metric, 'test': 'one_way_anova', 'statistic': float(anova_stat) if np.isfinite(anova_stat) else np.nan, 'p_value': float(anova_p) if np.isfinite(anova_p) else np.nan, 'groups': ';'.join(labels), 'note': 'Use Kruskal-Wallis if normality/variance assumptions are not satisfied.'})
            rows.append({'scenario': scenario, 'metric': metric, 'test': 'kruskal_wallis', 'statistic': float(kw_stat) if np.isfinite(kw_stat) else np.nan, 'p_value': float(kw_p) if np.isfinite(kw_p) else np.nan, 'groups': ';'.join(labels), 'note': 'Non-parametric comparison across modes.'})
    return pd.DataFrame(rows)

def override_params(params: Dict, key: str, value) -> Dict:
    new_params = dict(params)
    new_params[key] = value
    return new_params

def run_sensitivity_analysis(tasks_df: pd.DataFrame, params_df: pd.DataFrame, include_deadline_aware: bool=False) -> pd.DataFrame:
    """Run sensitivity analysis for the hybrid policies.

    Threshold sensitivity is specific to the baseline hybrid rule. Resource and
    network sensitivity are evaluated for the baseline hybrid mode by default
    to keep runtime manageable. Set include_deadline_aware=True to also include
    the deadline-aware hybrid mode in capacity and network sensitivity runs.
    """
    rows = []
    available_scenarios = set(tasks_df['scenario'].unique())
    scenarios = [s for s in SENSITIVITY_SCENARIOS if s in available_scenarios]
    capacity_modes = ['hybrid_cloud_edge']
    if include_deadline_aware:
        capacity_modes.append('deadline_aware_hybrid')
    for scenario in scenarios:
        base_params = get_scenario_params(params_df, scenario)
        scenario_df = tasks_df[tasks_df['scenario'] == scenario]
        for run_id in sorted(scenario_df['run_id'].unique()):
            subset_base = scenario_df[scenario_df['run_id'] == run_id].copy()
            for theta in THRESHOLD_VALUES_MS:
                _, summary = simulate_hybrid(subset_base, base_params, threshold_ms=float(theta))
                summary.update({'experiment_type': 'threshold_ms', 'parameter_value': theta})
                rows.append(summary)
            for edge_servers in EDGE_SERVER_VALUES:
                params = override_params(base_params, 'edge_nodes', int(edge_servers))
                for mode in capacity_modes:
                    _, summary = simulate_subset(subset_base, params, mode)
                    summary.update({'experiment_type': 'edge_servers', 'parameter_value': edge_servers})
                    rows.append(summary)
            for cloud_servers in CLOUD_SERVER_VALUES:
                params = override_params(base_params, 'cloud_servers', int(cloud_servers))
                for mode in capacity_modes:
                    _, summary = simulate_subset(subset_base, params, mode)
                    summary.update({'experiment_type': 'cloud_servers', 'parameter_value': cloud_servers})
                    rows.append(summary)
            for edge_queue in EDGE_QUEUE_VALUES:
                params = override_params(base_params, 'edge_queue_capacity', int(edge_queue))
                for mode in capacity_modes:
                    _, summary = simulate_subset(subset_base, params, mode)
                    summary.update({'experiment_type': 'edge_queue_capacity', 'parameter_value': edge_queue})
                    rows.append(summary)
            for multiplier in NETWORK_DELAY_MULTIPLIERS:
                subset = subset_base.copy()
                subset['edge_cloud_network_delay_ms'] = subset['edge_cloud_network_delay_ms'] * float(multiplier)
                for mode in capacity_modes:
                    _, summary = simulate_subset(subset, base_params, mode)
                    summary.update({'experiment_type': 'network_delay_multiplier', 'parameter_value': multiplier})
                    rows.append(summary)
    return pd.DataFrame(rows)

def plot_metric_grouped(agg_df: pd.DataFrame, metric_mean_col: str, ylabel: str, output_path: Path, metric_ci_col: Optional[str]=None) -> None:
    plot_df = sort_result_frame(agg_df.copy(), ['scenario', 'mode'])
    scenarios_present = set(plot_df['scenario'].astype(str))
    modes_present = set(plot_df['mode'].astype(str))
    scenarios = [s for s in SCENARIO_ORDER if s in scenarios_present] + sorted(s for s in scenarios_present if s not in SCENARIO_ORDER)
    modes = [m for m in PROCESSING_MODES if m in modes_present] + sorted(m for m in modes_present if m not in PROCESSING_MODES)
    x = np.arange(len(scenarios))
    width = min(0.18, 0.8 / max(len(modes), 1))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for idx, mode in enumerate(modes):
        vals = []
        errs = []
        for scenario in scenarios:
            row = plot_df[(plot_df['scenario'].astype(str) == scenario) & (plot_df['mode'].astype(str) == mode)]
            vals.append(float(row[metric_mean_col].iloc[0]) if len(row) else np.nan)
            if metric_ci_col:
                errs.append(float(row[metric_ci_col].iloc[0]) if len(row) else 0.0)
        offset = (idx - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, vals, width, yerr=errs if metric_ci_col else None, capsize=3, label=mode)
    ax.set_xlabel('Workload scenario')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def generate_controlled_graphs(agg_df: pd.DataFrame, output_dir: Path) -> None:
    plot_metric_grouped(agg_df, 'avg_response_time_ms_mean', 'Average response time (ms)', output_dir / 'fig_response_time_with_ci.png', 'avg_response_time_ms_ci95')
    plot_metric_grouped(agg_df, 'throughput_tasks_per_s_mean', 'Throughput (tasks/s)', output_dir / 'fig_throughput_with_ci.png', 'throughput_tasks_per_s_ci95')
    plot_metric_grouped(agg_df, 'blocking_probability_mean', 'Blocking probability', output_dir / 'fig_blocking_probability_with_ci.png', 'blocking_probability_ci95')
    plot_metric_grouped(agg_df, 'cloud_offloading_ratio_mean', 'Cloud offloading ratio', output_dir / 'fig_cloud_offloading_ratio_with_ci.png', 'cloud_offloading_ratio_ci95')
    plot_metric_grouped(agg_df, 'sla_violation_rate_mean', 'SLA violation rate', output_dir / 'fig_sla_violation_with_ci.png', 'sla_violation_rate_ci95')
    plot_metric_grouped(agg_df, 'avg_cost_proxy_mean', 'Average cost proxy', output_dir / 'fig_cost_proxy_with_ci.png', 'avg_cost_proxy_ci95')
    plot_metric_grouped(agg_df, 'avg_energy_proxy_mean', 'Average energy proxy', output_dir / 'fig_energy_proxy_with_ci.png', 'avg_energy_proxy_ci95')

def write_method_note(output_dir: Path, args: argparse.Namespace) -> None:
  
.strip()
    (output_dir / 'simulation_method_note.txt').write_text(note, encoding='utf-8')
    config = {'input_dir': str(args.input_dir), 'output_dir': str(args.output_dir), 'baseline_hybrid_threshold_ms': HYBRID_EDGE_DELAY_THRESHOLD_MS, 'service_time': {'min_service_time_ms': MIN_SERVICE_TIME_MS, 'nominal_cpu_demand_mi': NOMINAL_CPU_DEMAND_MI}, 'cloud_communication': {'edge_cloud_bandwidth_mbps': EDGE_CLOUD_BANDWIDTH_MBPS, 'result_return_delay_factor': RESULT_RETURN_DELAY_FACTOR, 'min_return_delay_ms': MIN_RETURN_DELAY_MS}, 'cost_energy_proxy': {'edge_cost_per_service_ms': EDGE_COST_PER_SERVICE_MS, 'cloud_cost_per_service_ms': CLOUD_COST_PER_SERVICE_MS, 'network_cost_per_mb': NETWORK_COST_PER_MB, 'edge_energy_per_service_ms': EDGE_ENERGY_PER_SERVICE_MS, 'cloud_energy_per_service_ms': CLOUD_ENERGY_PER_SERVICE_MS, 'network_energy_per_mb': NETWORK_ENERGY_PER_MB}, 'sensitivity_enabled': bool(args.sensitivity), 'sensitivity_scenarios': SENSITIVITY_SCENARIOS, 'sensitivity_capacity_modes': ['hybrid_cloud_edge', 'deadline_aware_hybrid'] if getattr(args, 'sensitivity_all_modes', False) else ['hybrid_cloud_edge'], 'threshold_values_ms': THRESHOLD_VALUES_MS, 'edge_server_values': EDGE_SERVER_VALUES, 'cloud_server_values': CLOUD_SERVER_VALUES, 'edge_queue_values': EDGE_QUEUE_VALUES, 'network_delay_multipliers': NETWORK_DELAY_MULTIPLIERS, 'scipy_available_for_stats': scipy_stats is not None}
    with open(output_dir / 'experiment_configuration.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run cloud-edge performance simulations.')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR, help='Generated dataset folder.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output folder.')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis after controlled simulations.')
    parser.add_argument('--no-detailed', action='store_true', help='Do not write detailed task-level results to save disk space.')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation.')
    parser.add_argument('--sensitivity-all-modes', action='store_true', help='Include deadline-aware hybrid in capacity/network sensitivity. This is slower.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_df, params_df = load_inputs(args.input_dir)
    print('Input files loaded successfully.')
    print(f'Total workload records: {len(tasks_df):,}')
    print(f"Scenarios: {', '.join(sorted(tasks_df['scenario'].unique(), key=scenario_sort_key))}")
    print(f"Runs detected: {len(sorted(tasks_df['run_id'].unique()))}")
    print()
    detailed_df, controlled_summary_df = run_controlled_simulations(tasks_df, params_df, include_detailed=not args.no_detailed)
    controlled_agg_df = aggregate_with_ci(controlled_summary_df, ['scenario', 'mode'])
    paper_table_df = build_paper_results_table_ci(controlled_agg_df)
    stats_df = run_statistical_tests(controlled_summary_df)
    controlled_summary_df.to_csv(args.output_dir / 'controlled_results_30runs.csv', index=False)
    controlled_agg_df.to_csv(args.output_dir / 'controlled_summary_with_ci.csv', index=False)
    paper_table_df.to_csv(args.output_dir / 'paper_results_table_with_ci.csv', index=False)
    stats_df.to_csv(args.output_dir / 'statistical_tests.csv', index=False)
    if not args.no_detailed and (not detailed_df.empty):
        detailed_df.to_csv(args.output_dir / 'detailed_task_results_.csv', index=False)
    if not args.no_plots:
        generate_controlled_graphs(controlled_agg_df, args.output_dir)
    if args.sensitivity:
        print()
        print('Running sensitivity analysis...')
        sensitivity_df = run_sensitivity_analysis(tasks_df, params_df, include_deadline_aware=args.sensitivity_all_modes)
        sensitivity_df.to_csv(args.output_dir / 'sensitivity_results_30runs.csv', index=False)
        sensitivity_agg_df = aggregate_with_ci(sensitivity_df, ['experiment_type', 'parameter_value', 'scenario', 'mode'])
        sensitivity_agg_df.to_csv(args.output_dir / 'sensitivity_summary_with_ci.csv', index=False)
    write_method_note(args.output_dir, args)
    print()
    print('Simulation completed successfully.')
    print(f'Output folder: {args.output_dir.resolve()}')
    print()
    print('Paper-ready results table:')
    print(paper_table_df.to_string(index=False))
if __name__ == '__main__':
    main()
