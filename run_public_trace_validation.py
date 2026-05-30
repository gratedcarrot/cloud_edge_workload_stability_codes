from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
INPUT_DIR = Path('public_trace_data')
OUTPUT_DIR = Path('public_trace_results')
TASK_FILE = INPUT_DIR / 'cloud_edge_workload_trace.csv'
PARAM_FILE = INPUT_DIR / 'scenario_parameters.csv'
PROCESSING_MODES = ['edge_only', 'cloud_only', 'hybrid_cloud_edge']
MODE_LABELS = {'edge_only': 'Edge-only', 'cloud_only': 'Cloud-only', 'hybrid_cloud_edge': 'Hybrid cloud-edge'}
HYBRID_EDGE_DELAY_THRESHOLD_MS = 220.0
MIN_SERVICE_TIME_MS = 5.0
NOMINAL_CPU_DEMAND_MI = 100.0
EDGE_CLOUD_BANDWIDTH_MBPS = 25.0
RESULT_RETURN_DELAY_FACTOR = 0.65
MIN_RETURN_DELAY_MS = 5.0

@dataclass
class LayerState:
    server_available_times: List[float]
    scheduled_start_times: List[float] = field(default_factory=list)
    queue_events: List[Tuple[float, int]] = field(default_factory=list)
    total_busy_time_ms: float = 0.0

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not TASK_FILE.exists():
        raise FileNotFoundError(f'Missing {TASK_FILE}. Run build_public_trace_data.py first.')
    if not PARAM_FILE.exists():
        raise FileNotFoundError(f'Missing {PARAM_FILE}. Run build_public_trace_data.py first.')
    tasks_df = pd.read_csv(TASK_FILE)
    params_df = pd.read_csv(PARAM_FILE)
    return (tasks_df, params_df)

def get_scenario_params(params_df: pd.DataFrame, scenario: str) -> Dict:
    row = params_df[params_df['scenario'] == scenario]
    if row.empty:
        raise ValueError(f'No parameters found for scenario: {scenario}')
    return row.iloc[0].to_dict()

def make_empty_layer_state(server_count: int) -> LayerState:
    return LayerState(server_available_times=[0.0 for _ in range(server_count)])

def service_time_ms(cpu_demand_mi: float, service_rate_tps_each: float) -> float:
    base_service_time_ms = 1000.0 / max(service_rate_tps_each, 0.001)
    demand_multiplier = max(cpu_demand_mi, 1.0) / NOMINAL_CPU_DEMAND_MI
    return max(MIN_SERVICE_TIME_MS, base_service_time_ms * demand_multiplier)

def transfer_delay_ms(task_size_kb: float, bandwidth_mbps: float) -> float:
    kb_per_ms = max(bandwidth_mbps * 0.125, 0.001)
    return max(0.0, task_size_kb / kb_per_ms)

def return_delay_ms(network_delay_ms: float) -> float:
    return max(MIN_RETURN_DELAY_MS, network_delay_ms * RESULT_RETURN_DELAY_FACTOR)

def waiting_queue_length(state: LayerState, time_ms: float) -> int:
    return sum((1 for start_time in state.scheduled_start_times if start_time > time_ms))

def choose_earliest_server(server_available_times: List[float]) -> int:
    return int(np.argmin(server_available_times))

def estimate_waiting_delay_ms(state: LayerState, arrival_time_ms: float) -> float:
    return max(0.0, min(state.server_available_times) - arrival_time_ms)

def schedule_on_layer(state: LayerState, arrival_to_layer_ms: float, service_ms: float, queue_capacity: int) -> Dict:
    queue_at_arrival = waiting_queue_length(state, arrival_to_layer_ms)
    if queue_at_arrival >= queue_capacity:
        return {'accepted': False, 'start_time_ms': None, 'service_finish_time_ms': None, 'waiting_time_ms': None, 'queue_length_at_arrival': queue_at_arrival, 'queue_length_after_assignment': queue_at_arrival}
    server_idx = choose_earliest_server(state.server_available_times)
    start_time_ms = max(arrival_to_layer_ms, state.server_available_times[server_idx])
    waiting_time_ms = start_time_ms - arrival_to_layer_ms
    service_finish_time_ms = start_time_ms + service_ms
    state.server_available_times[server_idx] = service_finish_time_ms
    state.scheduled_start_times.append(start_time_ms)
    state.total_busy_time_ms += service_ms
    if waiting_time_ms > 0:
        state.queue_events.append((arrival_to_layer_ms, +1))
        state.queue_events.append((start_time_ms, -1))
    queue_after = waiting_queue_length(state, arrival_to_layer_ms)
    return {'accepted': True, 'start_time_ms': start_time_ms, 'service_finish_time_ms': service_finish_time_ms, 'waiting_time_ms': waiting_time_ms, 'queue_length_at_arrival': queue_at_arrival, 'queue_length_after_assignment': queue_after}

def compute_average_queue_length(queue_events: List[Tuple[float, int]], simulation_end_ms: float) -> float:
    if not queue_events or simulation_end_ms <= 0:
        return 0.0
    events = sorted(queue_events, key=lambda x: x[0])
    current_q = 0
    previous_time = 0.0
    area = 0.0
    for event_time, delta in events:
        event_time = max(0.0, min(event_time, simulation_end_ms))
        area += current_q * max(0.0, event_time - previous_time)
        current_q += delta
        current_q = max(0, current_q)
        previous_time = event_time
    area += current_q * max(0.0, simulation_end_ms - previous_time)
    return area / simulation_end_ms

def row_result(task, mode, location, status, start, finish, service, wait, response, edge_q, cloud_q, transfer, ret):
    return {'task_id': task['task_id'], 'scenario': task['scenario'], 'run_id': int(task['run_id']), 'mode': mode, 'arrival_time_s': float(task['arrival_time_s']), 'processing_location': location, 'status': status, 'start_time_ms': round(start, 3) if start is not None else None, 'finish_time_ms': round(finish, 3) if finish is not None else None, 'service_time_ms': round(service, 3) if service is not None else None, 'waiting_time_ms': round(wait, 3) if wait is not None else None, 'response_time_ms': round(response, 3) if response is not None else None, 'edge_queue_estimate': edge_q, 'cloud_queue_estimate': cloud_q, 'cloud_transfer_delay_ms': round(transfer, 3), 'cloud_return_delay_ms': round(ret, 3)}

def summarize(rows, edge_state, cloud_state, subset, mode):
    df = pd.DataFrame(rows)
    completed = df[df['status'] == 'completed']
    blocked = df[df['status'] == 'blocked']
    completed_count = len(completed)
    blocked_count = len(blocked)
    total_tasks = len(df)
    max_finish = completed['finish_time_ms'].max() if completed_count else 0.0
    last_arrival = float(subset['arrival_time_s'].max()) * 1000.0
    sim_end = max(float(max_finish), last_arrival, 1.0)
    duration_s = sim_end / 1000.0
    edge_capacity_time = len(edge_state.server_available_times) * sim_end
    cloud_capacity_time = len(cloud_state.server_available_times) * sim_end
    edge_util = edge_state.total_busy_time_ms / edge_capacity_time if edge_capacity_time else 0.0
    cloud_util = cloud_state.total_busy_time_ms / cloud_capacity_time if cloud_capacity_time else 0.0
    edge_q = compute_average_queue_length(edge_state.queue_events, sim_end)
    cloud_q = compute_average_queue_length(cloud_state.queue_events, sim_end)
    cloud_completed = completed[completed['processing_location'] == 'cloud']
    return {'scenario': subset['scenario'].iloc[0], 'mode': mode, 'total_tasks': total_tasks, 'completed_tasks': completed_count, 'blocked_tasks': blocked_count, 'throughput_tasks_per_s': completed_count / duration_s if duration_s else 0.0, 'avg_response_time_ms': float(completed['response_time_ms'].mean()) if completed_count else np.nan, 'avg_waiting_time_ms': float(completed['waiting_time_ms'].mean()) if completed_count else np.nan, 'edge_utilization': edge_util, 'cloud_utilization': cloud_util, 'avg_edge_queue_length': edge_q, 'avg_cloud_queue_length': cloud_q, 'blocking_probability': blocked_count / total_tasks if total_tasks else 0.0, 'cloud_offloading_ratio': len(cloud_completed) / completed_count if completed_count else 0.0}

def simulate_mode(subset, params, mode):
    edge_state = make_empty_layer_state(int(params['edge_nodes']))
    cloud_state = make_empty_layer_state(int(params['cloud_servers']))
    edge_capacity = int(params['edge_queue_capacity'])
    cloud_capacity = int(params['cloud_queue_capacity'])
    edge_rate = float(params['edge_service_rate_tps_each'])
    cloud_rate = float(params['cloud_service_rate_tps_each'])
    subset = subset.sort_values('arrival_time_s').reset_index(drop=True)
    rows = []
    if mode == 'edge_only':
        for _, task in subset.iterrows():
            arrival = float(task['arrival_time_s']) * 1000.0
            service = service_time_ms(float(task['cpu_demand_mi']), edge_rate)
            sched = schedule_on_layer(edge_state, arrival, service, edge_capacity)
            if not sched['accepted']:
                rows.append(row_result(task, mode, 'blocked', 'blocked', None, None, None, None, None, sched['queue_length_at_arrival'], 0, 0.0, 0.0))
            else:
                finish = sched['service_finish_time_ms']
                rows.append(row_result(task, mode, 'edge', 'completed', sched['start_time_ms'], finish, service, sched['waiting_time_ms'], finish - arrival, sched['queue_length_after_assignment'], 0, 0.0, 0.0))
    elif mode == 'cloud_only':
        cloud_items = []
        for _, task in subset.iterrows():
            source_arrival = float(task['arrival_time_s']) * 1000.0
            network = float(task['edge_cloud_network_delay_ms'])
            upload = transfer_delay_ms(float(task['task_size_kb']), EDGE_CLOUD_BANDWIDTH_MBPS)
            cloud_arrival = source_arrival + network + upload
            cloud_items.append((cloud_arrival, source_arrival, task, upload, return_delay_ms(network)))
        cloud_items.sort(key=lambda x: x[0])
        for cloud_arrival, source_arrival, task, upload, ret in cloud_items:
            service = service_time_ms(float(task['cpu_demand_mi']), cloud_rate)
            sched = schedule_on_layer(cloud_state, cloud_arrival, service, cloud_capacity)
            if not sched['accepted']:
                rows.append(row_result(task, mode, 'blocked', 'blocked', None, None, None, None, None, 0, sched['queue_length_at_arrival'], upload, ret))
            else:
                finish = sched['service_finish_time_ms'] + ret
                rows.append(row_result(task, mode, 'cloud', 'completed', sched['start_time_ms'], finish, service, sched['waiting_time_ms'], finish - source_arrival, 0, sched['queue_length_after_assignment'], upload, ret))
        rows.sort(key=lambda r: r['arrival_time_s'])
    elif mode == 'hybrid_cloud_edge':
        cloud_items = []
        for _, task in subset.iterrows():
            arrival = float(task['arrival_time_s']) * 1000.0
            edge_q_now = waiting_queue_length(edge_state, arrival)
            edge_wait_est = estimate_waiting_delay_ms(edge_state, arrival)
            use_edge = edge_q_now < edge_capacity and edge_wait_est <= HYBRID_EDGE_DELAY_THRESHOLD_MS
            if use_edge:
                service = service_time_ms(float(task['cpu_demand_mi']), edge_rate)
                sched = schedule_on_layer(edge_state, arrival, service, edge_capacity)
                if sched['accepted']:
                    finish = sched['service_finish_time_ms']
                    rows.append(row_result(task, mode, 'edge', 'completed', sched['start_time_ms'], finish, service, sched['waiting_time_ms'], finish - arrival, sched['queue_length_after_assignment'], 0, 0.0, 0.0))
                    continue
            network = float(task['edge_cloud_network_delay_ms'])
            upload = transfer_delay_ms(float(task['task_size_kb']), EDGE_CLOUD_BANDWIDTH_MBPS)
            cloud_arrival = arrival + network + upload
            cloud_items.append((cloud_arrival, arrival, task, upload, return_delay_ms(network), edge_q_now))
        cloud_items.sort(key=lambda x: x[0])
        for cloud_arrival, source_arrival, task, upload, ret, edge_q_at_decision in cloud_items:
            service = service_time_ms(float(task['cpu_demand_mi']), cloud_rate)
            sched = schedule_on_layer(cloud_state, cloud_arrival, service, cloud_capacity)
            if not sched['accepted']:
                rows.append(row_result(task, mode, 'blocked', 'blocked', None, None, None, None, None, edge_q_at_decision, sched['queue_length_at_arrival'], upload, ret))
            else:
                finish = sched['service_finish_time_ms'] + ret
                rows.append(row_result(task, mode, 'cloud', 'completed', sched['start_time_ms'], finish, service, sched['waiting_time_ms'], finish - source_arrival, edge_q_at_decision, sched['queue_length_after_assignment'], upload, ret))
        rows.sort(key=lambda r: r['arrival_time_s'])
    else:
        raise ValueError(f'Unknown mode: {mode}')
    return (pd.DataFrame(rows), summarize(rows, edge_state, cloud_state, subset, mode))

def run_simulation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks, params_df = load_inputs()
    all_details = []
    all_summary = []
    for scenario in sorted(tasks['scenario'].unique()):
        params = get_scenario_params(params_df, scenario)
        subset = tasks[tasks['scenario'] == scenario]
        for mode in PROCESSING_MODES:
            print(f'Simulating public trace validation: scenario={scenario}, mode={mode}')
            details, summary = simulate_mode(subset, params, mode)
            all_details.append(details)
            all_summary.append(summary)
    detailed = pd.concat(all_details, ignore_index=True)
    summary = pd.DataFrame(all_summary)
    detailed.to_csv(OUTPUT_DIR / 'detailed_public_trace_results.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'public_trace_simulation_summary.csv', index=False)
    table = summary.copy()
    table['Mode'] = table['mode'].map(MODE_LABELS)
    table['Average response time (ms)'] = table['avg_response_time_ms'].round(2)
    table['Average waiting time (ms)'] = table['avg_waiting_time_ms'].round(2)
    table['Throughput (tasks/s)'] = table['throughput_tasks_per_s'].round(2)
    table['Edge utilization (%)'] = (table['edge_utilization'] * 100).round(2)
    table['Cloud utilization (%)'] = (table['cloud_utilization'] * 100).round(2)
    table['Average edge queue length'] = table['avg_edge_queue_length'].round(2)
    table['Average cloud queue length'] = table['avg_cloud_queue_length'].round(2)
    table['Blocking probability (%)'] = (table['blocking_probability'] * 100).round(2)
    table['Cloud offloading ratio (%)'] = (table['cloud_offloading_ratio'] * 100).round(2)
    paper_cols = ['Mode', 'Average response time (ms)', 'Average waiting time (ms)', 'Throughput (tasks/s)', 'Edge utilization (%)', 'Cloud utilization (%)', 'Average edge queue length', 'Average cloud queue length', 'Blocking probability (%)', 'Cloud offloading ratio (%)']
    table[paper_cols].to_csv(OUTPUT_DIR / 'public_trace_paper_results_table.csv', index=False)
    create_figures(summary)
    note = '\nPublic Trace-Based Validation Result Interpretation\n\nThe public trace-derived validation experiment uses task submission events and\nresource request patterns from Google ClusterData 2011-2. The trace is mapped\ninto the same cloud-edge simulation model used for the controlled synthetic\nexperiments. The objective is to test whether the relative performance trend\nobserved in the controlled experiment remains consistent under real production\ncloud workload arrival behavior.\n\nThe exact numerical values may differ from controlled synthetic results because\nthe public trace has its own temporal arrival pattern and resource demand\ndistribution. The main validation criterion is trend consistency across\nedge-only, cloud-only, and hybrid cloud-edge processing.\n'.strip()
    (OUTPUT_DIR / 'public_trace_result_interpretation_note.txt').write_text(note, encoding='utf-8')
    print()
    print('Public trace validation simulation completed.')
    print(f'Output folder: {OUTPUT_DIR.resolve()}')
    print()
    print(table[paper_cols].to_string(index=False))

def create_figures(summary):
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 10, 'figure.dpi': 150})
    plot_df = summary.copy()
    plot_df['Mode'] = plot_df['mode'].map(MODE_LABELS)
    metrics = [('avg_response_time_ms', 'Average response time (ms)', 'public_trace_response_time'), ('avg_waiting_time_ms', 'Average waiting time (ms)', 'public_trace_waiting_time'), ('throughput_tasks_per_s', 'Throughput (tasks/s)', 'public_trace_throughput'), ('blocking_probability', 'Blocking probability', 'public_trace_blocking_probability'), ('cloud_offloading_ratio', 'Cloud offloading ratio', 'public_trace_offloading_ratio')]
    for metric, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        ax.bar(plot_df['Mode'], plot_df[metric])
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Processing mode')
        ax.set_title(ylabel + ' for Public Trace Validation')
        ax.grid(axis='y', alpha=0.25)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=600, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
        plt.close()
if __name__ == '__main__':
    run_simulation()
