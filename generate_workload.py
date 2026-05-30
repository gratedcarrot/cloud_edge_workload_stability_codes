from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
BASE_RANDOM_SEED = 42
DEFAULT_RUNS_PER_SCENARIO = 30
DEFAULT_OUTPUT_DIR = Path('generated_workload')
SCENARIOS: List[Dict] = [{'scenario': 'low_load', 'description': 'Stable workload where arrival rate is below edge capacity.', 'duration_s': 300, 'base_lambda_tps': 8, 'burst_start_s': None, 'burst_end_s': None, 'burst_lambda_tps': None, 'heavy_tailed_demand': False, 'edge_nodes': 4, 'edge_service_rate_tps_each': 7, 'cloud_servers': 3, 'cloud_service_rate_tps_each': 18, 'edge_queue_capacity': 120, 'cloud_queue_capacity': 250, 'mean_network_delay_ms': 45, 'std_network_delay_ms': 8}, {'scenario': 'medium_load', 'description': 'Moderate workload where arrival rate approaches edge capacity.', 'duration_s': 300, 'base_lambda_tps': 22, 'burst_start_s': None, 'burst_end_s': None, 'burst_lambda_tps': None, 'heavy_tailed_demand': False, 'edge_nodes': 4, 'edge_service_rate_tps_each': 7, 'cloud_servers': 3, 'cloud_service_rate_tps_each': 18, 'edge_queue_capacity': 120, 'cloud_queue_capacity': 250, 'mean_network_delay_ms': 50, 'std_network_delay_ms': 10}, {'scenario': 'high_load', 'description': 'Overload condition where arrival rate exceeds edge capacity.', 'duration_s': 300, 'base_lambda_tps': 40, 'burst_start_s': None, 'burst_end_s': None, 'burst_lambda_tps': None, 'heavy_tailed_demand': False, 'edge_nodes': 4, 'edge_service_rate_tps_each': 7, 'cloud_servers': 3, 'cloud_service_rate_tps_each': 18, 'edge_queue_capacity': 120, 'cloud_queue_capacity': 250, 'mean_network_delay_ms': 55, 'std_network_delay_ms': 12}, {'scenario': 'burst_load', 'description': 'Normal workload with temporary large-scale burst demand.', 'duration_s': 300, 'base_lambda_tps': 14, 'burst_start_s': 100, 'burst_end_s': 160, 'burst_lambda_tps': 55, 'heavy_tailed_demand': False, 'edge_nodes': 4, 'edge_service_rate_tps_each': 7, 'cloud_servers': 3, 'cloud_service_rate_tps_each': 18, 'edge_queue_capacity': 120, 'cloud_queue_capacity': 250, 'mean_network_delay_ms': 60, 'std_network_delay_ms': 15}, {'scenario': 'extended_workload', 'description': 'Extended robustness scenario with heavier-tailed task demand variation.', 'duration_s': 300, 'base_lambda_tps': 28, 'burst_start_s': None, 'burst_end_s': None, 'burst_lambda_tps': None, 'heavy_tailed_demand': True, 'edge_nodes': 4, 'edge_service_rate_tps_each': 7, 'cloud_servers': 3, 'cloud_service_rate_tps_each': 18, 'edge_queue_capacity': 120, 'cloud_queue_capacity': 250, 'mean_network_delay_ms': 58, 'std_network_delay_ms': 14}]
EDGE_ZONES = ['edge_zone_A', 'edge_zone_B', 'edge_zone_C', 'edge_zone_D']
TASK_TYPE_PROBABILITIES = {'sensor_event': 0.34, 'video_frame': 0.18, 'transaction': 0.25, 'analytics_job': 0.15, 'control_signal': 0.08}
PRIORITY_LEVELS = [1, 2, 3, 4]
PRIORITY_PROBABILITIES = [0.5, 0.3, 0.15, 0.05]

def clipped_normal(rng: np.random.Generator, mean: float, std: float, minimum: float) -> float:
    return max(minimum, float(rng.normal(mean, std)))

def generate_task_profile(rng: np.random.Generator, task_type: str, heavy_tailed: bool=False) -> Dict[str, float]:
    if task_type == 'video_frame':
        task_size_kb = rng.lognormal(mean=7.2, sigma=0.35)
        cpu_demand_mi = rng.lognormal(mean=5.2, sigma=0.35)
        memory_mb = clipped_normal(rng, mean=180, std=35, minimum=32)
        deadline_ms = clipped_normal(rng, mean=450, std=80, minimum=100)
    elif task_type == 'analytics_job':
        task_size_kb = rng.lognormal(mean=7.6, sigma=0.45)
        cpu_demand_mi = rng.lognormal(mean=5.8, sigma=0.45)
        memory_mb = clipped_normal(rng, mean=260, std=60, minimum=64)
        deadline_ms = clipped_normal(rng, mean=850, std=150, minimum=200)
    elif task_type == 'transaction':
        task_size_kb = rng.lognormal(mean=5.4, sigma=0.3)
        cpu_demand_mi = rng.lognormal(mean=4.4, sigma=0.3)
        memory_mb = clipped_normal(rng, mean=90, std=20, minimum=16)
        deadline_ms = clipped_normal(rng, mean=300, std=60, minimum=80)
    elif task_type == 'control_signal':
        task_size_kb = rng.lognormal(mean=3.8, sigma=0.25)
        cpu_demand_mi = rng.lognormal(mean=3.4, sigma=0.25)
        memory_mb = clipped_normal(rng, mean=35, std=8, minimum=8)
        deadline_ms = clipped_normal(rng, mean=120, std=25, minimum=40)
    else:
        task_size_kb = rng.lognormal(mean=4.8, sigma=0.35)
        cpu_demand_mi = rng.lognormal(mean=4.0, sigma=0.35)
        memory_mb = clipped_normal(rng, mean=60, std=15, minimum=8)
        deadline_ms = clipped_normal(rng, mean=220, std=40, minimum=60)
    if heavy_tailed:
        multiplier = min(float(rng.pareto(a=2.5) + 1.0), 5.0)
        task_size_kb *= multiplier
        cpu_demand_mi *= multiplier
        memory_mb *= min(multiplier, 3.0)
        deadline_ms *= min(1.0 + 0.35 * multiplier, 2.4)
    return {'task_size_kb': round(float(task_size_kb), 3), 'cpu_demand_mi': round(float(cpu_demand_mi), 3), 'memory_demand_mb': round(float(memory_mb), 3), 'deadline_ms': int(round(deadline_ms))}

def lambda_for_second(scenario: Dict, second: int) -> Tuple[int, int]:
    in_burst = scenario['burst_start_s'] is not None and scenario['burst_start_s'] <= second < scenario['burst_end_s']
    if in_burst:
        return (int(scenario['burst_lambda_tps']), 1)
    return (int(scenario['base_lambda_tps']), 0)

def generate_arrival_times_for_run(rng: np.random.Generator, scenario: Dict) -> pd.DataFrame:
    arrival_times = []
    lambda_values = []
    burst_flags = []
    for second in range(int(scenario['duration_s'])):
        lam, burst_flag = lambda_for_second(scenario, second)
        arrivals_this_second = int(rng.poisson(lam))
        if arrivals_this_second > 0:
            second_offsets = rng.random(arrivals_this_second)
            for offset in second_offsets:
                arrival_times.append(second + float(offset))
                lambda_values.append(lam)
                burst_flags.append(burst_flag)
    arrivals_df = pd.DataFrame({'arrival_time_s': arrival_times, 'lambda_at_arrival_tps': lambda_values, 'burst_flag': burst_flags}).sort_values('arrival_time_s', ignore_index=True)
    arrivals_df['inter_arrival_time_s'] = arrivals_df['arrival_time_s'].diff()
    if not arrivals_df.empty:
        arrivals_df.loc[0, 'inter_arrival_time_s'] = arrivals_df.loc[0, 'arrival_time_s']
    return arrivals_df

def scenario_run_seed(base_seed: int, scenario_index: int, run_id: int) -> int:
    return int(base_seed + scenario_index * 1000 + run_id)

def generate_dataset(runs_per_scenario: int, base_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_tasks = []
    seed_rows = []
    task_counter = 1
    task_types = list(TASK_TYPE_PROBABILITIES.keys())
    task_probs = list(TASK_TYPE_PROBABILITIES.values())
    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        for run_id in range(1, runs_per_scenario + 1):
            seed = scenario_run_seed(base_seed, scenario_index, run_id)
            rng = np.random.default_rng(seed)
            seed_rows.append({'scenario': scenario['scenario'], 'run_id': run_id, 'seed': seed})
            arrivals = generate_arrival_times_for_run(rng, scenario)
            for _, row in arrivals.iterrows():
                task_type = str(rng.choice(task_types, p=task_probs))
                profile = generate_task_profile(rng, task_type, heavy_tailed=bool(scenario.get('heavy_tailed_demand', False)))
                source_zone = str(rng.choice(EDGE_ZONES))
                edge_index = EDGE_ZONES.index(source_zone) + 1
                network_delay_ms = clipped_normal(rng, mean=float(scenario['mean_network_delay_ms']), std=float(scenario['std_network_delay_ms']), minimum=5)
                all_tasks.append({'task_id': f'T{task_counter:09d}', 'scenario': scenario['scenario'], 'run_id': run_id, 'seed': seed, 'arrival_time_s': round(float(row['arrival_time_s']), 6), 'inter_arrival_time_s': round(float(row['inter_arrival_time_s']), 6), 'lambda_at_arrival_tps': int(row['lambda_at_arrival_tps']), 'burst_flag': int(row['burst_flag']), 'source_zone': source_zone, 'edge_node_candidate': f'edge_{edge_index}', 'task_type': task_type, 'priority': int(rng.choice(PRIORITY_LEVELS, p=PRIORITY_PROBABILITIES)), 'task_size_kb': profile['task_size_kb'], 'cpu_demand_mi': profile['cpu_demand_mi'], 'memory_demand_mb': profile['memory_demand_mb'], 'deadline_ms': profile['deadline_ms'], 'edge_cloud_network_delay_ms': round(float(network_delay_ms), 3)})
                task_counter += 1
    tasks_df = pd.DataFrame(all_tasks)
    seed_df = pd.DataFrame(seed_rows)
    tasks_df['arrival_second'] = np.floor(tasks_df['arrival_time_s']).astype(int)
    ts_rows = []
    for scenario in SCENARIOS:
        scenario_name = scenario['scenario']
        duration = int(scenario['duration_s'])
        for run_id in range(1, runs_per_scenario + 1):
            subset_run = tasks_df[(tasks_df['scenario'] == scenario_name) & (tasks_df['run_id'] == run_id)]
            grouped = subset_run.groupby('arrival_second')
            for second in range(duration):
                lam, burst_flag = lambda_for_second(scenario, second)
                subset_second = grouped.get_group(second) if second in grouped.groups else pd.DataFrame()
                ts_rows.append({'scenario': scenario_name, 'run_id': run_id, 'arrival_second': second, 'arrivals': int(len(subset_second)), 'configured_lambda_tps': lam, 'burst_flag': burst_flag, 'avg_task_size_kb': round(float(subset_second['task_size_kb'].mean()), 3) if len(subset_second) else 0.0, 'avg_cpu_demand_mi': round(float(subset_second['cpu_demand_mi'].mean()), 3) if len(subset_second) else 0.0})
    timeseries_df = pd.DataFrame(ts_rows)
    params_df = pd.DataFrame(SCENARIOS)
    params_df['runs_per_scenario'] = runs_per_scenario
    params_df['base_random_seed'] = base_seed
    params_df['total_edge_capacity_tps'] = params_df['edge_nodes'] * params_df['edge_service_rate_tps_each']
    params_df['total_cloud_capacity_tps'] = params_df['cloud_servers'] * params_df['cloud_service_rate_tps_each']
    params_df['total_system_capacity_tps'] = params_df['total_edge_capacity_tps'] + params_df['total_cloud_capacity_tps']
    run_summary_df = tasks_df.groupby(['scenario', 'run_id'], as_index=False).agg(seed=('seed', 'first'), total_tasks=('task_id', 'count'), burst_tasks=('burst_flag', 'sum'), avg_task_size_kb=('task_size_kb', 'mean'), avg_cpu_demand_mi=('cpu_demand_mi', 'mean'), avg_memory_demand_mb=('memory_demand_mb', 'mean'), avg_deadline_ms=('deadline_ms', 'mean'), avg_network_delay_ms=('edge_cloud_network_delay_ms', 'mean'))
    duration_lookup = {s['scenario']: int(s['duration_s']) for s in SCENARIOS}
    run_summary_df['observed_lambda_tps'] = run_summary_df.apply(lambda r: r['total_tasks'] / duration_lookup[r['scenario']], axis=1)
    summary_df = run_summary_df.groupby('scenario', as_index=False).agg(avg_tasks_per_run=('total_tasks', 'mean'), std_tasks_per_run=('total_tasks', 'std'), avg_observed_lambda_tps=('observed_lambda_tps', 'mean'), avg_burst_tasks=('burst_tasks', 'mean'), avg_task_size_kb=('avg_task_size_kb', 'mean'), avg_cpu_demand_mi=('avg_cpu_demand_mi', 'mean'), avg_memory_demand_mb=('avg_memory_demand_mb', 'mean'), avg_deadline_ms=('avg_deadline_ms', 'mean'), avg_network_delay_ms=('avg_network_delay_ms', 'mean'))
    return (tasks_df, timeseries_df, params_df, summary_df, seed_df)

def write_outputs(output_dir: Path, tasks_df: pd.DataFrame, timeseries_df: pd.DataFrame, params_df: pd.DataFrame, summary_df: pd.DataFrame, seed_df: pd.DataFrame, runs_per_scenario: int, base_seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_df.to_csv(output_dir / 'cloud_edge_workload_trace.csv', index=False)
    timeseries_df.to_csv(output_dir / 'workload_arrivals_timeseries.csv', index=False)
    params_df.to_csv(output_dir / 'scenario_parameters.csv', index=False)
    summary_df.to_csv(output_dir / 'dataset_summary.csv', index=False)
    seed_df.to_csv(output_dir / 'random_seeds.csv', index=False)
    metadata = {'dataset_name': 'Cloud-Edge Probabilistic Workload Dataset', 'dataset_source': 'Author-generated synthetic dataset', 'generator': 'Custom Python probabilistic workload generator', 'base_random_seed': base_seed, 'runs_per_scenario': runs_per_scenario, 'generation_method': 'Poisson arrival process for low, medium, high, and extended workload; non-homogeneous Poisson process for burst load; extended workload uses heavier-tailed task demand variation.', 'scenarios': [s['scenario'] for s in SCENARIOS], 'total_task_records': int(len(tasks_df)), 'outputs': ['cloud_edge_workload_trace.csv', 'workload_arrivals_timeseries.csv', 'scenario_parameters.csv', 'dataset_summary.csv', 'random_seeds.csv', 'metadata.json', 'README.txt']}
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    readme = f'\nCloud-Edge Probabilistic Workload Dataset\n=================================================\n\nThis dataset is generated for controlled simulation-based evaluation of\ncloud-edge workload handling strategies.\n\nKey settings\n------------\nRuns per scenario: {runs_per_scenario}\nBase random seed: {base_seed}\nTotal task records: {len(tasks_df):,}\n\nScenarios\n---------\n1. low_load\n2. medium_load\n3. high_load\n4. burst_load\n5. extended_workload\n\nFiles\n-----\ncloud_edge_workload_trace.csv    Task-level trace.\nworkload_arrivals_timeseries.csv   Per-second arrivals.\nscenario_parameters.csv       Scenario and system parameters.\ndataset_summary.csv         Scenario-level dataset summary.\nrandom_seeds.csv           Scenario/run seed values.\nmetadata.json            Machine-readable metadata.\n\nRecommended next command\n------------------------\npython run_simulation.py --sensitivity\n'.strip()
    with open(output_dir / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(readme)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate cloud-edge workload traces.')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS_PER_SCENARIO, help='Runs per scenario.')
    parser.add_argument('--seed', type=int, default=BASE_RANDOM_SEED, help='Base random seed.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory for generated dataset.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    tasks_df, timeseries_df, params_df, summary_df, seed_df = generate_dataset(runs_per_scenario=args.runs, base_seed=args.seed)
    write_outputs(output_dir=args.output_dir, tasks_df=tasks_df, timeseries_df=timeseries_df, params_df=params_df, summary_df=summary_df, seed_df=seed_df, runs_per_scenario=args.runs, base_seed=args.seed)
    print('Dataset generated successfully.')
    print(f'Output directory: {args.output_dir.resolve()}')
    print(f'Total task records: {len(tasks_df):,}')
    print()
    print('Scenario summary:')
    print(summary_df.to_string(index=False))
if __name__ == '__main__':
    main()
