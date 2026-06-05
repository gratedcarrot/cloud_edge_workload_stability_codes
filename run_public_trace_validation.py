from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_simulation import (
    PROCESSING_MODES,
    aggregate_with_ci,
    build_paper_results_table_ci,
    get_scenario_params,
    load_inputs as load_core_inputs,
    simulate_subset,
)

INPUT_DIR = Path('public_trace_data')
OUTPUT_DIR = Path('public_trace_results')
MODE_LABELS = {
    'edge_only': 'Edge-only',
    'cloud_only': 'Cloud-only',
    'hybrid_cloud_edge': 'Hybrid cloud-edge',
    'deadline_aware_hybrid': 'Deadline-aware hybrid',
}


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_core_inputs(INPUT_DIR)


def build_single_run_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table['Mode'] = table['mode'].map(MODE_LABELS).fillna(table['mode'])
    table['Average response time (ms)'] = table['avg_response_time_ms'].round(2)
    table['Average waiting time (ms)'] = table['avg_waiting_time_ms'].round(2)
    table['Throughput (tasks/s)'] = table['throughput_tasks_per_s'].round(2)
    table['Edge utilization (%)'] = (table['edge_utilization'] * 100).round(2)
    table['Cloud utilization (%)'] = (table['cloud_utilization'] * 100).round(2)
    table['Average edge queue length'] = table['avg_edge_queue_length'].round(2)
    table['Average cloud queue length'] = table['avg_cloud_queue_length'].round(2)
    table['Blocking probability (%)'] = (table['blocking_probability'] * 100).round(2)
    table['Cloud offloading ratio (%)'] = (table['cloud_offloading_ratio'] * 100).round(2)
    table['SLA violation rate (%)'] = (table['sla_violation_rate'] * 100).round(2)
    table['Average cost proxy'] = table['avg_cost_proxy'].round(5)
    table['Average energy proxy'] = table['avg_energy_proxy'].round(4)
    paper_cols = [
        'Mode',
        'Average response time (ms)',
        'Average waiting time (ms)',
        'Throughput (tasks/s)',
        'Edge utilization (%)',
        'Cloud utilization (%)',
        'Average edge queue length',
        'Average cloud queue length',
        'Blocking probability (%)',
        'Cloud offloading ratio (%)',
        'SLA violation rate (%)',
        'Average cost proxy',
        'Average energy proxy',
    ]
    return table[paper_cols]


def create_figures(summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })
    plot_df = summary.copy()
    plot_df['Mode'] = plot_df['mode'].map(MODE_LABELS).fillna(plot_df['mode'])
    plot_df['mode'] = pd.Categorical(plot_df['mode'], categories=PROCESSING_MODES, ordered=True)
    plot_df = plot_df.sort_values('mode')
    metrics = [
        ('avg_response_time_ms', 'Average response time (ms)', 'public_trace_response_time'),
        ('avg_waiting_time_ms', 'Average waiting time (ms)', 'public_trace_waiting_time'),
        ('throughput_tasks_per_s', 'Throughput (tasks/s)', 'public_trace_throughput'),
        ('blocking_probability', 'Blocking probability', 'public_trace_blocking_probability'),
        ('cloud_offloading_ratio', 'Cloud offloading ratio', 'public_trace_offloading_ratio'),
        ('sla_violation_rate', 'SLA violation rate', 'public_trace_sla_violation'),
        ('avg_cost_proxy', 'Average cost proxy', 'public_trace_cost_proxy'),
        ('avg_energy_proxy', 'Average energy proxy', 'public_trace_energy_proxy'),
    ]
    for metric, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.bar(plot_df['Mode'], plot_df[metric], width=0.55)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Processing mode')
        ax.set_title(ylabel + ' for Public Trace Validation')
        ax.grid(axis='y', alpha=0.25)
        plt.xticks(rotation=12, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=600, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / f'{name}.pdf', bbox_inches='tight')
        plt.close()


def run_simulation() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks, params_df = load_inputs()
    all_details: List[pd.DataFrame] = []
    all_summary: List[Dict] = []

    for scenario in sorted(tasks['scenario'].unique()):
        params = get_scenario_params(params_df, scenario)
        scenario_df = tasks[tasks['scenario'] == scenario]
        for run_id in sorted(scenario_df['run_id'].unique()):
            subset = scenario_df[scenario_df['run_id'] == run_id]
            for mode in PROCESSING_MODES:
                print(f'Public trace validation: scenario={scenario}, run={run_id}, mode={mode}')
                details, summary = simulate_subset(subset, params, mode)
                all_details.append(details)
                all_summary.append(summary)

    detailed = pd.concat(all_details, ignore_index=True)
    summary = pd.DataFrame(all_summary)
    detailed.to_csv(OUTPUT_DIR / 'detailed_public_trace_results.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'public_trace_simulation_summary.csv', index=False)

    paper_single_run_table = build_single_run_table(summary)
    paper_single_run_table.to_csv(OUTPUT_DIR / 'public_trace_paper_results_table.csv', index=False)

    agg = aggregate_with_ci(summary, ['scenario', 'mode'])
    agg.to_csv(OUTPUT_DIR / 'public_trace_summary_with_ci.csv', index=False)
    paper_ci_table = build_paper_results_table_ci(agg)
    paper_ci_table.to_csv(OUTPUT_DIR / 'public_trace_paper_results_table_with_ci.csv', index=False)

    create_figures(summary)

    note = """
Public Trace-Based Validation Result Interpretation

The public trace-derived validation experiment uses task submission events and
resource request patterns from Google ClusterData 2011-2. The trace is mapped
into the same cloud-edge simulation model used for the controlled synthetic
experiments. The objective is to test whether the relative performance trend
observed in controlled experiments remains consistent under real production
cloud workload arrival behavior.

This enhanced validation includes edge-only, cloud-only, baseline hybrid, and
deadline-aware hybrid modes. The deadline-aware mode reports the same response,
waiting, throughput, queueing, blocking, offloading, and SLA metrics as the
baseline modes, with additional cost and energy proxy indicators.

The exact numerical values may differ from controlled synthetic results because
the public trace has its own temporal arrival pattern and resource demand
distribution. The main validation criterion is trend consistency across
processing modes.
""".strip()
    (OUTPUT_DIR / 'public_trace_result_interpretation_note.txt').write_text(note, encoding='utf-8')

    print()
    print('Public trace validation simulation completed.')
    print(f'Output folder: {OUTPUT_DIR.resolve()}')
    print()
    print(paper_single_run_table.to_string(index=False))


if __name__ == '__main__':
    run_simulation()
