"""Export numerical manuscript and supplementary tables from saved run-level results.

These CSVs are numeric exports rather than Word layout replicas. Existing source
CSV files retain every realization, coefficient and mapping used in the study.
"""
from _paths import RESULTS, result_file
import numpy as np
import pandas as pd

OUT = RESULTS / 'tables'
OUT.mkdir(parents=True, exist_ok=True)
POLICIES = ['edge_only', 'cloud_only', 'hybrid_cloud_edge', 'deadline_aware_hybrid',
            'utilization_based', 'network_aware', 'queue_length_based', 'deadline_guarded']


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


def main():
    main_runs = pd.read_csv(result_file('main/runs.csv'))
    ci = pd.read_csv(result_file('tables/main_original_metrics_ci.csv'))
    sweep = pd.read_csv(result_file('tables/sensitivity_original_metrics_ci.csv'))
    diagnostics = pd.read_csv(result_file('analytical/workload_assumptions_summary.csv'))
    wanted = ['low_load', 'medium_load', 'high_load']
    save(diagnostics[diagnostics.scenario.isin(wanted)], 'table_II_queue_comparison.csv')
    save(ci[(ci.network_model == 'independent') & ci['mode'].isin(POLICIES[:4])],
         'table_VIIa_four_modes_with_intervals.csv')
    selected = [('high_load', 25), ('burst_load', 25), ('extended_workload', 25),
                ('high_load', 250), ('extended_workload', 250)]
    rows = []
    for scenario, bandwidth in selected:
        frame = main_runs[(main_runs.scenario == scenario) &
                          (main_runs.network_model == 'shared_fcfs') &
                          (main_runs.bandwidth_mbps == bandwidth)]
        for mode, group in frame.groupby('mode'):
            rows.append(dict(scenario=scenario, bandwidth_mbps=bandwidth, mode=mode,
                             independent_runs=len(group),
                             on_time_pct=100 * group.on_time_completion_rate.mean()))
    save(pd.DataFrame(rows), 'table_VIIb_shared_link_policy_comparison.csv')
    tests = pd.read_csv(result_file('tables/secondary_four_policy_tests.csv'))
    assert len(tests) == 40 and (tests.k_policies == 4).all()
    rows = []
    for metric, group in tests.groupby('metric'):
        rows.append(dict(metric=metric, scenarios=len(group), policies=4,
                         anova_GG_Holm_significant=int((group.rm_gg_holm_40 < .05).sum()),
                         friedman_Holm_significant=int((group.friedman_holm_40 < .05).sum())))
    save(pd.DataFrame(rows), 'table_VIII_four_policy_test_summary.csv')
    save(sweep[(sweep.parameter == 'threshold_ms') & (sweep.network_model == 'independent')],
         'table_IX_threshold_sensitivity.csv')
    save(sweep[sweep.parameter.isin(['edge_nodes', 'cloud_servers']) &
               (sweep.network_model == 'independent')], 'table_X_capacity_sensitivity.csv')
    google = pd.read_csv(result_file('google/base_outcomes_20k.csv'))
    reference = google[(google.anchor == 'affine600') &
                       (google.network_model == 'independent') & (google.bandwidth_mbps == 25)]
    save(reference[reference['mode'].isin(POLICIES[:4])], 'table_XII_google_reference.csv')
    s1 = google.assign(on_time_pct=100 * google.on_time_completion_rate)
    s1 = s1.pivot(index='mode', columns=['anchor', 'network_model', 'bandwidth_mbps'],
                  values='on_time_pct').reindex(POLICIES)
    s1.columns = [' | '.join(map(str, c)) for c in s1.columns]
    save(s1.reset_index(), 'table_S1_google_on_time.csv')
    mapping = pd.read_csv(result_file('google/mapping_sensitivity_20k.csv'))
    keys = ['anchor', 'network_model', 'bandwidth_mbps', 'mapping']
    pivot = mapping.pivot(index=keys, columns='mode', values='on_time_completion_rate')
    diff = 100 * (pivot.deadline_aware_hybrid - pivot.hybrid_cloud_edge)
    s2 = diff.groupby(level=[0, 1, 2]).agg(['min', 'max'])
    s2['base_mapping'] = diff.xs('base', level='mapping')
    save(s2.reset_index(), 'table_S2_mapping_sensitivity.csv')
    proxy = pd.read_csv(result_file('google/coefficient_sensitivity_20k.csv'))
    s3 = proxy[(proxy.anchor == 'affine600') & (proxy.network_model == 'independent') &
               (proxy.bandwidth_mbps == 25) & proxy.coefficient_case.isin(['base', 'ratio_c0_e0']) &
               proxy['mode'].isin(['edge_only', 'cloud_only'])]
    save(s3[['mode', 'coefficient_case', 'avg_cost_proxy', 'avg_energy_proxy']],
         'table_S3_proxy_scores.csv')
    locality = pd.read_csv(result_file('locality/condition_means.csv'))
    s4 = locality[(locality.radius_m == 600) | (locality.topology == 'unrestricted_pool')]
    s4 = s4.pivot(index='policy', columns=['bandwidth_mbps', 'topology'],
                  values='on_time_pct').reindex(POLICIES)
    s4.columns = [' | '.join(map(str, c)) for c in s4.columns]
    save(s4.reset_index(), 'table_S4_locality.csv')
    print('Exported manuscript Tables II, VII–X, XII and supplementary Tables S1–S4.')


if __name__ == '__main__':
    main()
