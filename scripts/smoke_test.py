"""Replay one complete high-load workload and selected Google/locality controls."""
from _paths import DATA, RESULTS, REFERENCE_RESULTS
import json
import numpy as np
import pandas as pd
from study_engine import simulate, prepare, POLICIES
from run_locality import geography, arrayout


def main():
    sc = next(p for p in json.loads((DATA/'synthetic/scenarios.json').read_text())
              if p['scenario'] == 'high_load')
    manifest = pd.read_csv(DATA/'synthetic/workloads.csv')
    row = manifest[(manifest.scenario == 'high_load') & (manifest.run_id == 1)].iloc[0]
    sc.update(run_id=1, seed=int(row.seed))
    tasks = prepare(pd.read_csv(DATA/'synthetic'/row.path))
    saved = pd.read_csv(REFERENCE_RESULTS/'main/runs.csv')
    count = 0
    for network, bandwidth in [('independent', 25.), ('shared_fcfs', 25.), ('shared_fcfs', 250.)]:
        for mode in POLICIES:
            summary, _ = simulate(tasks, sc, mode, network, bandwidth, backend='native')
            old = saved[(saved.scenario == 'high_load') & (saved.run_id == 1) &
                        (saved['mode'] == mode) & (saved.network_model == network) &
                        (saved.bandwidth_mbps == bandwidth)].iloc[0]
            for key in ['completed_tasks','blocked_tasks','on_time_completed_tasks','sla_violations',
                        'avg_response_time_ms','avg_waiting_time_ms','total_cost_proxy','total_energy_proxy']:
                np.testing.assert_allclose(summary[key],old[key],rtol=1e-11,atol=1e-8,err_msg=key)
            count += 1
    g = prepare(pd.read_csv(DATA/'google/cloud_edge_workload_trace.csv'))
    p = dict(duration_s=float(g.df.arrival_time_s.max()))
    reference = pd.read_csv(REFERENCE_RESULTS/'google/base_outcomes_20k.csv')
    for mode in ('hybrid_cloud_edge','deadline_aware_hybrid','deadline_guarded'):
        s,_ = simulate(g,p,mode,'shared_fcfs',25.,backend='native')
        r = reference[(reference.anchor=='native') & (reference.network_model=='shared_fcfs') &
                      (reference.bandwidth_mbps==25) & (reference['mode']==mode)].iloc[0]
        for key in ('on_time_completed_tasks','completed_tasks','avg_response_time_ms'):
            np.testing.assert_allclose(s[key],r[key],rtol=1e-11,atol=1e-8)
    # The spatial kernel's unrestricted pool must equal the main kernel.
    spatial = arrayout(tasks,sc,25.,'hybrid_cloud_edge',np.zeros(len(tasks.x),dtype=np.int32),1)
    main_s,_ = simulate(tasks,sc,'hybrid_cloud_edge','shared_fcfs',25.,backend='native')
    timely=(spatial[:,1]>0)&(spatial[:,6]<=tasks.x[:,3]+1e-9)
    assert int(timely.sum()) == main_s['on_time_completed_tasks']
    active, users = geography()
    assert len(active)==4 and len(users)==816
    report=dict(main_policy_cells=count,google_policy_cells=3,spatial_pool_check=True,
                reference_metrics_match=True)
    path=RESULTS/'verification';path.mkdir(parents=True,exist_ok=True)
    (path/'smoke_test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__': main()
