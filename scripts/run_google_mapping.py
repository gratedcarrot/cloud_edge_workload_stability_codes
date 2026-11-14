"""Matched Google 20k sensitivity: rerun without network from bundled extracted input."""
from pathlib import Path
import sys, itertools, hashlib, argparse
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from _paths import DATA, RESULTS
import study_engine as eng
INPUT=DATA/'google/cloud_edge_workload_trace.csv'
OUT=RESULTS/'google'; OUT.mkdir(parents=True,exist_ok=True)
CASES=dict([('base',('',1.))]+[(f'{key}_{label}',(key,m)) for key in ['arrival_span','cpu','deadline','payload','nominal_delay'] for label,m in [('half',0.5),('double',2.)]]+[('edge_servers_2',('edge_servers',2)),('edge_servers_8',('edge_servers',8)),('cloud_servers_1',('cloud_servers',1)),('cloud_servers_6',('cloud_servers',6))])
NETWORKS=[('independent',25.),('shared_fcfs',25.),('shared_fcfs',250.)]
def work(job):
    anchor,case=job
    df=pd.read_csv(INPUT); n=len(df); assert n==20000
    key,f=CASES[case]
    p=dict(edge_nodes=4,cloud_servers=3,edge_queue_capacity=120,cloud_queue_capacity=250,edge_service_rate_tps_each=7.,cloud_service_rate_tps_each=18.)
    native_span=df.arrival_time_s.max()
    span=600. if anchor=='affine600' else native_span
    if key=='arrival_span':span*=f
    if anchor=='affine600' or key=='arrival_span':df['arrival_time_s']*=span/native_span
    p['duration_s']=span
    c=dict(cpu='cpu_demand_mi',payload='task_size_kb',deadline='deadline_ms',nominal_delay='edge_cloud_network_delay_ms').get(key)
    if c:df[c]*=f
    if key=='edge_servers':p['edge_nodes']=f
    if key=='cloud_servers':p['cloud_servers']=f
    w=eng.prepare(df); rows=[]
    arr=w.x[:,0]
    for net,bw in NETWORKS:
        dests={}
        for pol in eng.POLICIES:
            s,d=eng.simulate(w,p,pol,net,bw,details=True,backend='native')
            dest=d.requested_destination.map({'edge':1,'cloud':2,'blocked':0}).to_numpy(dtype='uint8')
            dests[pol]=dest
            s.update(anchor=anchor,mapping=case,route_sha256=hashlib.sha256(dest.tobytes()).hexdigest(),least_lateness_count=(d.decision_reason=='least_predicted_lateness').sum(),peak_arrivals_per_clock_second=int(pd.Series(np.floor(arr)).value_counts().max()))
            assert s['completed_tasks']+s['blocked_tasks']==n and s['on_time_completed_tasks']+s['sla_violations']==s['completed_tasks']
            rows.append(s)
        for s in rows[-8:]:s['route_changes_vs_queue_aware']=int((dests[s['mode']]!=dests['hybrid_cloud_edge']).sum())
    return rows
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=3);args=a.parse_args()
    eng.native()
    with ProcessPoolExecutor(args.workers) as ex:
        out=pd.DataFrame([s for rows in ex.map(work,itertools.product(['native','affine600'],CASES)) for s in rows])
    g=out.groupby(['anchor','mapping','network_model','bandwidth_mbps'])
    assert len(out)==720 and len(g)==90
    assert g['mode'].nunique().eq(8).all() and g.workload_sha256.nunique().eq(1).all() and g.environment_sha256.nunique().eq(1).all()
    out.to_csv(OUT/'mapping_sensitivity_20k.csv',index=False)
    out.loc[out.mapping=='base'].to_csv(OUT/'base_outcomes_20k.csv',index=False)
    print('verified 90 matched conditions, 720 policy runs')
