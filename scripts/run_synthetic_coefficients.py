"""Coefficient sensitivity on six fixed synthetic workloads (three runs per scenario).
This is distinct from the 30-run main study and from the 20,000-task Google replay.
Policies whose routes depend on a changed proxy are replayed; invariant routes
are revalued exactly. Both operations are identified in the output.
"""
from pathlib import Path
from functools import lru_cache
import argparse, hashlib, json, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from _paths import DATA, RESULTS, ROOT
import sys
sys.path.insert(0,str(ROOT/'engine/reference'))
import frozen_policy_engine as core
import study_engine
from run_google_coefficients import coefficient_cases
POLICIES=['edge_only','cloud_only','hybrid_cloud_edge','deadline_aware_hybrid','deadline_guarded','utilization_based','network_aware','queue_length_based']
ADAPT={'deadline_aware_hybrid','deadline_guarded'}
BITS={'edge':1,'cloud':2,'blocked':0}
NETWORKS=[('independent',25.),('shared_fcfs',25.),('shared_fcfs',250.)]
def hash_obj(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,default=str,allow_nan=True).encode()).hexdigest()
def proxy_input(block):
    scenario,rid=block.rsplit('_',1)
    df=pd.read_csv(DATA/'synthetic_proxy'/f'{block}.csv.gz')
    rows=pd.read_csv(DATA/'synthetic_proxy/parameters.csv').set_index('scenario')
    p=dict(edge_nodes=4,cloud_servers=3,edge_service_rate_tps_each=7.,cloud_service_rate_tps_each=18.,edge_queue_capacity=120,cloud_queue_capacity=250,duration_s=300.,base_lambda_tps=float(rows.loc[scenario,'base_lambda_tps']),scenario=scenario)
    return df,p
def services(df,p):
    demand=np.maximum(df.cpu_demand_mi.to_numpy(),1.)/100.
    return tuple(np.maximum(5.,1000./p[rate]*demand) for rate in ('edge_service_rate_tps_each','cloud_service_rate_tps_each'))

def signature(es,cs,size,c):
    a=c['coefficients'];kb=size/c['divisor'];cw=c['proxy_cost_weight'];ew=c['proxy_energy_weight']
    ec=a['edge_cost_per_service_ms']*es;ee=a['edge_energy_per_service_ms']*es
    cc=a['cloud_cost_per_service_ms']*cs+a['network_cost_per_mb']*kb
    ce=a['cloud_energy_per_service_ms']*cs+a['network_energy_per_mb']*kb
    pref=cw*ec+ew*ee<=cw*cc+ew*ce
    return hashlib.sha256(np.packbits(pref).tobytes()).hexdigest(),pref

def run(df,p,pol,net,bw,c,backend):
    p=dict(p,bandwidth_mbps=bw,network_model=net,proxy_cost_weight=c['proxy_cost_weight'],proxy_energy_weight=c['proxy_energy_weight'])
    a=dict(c['coefficients'])
    if backend=='reference':
        # Original accounting computes size/1024. Scaling its two per-MB coefficients
        # by 1024/div implements exactly the desired per-decimal-MB rate (roundoff aside).
        a['network_cost_per_mb']*=1024./c['divisor'];a['network_energy_per_mb']*=1024./c['divisor']
        p.update(a); detail,s=core.simulate_subset(df,p,'deadline_aware_hybrid' if pol=='deadline_guarded' else pol,diagnostic_components={'least_lateness':False} if pol=='deadline_guarded' else None)
    else:
        import study_engine
        s,detail=study_engine.simulate(df,p,pol,network=net,bandwidth=bw,coefficients=a,details=True,proxy_divisor=c['divisor'])
    detail=detail.reset_index(drop=True)
    assert np.array_equal(np.array(df.task_id),np.array(detail.task_id)),'Task order mismatch'
    dest=detail.requested_destination.map(BITS).to_numpy(dtype='uint8'); ec=detail.processing_location.eq('edge');cc=detail.processing_location.eq('cloud')
    e_ms=detail.loc[ec,'service_time_ms'].sum();c_ms=detail.loc[cc,'service_time_ms'].sum();mb=df.task_size_kb.to_numpy()[dest==2].sum()/1000
    A=c['coefficients']; cost=e_ms*A['edge_cost_per_service_ms']+c_ms*A['cloud_cost_per_service_ms']+mb*1000/c['divisor']*A['network_cost_per_mb'];energy=e_ms*A['edge_energy_per_service_ms']+c_ms*A['cloud_energy_per_service_ms']+mb*1000/c['divisor']*A['network_energy_per_mb']
    for k,v in [('total_cost_proxy',cost),('total_energy_proxy',energy)]:np.testing.assert_allclose(s[k],v,rtol=2e-11,atol=1e-8)
    s=dict(s)
    s.update(mode=pol,edge_service_ms_sum=e_ms,cloud_service_ms_sum=c_ms,uploaded_decimal_mb=mb,route_signature=hashlib.sha256(dest.tobytes()).hexdigest(),tie_invocations=int(detail.decision_reason.astype(str).str.startswith('proxy_tie_break_to_').sum()),feasibility_overrides=int(detail.decision_reason.astype(str).str.startswith('deadline_override_').sum()),least_lateness=int(detail.decision_reason.eq('least_predicted_lateness').sum()),capacity_fallbacks=int(detail.decision_reason.eq('predicted_capacity_fallback').sum()))
    return s,dest

def revalue(s,c):
    A=c['coefficients'];ec,cc=s['edge_service_ms_sum'],s['cloud_service_ms_sum']; payload=s['uploaded_decimal_mb']*1000/c['divisor'];n=s['total_tasks'];nc=s['completed_tasks']
    C=ec*A['edge_cost_per_service_ms']+cc*A['cloud_cost_per_service_ms']+payload*A['network_cost_per_mb'];E=ec*A['edge_energy_per_service_ms']+cc*A['cloud_energy_per_service_ms']+payload*A['network_energy_per_mb']
    return dict(uploaded_proxy_mb=payload,total_cost_proxy=C,total_energy_proxy=E,cost_proxy_per_offered_task=C/n,energy_proxy_per_offered_task=E/n,avg_cost_proxy=C/nc if nc else np.nan,avg_energy_proxy=E/nc if nc else np.nan)

def proxy_worker(job):
    block,net,bw,backend=job;df,p=proxy_input(block);cases=coefficient_cases();es,cs=services(df,p);payload=df.task_size_kb.to_numpy();rows=[];w=core.workload_fingerprint(df)
    for pol in POLICIES:
        cache={};base_route=None
        for name,c in cases.items():
            sig,_=signature(es,cs,payload,c) if pol in ADAPT else ('proxy_blind',None)
            if sig not in cache:
                s,d=run(df,p,pol,net,bw,c,backend);cache[sig]=(s,d,name);how='event_replay'
            else:
                s,d,source=cache[sig];how='exact_revaluation'
                # Challenge cached results by replay on predeclared, identical-route tests.
                if block=='affine600' and net=='shared_fcfs' and bw==25 and name in ['common_x2','base_cost_only'] and pol in ['deadline_guarded','utilization_based']:
                    s2,d2=run(df,p,pol,net,bw,c,backend);np.testing.assert_array_equal(d,d2)
                    for k,v in revalue(s,c).items():
                        if k in s2:np.testing.assert_allclose(s2[k],v,rtol=2e-11,atol=1e-8)
                    how='independent_cache_check_replay'
            if base_route is None:base_route=d.copy()
            ans=dict(s);ans.update(revalue(s,c));ans.update(workload_sha256=w,environment_sha256=hash_obj(dict(p,network=net,bw=bw,coef=c)),block=block,setting=name,cost_ratio=c['cost_ratio'],energy_ratio=c['energy_ratio'],network_multiplier=c['wan_multiplier'],proxy_cost_weight=c['proxy_cost_weight'],proxy_energy_weight=c['proxy_energy_weight'],proxy_divisor=c['divisor'],bandwidth_mbps=bw,physical_env_sha256=hash_obj(dict(p,network=net,bw=bw)),route_changes_from_own_base=int(np.sum(d!=base_route)),evaluation_method=how,route_basis_setting=cache[sig][2],backend=backend)
            rows.append(ans)
    return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    study_engine.native()
    jobs=[(f'{s}_{r}',n,b,'fast') for s in ['high_load','burst_load'] for r in [1,2,3] for n,b in NETWORKS]
    with ProcessPoolExecutor(args.workers) as pool:
        out=pd.DataFrame([row for rows in pool.map(proxy_worker,jobs) for row in rows])
    assert len(out)==3744
    folder=RESULTS/'synthetic_proxy'; folder.mkdir(parents=True,exist_ok=True)
    out.to_csv(folder/'proxy_runs.csv',index=False)
    print('Saved 3,744 synthetic coefficient rows on six fixed workloads (not independent deployments).')
