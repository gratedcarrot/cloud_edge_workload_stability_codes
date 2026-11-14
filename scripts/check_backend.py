"""Compare native decisions, outcomes and event times with direct copied-state replay."""
from pathlib import Path
import json, numpy as np, pandas as pd
from _paths import DATA, RESULTS
from study_engine import simulate,POLICIES,prepare
HERE=RESULTS
rng=np.random.default_rng(57442)
def case(kind):
    n=360
    t=np.cumsum(rng.exponential(.009 if kind==0 else .030,n));t=np.round(t,2)
    return pd.DataFrame({'task_id':[f'x{i}' for i in range(n)],'arrival_time_s':t,'task_size_kb':rng.choice([0,5,122,1190,3500],n),'cpu_demand_mi':rng.choice([1,60,180,500],n),'deadline_ms':rng.choice([25,130,220,500,1100],n),'edge_cloud_network_delay_ms':rng.choice([0,15,45,180],n),'scenario':'crosscheck','run_id':1,'seed':57442})
counts=0;tasks=0;mx=0.;now=[];covered=set()
def compare(df,p,net,bw,divisor=1000.,policies=POLICIES):
  global counts,tasks,mx
  w=prepare(df)
  keys=['processing_location','requested_destination','status','decision_reason','on_time_completion','sla_violation','proxy_tie_break_applied','proxy_tie_break_changed_destination']
  for mode in policies:
    args=(w,p,mode,net,bw)
    a,da=simulate(*args,proxy_divisor=divisor,backend='native',details=True);b,db=simulate(*args,proxy_divisor=divisor,backend='reference',details=True)
    for k in keys:assert np.array_equal(da[k].to_numpy(),db[k].to_numpy()),(counts,mode,k,da[k].value_counts(),db[k].value_counts())
    for k in ['start_time_ms','finish_time_ms','response_time_ms','waiting_time_ms','service_time_ms','predicted_edge_response_ms','predicted_cloud_response_ms','predicted_edge_wait_ms','predicted_cloud_wait_ms','cloud_arrival_time_ms','network_queue_wait_ms','upload_serialization_ms','cost_proxy','energy_proxy','known_inflight_cloud_tasks','edge_queue_at_decision','cloud_queue_at_decision','edge_busy_servers_at_decision','cloud_busy_servers_at_decision','actual_queue_at_layer_arrival']:
        u=da[k].astype(float).to_numpy();v=db[k].astype(float).to_numpy()
        np.testing.assert_allclose(u,v,atol=1e-8,rtol=1e-12,equal_nan=True,err_msg=mode+':'+k)
        if np.isfinite(u).any():mx=max(mx,float(np.nanmax(abs(u-v))))
    for k in b:
        if isinstance(b[k],(int,float,np.number)):np.testing.assert_allclose(a[k],b[k],rtol=1e-10,atol=1e-8,equal_nan=True,err_msg=k)
        else:assert a[k]==b[k],(k,a[k],b[k])
    counts+=1;tasks+=len(df);covered.update(da.decision_reason.unique());now.append({'case':counts,'mode':mode,'net':net,'bandwidth':bw,'offered':len(df),'blocked':a['blocked_tasks'],'on_time':a['on_time_completed_tasks'],'proxy_divisor':divisor})
if __name__=='__main__':
  for i,cap in enumerate([0,5]):
    df=case(i);p={'duration_s':19,'edge_nodes':2,'cloud_servers':2,'edge_queue_capacity':cap,'cloud_queue_capacity':cap,'edge_service_rate_tps_each':10.,'cloud_service_rate_tps_each':13.}
    for net,bw in [('independent',25),('shared_fcfs',25),('shared_fcfs',250)]:compare(df,p,net,bw)
  df=pd.read_csv(DATA/'synthetic/high_load_01.csv.gz').iloc[:1750]
  p={'duration_s':300}
  compare(df,p,'shared_fcfs',25,1024.,['hybrid_cloud_edge','deadline_aware_hybrid','deadline_guarded'])
  compare(df,p,'independent',25,1000.,['hybrid_cloud_edge','deadline_aware_hybrid','deadline_guarded'])
  (HERE/'verification').mkdir(parents=True,exist_ok=True);pd.DataFrame(now).to_csv(HERE/'verification/native_equivalence.csv',index=False)
  report={'comparisons':counts,'task_evaluations_against_reference':tasks,'max_absolute_per_task_numeric_error':mx,'categories_exact':True,'all_common_summary_fields_checked':True,'decision_reasons_covered':sorted(covered)}
  (HERE/'verification/native_equivalence.json').write_text(json.dumps(report,indent=2));print(report)
