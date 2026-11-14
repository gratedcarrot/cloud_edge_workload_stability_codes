"""Matched run-level estimates; no task is counted as an independent replicate."""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
from scipy import stats
from _paths import RESULTS, result_file
from study_engine import POLICIES
DIR=RESULTS/'main'; DIR.mkdir(parents=True,exist_ok=True)
GX=['scenario','network_model','bandwidth_mbps']
PRE=['hybrid_cloud_edge','deadline_aware_hybrid','utilization_based','network_aware','queue_length_based']
def holm(p):
    p=np.asarray(p,dtype=float);s=np.argsort(p,kind='stable');adjusted=np.empty_like(p);adjusted[s]=np.minimum(1.,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[s]));return adjusted
def ci(x):
    x=np.asarray(x,dtype=float);n=len(x);m=float(x.mean());sd=float(x.std(ddof=1));z=stats.t.ppf(.975,n-1)*sd/np.sqrt(n);return m,sd,m-z,m+z
def write(df,path):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:df.to_csv(f,index=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def main():
    data=pd.read_csv(result_file('main/runs.csv'))
    assert len(data)==3600 and data.groupby(GX+['mode']).run_id.nunique().eq(30).all()
    assert data.groupby(['scenario','run_id']).workload_sha256.nunique().eq(1).all()
    metrics=['on_time_completion_rate','avg_response_time_ms','p95_response_time_ms','throughput_tasks_per_s','blocking_probability','sla_violation_rate','cloud_routing_fraction','cloud_offloading_ratio','cost_proxy_per_offered_task','energy_proxy_per_offered_task','avg_cost_proxy','avg_energy_proxy','edge_utilization','cloud_utilization','avg_network_queue_wait_ms','avg_waiting_time_ms','avg_edge_queue_length','avg_cloud_queue_length','proxy_tie_count','proxy_tie_changed_count','uplink_utilization']
    estimates=[]
    for keys,g in data.groupby(GX+['mode']):
        d=dict(zip(GX+['mode'],keys));d['n_runs']=len(g)
        for metric in metrics:
            vals=g[metric].dropna()
            mean,sd,lo,hi=ci(vals) if len(vals)>=2 else (float('nan'),)*4
            d.update({metric:mean,metric+'_sd':sd,metric+'_ci_lo':lo,metric+'_ci_hi':hi,metric+'_n':len(vals)})
        estimates.append(d)
    write(pd.DataFrame(estimates),DIR/'policy_summary.csv')
    signs=np.random.default_rng(20261103).choice([-1.,1.],size=(50000,30))
    pairs=[]
    for keys,g in data.groupby(GX):
        meta=dict(zip(GX,keys)); tab=g.pivot(index='run_id',columns='mode',values='on_time_completion_rate').sort_index()[POLICIES].to_numpy()
        n,k=tab.shape;assert (n,k)==(30,8)
        for comp in PRE:
            diff=(tab[:,7]-tab[:,POLICIES.index(comp)])*100
            m,sd,lo,hi=ci(diff)
            # Conditional sign-flip: paired differences assumed exchangeable under a sign reversal at the null.
            p=(1+np.count_nonzero(np.abs(signs@diff/n)>=abs(m)-1e-12))/(len(signs)+1)
            pairs.append({**meta,'candidate':'deadline_guarded','comparator':comp,'n_paired_runs':n,'delta_mean_pp':m,'delta_sd_pp':sd,'ci95_lower_pp':lo,'ci95_upper_pp':hi,'candidate_wins_runs':int(np.sum(diff>1e-12)),'same_runs':int(np.sum(abs(diff)<=1e-12)),'candidate_losses_runs':int(np.sum(diff< -1e-12)),'signflip_p':p})
    pc=pd.DataFrame(pairs);pc['holm_p_all_75']=holm(pc.signflip_p);pc['reject_holm_0p05']=pc.holm_p_all_75<.05
    write(pc,DIR/'paired_primary.csv')
    compact=pd.DataFrame(estimates).pivot(index=GX,columns='mode',values='on_time_completion_rate')*100
    write(compact.reset_index(),DIR/'ontime_percent_table.csv')
    print('120 policy cells; 75 paired contrasts; Holm-rejected contrasts:',int(pc.reject_holm_0p05.sum()))
    print(pc[(pc.network_model=='shared_fcfs')&(pc.bandwidth_mbps==25)&(pc.scenario.isin(['high_load','extended_workload']))][['scenario','comparator','delta_mean_pp','ci95_lower_pp','ci95_upper_pp','holm_p_all_75']].round(5).to_string(index=False))
if __name__=='__main__':main()
