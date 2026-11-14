"""Run-level confidence intervals and the four-policy secondary tests in Table VIII."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from _paths import RESULTS, result_file
OUT=RESULTS/'tables'; OUT.mkdir(parents=True,exist_ok=True)
SC=['low_load','medium_load','high_load','burst_load','extended_workload']; SL=['Low','Medium','High','Burst','Extended']
POL=['edge_only','cloud_only','hybrid_cloud_edge','deadline_aware_hybrid','utilization_based','network_aware','queue_length_based','deadline_guarded']
PL=['Edge-only','Cloud-only','Queue hybrid','Deadline (original)','Utilization','Network-aware','Queue length','Deadline (guarded)']
M=['avg_response_time_ms','avg_waiting_time_ms','throughput_tasks_per_s','blocking_probability','cloud_offloading_ratio','sla_violation_rate','avg_cost_proxy','avg_energy_proxy','on_time_completion_rate']
MNAME=['Response time','Waiting time','Throughput','Blocking probability','Cloud offloading ratio','SLA violation rate','Cost proxy','Energy proxy','On time / offered']
F=[1,1,1,100,100,100,1,1,100];D=[2,2,2,2,2,2,5,4,2]
def summarize(g):
    keys=list(g.keys); out=[]
    for ident,frame in g:
        assert len(frame)==30 and frame.seed.nunique()==30
        for j,m in enumerate(M):
            x=frame[m].to_numpy()*F[j];mean=x.mean(); half=stats.t.ppf(.975,29)*x.std(ddof=1)/np.sqrt(30)
            out.append(dict(zip(keys,ident),metric=m,mean=mean,halfwidth=half,lower=mean-half,upper=mean+half,runs=len(x)))
    return pd.DataFrame(out)
main=pd.read_csv(result_file('main/runs.csv')); sweep=pd.read_csv(result_file('sensitivity/sweep_runs.csv')); cap=pd.read_csv(result_file('sensitivity/capacity_runs.csv'))
assert len(main)==3600 and main.groupby(['scenario','run_id','network_model','bandwidth_mbps']).environment_sha256.nunique().eq(1).all()
assert main.groupby(['scenario','run_id']).workload_sha256.nunique().eq(1).all()
g=['scenario','mode','network_model','bandwidth_mbps']
mg=summarize(main.groupby(g)); mg.to_csv(OUT/'main_original_metrics_ci.csv',index=False)
sg=summarize(pd.concat([sweep,cap],ignore_index=True).groupby(['scenario','network_model','parameter','level']));sg.to_csv(OUT/'sensitivity_original_metrics_ci.csv',index=False)
orig=main[(main.network_model=='independent') & (main['mode'].isin(POL[:4]))]
def holm(p):
    p=np.array(p);i=np.argsort(p,kind='stable');o=np.empty_like(p);o[i]=np.minimum(1,np.maximum.accumulate((len(i)-np.arange(len(i)))*p[i]));return o
tests=[]
for sc in SC:
    for m in M[:8]:
        x=orig[orig.scenario==sc].pivot(index='seed',columns='mode',values=m)[POL[:4]].to_numpy();n,k=x.shape; assert(n,k)==(30,4)
        changed=bool(np.any(np.ptp(x,axis=1)>0)); cen=np.eye(k)-np.ones((k,k))/k
        # The all-zero blocking cell has no estimable routing effect; include p=1 in the Holm family.
        if changed:
            v=cen@np.cov(x,rowvar=False)@cen;vv=np.trace(v@v);eps=float(np.clip(np.trace(v)**2/((k-1)*vv),1/(k-1),1)) if vv>0 else 1.
            avg=x.mean();cm=x.mean(axis=0);rm=x.mean(axis=1,keepdims=True)
            eff=n*((cm-avg)**2).sum();err=((x-rm-cm+avg)**2).sum()
            f=(eff/(k-1))/(err/((n-1)*(k-1))) if err else float('inf')
            p=stats.f.sf(f,eps*(k-1),eps*(n-1)*(k-1));fr,fp=stats.friedmanchisquare(*x.T)
        else:eps=f=fr=np.nan;p=fp=1.
        tests.append(dict(scenario=sc,metric=m,n_blocks=n,k_policies=k,estimable=changed,gg_epsilon=eps,rm_anova_f=f,rm_gg_p=p,friedman_chi2=fr,friedman_p=fp))
tt=pd.DataFrame(tests);tt['rm_gg_holm_40']=holm(tt.rm_gg_p);tt['friedman_holm_40']=holm(tt.friedman_p);tt.to_csv(OUT/'secondary_four_policy_tests.csv',index=False)

print('Saved matched summaries and 40 four-policy ANOVA/Friedman comparisons.')
