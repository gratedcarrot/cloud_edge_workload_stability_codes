"""Secondary paired comparisons of the two submitted hybrids vs three simpler baselines.
Account for multiplicity across all 2 x 3 x 15 exploratory contrasts.
"""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
from _paths import RESULTS, result_file
h=RESULTS/'tables'; h.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(result_file('main/runs.csv')); signs=np.random.default_rng(20261104).choice([-1.,1.],size=(50000,30));g=['scenario','network_model','bandwidth_mbps']
rows=[]
for cell,frame in d.groupby(g):
    assert frame.groupby('seed').environment_sha256.nunique().eq(1).all() and frame.groupby('seed').workload_sha256.nunique().eq(1).all()
    x=frame.pivot(index='seed',columns='mode',values='on_time_completion_rate')
    for a in ('hybrid_cloud_edge','deadline_aware_hybrid'):
        for b in ('utilization_based','network_aware','queue_length_based'):
            diff=(x[a]-x[b]).to_numpy()*100;n=len(diff);assert n==30
            m=diff.mean();ci=t.ppf(.975,n-1)*np.std(diff,ddof=1)/np.sqrt(n)
            p=(1+np.count_nonzero(np.abs(signs@diff/n)>=abs(m)-1e-12))/(len(signs)+1)
            rows.append(dict(zip(g,cell),hybrid=a,baseline=b,n_pairs=n,mean_difference_pp=m,ci95_low_pp=m-ci,ci95_high_pp=m+ci,raw_signflip_p=p))
r=pd.DataFrame(rows);assert len(r)==90
p=r.raw_signflip_p.to_numpy();idx=np.argsort(p,kind='stable');adj=np.empty_like(p);adj[idx]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[idx]))
r['holm_p_90']=adj;r.to_csv(h/'secondary_baselines.csv',index=False)
print(r.query("scenario=='high_load' and bandwidth_mbps==25 and hybrid=='hybrid_cloud_edge' and baseline=='utilization_based'").to_string(index=False))
