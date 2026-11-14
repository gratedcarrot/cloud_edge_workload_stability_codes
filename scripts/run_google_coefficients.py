"""Additional coefficient-only sensitivity on 20,000 jobs, every policy rerun."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools,argparse
import numpy as np,pandas as pd
from _paths import result_file
import study_engine as eng
from run_google_mapping import INPUT,OUT,NETWORKS
BASE_COEF={k:eng.DEFAULTS[k] for k in ('edge_cost_per_service_ms','cloud_cost_per_service_ms','network_cost_per_mb','edge_energy_per_service_ms','cloud_energy_per_service_ms','network_energy_per_mb')}
def coefficient_cases():
    rows={}
    def add(label,rc=6.25,re=4.5,wn=1.,cw=1.,ew=1.,mult=1.,div=1000.):
        c=dict(BASE_COEF);c.update(cloud_cost_per_service_ms=c['edge_cost_per_service_ms']*rc,cloud_energy_per_service_ms=c['edge_energy_per_service_ms']*re,network_cost_per_mb=c['network_cost_per_mb']*wn,network_energy_per_mb=c['network_energy_per_mb']*wn)
        rows[label]=dict(coefficients={k:v*mult for k,v in c.items()},proxy_cost_weight=cw,proxy_energy_weight=ew,divisor=div,cost_ratio=rc,energy_ratio=re,wan_multiplier=wn,common_multiplier=mult)
    add('base')
    for (i,rc),(j,re) in itertools.product(enumerate([1.,18/7,6.25,12.5]),enumerate([1.,18/7,4.5,9.])):
        if i==2 and j==2:continue
        add(f'ratio_c{i}_e{j}',rc,re)
    for k in [0,10]:add(f'network_x{k}',wn=k)
    for desc,rc,re in [('base',6.25,4.5),('opposed',6.25,1.)]:
        add(desc+'_cost_only',rc,re,cw=1.,ew=0.)
        add(desc+'_energy_only',rc,re,cw=0.,ew=1.)
        add(desc+'_equal_normalized',rc,re,cw=1/(BASE_COEF['edge_cost_per_service_ms']*1000/7),ew=1/(BASE_COEF['edge_energy_per_service_ms']*1000/7))
    add('common_x2',mult=2.);add('legacy_1024',div=1024.)
    assert len(rows)==26
    return rows

def run(job):
    anchor,net,bw=job;df=pd.read_csv(INPUT)
    span=float(df.arrival_time_s.max())
    if anchor=='affine600':
        df['arrival_time_s']*=600./span;span=600.
    w=eng.prepare(df);out=[]
    for label,c in coefficient_cases().items():
        p=dict(duration_s=span,proxy_cost_weight=c['proxy_cost_weight'],proxy_energy_weight=c['proxy_energy_weight'])
        for pol in eng.POLICIES:
            s,_=eng.simulate(w,p,pol,net,bw,coefficients=c['coefficients'],proxy_divisor=c['divisor'],backend='native')
            out.append(dict(anchor=anchor,coefficient_case=label,**s))
    return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=2);args=a.parse_args()
    jobs=[(a,n,b) for a,(n,b) in itertools.product(['native','affine600'],NETWORKS)]
    with ProcessPoolExecutor(args.workers) as ex:out=pd.DataFrame([s for rows in ex.map(run,jobs) for s in rows])
    assert len(out)==1248
    k=['anchor','network_model','bandwidth_mbps','mode'];base=out.query('coefficient_case=="base"').set_index(k);ori=pd.read_csv(result_file('google/base_outcomes_20k.csv')).set_index(k).loc[base.index]
    for col in ['completed_tasks','on_time_completed_tasks','avg_cost_proxy','avg_energy_proxy']:np.testing.assert_allclose(base[col],ori[col],rtol=1e-12,atol=1e-12)
    assert out.groupby(['anchor','network_model','bandwidth_mbps','coefficient_case']).environment_sha256.nunique().eq(1).all()
    out.to_csv(OUT/'coefficient_sensitivity_20k.csv',index=False)
    print('verified 1,248 fully recomputed policy evaluations including base parity on 48 cells')
