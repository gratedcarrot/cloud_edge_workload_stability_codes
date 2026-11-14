"""Quantify the limits of the stationary single-tier surrogate on actual inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from _paths import ROOT, DATA, RESULTS
from check_mmck import HERE,load_core,theory,schedule,ci

def exp_distance(x):
    """Supremum EDF distance from an exponential of fitted mean; no p-value."""
    v=np.sort(np.asarray(x,float)); n=len(v)
    if n<2 or v.mean()<=0:return np.nan
    target=-np.expm1(-v/v.mean()); f=np.arange(n)/n
    return float(max(np.abs(f-target).max(),np.abs(f+1/n-target).max()))

def audit(core,path,scenario,run,seed,T,lam,classification):
    df=pd.read_csv(path,usecols=['arrival_time_s','cpu_demand_mi'])
    a=df.arrival_time_s.to_numpy(float); assert np.all(np.diff(a)>=0)
    cpu=df.cpu_demand_mi.to_numpy(float)
    s=np.asarray([core.service_time_ms(x,7.) for x in cpu])
    if T is None:T=float(a.max()-a.min())
    if lam is None:lam=len(a)/T
    gaps=np.diff(a)
    nsec=int(np.floor(T+1e-10)); early=a[a<nsec]
    counts=np.bincount(np.floor(early).astype(int),minlength=nsec)
    assert len(counts)==nsec
    finished,wait=schedule(core,1000*a,s,4,120); keep=np.isfinite(finished)
    _,exact=theory(lam,1000/s.mean(),4,124)
    span_drain_s=max(T,float(np.nanmax(finished))/1000)
    vals=dict(scenario=scenario,run=run,seed=seed,file=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              scope=classification,horizon_s=T,N=len(a),effective_mu_per_s=1000/s.mean(),
              lambda_for_theory=lam,observed_rate_per_s=len(a)/T,
              gap_cv=gaps.std(ddof=1)/gaps.mean(),gap_zero_fraction=np.mean(gaps==0),
              gap_mean_s=gaps.mean(),gap_exp_D=exp_distance(gaps),
              service_mean_ms=s.mean(),service_cv=s.std(ddof=1)/s.mean(),service_exp_D=exp_distance(s),
              dispersion_of_1s_counts=counts.var(ddof=1)/counts.mean(),complete_1s_bins=nsec,mean_count_per_complete_bin=counts.mean(),
              accepted_tasks=int(keep.sum()),blocked_tasks=int((~keep).sum()),
              empirical_blocking_fraction=(~keep).mean(),empirical_sojourn_ms=np.mean(finished[keep]-1000*a[keep]),
              empirical_wait_ms=wait[keep].mean(),empirical_admissions_per_s=keep.sum()/T,
              empirical_departures_per_s=np.sum(finished<T*1000)/T,empirical_drained_throughput=keep.sum()/span_drain_s,
              drain_horizon_s=span_drain_s,analytical_blocking_fraction=exact['blocking_fraction'],
              analytical_sojourn_ms=exact['sojourn_ms'],analytical_wait_ms=exact['wait_ms'],
              analytical_admissions_per_s=exact['admissions_per_s'])
    for key in ['blocking_fraction','sojourn_ms','wait_ms','admissions_per_s']:
        vals['observed_minus_analytical_'+key]=vals['empirical_'+key]-vals['analytical_'+key]
    if scenario=='burst_load':
        middle=(a>=100)&(a<160)
        vals['empirical_burst_rate_per_s']=middle.sum()/60
        vals['empirical_nonburst_rate_per_s']=(~middle).sum()/240
    return vals

def run(main_data,corepath,google_dir,out):
    out.mkdir(exist_ok=True,parents=True);core=load_core(corepath)
    manifest=pd.read_csv(main_data/'workloads.csv')
    assert len(manifest)==150 and (manifest.groupby('scenario').size()==30).all()
    params={d['scenario']:d for d in json.loads((main_data/'scenarios.json').read_text())}
    all_rows=[]
    for x in manifest.itertuples(index=False):
        p=params[x.scenario]; horizon=float(p['duration_s']); lam=float(p['base_lambda_tps'])
        if x.scenario=='burst_load':
            b=(p['burst_end_s']-p['burst_start_s'])/horizon
            lam=lam*(1-b)+float(p['burst_lambda_tps'])*b
        row=audit(core,main_data/x.path,x.scenario,x.run_id,x.seed,horizon,lam,'fresh_synthetic')
        assert row['source_sha256']==x.file_sha256 and row['N']==x.offered
        all_rows.append(row)
    for scenario in ['google_native','google_affine']:
        all_rows.append(audit(core,google_dir/(scenario+'.csv.gz'),scenario,1,None,None,None,'one_fixed_trace_window'))
    df=pd.DataFrame(all_rows);df.to_csv(out/'workload_assumptions_runs.csv',index=False)
    detail=[]
    metrics=['gap_cv','gap_exp_D','gap_zero_fraction','service_cv','service_exp_D','dispersion_of_1s_counts',
             'observed_rate_per_s','lambda_for_theory','effective_mu_per_s','service_mean_ms',
             'empirical_burst_rate_per_s','empirical_nonburst_rate_per_s','empirical_departures_per_s','empirical_drained_throughput']
    metrics += [f'{prefix}_{metric}' for metric in ['blocking_fraction','admissions_per_s','sojourn_ms','wait_ms']
                for prefix in ['empirical','analytical','observed_minus_analytical']]
    for scenario,subset in df.groupby('scenario',sort=False):
        for m in metrics:
            vals=subset[m].dropna().to_numpy()
            if not len(vals):continue
            mean,lo,hi=ci(vals)
            detail.append(dict(scenario=scenario,metric=m,replications=len(vals),offered_task_records=subset.N.sum(),
                               mean=mean,ci95_low=lo,ci95_high=hi))
    detail=pd.DataFrame(detail);detail.to_csv(out/'workload_assumptions_summary.csv',index=False)
    wide=detail.pivot(index='scenario',columns='metric',values='mean')
    wide.reindex(list(params)+['google_native','google_affine']).to_csv(out/'workload_assumptions_table.csv')
    truth=df[df.scenario.str.startswith('google_')]
    for m in ['gap_cv','gap_exp_D','gap_zero_fraction','service_cv','service_exp_D','service_mean_ms']:
        assert np.isclose(truth.iloc[0][m],truth.iloc[1][m],rtol=1e-11),m
    (out/'workload_verification.json').write_text(json.dumps({'synthetic_files_sha256_checked':len(manifest),
        'synthetic_offered_tasks':int(df[df.scope=='fresh_synthetic'].N.sum()),
        'google_observed_arrivals':2000,'google_time_axes_same_window':2,
        'normalized_gap_and_service_shape_invariant_under_google_clock_scaling':True,
        'intervals':'30-run two-sided pointwise Student-t; no interval for a single fixed trace window',
        'probability_fitting':'CDF supremum distances use fitted exponential means; no iid or pooled task p-values',
        'core_sha256':hashlib.sha256(corepath.read_bytes()).hexdigest()},indent=2)+'\n')
    print(wide[['gap_cv','gap_exp_D','service_cv','service_exp_D','dispersion_of_1s_counts','analytical_blocking_fraction',
        'empirical_blocking_fraction','analytical_sojourn_ms','empirical_sojourn_ms']].round(4).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-data',type=Path,default=DATA/'synthetic')
    p.add_argument('--core',type=Path,default=ROOT/'engine/reference/frozen_policy_engine.py')
    p.add_argument('--google-dir',type=Path,default=DATA/'diagnostic_google')
    p.add_argument('--out',type=Path,default=RESULTS/'analytical')
    a=p.parse_args();run(a.main_data,a.core,a.google_dir,a.out)
