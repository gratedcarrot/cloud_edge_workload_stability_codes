"""Finite M/M/c/K birth-death reference and queue-level verification.

Runs the frozen simulator's FCFS scheduler without a policy or network.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import logsumexp
from scipy.stats import t as student_t

from _paths import ROOT, RESULTS
HERE=ROOT
def load_core(path):
    spec=importlib.util.spec_from_file_location('analytical_frozen_core',str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def theory(lam,mu,c,K):
    if not (np.isfinite(lam) and np.isfinite(mu) and lam>0 and mu>0 and
            isinstance(c,int) and isinstance(K,int) and K>=c>=1):
        raise ValueError('Require finite positive rates and integers K>=c>=1')
    n=np.arange(K+1)
    logw=np.r_[0.,np.cumsum(np.log(lam)-np.log(np.minimum(n[1:],c)*mu))]
    pi=np.exp(logw-logsumexp(logw))
    assert np.isclose(pi.sum(),1)
    assert np.allclose(pi[:-1]*lam,pi[1:]*np.minimum(n[1:],c)*mu)
    # Sum active birth states instead of 1-pi[K] to avoid cancellation.
    eff=lam*pi[:-1].sum(); L=n@pi; Lq=np.maximum(n-c,0)@pi
    u=np.minimum(n,c)@pi/c
    assert np.isclose(eff,mu*c*u)
    return pi,{'blocking_fraction':pi[-1],'completions_per_s':eff,'admissions_per_s':eff,
               'sojourn_ms':1000*L/eff,'wait_ms':1000*Lq/eff,'mean_occupancy':L,
               'mean_queue':Lq,'server_utilization':u}

def transient_tv(lam,mu,c,K,seconds,pi):
    Q=np.zeros((K+1,K+1))
    for n in range(K+1):
        if n<K:Q[n,n+1]=lam
        if n:Q[n,n-1]=mu*min(n,c)
        Q[n,n]=-Q[n].sum()
    return 0.5*np.abs(expm(Q*seconds)[0]-pi).sum()

def schedule(core,a_ms,s_ms,c,q):
    layer=core.make_empty_layer_state(c); n=len(a_ms)
    finish=np.full(n,np.nan); wait=np.full(n,np.nan)
    for i,(a,service) in enumerate(zip(a_ms,s_ms)):
        decision=core.schedule_on_layer(layer,float(a),float(service),q)
        if decision['accepted']:
            finish[i]=decision['service_finish_time_ms']
            wait[i]=decision['waiting_time_ms']
    return finish,wait

def replicate(core,lam,mu,c,K,warm,obs,index,rep):
    total=obs+warm
    ra,rs=[np.random.default_rng(np.random.SeedSequence([8110317,index,rep,stream])) for stream in (0,1)]
    N=ra.poisson(lam*total)
    a=np.sort(ra.uniform(0,total,size=N))*1000.
    s=rs.exponential(1000/mu,N)
    finish,wait=schedule(core,a,s,c,K-c)
    keep=np.isfinite(finish); measured=a>=1000*warm; cohort=keep & measured
    # Closed at start, open at end; no rounding/jitter is used.
    times=np.r_[a[keep],finish[keep]]; changes=np.r_[np.ones(keep.sum(),int),-np.ones(keep.sum(),int)]
    order=np.lexsort((changes,times)); times=times[order]
    states=np.cumsum(changes[order]); assert states.min()>=0 and states.max()<=K
    boundaries=np.clip(times,1000*warm,1000*total)
    widths=np.diff(boundaries,append=1000*total)
    area=np.bincount(states,weights=widths,minlength=K+1)
    area[0]+=boundaries[0]-1000*warm
    state_probs=area/(1000*obs); assert np.isclose(state_probs.sum(),1)
    n=np.arange(K+1)
    vals={'blocking_fraction':(measured&~keep).sum()/measured.sum(),
          'completions_per_s':((finish>=warm*1000)&(finish<total*1000)).sum()/obs,
          'admissions_per_s':cohort.sum()/obs,
          'sojourn_ms':np.mean(finish[cohort]-a[cohort]),'wait_ms':np.mean(wait[cohort]),
          'mean_occupancy':n@state_probs,'mean_queue':np.maximum(n-c,0)@state_probs,
          'server_utilization':np.minimum(n,c)@state_probs/c,
          'offered_in_window':int(measured.sum()),'admitted_in_window':int(cohort.sum())}
    return vals,state_probs

def ci(values):
    values=np.asarray(values,float); n=len(values); m=values.mean()
    h=student_t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    return float(m),float(m-h),float(m+h)

def run(corepath,out):
    out.mkdir(exist_ok=True,parents=True);core=load_core(corepath)
    c,K,mu,warm,obs=4,12,7.,180.,1800.
    runs=[];summary=[];states=[];verification=[]
    for k,rho in enumerate([.6,1,1.4]):
        lam=mu*c*rho;pi,exact=theory(lam,mu,c,K); tv=transient_tv(lam,mu,c,K,warm,pi)
        case=[];freq=[]
        for rep in range(1,31):
            vals,p=replicate(core,lam,mu,c,K,warm,obs,k,rep)
            case.append(vals);freq.append(p)
            runs.append(dict(rho=rho,lambda_per_s=lam,mu_per_s=mu,c=c,K=K,run=rep,warmup_s=warm,measurement_s=obs,**vals))
        for metric,target in exact.items():
            mean,lo,hi=ci([x[metric] for x in case]);summary.append(dict(rho=rho,lambda_per_s=lam,mu_per_s=mu,c=c,K=K,replications=30,
                   metric=metric,analytical=target,simulation_mean=mean,ci95_low=lo,ci95_high=hi,
                   simulation_minus_analytical=mean-target,within_pointwise_ci=bool(lo<=target<=hi)))
        for n in range(K+1):
            mean,lo,hi=ci([p[n] for p in freq])
            states.append(dict(rho=rho,state=n,analytical_pi=pi[n],time_fraction_mean=mean,ci95_low=lo,ci95_high=hi))
        verification.append(dict(rho=rho,stationary_sum=float(pi.sum()),
            max_local_balance_residual=float(np.abs(pi[:-1]*lam-pi[1:]*np.minimum(np.arange(1,K+1),c)*mu).max()),
            total_variation_from_stationarity_after_warmup=tv))
    pd.DataFrame(runs).to_csv(out/'mmck_runs.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'mmck_comparison.csv',index=False)
    pd.DataFrame(states).to_csv(out/'mmck_state_probabilities.csv',index=False)
    (out/'mmck_verification.json').write_text(json.dumps({'core_sha256':hashlib.sha256(corepath.read_bytes()).hexdigest(),
         'replication_count':len(runs),'measured_offered':sum(x['offered_in_window'] for x in runs),
         'warmup_s':warm,'measurement_s':obs,'tests':verification},indent=2)+'\n')
    print(pd.DataFrame(summary).query("metric in ['blocking_fraction','completions_per_s','sojourn_ms']").to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core',type=Path,default=ROOT/'engine/reference/frozen_policy_engine.py')
    p.add_argument('--out',type=Path,default=RESULTS/'analytical')
    args=p.parse_args();run(args.core,args.out)
