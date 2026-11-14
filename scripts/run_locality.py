"""Static coverage and local-queue sensitivity with constant aggregate compute capacity."""
from pathlib import Path
import argparse, hashlib, json, os, ctypes, subprocess, tempfile, shutil
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import t
from _paths import DATA, RESULTS, ROOT
import study_engine as eng
HERE=ROOT; OUT=RESULTS/'locality'; OUT.mkdir(parents=True,exist_ok=True)
RADII=(450,600,750); NETS=(25.,250.)
def dist(a,b):
    ar,br=np.deg2rad(a),np.deg2rad(b);p=ar[:,None,0];q=br[None,:,0]
    d=np.sin((q-p)/2)**2+np.cos(p)*np.cos(q)*np.sin((br[None,:,1]-ar[:,None,1])/2)**2
    return 12742017.6*np.arcsin(np.sqrt(np.clip(d,0,1)))
@lru_cache(None)
def geography():
    sites=pd.read_csv(DATA/'eua/eua_candidate_sites.csv').sort_values('candidate_site_id').reset_index(drop=True)
    users=pd.read_csv(DATA/'eua/eua_generated_users.csv').sort_values('generated_user_id').reset_index(drop=True)
    sp=sites[['latitude_deg','longitude_deg']].to_numpy();up=users[['latitude_deg','longitude_deg']].to_numpy()
    pick=[dist(up.mean(axis=0,keepdims=True),sp)[0].argmin()]
    while len(pick)<4:
        ds=dist(sp,sp[pick]).min(axis=1);ds[pick]=-1;pick.append(ds.argmax())
    rank=dict(zip(pick,range(1,5))); pick=sorted(pick);active=sites.iloc[pick].copy()
    active['selection_order']=[rank[i] for i in pick];active['local_index']=range(4)
    d=dist(up,sp[pick]);j=d.argmin(axis=1);dmin=d[np.arange(len(up)),j]
    m=users[['generated_user_id']].copy();m['nearest_active_site_id']=active.candidate_site_id.to_numpy()[j];m['nearest_index']=j;m['distance_m']=dmin
    return active,m
@lru_cache(None)
def mod():
    src=ROOT/'engine/spatial_engine.cpp';so=Path(tempfile.gettempdir())/('eua_'+hashlib.sha256(src.read_bytes()).hexdigest()[:20]+('.dll' if os.name=='nt' else '.so'))
    if not so.exists():
        temp=str(so)+'.'+str(os.getpid())+'.tmp'
        subprocess.run([os.environ.get('CXX',shutil.which('g++') or 'clang++'),'-O3','-std=c++17','-ffp-contract=off','-shared']+([] if os.name=='nt' else ['-fPIC'])+[str(src),'-o',temp],check=True);os.replace(temp,so)
    f=ctypes.CDLL(str(so)).sim_eua;f64=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');i32=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int,f64,f64,ctypes.c_int,ctypes.c_int,i32,ctypes.c_int,f64];f.restype=ctypes.c_int;return f
def arrayout(w,p,bw,pol,j,sites):
    p=eng.effective(p,'shared_fcfs',bw,None,1000);cfg=np.array([float(p[k]=='paper_utilization') if k=='network_baseline_rule' else p[k] for k in eng.DEFAULTS],dtype=np.float64)
    n=len(w.x);out=np.empty((n,25),dtype=np.float64)
    assert mod()(n,w.x,cfg,eng.POLICIES.index(pol),1,np.ascontiguousarray(j,dtype=np.int32),sites,out)==0
    return out
def work(job):
    data,run=job
    p=next(i for i in json.loads((data/'scenarios.json').read_text()) if i['scenario']=='high_load')
    w=eng.prepare(pd.read_csv(data/f'high_load_{run:02d}.csv.gz'));_,m=geography()
    n=len(w.x);owner=np.random.Generator(np.random.PCG64(792000+run)).integers(len(m),size=n,dtype=np.int32);jj=m.nearest_index.to_numpy()[owner];dd=m.distance_m.to_numpy()[owner]
    h=hashlib.sha256(owner.astype('<i4').tobytes()).hexdigest()
    rows=[]
    for bw in NETS:
      cloud=None
      for topology,r in [('unrestricted_pool',0)]+[(structure,radius) for radius in RADII for structure in ('coverage_pool','four_local_queues')]:
        j=np.zeros(n,dtype=np.int32) if r==0 else np.where(dd<=r,(0 if topology=='coverage_pool' else jj),-1).astype(np.int32)
        sites=4 if topology=='four_local_queues' else 1
        for pol in eng.POLICIES:
          o=arrayout(w,p,bw,pol,j,sites)
          dst,ok=o[:,0],o[:,1].astype(bool);late=ok&(o[:,6]>w.x[:,3]+1e-9);on=ok&~late
          assert ((dst==1)|(dst==2)|(dst==0)).all() and not ((j<0)&(dst==0)).any()
          if pol=='cloud_only':
            if cloud is None:cloud=o.copy()
            else: np.testing.assert_array_equal(o[:,[0,1,2,3,6,7,8]],cloud[:,[0,1,2,3,6,7,8]])
          if topology=='unrestricted_pool' and run==1:
            summ,_=eng.simulate(w,p,pol,'shared_fcfs',bw,backend='native');assert summ['on_time_completed_tasks']==on.sum() and summ['blocked_tasks']==(~ok).sum()
            np.testing.assert_allclose(o[ok,6].mean(),summ['avg_response_time_ms'],atol=1e-10,rtol=1e-12)
          rows.append(dict(scenario='high_load',run=run,assignment_seed=792000+run,workload_sha256=w.fingerprint,owner_sha256=h,bandwidth_mbps=bw,topology=topology,radius_m=r,policy=pol,tasks=n,eligible_count=(j>=0).sum(),completed=ok.sum(),blocked=(~ok).sum(),late=late.sum(),on_time=on.sum(),on_time_pct=100.*on.mean(),block_pct=100.*(~ok).mean(),cloud_request_pct=100.*(dst==1).mean(),response_ms=o[ok,6].mean() if ok.any() else np.nan,net_wait_per_offered_ms=o[:,7].mean()))
    return rows
def analyze(out):
    by=['bandwidth_mbps','topology','radius_m','policy'];m=out.groupby(by)[['on_time_pct','block_pct','cloud_request_pct','response_ms']].mean().reset_index();m.to_csv(OUT/'condition_means.csv',index=False)
    key=['bandwidth_mbps','policy','run']
    def view(typ,r):return out[(out.topology==typ)&(out.radius_m==r)].set_index(key).on_time_pct
    rows=[]
    for r in RADII:
      pool=view('coverage_pool',r); full=view('unrestricted_pool',0); local=view('four_local_queues',r)
      for name,delta in [('coverage_pool_minus_unrestricted',pool-full),('local_minus_coverage_pool',local-pool),('local_minus_unrestricted',local-full)]:
       for (bw,policy),x in delta.groupby(level=[0,1]):
        assert len(x)==30 and np.isfinite(x).all()
        lo=t.ppf(.975,29)*x.std(ddof=1)/np.sqrt(30)
        rows.append(dict(radius_m=r,contrast=name,bandwidth_mbps=bw,policy=policy,n=30,mean_diff_on_time_pp=x.mean(),ci95_low_pp=x.mean()-lo,ci95_high_pp=x.mean()+lo))
    pd.DataFrame(rows).to_csv(OUT/'paired_contrasts.csv',index=False)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,default=DATA/'synthetic');a.add_argument('--workers',type=int,default=2);args=a.parse_args()
    active,m=geography();active.to_csv(OUT/'four_active_sites.csv',index=False);m.to_csv(OUT/'nearest_active_site.csv',index=False);mod()
    with ProcessPoolExecutor(args.workers) as ex: out=pd.DataFrame([s for rows in ex.map(work,[(args.data,i) for i in range(1,31)]) for s in rows])
    g=out.groupby(['run','bandwidth_mbps','topology','radius_m'])
    assert len(out)==3360 and g['policy'].nunique().eq(8).all() and g.workload_sha256.nunique().eq(1).all() and g.owner_sha256.nunique().eq(1).all() and (out.blocked+out.late+out.on_time==out.tasks).all()
    out.to_csv(OUT/'run_results.csv',index=False);analyze(out)
    print('verified 3,360 policy/condition/workload combinations; 30 paired independent synthetic workloads; cloud-only invariant')
