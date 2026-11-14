"""Portable matched-policy API: C++17 accelerated backend or frozen Python model.

Always use decimal MB by default. Only proxy_divisor=1024 reproduces prior accounting.
"""
from pathlib import Path
import hashlib, json, os, sys, subprocess, tempfile, ctypes, shutil, importlib.util
from functools import lru_cache
from dataclasses import dataclass
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
POLICIES=['edge_only','cloud_only','hybrid_cloud_edge','deadline_aware_hybrid','utilization_based','network_aware','queue_length_based','deadline_guarded']
REASONS=['fixed_edge','fixed_cloud','edge_busy_fraction_threshold','paper_network_utilization_rule_adapted','minimum_current_jobs_per_server','hybrid_queue_and_response_rule','predicted_capacity_fallback','deadline_override_to_cloud','deadline_override_to_edge','proxy_tie_break_to_edge','proxy_tie_break_to_cloud','least_predicted_lateness']
KEYS=['arrival_time_s','task_size_kb','cpu_demand_mi','deadline_ms','edge_cloud_network_delay_ms']
DEFAULTS={'edge_nodes':4,'cloud_servers':3,'edge_queue_capacity':120,'cloud_queue_capacity':250,'edge_service_rate_tps_each':7.,'cloud_service_rate_tps_each':18.,'bandwidth_mbps':25.,'threshold_ms':220.,'proxy_divisor':1000.,'edge_cost_per_service_ms':.00004,'cloud_cost_per_service_ms':.00025,'network_cost_per_mb':.00002,'edge_energy_per_service_ms':.004,'cloud_energy_per_service_ms':.018,'network_energy_per_mb':.002,'proxy_cost_weight':1.,'proxy_energy_weight':1.,'deadline_tie_gap_ms':25.,'deadline_tie_slack_ms':100.,'utilization_threshold':.8,'network_utilization_threshold':.8,'network_probe_kb':1024.,'network_baseline_rule':'paper_utilization','network_bandwidth_threshold_mbps':6.}
@dataclass
class Input:
    df:pd.DataFrame
    x:np.ndarray
    fingerprint:str
def prepare(tasks):
    df=tasks.sort_values('arrival_time_s',kind='stable').reset_index(drop=True)
    x=np.ascontiguousarray(df[KEYS].to_numpy(dtype=np.float64))
    if len(df)==0 or not np.isfinite(x).all() or np.min(x)<0 or np.min(x[:,2])<=0:raise ValueError('Empty workload or invalid numeric mark')
    ids=df['task_id'].astype(str)
    if ids.duplicated().any() or ids.str.contains('\x00',regex=False).any():raise ValueError('Each task must have a unique, NUL-free identifier')
    h=hashlib.sha256(b'sncs-v2:ids-utf8-nul;time,size,cpu,deadline,prop:f64le\x00');h.update('\x00'.join(ids).encode());h.update(b'\x00');h.update(x.astype('<f8',copy=False).tobytes())
    return Input(df,x,h.hexdigest())
@lru_cache(None)
def native():
    compiler=os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:return None
    src=HERE/'study_engine.cpp'; tmp=Path(tempfile.gettempdir())/'sncs_study_backend';tmp.mkdir(exist_ok=True)
    ext=('.dll' if os.name=='nt' else '.so')
    so=tmp/('native_'+hashlib.sha256(src.read_bytes()).hexdigest()[:20]+ext)
    if not so.exists():
        t=so.with_name(so.name+'.'+str(os.getpid())+'.tmp')
        cmd=[compiler,'-O3','-std=c++17','-ffp-contract=off','-shared']+([] if os.name=='nt' else ['-fPIC'])+[str(src),'-o',str(t)]
        subprocess.run(cmd,check=True);os.replace(t,so)
    lib=ctypes.CDLL(str(so))
    model=lib.sim_v2;ptr=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    model.argtypes=[ctypes.c_int,ptr,ptr,ctypes.c_int,ctypes.c_int,ptr];model.restype=ctypes.c_int
    return model
@lru_cache(None)
def refcore(divisor):
    name='sncs_reference_'+str(divisor).replace('.','_')
    spec=importlib.util.spec_from_file_location(name,HERE/'reference/frozen_policy_engine.py')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    m.task_size_mb=lambda kb:max(0.,float(kb)/divisor)
    return m
def effective(params,network,bandwidth,coefficients,proxy_divisor):
    p={**DEFAULTS,**dict(params),**(coefficients or {}),'network_model':network,'bandwidth_mbps':float(bandwidth),'proxy_divisor':float(proxy_divisor)}
    if network not in ('shared_fcfs','independent') or p['network_baseline_rule'] not in ('paper_utilization','reference_bandwidth'):raise ValueError('Unknown network model/rule')
    if not all(np.isfinite(float(p[k])) for k in DEFAULTS if k!='network_baseline_rule'):raise ValueError('Nonfinite parameter')
    for k in ('edge_nodes','cloud_servers','edge_queue_capacity','cloud_queue_capacity'):
        if int(p[k])!=p[k] or p[k]<(1 if k in ('edge_nodes','cloud_servers') else 0):raise ValueError(k)
    for k in ('bandwidth_mbps','proxy_divisor','edge_service_rate_tps_each','cloud_service_rate_tps_each','network_probe_kb'):
        if p[k]<=0:raise ValueError(k)
    if p['threshold_ms']<0 or not (0<=p['utilization_threshold']<=1 and 0<=p['network_utilization_threshold']<=1):raise ValueError('Threshold outside valid range')
    return p
def envhash(p):
    settings={k:p[k] for k in DEFAULTS};settings.update(network_model=p['network_model'],duration_s=float(p['duration_s']))
    return hashlib.sha256(json.dumps(settings,sort_keys=True,default=str).encode()).hexdigest()
def simulate(tasks,params,policy,network='shared_fcfs',bandwidth=25.,coefficients=None,details=False,proxy_divisor=1000.,backend='auto'):
    if policy not in POLICIES:raise ValueError(policy)
    w=tasks if isinstance(tasks,Input) else prepare(tasks);n=len(w.x);x=w.x;df=w.df
    p=effective(params,network,bandwidth,coefficients,proxy_divisor)
    p['duration_s']=float(p.get('duration_s',max(float(x[-1,0]),.001)))
    if not np.isfinite(p['duration_s']) or p['duration_s']<=0:raise ValueError('duration_s')
    if backend not in ('auto','native','reference'):raise ValueError(backend)
    model=None if backend=='reference' else native()
    if model is None:
        if backend=='native':raise RuntimeError('C++17 compiler not available')
        frame=df.copy()
        for k,default in [('scenario','custom'),('run_id',1),('seed',np.nan)]:
            if k not in frame:frame[k]=p.get(k,default)
        m=refcore(p['proxy_divisor'])
        det,s=m.simulate_subset(frame,p,'deadline_aware_hybrid' if policy=='deadline_guarded' else policy,p['threshold_ms'],{'least_lateness':False} if policy=='deadline_guarded' else None)
        det['mode']=policy
        for key in ('proxy_tie_break_applied','proxy_tie_break_changed_destination'):
            if key not in det:det[key]=False
        cm=det.requested_destination.eq('cloud');ep=det.processing_location.eq('edge');cp=det.processing_location.eq('cloud')
        s.update(mode=policy,workload_sha256=w.fingerprint,environment_sha256=envhash(p),bandwidth_mbps=bandwidth,proxy_divisor=proxy_divisor,proxy_tie_count=int(det.proxy_tie_break_applied.sum()),proxy_tie_changed_count=int(det.proxy_tie_break_changed_destination.sum()),edge_service_ms=det.loc[ep,'service_time_ms'].sum(),cloud_service_ms=det.loc[cp,'service_time_ms'].sum(),uploaded_mb=det.loc[cm,'task_size_kb'].sum()/1000,uploaded_proxy_mb=det.loc[cm,'task_size_kb'].sum()/proxy_divisor)
        return s,(det if details else None)
    cfg=np.array([float(p[k]) if k!='network_baseline_rule' else float(p[k]=='paper_utilization') for k in DEFAULTS])
    out=np.empty((n,25),dtype=np.float64);rc=model(n,x,cfg,POLICIES.index(policy),int(network=='shared_fcfs'),out)
    if rc!=0:raise RuntimeError('Native engine exception')
    dest,ok,start,finish,service,wait,response,nwait,upload,carrival,ret,cost,energy,reason,changed,epred,cpred,cpwait,eq,cq,eb,cb,ew,actual_q,inflight=out.T
    ok=ok.astype(bool);cm=dest==1;ep=(dest==0)&ok;cp=cm&ok;late=ok&(response>x[:,3]+1e-9);on=ok & ~late
    nc=int(ok.sum());nb=n-nc;nt=int(on.sum());nl=int(late.sum())
    end=max(float(p['duration_s'])*1000.,float(x[-1,0])*1000.,float(np.max(finish[ok])) if nc else 0.,float(np.max(carrival[cm])) if cm.any() else 0.,1.)
    tc=float(cost.sum());te=float(energy.sum());es=float(service[ep].sum());cs=float(service[cp].sum())
    en,cn=int(p['edge_nodes']),int(p['cloud_servers'])
    s=dict(scenario=str(df.scenario.iloc[0]) if 'scenario' in df else p.get('scenario','custom'),run_id=int(df.run_id.iloc[0]) if 'run_id' in df else int(p.get('run_id',1)),seed=df.seed.iloc[0] if 'seed' in df else p.get('seed',np.nan),mode=policy,network_model=network,bandwidth_mbps=bandwidth,environment_sha256=envhash(p),workload_sha256=w.fingerprint,total_tasks=n,completed_tasks=nc,blocked_tasks=nb,sla_violations=nl,on_time_completed_tasks=nt,unsuccessful_tasks=n-nt,on_time_completion_rate=nt/n,unsuccessful_task_rate=(n-nt)/n,throughput_tasks_per_s=nc/(end/1000.),offered_rate_tps=n/p['duration_s'],observation_duration_s=p['duration_s'],drain_end_s=end/1000.,avg_response_time_ms=float(response[ok].mean()) if nc else np.nan,p95_response_time_ms=float(np.quantile(response[ok],.95)) if nc else np.nan,avg_waiting_time_ms=float(wait[ok].mean()) if nc else np.nan,avg_network_queue_wait_ms=nwait.mean(),edge_utilization=es/(en*end),cloud_utilization=cs/(cn*end),avg_edge_queue_length=wait[ep].sum()/end,avg_cloud_queue_length=wait[cp].sum()/end,blocking_probability=nb/n,cloud_offloading_ratio=cp.sum()/nc if nc else 0.,cloud_routing_fraction=cm.mean(),sla_violation_rate=nl/nc if nc else np.nan,avg_cost_proxy=tc/nc if nc else np.nan,total_cost_proxy=tc,avg_energy_proxy=te/nc if nc else np.nan,total_energy_proxy=te,cost_proxy_per_offered_task=tc/n,energy_proxy_per_offered_task=te/n,uplink_utilization=upload.sum()/end if network=='shared_fcfs' else np.nan,cloud_forecasts_with_positive_wait=int((cpwait>0).sum()),proxy_divisor=proxy_divisor,proxy_tie_count=int(((reason==9)|(reason==10)).sum()),proxy_tie_changed_count=int(changed.sum()),edge_service_ms=es,cloud_service_ms=cs,uploaded_mb=x[cm,1].sum()/1000.,uploaded_proxy_mb=x[cm,1].sum()/proxy_divisor)
    d=None
    if details:
        labels=np.array(['edge','cloud','blocked'])[dest.astype(int)]
        d=df.copy();d['mode']=policy;d['requested_destination']=labels;d['processing_location']=np.where(ok,labels,'blocked');d['status']=np.where(ok,'completed','blocked');d['start_time_ms']=start;d['finish_time_ms']=finish;d['service_time_ms']=service;d['waiting_time_ms']=wait;d['response_time_ms']=response;d['on_time_completion']=on.astype(int);d['sla_violation']=late.astype(int);d['network_queue_wait_ms']=nwait;d['upload_serialization_ms']=upload;d['cloud_arrival_time_ms']=carrival;d['cost_proxy']=cost;d['energy_proxy']=energy;d['decision_reason']=np.array(REASONS)[reason.astype(int)]
        if p['network_baseline_rule']=='reference_bandwidth' and policy=='network_aware':d['decision_reason']='public_reference_bandwidth_rule_adapted'
        d['proxy_tie_break_applied']=(reason==9)|(reason==10);d['proxy_tie_break_changed_destination']=changed.astype(bool);d['predicted_edge_response_ms']=epred;d['predicted_cloud_response_ms']=cpred;d['predicted_cloud_wait_ms']=cpwait;d['predicted_edge_wait_ms']=ew;d['edge_queue_at_decision']=eq;d['cloud_queue_at_decision']=cq;d['edge_busy_servers_at_decision']=eb;d['cloud_busy_servers_at_decision']=cb;d['known_inflight_cloud_tasks']=inflight;d['actual_queue_at_layer_arrival']=actual_q;d['cloud_return_delay_ms']=ret
    return s,d
