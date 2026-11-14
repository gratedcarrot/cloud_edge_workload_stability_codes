"""Run frozen 150×3×8 matched cells. Each worker processes one complete workload."""
from pathlib import Path
import argparse,json,concurrent.futures,os,hashlib
import pandas as pd
from _paths import DATA, RESULTS
from study_engine import simulate,prepare,POLICIES,native
HERE=RESULTS; SYNTHETIC=DATA/'synthetic'; (RESULTS/'main').mkdir(parents=True,exist_ok=True)
NETS=[('independent',25),('shared_fcfs',25),('shared_fcfs',250)]
def one(item):
    params,row=item
    params.update(run_id=row['run_id'],seed=row['seed'])
    path=SYNTHETIC/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256'],path
    w=prepare(pd.read_csv(path))
    assert len(w.df)==row['offered']
    records=[]
    for net,bw in NETS:
        for policy in POLICIES:
            s,_=simulate(w,params,policy,net,bw)
            records.append(s)
    return records
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=4);o=a.parse_args();(HERE/'main').mkdir(parents=True,exist_ok=True)
    sc={s['scenario']:s for s in json.loads((SYNTHETIC/'scenarios.json').read_text())}
    rowlist=pd.read_csv(SYNTHETIC/'workloads.csv').to_dict('records')
    native()
    merged=[]
    with concurrent.futures.ProcessPoolExecutor(o.workers) as pool:
        for i,recs in enumerate(pool.map(one,[(sc[r['scenario']].copy(),r) for r in rowlist]),1):
            merged.extend(recs)
            if i%30==0:print('Workloads',i,'of',len(rowlist),flush=True)
    df=pd.DataFrame(merged)
    group=df.groupby(['scenario','run_id','network_model','bandwidth_mbps'])
    assert len(df)==3600 and len(group)==450 and group.workload_sha256.nunique().eq(1).all() and group.environment_sha256.nunique().eq(1).all() and group['mode'].nunique().eq(8).all()
    assert (df.completed_tasks+df.blocked_tasks==df.total_tasks).all() and (df.on_time_completed_tasks+df.sla_violations==df.completed_tasks).all()
    path=HERE/'main/runs.csv';temp=path.with_suffix('.tmp')
    with temp.open('w') as f:df.to_csv(f,index=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    print('Wrote',len(df),'rows,',df.total_tasks.sum(),'offered evaluations')
