"""Rerun all submitted server-count/buffer sweeps on the corrected engine. No network access."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,sys
import pandas as pd
import numpy as np
from _paths import DATA, RESULTS, result_file
HERE=RESULTS/'sensitivity'; HERE.mkdir(parents=True,exist_ok=True)
from study_engine import simulate,prepare,native
SC={p['scenario']:p for p in json.loads((DATA/'synthetic/scenarios.json').read_text())}
GRIDS={'edge_nodes':(2,4,6,8),'cloud_servers':(1,2,3,4),'edge_queue_capacity':(60,120,180,240)}
def one(record):
    scenario=record['scenario'];path=DATA/'synthetic'/record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['file_sha256']
    w=prepare(pd.read_csv(path));rows=[]
    for net in ('independent','shared_fcfs'):
        for k,levels in GRIDS.items():
            for level in levels:
                p=dict(SC[scenario],run_id=int(record['run_id']),seed=int(record['seed']))
                is_ref=(p[k]==level)
                p[k]=level
                s,_=simulate(w,p,'hybrid_cloud_edge',net,25.)
                rows.append(dict(s,parameter=k,level=level,is_main_setting=is_ref))
    return rows
if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('--workers',type=int,default=3);args=arg.parse_args();native()
    records=pd.read_csv(DATA/'synthetic/workloads.csv').query("scenario in ['high_load','burst_load']").to_dict('records')
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        data=pd.DataFrame([r for batch in pool.map(one,records) for r in batch])
    assert len(data)==1440 and data.groupby(['scenario','network_model','parameter','level']).size().eq(30).all()
    m=pd.read_csv(result_file('main/runs.csv')).query("mode=='hybrid_cloud_edge' and bandwidth_mbps==25")
    keys=['scenario','run_id','network_model','mode'];base=data[data.is_main_setting].merge(m,on=keys,suffixes=('','_orig'),validate='many_to_one')
    assert len(base)==360
    for col in m.columns.difference(keys):
        lhs,rhs=base[col],base[col+'_orig']
        if pd.api.types.is_numeric_dtype(lhs):
            np.testing.assert_allclose(lhs,rhs,rtol=1e-11,atol=1e-8,equal_nan=True,err_msg=col)
        else:assert (lhs.fillna('')==rhs.fillna('')).all(),col
    assert (data.total_tasks==data.completed_tasks+data.blocked_tasks).all() and (data.completed_tasks==data.on_time_completed_tasks+data.sla_violations).all()
    dst=HERE/'capacity_runs.csv';tmp=dst.with_suffix('.tmp')
    with tmp.open('w') as f:data.to_csv(f,index=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,dst)
    print('1440 runs; 360 identical control-setting matches. Unique physical settings per workload and link: 10 (3 shared anchors).')
