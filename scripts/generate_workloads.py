"""Generate compact routing inputs using the supplied, unmodified generator primitives."""
from pathlib import Path
import sys, concurrent.futures, json, hashlib, os, tempfile
import numpy as np
import pandas as pd
from _paths import ROOT, RESULTS
DIR=RESULTS
(RESULTS/'data').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'engine/reference'))
import generate_workload as gen
BASE=913700
def generate(item):
    i,r=item; s=gen.SCENARIOS[i]; seed=gen.scenario_run_seed(BASE,i+1,r); rng=np.random.default_rng(seed)
    arr=gen.generate_arrival_times_for_run(rng,s)
    typelist=list(gen.TASK_TYPE_PROBABILITIES); probs=list(gen.TASK_TYPE_PROBABILITIES.values())
    rows=[]
    for j,a in enumerate(arr.arrival_time_s):
        kind=str(rng.choice(typelist,p=probs)); mark=gen.generate_task_profile(rng,kind,heavy_tailed=bool(s['heavy_tailed_demand']))
        zone=rng.choice(gen.EDGE_ZONES)
        delay=gen.clipped_normal(rng,float(s['mean_network_delay_ms']),float(s['std_network_delay_ms']),5)
        priority=rng.choice(gen.PRIORITY_LEVELS,p=gen.PRIORITY_PROBABILITIES)
        rows.append((f'{i+1}.{r}.{j+1}',round(float(a),6),mark['task_size_kb'],mark['cpu_demand_mi'],mark['deadline_ms'],round(float(delay),3)))
    df=pd.DataFrame(rows,columns=['task_id','arrival_time_s','task_size_kb','cpu_demand_mi','deadline_ms','edge_cloud_network_delay_ms'])
    p=DIR/'data'/f"{s['scenario']}_{r:02d}.csv.gz"
    with tempfile.TemporaryDirectory(dir=p.parent) as temp:
        q=Path(temp)/p.name
        df.to_csv(q,index=False,compression={'method':'gzip','compresslevel':3,'mtime':0})
        with q.open('rb') as f: os.fsync(f.fileno())
        os.replace(q,p)
    if os.name!='nt':
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return {'scenario':s['scenario'],'run_id':r,'seed':seed,'offered':len(df),'path':p.name,'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if __name__=='__main__':
    (DIR/'data').mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(5) as pool:
        rows=list(pool.map(generate,[(i,r) for i in range(5) for r in range(1,31)],chunksize=2))
    pd.DataFrame(rows).to_csv(DIR/'data/workloads.csv',index=False)
    (DIR/'data/scenarios.json').write_text(json.dumps(gen.SCENARIOS,indent=2))
    print('Fresh inputs',len(rows),'workloads;',sum(x['offered'] for x in rows),'tasks',flush=True)
