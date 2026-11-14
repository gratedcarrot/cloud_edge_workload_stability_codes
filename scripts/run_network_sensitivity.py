from pathlib import Path
import json,sys,concurrent.futures,os,argparse
import pandas as pd
from _paths import DATA, RESULTS
from study_engine import simulate
HERE=RESULTS/'sensitivity'; HERE.mkdir(parents=True,exist_ok=True)
P={x['scenario']:x for x in json.loads((DATA/'synthetic/scenarios.json').read_text())}
def one(key):
    sc,rid=key
    df=pd.read_csv(DATA/'synthetic'/f'{sc}_{rid:02}.csv.gz')
    rows=[]
    for net in ['independent','shared_fcfs']:
        for kind,levels in [('threshold_ms',[25,50,100,150,220,300]),('propagation_multiplier',[0.5,1,1.5,2])]:
            for level in levels:
                p=dict(P[sc],run_id=rid,seed=913700+1000*(list(P).index(sc)+1)+rid);x=df
                if kind=='threshold_ms':p['threshold_ms']=float(level)
                else:
                    x=df.copy()
                    x['edge_cloud_network_delay_ms']*=level
                s,_=simulate(x,p,'hybrid_cloud_edge',network=net,bandwidth=25.)
                rows.append(dict(s,study_scenario=sc,study_run=rid,network=net,parameter=kind,level=level))
    return rows
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=3);args=a.parse_args()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        data=[r for rows in pool.map(one,[(s,i) for s in ['high_load','burst_load'] for i in range(1,31)]) for r in rows]
    out=pd.DataFrame(data);assert len(out)==1200
    HERE.mkdir(parents=True,exist_ok=True);out.to_csv(HERE/'sweep_runs.csv',index=False)
    print('Saved',len(out),'sensitivity runs')

