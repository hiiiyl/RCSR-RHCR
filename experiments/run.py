"""Run the manuscript experiment matrix from its per-seed result index."""
from pathlib import Path
import os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import main_runner as main
import panel_runner as panel
def output_path(job):
    return ROOT/'reruns'/job['suite']/f"{panel.METHODS[job['display_method']][2]}_s{job['seed']}_d{job['delay']}_n{job['fleet']}_k{job['commit_window']}_{job['condition']}_t{job['steps']}.json.gz"
panel.result_path=output_path
main.METHODS.update({
 'EMC-medoid history forward':('EMC-Medoid-Task-History','support_separated_forward_v1','mechanism17'),
 'Component without history forward':('RCSR-Component-Bayes-V2','support_separated_forward_v1','mechanism05'),
 'Component history nominal':('RCSR-Component-Task-History','nominal','mechanism18')})

def execute(row):
    seed=int(row['seed']);steps=int(row['steps']);fleet=int(row['fleet'])
    # Both runners read the same tapes; the panel helper generates them deterministically.
    tape=panel.tape_path(seed,fleet,steps)
    if row['suite'] in ('main','mechanism'):
        from urban_lifelong_planning.revision_experiment import EpisodeSpec,run_episode
        import dataclasses,gzip,platform,numpy as np
        from metric_contract import normalize_case
        method,reference,slug=main.METHODS[row['method']]
        target=ROOT/'reruns'/row['suite']/f'{slug}_s{seed}_d{row["delay"]}_t{steps}.json.gz'
        if target.exists():return {'status':'reused','path':str(target.relative_to(ROOT))}
        spec=EpisodeSpec(experiment='integrated_development_proposal',method=method,seed=seed,delay=int(row['delay']),steps=steps,reference_policy=reference,goal_release=True)
        summary,tasks,planning=run_episode(spec,tape)
        doc={'status':'completed','display_method':row['method'],'spec':dataclasses.asdict(spec),'summary':summary,'tasks':tasks,'planning':planning,'python_version':platform.python_version(),'numpy_version':np.__version__}
        doc['metrics']={k:normalize_case(doc)['metrics'][k]['value'] for k in main.METRICS}
        target.parent.mkdir(exist_ok=True,parents=True)
        with gzip.open(target,'wt',encoding='utf8') as f:json.dump(main.jsonable(doc),f,allow_nan=False)
        return {'status':'completed','path':str(target.relative_to(ROOT))}
    job=dict(suite=row['suite'],display_method=row['method'],seed=seed,delay=int(row['delay']),fleet=fleet,commit_window=int(row['commit_window']),condition=row['condition'],steps=steps)
    return panel.run_one(job)

def run(rows,workers):
    # A fresh process for each episode also isolates the extension's simulator adapters.
    with ProcessPoolExecutor(max_workers=workers,max_tasks_per_child=1) as pool:
        futures=[pool.submit(execute,r) for r in rows]
        for i,f in enumerate(as_completed(futures),1):print(json.dumps({'done':i,'total':len(rows),'result':f.result()}),flush=True)

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['all','main','mechanism','window','scale','async','continuous','cost'],default='main')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--methods',nargs='+')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--limit',type=int)
    p.add_argument('--list',action='store_true')
    args=p.parse_args()
    if args.suite=='cost' and args.workers!=1:p.error('cost requires --workers 1 and an otherwise idle machine')
    with (ROOT/'results/per_seed_core_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    rows=[dict(r,steps=args.steps) for r in rows if (args.suite=='all' or r['suite']==args.suite) and (not args.seeds or int(r['seed']) in args.seeds) and (not args.methods or r['method'] in args.methods)]
    if args.limit:rows=rows[:args.limit]
    if not rows:p.error('no matching episodes')
    print(json.dumps({'episodes':len(rows),'suite':args.suite,'steps':args.steps}),flush=True)
    if args.list:return
    panel.prepare_tapes([dict(seed=int(r['seed']),fleet=int(r['fleet']),steps=int(r['steps'])) for r in rows])
    # Redirect extension outputs without changing its tape lookup.
    run([r for r in rows if r['suite']!='cost'],args.workers)
    run([r for r in rows if r['suite']=='cost'],1)

if __name__=='__main__':cli()
