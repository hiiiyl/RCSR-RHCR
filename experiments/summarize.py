"""Recalculate core descriptive and paired statistics from per-seed results."""
from pathlib import Path
import csv,json,collections,statistics,math
import bootstrap
ROOT=Path(__file__).resolve().parent
FIELDS=('p90_response_time','mean_planning_start_error','steady_state_throughput','safety_interventions_per_uav_step')

def main():
    with (ROOT/'results/per_seed_core_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups=collections.defaultdict(dict)
    for r in rows:
        key=tuple(r[k] for k in ('suite','method','delay','fleet','commit_window','condition'))
        seed=int(r['seed']);assert seed not in groups[key]
        groups[key][seed]={k:float(r[k]) if r[k] else None for k in FIELDS}
    assert len(rows)==1360
    assert all(set(v)==set(range(9400,9420)) for v in groups.values())
    # Check the main panel against the shipped manuscript means before computing intervals.
    with (ROOT/'results/core_per_delay.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            values=groups['main',r['method'],r['delay'],'6','2','fixed']
            for field,prefix,scale in zip(FIELDS,('p90_steps','start_error_grid','throughput_tasks_per_step','interventions_per_1000_uav_steps'),(1,1,1,1000)):
                mean=statistics.mean(v[field] for v in values.values() if v[field] is not None)*scale
                assert math.isclose(mean,float(r[prefix+'_mean']),abs_tol=1e-10),(r['method'],field,mean)
    result=[]
    for key,values in sorted(groups.items()):
        suite,method,delay,fleet,window,condition=key
        if suite=='cost':continue  # Serial phase timings are in cost_per_seed.csv.
        control=groups.get((suite,'RCSR-RHCR',delay,fleet,window,condition))
        for field in FIELDS:
            v={seed:row[field] for seed,row in values.items()}
            record=dict(zip(('suite','method','delay','fleet','commit_window','condition'),key),metric=field,**bootstrap.summarize(v,paired=True))
            if control and method!='RCSR-RHCR':
                differences={s:control[s][field]-v[s] if control[s][field] is not None and v[s] is not None else None for s in v}
                record['rcsr_minus_method']=bootstrap.summarize(differences,paired=True)
            result.append(record)
    out=ROOT/'reruns';out.mkdir(exist_ok=True)
    (out/'core_statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(f'Validated {len(rows)} unique episode rows, 20 seeds in every group, and all main-panel means.')
    print('Wrote reruns/core_statistics.json (10,000 resamples; seed 20261107).')

if __name__=='__main__':main()
