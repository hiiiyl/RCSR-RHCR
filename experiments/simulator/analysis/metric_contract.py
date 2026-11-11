"""Pure descriptive adapter. No acceptance decisions, data writes, or implicit CI choice."""
from copy import deepcopy
from pathlib import Path
import json, math, statistics
REGISTRY=json.loads(Path(__file__).with_name('metric_registry.json').read_text())
VERSION='rcsr-descriptive-v1'
def finite(x): return type(x) in (int,float) and math.isfinite(x)
def quantile(a,q):
 a=sorted(a); x=(len(a)-1)*q; i=int(x)
 return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def clean(x):
 if isinstance(x,float) and not math.isfinite(x): return {'nonfinite_raw':str(x)}
 if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list): return [clean(v) for v in x]
 return deepcopy(x)
def normalize_case(document,context=None):
 """Return JSON-safe derivative with raw preserved, source trace and missing reason.
 Context may explicitly identify continuous=True and component_records, or warmup.
 Context is provenance, not permission to invent unrecorded measurements.
 """
 if document.get('contract_version')==VERSION:
  if context is not None and context!=document['context']: raise ValueError('Already normalized with different context')
  return deepcopy(document)
 if not isinstance(document.get('summary'),dict): raise ValueError('summary must be an object')
 context=deepcopy(context or {}); s=document['summary']; p=document.get('planning'); tasks=document.get('tasks'); audits=[]; m={}
 def put(k,v,why=None,source=None):
  if v is not None and (not finite(v) or v<0): v=None; why='invalid_or_nonfinite_measurement'
  m[k]={'value':v,'missing_reason':why if v is None else None,'source_columns':[source or REGISTRY[k]['aggregation_source']], 'raw_summary_value':clean(s.get(k))}
 def aggregate(k,records,field,agg):
  if not isinstance(records,list) or not records: put(k,None,'no_observed_records'); return
  vals=[row.get(field) if isinstance(row,dict) else None for row in records]
  # Partial observations cannot silently become an all-cycle mean.
  if any(not finite(v) or v<0 for v in vals): put(k,None,'field_missing_or_invalid_in_records'); return
  put(k, sum(vals) if agg=='sum' else max(vals) if agg=='max' else quantile(vals,.95) if agg=='p95' else quantile(vals,.9) if agg=='p90' else statistics.mean(vals),source=('tasks[].'+field if field in ('response_time','observed_duration') else None))
 for k,r in REGISTRY.items():
  src=r['aggregation_source']; agg=r['aggregation']
  if src.startswith('planning[].'):
   field=src.split('.')[-1]
   conditional=k in ('mean_history_after_filter_raw_error','mean_history_post_true_support_coverage','mean_conditional_evidence_mass','mean_conditional_true_state_probability')
   if conditional and (not p or any(row.get('conditional_posterior_valid') is not True for row in p)):
    put(k,None,'conditional_posterior_invalid_or_unobserved'); continue
   aggregate(k,p,field,agg)
  elif src.startswith('steps[].'): aggregate(k,document.get('steps'),src.split('.')[-1],agg)
  else: put(k,s.get(k),'field_absent_or_invalid')
  if r['availability'].startswith('continuous simulator') and context.get('continuous') is not True and s.get('continuous') is not True:
   put(k,None,'not_measured_continuous_NA')
 # Strict planning flags and their identities.
 flags={'joint_repair_count':'joint_repair','duplicate_raw_starts_count':'duplicate_raw_starts','planning_failure_count':'planning_failure','planner_wait_fallback_count':'planner_wait_fallback'}
 valid=isinstance(p,list) and all(isinstance(row,dict) and all(type(row.get(f)) is int and row[f]>=0 and (f=='duplicate_raw_starts' or row[f] in (0,1)) for f in flags.values()) and not(row['joint_repair']==1 and row['duplicate_raw_starts']==0) and not(row['duplicate_raw_starts']>0 and row['joint_repair']==0 and row['planning_failure']==0) for row in p)
 counts=list(flags)+['planning_cycle_count','start_repair_attempt_cycles','start_repair_success_cycles','start_repair_failure_cycles','post_repair_search_failure_cycles']
 if valid:
  vals={k:sum(row[f] for row in p) for k,f in flags.items()}
  vals.update(planning_cycle_count=len(p),start_repair_attempt_cycles=sum(row['duplicate_raw_starts']>0 for row in p),start_repair_success_cycles=sum(row['joint_repair'] for row in p),start_repair_failure_cycles=sum(row['duplicate_raw_starts']>0 and row['joint_repair']==0 for row in p),post_repair_search_failure_cycles=sum(row['joint_repair']==1 and row['planning_failure']==1 for row in p))
  for k,v in vals.items(): put(k,v)
  for k in flags: audits.append(dict(metric=k,reported=clean(s.get(k)),recomputed=vals[k],status='match' if s.get(k)==vals[k] else 'summary_mismatch'))
 else:
  for k in counts: put(k,None,'planning_flags_missing_or_invalid')
 for k,r in REGISTRY.items():
  if r['aggregation']=='ratio':
   num=m[r['aggregation_source'].split('.')[-1]]['value']; den=m['planning_cycle_count']['value']
   put(k,num/den if num is not None and den else None,'missing_or_zero_planning_denominator')
 validtasks=isinstance(tasks,list) and all(isinstance(t,dict) and type(t.get('event_observed')) is int and t['event_observed'] in (0,1) for t in tasks)
 if validtasks:
  done=[t for t in tasks if t['event_observed']==1]; cens=[t for t in tasks if t['event_observed']==0]
  for k,v in [('task_record_count',len(tasks)),('completed_targets',len(done)),('censored_task_count',len(cens)),('completion_fraction',len(done)/len(tasks) if tasks else None),('censoring_fraction',len(cens)/len(tasks) if tasks else None)]: put(k,v,'zero_task_denominator','tasks[].event_observed')
  for k,rows,field,agg in [('p90_response_time',done,'response_time','p90'),('mean_response_time',done,'response_time','mean'),('mean_unfinished_target_age',cens,'observed_duration','mean'),('p90_unfinished_target_age',cens,'observed_duration','p90')]: aggregate(k,rows,field,agg)
  v=m['p90_response_time']['value']; old=s.get('p90_response_time')
  audits.append(dict(metric='p90_response_time',reported=clean(old),recomputed=v,status='match' if finite(v) and finite(old) and math.isclose(v,old,rel_tol=1e-10,abs_tol=1e-9) else 'no_completed_tasks' if not done else 'summary_mismatch'))
 else:
  for k in ['p90_response_time','mean_response_time','task_record_count','completed_targets','censored_task_count','completion_fraction','censoring_fraction','mean_unfinished_target_age','p90_unfinished_target_age']: put(k,None,'task_records_missing_or_invalid')
 steps=s.get('episode_steps'); fleet=s.get('uav_count'); intervention=s.get('safety_intervention_count')
 denominator=steps*fleet if finite(steps) and finite(fleet) and steps>0 and fleet>0 else None
 put('safety_interventions_per_uav_step',intervention/denominator if finite(intervention) and denominator else None,'missing_or_zero_uav_step_denominator')
 warmup=context.get('warmup',min(100,int(steps)//10) if finite(steps) and steps>=0 else None)
 if validtasks and finite(steps) and finite(warmup) and 0<=warmup<steps and all(finite(t.get('completion_step')) for t in done):
  put('steady_state_throughput',sum(t['completion_step']>warmup for t in done)/(steps-warmup))
 else: put('steady_state_throughput',None,'task_completion_times_or_warmup_denominator_missing')
 # Explicit sampled contact denominators; no max(denominator,1) workaround.
 if context.get('continuous') is True or s.get('continuous') is True:
  substeps=s.get('continuous_substeps')
  for prefix,pair in [('uav_contact_pair',True),('obstacle_contact_uav',False),('boundary_violation_uav',False)]:
   key=prefix+'_substep_fraction'; numerator=s.get(prefix+'_substeps')
   den=substeps*fleet*((fleet-1)/2 if pair else 1) if finite(substeps) and finite(fleet) and substeps>0 and fleet>0 else None
   put(key,numerator/den if finite(numerator) and den and den>0 else None,'missing_or_zero_sample_denominator')
 aggregate('component_ess',context.get('component_records'),'ess','mean')
 return {'contract_version':VERSION,'context':clean(context),'raw':clean(document),'metrics':m,'audits':audits,'warmup':warmup if finite(warmup) else None,'inference_scope':'descriptive only; independent n=seed; no acceptance decision'}

CONFIGURATION_FIELDS=('method','task_endpoint_conditioning','reference_policy')
CONDITION_FIELDS=('experiment','condition','fleet','uav_count','steps','episode_steps','commit_window','p_true',
 'goal_release','forbid_following','allocator_age_weight','allocator_use_estimates',
 'planner_horizon','planning_horizon','horizon','cell_size','control_dt','continuous','observational_probability_trace')
_MISSING=('__missing_configuration_field__',)
def configuration_key(summary, *, legacy_defaults=None, require_complete=False):
 """Explicit configuration identity; defaults describe legacy schema, never mutate data."""
 defaults=legacy_defaults or {}
 if set(defaults)-set(CONFIGURATION_FIELDS): raise ValueError('legacy_defaults may contain only configuration fields')
 values=[]
 for field in CONFIGURATION_FIELDS:
  value=summary.get(field,defaults.get(field,_MISSING))
  if value==_MISSING:
   if require_complete: raise ValueError('Missing configuration field '+field+'; supply explicit legacy_defaults only if documented')
  elif field=='task_endpoint_conditioning':
   if type(value) is not bool: raise ValueError('task_endpoint_conditioning must be boolean')
  elif not isinstance(value,str) or not value: raise ValueError(field+' must be a nonempty string')
  values.append(value)
 return tuple(values)

def seed_statistics(cases,metric,method,seeds,delays,*,ci_method,control=None,bootstrap_seed=20261107,bootstrap_replicates=10000,configuration=None,control_configuration=None,legacy_defaults=None,condition_fields=CONDITION_FIELDS):
 """Prespecified complete seed clusters, optional explicit configuration selectors.
 Method-only legacy calls cannot silently collapse distinct endpoint/policy arms.
 Additional condition_fields may tighten the default fixed-condition guard.
 """
 if type(bootstrap_replicates) is not int or bootstrap_replicates<1: raise ValueError('Positive integer bootstrap_replicates required')
 if ci_method not in ('paired_t','fixedseedbootstrap'): raise ValueError('Explicit supported ci_method required')
 if len(set(seeds))!=len(seeds) or len(set(delays))!=len(delays) or not delays: raise ValueError('Unique seeds and nonempty unique delays required')
 if ci_method=='paired_t' and len(seeds)!=2: raise ValueError('paired_t is explicitly scoped to n=2 pilots')
 if ci_method=='fixedseedbootstrap' and len(seeds)!=16: raise ValueError('fixedseedbootstrap is explicitly scoped to n=16 development')
 if not set(CONDITION_FIELDS).issubset(condition_fields): raise ValueError('condition_fields may extend, not remove, fixed-condition guards')
 if control_configuration is not None and control is None: raise ValueError('control required with control_configuration')
 lookup={}; method_keys={}; signatures={}
 for d in cases:
  n=normalize_case(d); s=n['raw']['summary']
  # Explicit complete identities can be supplied by source or documented defaults.
  ck=configuration_key(s,legacy_defaults=legacy_defaults)
  key=(ck,s.get('seed'),s.get('delay'))
  if key in lookup: raise ValueError('Duplicate case '+str(key))
  lookup[key]=n; method_keys.setdefault(s.get('method'),set()).add(ck)
  sig=tuple((f,json.dumps(s[f],sort_keys=True,allow_nan=False) if f in s else '<absent>') for f in condition_fields)
  signatures.setdefault(ck,set()).add(sig)
 def select(arm,selector):
  if selector is not None:
   if set(selector)!=set(CONFIGURATION_FIELDS): raise ValueError('configuration selector requires exactly '+str(CONFIGURATION_FIELDS))
   ck=configuration_key(selector,require_complete=True)
   if ck[0]!=arm: raise ValueError('selector method differs from requested method')
   if any(_MISSING in key for key in method_keys.get(arm,set())): raise ValueError('Missing configuration fields for selected method; explicit documented legacy_defaults required')
  else:
   found=method_keys.get(arm,set())
   if len(found)>1: raise ValueError('Multiple configurations for method '+str(arm)+'; pass explicit configuration/control_configuration')
   ck=next(iter(found)) if found else configuration_key({'method':arm},legacy_defaults=legacy_defaults)
  if len(signatures.get(ck,set()))>1: raise ValueError('Mixed fixed experimental conditions for configuration '+str(ck)+'; partition cases before statistics')
  return ck
 candidate_key=select(method,configuration)
 control_key=select(control,control_configuration) if control is not None else None
 # A paired comparison must share fixed conditions, not merely seed and delay.
 if control_key is not None and signatures.get(candidate_key) and signatures.get(control_key) and signatures[candidate_key]!=signatures[control_key]:
  raise ValueError('Candidate/control fixed experimental conditions differ')
 rows=[]
 for seed in seeds:
  averages=[]; reasons=[]
  for arm,arm_key in [(method,candidate_key)]+([(control,control_key)] if control is not None else []):
   vals=[]
   for delay in delays:
    n=lookup.get((arm_key,seed,delay)); x=n['metrics'][metric] if n else None
    if x is None or x['value'] is None: reasons.append(f'{arm}/delay={delay}:'+('episode_missing' if x is None else x['missing_reason']))
    else: vals.append(x['value'])
   averages.append(statistics.mean(vals) if len(vals)==len(delays) else None)
  value=None if reasons else averages[0]-(averages[1] if control is not None else 0)
  rows.append(dict(seed=seed,value=value,missing_reasons=reasons))
 a=[r['value'] for r in rows if r['value'] is not None]; n=len(a); mean=statistics.mean(a) if n else None; sd=statistics.stdev(a) if n>1 else None; ci=[None,None]
 if n>1:
  if ci_method=='paired_t':
   half=12.706204736432095*sd/math.sqrt(n); ci=[mean-half,mean+half]
  else:
   import random
   rng=random.Random(bootstrap_seed)
   b=[statistics.mean(rng.choices(a,k=n)) for _ in range(bootstrap_replicates)]; ci=[quantile(b,.025),quantile(b,.975)]
 result=dict(metric=metric,method=method,control=control,ci_method=ci_method,mean=mean,sample_sd=sd,ci95=ci,n_valid=n,n_expected=len(seeds),missing_n=len(seeds)-n,seed_rows=rows,bootstrap_seed=bootstrap_seed if ci_method=='fixedseedbootstrap' else None)
 if configuration is not None or control_configuration is not None:
  result.update(configuration=deepcopy(configuration),control_configuration=deepcopy(control_configuration),legacy_defaults=deepcopy(legacy_defaults or {}))
 return result

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(); parser.add_argument('case'); parser.add_argument('--context'); args=parser.parse_args()
 print(json.dumps(normalize_case(json.loads(Path(args.case).read_text()),json.loads(Path(args.context).read_text()) if args.context else None),allow_nan=False,indent=2))
