"""Independently audit frozen dice case pairing, stop quaternions and exclusions."""
import argparse,json,math,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
release=json.loads(a.release.read_text());cases=release['cases'];methods=['astra','vlm','rule'];all_rows={}
for method in methods:
 root=a.results/method;result=json.loads((root/'result.json').read_text())
 if result['status']!='completed' or result['recorded']!=40:raise ValueError('Incomplete method: '+method)
 rows=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
 if [x['index'] for x in rows]!=list(range(40)):raise ValueError('Missing/repeated cases')
 for row,case in zip(rows,cases):
  if any(row[k]!=case[k] for k in case):raise ValueError('Case contract mismatch')
  q=row['final_quaternion_xyzw'];t=row['target_quaternion_xyzw']
  qn=math.sqrt(sum(x*x for x in q));tn=math.sqrt(sum(x*x for x in t))
  error=2*math.acos(min(1.,abs(sum(x*y for x,y in zip(q,t))/(qn*tn))))
  if abs(error-row['final_error_rad'])>1e-7:raise ValueError('Quaternion score disagreement')
  if row['reason'] in ('success','false_stop'):
   correct=row['declared_goal']==case['goal_index'] and row['declared_number']==case['goal_index']+1 and error<=.16
   if row['success']!=correct or row['success']!=(row['reason']=='success'):raise ValueError('Stop success disagreement')
  elif row['success']:raise ValueError('Success without autonomous stop')
  calls=json.loads((root/f"episode_{case['index']:03d}"/'calls.json').read_text())
  if len(calls)!=row['actual_model_calls'] or not 1<=len(calls)<=6:raise ValueError('Call ledger mismatch')
  service_failure=any(x['error'] is not None or x['elapsed_s']>120 for x in calls)
  if service_failure!=(row['exclusion_reason'] is not None):raise ValueError('Wrong service exclusion')
  for call in calls:
   if call.get('applied_step') is not None:
    expected=call['request_step']+math.ceil(call['elapsed_s']*20)
    if call['delivery_step']!=expected or call['applied_step']<expected:raise ValueError('Response applied before latency elapsed')
 all_rows[method]=rows
for i in range(40):
 if len({all_rows[m][i]['initial_state']['sha256'] for m in methods})!=1:raise ValueError('Initial states differ across methods')
excluded=[i for i in range(40) if any(all_rows[m][i]['exclusion_reason'] is not None for m in methods)]
eligible=[i for i in range(40) if i not in excluded]
def wilson(k,n):
 if not n:return None
 z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [center-half,center+half]
summary={}
for method,rows in all_rows.items():
 valid=[rows[i] for i in eligible];k=sum(x['success'] for x in valid);n=len(valid)
 summary[method]=dict(raw_trials=40,raw_success=sum(x['success'] for x in rows),matched_trials=n,matched_success=k,
  success_rate=k/n if n else None,wilson_95=wilson(k,n),reasons={s:sum(x['reason']==s for x in valid) for s in sorted({x['reason'] for x in valid})},
  drops_by_phase={s:sum(x['reason']=='drop' and x['drop_phase']==s for x in valid) for s in ['startup_wait','wait','execution']},
  model_calls=sum(x['actual_model_calls'] for x in rows),
  per_seed={str(seed):dict(n=sum(x['base_seed']==seed for x in valid),success=sum(x['base_seed']==seed and x['success'] for x in valid)) for seed in [186301,186302]})
report=dict(complete=True,release_sha256=hashlib.sha256(a.release.read_bytes()).hexdigest(),matched_excluded_cases=excluded,
 methods=summary,audits=['case coverage','paired initial states','independent final quaternion scoring','model call ledgers','response latency delivery','matched service exclusions'],
 scope='single native-layout dice object, noisy simulator pose, 40 scheduled cases per method; no pose estimator or unseen-layout claim')
(a.results/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
