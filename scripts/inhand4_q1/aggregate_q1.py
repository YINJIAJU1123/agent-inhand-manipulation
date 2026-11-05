"""Independent scoring/latency/latent audit for the complete six-cell study."""
import argparse,hashlib,json,math,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
r=read(a.release);count=40
if r['format']!='inhand4_marked_q1_v1':raise ValueError('Wrong release format')
from goals import load_bank
_,canonical,_=load_bank(r['shared_artifacts']['bank'],'cpu')
all_rows={};file_hashes={};methods=('astra','vlm','rule');layouts=('identity','permuted')
for layout in layouts:
 for method in methods:
  cell=a.results/f'{layout}_{method}';rows=[];actions=[];locations={}
  for path in sorted(cell.glob('*/batch_*')):
   result=read(path/'result.json');manifest=read(path/'manifest.json')
   if result['status'] not in ('completed','service_blocked') or manifest['release_sha256']!=sha(a.release) or manifest['development']:raise ValueError('Incomplete/wrong release: '+str(path))
   batch=[json.loads(s) for s in (path/'episodes.jsonl').read_text().splitlines()]
   if len(batch)!=result['recorded']:raise ValueError('Batch ledger mismatch')
   for row in batch:
    if row['index'] in locations:raise ValueError('Duplicate visited slot')
    locations[row['index']]=path
   rows+=batch
   if (path/'action_interface.jsonl').exists():actions += [json.loads(s) for s in (path/'action_interface.jsonl').read_text().splitlines()]
   for name in ('episodes.jsonl','manifest.json','result.json','action_interface.jsonl'):
    if (path/name).exists():file_hashes[str(path/name)]=sha(path/name)
  rows.sort(key=lambda x:x['index']);cases=r['cases_by_layout'][layout]
  if [x['index'] for x in rows]!=list(range(count)):raise ValueError('Incomplete 40-case cell: '+str(cell))
  # Reset makes step numbers repeat. Consume commands in call-ledger order.
  audited_commands=0
  for row,case in zip(rows,cases):
   path=locations[row['index']]
   if any(row[k]!=case[k] for k in case):raise ValueError('Case labels changed')
   q=row['final_quaternion_xyzw'];target=canonical[case['goal_index']].tolist()
   if target!=row['target_quaternion_xyzw']:raise ValueError('Scored target differs from frozen canonical goal')
   error=2*math.acos(min(1.,abs(sum(x*y for x,y in zip(q,target)))/math.sqrt(sum(x*x for x in q)*sum(x*x for x in target))))
   if abs(error-row['final_error_rad'])>1e-7:raise ValueError('Quaternion score')
   correct=row['declared_goal']==case['goal_index'] and row['declared_number']==case['requested_label_index'] and error<=.16
   if row['reason'] in ('success','false_stop'):
    if row['success']!=correct or row['success']!=(row['reason']=='success'):raise ValueError('Incorrect autonomous-stop success')
   elif row['success']:raise ValueError('Success without stop')
   case_actions=[command for command in actions if command['trial_seed']==case['trial_seed']]
   offset=0;known=False
   calls=read(path/f"episode_{case['index']:03d}/calls.json")
   if len(calls)!=row['actual_model_calls'] or not 1<=len(calls)<=6:raise ValueError('Call coverage')
   if any(c['error'] or c['elapsed_s']>120 for c in calls)!=(row['exclusion_reason'] is not None):raise ValueError('Incorrect service exclusion')
   for call in calls:
    if call.get('applied_step') is None:continue
    if call['delivery_step']!=call['request_step']+math.ceil(call['elapsed_s']*20) or call['applied_step']<call['delivery_step']:raise ValueError('Paused/early response')
    result=call['result']
    sets_skill=(result['tool']!='abstain') if method=='astra' else (result['accepted'] or not known)
    if sets_skill:
     if offset>=len(case_actions):raise ValueError('Missing command')
     command=case_actions[offset];offset+=1;audited_commands+=1
     if command['step']!=call['applied_step']:raise ValueError('Wrong action step')
     if method=='astra':
      if command['selected_skill']!=result['arguments']['number']-1 or command['continuous_latent_supplied']:raise ValueError('Astra action remap')
     elif result['accepted']:
      if command['selected_skill']!=result['skill']:raise ValueError('VLM action remap')
      if method=='vlm':
       vector=result['goal_embedding'];expected=hashlib.sha256(struct.pack('<768f',*vector)).hexdigest()
       if command['input_latent_sha256']!=expected or command['applied_latent_sha256']!=expected:raise ValueError('Latent changed in evaluator')
      elif command['continuous_latent_supplied']:raise ValueError('Rule received learned latent')
    if method=='astra':known=result['tool'] not in ('inspect_face','abstain')
    elif result['accepted']:known=True
   if offset!=len(case_actions):raise ValueError('Unmatched case command ledger')
  if audited_commands!=len(actions):raise ValueError('Unmatched command ledger')
  all_rows[layout,method]=rows
  for name in ('episodes.jsonl','manifest.json','result.json','action_interface.jsonl'):
   if (path/name).exists():file_hashes[str(path/name)]=sha(path/name)
reset_differences=[]
for i in range(count):
 hashes={rows[i]['initial_state']['sha256'] for rows in all_rows.values()}
 if len(hashes)!=1:
  review_path=a.results/'initial_state_review.json'
  review=read(review_path)
  entry=next((x for x in review['cases'] if x['index']==i),None)
  if not entry or set(entry['state_hashes'])!=hashes or not review['accepted_for_reporting']:raise ValueError('Unreviewed reset difference')
  for path,digest in review['input_sha256'].items():
   if sha(path)!=digest:raise ValueError('Changed reset review input')
  file_hashes[str(review_path.resolve())]=sha(review_path)
  reset_differences.append(i)
excluded=[i for i in range(count) if any(rows[i]['exclusion_reason'] is not None for rows in all_rows.values())]
def wilson(k,n):
 if not n:return None
 z=1.959963984540054;den=1+z*z/n;center=(k/n+z*z/(2*n))/den;half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
 return [center-half,center+half]
def paired(rows_a,rows_b,indices):
 b=sum(rows_a[i]['success'] and not rows_b[i]['success'] for i in indices)
 c=sum(rows_b[i]['success'] and not rows_a[i]['success'] for i in indices)
 n=b+c;pvalue=min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
 return dict(n=len(indices),a_only=b,b_only=c,mcnemar_exact_two_sided_p=pvalue)
summary={}
for layout in layouts:
 within=[i for i in range(count) if any(all_rows[layout,m][i]['exclusion_reason'] is not None for m in methods)]
 summary[layout]={}
 for method in methods:
  rows=all_rows[layout,method];valid=[x for x in rows if x['index'] not in within];common=[x for x in rows if x['index'] not in excluded]
  summary[layout][method]=dict(raw_trials=count,raw_success=sum(x['success'] for x in rows),matched_trials=len(valid),matched_success=sum(x['success'] for x in valid),wilson_95=wilson(sum(x['success'] for x in valid),len(valid)),six_cell_common_trials=len(common),six_cell_common_success=sum(x['success'] for x in common),reasons={reason:sum(x['reason']==reason for x in rows) for reason in sorted({x['reason'] for x in rows})})
contrasts={layout:{method:paired(all_rows[layout,'astra'],all_rows[layout,method],[i for i in range(count) if not any(all_rows[layout,m][i]['exclusion_reason'] for m in methods)]) for method in ('vlm','rule')} for layout in layouts}
layout_contrasts={method:paired(all_rows['identity',method],all_rows['permuted',method],[i for i in range(count) if i not in excluded]) for method in methods}
report=dict(complete=True,development=False,paired_astra_comparisons=contrasts,paired_identity_vs_permuted=layout_contrasts,release_sha256=sha(a.release),six_cell_excluded_cases=excluded,layouts=summary,input_sha256=file_hashes,object=r['object'],scope='Object-specific frozen marked-object Q1, six cells of 40 raw slots; original and changed mapping, noisy simulator pose')
report['reviewed_initial_state_difference_cases']=reset_differences
report['exact_reset_sensitivity']={layout:{method:dict(trials=len(valid := [i for i in range(count) if i not in reset_differences and not any(all_rows[layout,m][i]['exclusion_reason'] for m in methods)]),success=sum(all_rows[layout,method][i]['success'] for i in valid)) for method in methods} for layout in layouts}
(a.results/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))
