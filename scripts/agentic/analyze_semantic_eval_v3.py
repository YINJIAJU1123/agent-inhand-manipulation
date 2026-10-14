"""Audit and summarize v3 evidence; never mixes development and validation runs."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args()
root=Path(a.directory);freeze=json.loads((root/'freeze.json').read_text());protocol=freeze['protocol']
by_mode=defaultdict(list);by_scenario={};initial={};audit=[]
for scenario in protocol['scenarios']:
 for mode in protocol['modes']:
  seed=scenario['seed'];offset=scenario['instruction_offset'];template=scenario['template_set']
  name=f'seed{seed}_offset{offset}_{template}_{mode}'
  path=root/name/'report.json'
  if not path.exists():continue
  report=json.loads(path.read_text());meta=report['metadata'];rows=report['episodes']
  assert len(rows)==6 and len({r['case'] for r in rows})==6,name
  assert meta['checkpoint_sha256']==freeze['student_sha256'],name
  assert meta['adapter_sha256']==freeze['adapter_sha256'],name
  assert all(v==freeze['source_sha256'][k] for k,v in meta['source_sha256'].items()),name
  assert meta['seeds']==[seed] and meta['modes']==[mode] and meta['instruction_offset']==offset and meta['template_set']==template,name
  assert meta['call_budget']==protocol['call_budget'] and meta['max_steps']==protocol['max_steps'],name
  for row in rows:
   assert row['calls']==len(row['trace'])<=protocol['call_budget'],name
   if mode!='oracle':
    for call in row['trace']:
     assert json.dumps(row['instruction'],ensure_ascii=False) in call['prompt'],name
     assert call['instruction']==row['instruction'],name
    if row['success']:
     assert row['grounding_correct'] and row['language_correct'] and not row['drop'],name
   by_mode[mode].append(dict(row,template_set=template,instruction_offset=offset))
  by_scenario[(seed,offset,template,mode)]=rows
  initial[(seed,offset,template,mode)]=json.loads((path.parent/f'initial_{seed}_{mode}.json').read_text())
  audit.append(name)

def counts(rows):
 n=len(rows)
 return {'episodes':n,'success':sum(x['success'] for x in rows),'success_rate':sum(x['success'] for x in rows)/n if n else None,
         'drop':sum(x['drop'] for x in rows),
         'language_correct':None if rows and all(x['mode']=='oracle' for x in rows) else sum(x['language_correct'] for x in rows),
         'oracle_has_target_label':bool(rows) and all(x['mode']=='oracle' for x in rows),
         'grounding_correct':sum(x['grounding_correct'] for x in rows),'calls':sum(x['calls'] for x in rows),
         'text_calls':sum(t['operation']=='parse_language' for x in rows for t in x['trace']),
         'visual_calls':sum(t['operation']=='visual_grounding' for x in rows for t in x['trace']),
         'reobservations':sum(x['recovery_used'] for x in rows),'abstentions':sum(x['steps']==0 for x in rows),
         'event_counts':dict(Counter(t['reason'] for x in rows for t in x['trace']))}

# All instruction offsets for a seed preserve the same reset and layout.
differences={}
for seed in {x[0] for x in initial}:
 items=[(k,v) for k,v in initial.items() if k[0]==seed];base=items[0][1]
 for key,value in items:
  differences[str(key)]={field:float(np.abs(np.asarray(value[field])-np.asarray(base[field])).max())
                         for field in ('object_rotation','object_position','proprio','layout')}
  assert not any(differences[str(key)].values()),f'Initial state mismatch: {key}'

paired=[]
for (seed,offset,template,mode),rows in by_scenario.items():
 other=by_scenario.get((seed,offset,template,'periodic'))
 if mode=='feedback' and other:
  paired.append({'seed':seed,'offset':offset,'template_set':template,
                 'feedback_success':sum(r['success'] for r in rows),'periodic_success':sum(r['success'] for r in other),'episodes_per_mode':6})
counterfactual=[]
for mode in protocol['modes']:
 for seed in {x[0] for x in by_scenario}:
  base=by_scenario.get((seed,0,'canonical',mode))
  changed=by_scenario.get((seed,1,'paraphrase',mode))
  if base and changed:
   for before,after in zip(sorted(base,key=lambda r:r['case']),sorted(changed,key=lambda r:r['case'])):
    assert before['layout']==after['layout']
    assert before['target_color_index']!=after['target_color_index']
    counterfactual.append({'mode':mode,'seed':seed,'case':before['case'],
     'instructions':[before['instruction'],after['instruction']],
     'parsed_symbols':[before['parsed_goal_symbol'],after['parsed_goal_symbol']],
     'selected_skills':[before['final_goal'],after['final_goal']],
     'target_skills':[before['target_color_index'],after['target_color_index']],
     'both_language_correct':before['language_correct'] and after['language_correct'],
     'both_grounding_correct':before['grounding_correct'] and after['grounding_correct'],
     'both_success':before['success'] and after['success']})
summary={'status':'completed' if len(audit)==len(protocol['scenarios'])*len(protocol['modes']) else 'partial',
         'counts':{m:counts(rows) for m,rows in by_mode.items()},
         'by_template':{m:{t:counts([r for r in rows if r['template_set']==t]) for t in sorted({r['template_set'] for r in rows})} for m,rows in by_mode.items()},
         'by_symbol':{m:{s:counts([r for r in rows if r['target_symbol']==s]) for s in sorted({r['target_symbol'] for r in rows})} for m,rows in by_mode.items()},
         'counterfactual_same_scene':counterfactual,'paired_feedback_periodic':paired,'audited_jobs':audit,'initial_state_differences':differences,
         'scope':'Small v3 validation; no open-vocabulary, autonomous termination, slip estimation, or statistical superiority claim. No CI for this small seed sample.'}
probe_path=root/'language_probe.json'
if probe_path.exists():
 language_probe=json.loads(probe_path.read_text())
 summary['language_probe']={g:{'examples':sum(r['group']==g for r in language_probe['rows']),'correct':sum(r['correct'] for r in language_probe['rows'] if r['group']==g)} for g in {r['group'] for r in language_probe['rows']}}
(root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','counts','by_template','paired_feedback_periodic','scope']},indent=2))
