"""Read-only evaluation summary; seed-cluster bootstrap preserves paired cases."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bootstrap',type=int,default=10000);a=p.parse_args()
root=Path(a.directory);frozen=json.loads((root/'freeze.json').read_text());protocol=frozen['protocol']
rows=defaultdict(list);per_seed=defaultdict(dict);initials=defaultdict(dict)
for path in sorted(root.glob('seed*/report.json')):
 report=json.loads(path.read_text());seed=report['metadata']['seeds'][0];mode=report['metadata']['modes'][0]
 if seed not in protocol['seeds'] or mode not in protocol['modes']:raise ValueError(f'Unexpected report: {path}')
 meta=report['metadata']
 if meta['checkpoint_sha256']!=frozen['student_sha256'] or meta['adapter_sha256']!=frozen['adapter_sha256']:raise ValueError(f'Checkpoint fingerprint mismatch: {path}')
 for key,value in meta['source_sha256'].items():
  if value!=frozen['source_sha256'][key]:raise ValueError(f'Source fingerprint mismatch: {path}')
 records=report['episodes']
 if len(records)!=6 or sorted(r['case'] for r in records)!=list(range(6)):raise ValueError(f'Incomplete batch: {path}')
 rows[mode].extend(records);per_seed[mode][seed]=sum(r['success'] for r in records)/6
 initial=path.parent/f'initial_{seed}_{mode}.json';initials[seed][mode]=json.loads(initial.read_text())
def interval(values):
 values=np.asarray(values,dtype=float)
 if len(values)<10:return None
 rng=np.random.default_rng(20261014)
 draws=rng.integers(0,len(values),size=(a.bootstrap,len(values)))
 return np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist()
summary={}
for mode in protocol['modes']:
 records=rows[mode];n=len(records)
 summary[mode]={'episodes':n,'success':sum(r['success'] for r in records),'success_rate':sum(r['success'] for r in records)/n if n else None,'seed_cluster_bootstrap_95ci':interval(list(per_seed[mode].values())),'drop':sum(r['drop'] for r in records),'grounding_correct':sum(r['grounding_correct'] for r in records),'abstentions':sum(r['steps']==0 for r in records),'calls':sum(r['calls'] for r in records),'by_target_face':{i:{'episodes':sum(r['case']==i for r in records),'success':sum(r['case']==i and r['success'] for r in records)} for i in range(6)}}
paired={}
if 'periodic' in per_seed and 'once' in per_seed:
 common=sorted(set(per_seed['periodic'])&set(per_seed['once']))
 differences=[per_seed['periodic'][s]-per_seed['once'][s] for s in common]
 paired={'common_seeds':len(common),'episodes_per_mode':len(common)*6,'mean_success_rate_difference':float(np.mean(differences)) if differences else None,'seed_cluster_bootstrap_95ci':interval(differences)}
audit={}
for seed,modes in initials.items():
 reference=modes.get('oracle',next(iter(modes.values())))
 audit[seed]={mode:{key:float(np.max(np.abs(np.asarray(data[key])-np.asarray(reference[key])))) for key in ['object_rotation','object_position','proprio','layout']} for mode,data in modes.items()}
result={'status':'completed' if sum(len(r) for r in rows.values())==len(protocol['seeds'])*len(protocol['modes'])*6 else 'partial','planned_episodes':len(protocol['seeds'])*len(protocol['modes'])*6,'counts':summary,'paired_periodic_minus_once':paired,'initial_state_max_absolute_differences':audit,'uncertainty_note':'Seeds are resampled as clusters; all six target faces remain together. CI omitted with fewer than 10 completed seeds.'}
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
