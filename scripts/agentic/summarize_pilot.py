"""Aggregate counts and audit paired initial states, without selecting successes."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('directory'); a=p.parse_args()
root=Path(a.directory); grouped=defaultdict(list); initials=defaultdict(dict); sources={}
for path in sorted(root.glob('seed*/report.json')):
    report=json.loads(path.read_text())
    for row in report['episodes']: grouped[row['mode']].append(row)
    sources[path.parent.name]=report['metadata'].get('source_sha256',{})
    for initial in path.parent.glob('initial_*.json'):
        value=json.loads(initial.read_text()); seed=report['metadata']['seeds'][0]; mode=report['metadata']['modes'][0]
        initials[seed][mode]=value
summary={}
for mode,rows in grouped.items():
    summary[mode]={'episodes':len(rows),'success':sum(x['success'] for x in rows),'drop':sum(x['drop'] for x in rows),'grounding_correct':sum(x['grounding_correct'] for x in rows),'calls':sum(x['calls'] for x in rows),'recovery_episodes':sum(x['recovery_used'] for x in rows),'abstentions':sum(x['steps']==0 for x in rows),'vlm_time_s':sum(t.get('elapsed_s',0) for x in rows for t in x['trace'])}
pair_audit={}
for seed,modes in initials.items():
    baseline=modes.get('oracle',next(iter(modes.values())))
    pair_audit[seed]={m:{k:float(np.max(np.abs(np.asarray(v[k])-np.asarray(baseline[k])))) for k in ['object_rotation','object_position','proprio','layout']} for m,v in modes.items()}
result={'counts':summary,'paired_initial_max_absolute_difference':pair_audit,'source_hashes':sources}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
