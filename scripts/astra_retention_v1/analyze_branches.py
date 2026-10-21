"""Require identical physical intervention states before comparing rates."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
groups=defaultdict(dict);by_rate=defaultdict(list);audits=[]
for file in sorted((out/'data').glob('*/report.json')):
 report=json.loads(file.read_text());states=torch.load(file.parent/'branches.pt',map_location='cpu',weights_only=True)['states']
 for row in report['results']:
  key=(report['source'],row['case']);groups[key][report['rate']]=(row,states[row['case']]);by_rate[report['rate']].append(row)
for key,group in groups.items():
 if set(group)!={0.,.25,1.}:raise RuntimeError('Incomplete matched branches')
 reference=group[1.][1]
 maximum=max(float((state[name]-reference[name]).abs().max())for _,state in group.values()for name in reference)
 if maximum>1e-6:raise RuntimeError('Intervention states differ: '+str(key))
 audits.append(dict(source=key[0],case=key[1],max_abs_error=maximum))
summary={}
for rate,rows in by_rate.items():
 correct=[r for r in rows if r['binding_correct']]
 summary[str(rate)]=dict(branches=len(rows),drops=sum(r['drop']for r in rows),initial_ready=sum(r['initial_ready']for r in rows),final_ready=sum(r['final_ready']for r in rows),mean_ready_fraction=sum(r['ready_fraction']for r in rows)/len(rows),mean_quality=sum(r['quality']for r in rows)/len(rows),correct_binding_branches=len(correct),semantically_correct_final=sum(r['final_ready']for r in correct))
selection=[]
for key,group in groups.items():
 rows=[v[0]for v in group.values()];best=max(rows,key=lambda r:r['quality'])
 selection.append(dict(source=key[0],case=key[1],best_rate=best['rate'],best_quality=best['quality'],continue_quality=group[1.][0]['quality'],per_rate={str(r['rate']):r['quality']for r in rows}))
result=dict(rates=summary,paired_state_audits=audits,counterfactual_training_labels=selection,cloud_calls=0,interpretation='Local six-second interventions on recorded Astra hypotheses. These labels are for learning action selection; not actual online Astra task results. Best-rate labels use simulator outcomes, never an inference input.')
save(out/'summary.json',result);print(json.dumps(summary,indent=2))
