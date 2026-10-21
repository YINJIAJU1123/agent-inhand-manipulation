"""Local diagnostic audit; never promotes a threshold or reports live task success."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rows=[];branch_plan=[]
for shard in sorted(Path(a.data).glob('*/data.pt')):
 data=torch.load(shard,map_location='cpu',weights_only=True)
 manifest=json.loads((shard.parent/'manifest.json').read_text())
 assert manifest['replay_max_error']<=1e-6
 source=Path(manifest['source']);report=json.loads((source/'report.json').read_text())
 groups=defaultdict(list)
 for n,r in enumerate(data['rows']):groups[r['case']].append((n,r))
 for case,seq in groups.items():
  original=report['episodes'][case];target=original['target_color'];symbol=original['target_symbol']
  valid=[];pose_ready=[];consecutive=0;triggers={};last_key=None
  for n,r in seq:
   goal=r['claimed_goal'];bound=goal==target and r['claimed_symbol']==symbol and r['state'] in ('execute','verify','stop')
   good=bound and float(data['error'][n,target])<=.16
   if good:valid.append(r['step'])
   key=(goal,r['claimed_symbol']);hypothesis=goal is not None and r['state'] in ('execute','verify','stop')
   high=hypothesis and r['probability'][goal]>=.7
   consecutive=consecutive+1 if high and key==last_key else int(high)
   last_key=key
   for dwell in (1,2,3):
    if consecutive>=dwell and dwell not in triggers:
     triggers[dwell]=dict(step=r['step'],goal=goal,correct=good,pose_error=float(data['error'][n,goal]),binding_correct=bound)
   if hypothesis and float(data['error'][n,goal])<=.16:pose_ready.append(r['step'])
  intervals=[]
  for step in valid:
   if not intervals or step>intervals[-1][-1]+10:intervals.append([step,step])
   else:intervals[-1][-1]=step
  row=dict(source=str(source),case=case,reason=original['reason'],first_reach_step=original['first_reach_step'],finish_step=original['finish_step'],target=target,valid_samples=len(valid),first_valid_sample=valid[0] if valid else None,longest_sampled_valid_window_s=max(((b-a+10)/30 for a,b in intervals),default=0),valid_intervals=intervals,hypothesis_pose_ready_samples=len(pose_ready),sensor_triggers=triggers)
  rows.append(row)
  # Strict diagnostic only: trigger from public learned probability, never evaluator error.
  if 1 in triggers:branch_plan.append(dict(source=str(source),case=case,step=triggers[1]['step'],goal=triggers[1]['goal']))
summary={}
for dwell in (1,2,3):
 t=[r['sensor_triggers'][dwell]for r in rows if dwell in r['sensor_triggers']]
 summary[str(dwell)]=dict(proposals=len(t),correct=sum(r['correct']for r in t),wrong_binding=sum(not r['binding_correct']for r in t),precision=sum(r['correct']for r in t)/len(t)if t else None)
result=dict(episodes=len(rows),learned_first_trigger=summary,rows=rows,note='Fresh rendering, exact physical action replay. Sampled every 10 steps; short reaches may be missed. Dwell comparisons are development diagnostics, not calibrated selections. Recorded Astra schedules are not counterfactual live agent results.')
save(Path(a.output)/'audit.json',result);save(Path(a.output)/'sensor_branch_plan.json',branch_plan)
print(json.dumps(dict(episodes=len(rows),learned_first_trigger=summary),indent=2))
