"""Matched active-goal vs constant-goal W/E training, independent of formal data."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--project',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();project=a.project.resolve();src=root/'training_sources'
 if (root/'training_plan.json').exists():raise FileExistsError('Training already registered; inspect its process before resuming')
 src.mkdir()
 for f in (project/'scripts/pose_latent_student_v1').glob('*.py'):shutil.copy2(f,src/f.name)
 stages=[]
 for phase,updates,seed in [('waiting',1200,182841),('execution',2400,182871)]:
  for mode in ['active','center']:
   out=root/(phase+'_'+mode)
   command=[sys.executable,str(src/'train_modules.py'),'--data',str(root/('data_v1' if phase=='waiting' else 'data_v6')),
    '--text_cache',str(project/'outputs/visual_student_v3/checkpoints/siglip2_text_cache_v2.pt'),'--original_asset',str(a.asset.resolve()),
    '--output',str(out),'--updates',str(updates),'--seed',str(seed),'--actor_goal_input',mode]
   if phase=='execution':command+=['--actor_warm_start',str(root/('waiting_'+mode)/'maintenance_actor.pt'),
    '--paired_execution',str(root/'pairs'),'--balance_binding_state','--align_actor_elapsed']
   stages.append(dict(name=out.name,command=command))
 plan=dict(format='matched_goal_ablation_training_v1',created_at=datetime.now(timezone.utc).isoformat(),
  pid=os.getpid(),status='running',formal_trials_used=False,
  scope='Both active and center references retrained. Do not substitute original cube-v1 full results for the new active reference.',
  pipeline='Identical initial RNG and architecture. 1200-update waiting fit then own waiting warm-start for 2400-update paired execution fit.',
  teacher_student_goals_unchanged=True,actor_only_explicit_goal_ablation=True,
  source_sha256={f.name:sha(f) for f in src.glob('*.py')},stages=stages)
 (root/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 state=dict(pid=os.getpid(),status='running',completed=[])
 try:
  for s in stages:
   state['current']=s['name'];(root/'training_status.json').write_text(json.dumps(state,indent=2)+'\n')
   with (root/(s['name']+'.log')).open('x') as log:
    child=subprocess.Popen(s['command'],stdout=log,stderr=subprocess.STDOUT,cwd=project)
    state['child_pid']=child.pid;(root/'training_status.json').write_text(json.dumps(state,indent=2)+'\n')
    code=child.wait()
   if code:raise RuntimeError(s['name']+' exit '+str(code))
   completed=json.loads((root/s['name']/'completion.json').read_text());assert completed['completed']
   state['completed'].append(s['name']);print(s['name']+' completed',flush=True)
  state.update(status='completed',current=None,child_pid=None)
 except BaseException as e:
  state.update(status='failed',error=str(e));raise
 finally:(root/'training_status.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
