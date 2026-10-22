"""Finite 32-iteration maintenance training and two-seed paired validation."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/workspace/code');sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha,STUDENT,ADAPTER,CONTROLLER
from prepare import KIT
import torch
out=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_goal_maintenance_20261022/train_eval_r1');out.mkdir(parents=True,exist_ok=False)
for name in ('logs','tmp'):(out/name).mkdir()
script=ROOT/'scripts/astra_goal_maintenance_v1/train.py';model=script.with_name('model.py')
frozen={str(p):sha(p) for p in [script,model,STUDENT,ADAPTER,CONTROLLER]}
save(out/'protocol.json',dict(training_seeds=list(range(124101,124133)),iterations=32,eval_seeds=[124201,124202],eval_delay=6,cases_per_seed=12,source_sha256=frozen,cloud_calls=0,
 gate='Exact reply-state physical/sensor prefix, >=2 more final six-step correct holds, no extra drops, and >=.05 greater nominal ready-step fraction versus frozen original controller.',
 interpretation='Local public-color control through execution, a six-second wait and reaffirmation. No Astra, semantic-binding or autonomous-stop performance claim. No automatic deployment.'))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TMPDIR=str(out/'tmp'),PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),str(ROOT/'outputs/learning_dependencies'),os.environ.get('PYTHONPATH','')]))
child=None;state=dict(status='starting',completed=[],cloud_calls=0);started=time.monotonic()
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state);print('MAINTENANCE_PROGRESS '+json.dumps(state),flush=True)
def run(name,args,timeout):
 global child
 for p,h in frozen.items():assert sha(p)==h
 dest=out/name;cmd=[sys.executable,str(script),'--output',str(dest),'--num-envs','12','--headless','--device','cuda:0']+args+KIT
 save(out/'logs'/(name+'_command.json'),cmd)
 with (out/'logs'/(name+'.log')).open('w') as f:child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 start=time.monotonic()
 while child.poll() is None:
  detail={}
  for file in ('progress.json','rollout_progress.json'):
   q=dest/file
   if q.exists():detail[file]=json.loads(q.read_text())
  progress(status='training' if name=='train' else 'evaluating',current=name,child_pid=child.pid,detail=detail)
  if time.monotonic()-start>timeout or time.monotonic()-started>3900:raise TimeoutError(name)
  time.sleep(20)
 if child.returncode or not (dest/'result.json').exists():raise RuntimeError((dest/'failure.json').read_text() if (dest/'failure.json').exists() else (out/'logs'/(name+'.log')).read_text()[-1500:])
 progress(completed=state['completed']+[name],child_pid=None)
try:
 run('train',['--seed','124101','--iterations','32'],3000)
 candidate=out/'train/controller_final.pt';frozen[str(candidate)]=sha(candidate);audit=[];results={}
 for seed in (124201,124202):
  run(f'original_{seed}',['--seed',str(seed),'--mode','original'],500)
  run(f'learned_{seed}',['--seed',str(seed),'--mode','learned','--source',str(out/f'original_{seed}'),'--checkpoint',str(candidate)],500)
  b=torch.load(out/f'original_{seed}/branch.pt',map_location='cpu',weights_only=False);c=torch.load(out/f'learned_{seed}/branch.pt',map_location='cpu',weights_only=False)
  errors={k:float((b['physical'][k]-c['physical'][k]).abs().max()) for k in b['physical']}
  errors.update({k:float((b[k].float()-c[k].float()).abs().max()) for k in ['ih','ph','prop','active']});assert max(errors.values())==0
  audit.append(dict(seed=seed,errors=errors));save(out/'prefix_audit.json',audit)
 for arm in ('original','learned'):
  rows=sum([json.loads((out/f'{arm}_{seed}/trials.json').read_text()) for seed in (124201,124202)],[])
  results[arm]=dict(nominal=len(rows),successes=sum(r['success'] for r in rows),drops=sum(r['drop'] for r in rows),prefix_drops=sum(r['prefix_drop'] for r in rows),ready_fraction=sum(r['ready_steps'] for r in rows)/(len(rows)*300))
 b,c=results['original'],results['learned'];passed=c['successes']>=b['successes']+2 and c['drops']<=b['drops'] and c['ready_fraction']>=b['ready_fraction']+.05
 result=dict(status='completed',results=results,local_gate_passed=passed,cloud_calls=0,deployed=False,checkpoint_sha256=sha(candidate));save(out/'result.json',result);print('MAINTENANCE_RESULT '+json.dumps(result),flush=True);progress(status='completed',current=None)
except BaseException:save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=traceback.format_exc()[-1400:]);raise
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
