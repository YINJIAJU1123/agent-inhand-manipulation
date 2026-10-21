"""Finite local collection, counterfactual labels, and supervised value training.

Waits for a specific live diagnostic process, validates its finished results,
then uses independent new seeds. No cloud calls or automatic model promotion.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save,sha
from prepare import environment_variables,KIT
p=argparse.ArgumentParser();p.add_argument('--diagnostics',required=True);p.add_argument('--dependency-pid',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
diagnostics=Path(a.diagnostics).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
for d in ('logs','tmp','data'):(out/d).mkdir()
state=dict(pid=os.getpid(),status='waiting_for_diagnostics',completed_jobs=[]);child=None;lock=None
schedule=[('train',s)for s in range(104201,104209)]+[('calibration',104301),('calibration',104302),('audit',104401),('audit',104402)]
frozen={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')}
save(out/'protocol.json',dict(schedule=schedule,cases_per_seed=12,nominal_states=144,rates=[1.,.25,0.],horizon=180,source_sha256=frozen,cloud_calls=0,goal='Learn a sensor-conditioned action value for keeping the target during remote deliberation. Final goal remains >80% real Astra autonomous-stop success; local results do not fulfill it.',promotion='No automatic promotion. Check held-out local counterfactual results, then closed-loop and real Astra evaluation.',diagnostic_gate='At least eight matched states, oracle per-state rate quality exceeds continuing by 0.03 on average. No training if retention has no measurable local opportunity.',training_data='Balanced supplied goals arrive at six seconds; frozen V3 approaches target; shared batch branch once half the nominal cases reach frozen sensor readiness >=0.7, or at step 720; retain all active cases including non-ready states; include every prefix failure. Future outcomes only labels.'))
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
def interrupt(*_):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
def run(name,command,result,failure):
 global child
 for path,digest in frozen.items():
  if sha(path)!=digest:raise RuntimeError('Frozen learning source changed: '+path)
 save(out/'logs'/(name+'_command.json'),command)
 with (out/'logs'/(name+'.log')).open('w')as log:child=subprocess.Popen(command,cwd=ROOT,env=environment_variables(out),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 started=time.time()
 while child.poll() is None:
  progress(status='training' if name=='train'else 'collecting',current=name,child_pid=child.pid,elapsed_s=time.time()-started)
  if failure.exists():raise RuntimeError(failure.read_text())
  heartbeat=result.parent/'progress.json'
  if heartbeat.exists()and time.time()-heartbeat.stat().st_mtime>300:raise TimeoutError(name+' stopped reporting progress')
  if time.time()-started>2700:raise TimeoutError(name+' exceeded stage timeout')
  time.sleep(5)
 if child.returncode or not result.exists():raise RuntimeError(name+' incomplete')
 progress(completed_jobs=state['completed_jobs']+[name],child_pid=None)
try:
 start=time.time()
 while True:
  dep=json.loads((diagnostics/'progress.json').read_text())
  if dep['status']=='completed':break
  if dep['status']=='failed':raise RuntimeError('Local diagnostic failed')
  os.kill(a.dependency_pid,0)
  if b'continue_local.py'not in Path(f'/proc/{a.dependency_pid}/cmdline').read_bytes():raise RuntimeError('Dependency PID changed')
  progress(dependency_pid=a.dependency_pid,dependency_child=dep.get('child_pid'))
  if time.time()-start>14400:raise TimeoutError('Diagnostic dependency exceeded four hours')
  time.sleep(5)
 summary=json.loads((diagnostics/'branches/summary.json').read_text());labels=summary['counterfactual_training_labels']
 opportunity=sum(r['best_quality']-r['continue_quality']for r in labels)/max(1,len(labels))
 save(out/'diagnostic_gate.json',dict(states=len(labels),mean_oracle_improvement=opportunity,passed=len(labels)>=8 and opportunity>.03))
 if len(labels)<8 or opportunity<=.03:
  progress(status='diagnostic_gate_failed',reason='No supported retention opportunity; training not started');sys.exit(0)
 lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for split,seed in schedule:
  shard=out/'data'/f'{split}_{seed}';shard.mkdir();prefix=shard/'prefix'
  run(f'{split}_{seed}_prefix',[PYTHON,str(Path(__file__).with_name('collect_prefix.py')),'--output',str(prefix),'--seed',str(seed),'--split',split,'--cases','12','--headless','--device','cuda:0']+KIT,prefix/'report.json',prefix/'failure.json')
  for rate in (1.,.25,0.):
   branch=shard/('rate'+str(rate))
   run(f'{split}_{seed}_rate{rate}',[PYTHON,str(Path(__file__).with_name('branch.py')),'--source',str(prefix),'--plan',str(prefix/'sensor_branch_plan.json'),'--output',str(branch),'--rate',str(rate),'--headless','--device','cuda:0']+KIT,branch/'report.json',branch/'failure.json')
 train=out/'train'
 run('train',[PYTHON,str(Path(__file__).with_name('train_value.py')),'--data',str(out/'data'),'--output',str(train)],train/'result.json',train/'failure.json')
 result=json.loads((train/'result.json').read_text());progress(status='completed',current=None,child_pid=None,offline_gate_passed=result['offline_gate_passed'],cloud_eligible=False)
except SystemExit:
 raise
except BaseException as e:
 progress(status='failed',error=str(e));save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 if lock:lock.close()
