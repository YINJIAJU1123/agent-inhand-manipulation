"""Finite startup geometry/hold validation after the causal diagnostic."""
import argparse,fcntl,json,os,signal,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha,PYTHON
from prepare import environment_variables,KIT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for name in ('logs','tmp'):(out/name).mkdir()
 state=dict(pid=os.getpid(),status='waiting_previous',completed=[],cloud_calls=0);child=None;lock=None
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def interrupt(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 jobs=[dict(name=f'{arm}_{seed}',seed=seed,arm=arm,lift=0.,steps=900) for seed in (97401,97402) for arm in ('baseline','candidate')]
 frozen={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ('evaluate_switch_gated_v2.py',)}
 save(out/'protocol.json',dict(jobs=jobs,source_sha256=frozen,cloud_calls=0,interpretation='Gated candidate: original V3 for waiting and same-goal tasks, learned candidate only after a delivered target change. Compare with matching original V3 controls. No Astra calls; not an end-to-end agent result.'))
 try:
  progress();deadline=time.monotonic()+7200
  while True:
   prior=json.loads((ROOT/'outputs/async_stability_v1/first_skill_validation_20261021/progress.json').read_text())
   if prior['status'] in ('completed','failed','interrupted'):break
   if time.monotonic()>deadline:raise TimeoutError('Previous finite job did not finish')
   time.sleep(5)
  lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  env=environment_variables(out)
  for job in jobs:
   for f,h in frozen.items():
    if sha(f)!=h:raise RuntimeError('Frozen source changed')
   pilot=ROOT/'outputs/async_stability_v1/goal_switch_pilot_retry_20261021'
   cmd=[PYTHON,str(Path(__file__).with_name('evaluate_switch_gated_v2.py')),'--headless','--device','cuda:0','--output',str(out/job['name']),'--num-envs','12','--mode','learned','--seed',str(job['seed']),'--delay','6','--switch-goals','--request-cache',str(pilot/f"cache_{job['seed']}"/'request_state.pt'),'--checkpoint',str((pilot/'train/controller_final.pt') if job['arm']=='candidate' else (ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt')),'--anchor-checkpoint',str(ROOT/'outputs/recovery_controller_v3/pilot_20261018/train/controller_final.pt')]+KIT
   if job['arm']=='candidate':cmd+=['--arrival-cache',str(out/f"baseline_{job['seed']}"/'arrival_state.pt')]
   save(out/'logs'/(job['name']+'_command.json'),cmd)
   with (out/'logs'/(job['name']+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
    while child.poll() is None:
     progress(status='running',current=job['name'],child_pid=child.pid,wall_s=time.monotonic()-start)
     if time.monotonic()-start>600:raise TimeoutError(job['name'])
     time.sleep(5)
   if child.returncode:raise RuntimeError(job['name']+': exit '+str(child.returncode))
   if (out/job['name']/'failure.json').exists() or not (out/job['name']/'complete.json').exists():raise RuntimeError('Missing valid report: '+job['name'])
   state['completed'].append(job['name']);progress(child_pid=None)
  reports={f.parent.name:json.loads(f.read_text()) for f in out.glob('*/complete.json')}
  save(out/'summary.json',dict(reports=reports,cloud_calls=0));progress(status='completed',current=None,child_pid=None)
 except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
 finally:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
  if lock:lock.close()
if __name__=='__main__':main()
