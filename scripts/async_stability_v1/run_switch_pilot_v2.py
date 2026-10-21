"""Finite V3 goal-switch fine-tuning and two-seed paired validation; no API."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha,PYTHON,CONTROLLER,STUDENT,ADAPTER
from prepare import environment_variables,KIT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for x in ('tmp','logs'):(out/x).mkdir()
 script=Path(__file__).with_name('train_goal_switch.py')
 frozen={str(f):sha(f) for f in (script,CONTROLLER,STUDENT,ADAPTER,ROOT/'scripts/agentic/recovery_controller_v3.py',ROOT/'scripts/agentic/recovery_controller_v2.py')}
 jobs=[dict(name='train',args=['--mode','train','--seed','97311','--iterations','8','--train-delays','2','6','--checkpoint',str(CONTROLLER),'--switch-goals'],timeout=3600)]
 for seed in (97401,97402):
  cache=out/f'cache_{seed}'/'request_state.pt'
  jobs.append(dict(name=f'cache_{seed}',args=['--mode','cache','--seed',str(seed)],timeout=600))
  for name,ck in [('baseline',CONTROLLER),('candidate',out/'train/controller_final.pt')]:
   jobs.append(dict(name=f'{name}_{seed}',args=['--mode','learned','--seed',str(seed),'--delay','6','--request-cache',str(cache),'--checkpoint',str(ck),'--switch-goals'],timeout=1200))
 save(out/'protocol.json',dict(jobs=jobs,source_sha256=frozen,training_seeds=list(range(97311,97319)),evaluation_seeds=[97401,97402],cloud_calls=0,student_frozen=True,baseline_frozen=True,initialization_unchanged=True,interpretation='Synthetic same-goal and changed-goal replies, disjoint train/eval seeds; not fresh Astra evaluation. Report preparation failures and separate same/change strata. No automatic checkpoint promotion.'))
 state=dict(pid=os.getpid(),status='waiting_previous',completed=[],cloud_calls=0);child=None;lock=None
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def interrupt(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:
  progress();deadline=time.monotonic()+3600
  while True:
   prior=json.loads((ROOT/'outputs/async_stability_v1/startup_validation_retry_20261021/progress.json').read_text())
   if prior['status'] in ('completed','failed','interrupted'):break
   if time.monotonic()>deadline:raise TimeoutError('Previous finite job did not finish')
   time.sleep(5)
  lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=environment_variables(out)
  for job in jobs:
   for f,h in frozen.items():
    if sha(f)!=h:raise RuntimeError('Frozen source or weight changed')
   cmd=[PYTHON,str(script),'--headless','--device','cuda:0','--num-envs','12','--output',str(out/job['name'])]+job['args']+KIT
   save(out/'logs'/(job['name']+'_command.json'),cmd)
   with (out/'logs'/(job['name']+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
    while child.poll() is None:
     progress(status='training' if job['name']=='train' else 'evaluating',current=job['name'],child_pid=child.pid,wall_s=time.monotonic()-start)
     if time.monotonic()-start>job['timeout']:raise TimeoutError(job['name'])
     time.sleep(5)
   if child.returncode:raise RuntimeError(job['name']+': exit '+str(child.returncode))
   if (out/job['name']/'failure.json').exists() or not (out/job['name']/'complete.json').exists():raise RuntimeError('Missing completion artifact: '+job['name'])
   state['completed'].append(job['name']);progress(child_pid=None)
  results={}
  for name in ('baseline','candidate'):
   rows=sum([json.loads((out/f'{name}_{seed}'/'eval_trials.json').read_text()) for seed in (97401,97402)],[])
   results[name]={}
   for group,subset in [('all',rows),('same',[r for r in rows if not r['goal_changed']]),('changed',[r for r in rows if r['goal_changed']])]:
    results[name][group]=dict(n=len(subset),eligible=sum(r['eligible'] for r in subset),prefix_failures=sum(not r['eligible'] for r in subset),success=sum(r['eligible'] and r['success'] for r in subset),wait_drop=sum(r['wait_drop'] for r in subset),resume_drop=sum(r['resume_drop'] for r in subset))
  save(out/'summary.json',dict(results=results,cloud_calls=0,promoted=False));progress(status='completed',current=None,child_pid=None)
 except BaseException as error:progress(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
 finally:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
  if lock:lock.close()
if __name__=='__main__':main()
