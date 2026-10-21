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
 jobs=[dict(name=f'placement_{seed}',seed=seed,lift=0.,steps=900) for seed in (74101,74102)]
 frozen={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ('evaluate_placement.py','geometry_v2.py','placement.py')}
 save(out/'protocol.json',dict(jobs=jobs,source_sha256=frozen,cloud_calls=0,interpretation='Minimal vertical collision-clearance correction, compared with original oracle controls after six seconds of holding. No new Astra calls; not a learned-policy gain.'))
 try:
  progress();deadline=time.monotonic()+7200
  while True:
   prior=json.loads((ROOT/'outputs/async_stability_v1/goal_switch_pilot_retry_20261021/progress.json').read_text())
   if prior['status'] in ('completed','failed','interrupted'):break
   if time.monotonic()>deadline:raise TimeoutError('Previous finite job did not finish')
   time.sleep(5)
  lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  env=environment_variables(out)
  for job in jobs:
   for f,h in frozen.items():
    if sha(f)!=h:raise RuntimeError('Frozen source changed')
   source=ROOT/f"outputs/astra_repeated_eval_v1/repeat2_20261019/eval_r1_{job['seed']}_canonical_v3/report.json"
   cmd=[PYTHON,str(Path(__file__).with_name('evaluate_placement.py')),'--headless','--device','cuda:0','--source-report',str(source),'--output',str(out/job['name']),'--max-steps',str(job['steps']),'--initial-lift',str(job['lift']),'--mode','oracle','--handoff','direct','--collision-clearance','--record-video']+KIT
   save(out/'logs'/(job['name']+'_command.json'),cmd)
   with (out/'logs'/(job['name']+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
    while child.poll() is None:
     progress(status='running',current=job['name'],child_pid=child.pid,wall_s=time.monotonic()-start)
     if time.monotonic()-start>600:raise TimeoutError(job['name'])
     time.sleep(5)
   if child.returncode:raise RuntimeError(job['name']+': exit '+str(child.returncode))
   if (out/job['name']/'failure.json').exists() or not (out/job['name']/'report.json').exists():raise RuntimeError('Missing valid report: '+job['name'])
   state['completed'].append(job['name']);progress(child_pid=None)
  reports={f.parent.name:json.loads(f.read_text()) for f in out.glob('*/report.json')}
  save(out/'summary.json',dict(reports=reports,cloud_calls=0));progress(status='completed',current=None,child_pid=None)
 except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
 finally:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
  if lock:lock.close()
if __name__=='__main__':main()
