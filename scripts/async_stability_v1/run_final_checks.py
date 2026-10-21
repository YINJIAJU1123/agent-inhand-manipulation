"""Original-init V3 controls and API-free integration smoke; finite local queue."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import save,sha,PYTHON
from prepare import environment_variables,KIT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for x in ('logs','tmp'):(out/x).mkdir()
 jobs=[]
 for seed in (74101,74102):
  jobs.append(dict(name=f'legacy_v3_{seed}',script='evaluate_first_skill.py',args=['--source-report',str(ROOT/f'outputs/astra_repeated_eval_v1/repeat2_20261019/eval_r1_{seed}_canonical_v3/report.json'),'--mode','oracle','--handoff','v3','--max-steps','900']))
 for reset in ('legacy','collision_clearance'):
  jobs.append(dict(name='online_smoke_'+reset,script='evaluate_online.py',args=['--seed','74101','--arm','verify','--initialization',reset,'--evidence-model',str(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'),'--max-steps','120','--budget','3','--offline-smoke']))
 frozen={str(Path(__file__).with_name(f)):sha(Path(__file__).with_name(f)) for f in ('evaluate_first_skill.py','evaluate_online.py','placement.py','geometry_v2.py')}
 save(out/'protocol.json',dict(jobs=jobs,source_sha256=frozen,cloud_calls=0,interpretation='Legacy-initialization V3 first-skill controls plus offline plumbing tests. No API call authorized or made by this runner.'))
 state=dict(pid=os.getpid(),status='waiting_previous',completed=[],cloud_calls=0);child=None;lock=None
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def interrupt(*args):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:
  progress();deadline=time.monotonic()+3600
  while True:
   prior=json.loads((ROOT/'outputs/async_stability_v1/gated_validation_exact_20261021/progress.json').read_text())
   if prior['status'] in ('completed','failed','interrupted'):break
   if time.monotonic()>deadline:raise TimeoutError('Previous finite job did not finish')
   time.sleep(5)
  lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=environment_variables(out)
  for job in jobs:
   for f,h in frozen.items():
    if sha(f)!=h:raise RuntimeError('Frozen source changed')
   cmd=[PYTHON,str(Path(__file__).with_name(job['script'])),'--headless','--device','cuda:0','--output',str(out/job['name'])]+job['args']+KIT
   save(out/'logs'/(job['name']+'_command.json'),cmd)
   with (out/'logs'/(job['name']+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
    while child.poll() is None:
     progress(status='running',current=job['name'],child_pid=child.pid,wall_s=time.monotonic()-start)
     if time.monotonic()-start>600:raise TimeoutError(job['name'])
     time.sleep(5)
   report=out/job['name']/'report.json'
   if child.returncode or not report.exists() or any((out/job['name']/f).exists() for f in ('failure.json','runtime_failure.json')):raise RuntimeError('Missing valid report: '+job['name'])
   if job['name'].startswith('online_smoke'):
    data=json.loads(report.read_text());assert data['metadata']['offline_smoke']
    assert (out/job['name']/'actions.pt').exists() and (out/job['name']/'response_states.pt').exists()
   state['completed'].append(job['name']);progress(child_pid=None)
  progress(status='completed',current=None,child_pid=None)
 except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
 finally:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
  if lock:lock.close()
if __name__=='__main__':main()
