"""Bounded local causal follow-up, queued behind the initial cold diagnostic."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import save,PYTHON,sha
from prepare import environment_variables,KIT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for x in ['logs','tmp']:(out/x).mkdir()
 prior=ROOT/'outputs/async_stability_v1/cold_diagnostic_20261021/progress.json'
 state=dict(pid=os.getpid(),status='waiting_previous',completed=[],cloud_calls=0);child=None;lock=None
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def interrupt(*_):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 jobs=[]
 for goal in ['green','blue','yellow','magenta','cyan']:
  jobs.append(dict(name='boot_'+goal,script='evaluate_bootstrap.py',seed=74101,template='canonical',steps=90,args=['--cold-policy',goal,'--bootstrap-steps','15']))
 for lift in [.01,.02,.03]:
  jobs.append(dict(name='init_lift'+str(lift),script='evaluate_initialization.py',seed=74101,template='canonical',steps=180,args=['--initial-lift',str(lift),'--cold-policy','latch']))
 # Two representative live drops, one on each historical seed. Build fresh
 # continuous baselines and then replay exactly their action prefixes.
 for seed,template,t in [(74101,'canonical',1364),(74102,'paraphrase',669)]:
  base=f'base_{seed}_{template}'
  jobs.append(dict(name=base,script='evaluate_branch.py',seed=seed,template=template,steps=t+180,args=['--mode','replay','--handoff','v3']))
  for handoff in ['direct','ramp','cap']:
   jobs.append(dict(name=f'{base}_{handoff}',script='evaluate_branch.py',seed=seed,template=template,steps=t+180,args=['--mode','replay','--handoff',handoff,'--prefix-cache',str(out/base/'prefix.pt'),'--branch-step',str(t)]))
 hashes={job['script']:sha(Path(__file__).with_name(job['script'])) for job in jobs}
 save(out/'protocol.json',dict(jobs=jobs,source_sha256=hashes,cloud_calls=0,purpose='Mechanism diagnosis. Initial-lift results explicitly change initial placement; fixed-reply branching is not fresh Astra evaluation.'))
 try:
  progress();deadline=time.monotonic()+1800
  while True:
   previous=json.loads(prior.read_text())
   if previous['status']=='completed':break
   if previous['status'] in ('failed','interrupted'):raise RuntimeError('Prior diagnostic did not finish')
   if time.monotonic()>deadline:raise TimeoutError('Prior diagnostic wait')
   time.sleep(5)
  lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=environment_variables(out)
  for job in jobs:
   script=Path(__file__).with_name(job['script'])
   if sha(script)!=hashes[job['script']]:raise RuntimeError('Source changed')
   source=ROOT/f"outputs/astra_repeated_eval_v1/repeat2_20261019/eval_r1_{job['seed']}_{job['template']}_v3/report.json"
   cmd=[PYTHON,str(script),'--headless','--device','cuda:0','--output',str(out/job['name']),'--source-report',str(source),'--max-steps',str(job['steps'])]+job['args']+KIT
   save(out/'logs'/(job['name']+'_command.json'),cmd)
   with (out/'logs'/(job['name']+'.log')).open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
    while child.poll() is None:
     progress(status='running',current=job['name'],child_pid=child.pid,wall_s=time.monotonic()-start)
     if time.monotonic()-start>1800:raise TimeoutError(job['name'])
     time.sleep(5)
   if child.returncode:raise RuntimeError(job['name']+': exit '+str(child.returncode))
   state['completed'].append(job['name']);progress(child_pid=None)
   reports={p.parent.name:json.loads(p.read_text()) for p in out.glob('*/report.json')};save(out/'summary.json',dict(reports=reports,cloud_calls=0,status='running'))
  progress(status='completed',current=None,child_pid=None)
  summary=json.loads((out/'summary.json').read_text());summary['status']='completed';save(out/'summary.json',summary)
 except BaseException as error:progress(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
 finally:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
  if lock:lock.close()
if __name__=='__main__':main()
