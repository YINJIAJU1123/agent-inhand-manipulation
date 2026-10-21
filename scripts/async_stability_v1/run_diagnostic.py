"""Finite local diagnostic batches. No network calls or recurring automation."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'scripts/semantic_completion_v1'))
from common import save,PYTHON,sha
from prepare import environment_variables,KIT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=['cold','handoff'],required=True);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 for name in ['logs','tmp']:(out/name).mkdir()
 env=environment_variables(out);lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 evaluator=Path(__file__).with_name('evaluate.py');digest=sha(evaluator);child=None
 if a.stage=='cold':
  jobs=[dict(seed=seed,repeat=1,template='canonical',cold=mode,handoff='v3',steps=360) for seed in [74101,74102] for mode in ['latch','mean','median','nearest','red']]
 else:
  jobs=[dict(seed=seed,repeat=1,template=template,cold='latch',handoff=mode,steps=1800) for seed,template in [(74101,'canonical'),(74102,'paraphrase')] for mode in ['v3','direct','ramp','cap']]
 save(out/'protocol.json',dict(jobs=jobs,evaluator_sha256=digest,cloud_calls=0,scope='Development diagnostic on historical seeds, fixed recorded semantic calls. Not fresh Astra evaluation.'))
 state=dict(pid=os.getpid(),stage=a.stage,status='running',completed=[])
 def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
 def interrupt(*_):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 try:
  for job in jobs:
   if sha(evaluator)!=digest:raise RuntimeError('Diagnostic source changed')
   name=f"s{job['seed']}_{job['template']}_{job['cold']}_{job['handoff']}"
   source=ROOT/f"outputs/astra_repeated_eval_v1/repeat2_20261019/eval_r{job['repeat']}_{job['seed']}_{job['template']}_v3/report.json"
   cmd=[PYTHON,str(evaluator),'--headless','--device','cuda:0','--output',str(out/name),'--source-report',str(source),'--mode','cold' if a.stage=='cold' else 'replay','--cold-policy',job['cold'],'--handoff',job['handoff'],'--max-steps',str(job['steps'])]+KIT
   with (out/'logs'/f'{name}.log').open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);started=time.monotonic()
    while child.poll() is None:
     progress(current=name,child_pid=child.pid,wall_s=time.monotonic()-started)
     if time.monotonic()-started>1800:raise TimeoutError(name)
     time.sleep(5)
   if child.returncode:raise RuntimeError(name+': exit '+str(child.returncode))
   state['completed'].append(name);progress(child_pid=None)
   reports={p.parent.name:json.loads(p.read_text()) for p in out.glob('s*/report.json')}
   save(out/'summary.json',dict(status='running',reports=reports,cloud_calls=0))
  progress(status='completed',current=None,child_pid=None)
  summary=json.loads((out/'summary.json').read_text());summary['status']='completed';save(out/'summary.json',summary)
 except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
 finally:
  if child and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
  lock.close()
if __name__=='__main__':main()
