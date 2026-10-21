"""Finite GPU-exclusive replay audit, verifying process and output, no external calls."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save,sha
from prepare import environment_variables,KIT
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
for name in ('tmp','logs','data'):(out/name).mkdir()
state=dict(status='starting',pid=os.getpid(),completed=[]);child=None
lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
def interrupt(*_):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=sorted(Path(a.source).resolve().glob('eval_*collision_clearance/report.json'))
 frozen=sha(Path(__file__).with_name('replay.py'))
 save(out/'protocol.json',dict(jobs=[str(j.parent)for j in jobs],source_sha256=frozen,cloud_calls=0,purpose='Exact action replay and sensor evidence diagnostics; no change to frozen online results.'))
 for report in jobs:
  assert sha(Path(__file__).with_name('replay.py'))==frozen
  name=report.parent.name;dest=out/'data'/name
  command=[PYTHON,str(Path(__file__).with_name('replay.py')),'--source',str(report.parent),'--output',str(dest),'--headless','--device','cuda:0']+KIT
  with (out/'logs'/(name+'.log')).open('w')as log:
   child=subprocess.Popen(command,cwd=ROOT,env=environment_variables(out),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  start=time.time()
  while child.poll() is None:
   progress(status='replaying',current=name,child_pid=child.pid,elapsed_s=time.time()-start)
   if (dest/'failure.json').exists():raise RuntimeError((dest/'failure.json').read_text())
   if time.time()-start>1800:raise TimeoutError(name)
   time.sleep(5)
  if child.returncode or not(dest/'manifest.json').exists():raise RuntimeError('Replay incomplete: '+name)
  manifest=json.loads((dest/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['replay_max_error']<=1e-6
  progress(completed=state['completed']+[name],child_pid=None)
 progress(status='completed',current=None,child_pid=None)
except BaseException as e:
 progress(status='failed',error=str(e));save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 lock.close()
