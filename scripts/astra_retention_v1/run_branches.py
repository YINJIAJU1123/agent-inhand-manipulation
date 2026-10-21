"""Bounded, auditable local branch experiments with live process checks."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save,sha
from prepare import environment_variables,KIT
p=argparse.ArgumentParser();p.add_argument('--replays',required=True);p.add_argument('--output',required=True);a=p.parse_args()
replays=Path(a.replays).resolve();assert json.loads((replays/'progress.json').read_text())['status']=='completed'
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
for name in ('tmp','logs','data'):(out/name).mkdir()
state=dict(status='starting',pid=os.getpid(),completed=[]);child=None
lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
def interrupt(*_):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 subprocess.run([PYTHON,str(Path(__file__).with_name('analyze_replays.py')),'--data',str(replays/'data'),'--output',str(out)],check=True)
 plan=out/'sensor_branch_plan.json';entries=json.loads(plan.read_text());sources=sorted(set(r['source']for r in entries))
 jobs=[(r['source'],r['case'],rate)for r in entries for rate in (1.,.25,0.)]
 frozen=sha(Path(__file__).with_name('branch.py'))
 save(out/'protocol.json',dict(jobs=jobs,source_sha256=frozen,plan_sha256=sha(plan),cloud_calls=0,trigger='First frozen sensor pose probability >=0.7 after a delivered Astra binding, before episode termination. No ground-truth trigger.',horizon=180,quality='Fraction of six-second window within original 0.16rad orientation tolerance, minus drop indicator',purpose='Diagnose physical benefit of retaining completion while remote decision is pending. These inspected historical states are development, never final evaluation.'))
 for src,case,rate in jobs:
  assert sha(Path(__file__).with_name('branch.py'))==frozen
  name=Path(src).name+'_case'+str(case)+'_rate'+str(rate);dest=out/'data'/name
  command=[PYTHON,str(Path(__file__).with_name('branch.py')),'--source',src,'--plan',str(plan),'--output',str(dest),'--rate',str(rate),'--case',str(case),'--headless','--device','cuda:0']+KIT
  with (out/'logs'/(name+'.log')).open('w')as log:
   child=subprocess.Popen(command,cwd=ROOT,env=environment_variables(out),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  start=time.time()
  while child.poll() is None:
   progress(status='branching',current=name,child_pid=child.pid,elapsed_s=time.time()-start)
   if (dest/'failure.json').exists():raise RuntimeError((dest/'failure.json').read_text())
   if time.time()-start>1800:raise TimeoutError(name)
   time.sleep(5)
  if child.returncode or not(dest/'report.json').exists():raise RuntimeError('Branch incomplete: '+name)
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
