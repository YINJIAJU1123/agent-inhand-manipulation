"""Finite dependency runner, tied to a live replay process, not a scheduled task."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save
p=argparse.ArgumentParser();p.add_argument('--replays',required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
replays=Path(a.replays).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);child=None
state=dict(pid=os.getpid(),status='waiting_for_replays',dependency_pid=a.pid)
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
def interrupt(*_):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
try:
 while True:
  statefile=json.loads((replays/'progress.json').read_text())
  if statefile['status']=='completed':break
  if statefile['status']=='failed':raise RuntimeError('Replay failed; not proceeding')
  # A stale status file is not sufficient evidence that a job is alive.
  os.kill(a.pid,0);cmd=Path(f'/proc/{a.pid}/cmdline').read_bytes()
  if b'run_replays.py'not in cmd:raise RuntimeError('Dependency PID no longer identifies replay runner')
  progress(dependency_status=statefile['status'],dependency_child_pid=statefile.get('child_pid'));time.sleep(5)
 for name,command in [
  ('branches',[PYTHON,str(Path(__file__).with_name('run_branches.py')),'--replays',str(replays),'--output',str(out/'branches')]),
  ('analyze',[PYTHON,str(Path(__file__).with_name('analyze_branches.py')),'--output',str(out/'branches')])]:
  with (out/(name+'.log')).open('w')as log:child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  while child.poll() is None:progress(status=name,child_pid=child.pid);time.sleep(5)
  if child.returncode:raise RuntimeError(name+' failed')
 progress(status='completed',child_pid=None)
except BaseException as e:
 progress(status='failed',error=str(e));save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
