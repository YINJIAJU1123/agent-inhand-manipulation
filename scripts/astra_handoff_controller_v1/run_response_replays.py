"""Bounded six-way diagnostic on two already-inspected real reply states."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save,sha
from prepare import environment_variables,KIT
out=ROOT/'outputs/astra_handoff_controller_v1/real_response_diagnostic_20261022';out.mkdir(parents=True,exist_ok=False)
for name in ('tmp','logs'):(out/name).mkdir()
script=Path(__file__).with_name('replay_response.py');frozen=sha(script);state=dict(status='starting',pid=os.getpid(),completed=[],cloud_calls=0);child=None
lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
def interrupt(*_):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
protocol=dict(cases=[dict(seed=110101,case=3,step=372),dict(seed=110102,case=3,step=404)],modes=['original','handoff','smooth'],source_sha256=frozen,cloud_calls=0,
 scope='Selected known development drops after a changed-color reply; not an unbiased success estimate or a model-selection validation set. Fixed delivered command for six seconds, no future recorded agent replies.')
save(out/'protocol.json',protocol)
try:
 results=[]
 for entry in protocol['cases']:
  for mode in protocol['modes']:
   assert sha(script)==frozen
   name=f"{entry['seed']}_{mode}";dest=out/name;src=ROOT/f"outputs/astra_startup_live_v1/continuation12_20261022/eval_{entry['seed']}_fixed_green"
   command=[PYTHON,str(script),'--source',str(src),'--output',str(dest),'--case',str(entry['case']),'--step',str(entry['step']),'--mode',mode,'--headless','--device','cuda:0']+KIT
   save(out/'logs'/(name+'_command.json'),command)
   with (out/'logs'/(name+'.log')).open('w') as f:child=subprocess.Popen(command,cwd=ROOT,env=environment_variables(out),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   started=time.monotonic()
   while child.poll() is None:
    q=dest/'progress.json';progress(status='running',current=name,child_pid=child.pid,detail=json.loads(q.read_text()) if q.exists() else None)
    if time.monotonic()-started>600:raise TimeoutError(name)
    time.sleep(10)
   if child.returncode or not (dest/'result.json').exists():raise RuntimeError(name+' failed; '+(dest/'failure.json').read_text() if (dest/'failure.json').exists() else name+' failed')
   r=json.loads((dest/'result.json').read_text());results.append(dict(seed=entry['seed'],**r));progress(completed=state['completed']+[name],child_pid=None)
 save(out/'result.json',dict(status='completed',cases=results,cloud_calls=0,deployed=False));progress(status='completed',current=None)
except BaseException as e:progress(status='failed',error=str(e));raise
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 lock.close()
