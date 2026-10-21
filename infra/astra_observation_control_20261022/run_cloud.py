"""Finite two-seed control probe. No external agent requests."""
import hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
out=Path(os.environ['OBSERVATION_CONTROL_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
(out/'tmp').mkdir();(out/'logs').mkdir()
manifest=json.loads((ROOT/'package_manifest.json').read_text())
for name,digest in manifest['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
schedule=[(seed,mode)for seed in (105101,105102)for mode in ('none','execute','all')]
state=dict(status='starting',completed=[],cloud_calls=0);child=None
def progress(**values):
 state.update(values,updated=time.time());save(out/'progress.json',state);print('OBSERVATION_CONTROL_PROGRESS '+json.dumps(state),flush=True)
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',TMPDIR=str(out/'tmp'),PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),str(ROOT/'outputs/learning_dependencies'),os.environ.get('PYTHONPATH','')]))
results=[]
try:
 save(out/'protocol.json',dict(schedule=schedule,source_sha256=manifest['sha256'],cases_per_run=6,max_steps=1800,cloud_calls=0,scope='Fixed public observation and execution colors with deterministic six-second requests. Continuous physics. No autonomous semantic task success.'))
 for seed,mode in schedule:
  name=f'{seed}_{mode}';dest=out/name
  command=[sys.executable,str(ROOT/'scripts/astra_observation_control_v1/evaluate.py'),'--output',str(dest),'--seed',str(seed),'--mode',mode,'--max-steps','1800','--headless','--device','cuda:0']+KIT
  with(out/'logs'/(name+'.log')).open('w')as log:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start=time.time()
  while child.poll()is None:
   p=dest/'progress.json';measured=json.loads(p.read_text())if p.exists()else None
   progress(status='probing',current=name,child_pid=child.pid,child_progress=measured,elapsed_s=time.time()-start)
   if(dest/'failure.json').exists():raise RuntimeError((dest/'failure.json').read_text())
   if time.time()-start>1800:raise TimeoutError(name)
   time.sleep(15)
  if child.returncode or not(dest/'report.json').exists():raise RuntimeError((out/'logs'/(name+'.log')).read_text()[-10000:])
  initial=torch.load(dest/'initial.pt',map_location='cpu',weights_only=True)
  reference=torch.load(out/f'{seed}_none/initial.pt',map_location='cpu',weights_only=True)
  delta=max(float((initial[k]-v).abs().max())for k,v in reference.items())
  if delta>1e-6:raise RuntimeError('Initial physical states differ: '+str(delta))
  result=json.loads((dest/'report.json').read_text());result['initial_max_difference']=delta
  results.append(result);save(out/'partial_results.json',results)
  print('OBSERVATION_CONTROL_RUN '+json.dumps(result),flush=True)
  progress(completed=state['completed']+[name],child_pid=None)
 save(out/'result.json',dict(status='completed',runs=results,cloud_calls=0))
 print('OBSERVATION_CONTROL_RESULT '+json.dumps(dict(runs=results,cloud_calls=0)),flush=True);progress(status='completed',current=None)
except BaseException as error:
 save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=str(error));raise
finally:
 if child and child.poll()is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
