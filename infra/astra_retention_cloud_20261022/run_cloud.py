"""One finite cloud compatibility trial using byte-identical policy sources."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
out=Path(os.environ['RETENTION_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
(out/'logs').mkdir();(out/'tmp').mkdir()
state=dict(status='starting',cloud_calls=0,pid=os.getpid(),completed=[])
def progress(**kw):
 state.update(kw,updated=time.time());save(out/'progress.json',state);print('RETENTION_PROGRESS '+json.dumps(state),flush=True)
manifest=json.loads((ROOT/'package_manifest.json').read_text())
for path,h in manifest['sha256'].items():
 if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen package mismatch: '+path)
save(out/'package_manifest.json',manifest)
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',ASTRA_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'),PYTHONPATH=':'.join([str(ROOT/p)for p in ['scripts/agentic','outputs/learning_dependencies']]+[os.environ.get('PYTHONPATH','')]))
child=None
try:
 subprocess.run([sys.executable,'-c','import isaaclab,toml,torch; print("CLOUD_CHILD_IMPORT_OK", flush=True)'],env=env,check=True)
 import torch,importlib.metadata
 save(out/'environment.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,isaaclab=getattr(__import__('isaaclab'),'__version__','unknown')))
 def run(name,command,result):
  global child
  save(out/'logs'/(name+'_command.json'),command)
  with (out/'logs'/(name+'.log')).open('w')as log:
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start=time.time()
  while child.poll()is None:
   progress(status='running',current=name,child_pid=child.pid,elapsed_s=time.time()-start)
   if time.time()-start>1800:raise TimeoutError(name)
   time.sleep(10)
  if child.returncode or not result.exists():
   print((out/'logs'/(name+'.log')).read_text()[-12000:],flush=True)
   raise RuntimeError(name+' failed')
  progress(completed=state['completed']+[name],child_pid=None)
 prefix=out/'prefix'
 run('prefix',[sys.executable,str(ROOT/'scripts/astra_retention_v1/collect_prefix.py'),'--output',str(prefix),'--seed','104601','--split','audit','--cases','12','--headless','--device','cuda:0']+KIT,prefix/'report.json')
 for rate in (1.,.25,0.):
  dest=out/'branches/data'/('rate'+str(rate))
  run('rate'+str(rate),[sys.executable,str(ROOT/'scripts/astra_retention_v1/branch.py'),'--source',str(prefix),'--plan',str(prefix/'sensor_branch_plan.json'),'--output',str(dest),'--rate',str(rate),'--headless','--device','cuda:0']+KIT,dest/'report.json')
 run('audit',[sys.executable,str(ROOT/'scripts/astra_retention_v1/analyze_branches.py'),'--output',str(out/'branches')],out/'branches/summary.json')
 summary=json.loads((out/'branches/summary.json').read_text())
 result=dict(status='completed',purpose='Cloud compatibility trial only; not training or Astra evaluation',seed=104601,cloud_calls=0,matched_states=len(summary['paired_state_audits']),max_state_error=max((r['max_abs_error']for r in summary['paired_state_audits']),default=None),rates=summary['rates'])
 save(out/'result.json',result);progress(status='completed',current=None,child_pid=None)
 print('RETENTION_RESULT '+json.dumps(result),flush=True)
except BaseException as e:
 save(out/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=str(e));raise
finally:
 if child and child.poll()is None:
  import signal
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
