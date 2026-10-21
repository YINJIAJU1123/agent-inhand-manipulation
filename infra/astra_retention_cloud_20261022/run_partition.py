"""Three finite GPU partitions, independent seed splits, one shared NAS run."""
from pathlib import Path
import base64,hashlib,io,json,os,signal,subprocess,sys,tarfile,time,traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import save
from prepare import KIT
rank=int(os.environ['RETENTION_PARTITION']);assert rank in (0,1,2)
out=Path(os.environ['RETENTION_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
worker=out/f'worker{rank}';worker.mkdir(exist_ok=False)
(worker/'logs').mkdir();(worker/'tmp').mkdir()
schedule=[('train',s)for s in range(104211,104219)]+[('calibration',104311),('calibration',104312),('audit',104411),('audit',104412)]
state=dict(status='starting',rank=rank,pid=os.getpid(),completed=[],cloud_calls=0)
def progress(**kw):
 state.update(kw,updated=time.time());save(worker/'progress.json',state);print('RETENTION_PROGRESS '+json.dumps(state),flush=True)
manifest=json.loads((ROOT/'package_manifest.json').read_text())
for path,digest in manifest['sha256'].items():
 if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen package changed: '+path)
gate=json.loads((ROOT/'launch_gate.json').read_text())
assert gate['cloud_compatibility_passed'] and gate['full_local_diagnostic_passed']
assert gate['cloud_calls']==0
save(worker/'protocol.json',dict(schedule=schedule,rank=rank,assigned=[r for i,r in enumerate(schedule)if i%3==rank],nominal_states=144,source_sha256=manifest['sha256'],launch_gate=gate,cloud_calls=0,interpretation='Independent supplied-goal action selection data. No Astra task-success claim or automatic promotion.'))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',ASTRA_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(worker/'tmp'),PYTHONPATH=':'.join([str(ROOT/p)for p in ['scripts/agentic','outputs/learning_dependencies']]+[os.environ.get('PYTHONPATH','')]))
child=None
try:
 subprocess.run([sys.executable,'-c','import isaaclab,toml,torch; print("CLOUD_CHILD_IMPORT_OK",flush=True)'],env=env,check=True)
 def run(name,command,result):
  global child
  save(worker/'logs'/(name+'_command.json'),command)
  with (worker/'logs'/(name+'.log')).open('w')as log:
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  started=time.time()
  while child.poll() is None:
   heartbeat=result.parent/'progress.json'
   measured=json.loads(heartbeat.read_text())if heartbeat.exists()else None
   progress(status='training'if name=='train'else'collecting',current=name,child_pid=child.pid,elapsed_s=time.time()-started,child_progress=measured)
   if heartbeat.exists()and time.time()-heartbeat.stat().st_mtime>300:raise TimeoutError(name+' stopped reporting')
   if time.time()-started>1800:raise TimeoutError(name)
   time.sleep(10)
  if child.returncode or not result.exists():
   print((worker/'logs'/(name+'.log')).read_text()[-12000:],flush=True)
   raise RuntimeError(name+' failed')
  progress(completed=state['completed']+[name],child_pid=None)
 for i,(split,seed)in enumerate(schedule):
  if i%3!=rank:continue
  folder=out/'data'/f'{split}_{seed}';folder.mkdir(parents=True,exist_ok=False);prefix=folder/'prefix'
  run(f'{split}_{seed}_prefix',[sys.executable,str(ROOT/'scripts/astra_retention_v1/collect_prefix.py'),'--output',str(prefix),'--seed',str(seed),'--split',split,'--cases','12','--headless','--device','cuda:0']+KIT,prefix/'report.json')
  for rate in (1.,.25,0.):
   dest=folder/('rate'+str(rate))
   run(f'{split}_{seed}_rate{rate}',[sys.executable,str(ROOT/'scripts/astra_retention_v1/branch.py'),'--source',str(prefix),'--plan',str(prefix/'sensor_branch_plan.json'),'--output',str(dest),'--rate',str(rate),'--headless','--device','cuda:0']+KIT,dest/'report.json')
 if rank==0:
  start=time.time()
  required=[out/'data'/f'{s}_{seed}'/('rate'+str(r))/'report.json'for s,seed in schedule for r in (1.,.25,0.)]
  while not all(p.exists()for p in required):
   if any((out/f'worker{i}/failure.json').exists()for i in (1,2)):raise RuntimeError('A collection partition failed')
   progress(status='waiting_for_partitions',complete_reports=sum(p.exists()for p in required),expected_reports=len(required))
   if time.time()-start>5400:raise TimeoutError('Other partitions exceeded 90-minute wait')
   time.sleep(15)
  run('train',[sys.executable,str(ROOT/'scripts/astra_retention_v1/train_value.py'),'--data',str(out/'data'),'--output',str(out/'train')],out/'train/result.json')
  result=json.loads((out/'train/result.json').read_text());print('RETENTION_RESULT '+json.dumps(result),flush=True)
  if result['offline_gate_passed']:
   probes=[]
   for mode in ('none','hold','learned'):
    dest=out/'probe'/mode
    command=[sys.executable,str(ROOT/'scripts/astra_retention_online_v1/evaluate.py'),'--output',str(dest),'--seed','104511','--arm','verify','--initialization','collision_clearance','--retention-mode',mode,'--evidence-model',str(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'),'--max-steps','1800','--budget','6','--offline-smoke','--offline-retention-probe','--headless','--device','cuda:0']+KIT
    if mode=='learned':command+=['--retention-head',str(out/'train/head.pt')]
    run('probe_'+mode,command,dest/'report.json')
    events=json.loads((dest/'retention_events.json').read_text())
    probes.append(dict(mode=mode,retention_choices=sum(e['event']=='retention_choice'and e['rate']is not None for e in events),releases=sum(e['event']=='retention_release'for e in events),report=str(dest/'report.json')))
   probe_result=dict(status='completed',seed=104511,cloud_calls=0,modes=probes,note='Fixed-color simulated-delay integration, not Astra task performance')
   save(out/'probe_result.json',probe_result);print('RETENTION_PROBE_RESULT '+json.dumps(probe_result),flush=True)
  else:
   save(out/'probe_result.json',dict(status='not_started',reason='Local held-out gate failed',cloud_calls=0))
  # A bounded model/report bundle in the authenticated DLC log permits exact
  # retrieval without copying cloud credentials into the training container.
  stream=io.BytesIO()
  with tarfile.open(fileobj=stream,mode='w:gz')as archive:
   for name in ('head.pt','result.json','protocol.json','history.json'):
    archive.add(out/'train'/name,arcname=name)
  blob=stream.getvalue();assert len(blob)<2000000
  save(out/'artifact.json',dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
  print('RETENTION_ARTIFACT_SHA256 '+hashlib.sha256(blob).hexdigest(),flush=True)
  encoded=base64.b64encode(blob).decode()
  print('RETENTION_ARTIFACT_BEGIN',flush=True)
  for offset in range(0,len(encoded),64):print(encoded[offset:offset+64],flush=True)
  print('RETENTION_ARTIFACT_END',flush=True)
 progress(status='completed',current=None,child_pid=None)
except BaseException as e:
 save(worker/'failure.json',dict(error=traceback.format_exc()));progress(status='failed',error=str(e));raise
finally:
 if child and child.poll()is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
