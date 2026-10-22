"""Independent calibration and audit of two already frozen completion heads."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/workspace/code');sys.path[:0]=[str(ROOT/'scripts'/s) for s in ('semantic_completion_v1','astra_transition_completion_v1','semantic_completion_v2','semantic_completion_v3')]
from common import save,sha
from prepare import KIT
from train_dense import read,assess,decision_data
from sequential import score,THRESHOLD_GRID
from dense_model import load_model
out=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_transition_completion_20261022/recalibration_r2');out.mkdir(parents=True,exist_ok=False)
for name in ('tmp','logs','data'):(out/name).mkdir()
old=ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
new=Path('/mnt/data_nas/jiaju/agent-inhand-manipulation/astra_transition_completion_20261022/adaptation_r2/training/head.pt')
assert sha(new)=='d64787da5d9d19478937c72441d7b5e8fc071c3d662e2b9e6a5d130add0e27c8'
save(out/'protocol.json',dict(calibration_seed=121301,audit_seed=121302,cases=24,max_steps=1200,
 models={'original':sha(old),'adapted':sha(new)},training=False,cloud_calls=0,
 selection='Frozen weights. Calibrate each head independently: >=8 first stops, >=.9 precision, maximize correct then precision then prefer lower threshold. Original fixed .7 also reported.',
 gate='Adapted audit >=8 first stops, precision>=.9, reached_recall>=.3, no fewer correct and fewer false first stops than recalibrated original. No automatic deployment.',
 motivation='Earlier calibration had only eight reached segments. Its failed gate remains recorded. This new pair increases the cases per seed; no old threshold is silently relaxed.'))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',TMPDIR=str(out/'tmp'),PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),str(ROOT/'outputs/learning_dependencies'),os.environ.get('PYTHONPATH','')]))
try:
 for split,seed in [('calibration',121301),('audit',121302)]:
  dest=out/'data'/split
  cmd=[sys.executable,str(ROOT/'scripts/astra_transition_completion_v1/collect.py'),'--output',str(dest),'--seed',str(seed),'--split',split,'--cases','24','--max-steps','1200','--headless','--device','cuda:0']+KIT
  with (out/'logs'/(split+'.log')).open('w') as f:
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  start=time.monotonic()
  while p.poll() is None:
   q=dest/'progress.json';r=dict(status='collecting',split=split,seed=seed,child=json.loads(q.read_text()) if q.exists() else None)
   save(out/'progress.json',r);print('RECAL_PROGRESS '+json.dumps(r),flush=True)
   if time.monotonic()-start>600:p.kill();raise TimeoutError(split)
   time.sleep(15)
  if p.returncode:raise RuntimeError((out/'logs'/(split+'.log')).read_text()[-1800:])
 models={name:load_model(path,'cuda:0') for name,path in [('original',old),('adapted',new)]}
 cal=read([out/'data/calibration/data.pt']);calids=[i for i,r in enumerate(cal['rows']) if r.get('phase')!='initial']
 thresholds={};curves={}
 for name,(m,pack) in models.items():
  _,prob=assess(m,cal,'cuda:0');curve=[dict(threshold=t,**score(prob[calids],decision_data(cal),t)) for t in THRESHOLD_GRID]
  eligible=[r for r in curve if r['accepted']>=8 and r['precision']>=.9]
  selected=max(eligible,key=lambda r:(r['correct'],r['precision'],-r['threshold'])) if eligible else None
  thresholds[name]=selected['threshold'] if selected else 1.;curves[name]=curve
 save(out/'calibration.json',dict(thresholds=thresholds,curves=curves))
 audit=read([out/'data/audit/data.pt']);auditids=[i for i,r in enumerate(audit['rows']) if r.get('phase')!='initial'];scores={}
 for name,(m,pack) in models.items():
  metrics,prob=assess(m,audit,'cuda:0');scores[name]=dict(metrics=metrics,**score(prob[auditids],decision_data(audit),thresholds[name]))
  if name=='original':scores['original_fixed']=score(prob[auditids],decision_data(audit),pack['threshold'])
 a,b=scores['adapted'],scores['original'];passed=a['accepted']>=8 and a['precision']>=.9 and (a['reached_recall'] or 0)>=.3 and a['correct']>=b['correct'] and a['false_first_stops']<b['false_first_stops']
 result=dict(status='completed',thresholds=thresholds,audit=scores,offline_gate_passed=passed,cloud_calls=0,deployed=False)
 save(out/'result.json',result)
 compact=dict(result);compact['audit']={k:{x:y for x,y in v.items() if x not in ('details','metrics')} for k,v in scores.items()}
 print('RECAL_RESULT '+json.dumps(compact),flush=True);save(out/'progress.json',dict(status='completed',offline_gate_passed=passed))
except BaseException:
 save(out/'failure.json',dict(error=traceback.format_exc()));print(traceback.format_exc(),flush=True);raise
