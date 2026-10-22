"""One training rollout and exact-prefix simulation check, no model calls."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save,sha
from prepare import environment_variables,KIT
import torch
out=ROOT/'outputs/astra_goal_maintenance_v2/smoke_20261022'
out.mkdir(parents=True,exist_ok=False)
for name in ('logs','tmp'):(out/name).mkdir()
script=Path(__file__).with_name('train.py');frozen={str(p):sha(p) for p in [script,script.with_name('model.py')]}
save(out/'protocol.json',dict(source_sha256=frozen,seed_train=125001,seed_eval=125002,nominal=6,scope='Integration only, not model performance',cloud_calls=0))
with open('/tmp/agent_inhand_paired_gpu_0.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name,args in [('train',['--seed','125001','--iterations','1']),('original',['--seed','125002','--mode','original']),('learned',['--seed','125002','--mode','learned','--source',str(out/'original'),'--checkpoint',str(out/'train/controller_final.pt')])]:
  assert all(sha(p)==h for p,h in frozen.items())
  save(out/'progress.json',dict(current=name,status='running',pid=os.getpid(),updated=time.time(),cloud_calls=0))
  cmd=[PYTHON,str(script),'--output',str(out/name),'--num-envs','6','--headless','--device','cuda:0']+args+KIT
  with (out/'logs'/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,env=environment_variables(out),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=650,check=True)
 b=torch.load(out/'original/branch.pt',map_location='cpu',weights_only=False);c=torch.load(out/'learned/branch.pt',map_location='cpu',weights_only=False)
 errors={k:float((b['physical'][k]-c['physical'][k]).abs().max()) for k in b['physical']}
 errors.update({k:float((b[k].float()-c[k].float()).abs().max()) for k in ['ih','ph','prop','active','dense']})
 assert max(errors.values())==0
 save(out/'replay_check.json',dict(status='completed',max_difference=max(errors.values()),errors=errors,cloud_calls=0))
 save(out/'progress.json',dict(status='completed',updated=time.time(),cloud_calls=0))
 print(json.dumps(dict(status='completed',max_difference=0,cloud_calls=0)),flush=True)
