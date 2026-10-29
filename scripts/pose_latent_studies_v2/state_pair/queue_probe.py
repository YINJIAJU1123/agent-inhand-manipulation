"""Run exactly one pair of state reconstruction probes on an idle GPU."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
lock=(a.root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (a.root/'queue_status.json').exists():raise FileExistsError('Inspect previous probe before starting again')
state={'pid':os.getpid(),'status':'waiting_for_idle_gpu','actual_model_calls':0}
def save(**updates):
 state.update(updates,observed_at=datetime.now(timezone.utc).isoformat())
 tmp=a.root/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.root/'queue_status.json')
try:
 while True:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  readings=[list(map(int,x.split(','))) for x in raw.strip().splitlines()];chosen=None
  for gpu,mem,util in readings:
   if mem>=1000 or util>10:continue
   candidate=Path(f'/tmp/agentic_rotation_experiment_gpu_{gpu}.lock').open('a')
   try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:candidate.close();continue
   chosen=(gpu,candidate);break
  save(gpu_observations=readings)
  if chosen:break
  time.sleep(30)
 gpu,gpu_lock=chosen
 with (a.root/'run.log').open('x') as log:
  child=subprocess.Popen(['bash',str(a.root/'run_probe.sh')],env=dict(os.environ,RUNTIME_ROOT=str(a.runtime),STUDY_ROOT=str(a.root),CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
  save(status='running',gpu=gpu,child_pid=child.pid);code=child.wait()
 if code:raise RuntimeError('Reconstruction probe exit '+str(code))
 result=json.loads((a.root/'repeat/result.json').read_text())
 if not result['paired_state_admitted']:raise RuntimeError('State-pair admission failed')
 save(status='completed',child_pid=None,paired_state_admitted=True)
except BaseException as e:
 save(status='failed',error=str(e));raise
