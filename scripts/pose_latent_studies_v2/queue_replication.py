"""Finite experiment queue. Only starts on an actually idle GPU, no preemption."""
import argparse,fcntl,json,os,subprocess,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path

def save(path,d):
 p=path.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args();a.root=a.root.resolve()
 lock=(a.root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (a.root/'queue_status.json').exists():raise FileExistsError('Inspect existing queue and outputs before a new launch')
 state=dict(pid=os.getpid(),status='waiting_for_idle_gpu',completed=[],created_at=datetime.now(timezone.utc).isoformat())
 save(a.root/'queue_status.json',state)
 for condition in ['astra_full','finetuned_vlm_bare','rule_vlm_bare']:
  state.update(condition=condition,status='waiting_for_idle_gpu',child_pid=None,gpu=None)
  while True:
   # Queries observe all cards but never modify other processes.
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
   readings=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
   state.update(observed_at=datetime.now(timezone.utc).isoformat(),gpu_observations=readings)
   chosen=None
   for gpu,mem,util in readings:
    if mem>=1000 or util>10:continue
    candidate=Path(f'/tmp/agentic_rotation_experiment_gpu_{gpu}.lock').open('a')
    try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:candidate.close();continue
    chosen=(gpu,candidate);break
   save(a.root/'queue_status.json',state)
   if chosen:break
   time.sleep(30)
  gpu,gpu_lock=chosen
  # Health must pass before any trial. The independent service keeps its own cap.
  try:
   endpoints=[(18794,None),(18795,None),(18811,a.runtime/'outputs/live_services_20261029/bridge.key')]
   for port,key in endpoints:
    headers={} if key is None else {'Authorization':'Bearer '+key.read_text().strip()}
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/health',headers=headers),timeout=10) as r:
     if r.status!=200:raise RuntimeError('Unhealthy endpoint')
   env=dict(os.environ,RUNTIME_ROOT=str(a.runtime),STUDY_ROOT=str(a.root),CONDITION=condition,CUDA_VISIBLE_DEVICES=str(gpu))
   with (a.root/(condition+'.queue.log')).open('x') as log:
    child=subprocess.Popen(['bash',str(a.root/'run_replication_condition.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='running',gpu=gpu,child_pid=child.pid);save(a.root/'queue_status.json',state)
    code=child.wait()
   if code:raise RuntimeError(condition+' exit '+str(code))
   state['completed'].append(condition);save(a.root/'queue_status.json',state)
  except BaseException as e:
   state.update(status='failed',error=str(e));save(a.root/'queue_status.json',state);raise
  finally:gpu_lock.close()
 state.update(status='completed',condition=None,child_pid=None);save(a.root/'queue_status.json',state)
if __name__=='__main__':main()
