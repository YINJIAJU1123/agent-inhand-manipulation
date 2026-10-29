"""Local development probe with an exclusive lease against our remote queues.

The lease uses only our experiment lock files. It never stops another user's
GPU work. Holding it prevents our formal inference from overlapping the local
probe, which shares the GPU hosting our VLM services.
"""
import argparse,json,os,shlex,signal,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
if (a.study/'launch.json').exists():raise FileExistsError('Inspect existing local probe before launching')
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<10000:raise RuntimeError('At least 10GB free GPU memory required')
pids={int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()}
if not pids.issubset({95000,95138}):raise RuntimeError('Unexpected GPU compute process; do not contend')
lease_code='''import fcntl,sys
from pathlib import Path
locks=[]
for gpu in range(8):
 f=Path(f'/tmp/agentic_rotation_experiment_gpu_{gpu}.lock').open('a')
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 locks.append(f)
print('LEASE_ACQUIRED',flush=True)
sys.stdin.buffer.read()
'''
lease=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','brainco@10.3.100.20','python3 -u -c '+shlex.quote(lease_code)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
child=None
try:
 if lease.stdout.readline().strip()!='LEASE_ACQUIRED':raise RuntimeError('Cannot acquire remote experiment lease: '+lease.stderr.read()[:300])
 env=dict(os.environ,RUNTIME_ROOT=str(a.runtime),STUDY_ROOT=str(a.study),CUDA_VISIBLE_DEVICES='0',RELOCATION_MANIFEST=str(a.runtime/'relocation.json'))
 with (a.study/'supervisor.log').open('x') as log:
  child=subprocess.Popen(['bash',str(a.study/'run_probe.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launch={'pid':os.getpid(),'child_pid':child.pid,'lease_pid':lease.pid,'initial_gpu_free_mib':free,'scope':'Local development only; no high-level model requests'}
  (a.study/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
  while True:
   try:code=child.wait(timeout=5);break
   except subprocess.TimeoutExpired:
    if lease.poll() is not None:
     os.killpg(child.pid,signal.SIGTERM);raise RuntimeError('Remote experiment lease lost; stopped local workload')
 if code:raise RuntimeError('Local probe exited '+str(code))
 (a.study/'supervisor_result.json').write_text(json.dumps({'status':'completed','exit_code':code})+'\n')
except BaseException as e:
 if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 (a.study/'supervisor_result.json').write_text(json.dumps({'status':'failed','error':str(e)})+'\n');raise
finally:
 if lease.stdin:lease.stdin.close()
 try:lease.wait(timeout=10)
 except subprocess.TimeoutExpired:lease.terminate()
