"""Finite original-render diagnostics. The two repetitions are not new seeds."""
import os,sys,json,time,signal,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/latency_counterfactual'))
from run_original import gpu_available
out=ROOT/'outputs/recovery_controller_v3/variance_20261018'
out.mkdir(parents=True,exist_ok=False);(out/'tmp').mkdir()
lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert gpu_available(0)['available']
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',PAIRED_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
env['PYTHONPATH']=':'.join(str(ROOT/p) for p in ['scripts/agentic','source/BrainCo_DexHand','outputs/video_dependencies'])
python='/home/yin/Grasping/Grasppose/env_isaaclab/bin/python'
state={'pid':os.getpid(),'status':'starting'};child=None

def save(**kwargs):
 state.update(kwargs,updated=time.time());tmp=out/'progress.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'progress.json')

def interrupt(*args):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
try:
 for name,filt,playback in [('original_r1','none',None),('original_r2','none',None),('action_playback','none','original_r1'),('filtered_r1','gaussian3',None),('filtered_r2','gaussian3',None)]:
  cmd=[python,str(ROOT/'scripts/recovery_controller_v3/diagnose_variance.py'),'--headless','--device','cuda:0','--output',str(out/name),'--seed','82002','--num-envs','12','--mode','continue','--delay','0','--request-cache',str(ROOT/'outputs/recovery_controller_v2/pilot_20261018/cache_82002/request_state.pt'),'--rgb-filter',filt,'--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']
  if playback:cmd+=['--replay-actions',str(out/playback/'trace.pt')]
  (out/(name+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (out/(name+'.log')).open('w') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.monotonic();save(status='running',arm=name,child_pid=child.pid)
   while child.poll() is None:
    if time.monotonic()-start>900:raise TimeoutError(name)
    time.sleep(5)
   if child.returncode:raise RuntimeError(f'{name}: {child.returncode}')
  save(last_completed=name,child_pid=None)
 save(status='completed')
except BaseException as exc:
 save(status='failed',error=str(exc));raise
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 lock.close()
