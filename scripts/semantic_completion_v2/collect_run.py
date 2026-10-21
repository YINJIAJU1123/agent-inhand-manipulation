"""Bounded local data collection with balanced fixed-goal assignments."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from assets import ROOT,PYTHON,save,sha

def env_variables(out):
    e=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',ASTRA_RENDER_GPU='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',TMPDIR=str(out/'tmp'))
    e['PYTHONPATH']=':'.join(str(ROOT/x) for x in ['outputs/vlm_dependencies','outputs/learning_dependencies','outputs/video_dependencies','scripts/agentic']);return e
KIT=['--kit_args','--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for n in ['data','logs','tmp']:(out/n).mkdir()
    state=dict(pid=os.getpid(),status='collecting',completed=0);child=None
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    frozen={str(ROOT/'scripts/semantic_completion_v2'/x):sha(ROOT/'scripts/semantic_completion_v2'/x) for x in ['collect.py','assets.py','collect_run.py']}
    schedule=[('train',s) for s in range(96121,96129)]+[('val',96311),('val',96312)]
    save(out/'protocol.json',dict(schedule=schedule,source_sha256=frozen,steps=900,formal_seeds_untouched=[86101,86102],
         purpose='Balanced fixed-goal trajectories, including goal approach and drift; 96311 for calibration, 96312 for audit only. Two fresh development evaluation seeds.',cloud_calls=0))
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=env_variables(out)
    try:
        progress()
        for split,seed in schedule:
            for f,h in frozen.items():
                if sha(f)!=h:raise RuntimeError('Frozen collection code changed')
            name=f'{split}_{seed}';cmd=[PYTHON,str(ROOT/'scripts/semantic_completion_v2/collect.py'),'--output',str(out/'data'/name),'--seed',str(seed),'--split',split,'--headless','--device','cuda:0']+KIT
            with (out/'logs'/(name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
                while child.poll() is None:
                    progress(current=name,child_pid=child.pid,wall_s=time.monotonic()-start)
                    if (out/'data'/name/'failure.json').exists():raise RuntimeError(name+' failed')
                    if time.monotonic()-start>1800:raise TimeoutError(name)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(name+' exited '+str(child.returncode))
            manifest=json.loads((out/'data'/name/'manifest.json').read_text());progress(completed=state['completed']+1,last_counts=manifest['ready_counts'])
        progress(status='completed',child_pid=None,current=None)
    except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        lock.close()
if __name__=='__main__':main()
