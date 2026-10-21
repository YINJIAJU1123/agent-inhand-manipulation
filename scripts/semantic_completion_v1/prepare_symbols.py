"""Finite local data repair. No network clients and no formal test data."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from common import ROOT,PYTHON,save,sha
from prepare import environment_variables,KIT

def main():
    p=argparse.ArgumentParser();p.add_argument('--learning',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    src=Path(a.learning).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ['data','logs','tmp']:(out/name).mkdir()
    state=dict(pid=os.getpid(),status='collecting',completed_shards=0);child=None
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    schedule=[('train',s) for s in [96111,96112,96114,96115]]+[('val',s) for s in [96211,96212]]
    frozen={str(ROOT/'scripts/semantic_completion_v1'/x):sha(ROOT/'scripts/semantic_completion_v1'/x) for x in ['collect_symbols.py','common.py','prepare_symbols.py']}
    save(out/'protocol.json',dict(schedule=schedule,views_per_state=12,state_stride=60,max_steps=480,source_sha256=frozen,
         source_learning=str(src),api_calls=0,test_seeds_used=[],reason='Unknown-class collapse and insufficient six-symbol coverage.'))
    env=environment_variables(out);lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        progress()
        for split,seed in schedule:
            for path,digest in frozen.items():
                if sha(path)!=digest:raise RuntimeError('Collection source changed')
            name=f'{split}_{seed}';dest=out/'data'/name
            cmd=[PYTHON,str(ROOT/'scripts/semantic_completion_v1/collect_symbols.py'),'--headless','--device','cuda:0',
                 '--source',str(src/'data'/name),'--output',str(dest),'--seed',str(seed),'--split',split]+KIT
            with (out/'logs'/(name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
                while child.poll() is None:
                    progress(current=name,child_pid=child.pid,wall_s=time.monotonic()-start)
                    if (dest/'failure.json').exists():raise RuntimeError(name+' failed')
                    if time.monotonic()-start>2400:raise TimeoutError(name)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(name+' exited '+str(child.returncode))
            counts=json.loads((dest/'manifest.json').read_text())['label_counts']
            progress(completed_shards=state['completed_shards']+1,last_label_counts=counts)
        progress(status='completed',current=None,child_pid=None)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        lock.close()

if __name__=='__main__':main()
