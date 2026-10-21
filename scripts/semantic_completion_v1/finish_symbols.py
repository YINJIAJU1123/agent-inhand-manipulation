"""Finish a finite local repair: wait for collection, train, then offline checks."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from common import ROOT,PYTHON,save,sha
from prepare import environment_variables

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-learning',required=True);p.add_argument('--symbols',required=True)
    p.add_argument('--initial',required=True);p.add_argument('--account',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    src=Path(a.source_learning).resolve();symbols=Path(a.symbols).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    learning=out/'learning';learning.mkdir();(learning/'data').symlink_to(src/'data',target_is_directory=True)
    (out/'tmp').mkdir();state=dict(pid=os.getpid(),status='waiting_symbols',api_calls=0);child=None;lock=None
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    env=environment_variables(out)
    frozen={str(ROOT/'scripts/semantic_completion_v1'/x):sha(ROOT/'scripts/semantic_completion_v1'/x) for x in ['train_symbols.py','model.py','common.py','finish_symbols.py']}
    save(out/'protocol.json',dict(source_learning=str(src),symbol_data=str(symbols),initial_checkpoint=str(Path(a.initial).resolve()),source_sha256=frozen,cloud_calls_allowed=False))
    def run(cmd,logfile,status,timeout):
        nonlocal child
        for path,digest in frozen.items():
            if sha(path)!=digest:raise RuntimeError('Local repair source changed')
        with (out/logfile).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
            while child.poll() is None:
                progress(status=status,child_pid=child.pid,wall_s=time.monotonic()-start)
                if time.monotonic()-start>timeout:raise TimeoutError(status)
                time.sleep(5)
            if child.returncode:raise RuntimeError(status+' failed with exit '+str(child.returncode))
    try:
        progress();deadline=time.monotonic()+2400
        while True:
            data=json.loads((symbols/'progress.json').read_text())
            if data['status']=='completed':break
            if data['status'] in ['failed','interrupted']:raise RuntimeError('Symbol collection failed')
            if time.monotonic()>deadline:raise TimeoutError('Symbol collection wait')
            progress(completed_shards=data.get('completed_shards',0),collection_item=data.get('current'));time.sleep(10)
        lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        save(learning/'progress.json',dict(status='training'))
        run([PYTHON,str(ROOT/'scripts/semantic_completion_v1/train_symbols.py'),'--learning',str(src),'--symbols',str(symbols),
             '--initial',str(Path(a.initial).resolve()),'--output',str(learning/'train'),'--epochs','80'],'train.log','training',1800)
        result=json.loads((learning/'train/result.json').read_text());save(learning/'progress.json',dict(status='completed',training_result=result))
        lock.close();lock=None
        run([PYTHON,str(ROOT/'scripts/semantic_completion_v1/run.py'),'--learning',str(learning),'--output',str(out/'offline'),
             '--account',str(Path(a.account).resolve()),'--offline-only'],'offline.log','offline_validation',1200)
        progress(status='ready_for_cloud_approval' if result['cloud_eligible'] else 'needs_improvement',child_pid=None,training_result=result)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        if lock:lock.close()

if __name__=='__main__':main()
