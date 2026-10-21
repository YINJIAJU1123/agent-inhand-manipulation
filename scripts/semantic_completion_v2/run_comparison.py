"""Run the finite two-seed paired shadow diagnostic without cloud services."""
import argparse,fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
from assets import ROOT,PYTHON,save,sha
from collect_run import env_variables,KIT

def main():
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--cases',type=int,default=6);a=p.parse_args()
    train=Path(a.training).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ['logs','tmp']:(out/name).mkdir()
    recal=json.loads((train/'recalibration_only.json').read_text())
    specs=dict(frame_gate=dict(checkpoint=str(Path(a.base).resolve())),
        recalibration_only=dict(checkpoint=str(Path(a.base).resolve()),threshold=recal['threshold']),
        continuation_control=dict(checkpoint=str(train/'continuation_control/head.pt')),
        first_stop=dict(checkpoint=str(train/'first_stop/head.pt')))
    save(out/'models.json',specs);frozen={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    (out/'source').mkdir()
    for file in frozen:shutil.copy2(file,out/'source'/Path(file).name)
    save(out/'protocol.json',dict(seeds=[96311,96312],cases_per_seed=a.cases,source_sha256=frozen,models=specs,cloud_calls=0))
    state=dict(pid=os.getpid(),status='running',completed=0,cloud_calls=0);child=None
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        progress()
        for seed in [96311,96312]:
            for p,h in frozen.items():
                if sha(p)!=h:raise RuntimeError('Frozen comparison code changed')
            cmd=[PYTHON,str(ROOT/'scripts/semantic_completion_v2/compare_shadow.py'),'--models',str(out/'models.json'),
                '--output',str(out/str(seed)),'--seed',str(seed),'--cases',str(a.cases),'--headless','--device','cuda:0']+KIT
            with (out/'logs'/f'{seed}.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env_variables(out),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                started=time.monotonic()
                while child.poll() is None:
                    progress(seed=seed,child_pid=child.pid,wall_s=time.monotonic()-started)
                    if time.monotonic()-started>1800:raise TimeoutError(str(seed))
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Simulation exited '+str(child.returncode))
            progress(completed=state['completed']+1)
        reports=[json.loads((out/str(s)/'report.json').read_text()) for s in [96311,96312]]
        summary={name:{k:sum(r['heads'][name][k] for r in reports) for k in ['proposals','correct','false_first_stops']} for name in specs}
        save(out/'summary.json',dict(episodes=2*a.cases,heads=summary,scope='Prospective paired known-goal shadow diagnostic; no Astra task-success claim.',cloud_calls=0))
        progress(status='completed',child_pid=None,summary=summary)
    except BaseException as e:progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        lock.close()
if __name__=='__main__':main()
