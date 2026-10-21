"""One finite local run: corrected temporal labels, symbol coverage, training, smoke."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from common import ROOT,PYTHON,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--account',required=True);p.add_argument('--pose-initial',required=True);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);state=dict(pid=os.getpid(),status='starting',api_calls=0);child=None
    def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    frozen={str(x):sha(x) for x in (ROOT/'scripts/semantic_completion_v1').glob('*.py')}
    save(out/'protocol.json',dict(source_sha256=frozen,api_calls=0,formal_test_seeds_used=[],
        correction='Existing prepared-label rule: expected ink area uses (7.5/9)^2; ambiguous partial views excluded from semantic classification.',
        supersedes=['learning_20261020','balanced_train_20261021','symbol_repair_20261021','repair_pipeline_20261021'],
        pose_initial_sha256=sha(a.pose_initial),note='Prior exploratory artifacts retained. Cloud evaluation requires separate explicit transfer approval and validation gates.'))
    commands=[
      ('corrected_collection',[PYTHON,str(ROOT/'scripts/semantic_completion_v1/prepare.py'),'--output',str(out/'base')]),
      ('symbol_coverage',[PYTHON,str(ROOT/'scripts/semantic_completion_v1/prepare_symbols.py'),'--learning',str(out/'base'),'--output',str(out/'symbols')]),
      ('learning_and_offline_validation',[PYTHON,str(ROOT/'scripts/semantic_completion_v1/finish_symbols.py'),'--source-learning',str(out/'base'),
        '--symbols',str(out/'symbols'),'--initial',str(Path(a.pose_initial).resolve()),'--account',str(Path(a.account).resolve()),'--output',str(out/'final')])]
    try:
        for stage,cmd in commands:
            for path,digest in frozen.items():
                if sha(path)!=digest:raise RuntimeError('Repair source changed after freeze')
            with (out/(stage+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
                while child.poll() is None:
                    progress(status=stage,child_pid=child.pid,wall_s=time.monotonic()-start)
                    if time.monotonic()-start>7200:raise TimeoutError(stage)
                    time.sleep(5)
                if child.returncode:raise RuntimeError(stage+' exited '+str(child.returncode))
        result=json.loads((out/'final/progress.json').read_text());progress(status=result['status'],child_pid=None,result=result)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)

if __name__=='__main__':main()
