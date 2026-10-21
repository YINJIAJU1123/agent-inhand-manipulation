"""Finite dependency runner: completed data jobs -> two matched learning jobs.

This is not a scheduled monitor. It runs once with a three-hour deadline, does
not restart remote work, and never makes Astra requests or deploys a model.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'infra/astra_symbol_trajectory_20261022'
BCTRL=['/home/yin/.local/bin/bctrl','--region','cn-wulanchabu','dlc']

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs=2,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting_for_data',pid=os.getpid(),data_jobs=a.jobs,submitted=[],cloud_calls=0)
    sources=[HERE/'train_invariant.yaml',HERE/'train_verified.yaml']+list((HERE/'code_train').rglob('*'))
    frozen={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources if f.is_file()}
    save(a.output/'protocol.json',dict(data_jobs=a.jobs,source_sha256=frozen,variants=['invariant','verified'],
        wait_seconds=10800,cloud_calls=0,scope='Only two fixed training jobs, followed by recorded-video diagnostics. No promotion or real Astra requests.'))
    start=time.monotonic()
    def progress(**changes):
        state.update(changes,updated=time.time());save(a.output/'progress.json',state)
    try:
        while True:
            if time.monotonic()-start>10800:raise TimeoutError('Observation deadline; data jobs are not inferred failed')
            statuses={};errors={}
            for job in a.jobs:
                try:
                    r=subprocess.run(BCTRL+['show',job,'--json'],capture_output=True,text=True,check=True,timeout=40)
                    statuses[job]=json.loads(r.stdout)['Status']
                except (subprocess.SubprocessError,ValueError,KeyError) as e:errors[job]=type(e).__name__
            progress(status='waiting_for_data',job_status=statuses,observation_errors=errors)
            if any(status in ('Failed','Stopped','Cancelled','Canceled') for status in statuses.values()):
                progress(status='not_started',reason='A data job ended without success');return
            if len(statuses)==2 and all(s=='Succeeded' for s in statuses.values()):break
            time.sleep(45)
        for name,digest in frozen.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen training source changed: '+name)
        for variant in ('invariant','verified'):
            progress(status='submitting',variant=variant)
            with (a.output/(variant+'_submit.log')).open('w') as log:
                r=subprocess.run(BCTRL+['submit',str(HERE/f'train_{variant}.yaml'),'--yes'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            text=(a.output/(variant+'_submit.log')).read_text()
            ids=re.findall(r'<(dlc[a-z0-9]+)>',text)
            if r.returncode or len(set(ids))!=1:raise RuntimeError('Submission needs inspection; it will not be retried automatically: '+variant)
            job=ids[0];state['submitted'].append(dict(variant=variant,job_id=job));progress(status='submitted')
            output=a.output/(variant+'_shadow')
            model=ROOT/('scripts/astra_symbol_verified_v1/model.py' if variant=='verified' else 'scripts/astra_symbol_invariant_v1/model.py')
            command=[sys.executable,str(ROOT/'scripts/astra_symbol_regions_audit_v1/run_after_training.py'),
                '--job-id',job,'--cache',str(ROOT/'outputs/astra_symbol_regions_20261022/request_cache'),
                '--output',str(output),'--model-source',str(model)]
            with (a.output/(variant+'_shadow.log')).open('w') as log:
                watcher=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            state['submitted'][-1]['watcher_pid']=watcher.pid
            progress(status='submitted')
        progress(status='completed')
    except BaseException:
        progress(status='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
