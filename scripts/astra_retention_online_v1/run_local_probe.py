"""Finite local probe after a particular training job; never calls Astra."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON,save
from prepare import environment_variables,KIT
p=argparse.ArgumentParser()
p.add_argument('--training',required=True)
p.add_argument('--training-pid',type=int,required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
training=Path(a.training).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'logs').mkdir();child=None;lock=None
state=dict(status='waiting_for_training',pid=os.getpid(),dependency_pid=a.training_pid,completed=[])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')}
save(out/'protocol.json',dict(seed=104501,modes=['none','hold','learned'],cases_per_mode=6,cloud_calls=0,training=str(training),source_sha256=frozen,scope='Fixed public color probes and simulated six-second requests; validates moving physics and control handoff only. Not Astra results. No new external calls or budget reset.',requires='Training completes and its predeclared local audit gate passes',max_steps=1800))
def progress(**kw):state.update(kw,updated=time.time());save(out/'progress.json',state)
def stop(*args):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
    start=time.monotonic()
    while True:
        d=json.loads((training/'progress.json').read_text())
        if d['status'] in ('completed','failed','diagnostic_gate_failed'):
            break
        os.kill(a.training_pid,0)
        if b'run_learning_pipeline.py' not in Path(f'/proc/{a.training_pid}/cmdline').read_bytes():
            raise RuntimeError('Dependency PID is no longer the learning runner')
        progress(dependency_status=d['status'],dependency_child=d.get('child_pid'))
        if time.monotonic()-start>21600:raise TimeoutError('Training dependency exceeded six hours')
        time.sleep(5)
    if d['status']!='completed':
        progress(status='not_started',reason='Training did not complete: '+d['status']);sys.exit(0)
    result=json.loads((training/'train/result.json').read_text())
    if not result['offline_gate_passed']:
        progress(status='not_started',reason='Retention did not pass its held-out local gate');sys.exit(0)
    lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
    # The dependency writes completion before its finally releases the lock.
    started=time.monotonic()
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            progress(status='waiting_for_gpu')
            if time.monotonic()-started>300:raise TimeoutError('GPU lock still held after training completed')
            time.sleep(2)
    for mode in ('none','hold','learned'):
        for path,digest in frozen.items():
            if sha(path)!=digest:raise RuntimeError('Frozen local probe source changed: '+path)
        output=out/mode
        command=[PYTHON,str(Path(__file__).with_name('evaluate.py')),'--output',str(output),'--seed','104501','--arm','verify','--initialization','collision_clearance','--retention-mode',mode,'--evidence-model',str(ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'),'--max-steps','1800','--budget','6','--offline-smoke','--offline-retention-probe','--headless','--device','cuda:0']+KIT
        if mode=='learned':command+=['--retention-head',str(training/'train/head.pt')]
        save(out/'logs'/(mode+'_command.json'),command)
        with (out/'logs'/(mode+'.log')).open('w')as log:
            child=subprocess.Popen(command,cwd=ROOT,env=environment_variables(out),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        started=time.monotonic()
        while child.poll() is None:
            progress(status='probing',mode=mode,child_pid=child.pid,elapsed_s=time.monotonic()-started)
            failure=output/'runtime_failure.json'
            if failure.exists():raise RuntimeError(failure.read_text())
            heartbeat=output/'progress.json'
            if heartbeat.exists()and time.time()-heartbeat.stat().st_mtime>300:raise TimeoutError(mode+' stopped reporting')
            if time.monotonic()-started>1800:raise TimeoutError(mode+' exceeded probe timeout')
            time.sleep(5)
        if child.returncode or not(output/'report.json').exists():raise RuntimeError(mode+' probe incomplete')
        events=json.loads((output/'retention_events.json').read_text())
        progress(completed=state['completed']+[mode],child_pid=None)
        save(out/'logs'/(mode+'_events_summary.json'),dict(events=len(events),choices=sum(e['event']=='retention_choice'and e['rate']is not None for e in events),releases=sum(e['event']=='retention_release'for e in events)))
    progress(status='completed',mode=None,child_pid=None,cloud_eligible=False)
except SystemExit:
    raise
except BaseException as e:
    progress(status='failed',error=str(e));save(out/'failure.json',dict(error=traceback.format_exc()));raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    if lock:lock.close()
