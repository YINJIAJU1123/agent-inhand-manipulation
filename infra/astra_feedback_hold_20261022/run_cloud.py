"""Two fresh seeds, shared prefixes, fixed and feedback-released hold comparison."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'scripts/astra_control_prefix_v1'), str(ROOT/'scripts/semantic_completion_v1')]
from common import save
from prepare import KIT
import importlib.util
_spec=importlib.util.spec_from_file_location('_feedback_prefix_audit', ROOT/'scripts/astra_control_prefix_v1/run.py')
_audit=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
audit_branch=_audit.audit_branch

out = Path(os.environ['FEEDBACK_HOLD_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
for name in ('logs', 'tmp', 'cases'):
    (out/name).mkdir()
manifest = json.loads((ROOT/'package_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen package mismatch '+name)
save(out/'protocol.json', dict(seeds=[117101,117102], modes=['record','none','hold','feedback'],
    cases_per_seed=6, max_steps=1800, cloud_calls=0, package=manifest,
    intervention='Same hold entry threshold and request horizon. Release on the first fresh below-threshold observation; remain with original V3 until the next request. No new goal, no oracle trigger, no learned model change.',
    primary='Nominal execute-phase target-pose occupancy, retaining prefix and later drops.',
    gate='Feedback occupancy at least .05 above each of none/hold, no more drops than either, and no fewer final-ready states than either. No automatic deployment.',
    scope='Fixed supplied colors with advancing physics, not Astra or semantic task success.'))
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='0',
    TMPDIR=str(out/'tmp'), PYTHONPATH=':'.join([str(ROOT/'scripts/agentic'),
    str(ROOT/'outputs/learning_dependencies'), os.environ.get('PYTHONPATH','')]))
state = dict(status='starting', completed=[], cloud_calls=0)
started = time.monotonic()
child = None
summary = {}


def progress(**kw):
    state.update(kw, updated=time.time())
    save(out/'progress.json', state)
    print('FEEDBACK_HOLD_PROGRESS '+json.dumps(state), flush=True)


def run_stage(seed, mode):
    global child
    dest = out/'cases'/str(seed)/mode
    source = dest.parent/'record'
    command = [sys.executable, str(ROOT/'scripts/astra_feedback_hold_v1/evaluate.py'),
        '--mode', 'hold' if mode=='feedback' else mode, '--output', str(dest), '--headless', '--device', 'cuda:0']
    command += ['--seed',str(seed)] if mode=='record' else ['--source',str(source)]
    if mode=='feedback': command += ['--feedback-release']
    command += KIT
    name=str(seed)+'_'+mode
    save(out/'logs'/(name+'_command.json'), command)
    with (out/'logs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    begin=time.monotonic()
    while child.poll() is None:
        p=dest/'progress.json'
        progress(status='running',current=name,child_pid=child.pid,
            child_progress=json.loads(p.read_text()) if p.exists() else None,elapsed_s=time.monotonic()-begin)
        if time.monotonic()-begin>1800 or time.monotonic()-started>3600:
            raise TimeoutError('Feedback comparison exceeded its bound')
        time.sleep(15)
    if child.returncode or not (dest/'report.json').exists():
        raise RuntimeError((out/'logs'/(name+'.log')).read_text()[-8000:])
    report=json.loads((dest/'report.json').read_text())
    intervention=json.loads((dest.parent/(dest.name+'_intervention.json')).read_text())
    if report['status']!='completed' or intervention['feedback_release']!=(mode=='feedback'):
        raise RuntimeError('Wrong intervention or incomplete result')
    for name,key in [('evaluate.py','feedback_wrapper_sha256'),('feedback_window.py','feedback_window_sha256')]:
        if intervention[key]!=hashlib.sha256((ROOT/'scripts/astra_feedback_hold_v1'/name).read_bytes()).hexdigest():
            raise RuntimeError('Intervention implementation hash mismatch')
    report.update(intervention)
    if mode!='record':audit_branch(source,dest)
    progress(completed=state['completed']+[name],child_pid=None)
    return report


try:
    for seed in (117101,117102):
        summary[str(seed)]={}
        for mode in ('record','none','hold','feedback'):
            report=run_stage(seed,mode)
            summary[str(seed)][mode]=report
            if mode=='record' and report['branch_step'] is None:
                summary[str(seed)]['not_started']='No eligible branch. Keep source and do not replace seed.'
                break
    totals={}
    if all(all(m in group for m in ('none','hold','feedback')) for group in summary.values()):
        for mode in ('none','hold','feedback'):
            reports=[group[mode] for group in summary.values()]
            totals[mode]={k:sum(r[k] for r in reports) for k in ('cases','drops','prefix_drops','final_ready')}
            for metric in ('execute_ready_fraction','inspect_ready_fraction'):
                totals[mode][metric]=sum(r[metric]*r['cases'] for r in reports)/totals[mode]['cases']
        f=totals['feedback']
        passed=all(f['execute_ready_fraction']>=totals[m]['execute_ready_fraction']+.05 and
            f['drops']<=totals[m]['drops'] and f['final_ready']>=totals[m]['final_ready'] for m in ('none','hold'))
    else:passed=False
    result=dict(status='completed',by_mode=totals,local_gate_passed=passed,by_seed=summary,
        cloud_calls=0,deployed=False,note='Paired fixed-color control only. No Astra task success estimate.')
    save(out/'result.json',result)
    blob=gzip.compress(json.dumps(result).encode(),mtime=0)
    print('FEEDBACK_HOLD_RESULT_SHA256 '+hashlib.sha256(blob).hexdigest(),flush=True)
    print('FEEDBACK_HOLD_RESULT_BEGIN',flush=True)
    encoded=base64.b64encode(blob).decode()
    for i in range(0,len(encoded),64):print(encoded[i:i+64],flush=True)
    print('FEEDBACK_HOLD_RESULT_END',flush=True)
    progress(status='completed',current=None,child_pid=None)
except BaseException:
    save(out/'failure.json',dict(error=traceback.format_exc()))
    progress(status='failed',error=traceback.format_exc());raise
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
