"""Two-seed finite continuous-control diagnostic with a shared recorded prefix."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/semantic_completion_v1'))
from common import PYTHON, save, sha, STUDENT, ADAPTER, CONTROLLER
from prepare import environment_variables, KIT
from control_protocol import assert_equal


def audit_branch(source, branch):
    bootstrap = torch.load(source/'bootstrap.pt', map_location='cpu', weights_only=True)
    start = torch.load(branch/'branch_start.pt', map_location='cpu', weights_only=True)
    checks = {'physics':assert_equal(start['physical'],bootstrap['physical'],tolerance=0.)}
    fields = ('ih','ph','proprio','current','proposal','base_action','ready_probability','feature_bank')
    checks['sensors_and_proposals'] = assert_equal({k:start[k] for k in fields}, {k:bootstrap[k] for k in fields}, tolerance=0.)
    before = torch.load(source/'actions.pt', map_location='cpu', weights_only=True)
    after = torch.load(branch/'actions.pt', map_location='cpu', weights_only=True)
    step = bootstrap['step']
    if not torch.equal(before[:step],after[:step]):
        raise RuntimeError('Command prefix is not bitwise identical')
    report = json.loads((branch/'report.json').read_text())
    if report['mode'] == 'none' and not torch.equal(before[step],after[step]):
        raise RuntimeError('No-treatment first command differs despite shared inputs')
    for i, prefix in enumerate(bootstrap['rows']):
        if prefix['drop'] and report['episodes'][i]['drop_step'] != prefix['drop_step']:
            raise RuntimeError('Prefix failure was not retained')
    checks.update(branch_step=step, action_prefix_bitwise_equal=True,
                  original_prefix_drops=int((~bootstrap['active']).sum()), cases=report['cases'])
    save(branch/'shared_state_audit.json',checks)
    return checks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ('logs','tmp'):(out/name).mkdir()
    seeds=(105601,105602)
    head=ROOT/'outputs/astra_retention_v1/learning_20261022/train/head.pt'
    files=list(Path(__file__).parent.glob('*.py'))
    for directory in ('astra_retention_online_v1','astra_retention_v1','astra_learning_v1',
                      'astra_runtime_v2','astra_completion_v3','semantic_completion_v1',
                      'semantic_completion_v2','semantic_completion_v3','async_stability_v1','agentic'):
        files.extend((ROOT/'scripts'/directory).glob('*.py'))
    files.extend((ROOT/'source/BrainCo_DexHand').rglob('*.py'))
    files.extend([STUDENT,ADAPTER,CONTROLLER,head,head.parent/'result.json',
                  ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'])
    frozen={str(path):sha(path) for path in files}
    save(out/'protocol.json',dict(seeds=seeds,modes=['record','none','hold','learned'],cases_per_seed=6,
        max_steps=1800,stage_timeout_s=1800,source_sha256=frozen,cloud_calls=0,
        scope='Control diagnostic with predetermined public colors and six-second simulated latency. No language success score.',
        selection='First eligible pending inspection frame by measured pose probability. Entire batch shares prefix. No outcome-based state selection.',
        no_eligible_seed='Retain record and mark other arms not_started. Do not replace the seed.',
        candidate='Exploratory retention head failed actual runtime-cohort gate; no deployment or Astra calls.'))
    state=dict(status='starting',pid=os.getpid(),completed=[],cloud_calls=0)
    child=None;lock=None;summary={}
    def progress(**kw):
        state.update(kw,updated=time.time());save(out/'progress.json',state)
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        lock=Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in seeds:
            summary[str(seed)]={}
            source=out/str(seed)/'record'
            for mode in ('record','none','hold','learned'):
                for path,digest in frozen.items():
                    if sha(path)!=digest:raise RuntimeError('Frozen dependency changed: '+path)
                target=out/str(seed)/mode
                name=f'{seed}_{mode}'
                command=[PYTHON,str(Path(__file__).with_name('evaluate.py')),'--mode',mode,
                    '--output',str(target),'--retention-head',str(head),'--headless','--device','cuda:0']
                command+=['--seed',str(seed)] if mode=='record' else ['--source',str(source)]
                command+=KIT
                save(out/'logs'/(name+'_command.json'),command)
                with (out/'logs'/(name+'.log')).open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=environment_variables(out),stdout=log,
                        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                started=time.monotonic()
                while child.poll() is None:
                    sub=None
                    if (target/'progress.json').exists():
                        sub=json.loads((target/'progress.json').read_text())
                    progress(status='running',current=name,child_pid=child.pid,child_progress=sub,elapsed_s=time.monotonic()-started)
                    if (target/'failure.json').exists():raise RuntimeError((target/'failure.json').read_text())
                    if time.monotonic()-started>1800:raise TimeoutError(name+' exceeded stage timeout')
                    if sub and time.time()-(target/'progress.json').stat().st_mtime>300:
                        raise TimeoutError(name+' stopped reporting progress')
                    time.sleep(5)
                if child.returncode or not (target/'report.json').exists():
                    raise RuntimeError(name+' incomplete; exit '+str(child.returncode))
                report=json.loads((target/'report.json').read_text())
                if report['status']!='completed':raise RuntimeError(name+' not completed')
                if mode!='record':audit_branch(source,target)
                summary[str(seed)][mode]={k:report[k] for k in ('cases','drops','prefix_drops','branch_step','final_ready','inspect_ready_fraction','execute_ready_fraction')}
                progress(completed=state['completed']+[name],child_pid=None)
                save(out/'summary.json',dict(by_seed=summary,cloud_calls=0,
                    note='Fixed-color continuous-control diagnostic. Record is an extra no-treatment repeat; none is the matched intervention baseline. No task or Astra success estimate.'))
                if mode=='record' and report['branch_step'] is None:
                    summary[str(seed)]['interventions']='not_started_no_eligible_state'
                    break
        progress(status='completed',current=None,child_pid=None)
    except BaseException as e:
        progress(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e))
        save(out/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        if lock:lock.close()


if __name__=='__main__':main()
