"""Validate the pilot-selected fixed startup action on new paired seeds.

This is a finite local control experiment. It never calls Astra, trains a
selector, or deploys an action based on test outcomes.
"""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/semantic_completion_v1'))
from common import PYTHON, save, sha
from prepare import environment_variables, KIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'tmp', 'cases'):
        (out / name).mkdir()
    pilot = ROOT / 'outputs/astra_startup_value_20261022/pilot_r1/result.json'
    result = json.loads(pilot.read_text())
    if result['status'] != 'completed' or result['descriptive_best_fixed'] != 'skill1':
        raise RuntimeError('Expected the completed pilot and its frozen skill1 selection')
    source_protocol = ROOT / 'outputs/astra_startup_value_20261022/local_smoke_r1/record/protocol.json'
    frozen = json.loads(source_protocol.read_text())['source_sha256']
    frozen.update({str(Path(__file__).relative_to(ROOT)): sha(__file__),
                   str(pilot.relative_to(ROOT)): sha(pilot)})
    schedule = [(115101, 180), (115102, 360)]
    modes = ['latch', 'pose', 'skill1']
    save(out / 'protocol.json', dict(schedule=schedule, modes=modes, cases_per_seed=6,
        selected_fixed='skill1', selection_source=str(pilot), source_sha256=frozen,
        primary_comparison='Frozen skill1 against original latch; pose heuristic is secondary.',
        decision='At least two more completions than latch with no additional drops, and no fewer completions than pose. This only permits further integration validation.',
        scope='New local control states, not autonomous semantic tasks. No Astra calls or automatic deployment.',
        cloud_calls=0))
    state = dict(status='starting', pid=os.getpid(), completed=[], cloud_calls=0)
    child = None
    lock = None
    started = time.monotonic()

    def progress(**kw):
        state.update(kw, updated=time.time(), wall_s=time.monotonic()-started)
        save(out / 'progress.json', state)
        print(json.dumps(state), flush=True)

    def run(seed, wait, mode):
        nonlocal child
        for path, digest in frozen.items():
            if sha(ROOT / path) != digest:
                raise RuntimeError('Frozen source changed: ' + path)
        dest = out / 'cases' / str(seed) / mode
        command = [PYTHON, str(ROOT / 'scripts/astra_startup_value_v1/collect.py'),
            '--mode', mode, '--seed', str(seed), '--wait-steps', str(wait),
            '--output', str(dest), '--headless', '--device', 'cuda:0']
        if mode != 'record':
            command += ['--source', str(dest.parent / 'record')]
        command += KIT
        name = str(seed) + '_' + mode
        save(out / 'logs' / (name + '_command.json'), command)
        with (out / 'logs' / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment_variables(out),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        begin = time.monotonic()
        while child.poll() is None:
            p = dest / 'progress.json'
            progress(status='running', current=name, child_pid=child.pid,
                child_progress=json.loads(p.read_text()) if p.exists() else None)
            if time.monotonic()-begin > 1800 or time.monotonic()-started > 5400:
                raise TimeoutError('Fixed startup validation time bound exceeded')
            time.sleep(10)
        if child.returncode or not (dest / 'report.json').exists():
            raise RuntimeError((out / 'logs' / (name+'.log')).read_text()[-8000:])
        progress(completed=state['completed']+[name], child_pid=None)

    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        lock = Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed, wait in schedule:
            for mode in ['record'] + modes:
                run(seed, wait, mode)
        groups = []
        for seed, _ in schedule:
            folder = out / 'cases' / str(seed)
            expected = json.loads((folder / 'record/report.json').read_text())['decision_inputs_sha256']
            reports = {}
            for mode in modes:
                audit = json.loads((folder / mode / 'initial_audit.json').read_text())
                report = json.loads((folder / mode / 'report.json').read_text())
                if audit['decision_inputs_sha256'] != expected or any(audit['physical_errors'].values()):
                    raise RuntimeError('Initial physical or measured input mismatch')
                if report['status'] != 'completed' or report['cases'] != 6:
                    raise RuntimeError('Incomplete branch')
                reports[mode] = report
            groups.append(reports)
        totals = {m: {k: sum(g[m][k] for g in groups) for k in
            ('cases', 'completed', 'drops', 'startup_drops', 'post_handoff_drops')} for m in modes}
        candidate, base, pose = [totals[m] for m in ('skill1', 'latch', 'pose')]
        wins = sum(g['skill1']['episodes'][i]['completed'] and not g['latch']['episodes'][i]['completed']
                   for g in groups for i in range(6))
        losses = sum(g['latch']['episodes'][i]['completed'] and not g['skill1']['episodes'][i]['completed']
                     for g in groups for i in range(6))
        passed = (candidate['completed'] >= base['completed']+2 and candidate['drops'] <= base['drops']
                  and candidate['completed'] >= pose['completed'])
        save(out / 'result.json', dict(status='completed', by_mode=totals,
            paired_completion_wins=wins, paired_completion_losses=losses,
            fresh_local_gate_passed=passed, groups=groups, cloud_calls=0, deployed=False,
            note='All initial cases retained. Two seeds only. Local first-skill completion, not Astra task success.'))
        progress(status='completed', current=None, child_pid=None, fresh_local_gate_passed=passed)
    except BaseException:
        save(out / 'failure.json', dict(error=traceback.format_exc()))
        progress(status='failed', error=traceback.format_exc())
        raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
