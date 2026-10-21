"""Finite fresh hold calibration, then a separately seeded audit if it passes.

No fitting, remote calls or controller deployment. Uses the established shared
prefix collector and a single six-second hold branch. All prefix failures stay
in the nominal denominator; only measured surviving states have hold labels.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from model import HoldEvidence
from train import THRESHOLDS, score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts' / name) for name in
    ('semantic_completion_v1', 'semantic_completion_v3', 'astra_retention_v1')]
from common import PYTHON, save, sha
from prepare import environment_variables, KIT
from dense_model import load_model
from value_model import frozen_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'tmp', 'data'):
        (out / name).mkdir()
    checkpoint = ROOT / 'outputs/astra_hold_evidence_20261022/pooled_r1/head.pt'
    dense = ROOT / 'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
    model = HoldEvidence().eval()
    pack = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if pack['encoder_sha256'] != sha(dense):
        raise RuntimeError('Feature encoder mismatch')
    model.load_state_dict(pack['model'])
    model.requires_grad_(False)
    torch.set_num_threads(2)
    encoder, _ = load_model(dense, 'cpu')
    frozen = {str(path): sha(path) for path in [checkpoint, dense, Path(__file__),
        Path(__file__).with_name('model.py'), Path(__file__).with_name('train.py'),
        ROOT / 'scripts/astra_retention_v1/collect_prefix.py', ROOT / 'scripts/astra_retention_v1/branch.py',
        ROOT / 'scripts/astra_retention_v1/value_model.py']}
    save(out / 'protocol.json', dict(calibration_seeds=[114301, 114302], audit_seeds=[114401, 114402],
        cases_per_seed=24, frozen=frozen, thresholds=THRESHOLDS, cloud_calls=0,
        trigger='Existing collector: common branch when half the nominal batch has measured p>=0.7, or step720.',
        label='At least 95% of six seconds within 0.16rad and no drop under hold.',
        calibration='Maximum coverage, at least eight accepts, >=90% precision, and at least 80% of baseline reliable accepts.',
        audit='Same minimum count and precision, fewer unsafe accepts than original p>=0.7, retaining >=80% of its reliable accepts.',
        scope='Fresh local hold reliability only. No fitting, semantic success estimate, Astra requests or automatic deployment.'))
    state = dict(status='starting', pid=os.getpid(), completed=[], cloud_calls=0)
    child = None
    lock = None
    started = time.monotonic()

    def progress(**kw):
        state.update(kw, updated=time.time(), wall_s=time.monotonic()-started)
        save(out / 'progress.json', state)
        print(json.dumps(state), flush=True)

    def run(name, command, report):
        nonlocal child
        for path, digest in frozen.items():
            if sha(path) != digest:
                raise RuntimeError('Frozen model or collector changed: ' + path)
        save(out / 'logs' / (name+'_command.json'), command)
        with (out / 'logs' / (name+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment_variables(out),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        begin = time.monotonic()
        while child.poll() is None:
            path = report.parent / 'progress.json'
            progress(status='collecting', current=name, child_pid=child.pid,
                child_progress=json.loads(path.read_text()) if path.exists() else None)
            if time.monotonic()-begin > 1800 or time.monotonic()-started > 7200:
                raise TimeoutError('Fresh validation exceeded its time bound')
            time.sleep(10)
        if child.returncode or not report.exists():
            raise RuntimeError((out / 'logs' / (name+'.log')).read_text()[-8000:])
        progress(completed=state['completed']+[name], child_pid=None)

    def collect(split, seeds):
        rows, nominal, prefix_drops = [], 0, 0
        for seed in seeds:
            shard = out / 'data' / str(seed)
            shard.mkdir()
            prefix, branch = shard / 'prefix', shard / 'hold'
            run(str(seed)+'_prefix', [PYTHON, str(ROOT / 'scripts/astra_retention_v1/collect_prefix.py'),
                '--output', str(prefix), '--seed', str(seed), '--split', split, '--cases', '24',
                '--headless', '--device', 'cuda:0']+KIT, prefix / 'report.json')
            run(str(seed)+'_hold', [PYTHON, str(ROOT / 'scripts/astra_retention_v1/branch.py'),
                '--source', str(prefix), '--plan', str(prefix / 'sensor_branch_plan.json'),
                '--output', str(branch), '--rate', '0.0', '--headless', '--device', 'cuda:0']+KIT,
                branch / 'report.json')
            report = json.loads((branch / 'report.json').read_text())
            source = json.loads((prefix / 'report.json').read_text())
            nominal += len(source['episodes'])
            prefix_drops += sum(r['reason'] == 'prefix_drop' for r in source['episodes'])
            if any(max(r['errors'].values()) > 1e-6 for r in report['audits']):
                raise RuntimeError('Physical prefix replay mismatch')
            snapshot = torch.load(branch / 'branches.pt', map_location='cpu', weights_only=True)
            for r in report['results']:
                case = r['case']
                x = {k: v[None] for k, v in snapshot['features'][case].items()}
                f = frozen_features(encoder, x, torch.tensor([r['goal']]))[0]
                rows.append(dict(seed=seed, case=case, feature=f, occupancy=r['ready_fraction'],
                    reliable=float(r['ready_fraction'] >= .95 and not r['drop']), drop=float(r['drop'])))
        if len(rows) < 8:
            raise RuntimeError('Fewer than eight surviving states, retain prefix failures and stop')
        data = {k: torch.stack([r[k] for r in rows]) if k == 'feature' else torch.tensor([r[k] for r in rows])
                for k in ('feature', 'reliable', 'occupancy', 'drop')}
        torch.save(dict(data=data, keys=[dict(seed=r['seed'], case=r['case']) for r in rows]), out / (split+'.pt'))
        with torch.inference_mode():
            probability = model(data['feature'])['reliable_logit'].sigmoid()
        save(out / (split+'_cohort.json'), dict(nominal_cases=nominal, prefix_drops=prefix_drops,
            measured_surviving_states=len(rows), note='Reliability precision is conditional on a surviving branch state.'))
        return data, probability

    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        lock = Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cal, probability = collect('calibration', [114301, 114302])
        baseline = score(cal['feature'][:, 98].sigmoid(), .7, cal)
        curve = [score(probability, t, cal) for t in THRESHOLDS]
        eligible = [r for r in curve if r['accepted'] >= 8 and r['precision'] >= .9
                    and r['reliable'] >= .8*baseline['reliable']]
        selected = max(eligible, key=lambda r:(r['accepted'], r['precision'])) if eligible else None
        save(out / 'calibration.json', dict(selected=selected, curve=curve, baseline=baseline,
            checkpoint_sha256=sha(checkpoint)))
        if not selected:
            save(out / 'result.json', dict(status='completed', calibration_gate_passed=False,
                audit_status='not_started', cloud_calls=0, deployed=False))
            progress(status='completed', current=None, child_pid=None, calibration_gate_passed=False)
            return
        audit, probability = collect('audit', [114401, 114402])
        measured = score(probability, selected['threshold'], audit)
        baseline = score(audit['feature'][:, 98].sigmoid(), .7, audit)
        passed = bool(measured['accepted'] >= 8 and measured['precision'] >= .9
            and measured['unsafe'] < baseline['unsafe'] and measured['reliable'] >= .8*baseline['reliable'])
        save(out / 'result.json', dict(status='completed', calibration=selected, audit=measured,
            baseline_audit=baseline, fresh_local_gate_passed=passed, checkpoint_sha256=sha(checkpoint),
            cloud_calls=0, deployed=False, note='Still requires closed-loop control and real Astra validation.'))
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
