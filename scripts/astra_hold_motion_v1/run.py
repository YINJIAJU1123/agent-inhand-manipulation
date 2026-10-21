"""Finite matched motion-feature ablation with fresh calibration and audit.

Previously inspected local data are explicitly development training data now.
No external calls, simulator object-state inputs, or automatic deployment.
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

import torch
from torch.nn import functional as F
from model import HoldMotion, measured_motion

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend(str(ROOT/'scripts'/name) for name in
    ('semantic_completion_v1', 'semantic_completion_v3', 'astra_retention_v1'))
from common import PYTHON, save, sha
from prepare import environment_variables, KIT
from dense_model import load_model
from value_model import frozen_features

THRESHOLDS = (.5, .6, .7, .8, .85, .9, .95, .975, .99)


def score(prob, threshold, data):
    mask = prob >= threshold
    count = int(mask.sum())
    correct = int((mask & data['reliable'].bool()).sum())
    return dict(threshold=threshold, states=len(mask), accepted=count, reliable=correct,
        unsafe=count-correct, precision=correct/count if count else None,
        accepted_drops=int(data['drop'][mask].sum()),
        mean_occupancy=float(data['occupancy'][mask].mean()) if count else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'tmp', 'data', 'models'):
        (out/name).mkdir()
    torch.set_num_threads(2)
    dense = ROOT/'outputs/semantic_completion_v3/training_20261021/dense/head.pt'
    encoder, _ = load_model(dense, 'cpu')
    folders = sorted((ROOT/'outputs/astra_retention_v1/learning_20261022/data').glob('*/rate0.0'))
    folders += [ROOT/f'outputs/astra_hold_evidence_20261022/fresh_r1/data/{seed}/hold'
                for seed in (114301, 114302)]
    frozen_paths = [Path(__file__), Path(__file__).with_name('model.py'), dense,
        ROOT/'scripts/astra_retention_v1/collect_prefix.py', ROOT/'scripts/astra_retention_v1/branch.py',
        ROOT/'scripts/astra_retention_v1/value_model.py']
    frozen_paths += [p/name for p in folders for name in ('report.json', 'branches.pt')]
    frozen = {str(p): sha(p) for p in frozen_paths}
    save(out/'protocol.json', dict(train_sources=[str(p) for p in folders],
        training_scope='All listed old local train/calibration/audit and failed fresh-calibration data are now development training data. No old held-out claim remains.',
        calibration_seeds=[116301, 116302], audit_seeds=[116401, 116402], cases_per_seed=24,
        source_sha256=frozen, variants=['pose_features', 'pose_plus_motion'],
        measured_motion='21 scaled joint velocities, 30 fingertip kinematic velocities, 21 previous-command minus joint-position differences, four target RGB-marker history differences.',
        forbidden_inputs='Object pose/velocity, goal truth, seed, future outcome. Simulation truth supplies labels only.',
        fit='Same 100 epochs, AdamW lr=.001 wd=.1, hidden16, BCE plus .5 occupancy smooth-L1, minimum calibration BCE checkpoint, same training shuffle seed.',
        calibration='Frozen threshold grid; maximum coverage with >=8 accepted, >=90% precision, retaining >=80% of dense-pose reliable accepts.',
        audit='Only if motion calibration passes. Same count/precision/coverage and fewer unsafe accepts than dense-pose baseline. Report matched 125-feature ablation separately without reselecting on audit.',
        thresholds=THRESHOLDS, cloud_calls=0, deployment=False))
    state = dict(status='starting', pid=os.getpid(), completed=[], cloud_calls=0)
    child, lock = None, None
    started = time.monotonic()

    def progress(**kw):
        state.update(kw, updated=time.time(), wall_s=time.monotonic()-started)
        save(out/'progress.json', state)
        print(json.dumps(state), flush=True)

    def check():
        for path, digest in frozen.items():
            if sha(path) != digest:
                raise RuntimeError('Frozen dependency changed: '+path)

    def extract(sources, split):
        features, reliability, occupancy, drops, keys = [], [], [], [], []
        for folder in sources:
            report = json.loads((folder/'report.json').read_text())
            prefix = json.loads((Path(report['source'])/'report.json').read_text())
            seed = prefix['metadata']['seed']
            data = torch.load(folder/'branches.pt', map_location='cpu', weights_only=True)
            if report['rate'] != 0. or report['horizon'] != 180:
                raise RuntimeError('Mixed hold interventions')
            if max(max(x['errors'].values()) for x in report['audits']) > 1e-6:
                raise RuntimeError('Physical prefix mismatch')
            for row in report['results']:
                case, goal = row['case'], row['goal']
                # Explicitly use only the measured feature dict. data['states'] is never read.
                sensed = {k:v[None] for k,v in data['features'][case].items()}
                base = frozen_features(encoder, sensed, torch.tensor([goal]))[0]
                motion = measured_motion(sensed['x'][0, goal])
                features.append(torch.cat((base, motion)))
                reliability.append(float(row['ready_fraction'] >= .95 and not row['drop']))
                occupancy.append(row['ready_fraction'])
                drops.append(float(row['drop']))
                keys.append(dict(seed=seed, case=case))
        if len({(r['seed'], r['case']) for r in keys}) != len(keys):
            raise RuntimeError('Duplicate state labels')
        result = dict(feature=torch.stack(features), reliable=torch.tensor(reliability),
                      occupancy=torch.tensor(occupancy), drop=torch.tensor(drops))
        torch.save(dict(data=result, keys=keys), out/(split+'.pt'))
        return result, keys

    def run(seed, split, mode):
        nonlocal child
        check()
        folder = out/'data'/str(seed)
        dest = folder/mode
        if mode == 'prefix':
            command = [PYTHON, str(ROOT/'scripts/astra_retention_v1/collect_prefix.py'),
                '--seed', str(seed), '--cases', '24', '--split', split]
        else:
            command = [PYTHON, str(ROOT/'scripts/astra_retention_v1/branch.py'),
                '--source', str(folder/'prefix'), '--plan', str(folder/'prefix/sensor_branch_plan.json'), '--rate', '0.0']
        command += ['--output', str(dest), '--headless', '--device', 'cuda:0']+KIT
        name = str(seed)+'_'+mode
        save(out/'logs'/(name+'_command.json'), command)
        with (out/'logs'/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment_variables(out),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        begin = time.monotonic()
        while child.poll() is None:
            p = dest/'progress.json'
            progress(status='collecting', current=name, child_pid=child.pid,
                child_progress=json.loads(p.read_text()) if p.exists() else None)
            if time.monotonic()-begin > 1800 or time.monotonic()-started > 10800:
                raise TimeoutError('Motion ablation exceeded time bound')
            time.sleep(10)
        if child.returncode or not (dest/'report.json').exists():
            raise RuntimeError((out/'logs'/(name+'.log')).read_text()[-8000:])
        progress(completed=state['completed']+[name], child_pid=None)

    def collect(split, seeds):
        for seed in seeds:
            for mode in ('prefix', 'hold'):
                run(seed, split, mode)
        sources = [out/'data'/str(seed)/'hold' for seed in seeds]
        reports = [json.loads((out/'data'/str(seed)/'prefix/report.json').read_text()) for seed in seeds]
        save(out/(split+'_cohort.json'), dict(nominal_cases=sum(len(r['episodes']) for r in reports),
            prefix_drops=sum(e['reason']=='prefix_drop' for r in reports for e in r['episodes'])))
        return extract(sources, split)

    def fit(name, motion, train, cal):
        torch.manual_seed(202610223)
        model = HoldMotion(motion)
        dimension = model.dimension
        x, y = train['feature'][:, :dimension], cal['feature'][:, :dimension]
        model.mean.copy_(x.mean(0)); model.scale.copy_(x.std(0).clamp_min(.05))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
        best, history = float('inf'), []
        folder = out/'models'/name; folder.mkdir()
        for epoch in range(100):
            model.train(); losses = []
            # Same shuffle for both variants, independent of model initialization width.
            indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(202610223+epoch))
            for index in indices.split(32):
                pred = model(x[index])
                loss = F.binary_cross_entropy_with_logits(pred['reliable_logit'], train['reliable'][index])
                loss += .5*F.smooth_l1_loss(pred['occupancy'], train['occupancy'][index])
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                loss = float(F.binary_cross_entropy_with_logits(model(y)['reliable_logit'], cal['reliable']))
            history.append(dict(epoch=epoch+1, train_loss=sum(losses)/len(losses), calibration_bce=loss))
            if loss < best:
                best = loss
                torch.save(dict(model={k:v.clone() for k,v in model.state_dict().items()},
                    epoch=epoch+1, motion=motion, encoder_sha256=sha(dense), deployment_eligible=False), folder/'head.pt')
            if epoch % 10 == 0:
                progress(status='training', current=name, epoch=epoch+1, calibration_bce=loss)
        model.load_state_dict(torch.load(folder/'head.pt', weights_only=True)['model']); model.eval()
        with torch.no_grad():prob = model(y)['reliable_logit'].sigmoid()
        baseline = score(cal['feature'][:, 98].sigmoid(), .7, cal)
        curve = [score(prob, t, cal) for t in THRESHOLDS]
        good = [r for r in curve if r['accepted']>=8 and r['precision']>=.9 and r['reliable']>=.8*baseline['reliable']]
        selected = max(good, key=lambda r:(r['accepted'],r['precision'])) if good else None
        save(folder/'history.json', history)
        save(folder/'selection.json', dict(selected=selected, curve=curve, baseline=baseline,
            checkpoint_sha256=sha(folder/'head.pt'), calibration_brier=float(((prob-cal['reliable'])**2).mean())))
        return model, selected

    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    try:
        train, train_keys = extract(folders, 'train')
        progress(status='training_data_ready', train_states=len(train_keys), train_reliable=int(train['reliable'].sum()))
        if args.wait_pid:
            while True:
                try:
                    os.kill(args.wait_pid, 0)
                    command = Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
                except (ProcessLookupError, FileNotFoundError):
                    break
                if b'astra_startup_value_v1/validate_fixed.py' not in command:
                    raise RuntimeError('Dependency PID changed identity')
                progress(status='waiting_for_gpu', dependency_pid=args.wait_pid)
                if time.monotonic()-started > 5400: raise TimeoutError('Dependency wait bound exceeded')
                time.sleep(10)
        lock = Path('/tmp/agent_inhand_paired_gpu_0.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cal, cal_keys = collect('calibration', (116301, 116302))
        if {r['seed'] for r in train_keys} & {r['seed'] for r in cal_keys}: raise RuntimeError('Split overlap')
        models = {name:fit(name, motion, train, cal) for name,motion in
                  (('pose_features', False), ('pose_plus_motion', True))}
        selections = {k:v[1] for k,v in models.items()}
        save(out/'frozen_selection.json', selections)
        if selections['pose_plus_motion'] is None:
            save(out/'result.json', dict(status='completed', calibration_gate_passed=False,
                selection=selections, audit_status='not_started', cloud_calls=0, deployed=False))
            progress(status='completed', current=None, calibration_gate_passed=False); return
        audit, audit_keys = collect('audit', (116401,116402))
        if {r['seed'] for r in audit_keys} & {r['seed'] for r in train_keys+cal_keys}: raise RuntimeError('Audit overlap')
        baseline = score(audit['feature'][:,98].sigmoid(), .7, audit)
        measured = {}
        with torch.no_grad():
            for name,(model,selected) in models.items():
                measured[name] = score(model(audit['feature'][:,:model.dimension])['reliable_logit'].sigmoid(), selected['threshold'], audit) if selected else None
        m=measured['pose_plus_motion']
        passed=bool(m['accepted']>=8 and m['precision']>=.9 and m['unsafe']<baseline['unsafe'] and m['reliable']>=.8*baseline['reliable'])
        save(out/'result.json', dict(status='completed', selection=selections, audit=measured,
            dense_baseline=baseline, fresh_local_gate_passed=passed, cloud_calls=0, deployed=False,
            note='Local hold prediction only. Closed-loop and real Astra success still untested.'))
        progress(status='completed', current=None, fresh_local_gate_passed=passed)
    except BaseException:
        save(out/'failure.json', dict(error=traceback.format_exc()))
        progress(status='failed', error=traceback.format_exc()); raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
        if lock: lock.close()


if __name__ == '__main__':
    main()
