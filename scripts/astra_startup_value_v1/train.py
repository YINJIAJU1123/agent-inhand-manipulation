"""Train on complete matched outcomes, freeze on calibration, then open audit.

One random future command labels each initial state. It is absent from model
inputs, so this estimates completion averaged over the command distribution.
An oracle using those future outcomes is not an achievable startup policy.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from model import StartupValue, ARMS, FEATURE_DIM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_split(root, seeds, reference=None):
    rows, files = [], {}
    for seed in seeds:
        parent = Path(root) / str(seed)
        record = read(parent / 'record/report.json')
        protocol = read(parent / 'record/protocol.json')
        if record['status'] != 'completed' or record['seed'] != seed:
            raise RuntimeError('Wrong or incomplete source record')
        if reference is None:
            reference = protocol['source_sha256']
        if protocol['source_sha256'] != reference:
            raise RuntimeError('Source or weights changed across seeds')
        inp = parent / 'record/decision_inputs.pt'
        if sha(inp) != record['decision_inputs_sha256']:
            raise RuntimeError('Measured inputs changed')
        data = torch.load(inp, map_location='cpu', weights_only=True)
        if set(data) != {'features', 'rgb', 'ih', 'ph', 'compact', 'proprio'}:
            raise RuntimeError('Unexpected fields in measured inputs')
        features = data['features'].float()
        if features.shape != (record['cases'], 7, FEATURE_DIM) or not torch.isfinite(features).all():
            raise RuntimeError('Invalid measured feature tensor')
        bootstrap = parent / 'record/bootstrap.pt'
        if sha(bootstrap) != record['bootstrap_sha256']:
            raise RuntimeError('Initial snapshot changed')
        target = torch.load(bootstrap, map_location='cpu', weights_only=True)['future_colors']
        reports = {}
        for arm in ARMS + ('pose',):
            report_path = parent / arm / 'report.json'
            audit_path = parent / arm / 'initial_audit.json'
            r, audit = read(report_path), read(audit_path)
            if (r['status'] != 'completed' or r['seed'] != seed or r['cases'] != record['cases']
                    or r['future_colors'] != target or r['wait_steps'] != protocol['wait_steps']
                    or r['decision_inputs_sha256'] != record['decision_inputs_sha256']):
                raise RuntimeError('Unmatched or incomplete action outcomes')
            if (not audit['shared_sensor_state'] or any(v != 0 for v in audit['physical_errors'].values())
                    or audit['decision_inputs_sha256'] != record['decision_inputs_sha256']):
                raise RuntimeError('Action branches did not share initial measurements and physics')
            if [x['case'] for x in r['episodes']] != list(range(record['cases'])):
                raise RuntimeError('Case ordering changed')
            if any(x['completed'] and x['drop'] for x in r['episodes']):
                raise RuntimeError('Mutually inconsistent endpoint labels')
            reports[arm] = r
            files[str(report_path)] = sha(report_path)
            files[str(audit_path)] = sha(audit_path)
        for case in range(record['cases']):
            outcomes = [reports[arm]['episodes'][case] for arm in ARMS]
            rows.append(dict(seed=seed, case=case, features=features[case],
                label=torch.tensor([1 if r['completed'] else 2 if r['drop'] else 0 for r in outcomes]),
                startup_drop=torch.tensor([float(r['drop'] and r['drop_step'] <= protocol['wait_steps']) for r in outcomes]),
                pose_completed=reports['pose']['episodes'][case]['completed'],
                pose_drop=reports['pose']['episodes'][case]['drop']))
        for file in (inp, bootstrap, parent / 'record/protocol.json', parent / 'record/report.json'):
            files[str(file)] = sha(file)
    if not rows:
        raise RuntimeError('Empty split')
    return dict(features=torch.stack([r['features'] for r in rows]),
        label=torch.stack([r['label'] for r in rows]),
        startup_drop=torch.stack([r['startup_drop'] for r in rows]),
        keys=[dict(seed=r['seed'], case=r['case']) for r in rows],
        pose_completed=sum(r['pose_completed'] for r in rows), pose_drops=sum(r['pose_drop'] for r in rows),
        source_sha256=reference, files=files)


@torch.inference_mode()
def metrics(model, data):
    model.eval()
    choice = model.choose(data['features'])
    picked = data['label'][torch.arange(len(choice)), choice]
    fixed_completed = (data['label'] == 1).sum(0).tolist()
    fixed_drops = (data['label'] == 2).sum(0).tolist()
    return dict(states=len(choice), completed=int((picked == 1).sum()), drops=int((picked == 2).sum()),
        alive_incomplete=int((picked == 0).sum()), choices=choice.tolist(),
        fixed_completed=dict(zip(ARMS, fixed_completed)), fixed_drops=dict(zip(ARMS, fixed_drops)),
        pose_completed=data['pose_completed'], pose_drops=data['pose_drops'],
        outcome_oracle_completed=int((data['label'] == 1).any(-1).sum()),
        per_state=[dict(**key, action=ARMS[int(c)], outcome=int(y))
                   for key, c, y in zip(data['keys'], choice, picked)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    splits = read(a.splits)
    if set(splits) != {'train', 'calibration', 'audit'}:
        raise ValueError('Provide exactly the three independent seed splits')
    if any(len(v) != len(set(v)) for v in splits.values()):
        raise ValueError('Duplicate seeds')
    groups = [set(splits[k]) for k in ('train', 'calibration', 'audit')]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Seed leakage between splits')
    if len(splits['train']) < 8 or len(splits['calibration']) < 2 or len(splits['audit']) != 2:
        raise ValueError('Need at least eight training/two calibration seeds and exactly two audit seeds')
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(202610221)
    train = load_split(a.data, splits['train'])
    cal = load_split(a.data, splits['calibration'], train['source_sha256'])
    model = StartupValue()
    model.mean.copy_(train['features'].mean((0, 1)))
    model.scale.copy_(train['features'].flatten(0, 1).std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.03)
    best, history, start = float('inf'), [], time.time()
    save(out / 'protocol.json', dict(splits=splits, epochs=100, model_sha256=sha(Path(__file__).with_name('model.py')),
        trainer_sha256=sha(__file__), collector_sha256=train['source_sha256'],
        training_files=train['files'], calibration_files=cal['files'],
        selection='Minimum calibration completion-probability Brier loss; audit files opened only afterward.',
        action_choice='Maximum predicted eventual completion probability; survival is auxiliary supervision.',
        input='Measured initial frozen RGB-D features and candidate motor proposals only. No future command or latency.',
        limitation='One independent random future command per state supplies a noisy marginal completion label.',
        gate='At least two more audit completions than the calibration-selected fixed action, no additional drops, and no fewer completions than the measured pose heuristic.',
        scope='Counterfactual development test only. Mixed selected actions need fresh closed-loop validation.', cloud_calls=0))
    for epoch in range(100):
        model.train()
        losses = []
        for ids in torch.randperm(len(train['features'])).split(16):
            pred = model(train['features'][ids])
            label = train['label'][ids]
            loss = F.cross_entropy(pred['outcome_logits'].flatten(0, 1), label.flatten())
            loss = loss + .2 * F.binary_cross_entropy_with_logits(pred['startup_drop_logit'], train['startup_drop'][ids])
            successes = label == 1
            pairs = successes[:, :, None] & ~successes[:, None, :]
            if pairs.any():
                score = pred['completion_probability']
                loss = loss + .2 * F.softplus(.1 - (score[:, :, None] - score[:, None, :]))[pairs].mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            brier = float(F.mse_loss(model(cal['features'])['completion_probability'], (cal['label'] == 1).float()))
        history.append(dict(epoch=epoch + 1, loss=sum(losses) / len(losses), calibration_brier=brier))
        save(out / 'progress.json', dict(status='training', pid=os.getpid(), **history[-1], elapsed_s=time.time() - start))
        if brier < best:
            best = brier
            torch.save(dict(model={k: v.clone() for k, v in model.state_dict().items()}, epoch=epoch + 1,
                collector_sha256=train['source_sha256'], arms=ARMS, feature_dim=FEATURE_DIM,
                cloud_eligible=False), out / 'head.pt')
    frozen_sha = sha(out / 'head.pt')
    model.load_state_dict(torch.load(out / 'head.pt', map_location='cpu', weights_only=True)['model'])
    cal_score = metrics(model, cal)
    fixed = max(ARMS, key=lambda arm: (cal_score['fixed_completed'][arm], -cal_score['fixed_drops'][arm]))
    save(out / 'frozen_selection.json', dict(checkpoint_sha256=frozen_sha, fixed_arm=fixed,
        epoch=torch.load(out / 'head.pt', weights_only=True)['epoch'], calibration=cal_score))
    # Audit data is not loaded until the checkpoint and comparison are frozen.
    audit = load_split(a.data, splits['audit'], train['source_sha256'])
    audit_score = metrics(model, audit)
    passed = (audit_score['completed'] >= audit_score['fixed_completed'][fixed] + 2
        and audit_score['drops'] <= audit_score['fixed_drops'][fixed]
        and audit_score['completed'] >= audit_score['pose_completed'])
    assert sha(out / 'head.pt') == frozen_sha
    torch.save(dict(train={k: train[k] for k in ('features', 'label', 'startup_drop', 'keys')},
        calibration={k: cal[k] for k in ('features', 'label', 'startup_drop', 'keys')},
        audit={k: audit[k] for k in ('features', 'label', 'startup_drop', 'keys')}), out / 'features.pt')
    save(out / 'history.json', history)
    result = dict(status='completed', checkpoint_sha256=frozen_sha, calibration=cal_score, audit=audit_score,
        calibration_selected_fixed=fixed, offline_gate_passed=passed, audit_files=audit['files'], cloud_calls=0,
        note='No autonomous Astra evaluation or deployment. Outcome oracle uses unobserved future command and cannot be treated as an achievable policy.')
    save(out / 'result.json', result)
    save(out / 'progress.json', dict(status='completed', pid=os.getpid(), offline_gate_passed=passed, elapsed_s=time.time() - start))
    print(json.dumps({k: v for k, v in result.items() if k != 'audit_files'}, indent=2))


if __name__ == '__main__':
    main()
