"""Learn six-second hold reliability from existing independent physical branches.

These datasets and audit splits have already been inspected in development.
This is a new candidate check, not an untouched final benchmark or deployment.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from model import HoldEvidence

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / 'outputs/astra_retention_v1/learning_20261022/train',
           ROOT / 'outputs/astra_retention_cloud_20261022/retrieved_train_r1']
THRESHOLDS = (.5, .6, .7, .8, .85, .9, .95, .975, .99)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def score(probability, threshold, data):
    accepted = probability >= threshold
    reliable = data['reliable'].bool()
    n = int(accepted.sum())
    correct = int((accepted & reliable).sum())
    return dict(threshold=threshold, states=len(accepted), accepted=n, reliable=correct,
        unsafe=n-correct, precision=correct/n if n else None,
        accepted_mean_occupancy=float(data['occupancy'][accepted].mean()) if n else None,
        accepted_drops=int(data['drop'][accepted].sum()), choices=accepted.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(202610222)
    packs = [torch.load(path / 'features.pt', map_location='cpu', weights_only=True) for path in SOURCES]
    protocols = [json.loads((path / 'protocol.json').read_text()) for path in SOURCES]
    encoder = sha(ROOT / 'outputs/semantic_completion_v3/training_20261021/dense/head.pt')
    if any(tuple(p['rates']) != (1., .25, 0.) or p['frozen_evidence_sha256'] != encoder for p in protocols):
        raise RuntimeError('Mixed feature encoders or action definitions')
    data, keys = {}, {}
    for split in ('train', 'calibration', 'audit'):
        features = torch.cat([p['data'][split]['feature'] for p in packs])
        quality = torch.cat([p['data'][split]['quality'][:, 2] for p in packs])
        drop = torch.cat([p['data'][split]['drop'][:, 2] for p in packs])
        occupancy = quality + drop
        if not torch.isfinite(features).all() or (occupancy < -1e-6).any() or (occupancy > 1.000001).any():
            raise RuntimeError('Invalid saved features or hold labels')
        data[split] = dict(feature=features, reliable=((quality >= .95) & (drop == 0)).float(),
                           occupancy=occupancy.clamp(0, 1), drop=drop)
        keys[split] = sum([p['keys'][split] for p in packs], [])
    sets = [{r['seed'] for r in keys[s]} for s in ('train', 'calibration', 'audit')]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError('Seed overlap between splits')
    save(out / 'protocol.json', dict(source_features={str(p):sha(p / 'features.pt') for p in SOURCES},
        encoder_sha256=encoder, model_sha256=sha(Path(__file__).with_name('model.py')), trainer_sha256=sha(__file__),
        split_seeds={k: sorted({r['seed'] for r in v}) for k, v in keys.items()},
        states={k: len(v['feature']) for k, v in data.items()},
        target='At least 95% of six seconds at <=0.16rad and no drop under hold.',
        checkpoint='Minimum calibration binary cross entropy across 100 fixed epochs.',
        threshold='Maximum calibration coverage with >=90% precision and at least eight accepted states.',
        audit_gate='At least eight accepted states, >=90% precision, fewer unsafe accepts than original p>=0.7, and at least 80% of its reliable accepts retained.',
        limitation='Previously inspected development datasets. New candidate, no untouched-test claim, no deployment.',
        thresholds=THRESHOLDS, cloud_calls=0))
    model = HoldEvidence()
    train, cal = data['train'], data['calibration']
    model.mean.copy_(train['feature'].mean(0))
    model.scale.copy_(train['feature'].std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    best, history, started = float('inf'), [], time.time()
    for epoch in range(100):
        model.train()
        losses = []
        for indices in torch.randperm(len(train['feature'])).split(32):
            y = model(train['feature'][indices])
            loss = F.binary_cross_entropy_with_logits(y['reliable_logit'], train['reliable'][indices])
            loss += .5 * F.smooth_l1_loss(y['occupancy'], train['occupancy'][indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = float(F.binary_cross_entropy_with_logits(model(cal['feature'])['reliable_logit'], cal['reliable']))
        history.append(dict(epoch=epoch+1, loss=sum(losses)/len(losses), calibration_bce=validation))
        save(out / 'progress.json', dict(status='training', pid=os.getpid(), **history[-1], elapsed_s=time.time()-started))
        if validation < best:
            best = validation
            torch.save(dict(model={k:v.clone() for k,v in model.state_dict().items()}, epoch=epoch+1,
                encoder_sha256=encoder, label='six-second hold reliability', deployment_eligible=False), out / 'head.pt')
    model.load_state_dict(torch.load(out / 'head.pt', map_location='cpu', weights_only=True)['model'])
    model.eval()
    with torch.no_grad():
        probabilities = {k:model(v['feature'])['reliable_logit'].sigmoid() for k,v in data.items()}
    curve = [score(probabilities['calibration'], t, cal) for t in THRESHOLDS]
    eligible = [r for r in curve if r['accepted'] >= 8 and r['precision'] >= .9]
    selected = max(eligible, key=lambda r:(r['accepted'], r['precision'])) if eligible else None
    save(out / 'calibration.json', dict(selected=selected, curve=curve, checkpoint_sha256=sha(out / 'head.pt')))
    audit = data['audit']
    baseline = score(audit['feature'][:, 98].sigmoid(), .7, audit)
    measured = score(probabilities['audit'], selected['threshold'], audit) if selected else None
    passed = bool(measured and measured['accepted'] >= 8 and measured['precision'] >= .9
        and measured['unsafe'] < baseline['unsafe'] and measured['reliable'] >= .8*baseline['reliable'])
    result = dict(status='completed', calibration=selected, audit=measured, baseline_audit=baseline,
        offline_gate_passed=passed, checkpoint_sha256=sha(out / 'head.pt'),
        epoch=torch.load(out / 'head.pt', weights_only=True)['epoch'], cloud_calls=0,
        note='Goal-relative local hold labels only, no semantic correctness or Astra success claim. No automatic deployment.')
    torch.save(dict(probabilities=probabilities, keys=keys), out / 'predictions.pt')
    save(out / 'history.json', history)
    save(out / 'result.json', result)
    save(out / 'progress.json', dict(status='completed', pid=os.getpid(), offline_gate_passed=passed, elapsed_s=time.time()-started))
    print(json.dumps({k:({x:y for x,y in v.items() if x!='choices'} if isinstance(v,dict) else v) for k,v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
