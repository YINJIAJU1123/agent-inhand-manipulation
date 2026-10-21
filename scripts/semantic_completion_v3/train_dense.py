"""Matched compact/dense completion pilots. Audit data is evaluated only at end."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parents[1]/'semantic_completion_v2'))
from assets import save, sha
from sequential import THRESHOLD_GRID, score
from dense_model import DenseCompletion, TOLERANCES, quaternion_matrix

SENSORS = ('x', 'roi_rgb', 'roi_depth', 'roi_box', 'roi_found')


def read(paths):
    shards = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    data = {key: torch.cat([shard[key] for shard in shards]) for key in (*SENSORS, 'error', 'rotation_label')}
    data['rows'] = [row for shard in shards for row in shard['rows']]
    return data


@torch.inference_mode()
def assess(model, data, device):
    model.eval()
    outputs = []
    for start in range(0, len(data['rows']), 64):
        output = model(*(data[key][start:start+64].to(device) for key in SENSORS))
        outputs.append({key: value.cpu() for key, value in output.items()})
    result = {key: torch.cat([output[key] for output in outputs]) for key in outputs[0]}
    true = data['error'] <= .16
    near = (data['error'] > .16) & (data['error'] <= .5)
    far = data['error'] > .5
    bce = F.binary_cross_entropy_with_logits(result['ordinal'][..., 2], true.float(), reduction='none')
    # A near/positive-only selection metric can ignore catastrophic distant errors.
    balanced = torch.stack([bce[mask].mean() for mask in (true, near, far)]).mean()
    target = quaternion_matrix(data['rotation_label'])
    relative_trace = (result['rotation'] * target).sum((-1, -2))
    angle = torch.acos(((relative_trace-1)/2).clamp(-1, 1))
    probabilities = result['ordinal'][..., 2].sigmoid()
    return dict(selection_loss=float(balanced),
                completion_brier=float((probabilities-true.float()).square().mean()),
                rotation_median_rad=float(angle.median()),
                rotation_mean_rad=float(angle.mean()), positives=int(true.sum())), probabilities


def frame_statistics(probabilities, data, threshold):
    accepted = probabilities >= threshold if threshold < 1 else torch.zeros_like(probabilities, dtype=torch.bool)
    true = data['error'] <= .16
    n, correct = int(accepted.sum()), int((accepted & true).sum())
    return dict(accepted=n, correct=correct, precision=correct/n if n else None,
                recall=correct/int(true.sum()) if true.any() else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=('compact', 'dense'), required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    out, root = Path(args.output), Path(args.cache)
    out.mkdir(parents=True, exist_ok=False)
    train_files = sorted(root.glob('train_*.pt'))
    calibration_file, audit_file = root/'val_96311.pt', root/'val_96312.pt'
    assert len(train_files) == 8 and calibration_file.exists() and audit_file.exists()
    train, calibration = read(train_files), read([calibration_file])
    model = DenseCompletion(dense=args.variant == 'dense').to(args.device)
    model.mean.copy_(train['x'].flatten(0, 1).mean(0))
    model.scale.copy_(train['x'].flatten(0, 1).std(0).clamp_min(.02))
    error = train['error']
    weights = 1 + 8*((error > .16) & (error < .4)).any(-1).float() + 12*(error <= .16).any(-1).float()
    dataset = TensorDataset(*(train[key] for key in (*SENSORS, 'error', 'rotation_label')))
    generator = torch.Generator().manual_seed(20261021)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)
    tolerances = torch.tensor(TOLERANCES, device=args.device)
    protocol = dict(variant=args.variant, epochs=args.epochs, train_frames=len(dataset),
        train_files=[str(path) for path in train_files], calibration_file=str(calibration_file),
        audit_file=str(audit_file), trainable_parameters=sum(p.numel() for p in model.parameters()),
        learning_rate=3e-4, weight_decay=.02, batch_size=64, sampler_seed=20261021,
        source_sha256=sha(__file__), model_source_sha256=sha(Path(__file__).with_name('dense_model.py')),
        cache_manifest_sha256=sha(root/'manifest.json'), sensor_fields=list(SENSORS),
        supervision_only=['error', 'rotation_label'], pose_auxiliary_coefficient=.25,
        selection='Lowest mean BCE across positive, near-negative, far-negative calibration groups. Audit only after checkpoint selection.',
        calibration='First stop per trajectory, calibration seed only. At least eight proposals and precision >= 0.90. Maximize correct proposals, then precision, then prefer lower threshold.',
        matched_control='Both variants have identical train/calibration data, sampled indices, losses and epoch budget. Dense adds measured RGB-D/box/visibility and model capacity. This pilot does not isolate capacity from visual information.',
        limits='Development seeds reused. Known-goal stop evidence only, no semantic or Astra task success. Passing audit requires subsequent live shadow validation.',
        tolerance_rad=.16, progress_enabled=False, cloud_calls=0, formal_astra_episodes=0)
    save(out/'protocol.json', protocol)
    history, best, started = [], float('inf'), time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for batch in loader:
            tensors = [value.to(args.device, non_blocking=True) for value in batch]
            y = model(*tensors[:len(SENSORS)])
            e, q = tensors[-2:]
            truth = (e[..., None] <= tolerances).float()
            weight = 1+3*(e <= .16).float()+2*((e > .16) & (e <= .4)).float()
            ordinal = F.binary_cross_entropy_with_logits(y['ordinal'], truth, reduction='none').mean(-1)
            distance = F.smooth_l1_loss(y['log_error'], e.clamp_min(.02).log(), reduction='none')
            loss = ((ordinal+.3*distance)*weight).sum()/weight.sum()
            positive = e <= .16
            pairs = positive[:, :, None] & ~positive[:, None, :]
            logits = y['ordinal'][..., 2]
            if pairs.any():
                loss = loss+.2*F.softplus(1-logits[:, :, None]+logits[:, None, :])[pairs].mean()
            loss = loss+.1*F.relu(y['ordinal'][..., :-1]-y['ordinal'][..., 1:]).mean()
            loss = loss+.25*(y['rotation']-quaternion_matrix(q)).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, _ = assess(model, calibration, args.device)
        row = dict(epoch=epoch, train_loss=sum(losses)/len(losses), calibration=metrics, elapsed_s=time.time()-started)
        history.append(row)
        save(out/'history.json', history)
        save(out/'progress.json', dict(status='training', pid=os.getpid(), updated=time.time(), **row))
        print(json.dumps(row), flush=True)
        if metrics['selection_loss'] < best:
            best = metrics['selection_loss']
            torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                dense=args.variant == 'dense', epoch=epoch, threshold=1., calibration=None,
                cloud_eligible=False, live_validated=False, progress_enabled=False), out/'head.pt')
    pack = torch.load(out/'head.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(pack['model'])
    calibration_metrics, probabilities = assess(model, calibration, args.device)
    curve = [dict(threshold=t, **score(probabilities, calibration, t)) for t in THRESHOLD_GRID]
    eligible = [row for row in curve if row['accepted'] >= 8 and row['precision'] >= .9]
    selected = max(eligible, key=lambda row: (row['correct'], row['precision'], -row['threshold'])) if eligible else None
    threshold = selected['threshold'] if selected else 1.
    # Never load the audit split during fitting or checkpoint/threshold selection.
    audit = read([audit_file])
    audit_metrics, audit_probabilities = assess(model, audit, args.device)
    audit_score = score(audit_probabilities, audit, threshold)
    passed = bool(selected and audit_score['accepted'] >= 8 and audit_score['precision'] >= .9
                  and (audit_score['reached_recall'] or 0) >= .3)
    pack.update(threshold=threshold, calibration=selected, audit=audit_score,
                offline_gate_passed=passed, source='dense_completion_v3',
                feature_manifest=json.loads((root/'manifest.json').read_text()))
    torch.save(pack, out/'head.pt')
    result = dict(status='needs_live_validation' if passed else 'needs_improvement', variant=args.variant,
        best_epoch=pack['epoch'], threshold=threshold, calibration=selected, calibration_curve=curve,
        calibration_metrics=calibration_metrics, audit=audit_score, audit_metrics=audit_metrics,
        audit_frames=frame_statistics(audit_probabilities, audit, threshold),
        offline_gate_passed=passed, cloud_eligible=False, live_validated=False,
        checkpoint_sha256=sha(out/'head.pt'), cloud_calls=0, formal_astra_episodes=0,
        elapsed_s=time.time()-started,
        note='Correlated development trajectories. Pose auxiliary and completion use sensors only at inference. No baseline replacement or autonomous task success claim.')
    save(out/'result.json', result)
    save(out/'progress.json', dict(status='completed', pid=None, updated=time.time(), best_epoch=pack['epoch'], result_status=result['status']))
    print(json.dumps({key: result[key] for key in ('status', 'variant', 'best_epoch', 'threshold', 'audit', 'audit_metrics')}), flush=True)


if __name__ == '__main__':
    main()
