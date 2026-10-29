"""Audit explicit goal conditioning on saved validation states, without rollouts.

Changing only the actor's goal input holds the student's proposal and belief
fixed. This describes local input dependence, not an online component ablation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch

from control_modules import FIELDS, load_maintenance_actor
from train_modules import load_data


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def evaluate(actor, bank, sensor, goals, proposal, current, pending, target):
    names = ('matched_goal', 'goal_center', 'mean_of_five_wrong_canonical_goals')
    sums = {name: dict(teacher_mse=0., action_change_mse=0.) for name in names}
    total = 0
    for ids in torch.arange(len(goals)).split(1024):
        x, g, p, c, wait, y = (value[ids] for value in (sensor, goals, proposal, current, pending, target))
        matched = actor(x, bank[g], p, c, wait)['action']
        center = actor(x, bank.mean(0).expand(len(ids), -1), p, c, wait)['action']
        mismatch_error = mismatch_change = 0.
        for shift in range(1, len(bank)):
            altered = actor(x, bank[(g + shift) % len(bank)], p, c, wait)['action']
            mismatch_error += (altered - y).square().mean(-1).sum().item() / (len(bank) - 1)
            mismatch_change += (altered - matched).square().mean(-1).sum().item() / (len(bank) - 1)
        sums[names[0]]['teacher_mse'] += (matched - y).square().mean(-1).sum().item()
        sums[names[1]]['teacher_mse'] += (center - y).square().mean(-1).sum().item()
        sums[names[1]]['action_change_mse'] += (center - matched).square().mean(-1).sum().item()
        sums[names[2]]['teacher_mse'] += mismatch_error
        sums[names[2]]['action_change_mse'] += mismatch_change
        total += len(ids)
    baseline = torch.where(pending[:, None], current + .15 * (proposal.clamp(-1, 1) - current), proposal.clamp(-1, 1))
    return dict(queries=total, baseline_teacher_mse=(baseline - target).square().mean().item(),
                variants={name: {key: value / total for key, value in result.items()} for name, result in sums.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--actor', action='append', required=True, metavar='NAME=CHECKPOINT')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest, data = load_data(args.data)
    if manifest['seed'] in (183101, 183102):
        raise ValueError('Formal data cannot be used for this development diagnostic')
    validity = args.data.parent / (args.data.name + '.validity.json')
    if not json.loads(validity.read_text())['physical_log_valid']:
        raise ValueError('Validation capture has invalid physics')
    pair_manifest = json.loads((args.pairs / 'manifest.json').read_text())
    if not pair_manifest['completed']:
        raise ValueError('Paired features incomplete')
    paired = torch.load(args.pairs / 'validation.pt', map_location='cpu', weights_only=True)
    if paired['source_manifest_sha256'] != sha(args.data / 'manifest.json'):
        raise ValueError('Paired features do not match the validation capture')
    for key in ('sim_step', 'env_id', 'episode_id', 'goal_index'):
        if not torch.equal(paired[key], data[key]):
            raise ValueError('Paired state alignment differs: ' + key)
    elapsed_offset = sum(data[key].shape[-1] if data[key].ndim > 1 else 1
                         for key in FIELDS[:FIELDS.index('elapsed_pending_s')])
    sensor = data['sensor'].clone()
    sensor[~data['pending'], elapsed_offset] = 0.
    groups = {name: mask.nonzero().flatten() for name, mask in (
        ('waiting', data['pending']),
        ('unbound_execution', ~data['pending'] & ~data['known']),
        ('bound_execution', ~data['pending'] & data['known']))}
    specs = dict(item.split('=', 1) for item in args.actor)
    first = torch.load(next(iter(specs.values())), map_location='cpu', weights_only=True)
    bank = first['bank']
    results = {}
    for name, path in specs.items():
        actor = load_maintenance_actor(Path(path), manifest['checkpoint_sha256'], bank, 'cpu')
        observed = {}; counterfactual = {}
        for group, ids in groups.items():
            goals = data['goal_index'][ids]
            target = torch.where(data['pending'][ids, None], data['teacher_hold_action'][ids],
                                 data['teacher_actions'][ids, goals]).clamp(-1, 1)
            observed[group] = evaluate(actor, bank, sensor[ids], goals, data['proposal'][ids],
                                       data['proprio'][ids, :21].clamp(-1, 1), data['pending'][ids], target)
            if group == 'waiting':
                continue
            x = paired['features'][ids].reshape(-1, sensor.shape[1]).clone()
            x[:, elapsed_offset] = 0.
            goals = torch.arange(len(bank)).repeat(len(ids))
            proposal = paired['proposals'][ids].reshape(-1, 21)
            current = data['proprio'][ids, :21].clamp(-1, 1).repeat_interleave(len(bank), 0)
            pending = torch.zeros(len(goals), dtype=torch.bool)
            target = data['teacher_actions'][ids].reshape(-1, 21).clamp(-1, 1)
            counterfactual[group] = evaluate(actor, bank, x, goals, proposal, current, pending, target)
        results[name] = dict(checkpoint_sha256=sha(Path(path)), observed=observed, paired_execution=counterfactual)
        print(json.dumps(dict(actor=name, observed=observed, paired_execution=counterfactual)), flush=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope='CPU analysis of frozen models on correlated validation states. Not an online ablation or semantic task success rate.',
                  validation_seed=manifest['seed'], physical_states=len(sensor),
                  states_per_phase={name: len(ids) for name, ids in groups.items()}, results=results,
                  source_sha256={name: sha(Path(__file__).with_name(name))
                                 for name in ('analyze_controller_inputs.py', 'control_modules.py', 'train_modules.py')},
                  validation_manifest_sha256=sha(args.data / 'manifest.json'),
                  paired_features_sha256=sha(args.pairs / 'validation.pt'),
                  frozen_teacher_sha256=manifest['teacher_sha256'], formal_trials_used=False,
                  cautions=[
                      'Wrong-goal interventions deliberately mismatch explicit actor goals with otherwise fixed student features. They are not retrained baselines.',
                      'Paired execution uses six alternative goals from the same sensor state and pre-decision recurrent history. These queries are correlated.',
                      'Goal information also enters the student proposal and belief, so this does not remove all goal information.',
                      'The actor sees measured pose and student belief, not raw RGB. Ground-truth pose and teacher actions are used only for offline labels.',
                      'No rollout, live-model request, checkpoint update or final actor selection is performed.'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
