"""Create an audited, UNTRAINED latent student initialization from a release.

This is an artifact preparation command, not a simulator or performance test.
The source checkpoint and its optimizer are never modified.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from model import LatentStudent, PROP_KEEP, configure_training, migrate
from viserdex.distillation.model import BeliefStudent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=182801)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(4)
    payload = torch.load(args.source, map_location='cpu', weights_only=False)
    student, report = migrate(payload, args.seed)
    reference = BeliefStudent(student.original_cfg).eval()
    reference.load_state_dict(payload['student'], strict=True)
    student.eval()

    # Center removed target channels at their saved means so their normalized
    # contribution is exactly zero. This audits remapping, not policy quality.
    prop = reference.prop_norm.mean.expand(4, -1).clone()
    prop[:, PROP_KEEP] += .01 * torch.randn(4, 109)
    pose = reference.extero_norm.mean.expand(4, -1).clone()
    pose[:, :7] += .01 * torch.randn(4, 7)
    latent = torch.randn(4, 768)
    hidden = torch.randn_like(reference.initial_state(4, 'cpu')) * .01
    with torch.no_grad():
        expected, _, expected_hidden = reference(prop, pose, hidden)
        actual, _, actual_hidden = student(prop[:, PROP_KEEP], pose[:, :7], latent, hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-5, atol=2e-6)
    report['neutralized_goal_action_max_abs_error'] = float((actual - expected).abs().max())
    report['neutralized_goal_hidden_max_abs_error'] = float((actual_hidden - expected_hidden).abs().max())

    # Exercise the actual warmup optimizer on a disposable copy. Adam without
    # weight decay preserves inherited columns under the gradient masks.
    probe = deepcopy(student)
    hooks = configure_training(probe, 'interface')
    optimizer = torch.optim.Adam([p for p in probe.parameters() if p.requires_grad], lr=1e-3)
    before = {name: p.detach().clone() for name, p in probe.named_parameters()}
    for _ in range(3):
        prediction, _, _ = probe(prop[:, PROP_KEEP], pose[:, :7], latent, hidden)
        loss = (prediction - (expected + .1)).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    changed = []
    for name, parameter in probe.named_parameters():
        delta = parameter.detach() - before[name]
        if name in ('core.rnn.weight_ih_l0', 'core.actor.0.weight'):
            assert torch.count_nonzero(delta[:, :109]) == 0
            assert torch.count_nonzero(delta[:, 109 + probe.goal_width:]) == 0
            assert torch.count_nonzero(delta[:, 109:109 + probe.goal_width]) > 0
        elif not name.startswith('goal_encoder.'):
            assert torch.count_nonzero(delta) == 0, name
        if torch.count_nonzero(delta):
            changed.append(name)
    assert 'goal_encoder.weight' in changed
    for hook in hooks:
        hook.remove()
    report['interface_optimizer_changed_parameters'] = changed
    report['inherited_coefficients_unchanged_during_interface_probe'] = True
    report['probe_is_not_saved_as_trained_checkpoint'] = True

    # Sequence API must give the same actions as repeated calls with the same
    # episode resets. Training uses this API for recurrent distillation.
    ps = prop[:, PROP_KEEP].unsqueeze(0).repeat(3, 1, 1)
    xs = pose[:, :7].unsqueeze(0).repeat(3, 1, 1)
    zs = latent.unsqueeze(0).repeat(3, 1, 1)
    resets = torch.zeros(3, 4, dtype=torch.bool)
    resets[0, 0] = True
    resets[1, 2] = True
    with torch.no_grad():
        sequence_actions, _, sequence_hidden = student.sequence(ps, xs, zs, hidden, resets)
        h, actions = hidden.clone(), []
        for t in range(3):
            h *= (~resets[t])[None, :, None]
            action, _, h = student(ps[t], xs[t], zs[t], h)
            actions.append(action)
    torch.testing.assert_close(sequence_actions, torch.stack(actions), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(sequence_hidden, h, rtol=1e-5, atol=2e-6)
    report['reset_aware_sequence_verified'] = True
    report['source_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    report['source_iteration'] = payload.get('iteration')
    report['seed'] = args.seed
    report['status'] = 'audited_initialization_not_trained'
    args.output.mkdir(parents=True)
    checkpoint = args.output / 'latent_student_initial.pt'
    torch.save({
        'format': 'pose_latent_student_v1', 'stage': 'initialization',
        'completed_updates': 0, 'model': student.state_dict(),
        'original_student_cfg': asdict(student.original_cfg),
        'latent_dim': student.latent_dim, 'goal_width': student.goal_width,
        'source_metadata': payload['metadata'], 'noise_cfg': payload['noise_cfg'],
        'migration': report,
    }, checkpoint)
    restored = LatentStudent(student.original_cfg)
    restored.load_state_dict(torch.load(checkpoint, weights_only=False)['model'], strict=True)
    report['initial_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (args.output / 'migration_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
