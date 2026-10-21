"""Training-only sampling and physics checks for trajectory replay."""
import itertools
import numpy as np
import torch


def layout_bank(split):
    bank = np.asarray(list(itertools.permutations(range(6))))
    bank = bank[np.random.default_rng(31990).permutation(len(bank))]
    lo, hi = {'train': (0, 480), 'calibration': (480, 540), 'audit': (540, 600)}[split]
    return bank[lo:hi]


def snapshot(raw):
    return {key: value.detach().cpu().clone() for key, value in dict(
        joint_pos=raw.hand.data.joint_pos, joint_vel=raw.hand.data.joint_vel,
        object_root=raw.object.data.root_state_w, prev_targets=raw.prev_targets).items()}


def compare(actual, expected, tolerance=1e-6):
    if actual.keys() != expected.keys():
        raise ValueError('Physical state fields differ')
    errors = {}
    for key, reference in expected.items():
        value = actual[key]
        if value.shape != reference.shape or not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise ValueError('Invalid physical state: ' + key)
        errors[key] = float((value - reference).abs().max())
    if max(errors.values()) > tolerance:
        raise ValueError('Replay physical state diverged: ' + str(errors))
    return errors


def projected_boxes(points, resolution=512):
    """Supervision boxes only. Never called by a deployed RGB model."""
    points = torch.as_tensor(points)
    lower, upper = points.min(2).values, points.max(2).values
    center = (lower + upper) / 2
    extent = (upper - lower) * 1.7 + 4
    lower = (center - extent / 2).clamp(0, resolution - 1)
    upper = (center + extent / 2).clamp(1, resolution)
    return torch.cat(((lower + upper) / 2, (upper - lower).clamp_min(1)), -1) / resolution
