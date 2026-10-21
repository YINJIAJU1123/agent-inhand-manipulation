"""Unused permutations for frozen-weight validation, never training."""
import itertools
import numpy as np


def layout_bank(split):
    if split != 'fresh':
        raise ValueError('This collector only uses the reserved permutation bank')
    bank = np.asarray(list(itertools.permutations(range(6))))
    return bank[np.random.default_rng(31990).permutation(len(bank))][600:720]


def snapshot(raw):
    return {key:value.detach().cpu().clone() for key,value in dict(
        joint_pos=raw.hand.data.joint_pos,joint_vel=raw.hand.data.joint_vel,
        object_root=raw.object.data.root_state_w,prev_targets=raw.prev_targets).items()}
