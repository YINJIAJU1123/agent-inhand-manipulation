"""Hold reliability with a measured motion bypass around the frozen pose head."""
import torch
from torch import nn


def measured_motion(candidates):
    if candidates.shape[-1] != 731 or not torch.isfinite(candidates).all():
        raise ValueError('Expected finite measured frozen policy features')
    # FrozenFeatures layout: memory 512, commands 42, proprio 128, image/query 49.
    # Proprio: joints 21, velocities 21, fingertip positions 15, rotations 20,
    # fingertip velocities 30, previous normalized commands 21.
    prop = candidates[..., 554:682]
    return torch.cat((prop[..., 21:42], prop[..., 77:107],
                      prop[..., 107:128]-prop[..., :21], candidates[..., 727:731]), -1)


class HoldMotion(nn.Module):
    def __init__(self, motion):
        super().__init__()
        self.dimension = 201 if motion else 125
        self.register_buffer('mean', torch.zeros(self.dimension))
        self.register_buffer('scale', torch.ones(self.dimension))
        self.net = nn.Sequential(nn.Linear(self.dimension, 16), nn.Tanh(), nn.Dropout(.1), nn.Linear(16, 2))

    def forward(self, features):
        if features.shape[-1] != self.dimension or not torch.isfinite(features).all():
            raise ValueError('Wrong or nonfinite measured feature contract')
        y = self.net(((features-self.mean)/self.scale).clamp(-8, 8))
        return dict(reliable_logit=y[..., 0], occupancy=y[..., 1].sigmoid())
