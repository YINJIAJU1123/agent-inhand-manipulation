"""Measured-state evidence that holding will preserve goal alignment."""
import torch
from torch import nn


class HoldEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(125))
        self.register_buffer('scale', torch.ones(125))
        self.net = nn.Sequential(nn.Linear(125, 16), nn.Tanh(), nn.Dropout(.1), nn.Linear(16, 2))

    def forward(self, features):
        if features.shape[-1] != 125 or not torch.isfinite(features).all():
            raise ValueError('Expected finite measured retention features')
        y = self.net(((features - self.mean) / self.scale).clamp(-8, 8))
        return dict(reliable_logit=y[..., 0], occupancy=y[..., 1].sigmoid())
