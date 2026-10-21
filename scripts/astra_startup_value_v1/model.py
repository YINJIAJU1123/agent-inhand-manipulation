"""Predict eventual local skill completion from measured startup features.

Forward accepts only the seven candidate feature vectors. Neither the future
goal, actual future latency nor simulator labels are available at startup.
"""
import torch
from torch import nn

ARMS = tuple([f'skill{i}' for i in range(6)] + ['latch'])
FEATURE_DIM = 168


class StartupValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(FEATURE_DIM))
        self.register_buffer('scale', torch.ones(FEATURE_DIM))
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM * 2, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(.15),
            nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 4))

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (7, FEATURE_DIM):
            raise ValueError('Expected seven measured 168-dimensional candidates')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite startup features')
        x = ((features - self.mean) / self.scale).clamp(-8, 8)
        context = x.mean(1, keepdim=True).expand_as(x)
        y = self.net(torch.cat((x, context), dim=-1))
        probability = y[..., :3].softmax(-1)
        return dict(outcome_logits=y[..., :3], startup_drop_logit=y[..., 3],
            # Classes: alive but unfinished, completed, dropped before completion.
            completion_probability=probability[..., 1], drop_probability=probability[..., 2])

    @torch.inference_mode()
    def choose(self, features):
        # Survival is auxiliary supervision, never a substitute for completion.
        return self(features)['completion_probability'].argmax(-1)
