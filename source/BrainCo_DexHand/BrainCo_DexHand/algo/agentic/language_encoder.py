"""Language feature adapters used by the Revo3 teacher/student pipeline.

The control policy only depends on a fixed feature contract.  During the
teacher stage the input is the six-dimensional structured face goal.  During
student training the same adapter can consume cached CLIP/SigLIP text
features, avoiding a large language model in the high-frequency action loop.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StructuredGoalEncoder(nn.Module):
    """Project the six-dimensional face goal into the policy hidden space."""

    def __init__(self, goal_dim: int = 6, output_dim: int = 128) -> None:
        super().__init__()
        if goal_dim <= 0 or output_dim <= 0:
            raise ValueError("goal_dim and output_dim must be positive")
        self.goal_dim = goal_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, goal: Tensor) -> Tensor:
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"expected goal features with last dimension {self.goal_dim}")
        return self.net(goal)


class FrozenTextFeatureAdapter(nn.Module):
    """Adapt cached CLIP/SigLIP text features without updating the encoder.

    The upstream text encoder is intentionally outside this module.  A
    rollout can cache one embedding per instruction and train this small
    projection jointly with the action policy.  This keeps the experiment
    reproducible even when the VLM service or HuggingFace weights are absent.
    """

    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"expected text features with last dimension {self.input_dim}")
        return self.net(features)

