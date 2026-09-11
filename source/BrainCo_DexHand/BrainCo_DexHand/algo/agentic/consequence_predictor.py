"""Action-consequence prediction and scoring for agentic reorientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ConsequenceBatch:
    """Training batch for candidate action chunks.

    ``state`` is a policy/memory latent ``[B,state_dim]``.  Candidates have
    shape ``[B,K,H,action_dim]`` where K is the number of short action chunks
    proposed by the controller and H is their horizon.  Targets are
    ``[B,K,4]`` = (delta visibility, delta occlusion, drop probability,
    rotation progress), all continuous values in [0,1] except the deltas.
    ``mask`` optionally marks valid candidates as ``[B,K]``.
    """

    state: Tensor
    candidate_actions: Tensor
    targets: Optional[Tensor] = None
    mask: Optional[Tensor] = None

    def validate(self, target_dim: int = 4) -> None:
        if self.state.ndim != 2 or self.candidate_actions.ndim != 4:
            raise ValueError("state must be [B,D], actions must be [B,K,H,A]")
        b, k = self.candidate_actions.shape[:2]
        if self.state.shape[0] != b:
            raise ValueError("state and candidate_actions batch dimensions differ")
        if self.targets is not None and self.targets.shape != (b, k, target_dim):
            raise ValueError(f"targets must be [B,K,{target_dim}]")
        if self.mask is not None and self.mask.shape != (b, k):
            raise ValueError("mask must be [B,K]")


class ConsequencePredictor(nn.Module):
    """Predict visual progress and safety for short candidate action chunks."""

    target_names = ("delta_visibility", "delta_occlusion", "drop_probability", "rotation_progress")

    def __init__(self, state_dim: int, action_dim: int = 21, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.SiLU())
        self.action_encoder = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim), nn.SiLU())
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(self.target_names))
        )

    def forward(self, batch: ConsequenceBatch | None = None, *, state: Tensor | None = None, candidate_actions: Tensor | None = None) -> Tensor:
        if batch is not None:
            batch.validate()
            state, candidate_actions = batch.state, batch.candidate_actions
        if state is None or candidate_actions is None:
            raise ValueError("provide batch or state and candidate_actions")
        if candidate_actions.ndim != 4 or state.ndim != 2:
            raise ValueError("state must be [B,D], candidate_actions [B,K,H,A]")
        b, k, h, a = candidate_actions.shape
        if a != self.action_dim or state.shape[0] != b:
            raise ValueError("candidate action/state dimensions do not match predictor")
        action_tokens = self.action_encoder(candidate_actions.reshape(b * k, h, a))
        encoded, _ = self.temporal(action_tokens)
        action_summary = encoded[:, -1].reshape(b, k, -1)
        state_summary = self.state_encoder(state)[:, None, :].expand(-1, k, -1)
        # Deltas may be negative, while probabilities/progress stay [0,1].
        raw = self.head(torch.cat((state_summary, action_summary), dim=-1))
        raw[..., 2:] = raw[..., 2:].sigmoid()
        return raw


def consequence_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None, weights: Tensor | None = None) -> Tensor:
    """Masked Smooth-L1 loss used with simulator-derived consequence labels."""

    if prediction.shape != target.shape or prediction.shape[-1] != 4:
        raise ValueError("prediction and target must have shape [B,K,4]")
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    if weights is not None:
        error = error * weights.to(device=error.device, dtype=error.dtype).view(1, 1, -1)
    error = error.mean(dim=-1)
    if mask is not None:
        mask = mask.to(device=error.device, dtype=error.dtype)
        return (error * mask).sum() / mask.sum().clamp_min(1.0)
    return error.mean()


def score_action_chunks(prediction: Tensor, *, visibility_weight: float = 1.0, occlusion_weight: float = 0.5, drop_weight: float = 2.0, rotation_weight: float = 0.25) -> Tensor:
    """Convert predictions to higher-is-better agentic action scores.

    A candidate is preferred when visibility increases, occlusion decreases,
    drop risk is low and rotation progresses.  This function is intentionally
    differentiable, so scores can bias a policy distribution during PPO.
    """

    if prediction.shape[-1] != 4:
        raise ValueError("prediction last dimension must be 4")
    return (
        visibility_weight * prediction[..., 0]
        - occlusion_weight * prediction[..., 1]
        - drop_weight * prediction[..., 2]
        + rotation_weight * prediction[..., 3]
    )
