"""Visual-language student that preserves Revo3's 21-D action interface.

This is deliberately a small fusion policy rather than a large VLM.  A
frozen CLIP/SigLIP (or another encoder) can provide ``rgb_features`` and
``language_features``; the GRU below owns temporal memory and the action
head.  The same forward pass exposes auxiliary evidence predictions used for
supervised pre-training and PPO diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class VisualStudentBatch:
    """Batch contract for exported RevoLab trajectories.

    Tensors use batch-first layout.  ``rgb_features`` is normally produced by
    a frozen image encoder and has shape ``[B, T, rgb_dim]``.  Language can be
    static per episode (``[B, language_dim]``) or time-aligned
    (``[B, T, language_dim]``).  ``touch`` and ``action_history`` are
    optional, but when present must have the same ``B,T`` dimensions.
    ``valid_mask`` marks real history entries for padded sequences.
    """

    rgb_features: Tensor
    language_features: Tensor
    proprio: Tensor
    touch: Optional[Tensor] = None
    action_history: Optional[Tensor] = None
    valid_mask: Optional[Tensor] = None

    def validate(self) -> None:
        if self.rgb_features.ndim != 3 or self.proprio.ndim != 3:
            raise ValueError("rgb_features and proprio must be [B,T,D]")
        b, t, _ = self.rgb_features.shape
        if self.proprio.shape[:2] != (b, t):
            raise ValueError("proprio must share B,T with rgb_features")
        if self.language_features.ndim == 2:
            if self.language_features.shape[0] != b:
                raise ValueError("language batch dimension does not match")
        elif self.language_features.ndim == 3:
            if self.language_features.shape[:2] != (b, t):
                raise ValueError("time-aligned language must share B,T")
        else:
            raise ValueError("language_features must be [B,D] or [B,T,D]")
        for name, value in (("touch", self.touch), ("action_history", self.action_history)):
            if value is not None and value.shape[:2] != (b, t):
                raise ValueError(f"{name} must share B,T with rgb_features")
        if self.valid_mask is not None and self.valid_mask.shape != (b, t):
            raise ValueError("valid_mask must be [B,T]")


class VisualLanguageStudent(nn.Module):
    """Compact image-language-proprioception policy with evidence heads.

    ``action`` is bounded to [-1, 1] and has shape ``[B, action_dim]``.  It is
    intended to be passed through the existing RevoLab action scaling, so the
    default ``action_dim=21`` does not alter the low-level interface.
    """

    def __init__(
        self,
        rgb_dim: int,
        language_dim: int,
        proprio_dim: int,
        action_dim: int = 21,
        touch_dim: int = 0,
        action_history_dim: int = 0,
        hidden_dim: int = 256,
        memory_layers: int = 1,
    ) -> None:
        super().__init__()
        if min(rgb_dim, language_dim, proprio_dim, action_dim, hidden_dim) <= 0:
            raise ValueError("feature, action and hidden dimensions must be positive")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rgb_proj = nn.Sequential(nn.LayerNorm(rgb_dim), nn.Linear(rgb_dim, hidden_dim), nn.SiLU())
        self.lang_proj = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, hidden_dim), nn.SiLU())
        self.proprio_proj = nn.Sequential(nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, hidden_dim), nn.SiLU())
        self.touch_proj = (
            nn.Sequential(nn.LayerNorm(touch_dim), nn.Linear(touch_dim, hidden_dim), nn.SiLU())
            if touch_dim
            else None
        )
        self.action_hist_proj = (
            nn.Sequential(
                nn.LayerNorm(action_history_dim), nn.Linear(action_history_dim, hidden_dim), nn.SiLU()
            )
            if action_history_dim
            else None
        )
        self.memory = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=memory_layers,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim)
        )
        # Auxiliary labels are [visibility, occlusion, confidence].  Keeping
        # these in the policy gives the agentic layer a differentiable signal.
        self.evidence_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 3))
        self.stop_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))

    def _language_sequence(self, language: Tensor, t: int) -> Tensor:
        if language.ndim == 2:
            return language[:, None, :].expand(-1, t, -1)
        if language.ndim == 3:
            return language
        raise ValueError("language_features must be [B,D] or [B,T,D]")

    def forward(self, batch: VisualStudentBatch | None = None, **kwargs: Tensor) -> dict[str, Tensor]:
        if batch is None:
            batch = VisualStudentBatch(**kwargs)
        batch.validate()
        rgb, proprio = batch.rgb_features, batch.proprio
        b, t, _ = rgb.shape
        x = self.rgb_proj(rgb) + self.lang_proj(self._language_sequence(batch.language_features, t))
        x = x + self.proprio_proj(proprio)
        if batch.touch is not None:
            if self.touch_proj is None:
                raise ValueError("touch was supplied but touch_dim=0")
            x = x + self.touch_proj(batch.touch)
        if batch.action_history is not None:
            if self.action_hist_proj is None:
                raise ValueError("action_history was supplied but action_history_dim=0")
            x = x + self.action_hist_proj(batch.action_history)
        x = self.fusion_norm(x)
        if batch.valid_mask is not None:
            # Zero padded entries before recurrent memory.  This preserves a
            # simple, stable interface for replay buffers with left padding.
            x = x * batch.valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        memory, _ = self.memory(x)
        if batch.valid_mask is None:
            last = memory[:, -1]
        else:
            lengths = batch.valid_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
            last = memory[torch.arange(b, device=memory.device), lengths]
        action = torch.tanh(self.action_head(last))
        evidence_logits = self.evidence_head(last)
        stop_logit = self.stop_head(last).squeeze(-1)
        return {
            "action": action,
            "evidence_logits": evidence_logits,
            "visibility": evidence_logits[:, 0].sigmoid(),
            "occlusion": evidence_logits[:, 1].sigmoid(),
            "confidence": evidence_logits[:, 2].sigmoid(),
            "stop_logit": stop_logit,
            "memory": memory,
        }
