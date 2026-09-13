"""Small, deployment-independent language goal contract for Revo3.

The first teacher does not need a large VLM.  It receives the structured goal
that a language parser would produce, while the visual student can later
replace this vector with a frozen CLIP/SigLIP text embedding without changing
the action head.
"""

from __future__ import annotations

from typing import Iterable

import torch

FACE_NAMES = ("red", "green", "blue", "yellow", "magenta", "cyan")
LANGUAGE_GOAL_DIM = len(FACE_NAMES)


def face_name(face: int) -> str:
    """Return the canonical English name for a cube face id."""
    if not 0 <= int(face) < len(FACE_NAMES):
        raise ValueError(f"face id must be in [0, {len(FACE_NAMES) - 1}]")
    return FACE_NAMES[int(face)]


def instruction(face: int, keep_visible: bool = False) -> str:
    """Render one of the language templates used by data collection."""
    text = f"show the {face_name(face)} marker"
    return text + " and keep it visible" if keep_visible else text


def encode_face_goal(face: torch.Tensor | Iterable[int] | int, *, device=None) -> torch.Tensor:
    """Encode face ids as the stable six-dimensional teacher contract.

    Keeping this helper independent of Isaac Lab makes the exact same goal
    representation available to PPO, rollout collection and the student.
    """
    ids = torch.as_tensor(face, dtype=torch.long, device=device)
    if torch.any((ids < 0) | (ids >= LANGUAGE_GOAL_DIM)):
        raise ValueError("face ids are outside the six-face vocabulary")
    return torch.nn.functional.one_hot(ids, num_classes=LANGUAGE_GOAL_DIM).to(torch.float32)


def batch_instructions(face_ids: torch.Tensor, keep_visible: torch.Tensor | None = None) -> list[str]:
    """Render instructions for logging and reproducible rollout metadata."""
    ids = face_ids.detach().to(device="cpu").reshape(-1).tolist()
    if keep_visible is None:
        keep = [False] * len(ids)
    else:
        keep = keep_visible.detach().to(device="cpu").reshape(-1).bool().tolist()
    return [instruction(face, hold) for face, hold in zip(ids, keep)]

