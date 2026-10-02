"""Small deterministic text feature fallback for language-conditioned control.

This is a versioned fallback for hosts without a cached CLIP/SigLIP model. It
uses signed hashed word and character n-grams, so the policy consumes the
actual instruction string and can be evaluated on paraphrases. It is not a
semantic pretrained language model; the feature contract is intentionally
replaceable by a frozen text encoder later.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import torch

HASH_TEXT_FEATURE_DIM = 64
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _bucket(key: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return value % dim, (1.0 if (value >> 63) == 0 else -1.0)

def hashed_text_features(texts: Iterable[str], dim: int = HASH_TEXT_FEATURE_DIM) -> torch.Tensor:
    """Encode instruction strings into deterministic normalized vectors."""
    if dim <= 0:
        raise ValueError("text feature dimension must be positive")
    rows = []
    for raw in texts:
        text = " ".join(str(raw).lower().split())
        values = torch.zeros(dim, dtype=torch.float32)
        tokens = _TOKEN_RE.findall(text)
        for token in tokens:
            index, sign = _bucket("word:" + token, dim)
            values[index] += sign
        compact = "_" + "_".join(tokens) + "_"
        for start in range(max(0, len(compact) - 2)):
            index, sign = _bucket("char:" + compact[start : start + 3], dim)
            values[index] += 0.5 * sign
        rows.append(values)
    output = torch.stack(rows) if rows else torch.empty((0, dim), dtype=torch.float32)
    return torch.nn.functional.normalize(output, dim=-1)
