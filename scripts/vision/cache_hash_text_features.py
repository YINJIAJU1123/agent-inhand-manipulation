"""Replace structured face one-hot goals with deterministic text-string features.

This is a reproducible fallback when a frozen CLIP/SigLIP checkpoint is not
available. It preserves the visual/action data and changes only the language
feature contract. The resulting model must be described as hash-text, not as a
pretrained language encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "BrainCo_DexHand"))
from BrainCo_DexHand.algo.agentic.language_goal import FACE_NAMES  # noqa: E402
from BrainCo_DexHand.algo.agentic.text_features import HASH_TEXT_FEATURE_DIM, hashed_text_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, default=HASH_TEXT_FEATURE_DIM)
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    face = data["target_face"].reshape(-1).long()
    prompts = [f"show the {FACE_NAMES[int(i)]} marker" for i in face]
    result = dict(data)
    result["language_features"] = hashed_text_features(prompts, dim=args.dim)
    result["language_mode"] = "hash"
    result["language_dim"] = args.dim
    result["model"] = str(data.get("model", "")) + "+hash_text64"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {
        "output": str(output),
        "samples": int(result["actions"].shape[0]),
        "image_dim": int(result["image_features"].shape[-1]),
        "language_dim": int(result["language_features"].shape[-1]),
        "language_mode": "hash",
        "model": result["model"],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
