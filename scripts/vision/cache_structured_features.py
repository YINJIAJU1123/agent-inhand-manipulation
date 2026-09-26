"""Build a deterministic image/language feature cache without a network model.

This is the offline fallback for the first student smoke run.  It keeps the
same cache contract as ``cache_vlm_features.py``: RGB statistics stand in for
the frozen image encoder and the six-dimensional face goal stands in for the
language embedding.  The student architecture and action targets are
unchanged, so a later SigLIP cache is a drop-in replacement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    image_features = []
    language_features = []
    for frame_batch, face_batch in zip(data["frames"], data["target_face"]):
        rgb = frame_batch["rgb"].to(torch.float32) / 255.0
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        n, h, w, _ = rgb.shape
        center = rgb[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        # Global and center mean/std provide a compact, deterministic visual
        # signal while retaining the exact feature-cache interface.
        visual = torch.cat((rgb.mean((1, 2)), rgb.std((1, 2)),
                            center.mean((1, 2)), center.std((1, 2))), dim=-1)
        faces = torch.as_tensor(face_batch, dtype=torch.long).reshape(-1)
        language = torch.nn.functional.one_hot(faces, num_classes=6).float()
        image_features.append(visual)
        language_features.append(language)

    result = {
        "image_features": torch.cat(image_features),
        "language_features": torch.cat(language_features),
        "student_proprio": torch.cat(data["student_proprio"]),
        "actions": torch.cat(data["actions"]),
        "teacher_actions": torch.cat(data.get("teacher_actions", data["actions"])),
        "target_face": torch.cat(data["target_face"]),
        "episode_id": torch.cat(data["episode_id"]),
        "step_index": torch.cat(data["step_index"]),
        "env_id": torch.cat(data.get("env_id", [torch.zeros_like(x) for x in data["episode_id"]])),
        "terminal": torch.cat(data.get("terminal", [torch.zeros_like(x) for x in data["episode_id"]])).bool(),
        "model": "structured_rgb_stats+face_onehot",
        "source": args.data,
        "source_checkpoint": data.get("checkpoint"),
        "source_checkpoint_sha256": data.get("checkpoint_sha256"),
        "freeze_id": data.get("freeze_id"),
        "action_storage": data.get("action_storage"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {"output": str(output), "samples": int(result["actions"].shape[0]),
              "image_dim": int(result["image_features"].shape[-1]),
              "language_dim": int(result["language_features"].shape[-1]),
              "model": result["model"]}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
