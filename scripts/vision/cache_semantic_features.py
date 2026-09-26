"""Cache compact marker-aware RGB features for a visual student baseline.

This is a deterministic fallback for hosts where the frozen SigLIP checkpoint
cannot be downloaded.  It keeps the same feature-cache contract as the VLM
path, while exposing the colored face marker's mass, centroid and spread so a
student can tell which face is visible and where it is in the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


FACE_COLORS = torch.tensor(
    [[0.85, 0.05, 0.05], [0.05, 0.75, 0.15], [0.05, 0.25, 0.90],
     [0.95, 0.75, 0.05], [0.80, 0.05, 0.75], [0.05, 0.80, 0.85]],
    dtype=torch.float32,
)


def semantic_rgb_features(rgb: torch.Tensor) -> torch.Tensor:
    """Return the 36-D feature used by both offline and live evaluation."""
    rgb = rgb.to(torch.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    n, height, width, _ = rgb.shape
    pixels = rgb.reshape(n, height * width, 3)
    colors = FACE_COLORS.to(device=rgb.device, dtype=rgb.dtype)
    distance = (pixels[:, None] - colors[None, :, None]).square().mean(dim=-1)
    # White background and dark table are suppressed by the color distance;
    # the brightness term removes residual responses in black regions.
    weights = torch.exp(-distance / 0.025) * (pixels.mean(dim=-1)[:, None] > 0.06)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype),
        torch.linspace(-1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype),
        indexing="ij",
    )
    xx, yy = xx.reshape(1, 1, -1), yy.reshape(1, 1, -1)
    mass = weights.mean(dim=-1)
    denom = weights.sum(dim=-1).clamp_min(1e-6)
    cx = (weights * xx).sum(dim=-1) / denom
    cy = (weights * yy).sum(dim=-1) / denom
    sx = torch.sqrt((weights * (xx - cx[..., None]).square()).sum(dim=-1) / denom)
    sy = torch.sqrt((weights * (yy - cy[..., None]).square()).sum(dim=-1) / denom)
    marker = torch.stack((mass, cx, cy, sx + sy), dim=-1).reshape(n, -1)
    center = rgb[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    global_stats = torch.cat((rgb.mean((1, 2)), rgb.std((1, 2)),
                              center.mean((1, 2)), center.std((1, 2))), dim=-1)
    return torch.cat((global_stats, marker), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    image_features = []
    for frame_batch in data["frames"]:
        image_features.append(semantic_rgb_features(frame_batch["rgb"]))
    result = {
        "image_features": torch.cat(image_features),
        "language_features": torch.cat([
            torch.nn.functional.one_hot(torch.as_tensor(x, dtype=torch.long).reshape(-1), num_classes=6).float()
            for x in data["target_face"]
        ]),
        "student_proprio": torch.cat(data["student_proprio"]),
        "actions": torch.cat(data["actions"]),
        "teacher_actions": torch.cat(data.get("teacher_actions", data["actions"])),
        "target_face": torch.cat(data["target_face"]),
        "episode_id": torch.cat(data["episode_id"]),
        "step_index": torch.cat(data["step_index"]),
        "env_id": torch.cat(data.get("env_id", [torch.zeros_like(x) for x in data["episode_id"]])),
        "terminal": torch.cat(data.get("terminal", [torch.zeros_like(x) for x in data["episode_id"]])).bool(),
        "model": "semantic_rgb_marker_stats+face_onehot",
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
              "language_dim": int(result["language_features"].shape[-1]), "model": result["model"]}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
