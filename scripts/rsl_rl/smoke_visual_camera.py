"""Render actual task RGB-D frames before collecting training data.

No checkpoint or VLM is required. A pass establishes sensor output, not policy
success or semantic grounding; inspect rgb.png for framing and occlusion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="outputs/camera_smoke")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
import BrainCo_DexHand  # noqa: F401, E402


def main():
    task = "BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=1)
    cfg.seed = 0
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        with torch.inference_mode():
            # Repeated resets catch stale multi-hot labels. Verify each
            # sampled local face actually points up under its goal rotation.
            raw = env.unwrapped
            face_normals = torch.tensor(
                [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                 [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]],
                device=raw.device,
            )
            seen = set()
            for _ in range(128):
                raw._reset_target_pose(torch.arange(raw.num_envs, device=raw.device))
                assert torch.all(raw.target_face_onehot.sum(-1) == 1), "Stale goal labels"
                assert torch.equal(raw.target_face_onehot.argmax(-1), raw.target_face)
                normal = quat_apply(raw.goal_rot, face_normals[raw.target_face])
                assert torch.allclose(normal, face_normals[4:5], atol=1e-5), "Wrong goal face"
                seen.update(raw.target_face.tolist())
            assert len(seen) == 6, "Smoke did not exercise all six goal faces"
            action = torch.zeros((1, cfg.action_space), device=env.unwrapped.device)
            for _ in range(8):
                env.step(action)
            frames = env.unwrapped.capture_camera()
        rgb = frames["rgb"][0, ..., :3].cpu().numpy()
        depth = frames["depth"][0].cpu().numpy()
        assert rgb.shape == (cfg.tiled_camera.height, cfg.tiled_camera.width, 3)
        assert rgb.dtype == np.uint8, f"Unexpected RGB dtype: {rgb.dtype}"
        assert float(rgb.std()) > 1.0, "RGB is blank or constant"
        assert np.any(np.isfinite(depth) & (depth > 0)), "No valid depth samples"
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output / "rgb.png")
        np.save(output / "depth.npy", depth)
        report = {
            "task": task, "passed": True, "rgb_shape": list(rgb.shape),
            "rgb_std": float(rgb.std()),
            "valid_depth_fraction": float(np.mean(np.isfinite(depth) & (depth > 0))),
            "camera_offset": cfg.tiled_camera.offset.to_dict(),
            "goal_faces_checked": sorted(seen),
            "scope": "RGB-D sensor smoke only; inspect framing and occlusion manually",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit's default fast shutdown exits with status 0 even during Python
        # exception handling. A failed isolated smoke worker must return 1.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        app.close()
