"""Closed-loop evaluation for the RGB-D + language behavior-cloning baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--report", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import BrainCo_DexHand  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_visual_bc import VisualBC  # noqa: E402


def main():
    task = "BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    ckpt = torch.load(args.checkpoint, map_location=raw.device, weights_only=False)
    model = VisualBC(ckpt["proprio_dim"], ckpt["action_dim"]).to(raw.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    obs, _ = env.reset()
    completed = 0
    dropped = 0
    steps = 0
    previous_successes = raw.successes.clone()
    success_total = 0
    with torch.inference_mode():
        while app.is_running() and completed < args.episodes and steps < args.max_steps:
            camera = raw.capture_camera()
            rgb = camera["rgb"].to(torch.float32) / 255.0
            depth = camera["depth"].to(torch.float32).clamp(0.0, 2.0) / 2.0
            image = torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2)
            language = torch.nn.functional.one_hot(raw.target_face, num_classes=6).float()
            proprio = raw.compute_student_proprio()
            action = model(image, language, proprio).clamp(-1.0, 1.0)
            obs, _, dones, _ = env.step(action)
            # ``successes`` survives a goal reset but is cleared on an episode
            # reset; count only positive increments before updating the
            # per-environment reference.
            delta_success = raw.successes - previous_successes
            success_total += int(delta_success.clamp_min(0).sum().item())
            previous_successes = raw.successes.clone()
            done = dones[0] if isinstance(dones, tuple) else dones
            done = torch.as_tensor(done, device=raw.device)
            completed += int(done.sum().item())
            dropped += int((raw.object_pos - raw.in_hand_pos).norm(dim=-1).ge(raw.cfg.fall_dist).sum().item())
            steps += 1
    report = {
        "task": task,
        "checkpoint": args.checkpoint,
        "requested_episodes": args.episodes,
        "completed_episode_resets": completed,
        "steps": steps,
        "success_events": success_total,
        "drop_events": dropped,
        "success_events_per_reset": float(success_total / max(completed, 1)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
