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
    cfg.max_consecutive_success = 1
    cfg.seed = 0
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    ckpt = torch.load(args.checkpoint, map_location=raw.device, weights_only=False)
    history = int(ckpt.get("history", 1))
    model = VisualBC(ckpt["proprio_dim"], ckpt["action_dim"], history=history).to(raw.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    obs, _ = env.reset()
    records = []
    steps = 0
    episode_success = torch.zeros(args.num_envs, dtype=torch.bool, device=raw.device)
    episode_face = raw.target_face.clone()
    episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=raw.device)
    image_history = []
    safety_cap = args.max_steps * args.episodes
    with torch.inference_mode():
        while app.is_running() and len(records) < args.episodes and steps < safety_cap:
            camera = raw.capture_camera()
            rgb = camera["rgb"].to(torch.float32) / 255.0
            depth = camera["depth"].to(torch.float32).clamp(0.0, 2.0) / 2.0
            current_image = torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2)
            image_history.append(current_image)
            image_history = image_history[-max(history, 1):]
            image = torch.cat(([image_history[0]] * (history - len(image_history))) + image_history, dim=1)
            language = torch.nn.functional.one_hot(raw.target_face, num_classes=6).float()
            proprio = raw.compute_student_proprio()
            action = model(image, language, proprio).clamp(-1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            episode_success |= reward >= 0.5 * raw.cfg.reach_goal_bonus
            episode_steps += 1
            for index in torch.nonzero(done).flatten().tolist():
                records.append({
                    "face": int(episode_face[index].item()),
                    "success": bool(episode_success[index].item()),
                    "drop": bool(terminated[index].item()),
                    "steps": int(episode_steps[index].item()),
                })
                episode_success[index] = False
                episode_steps[index] = 0
                episode_face[index] = raw.target_face[index]
                # Do not carry frames from the previous object/task across a
                # reset for this environment slot.
                for old in image_history[:-1]:
                    old[index].copy_(current_image[index])
                if len(records) >= args.episodes:
                    break
            steps += 1
            if steps % 50 == 0:
                print(f"[eval] steps={steps} episodes={len(records)}", flush=True)
    report = {
        "task": task,
        "checkpoint": args.checkpoint,
        "episodes": len(records),
        "vector_steps": steps,
        "success_rate": sum(r["success"] for r in records) / max(len(records), 1),
        "drop_rate": sum(r["drop"] for r in records) / max(len(records), 1),
        "records": records,
        "scope": "Small color-token BC smoke; no open-vocabulary language claim",
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
