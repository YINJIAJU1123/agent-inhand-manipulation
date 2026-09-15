"""Closed-loop evaluation for the frozen-VLM Revo3 student.

The SigLIP2 encoder is frozen and runs only as an observation encoder.  The
GRU/action head remains the single low-level controller and emits the same
21-dimensional bounded command as the Revo3 teacher interface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--model", default="google/siglip2-base-patch16-224")
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--vision-stride", type=int, default=4, help="Run the frozen VLM every N control steps.")
parser.add_argument("--report", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import BrainCo_DexHand  # noqa: F401, E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_feature_student import VisualLanguageStudent, VisualStudentBatch  # noqa: E402


def _features(model, processor, images, texts, device):
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items() if torch.is_tensor(v)}
    image_features = model.get_image_features(**image_inputs)
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items() if torch.is_tensor(v)}
    text_features = model.get_text_features(**text_inputs)
    return (
        torch.nn.functional.normalize(image_features.float(), dim=-1),
        torch.nn.functional.normalize(text_features.float(), dim=-1),
    )


def main():
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    task = "BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0"
    cfg = parse_env_cfg(task, device=str(device), num_envs=args.num_envs)
    cfg.max_consecutive_success = 1
    cfg.record_eval_metrics = True
    cfg.seed = 123
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = VisualLanguageStudent(
        ckpt["rgb_dim"], ckpt["language_dim"], ckpt["proprio_dim"], ckpt["action_dim"]
    ).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    vlm = AutoModel.from_pretrained(args.model).to(device).eval()
    face_names = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    text_prompts = [f"show the {x} marker" for x in face_names]
    with torch.inference_mode():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items() if torch.is_tensor(v)}
        text_features = torch.nn.functional.normalize(vlm.get_text_features(**text_inputs).float(), dim=-1)

    obs, _ = env.reset()
    n = raw.num_envs
    history = int(ckpt.get("history", 8))
    image_hist = torch.zeros((n, history, ckpt["rgb_dim"]), device=device)
    prop_hist = torch.zeros((n, history, ckpt["proprio_dim"]), device=device)
    image_features = torch.zeros((n, ckpt["rgb_dim"]), device=device)
    done_count = 0
    step_count = 0
    episode_steps = torch.zeros(n, dtype=torch.long, device=device)
    episode_success = torch.zeros(n, dtype=torch.bool, device=device)
    episode_drop = torch.zeros(n, dtype=torch.bool, device=device)
    episode_min_error = torch.full((n,), float("inf"), device=device)
    episode_face = raw.target_face.clone()
    records = []
    max_vector_steps = args.max_steps * math.ceil(args.episodes / max(n, 1))
    while app.is_running() and done_count < args.episodes and step_count < max_vector_steps:
        with torch.inference_mode():
            if step_count % max(args.vision_stride, 1) == 0:
                camera = raw.capture_camera()
                rgb = camera["rgb"][..., :3].to(torch.uint8)
                images = [Image.fromarray(x.cpu().numpy()) for x in rgb]
                image_features, _ = _features(vlm, processor, images, text_prompts, device)
            face_text = text_features[raw.target_face]
            image_hist = torch.cat((image_hist[:, 1:], image_features[:, None]), dim=1)
            proprio = raw.compute_student_proprio().float()
            prop_hist = torch.cat((prop_hist[:, 1:], proprio[:, None]), dim=1)
            out = policy(VisualStudentBatch(image_hist, face_text, prop_hist))
            action = out["action"].clamp(-1.0, 1.0)
            _, _, terminated, truncated, info = env.step(action)
            done = (terminated | truncated).to(device=device, dtype=torch.bool)
            metrics = info["semantic_metrics"]
            episode_steps += 1
            episode_min_error = torch.minimum(episode_min_error, metrics["orientation_error"])
            episode_success |= metrics["goal_reached"] & ~metrics["dropped"]
            episode_drop |= metrics["dropped"]
            # Apply an evaluator-side horizon even when the environment does
            # not emit a terminal signal.  This keeps each rollout bounded
            # and lets us compare success/timeout rates fairly.
            timeout = episode_steps >= args.max_steps
            done_eval = done | timeout
        timeout_ids = torch.nonzero(timeout & ~done, as_tuple=False).flatten()
        if len(timeout_ids) > 0:
            raw._reset_idx(timeout_ids)
        for idx_t in torch.nonzero(done_eval, as_tuple=False).flatten():
            idx = int(idx_t.item())
            if done_count >= args.episodes:
                break
            records.append({
                "face": int(episode_face[idx].item()),
                "success": bool(episode_success[idx].item()) and not bool(episode_drop[idx].item()),
                "drop": bool(episode_drop[idx].item()),
                "timeout": bool(timeout[idx].item()),
                "min_orientation_error_rad": float(episode_min_error[idx].item()),
                "steps": int(episode_steps[idx].item()),
            })
            done_count += 1
            episode_steps[idx] = 0
            episode_success[idx] = False
            episode_drop[idx] = False
            episode_min_error[idx] = float("inf")
            episode_face[idx] = raw.target_face[idx]
            image_hist[idx] = 0
            prop_hist[idx] = 0
            image_features[idx] = 0
        step_count += 1
        if step_count % 50 == 0:
            print(f"[eval] vector_steps={step_count} episodes={done_count}", flush=True)

    # A visual student can fail to trigger the environment's terminal condition
    # (for example, it may neither reach the pose nor drop the object).  Keep
    # those partial rollouts in the report as explicit timeouts instead of
    # silently returning ``episodes: 0``.
    if done_count < args.episodes:
        for idx in range(n):
            if done_count >= args.episodes or episode_steps[idx].item() == 0:
                continue
            records.append({
                "face": int(episode_face[idx].item()),
                "success": False,
                "drop": bool(episode_drop[idx].item()),
                "timeout": True,
                "min_orientation_error_rad": float(episode_min_error[idx].item()),
                "steps": int(episode_steps[idx].item()),
            })
            done_count += 1

    per_face = defaultdict(list)
    for record in records:
        per_face[record["face"]].append(record)
    report = {
        "task": task, "checkpoint": os.path.abspath(args.checkpoint), "model": args.model,
        "vision_stride": args.vision_stride,
        "episodes": len(records), "vector_steps": step_count,
        "success_rate": sum(x["success"] for x in records) / max(len(records), 1),
        "drop_rate": sum(x["drop"] for x in records) / max(len(records), 1),
        "per_face": {str(k): {"episodes": len(v), "success_rate": sum(x["success"] for x in v) / len(v),
                              "drop_rate": sum(x["drop"] for x in v) / len(v)} for k, v in sorted(per_face.items())},
        "records": records,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


try:
    main()
finally:
    app.close()
