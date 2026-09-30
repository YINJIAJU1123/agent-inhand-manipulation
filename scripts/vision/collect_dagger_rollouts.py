"""Collect teacher labels on states visited by a visual student (DAgger)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger labels for a visual Revo3 student.")
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--student-checkpoint", required=True)
parser.add_argument("--task", default="BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0")
parser.add_argument("--episodes", type=int, default=200, help="Number of recorded target episodes.")
parser.add_argument("--target-faces", default="", help="Comma-separated target face ids to record (for example 4,5). Empty records all faces.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--vision-stride", type=int, default=2)
parser.add_argument("--semantic-grid-size", type=int, default=16)
parser.add_argument("--semantic-depth-grid-size", type=int, default=16)
parser.add_argument("--output", required=True)
parser.add_argument("--freeze-manifest", default="configs/visual_student_freeze.json")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.enable_cameras = True
args.target_faces = tuple(sorted({int(x) for x in args.target_faces.split(",") if x.strip()}))
if any(x < 0 or x >= 6 for x in args.target_faces):
    raise ValueError("target face ids must be in [0, 5]")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import *  # noqa: F401,F403,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "BrainCo_DexHand"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "vision"))
import BrainCo_DexHand  # noqa: F401,E402
from train_feature_student import VisualLanguageStudent, VisualStudentBatch  # noqa: E402
from BrainCo_DexHand.algo.agentic.language_goal import FACE_NAMES  # noqa: E402


def semantic_features(rgb, depth, grid_size, depth_grid_size):
    rgb = rgb.to(torch.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = rgb[..., :3]
    n, h, w, _ = rgb.shape
    pixels = rgb.reshape(n, h * w, 3)
    colors = torch.tensor([[0.85, 0.05, 0.05], [0.05, 0.75, 0.15], [0.05, 0.25, 0.90],
                           [0.95, 0.75, 0.05], [0.80, 0.05, 0.75], [0.05, 0.80, 0.85]],
                          device=rgb.device, dtype=rgb.dtype)
    weights = torch.exp(-(pixels[:, None] - colors[None, :, None]).square().mean(-1) / 0.025)
    weights = weights * (pixels.mean(-1)[:, None] > 0.06)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype),
                            torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype), indexing="ij")
    xx, yy = xx.reshape(1, 1, -1), yy.reshape(1, 1, -1)
    mass = weights.mean(-1)
    denom = weights.sum(-1).clamp_min(1e-6)
    cx = (weights * xx).sum(-1) / denom
    cy = (weights * yy).sum(-1) / denom
    sx = torch.sqrt((weights * (xx - cx[..., None]).square()).sum(-1) / denom)
    sy = torch.sqrt((weights * (yy - cy[..., None]).square()).sum(-1) / denom)
    marker = torch.stack((mass, cx, cy, sx + sy), -1).reshape(n, -1)
    center = rgb[:, h // 4:3 * h // 4, w // 4:3 * w // 4]
    global_stats = torch.cat((rgb.mean((1, 2)), rgb.std((1, 2)), center.mean((1, 2)), center.std((1, 2))), -1)
    pieces = [global_stats]
    if grid_size:
        grid = torch.nn.functional.interpolate(rgb.permute(0, 3, 1, 2), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        pieces.append(grid.reshape(n, -1))
    if depth_grid_size:
        d = torch.nan_to_num(depth.to(rgb.dtype), nan=2.0, posinf=2.0, neginf=0.0).clamp(0, 2) / 2
        if d.ndim == 3:
            d = d.unsqueeze(-1)
        d = torch.nn.functional.interpolate(d.permute(0, 3, 1, 2), size=(depth_grid_size, depth_grid_size), mode="bilinear", align_corners=False)
        pieces.append(d.reshape(n, -1))
    pieces.append(marker)
    return torch.cat(pieces, -1)


def instruction_faces(raw, device):
    names = {name: i for i, name in enumerate(FACE_NAMES)}
    values = []
    for text in raw.current_instructions():
        matches = [i for name, i in names.items() if name in str(text).lower().split()]
        if len(matches) != 1:
            raise RuntimeError(f"cannot parse instruction {text!r}")
        values.append(matches[0])
    return torch.tensor(values, dtype=torch.long, device=device)


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.max_consecutive_success = 1
    env_cfg.record_eval_metrics = True
    env_cfg.seed = 987
    env_cfg.goal_yaw = 0.0
    env_cfg.sim.device = args.device or env_cfg.sim.device
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    teacher_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    teacher_runner.load(retrieve_file_path(args.teacher_checkpoint), map_location=agent_cfg.device)
    teacher = teacher_runner.get_inference_policy(device=env.unwrapped.device)
    teacher_nn = getattr(teacher_runner.alg, "policy", None) or getattr(teacher_runner.alg, "actor_critic", None)
    device = env.unwrapped.device
    ckpt = torch.load(args.student_checkpoint, map_location=device, weights_only=False)
    manifest = json.loads(Path(args.freeze_manifest).read_text())
    student = VisualLanguageStudent(ckpt["rgb_dim"], ckpt["language_dim"], ckpt["proprio_dim"], ckpt["action_dim"], memory_mode=ckpt.get("memory_mode", "plain")).to(device)
    student.load_state_dict(ckpt["model"]); student.eval()
    history = int(ckpt.get("history", 8)); n = env.unwrapped.num_envs
    rgb_hist = torch.zeros((n, history, ckpt["rgb_dim"]), device=device)
    prop_hist = torch.zeros((n, history, ckpt["proprio_dim"]), device=device)
    initialized = torch.zeros(n, dtype=torch.bool, device=device)
    record_episode = torch.zeros(n, dtype=torch.bool, device=device)
    record_face = torch.full((n,), -1, dtype=torch.long, device=device)
    episode_id = torch.zeros(n, dtype=torch.long, device=device)
    step_index = torch.zeros(n, dtype=torch.long, device=device)
    env_ids = torch.arange(n, device=device)
    completed = 0; step = 0
    image_features, language_features, proprio, actions, teacher_actions = [], [], [], [], []
    target_faces, episodes, steps, env_batches = [], [], [], []
    quality = []
    obs = env.get_observations()
    while app.is_running() and completed < args.episodes:
        capture = step % max(args.vision_stride, 1) == 0
        with torch.no_grad():
            if capture:
                camera = env.unwrapped.capture_camera()
                image = semantic_features(camera["rgb"][..., :3], camera.get("depth"), args.semantic_grid_size, args.semantic_depth_grid_size)
                face = instruction_faces(env.unwrapped, device)
                prop = env.unwrapped.compute_student_proprio().float()
                rgb_hist = torch.cat((rgb_hist[:, 1:], image[:, None]), 1)
                prop_hist = torch.cat((prop_hist[:, 1:], prop[:, None]), 1)
                fresh = ~initialized
                if fresh.any():
                    rgb_hist[fresh] = image[fresh, None]
                    prop_hist[fresh] = prop[fresh, None]
                    initialized[fresh] = True
                    record_face[fresh] = face[fresh]
                    if args.target_faces:
                        record_episode[fresh] = torch.isin(face[fresh], torch.tensor(args.target_faces, device=device))
                    else:
                        record_episode[fresh] = True
                lang = torch.eye(6, device=device)[face]
                out = student(VisualStudentBatch(rgb_hist, lang, prop_hist))
                teacher_action = teacher(obs)
                keep = record_episode
                if keep.any():
                    image_features.append(image[keep].cpu()); language_features.append(lang[keep].cpu()); proprio.append(prop[keep].cpu())
                    actions.append(teacher_action[keep].clamp(-1, 1).cpu()); teacher_actions.append(teacher_action[keep].cpu())
                    target_faces.append(face[keep].cpu()); episodes.append(episode_id[keep].cpu()); steps.append(step_index[keep].cpu()); env_batches.append(env_ids[keep].cpu())
            else:
                out = student(VisualStudentBatch(rgb_hist, lang, prop_hist))
                teacher_action = teacher(obs)
            student_action = out["action"]
            obs, _, dones, _ = env.step(student_action)
            if teacher_nn is not None and hasattr(teacher_nn, "reset"):
                teacher_nn.reset(dones)
        done = torch.as_tensor(dones, device=device).bool().reshape(-1)
        if done.any():
            metrics = env.unwrapped.extras.get("semantic_metrics", {})
            reached = metrics.get("goal_reached", torch.zeros_like(done)); dropped = metrics.get("dropped", torch.zeros_like(done))
            for idx in torch.nonzero(done, as_tuple=False).flatten().tolist():
                if bool(record_episode[idx]):
                    quality.append({"env_id": int(env_ids[idx]), "episode_id": int(episode_id[idx]), "face": int(record_face[idx]), "success": bool(reached[idx]), "drop": bool(dropped[idx])})
            completed += int((done & record_episode).sum())
            episode_id[done] += 1; step_index[done] = 0; initialized[done] = False; record_episode[done] = False; record_face[done] = -1
            rgb_hist[done] = 0; prop_hist[done] = 0
        step_index += 1; step += 1
        if step % 100 == 0:
            print(f"[dagger] steps={step} episodes={completed}", flush=True)
    result = {
        "image_features": torch.cat(image_features), "language_features": torch.cat(language_features),
        "student_proprio": torch.cat(proprio), "actions": torch.cat(actions), "teacher_actions": torch.cat(teacher_actions),
        "target_face": torch.cat(target_faces), "episode_id": torch.cat(episodes), "step_index": torch.cat(steps), "env_id": torch.cat(env_batches),
        "model": f"dagger_semantic_rgb_grid{args.semantic_grid_size}_depth{args.semantic_depth_grid_size}",
        "freeze_id": manifest["freeze_id"], "action_storage": "bounded_projection_of_teacher_output",
        "source_checkpoint_sha256": manifest["teacher"]["sha256"], "quality": quality, "target_faces": list(args.target_faces),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); torch.save(result, out)
    print(json.dumps({"output": str(out), "samples": int(result["actions"].shape[0]), "episodes": completed, "quality": quality}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
