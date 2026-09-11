# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a SemanticReorient RSL-RL checkpoint.

The task resets an environment after the first successful goal when
``max_consecutive_success`` is enabled.  We therefore identify the success
from the reward spike on the transition, while reading ``target_face`` before
stepping.  This keeps per-face statistics correct even though Isaac Lab has
already reset the goal by the time ``step`` returns.

Example (from the repository root)::

    python scripts/rsl_rl/evaluate_semantic_reorient.py \
        --task BrainCo-Direct-Revo3-SemanticReorient-Cube-v0 \
        --checkpoint logs/rsl_rl/brainco_hand/<run>/model_1000.pt \
        --num_envs 256 --episodes 100 --headless

The output is a JSON report containing overall and per-face success, drop,
orientation and episode-length metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate a SemanticReorient RSL-RL checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="BrainCo-Direct-Revo3-SemanticReorient-Cube-v0",
    help="Gym task id.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent config entry point.")
parser.add_argument("--episodes", type=int, default=100, help="Number of completed episodes to collect.")
parser.add_argument("--max_steps", type=int, default=0, help="Safety cap; 0 uses the environment episode length.")
parser.add_argument("--report", type=str, default=None, help="Optional JSON report output path.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel evaluation environments.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric.")
parser.add_argument("--real-time", action="store_true", help="Run at the simulated wall-clock rate.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import BrainCo_DexHand  # noqa: F401, E402


def _to_bool_tensor(value, device: torch.device) -> torch.Tensor:
    """Normalize Gym/RSL-RL done outputs to a one-dimensional bool tensor."""
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device)
    return value.to(device=device, dtype=torch.bool).reshape(-1)


def _to_cpu_list(value: torch.Tensor) -> list:
    return value.detach().to(device="cpu").tolist()


def _empty_face_stats() -> dict:
    return defaultdict(lambda: {"episodes": 0, "successes": 0, "drops": 0, "min_orientation_error": [], "steps": []})


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive")

    # A one-success episode makes the evaluation unit unambiguous.  The
    # training environment keeps its original setting (0 = continue goals).
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.max_consecutive_success = 1
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = str(env_cfg.sim.device)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    print(f"[INFO] Evaluating {args_cli.episodes} episodes with {args_cli.num_envs} environments")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    max_steps = args_cli.max_steps or int(env.unwrapped.max_episode_length)
    tolerance = float(env.unwrapped.cfg.success_tolerance)
    success_reward_threshold = 0.5 * float(env.unwrapped.cfg.reach_goal_bonus)
    face_stats = _empty_face_stats()
    all_records: list[dict] = []
    episodes_done = 0
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    min_rot = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
    max_obj_dist = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Face at the beginning of the current transition.  SemanticReorientEnv
    # changes this buffer immediately after a successful reward.
    episode_face = env.unwrapped.target_face.clone()

    obs = env.get_observations()
    start_time = time.time()
    while simulation_app.is_running() and episodes_done < args_cli.episodes:
        face_before = episode_face.clone()
        with torch.inference_mode():
            # Track the best state before stepping.  This remains valid even
            # when the environment auto-resets on the transition.
            current_rot = env.unwrapped.object_rot
            goal_rot = env.unwrapped.goal_rot
            # Keep the environment's TorchScript rotation helper local to
            # avoid importing task implementation details in this script.
            # Quaternion distance is sign-invariant; use the standard absolute
            # dot-product formulation for an evaluation-only metric.
            dot = torch.abs(torch.sum(current_rot * goal_rot, dim=-1)).clamp(max=1.0)
            rot_dist = 2.0 * torch.acos(dot)
            min_rot = torch.minimum(min_rot, rot_dist)
            obj_dist = torch.linalg.vector_norm(env.unwrapped.object_pos - env.unwrapped.in_hand_pos, dim=-1)
            max_obj_dist = torch.maximum(max_obj_dist, obj_dist)
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
            dones = _to_bool_tensor(dones, device)
            rewards = rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, device=device)
            rewards = rewards.reshape(-1)
            episode_steps += 1

            # A successful transition receives the configured success bonus.
            # Record it against face_before because the env samples the next
            # target face during reward computation.
            success_now = rewards >= success_reward_threshold
            episode_success |= success_now
            if success_now.any():
                min_rot = torch.where(success_now, torch.minimum(min_rot, torch.full_like(min_rot, tolerance)), min_rot)

            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        done_ids = torch.nonzero(dones, as_tuple=False).reshape(-1)
        for idx_t in done_ids:
            idx = int(idx_t.item())
            face = int(face_before[idx].item())
            success = bool(episode_success[idx].item())
            # A terminal episode without success is either a timeout or a
            # fall.  max_obj_dist separates the two for the drop metric.
            dropped = (not success) and bool(max_obj_dist[idx].item() >= float(env.unwrapped.cfg.fall_dist))
            error = float(min_rot[idx].item())
            if not error < float("inf"):
                error = float("nan")
            steps = int(episode_steps[idx].item())
            record = {
                "face": face,
                "success": success,
                "drop": dropped,
                "min_orientation_error_rad": error,
                "steps": steps,
                "time_s": steps * float(env.unwrapped.step_dt),
            }
            all_records.append(record)
            stats = face_stats[face]
            stats["episodes"] += 1
            stats["successes"] += int(success)
            stats["drops"] += int(dropped)
            if error == error:
                stats["min_orientation_error"].append(error)
            stats["steps"].append(steps)
            episodes_done += 1

            # The underlying env has reset this slot.  Initialize tracking
            # from the new target rather than retaining the old face.
            episode_steps[idx] = 0
            min_rot[idx] = float("inf")
            max_obj_dist[idx] = 0.0
            episode_success[idx] = False
            episode_face[idx] = env.unwrapped.target_face[idx]
            if episodes_done >= args_cli.episodes:
                break

        if args_cli.real_time:
            elapsed = time.time() - start_time
            expected = episode_steps.float().mean().item() * float(env.unwrapped.step_dt)
            if expected > elapsed:
                time.sleep(expected - elapsed)

    env.close()
    total = len(all_records)
    def mean(key: str) -> float:
        values = [r[key] for r in all_records if isinstance(r[key], (int, float)) and r[key] == r[key]]
        return float(sum(values) / len(values)) if values else float("nan")

    report = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(resume_path),
        "num_envs": num_envs,
        "episodes": total,
        "success_rate": sum(int(r["success"]) for r in all_records) / max(total, 1),
        "drop_rate": sum(int(r["drop"]) for r in all_records) / max(total, 1),
        "mean_min_orientation_error_rad": mean("min_orientation_error_rad"),
        "mean_steps": mean("steps"),
        "mean_time_s": mean("time_s"),
        "per_face": {},
    }
    for face, stats in sorted(face_stats.items()):
        n = stats["episodes"]
        report["per_face"][str(face)] = {
            "episodes": n,
            "success_rate": stats["successes"] / max(n, 1),
            "drop_rate": stats["drops"] / max(n, 1),
            "mean_min_orientation_error_rad": sum(stats["min_orientation_error"]) / max(len(stats["min_orientation_error"]), 1),
            "mean_steps": sum(stats["steps"]) / max(len(stats["steps"]), 1),
        }

    # JSON has no NaN in strict parsers; omit non-finite means as null.
    def json_safe(value):
        if isinstance(value, float) and value != value:
            return None
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        return value

    report = json_safe(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args_cli.report:
        report_path = os.path.abspath(args_cli.report)
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"[INFO] Wrote evaluation report to {report_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
