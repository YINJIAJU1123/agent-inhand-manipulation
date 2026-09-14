"""Evaluate semantic reorientation together with action smoothness metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from evaluation_protocol import EpisodeQuota

parser = argparse.ArgumentParser(description="Evaluate Revo3 success and control quality.")
parser.add_argument("--task", type=str, default="BrainCo-Direct-Revo3-SemanticReorient-Cube-v0")
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--report", type=str, default="outputs/control_quality.json")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import BrainCo_DexHand  # noqa: F401, E402


def _done_tensor(value, device):
    if isinstance(value, tuple):
        value = value[0]
    return torch.as_tensor(value, device=device).bool().reshape(-1)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.max_consecutive_success = 1
    env_cfg.record_eval_metrics = True
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = str(env_cfg.sim.device)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)

    device = env.unwrapped.device
    n = env.unwrapped.num_envs
    quota = EpisodeQuota(args_cli.episodes, n)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    dropped = torch.zeros(n, dtype=torch.bool, device=device)
    min_error = torch.full((n,), float("inf"), device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    count = torch.zeros(n, dtype=torch.long, device=device)
    sum_abs = torch.zeros(n, device=device)
    sum_l2 = torch.zeros(n, device=device)
    sum_slew = torch.zeros(n, device=device)
    sum_target_delta = torch.zeros(n, device=device)
    prev_action = torch.zeros((n, env.unwrapped.num_hand_dofs), device=device)
    has_prev = torch.zeros(n, dtype=torch.bool, device=device)
    records = []
    face_before = env.unwrapped.target_face.clone()
    obs = env.get_observations()

    while simulation_app.is_running() and not quota.complete:
        with torch.inference_mode():
            # Keep the action passed to the wrapper identical to the official
            # evaluator.  Measure the clipped command that reaches the task.
            raw_action = policy(obs).detach()
            action = raw_action.clamp(-1.0, 1.0)
            sum_abs += action.abs().mean(-1)
            sum_l2 += torch.linalg.vector_norm(action, dim=-1)
            sum_slew += torch.where(
                has_prev, torch.linalg.vector_norm(action - prev_action, dim=-1), torch.zeros_like(sum_slew)
            )
            lower = env.unwrapped.hand_dof_lower_limits[:, env.unwrapped.actuated_dof_indices]
            upper = env.unwrapped.hand_dof_upper_limits[:, env.unwrapped.actuated_dof_indices]
            target = lower + 0.5 * (action + 1.0) * (upper - lower)
            current_target = env.unwrapped.cur_targets[:, env.unwrapped.actuated_dof_indices]
            sum_target_delta += torch.abs(target - current_target).mean(-1)

            obs, _, dones, info = env.step(raw_action)
            dones = _done_tensor(dones, device)
            metrics = info["semantic_metrics"]
            min_error = torch.minimum(min_error, metrics["orientation_error"])
            success |= metrics["goal_reached"] & ~metrics["dropped"]
            dropped |= metrics["dropped"]
            steps += 1
            count += 1
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        # Reductions involving simulator tensors inherit PyTorch's inference
        # tensor type. Clone them before indexed bookkeeping below.
        min_error = min_error.clone()
        success = success.clone()
        dropped = dropped.clone()

        for idx_t in torch.nonzero(dones, as_tuple=False).reshape(-1):
            idx = int(idx_t.item())
            if not quota.accept(idx):
                continue
            face = int(face_before[idx].item())
            records.append(
                {
                    "face": face,
                    "success": bool(success[idx].item()) and not bool(dropped[idx].item()),
                    "drop": bool(dropped[idx].item()),
                    "steps": int(steps[idx].item()),
                    "mean_abs_action": float((sum_abs[idx] / count[idx].clamp_min(1)).item()),
                    "mean_action_l2": float((sum_l2[idx] / count[idx].clamp_min(1)).item()),
                    "mean_action_slew_l2": float((sum_slew[idx] / count[idx].clamp_min(1)).item()),
                    "mean_target_delta_rad": float((sum_target_delta[idx] / count[idx].clamp_min(1)).item()),
                    "min_orientation_error_rad": float(min_error[idx].item()),
                }
            )
            success[idx] = False
            dropped[idx] = False
            min_error[idx] = float("inf")
            steps[idx] = 0
            count[idx] = 0
            sum_abs[idx] = 0
            sum_l2[idx] = 0
            sum_slew[idx] = 0
            sum_target_delta[idx] = 0
            has_prev[idx] = False
            prev_action[idx] = 0
            face_before[idx] = env.unwrapped.target_face[idx]
        active = ~dones
        prev_action[active] = action[active]
        has_prev[active] = True

    env.close()
    per_face = defaultdict(list)
    for record in records:
        per_face[record["face"]].append(record)

    def mean(key, values):
        return sum(v[key] for v in values) / max(len(values), 1)

    report = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(resume_path),
        "requested_episodes": args_cli.episodes,
        "episodes": len(records),
        "complete": quota.complete,
        "success_rate": sum(r["success"] for r in records) / max(len(records), 1),
        "drop_rate": sum(r["drop"] for r in records) / max(len(records), 1),
        "mean_abs_action": mean("mean_abs_action", records),
        "mean_action_l2": mean("mean_action_l2", records),
        "mean_action_slew_l2": mean("mean_action_slew_l2", records),
        "mean_target_delta_rad": mean("mean_target_delta_rad", records),
        "per_face": {
            str(face): {
                "episodes": len(values),
                "success_rate": sum(v["success"] for v in values) / max(len(values), 1),
                "drop_rate": sum(v["drop"] for v in values) / max(len(values), 1),
                "mean_abs_action": mean("mean_abs_action", values),
                "mean_action_slew_l2": mean("mean_action_slew_l2", values),
                "mean_target_delta_rad": mean("mean_target_delta_rad", values),
            }
            for face, values in sorted(per_face.items())
        },
        "records": records,
    }
    path = os.path.abspath(args_cli.report)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[INFO] Wrote report to {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
