"""Reproducible, camera-free evaluation of the semantic Revo3 state policy.

The protocol evaluates every requested face and seed separately.  Episodes run
to a fixed horizon (or a drop), even after an instantaneous reach, so fast
successes cannot replace slower trials.  All terminal metrics are copied from
the environment's pre-reset ``semantic_metrics`` payload before bookkeeping.

Example (from the repository root)::

    python scripts/rsl_rl/evaluate_state_baseline.py \
      --checkpoint logs/rsl_rl/brainco_hand/<run>/model_1000.pt \
      --episodes-per-face 100 --num-envs 64 --seeds 0,1,2 --headless \
      --report outputs/state-baseline.json

This script intentionally does not enable cameras or require RTX rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a Revo3 semantic state-policy baseline.")
parser.add_argument("--task", type=str, default="BrainCo-Direct-Revo3-SemanticReorient-Cube-v0")
parser.add_argument("--episodes-per-face", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated simulator seeds.")
parser.add_argument("--faces", type=str, default="0,1,2,3,4,5", help="Comma-separated face IDs.")
parser.add_argument("--goal-yaw", type=float, default=None, help="Fixed goal yaw in radians; omit for random yaw.")
parser.add_argument("--report", type=str, default="outputs/state-baseline.json")
parser.add_argument("--object-scale", type=float, default=None)
parser.add_argument("--object-density", type=float, default=None)
parser.add_argument("--static-friction", type=float, default=None)
parser.add_argument("--dynamic-friction", type=float, default=None)
parser.add_argument("--reset-position-noise", type=float, default=None)
parser.add_argument("--reset-dof-pos-noise", type=float, default=None)
parser.add_argument(
    "--initial-rotation-noise",
    type=float,
    default=None,
    help="Unsupported by the current environment; passing it fails instead of silently doing nothing.",
)
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

from state_baseline_protocol import EpisodeQuota, aggregate_records  # noqa: E402


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _done_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    return torch.as_tensor(value, device=device).bool().reshape(-1)


def _metric_tensor(metrics: dict[str, Any], key: str, fallback: torch.Tensor, *, dtype=None) -> torch.Tensor:
    value = metrics.get(key, fallback)
    value = torch.as_tensor(value, device=fallback.device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.reshape(fallback.shape).clone()


def _require_protocol_metrics(metrics: Any) -> dict[str, Any]:
    required = {
        "orientation_error",
        "object_distance",
        "goal_reached",
        "dropped",
        "hold_complete",
        "hold_steps",
        "target_face",
        "goal_rotation",
    }
    if not isinstance(metrics, dict):
        raise RuntimeError(
            "semantic_metrics is unavailable. This evaluator requires the updated SemanticReorientEnv "
            "with record_eval_metrics=True; old checkpoints/code cannot produce a hold result."
        )
    missing = sorted(required.difference(metrics))
    if missing:
        raise RuntimeError(
            "semantic_metrics is missing protocol fields: " + ", ".join(missing) + ". "
            "Update semantic_reorient.py before running the state baseline; hold metrics are never fabricated."
        )
    return metrics


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Convert Isaac config objects into bounded JSON manifest data."""

    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value), depth + 1)
        except Exception:
            pass
    return repr(value)


def _set_attr_or_fail(obj: Any, path: str, value: Any) -> None:
    current = obj
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise ValueError(f"stress override {path!r} is unsupported by this environment")
        current = getattr(current, part)
    if not hasattr(current, parts[-1]):
        raise ValueError(f"stress override {path!r} is unsupported by this environment")
    setattr(current, parts[-1], value)


def _apply_stress_overrides(env_cfg: Any) -> dict[str, Any]:
    if args_cli.initial_rotation_noise is not None:
        raise ValueError(
            "--initial-rotation-noise is not supported: the current environment hard-codes X/Y reset rotation noise. "
            "No silent approximation is applied."
        )
    overrides: dict[str, Any] = {}
    for cli_name, path in (
        ("reset_position_noise", "reset_position_noise"),
        ("reset_dof_pos_noise", "reset_dof_pos_noise"),
    ):
        value = getattr(args_cli, cli_name)
        if value is not None:
            _set_attr_or_fail(env_cfg, path, value)
            overrides[path] = value
    if args_cli.object_scale is not None:
        _set_attr_or_fail(env_cfg, "object_cfg.spawn.scale", (args_cli.object_scale,) * 3)
        overrides["object_cfg.spawn.scale"] = (args_cli.object_scale,) * 3
    if args_cli.object_density is not None:
        _set_attr_or_fail(env_cfg, "object_cfg.spawn.mass_props.density", args_cli.object_density)
        overrides["object_cfg.spawn.mass_props.density"] = args_cli.object_density
    for cli_name, path in (("static_friction", "sim.physics_material.static_friction"), ("dynamic_friction", "sim.physics_material.dynamic_friction")):
        value = getattr(args_cli, cli_name)
        if value is not None:
            _set_attr_or_fail(env_cfg, path, value)
            overrides[path] = value
    return overrides


def _runner_for(env: Any, agent_cfg: Any, checkpoint: str):
    cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint, map_location=agent_cfg.device)
    return runner, runner.get_inference_policy(device=env.unwrapped.device)


def _run_block(env_cfg: Any, agent_cfg: Any, checkpoint: str, seed: int, face: int) -> list[dict[str, Any]]:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = seed
    env_cfg.fixed_target_face = face
    env_cfg.goal_yaw = args_cli.goal_yaw
    env_cfg.max_consecutive_success = 0
    env_cfg.freeze_goal_for_episode = True
    env_cfg.goal_hold_time_s = 0.5
    env_cfg.record_eval_metrics = True
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.dirname(checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner, policy = _runner_for(env, agent_cfg, checkpoint)
    policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "actor_critic", None)
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    episode_quota = EpisodeQuota(args_cli.episodes_per_face, num_envs)
    quotas = episode_quota.quotas
    accepted = episode_quota.counts()
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    min_error = torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device)
    dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    instant_reach = torch.zeros(num_envs, dtype=torch.bool, device=device)
    held_at_end = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_action = torch.zeros((num_envs, env.unwrapped.num_hand_dofs), device=device)
    has_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)
    sum_clipped_l2 = torch.zeros(num_envs, device=device)
    sum_slew_l2 = torch.zeros(num_envs, device=device)
    sum_target_delta = torch.zeros(num_envs, device=device)
    face_before = env.unwrapped.target_face.detach().clone()
    goal_before = env.unwrapped.goal_rot.detach().clone()
    obs = env.get_observations()
    records: list[dict[str, Any]] = []
    while simulation_app.is_running() and sum(accepted) < args_cli.episodes_per_face:
        with torch.no_grad():
            raw = policy(obs).detach()
            action = raw.clamp(-1.0, 1.0)
            lower = env.unwrapped.hand_dof_lower_limits[:, env.unwrapped.actuated_dof_indices]
            upper = env.unwrapped.hand_dof_upper_limits[:, env.unwrapped.actuated_dof_indices]
            previous_target = env.unwrapped.cur_targets[:, env.unwrapped.actuated_dof_indices].detach().clone()
            scaled = lower + 0.5 * (action + 1.0) * (upper - lower)
            moving_average = float(env.unwrapped.cfg.act_moving_average)
            actual_target = moving_average * scaled + (1.0 - moving_average) * previous_target
            target_delta = torch.linalg.vector_norm(actual_target - previous_target, dim=-1)
            clipped_l2 = torch.linalg.vector_norm(action, dim=-1)
            slew_l2 = torch.where(
                has_prev, torch.linalg.vector_norm(action - prev_action, dim=-1), torch.zeros_like(clipped_l2)
            )
            obs, _, dones_raw, info = env.step(action)
            dones = _done_tensor(dones_raw, device)
            metrics = _require_protocol_metrics(info.get("semantic_metrics") if isinstance(info, dict) else None)
            orientation = _metric_tensor(metrics, "orientation_error", min_error.new_zeros(num_envs))
            object_distance = _metric_tensor(metrics, "object_distance", min_error.new_zeros(num_envs))
            reach = _metric_tensor(metrics, "goal_reached", instant_reach, dtype=torch.bool)
            drop = _metric_tensor(metrics, "dropped", dropped, dtype=torch.bool)
            hold_complete = _metric_tensor(metrics, "hold_complete", held_at_end, dtype=torch.bool)
            hold_steps = _metric_tensor(metrics, "hold_steps", episode_steps, dtype=torch.long)
            metric_face = _metric_tensor(metrics, "target_face", face_before, dtype=torch.long)
            metric_goal = _metric_tensor(metrics, "goal_rotation", goal_before)
            min_error = torch.minimum(min_error, orientation)
            dropped |= drop
            instant_reach |= reach
            held_at_end = hold_complete
            episode_steps += 1
            sum_clipped_l2 += clipped_l2
            sum_slew_l2 += slew_l2
            sum_target_delta += target_delta
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        # Clone all values derived from the simulator before any reset-slot
        # assignment.  This also avoids mutating PyTorch inference tensors.
        dones_cpu = dones.detach().cpu().tolist()
        for idx, done in enumerate(dones_cpu):
            if not done or accepted[idx] >= quotas[idx]:
                continue
            accepted[idx] += 1
            n_steps = int(episode_steps[idx].item())
            metric_face_value = int(metric_face[idx].item())
            if metric_face_value != face:
                raise RuntimeError(
                    f"terminal target_face mismatch: block requested {face}, metric reported {metric_face_value}"
                )
            record = {
                "seed": seed,
                "face": metric_face_value,
                "success": bool(held_at_end[idx].item()) and not bool(dropped[idx].item()),
                "instant_reach": bool(instant_reach[idx].item()) and not bool(dropped[idx].item()),
                "continuous_hold": bool(hold_complete[idx].item()) and not bool(dropped[idx].item()),
                "held_at_end": bool(held_at_end[idx].item()) and not bool(dropped[idx].item()),
                "drop": bool(dropped[idx].item()) or bool(object_distance[idx].item() >= float(env.unwrapped.cfg.fall_dist)),
                "hold_steps": int(hold_steps[idx].item()),
                "steps": n_steps,
                "time_s": n_steps * float(env.unwrapped.step_dt),
                "final_orientation_error_rad": float(orientation[idx].item()),
                "min_orientation_error_rad": float(min_error[idx].item()),
                "goal_rotation": metric_goal[idx].detach().cpu().tolist(),
                "mean_clipped_action_l2": float((sum_clipped_l2[idx] / max(n_steps, 1)).item()),
                "mean_action_slew_l2": float((sum_slew_l2[idx] / max(n_steps, 1)).item()),
                "mean_target_delta_rad": float((sum_target_delta[idx] / max(n_steps, 1)).item()),
            }
            records.append(record)
            episode_steps[idx] = 0
            min_error[idx] = float("inf")
            dropped[idx] = False
            instant_reach[idx] = False
            held_at_end[idx] = False
            sum_clipped_l2[idx] = 0
            sum_slew_l2[idx] = 0
            sum_target_delta[idx] = 0
            has_prev[idx] = False
            prev_action[idx] = 0
            face_before[idx] = env.unwrapped.target_face[idx]
            goal_before[idx] = env.unwrapped.goal_rot[idx]
        active = ~dones
        prev_action[active] = action[active]
        has_prev[active] = True
    env.close()
    return records


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args_cli.episodes_per_face <= 0 or args_cli.num_envs <= 0:
        raise ValueError("--episodes-per-face and --num-envs must be positive")
    seeds = _parse_ints(args_cli.seeds, "--seeds")
    faces = _parse_ints(args_cli.faces, "--faces")
    if any(face < 0 or face >= 6 for face in faces):
        raise ValueError("face IDs must be in [0, 5]")
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    stress = _apply_stress_overrides(env_cfg)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = str(args_cli.device or env_cfg.sim.device)
    all_records: list[dict[str, Any]] = []
    for seed in seeds:
        for face in faces:
            # The environment is recreated and closed for every fixed
            # face/seed block.  _run_block assigns all per-block fields, so
            # reusing the Hydra config avoids deepcopying Isaac config objects.
            all_records.extend(_run_block(env_cfg, agent_cfg, checkpoint, seed, face))
    by_face: dict[str, Any] = {}
    by_seed: dict[str, Any] = {}
    for face in faces:
        by_face[str(face)] = aggregate_records([row for row in all_records if row["face"] == face])
    for seed in seeds:
        by_seed[str(seed)] = aggregate_records([row for row in all_records if row["seed"] == seed])
    report = {
        "protocol": "state_baseline_v1",
        "task": args_cli.task,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "git_revision": _git_revision(),
        "episodes_per_face_per_seed": args_cli.episodes_per_face,
        "seeds": seeds,
        "faces": faces,
        "goal_yaw_rad": args_cli.goal_yaw,
        "goal_yaw_mode": "fixed" if args_cli.goal_yaw is not None else "random",
        "num_envs": args_cli.num_envs,
        "hold_tolerance_rad": float(env_cfg.success_tolerance),
        "hold_time_s": 0.5,
        "stress_overrides": stress,
        "env_cfg": _json_safe(env_cfg),
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        "summary": aggregate_records(all_records),
        "per_face": by_face,
        "per_seed": by_seed,
        "records": all_records,
        "complete": len(all_records) == args_cli.episodes_per_face * len(faces) * len(seeds),
        "argv": sys.argv,
    }
    output = Path(args_cli.report).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"[INFO] Wrote report to {output}")
    if not report["complete"]:
        raise RuntimeError("evaluation ended before the fixed episode quota was complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
