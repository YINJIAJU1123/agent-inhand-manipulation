"""Collect RGB-D demonstrations from the camera-enabled Revo3 teacher task.

The policy remains the 21-DoF semantic state teacher while this collector
records the camera stream, language/face labels, proprioception and actions.
The resulting torch shard is the input for visual-student and consequence
predictor training.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect RGB-D rollouts for visual Revo3 training.")
parser.add_argument("--task", type=str, default="BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--stride", type=int, default=1, help="Record every Nth environment step.")
parser.add_argument("--output", type=str, default="data/visual_rollouts.pt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import BrainCo_DexHand  # noqa: F401, E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        policy_nn = getattr(runner.alg, "actor_critic", None)
    obs = env.get_observations()

    frames, actions, proprio, labels = [], [], [], []
    completed = 0
    step = 0
    while simulation_app.is_running() and completed < args_cli.episodes:
        with torch.inference_mode():
            # Keep each frame/label/proprio sample aligned with the action
            # computed from that same observation.  The environment may
            # resample a target face during ``step`` when an episode ends.
            if hasattr(env.unwrapped, "capture_camera") and step % max(args_cli.stride, 1) == 0:
                camera = env.unwrapped.capture_camera()
                frames.append({k: v.detach().cpu() for k, v in camera.items()})
                action = policy(obs)
                actions.append(action.detach().cpu())
                # Store only robot state that is observable at deployment.
                # ``obs["policy"]`` contains object pose and goal error for
                # the privileged state teacher and must not become student
                # input by accident.
                student_proprio = env.unwrapped.compute_student_proprio()
                proprio.append(student_proprio.detach().cpu())
                labels.append(env.unwrapped.target_face.detach().cpu())
            else:
                action = policy(obs)
            obs, _, dones, _ = env.step(action)
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        done_tensor = dones[0] if isinstance(dones, tuple) else dones
        completed += int(torch.as_tensor(done_tensor).sum().item())
        step += 1

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)) or ".", exist_ok=True)
    torch.save(
        {
            "frames": frames,
            "actions": actions,
            "student_proprio": proprio,
            "target_face": labels,
            "instruction_templates": [
                "show the {face} marker",
                "show the {face} marker and keep it visible",
            ],
            # The visual scene renders these colors as six face patches.  The
            # language target is therefore grounded in RGB evidence rather
            # than being a hidden face-id token.
            "face_names": ["red", "green", "blue", "yellow", "magenta", "cyan"],
            "task": args_cli.task,
            "episodes": completed,
            "stride": args_cli.stride,
        },
        args_cli.output,
    )
    print(f"[INFO] Wrote {len(frames)} frames from {completed} episodes to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
