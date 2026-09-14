"""Record the task's fixed RGB camera while a policy controls the hand.

Unlike ``play.py`` (which records Isaac Sim's viewer), this script writes the
camera sensor stream exposed by ``capture_camera``.  It is intended for
debugging and for short qualitative rollouts of the visual Revo3 task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record RGB camera frames from a Revo3 rollout.")
parser.add_argument("--task", type=str, default="BrainCo-Direct-Revo3-VisualSemanticReorient-Cube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=120)
parser.add_argument("--output", type=str, default="outputs/camera-video")
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
from PIL import Image  # noqa: E402
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)

    output = os.path.abspath(args_cli.output)
    os.makedirs(output, exist_ok=True)
    obs = env.get_observations()

    # Let the tiled camera and renderer settle before saving the first frame.
    with torch.inference_mode():
        for _ in range(5):
            obs, _, dones, _ = env.step(policy(obs).clamp(-1.0, 1.0))
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        for step in range(args_cli.video_length):
            camera = env.unwrapped.capture_camera()
            rgb = camera["rgb"][0].detach().cpu().numpy()
            Image.fromarray(rgb).save(os.path.join(output, f"frame_{step:06d}.png"))
            obs, _, dones, _ = env.step(policy(obs).clamp(-1.0, 1.0))
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

    with open(os.path.join(output, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"task": args_cli.task, "frames": args_cli.video_length, "camera": "semantic_camera", "fps": 30}, f, indent=2)
    print(f"[INFO] Wrote {args_cli.video_length} RGB camera frames to {output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
