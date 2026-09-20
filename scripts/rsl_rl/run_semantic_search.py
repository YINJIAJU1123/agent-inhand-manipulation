"""Train/evaluate recurrent RGB-D PPO and an image-only scan baseline.

v0 uses a fixed spatial RGB-D encoder plus color statistics and a small explicit
language grammar. No pretrained vision or language model, teacher imitation,
or object-state actor input is used. Results are engineering pilots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["smoke", "train", "eval", "scan"], required=True)
parser.add_argument("--initial", choices=["visible", "hidden", "mixed"], default="mixed")
parser.add_argument("--split", choices=["train", "val", "test"], default="train")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--steps-per-env", type=int, default=32)
parser.add_argument("--episodes", type=int, default=96)
parser.add_argument("--checkpoint")
parser.add_argument("--output", required=True)
parser.add_argument("--episode-seconds", type=float, default=20.)
parser.add_argument("--hold-seconds", type=float, default=1.)
parser.add_argument("--save-interval", type=int, default=50)
parser.add_argument("--object-split", choices=["legacy", "train", "val", "test"], default="legacy")
parser.add_argument("--objects", nargs="+", help="Optional object IDs within the declared split")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
import BrainCo_DexHand  # noqa: F401, E402
from BrainCo_DexHand.tasks.direct.brainco.brainco_hand_search_env_cfg import BrainCoHandSearchEnvCfg  # noqa: E402
from BrainCo_DexHand.algo.agentic.search_protocol import ScanRecognizeHold, parse_instruction  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def runner_config():
    return {
        "seed": args.seed, "device": args.device or "cuda:0",
        "num_steps_per_env": args.steps_per_env, "save_interval": args.save_interval,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "logger": "tensorboard", "experiment_name": "semantic_search_v0",
        "policy": {
            "class_name": "ActorCriticRecurrent", "init_noise_std": .35,
            "noise_std_type": "log", "actor_obs_normalization": True,
            "critic_obs_normalization": True, "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128], "activation": "elu",
            "rnn_type": "gru", "rnn_hidden_dim": 256, "rnn_num_layers": 1,
        },
        "algorithm": {
            "class_name": "PPO", "value_loss_coef": 1., "use_clipped_value_loss": True,
            "clip_param": .2, "entropy_coef": .005, "num_learning_epochs": 4,
            "num_mini_batches": min(4, args.num_envs), "learning_rate": 3e-4,
            "schedule": "adaptive", "gamma": .99, "lam": .95,
            "desired_kl": .01, "max_grad_norm": 1.,
        },
    }


class LoggingWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, output):
        self.output = output
        self.control_steps = 0
        self.completed = 0
        self.terminal_successes = 0
        self.drops = 0
        self.object_totals = {}
        self.start = time.monotonic()
        super().__init__(env, clip_actions=1.)

    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        self.control_steps += 1
        self.completed += int(done.sum())
        metrics = extras["search_metrics"]
        finished = done.bool()
        self.terminal_successes += int((metrics["held_at_end"] & finished & ~metrics["dropped"]).sum())
        self.drops += int((metrics["dropped"] & finished).sum())
        for slot in finished.nonzero(as_tuple=False).flatten().tolist():
            name = self.unwrapped.object_names[slot]
            totals = self.object_totals.setdefault(name, {"episodes": 0, "successes": 0, "drops": 0})
            totals["episodes"] += 1
            totals["successes"] += int(metrics["held_at_end"][slot] & ~metrics["dropped"][slot])
            totals["drops"] += int(metrics["dropped"][slot])
        if self.control_steps % 32 == 0:
            json_write(self.output / "heartbeat.json", {
                "unix_time": time.time(), "pid": os.getpid(), "control_steps": self.control_steps,
                "environment_steps": self.control_steps * self.num_envs,
                "completed_episodes": self.completed, "terminal_successes": self.terminal_successes,
                "drops": self.drops, "elapsed_s": time.monotonic() - self.start,
                "per_object": self.object_totals,
                "scope": "online training/rollout totals; not held-out evaluation",
            })
        return obs, reward, done, extras


def save_frames(raw, output, prefix):
    frames = raw.capture_camera()
    fraction, area, facing, _ = raw._surface_metrics()
    rows = []
    for i in range(min(raw.num_envs, 8)):
        Image.fromarray(frames["rgb"][i].cpu().numpy()).save(output / f"{prefix}_{i:02d}.png")
        face = int(raw.target_face[i])
        rows.append({"env": i, "instruction": raw.current_instructions()[i],
                     "object_id": raw.object_names[i],
                     "layout": raw.face_colors[i].tolist(), "target_face": face,
                     "initial_hidden": bool(raw.initial_hidden[i]),
                     "visible_fraction": float(fraction[i, face]),
                     "projected_area": float(area[i, face]), "facing_cosine": float(facing[i, face])})
    json_write(output / f"{prefix}.json", rows)


def smoke(env, output):
    raw = env.unwrapped
    cases = []
    for initial in ("visible", "hidden"):
        raw.cfg.initial_visibility = initial
        obs, _ = env.reset()
        assert obs["policy"].shape == (args.num_envs, 1111)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.equal(obs["policy"][:, -6:].argmax(-1), raw.target_color)
        for i, text in enumerate(raw.current_instructions()):
            assert parse_instruction(text) == int(raw.target_color[i])
        fraction, area, facing, _ = raw._surface_metrics()
        row = torch.arange(raw.num_envs, device=raw.device)
        if initial == "visible":
            assert (fraction[row, raw.target_face] >= raw.cfg.minimum_visible_fraction).all()
        else:
            assert (fraction[row, raw.target_face] <= .02).all()
            assert (facing[row, raw.target_face] < -.2).all()
        save_frames(raw, output, f"reset_{initial}")
        # Holding current joint positions avoids interpreting zero normalized
        # commands as a physically neutral action.
        ids = raw.actuated_dof_indices
        low, high = raw.hand_dof_lower_limits[:, ids], raw.hand_dof_upper_limits[:, ids]
        action = (2 * (raw.hand_dof_pos[:, ids] - low) / (high - low) - 1).clamp(-1, 1)
        for _ in range(12):
            obs, reward, terminated, truncated, info = env.step(action)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            assert "search_metrics" in info
        save_frames(raw, output, f"after_{initial}")
        cases.append({"initial": initial, "passed": True})
    # Force a timeout to test pre-reset metrics vs post-reset visual/language obs.
    raw.episode_length_buf[:] = raw.max_episode_length - 1
    previous_ids = raw.episode_id.clone()
    obs, reward, terminated, truncated, info = env.step(action)
    assert (terminated | truncated).all()
    assert torch.equal(info["search_metrics"]["episode_id"], previous_ids)
    assert (raw.episode_id > previous_ids).all()
    assert torch.equal(obs["policy"][:, -6:].argmax(-1), raw.target_color)
    json_write(output / "smoke.json", {"passed": True, "cases": cases,
                                       "reset_rejections": raw.reset_rejections,
                                       "scope": "protocol/rendering only; not trained performance"})


def evaluate(env, runner, output):
    actor = runner.get_inference_policy(device=env.device)
    model = runner.alg.policy
    model.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    obs = env.get_observations()
    scan = ScanRecognizeHold(env.num_envs, env.device) if args.mode == "scan" else None
    previous_goal = None
    quotas = [args.episodes // env.num_envs + (i < args.episodes % env.num_envs) for i in range(env.num_envs)]
    counts = [0] * env.num_envs
    records = []
    # Per-slot quotas prevent early drop episodes from dominating the sample.
    max_steps = (max(quotas) + 1) * env.max_episode_length
    with torch.inference_mode():
        for _ in range(max_steps):
            if scan is not None:
                requested = obs["policy"][:, -6:].argmax(-1)
                goal = scan(env.unwrapped.capture_camera()["rgb"], requested)
                if previous_goal is not None:
                    model.reset(goal != previous_goal)
                previous_goal = goal.clone()
                obs = obs.clone()
                obs["policy"][:, -6:] = torch.nn.functional.one_hot(goal, 6).float()
            actions = actor(obs)
            obs, reward, done, extras = env.step(actions)
            model.reset(done)
            if scan is not None:
                scan.reset(done.bool())
            metrics = extras["search_metrics"]
            for slot in done.nonzero(as_tuple=False).flatten().tolist():
                if counts[slot] >= quotas[slot]:
                    continue
                rec = {k: v[slot].tolist() for k, v in metrics.items()}
                rec.update(slot=slot, evaluation_seed=args.seed, layout_split=args.split,
                           object_id=env.unwrapped.object_names[slot], object_split=args.object_split,
                           method=args.mode, success=bool(metrics["held_at_end"][slot] & ~metrics["dropped"][slot]))
                counts[slot] += 1
                records.append(rec)
                with (output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(rec, allow_nan=False) + "\n")
            if len(records) == args.episodes:
                break
    assert len(records) == args.episodes, (len(records), args.episodes)
    summary = {"episodes": len(records), "method": args.mode, "initial": args.initial,
               "layout_split": args.split, "seed": args.seed, "checkpoint_sha256": sha256(args.checkpoint),
               "successes": sum(r["success"] for r in records),
               "ever_display": sum(r["ever_display"] for r in records),
               "ever_exposed": sum(r["ever_exposed"] for r in records),
               "drops": sum(r["dropped"] for r in records), "step_dt": env.unwrapped.step_dt}
    summary["success_rate"] = summary["successes"] / len(records)
    summary["object_split"] = args.object_split
    summary["per_object"] = {}
    for name in sorted(set(r["object_id"] for r in records)):
        rows = [r for r in records if r["object_id"] == name]
        summary["per_object"][name] = {
            "episodes": len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows),
            "drops": sum(r["dropped"] for r in rows), "ever_exposed": sum(r["ever_exposed"] for r in rows)}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists() or (output / "episodes.jsonl").exists():
        raise RuntimeError(f"Use a fresh output directory to preserve prior runs: {output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    cfg = BrainCoHandSearchEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.initial_visibility, cfg.layout_split = args.initial, args.split
    cfg.episode_length_s, cfg.goal_hold_time_s = args.episode_seconds, args.hold_seconds
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    specs = None
    if args.object_split != "legacy":
        from BrainCo_DexHand.algo.agentic.object_catalog import select_objects, PATCH_SIZE, PATCH_THICKNESS
        from BrainCo_DexHand.assets.search_objects import configure_objects
        specs = select_objects(args.object_split, args.objects)
        if args.mode == "train" and (args.object_split != "train" or args.split != "train"):
            raise ValueError("Training must use training objects and marker layouts")
        if args.mode in ("eval", "scan") and args.episodes % args.num_envs:
            raise ValueError("Multi-object evaluation requires equal per-slot episode quotas")
        configure_objects(cfg, specs)
        for marker in cfg.face_marker_cfg.markers.values():
            marker.size = (PATCH_SIZE, PATCH_SIZE, PATCH_THICKNESS)
    elif args.objects:
        raise ValueError("--objects requires a non-legacy --object-split")
    json_write(output / "manifest.json", {
        "args": vars(args), "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "actor_inputs": ["pooled RGB-D", "RGB color statistics", "joint position/velocity", "previous action", "instruction color token"],
        "privileged": "object pose/velocity and target normal for critic; geometry/depth for reward and scoring",
        "human_task_demonstrations": 0, "pretraining": None,
        "objects": specs, "object_assignment": "fixed balanced slots; read actual USD identity",
        "visual_frontend": "fixed RGB-D pooling and color statistics; no distillation ablation",
        "protocol": "semantic_search_v0_pilot", "unix_time": time.time(),
    })
    dump_yaml(str(output / "env.yaml"), cfg)
    json_write(output / "runner.json", runner_config())
    env = gym.make("BrainCo-Direct-Revo3-SemanticSearch-Cube-v0", cfg=cfg)
    try:
        if args.mode == "smoke":
            smoke(env, output)
        else:
            env = LoggingWrapper(env, output)
            runner = OnPolicyRunner(env, runner_config(), log_dir=str(output / "train"), device=cfg.sim.device)
            if args.checkpoint:
                runner.load(args.checkpoint, load_optimizer=args.mode == "train")
            if args.mode == "train":
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
                runner.save(str(output / "final.pt"))
            else:
                if not args.checkpoint:
                    raise ValueError("Evaluation and scan require a controller checkpoint")
                evaluate(env, runner, output)
        json_write(output / "status.json", {"status": "completed", "unix_time": time.time()})
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        json_write(output / "failure.json", {"traceback": traceback.format_exc(), "unix_time": time.time()})
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Kit fast shutdown must not turn a Python failure into exit 0.
    else:
        sys.stdout.flush()
        app.close()
