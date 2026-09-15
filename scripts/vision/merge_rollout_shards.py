"""Merge per-goal RGB-D rollout shards without mixing episode ids."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+", help="rollout .pt shards")
    args = parser.parse_args()
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    if not shards:
        raise ValueError("at least one shard is required")
    merged = {}
    list_keys = {
        "frames", "actions", "teacher_actions", "student_proprio", "target_face",
        "instructions", "episode_id", "step_index", "env_id", "terminal",
        "transition_goal_reached", "transition_dropped",
    }
    # Each collection process restarts its environment ids at zero. Allocate a
    # disjoint id range per shard so the trainer cannot join histories from
    # different shards into one fictitious episode.
    env_offset = 0
    for shard in shards:
        shard_env = shard.get("env_id")
        if shard_env is None:
            shard_env = [torch.zeros_like(x) for x in shard["episode_id"]]
        max_env = max((int(torch.as_tensor(x).max().item()) for x in shard_env), default=-1)
        for key in list_keys:
            if key not in shard:
                continue
            values = shard[key]
            if key == "env_id":
                values = [torch.as_tensor(x).clone() + env_offset for x in values]
            merged.setdefault(key, []).extend(values)
        env_offset += max_env + 1
    merged.update({key: shards[0][key] for key in ("instruction_templates", "face_names", "task", "stride", "depth_dtype", "action_storage") if key in shards[0]})
    merged["episodes"] = sum(int(shard.get("episodes", 0)) for shard in shards)
    merged["shards"] = [str(path) for path in args.inputs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    samples = sum(int(x.shape[0]) for x in merged["actions"])
    print(f"[INFO] merged {len(shards)} shards, {merged['episodes']} episodes, {samples} samples -> {output}")


if __name__ == "__main__":
    main()
