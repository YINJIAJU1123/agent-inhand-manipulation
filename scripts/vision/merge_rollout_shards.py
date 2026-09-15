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
    for key in list_keys:
        values = [shard[key] for shard in shards if key in shard]
        if values:
            merged[key] = sum(values, [])
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
