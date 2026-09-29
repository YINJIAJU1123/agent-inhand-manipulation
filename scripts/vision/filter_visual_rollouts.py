"""Filter rollout shards by complete episode-level teacher quality records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_value(value, mask: torch.Tensor):
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == mask.shape[0]:
            return value[mask]
        return value
    if isinstance(value, list) and len(value) == mask.shape[0]:
        return [item for item, keep in zip(value, mask.tolist()) if keep]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=("all", "success_only", "no_drop"), default="success_only")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source = Path(args.data)
    data = torch.load(source, map_location="cpu", weights_only=False)

    quality = data.get("episode_quality", [])
    quality_by_key = {(int(row["env_id"]), int(row["episode_id"])): row for row in quality}
    all_keys = set()
    for envs, episodes in zip(data["env_id"], data["episode_id"]):
        all_keys.update((int(e), int(ep)) for e, ep in zip(torch.as_tensor(envs).reshape(-1), torch.as_tensor(episodes).reshape(-1)))
    if args.policy == "all":
        selected = all_keys
    elif not quality_by_key:
        raise ValueError("quality policy requires episode_quality records in the source shard")
    elif args.policy == "success_only":
        selected = {key for key, row in quality_by_key.items() if bool(row.get("success")) and not bool(row.get("drop")) and not bool(row.get("timeout"))}
    else:
        selected = {key for key, row in quality_by_key.items() if not bool(row.get("drop"))}

    list_keys = {
        "frames", "actions", "teacher_actions", "student_proprio", "target_face", "instructions",
        "episode_id", "step_index", "env_id", "terminal", "transition_goal_reached", "transition_dropped",
    }
    filtered = {key: value for key, value in data.items() if key not in list_keys and key != "episode_quality"}
    for key in list_keys:
        if key not in data:
            continue
        filtered[key] = []

    kept_samples = 0
    for batch_idx, env_batch in enumerate(data["env_id"]):
        env_tensor = torch.as_tensor(env_batch).reshape(-1)
        ep_tensor = torch.as_tensor(data["episode_id"][batch_idx]).reshape(-1)
        mask = torch.tensor([(int(e), int(ep)) in selected for e, ep in zip(env_tensor, ep_tensor)], dtype=torch.bool)
        if not bool(mask.any()):
            continue
        for key in list_keys:
            if key not in data:
                continue
            value = data[key][batch_idx]
            if key == "frames":
                filtered[key].append({name: mask_value(tensor, mask) for name, tensor in value.items()})
            else:
                filtered[key].append(mask_value(value, mask))
        kept_samples += int(mask.sum())

    filtered["episodes"] = len(selected)
    filtered["source_data"] = str(source)
    filtered["source_sha256"] = sha256(source)
    filtered["quality_policy"] = args.policy
    filtered["selected_episode_keys"] = sorted([list(key) for key in selected])
    filtered["episode_quality"] = [row for key, row in quality_by_key.items() if key in selected]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, output)
    report = {
        "source": str(source),
        "source_sha256": filtered["source_sha256"],
        "output": str(output),
        "policy": args.policy,
        "source_episodes": len(all_keys),
        "selected_episodes": len(selected),
        "samples": kept_samples,
        "quality_records": len(filtered["episode_quality"]),
        "freeze_id": data.get("freeze_id"),
        "checkpoint_sha256": data.get("checkpoint_sha256"),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
