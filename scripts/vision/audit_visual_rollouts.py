"""Audit visual rollout integrity and teacher trajectory quality.

The collector stores one list entry per captured simulation tick, with each
entry containing a batch for all vectorized environments.  This audit checks
alignment, episode boundaries, camera validity, action saturation, duplicate
keys, and (when available) per-episode quality records.  It never changes the
source shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_flat(values, dtype=None):
    if not values:
        return torch.empty(0, dtype=dtype or torch.float32)
    tensors = [torch.as_tensor(x) for x in values]
    return torch.cat([x.reshape(-1) for x in tensors])



def _cat_samples(values):
    tensors = [torch.as_tensor(x) for x in values]
    return torch.cat([x.reshape(x.shape[0], -1) for x in tensors], dim=0)

def _batch_lengths(data: dict, keys: list[str]) -> dict:
    lengths = {}
    for key in keys:
        values = data.get(key)
        if values is None:
            continue
        if key == "frames":
            lengths[key] = [int(torch.as_tensor(x["rgb"]).shape[0]) for x in values]
        elif key == "instructions":
            lengths[key] = [len(x) for x in values]
        else:
            lengths[key] = [int(torch.as_tensor(x).shape[0]) for x in values]
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-stride", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.data)
    data = torch.load(source, map_location="cpu", weights_only=False)
    required = ["frames", "actions", "student_proprio", "target_face", "instructions",
                "episode_id", "step_index", "env_id", "terminal"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"rollout is missing required keys: {missing}")

    list_keys = ["frames", "actions", "teacher_actions", "student_proprio", "target_face",
                 "instructions", "episode_id", "step_index", "env_id", "terminal"]
    list_lengths = {key: len(data[key]) for key in list_keys if key in data}
    n_batches = len(data["actions"])
    batch_lengths = _batch_lengths(data, list_keys)
    batch_shape_mismatches = []
    for batch_idx in range(n_batches):
        observed = {key: batch_lengths[key][batch_idx] for key in batch_lengths if len(batch_lengths[key]) > batch_idx}
        if len(set(observed.values())) > 1:
            batch_shape_mismatches.append({"batch": batch_idx, "lengths": observed})

    actions = _cat_samples(data["actions"])
    teacher_actions = _cat_samples(data.get("teacher_actions", data["actions"]))
    episode_id = _as_flat(data["episode_id"], torch.long).long()
    step_index = _as_flat(data["step_index"], torch.long).long()
    env_id = _as_flat(data["env_id"], torch.long).long()
    terminal = _as_flat(data["terminal"], torch.bool).bool()
    target_face = _as_flat(data["target_face"], torch.long).long()
    n_samples = int(actions.shape[0])
    if not all(int(x.shape[0]) == n_samples for x in [teacher_actions, episode_id, step_index, env_id, terminal, target_face]):
        raise ValueError("flattened rollout fields have different sample counts")

    key_counts = Counter((int(e), int(ep), int(step)) for e, ep, step in zip(env_id, episode_id, step_index))
    duplicate_keys = sum(count - 1 for count in key_counts.values() if count > 1)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (e, ep) in enumerate(zip(env_id.tolist(), episode_id.tolist())):
        groups[(int(e), int(ep))].append(i)
    quality_records = data.get("episode_quality", [])
    quality_keys = {(int(row["env_id"]), int(row["episode_id"])) for row in quality_records}
    checked_keys = quality_keys or set(groups)

    episode_rows = []
    incomplete = 0
    stride_mismatches = 0
    terminal_mismatches = 0
    observed_incomplete = 0
    for (env, ep), indices in sorted(groups.items()):
        ordered = sorted(indices, key=lambda i: int(step_index[i]))
        steps = [int(step_index[i]) for i in ordered]
        diffs = [b - a for a, b in zip(steps, steps[1:])]
        expected_stride = args.expected_stride or int(data.get("stride", 1))
        contiguous = bool(steps) and 0 <= steps[0] < expected_stride and all(diff == expected_stride for diff in diffs)
        terminal_indices = [i for i in ordered if bool(terminal[i])]
        is_checked = (env, ep) in checked_keys
        if not contiguous:
            observed_incomplete += 1
            if is_checked:
                incomplete += 1
        if diffs and any(diff != expected_stride for diff in diffs) and is_checked:
            stride_mismatches += 1
        if len(terminal_indices) != 1 and is_checked:
            terminal_mismatches += 1
        faces = sorted(set(int(target_face[i]) for i in ordered))
        episode_rows.append({
            "env_id": env,
            "episode_id": ep,
            "samples": len(ordered),
            "first_step": steps[0] if steps else None,
            "last_step": steps[-1] if steps else None,
            "step_diffs": sorted(set(diffs)),
            "contiguous_at_stride": contiguous,
            "terminal_count": len(terminal_indices),
            "faces": faces,
            "quality_record": is_checked,
        })

    valid_depth = []
    rgb_std = []
    for frame_batch in data["frames"]:
        rgb = torch.as_tensor(frame_batch["rgb"]).float()
        depth = torch.as_tensor(frame_batch["depth"]).float()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        # Per-sample camera checks, kept compact as summary quantiles.
        rgb_std.extend(rgb.reshape(rgb.shape[0], -1).std(dim=1).tolist())
        valid_depth.extend(torch.isfinite(depth).logical_and(depth > 0).reshape(depth.shape[0], -1).float().mean(dim=1).tolist())

    def quantiles(values):
        if not values:
            return {"min": None, "p05": None, "median": None, "p95": None, "max": None}
        x = torch.tensor(values, dtype=torch.float64)
        q = torch.quantile(x, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64))
        return {k: float(v) for k, v in zip(("min", "p05", "median", "p95", "max"), q)}

    quality = data.get("episode_quality", [])
    quality_summary = {}
    if quality:
        quality_summary = {
            "records": len(quality),
            "success": sum(bool(x.get("success", False)) for x in quality),
            "drop": sum(bool(x.get("drop", False)) for x in quality),
            "timeout": sum(bool(x.get("timeout", False)) for x in quality),
            "usable": sum(bool(x.get("success", False)) and not bool(x.get("drop", False)) for x in quality),
        }
    report = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "task": data.get("task"),
        "freeze_id": data.get("freeze_id"),
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "requested_episodes": data.get("requested_episodes", data.get("episodes")),
        "completed_episodes": int(data.get("episodes", len(groups))),
        "observed_episodes": len(groups),
        "quality_record_episodes": len(quality_keys),
        "uncompleted_observed_episodes": max(0, len(groups) - len(quality_keys)) if quality_keys else 0,
        "captured_batches": n_batches,
        "samples": n_samples,
        "stride": int(data.get("stride", 1)),
        "list_lengths": list_lengths,
        "batch_shape_mismatches": batch_shape_mismatches[:20],
        "duplicate_sample_keys": duplicate_keys,
        "episodes_with_noncontiguous_steps": incomplete,
        "observed_episodes_with_noncontiguous_steps": observed_incomplete,
        "episodes_with_stride_mismatch": stride_mismatches,
        "episodes_with_terminal_count_not_one": terminal_mismatches,
        "camera": {
            "rgb_std": quantiles(rgb_std),
            "valid_depth_fraction": quantiles(valid_depth),
            "low_rgb_std_count": sum(x < 0.01 for x in rgb_std),
            "low_depth_fraction_count": sum(x < 0.5 for x in valid_depth),
        },
        "actions": {
            "bounded_fraction": float((actions.abs() <= 1.0 + 1e-6).float().mean()) if n_samples else None,
            "teacher_out_of_bounds_fraction": float((teacher_actions.abs() > 1.0 + 1e-6).float().mean()) if n_samples else None,
            "student_action_min": float(actions.min()) if n_samples else None,
            "student_action_max": float(actions.max()) if n_samples else None,
        },
        "episode_quality": quality_summary,
        "episode_rows": episode_rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "episode_rows"}, indent=2))


if __name__ == "__main__":
    main()
