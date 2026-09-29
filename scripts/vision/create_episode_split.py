"""Create and persist a deterministic episode-level train/validation split."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val fraction must be between zero and one")
    source = Path(args.data)
    data = torch.load(source, map_location="cpu", weights_only=False)
    env = torch.cat([torch.as_tensor(x).reshape(-1) for x in data["env_id"]]).long()
    episode = torch.cat([torch.as_tensor(x).reshape(-1) for x in data["episode_id"]]).long()
    groups = sorted({(int(e), int(ep)) for e, ep in zip(env, episode)})
    if len(groups) < 2:
        raise ValueError(f"need at least two episodes, found {len(groups)}")
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(groups), generator=generator).tolist()
    val_count = max(1, int(round(len(groups) * args.val_fraction)))
    val_indices = set(permutation[:val_count])
    val = sorted([list(groups[i]) for i in val_indices])
    train = sorted([list(groups[i]) for i in range(len(groups)) if i not in val_indices])
    report = {
        "source": str(source),
        "source_sha256": sha256(source),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "episodes": len(groups),
        "train_episodes": len(train),
        "val_episodes": len(val),
        "train": train,
        "val": val,
        "freeze_id": data.get("freeze_id"),
        "checkpoint_sha256": data.get("checkpoint_sha256", data.get("source_checkpoint_sha256")),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k not in {"train", "val"}}, indent=2))


if __name__ == "__main__":
    main()
