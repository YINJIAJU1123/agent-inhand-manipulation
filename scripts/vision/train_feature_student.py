"""Train the first recurrent student on cached frozen VLM features.

This is the reproducible baseline between the privileged state teacher and a
later on-policy visual-language policy.  The image and instruction encoders
are frozen upstream; only the GRU, proprioception projection and 21-D action
head are optimized here.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "BrainCo_DexHand"))
DEFAULT_FREEZE_MANIFEST = ROOT / "configs" / "visual_student_freeze.json"
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch  # noqa: E402


class FeatureSequenceDataset(Dataset):
    def __init__(self, path: str, history: int = 8, split: str = "train", seed: int = 0):
        if history <= 0:
            raise ValueError("history must be positive")
        data = torch.load(path, map_location="cpu", weights_only=False)
        n = int(data["actions"].shape[0])
        env_id = data.get("env_id", torch.zeros(n, dtype=torch.long)).reshape(-1)
        episode_id = data.get("episode_id", torch.zeros(n, dtype=torch.long)).reshape(-1)
        keys = [(int(e), int(ep)) for e, ep in zip(env_id, episode_id)]
        groups = sorted(set(keys))
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(groups), generator=generator).tolist()
        cut = max(1, int(0.8 * len(groups)))
        selected = set(groups[i] for i in permutation[:cut]) if split == "train" else set(groups[i] for i in permutation[cut:])
        if split == "val" and not selected:
            selected = set(groups[-1:])

        order = sorted(range(n), key=lambda i: (keys[i], int(data["step_index"].reshape(-1)[i])))
        self.rgb, self.lang, self.proprio, self.actions = [], [], [], []
        last_key = None
        rgb_hist, lang_hist, prop_hist = [], [], []
        for index in order:
            key = keys[index]
            if key not in selected:
                continue
            if key != last_key:
                rgb_hist, lang_hist, prop_hist = [], [], []
                last_key = key
            rgb_hist.append(data["image_features"][index].float())
            lang_hist.append(data["language_features"][index].float())
            prop_hist.append(data["student_proprio"][index].float())
            while len(rgb_hist) < history:
                rgb_hist.insert(0, rgb_hist[0].clone())
                lang_hist.insert(0, lang_hist[0].clone())
                prop_hist.insert(0, prop_hist[0].clone())
            self.rgb.append(torch.stack(rgb_hist[-history:]))
            self.lang.append(torch.stack(lang_hist[-history:]))
            self.proprio.append(torch.stack(prop_hist[-history:]))
            self.actions.append(data["actions"][index].float())
        if not self.actions:
            raise RuntimeError(f"no samples found for split={split} in {path}")
        self.rgb = torch.stack(self.rgb)
        self.lang = torch.stack(self.lang)
        self.proprio = torch.stack(self.proprio)
        self.actions = torch.stack(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.rgb[index], self.lang[index], self.proprio[index], self.actions[index]


def _load_freeze_manifest(path: str | None) -> dict:
    manifest_path = Path(path) if path else DEFAULT_FREEZE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"freeze manifest not found: {manifest_path}")
    import json as _json
    manifest = _json.loads(manifest_path.read_text())
    contract = manifest.get("action_contract", {})
    if contract.get("action_dim") != 21:
        raise ValueError("visual student freeze requires the 21-D Revo3 action contract")
    if float(contract.get("action_scale", 0.0)) <= 0:
        raise ValueError("freeze manifest must define a positive action scale")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--memory-mode", choices=("plain", "evidence"), default="plain")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest = _load_freeze_manifest(args.freeze_manifest)
    torch.manual_seed(args.seed)
    train = FeatureSequenceDataset(args.data, args.history, "train", args.seed)
    val = FeatureSequenceDataset(args.data, args.history, "val", args.seed)
    if train.actions.shape[-1] != int(manifest["action_contract"]["action_dim"]):
        raise ValueError(
            f"dataset action dimension {train.actions.shape[-1]} does not match frozen contract "
            f"{manifest['action_contract']['action_dim']}"
        )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device(args.device)
    model = VisualLanguageStudent(
        rgb_dim=train.rgb.shape[-1], language_dim=train.lang.shape[-1],
        proprio_dim=train.proprio.shape[-1], action_dim=train.actions.shape[-1],
        memory_mode=args.memory_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for rgb, lang, prop, action in train_loader:
            out = model(VisualStudentBatch(rgb.to(device), lang.to(device), prop.to(device)))
            loss = torch.nn.functional.mse_loss(out["action"], action.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss += float(loss.item()) * len(action)
        model.eval(); val_loss = 0.0
        with torch.inference_mode():
            for rgb, lang, prop, action in val_loader:
                out = model(VisualStudentBatch(rgb.to(device), lang.to(device), prop.to(device)))
                val_loss += float(torch.nn.functional.mse_loss(out["action"], action.to(device)).item()) * len(action)
        row = {"epoch": epoch + 1, "train_mse": train_loss / len(train), "val_mse": val_loss / len(val)}
        history.append(row); print(json.dumps(row), flush=True)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    data_sha256 = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    torch.save({"model": model.cpu().state_dict(), "rgb_dim": train.rgb.shape[-1],
                "language_dim": train.lang.shape[-1], "proprio_dim": train.proprio.shape[-1],
                "action_dim": train.actions.shape[-1], "history": args.history,
                "action_scale": float(manifest["action_contract"]["action_scale"]),
                "memory_mode": args.memory_mode,
                "freeze_id": manifest["freeze_id"],
                "freeze_manifest": str(Path(args.freeze_manifest).resolve()),
                "data_sha256": data_sha256,
                "metrics": history}, output)
    report = {"data": args.data, "data_sha256": data_sha256, "output": str(output),
              "train_samples": len(train), "val_samples": len(val), "device": str(device),
              "seed": args.seed, "history": args.history,
              "action_scale": float(manifest["action_contract"]["action_scale"]),
              "memory_mode": args.memory_mode, "freeze_id": manifest["freeze_id"], "metrics": history}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
