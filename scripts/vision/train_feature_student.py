"""Train the first recurrent student on cached frozen VLM features.

This is the reproducible baseline between the privileged state teacher and a
later on-policy visual-language policy.  The image and instruction encoders
are frozen upstream; only the GRU, proprioception projection and 21-D action
head are optimized here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "BrainCo_DexHand"))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train = FeatureSequenceDataset(args.data, args.history, "train", args.seed)
    val = FeatureSequenceDataset(args.data, args.history, "val", args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device(args.device)
    model = VisualLanguageStudent(
        rgb_dim=train.rgb.shape[-1], language_dim=train.lang.shape[-1],
        proprio_dim=train.proprio.shape[-1], action_dim=train.actions.shape[-1],
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
    torch.save({"model": model.cpu().state_dict(), "rgb_dim": train.rgb.shape[-1],
                "language_dim": train.lang.shape[-1], "proprio_dim": train.proprio.shape[-1],
                "action_dim": train.actions.shape[-1], "history": args.history,
                "metrics": history}, output)
    report = {"data": args.data, "output": str(output), "train_samples": len(train),
              "val_samples": len(val), "device": str(device), "metrics": history}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
