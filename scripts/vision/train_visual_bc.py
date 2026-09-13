"""Train a small RGB-D + language + proprioception behavior-cloning baseline.

This is a deliberately compact sanity-check model for the first visual stage:
it consumes the camera frame, a color-language token, and deployment-visible
robot proprioception, then emits the same 21-D action as the RL teacher.  It is
not the final VLA; it verifies that the collected shard has a usable
image/language/action path before adding temporal memory or a larger VLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class RolloutDataset(Dataset):
    def __init__(self, path: str, max_samples: int | None = None):
        data = torch.load(path, map_location="cpu", weights_only=False)
        images, proprio, language, actions = [], [], [], []
        for frame, prop, label, action in zip(
            data["frames"], data["student_proprio"], data["target_face"], data["actions"]
        ):
            rgb = frame["rgb"].to(torch.float32) / 255.0
            depth = frame["depth"].to(torch.float32).clamp(0.0, 2.0) / 2.0
            if depth.ndim == 3:
                depth = depth.unsqueeze(-1)
            # Keep the batch/environment dimension; flatten it below.
            images.append(torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2))
            proprio.append(prop.to(torch.float32))
            language.append(torch.nn.functional.one_hot(label.long(), num_classes=6).float())
            actions.append(action.to(torch.float32))
        self.images = torch.cat(images)
        self.proprio = torch.cat(proprio)
        self.language = torch.cat(language)
        self.actions = torch.cat(actions)
        if max_samples is not None and len(self.actions) > max_samples:
            keep = torch.randperm(len(self.actions))[:max_samples]
            self.images, self.proprio = self.images[keep], self.proprio[keep]
            self.language, self.actions = self.language[keep], self.actions[keep]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.images[index], self.language[index], self.proprio[index], self.actions[index]


class VisualBC(nn.Module):
    def __init__(self, proprio_dim: int = 128, action_dim: int = 21):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU())
        self.language = nn.Sequential(nn.Linear(6, 32), nn.ELU())
        self.head = nn.Sequential(nn.Linear(128 + 128 + 32, 256), nn.ELU(), nn.Linear(256, action_dim))

    def forward(self, image, language, proprio):
        return self.head(torch.cat((self.visual(image), self.language(language), self.proprio(proprio)), dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dataset = RolloutDataset(args.data, args.max_samples or None)
    n_train = max(1, int(0.8 * len(dataset)))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisualBC(proprio_dim=dataset.proprio.shape[-1], action_dim=dataset.actions.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for image, language, proprio, action in train_loader:
            pred = model(image.to(device), language.to(device), proprio.to(device))
            loss = loss_fn(pred, action.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss += loss.item() * len(action)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for image, language, proprio, action in val_loader:
                val_loss += loss_fn(model(image.to(device), language.to(device), proprio.to(device)), action.to(device)).item() * len(action)
        row = {"epoch": epoch + 1, "train_mse": train_loss / n_train, "val_mse": val_loss / max(n_val, 1)}
        history.append(row); print(json.dumps(row), flush=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "proprio_dim": dataset.proprio.shape[-1], "action_dim": dataset.actions.shape[-1], "history": history}, out)
    report = {"data": args.data, "samples": len(dataset), "train_samples": n_train, "val_samples": n_val, "device": str(device), "history": history, "model": str(out)}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
