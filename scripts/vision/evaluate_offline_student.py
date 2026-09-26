"""Evaluate a visual student on a frozen feature cache.

This is an offline action-prediction check only. It never reports closed-loop
control success and never replaces ``evaluate_feature_student.py`` with a live
RGB-D evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_feature_student import FeatureSequenceDataset
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    history = int(ckpt.get("history", 8))
    # Build the dataset with the checkpoint's history so the offline evaluator
    # cannot silently score a different temporal contract.
    dataset = FeatureSequenceDataset(args.data, history=history, split=args.split, seed=args.seed)
    model = VisualLanguageStudent(
        ckpt["rgb_dim"], ckpt["language_dim"], ckpt["proprio_dim"], ckpt["action_dim"],
        memory_mode=ckpt.get("memory_mode", "plain")
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    squared = 0.0
    absolute = 0.0
    count = 0
    out_of_bounds = 0
    with torch.inference_mode():
        for rgb, lang, prop, action in loader:
            pred = model(VisualStudentBatch(rgb.to(args.device), lang.to(args.device), prop.to(args.device)))["action"]
            target = action.to(args.device)
            squared += float((pred - target).square().sum().item())
            absolute += float((pred - target).abs().sum().item())
            out_of_bounds += int(((pred < -1.0) | (pred > 1.0)).sum().item())
            count += int(pred.numel())
    report = {
        "scope": "offline feature action prediction only",
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "seed": args.seed,
        "samples": len(dataset),
        "action_elements": count,
        "mse": squared / max(count, 1),
        "mae": absolute / max(count, 1),
        "predicted_out_of_bounds_fraction": out_of_bounds / max(count, 1),
        "action_scale": ckpt.get("action_scale"),
        "freeze_id": ckpt.get("freeze_id"),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
