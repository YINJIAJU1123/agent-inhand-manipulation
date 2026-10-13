"""Train a frozen visual student's small SigLIP2 language adapter.

The RGB-D frontend, recurrent memory and action head are loaded from a
validated visual student checkpoint.  Only language-facing fusion layers are
updated against the existing teacher actions while each target face is paired
with several equivalent natural-language prompts.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "source" / "BrainCo_DexHand"))
from BrainCo_DexHand.algo.agentic.visual_student import VisualLanguageStudent, VisualStudentBatch

FACE_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan"]
TEMPLATES = [
    "show the {color} marker",
    "reorient to the {color} marker",
    "turn toward the {color} marker",
    "bring the {color} marker to the top",
    "move the {color} marker into view",
    "place the {color} marker upward",
]

class SequenceDataset(Dataset):
    def __init__(self, data_path, history, split, seed):
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        n = int(data["actions"].shape[0])
        env = data.get("env_id", torch.zeros(n, dtype=torch.long)).reshape(-1)
        ep = data.get("episode_id", torch.zeros(n, dtype=torch.long)).reshape(-1)
        step = data["step_index"].reshape(-1)
        keys = [(int(e), int(p)) for e, p in zip(env, ep)]
        groups = sorted(set(keys))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(groups), generator=g).tolist()
        cut = max(1, int(0.8 * len(groups)))
        selected = set(groups[i] for i in (perm[:cut] if split == "train" else perm[cut:]))
        if split == "val" and not selected:
            selected = set(groups[-1:])
        order = sorted(range(n), key=lambda i: (keys[i], int(step[i])))
        self.rgb, self.prop, self.action, self.face = [], [], [], []
        last = None
        rgb_hist, prop_hist = [], []
        for i in order:
            key = keys[i]
            if key not in selected:
                continue
            if key != last:
                rgb_hist, prop_hist = [], []
                last = key
            rgb_hist.append(data["image_features"][i].float())
            prop_hist.append(data["student_proprio"][i].float())
            while len(rgb_hist) < history:
                rgb_hist.insert(0, rgb_hist[0].clone())
                prop_hist.insert(0, prop_hist[0].clone())
            self.rgb.append(torch.stack(rgb_hist[-history:]))
            self.prop.append(torch.stack(prop_hist[-history:]))
            self.action.append(data["actions"][i].float())
            self.face.append(int(data["target_face"][i].item()))
        if not self.action:
            raise RuntimeError(f"no samples for {split}")
        self.rgb = torch.stack(self.rgb)
        self.prop = torch.stack(self.prop)
        self.action = torch.stack(self.action)
        self.face = torch.tensor(self.face, dtype=torch.long)
    def __len__(self):
        return len(self.action)
    def __getitem__(self, i):
        return self.rgb[i], self.prop[i], self.action[i], self.face[i]

def encode_texts(model, processor, texts, device, batch_size=32):
    outs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inp = processor(text=texts[start:start+batch_size], return_tensors="pt",
                            padding=True, truncation=True)
            inp = {k: v.to(device) for k,v in inp.items() if torch.is_tensor(v)}
            if hasattr(model, "get_text_features"):
                feat = model.get_text_features(**inp)
            else:
                out = model.text_model(input_ids=inp["input_ids"],
                                       attention_mask=inp.get("attention_mask"))
                feat = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
                if hasattr(model, "text_projection"):
                    feat = model.text_projection(feat)
            outs.append(torch.nn.functional.normalize(feat.float(), dim=-1).cpu())
    return torch.cat(outs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--freeze-manifest", required=True)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text-batch-size", type=int, default=32)
    p.add_argument("--anchor-weight", type=float, default=5.0)
    args = p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(args.freeze_manifest).read_text())
    if base.get("freeze_id") and base["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("base checkpoint freeze_id does not match freeze manifest")
    train = SequenceDataset(args.data, args.history, "train", args.seed)
    val = SequenceDataset(args.data, args.history, "val", args.seed)
    model = VisualLanguageStudent(base["rgb_dim"], base["language_dim"], base["proprio_dim"],
                                  base["action_dim"], memory_mode=base.get("memory_mode", "plain"))
    model.load_state_dict(base["model"])
    model.to(device)
    reference = VisualLanguageStudent(base["rgb_dim"], base["language_dim"], base["proprio_dim"],
                                      base["action_dim"], memory_mode=base.get("memory_mode", "plain"))
    reference.load_state_dict(base["model"]); reference.to(device).eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_names = {"lang_proj"}
    trainable = []
    for name, param in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in trainable_names):
            param.requires_grad_(True); trainable.append(name)
    if not trainable:
        raise RuntimeError("no language adapter parameters selected")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    vlm = AutoModel.from_pretrained(args.model).to(device).eval()
    prompts = [template.format(color=color) for template in TEMPLATES for color in FACE_NAMES]
    table = encode_texts(vlm, processor, prompts, device, args.text_batch_size).reshape(len(TEMPLATES), len(FACE_NAMES), -1).to(device)
    for param in vlm.parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    tr_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=False)
    hist = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for rgb, prop, action, face in tr_loader:
            rgb, prop, action, face = rgb.to(device), prop.to(device), action.to(device), face.to(device)
            canonical = table[0, face]
            template = torch.randint(0, len(TEMPLATES), (len(face),), device=device)
            lang = table[template, face]
            out_canonical = model(VisualStudentBatch(rgb, canonical, prop))
            out_augmented = model(VisualStudentBatch(rgb, lang, prop))
            with torch.no_grad():
                reference_out = reference(VisualStudentBatch(rgb, canonical, prop))["action"].detach()
            teacher_loss = 0.5 * torch.nn.functional.mse_loss(out_canonical["action"], action) + 0.5 * torch.nn.functional.mse_loss(out_augmented["action"], action)
            anchor_loss = 0.5 * torch.nn.functional.mse_loss(out_canonical["action"], reference_out) + 0.5 * torch.nn.functional.mse_loss(out_augmented["action"], reference_out)
            loss = teacher_loss + args.anchor_weight * anchor_loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss.item()) * len(face)
        model.eval(); val_total = 0.0; aug_total = 0.0; count = 0
        with torch.inference_mode():
            for rgb, prop, action, face in va_loader:
                rgb, prop, action, face = rgb.to(device), prop.to(device), action.to(device), face.to(device)
                canonical = table[0, face]
                out = model(VisualStudentBatch(rgb, canonical, prop))
                val_total += float(torch.nn.functional.mse_loss(out["action"], action).item()) * len(face)
                template = torch.randint(0, len(TEMPLATES), (len(face),), device=device)
                out = model(VisualStudentBatch(rgb, table[template, face], prop))
                aug_total += float(torch.nn.functional.mse_loss(out["action"], action).item()) * len(face)
                count += len(face)
        row = {"epoch": epoch + 1, "train_mse": total / len(train),
               "val_canonical_mse": val_total / count, "val_augmented_mse": aug_total / count}
        hist.append(row); print(json.dumps(row), flush=True)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    data_sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    base_sha = hashlib.sha256(Path(args.base_checkpoint).read_bytes()).hexdigest()
    payload = dict(base)
    payload["model"] = model.cpu().state_dict()
    payload["language_adapter"] = {
        "version": "siglip2_language_adapter_v1",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": base_sha,
        "templates": TEMPLATES,
        "face_names": FACE_NAMES,
        "language_model": args.model,
        "trainable_modules": sorted(trainable_names),
        "epochs": args.epochs, "seed": args.seed, "metrics": hist,
    }
    payload["data_sha256"] = data_sha
    payload["language_mode"] = "vlm"
    torch.save(payload, out_path)
    report = {"output": str(out_path), "data": args.data, "data_sha256": data_sha,
              "base_checkpoint": args.base_checkpoint, "base_checkpoint_sha256": base_sha,
              "language_model": args.model, "templates": TEMPLATES, "face_names": FACE_NAMES,
              "trainable_modules": sorted(trainable_names), "train_samples": len(train),
              "val_samples": len(val), "epochs": args.epochs, "seed": args.seed,
              "metrics": hist}
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()

