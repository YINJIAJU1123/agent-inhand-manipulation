"""Train a linear goal classifier on frozen SigLIP2; evaluate unseen templates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModel, AutoProcessor
from text_grounding import FACE_NAMES, TRAIN_TEMPLATES, HELDOUT_TEMPLATES, encode_texts, ground_features


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--canonical-source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    model.requires_grad_(False)
    train = [t.format(color=c) for t in TRAIN_TEMPLATES for c in FACE_NAMES]
    heldout = [t.format(color=c) for t in HELDOUT_TEMPLATES for c in FACE_NAMES]
    assert not set(train) & set(heldout)
    print('encoding 108 training instructions', flush=True)
    x = encode_texts(model, processor, train, args.device).detach()
    labels = torch.arange(6, device=args.device).repeat(len(TRAIN_TEMPLATES))
    head = torch.nn.Linear(x.shape[-1], 6).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=0.01)
    for step in range(1000):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(x), labels)
        loss.backward()
        opt.step()
    source = torch.load(args.canonical_source, map_location='cpu', weights_only=False)
    adapter = dict(version='siglip2_text_classifier_v2', weight=head.weight.detach().cpu(),
                   bias=head.bias.detach().cpu(), canonical_embeddings=source['canonical_embeddings'],
                   face_names=FACE_NAMES, train_templates=TRAIN_TEMPLATES, max_length=64,
                   language_model=args.model, seed=args.seed,
                   canonical_source_sha256=sha256(args.canonical_source))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter, out)
    print('checkpoint frozen; evaluating 72 held-out instructions', flush=True)
    test_x = encode_texts(model, processor, heldout, args.device)
    _, pred, conf = ground_features(test_x, adapter)
    expected = torch.arange(6, device=args.device).repeat(len(HELDOUT_TEMPLATES))
    confusion = torch.bincount(expected * 6 + pred, minlength=36).reshape(6, 6)
    # Check invariance to batch composition and a reordered text-only request.
    single = encode_texts(model, processor, heldout[:1], args.device)
    batch_error = float((single - test_x[:1]).abs().max())
    _, reverse_pred, _ = ground_features(test_x.flip(0), adapter)
    assert torch.equal(reverse_pred, pred.flip(0))
    assert batch_error < 1e-4, batch_error
    report = dict(version=adapter['version'], checkpoint=str(out), checkpoint_sha256=sha256(out),
                  train_count=len(train), test_count=len(heldout), seed=args.seed,
                  train_accuracy=float((head(x).argmax(-1) == labels).float().mean()),
                  heldout_accuracy=float((pred == expected).float().mean()),
                  batch_invariance_max_abs_error=batch_error, confusion_matrix=confusion.tolist(),
                  train_templates=TRAIN_TEMPLATES, heldout_templates=HELDOUT_TEMPLATES,
                  limitations='Six English color goals; template holdout only. No negation, spatial reasoning, or VLM recovery claims.',
                  predictions=[dict(text=t, expected=int(y), predicted=int(z), confidence=float(c))
                               for t,y,z,c in zip(heldout, expected, pred, conf)])
    out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('predictions','train_templates','heldout_templates')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
