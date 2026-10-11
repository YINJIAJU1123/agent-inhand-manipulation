"""Replace one-hot language features with frozen SigLIP/SigLIP2 text features.

The RGB-D frontend is intentionally preserved from ``cache_semantic_features``.
Only the language contract changes, so this is a controlled first experiment:
the current visual student receives the same RGB-D features and proprioception
with a frozen pretrained text embedding instead of a face one-hot token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor


def encode_texts(model, processor, texts: list[str], device: torch.device, batch_size: int) -> torch.Tensor:
    if not texts:
        raise ValueError("feature cache does not contain raw instructions")
    unique = list(dict.fromkeys(texts))
    encoded: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            if hasattr(model, "get_text_features"):
                features = model.get_text_features(**inputs)
            else:
                output = model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                features = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
                if hasattr(model, "text_projection"):
                    features = model.text_projection(features)
            encoded.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
    lookup = torch.cat(encoded)
    index = {text: i for i, text in enumerate(unique)}
    return lookup[torch.tensor([index[text] for text in texts], dtype=torch.long)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="RGB-D semantic feature cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--canonical-from-target-face", action="store_true",
                        help="Use canonical prompts reconstructed from target_face when raw instructions are absent.")
    args = parser.parse_args()

    source = Path(args.data)
    data = torch.load(source, map_location="cpu", weights_only=False)
    instructions = [str(text) for text in data.get("instructions", [])]
    instruction_source = "raw_instruction"
    if not instructions and args.canonical_from_target_face:
        face_names = data.get("face_names", ["red", "green", "blue", "yellow", "magenta", "cyan"])
        faces = torch.as_tensor(data["target_face"]).reshape(-1).tolist()
        instructions = [f"show the {face_names[int(face)]} marker" for face in faces]
        instruction_source = "canonical_from_target_face"
    if not instructions:
        raise ValueError(
            "source cache has no raw instructions; regenerate it with cache_semantic_features.py "
            "after collecting a shard that stores instructions, or pass --canonical-from-target-face"
        )
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    language_features = encode_texts(model, processor, instructions, device, args.batch_size)

    result = dict(data)
    result["language_features"] = language_features
    result["language_mode"] = "vlm"
    result["language_dim"] = int(language_features.shape[-1])
    result["language_model"] = args.model
    result["language_model_sha256"] = hashlib.sha256(
        json.dumps({"model": args.model}, sort_keys=True).encode()
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {
        "output": str(output),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "samples": int(result["actions"].shape[0]),
        "image_dim": int(result["image_features"].shape[-1]),
        "language_dim": int(result["language_features"].shape[-1]),
        "language_mode": "vlm",
        "language_model": args.model,
        "instruction_source": instruction_source,
        "unique_instructions": len(set(instructions)),
        "freeze_id": result.get("freeze_id"),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
