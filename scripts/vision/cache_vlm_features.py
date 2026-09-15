"""Cache frozen image and language features for visual Revo3 rollouts.

The high-frequency controller never calls a generative VLM.  This utility
uses a frozen Hugging Face vision-language encoder once, offline, and stores
one image embedding and one instruction embedding per recorded transition.
The recurrent/action head can then be trained repeatedly without loading the
large encoder or changing the 21-D Revo3 action interface.

The default checkpoint is SigLIP2.  Any compatible CLIP/SigLIP checkpoint can
be supplied with ``--model``; the script handles both ``get_image_features``
and pooled ``last_hidden_state`` model APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def _image_features(model, processor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    if hasattr(model, "get_image_features"):
        output = model.get_image_features(**inputs)
    else:
        output = model.vision_model(pixel_values=inputs["pixel_values"])
        output = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
        if hasattr(model, "visual_projection"):
            output = model.visual_projection(output)
    return torch.nn.functional.normalize(output.float(), dim=-1).cpu()


def _text_features(model, processor, texts: list[str], device: torch.device) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    if hasattr(model, "get_text_features"):
        output = model.get_text_features(**inputs)
    else:
        output = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        output = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
        if hasattr(model, "text_projection"):
            output = model.text_projection(output)
    return torch.nn.functional.normalize(output.float(), dim=-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="visual_rollouts.pt")
    parser.add_argument("--output", required=True, help="cached feature shard")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    image_features, text_features = [], []
    # Each collector entry contains [num_envs,H,W,C], while labels and
    # instructions contain the same leading environment dimension.
    with torch.inference_mode():
        for frame_batch, text_batch in zip(data["frames"], data["instructions"]):
            rgb = frame_batch["rgb"].to(torch.uint8)
            images = [Image.fromarray(x.numpy()) for x in rgb]
            for start in range(0, len(images), args.batch_size):
                image_features.append(_image_features(model, processor, images[start:start + args.batch_size], device))
            texts = [str(x) for x in text_batch]
            for start in range(0, len(texts), args.batch_size):
                text_features.append(_text_features(model, processor, texts[start:start + args.batch_size], device))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image_features": torch.cat(image_features),
        "language_features": torch.cat(text_features),
        "student_proprio": torch.cat(data["student_proprio"]),
        "actions": torch.cat(data["actions"]),
        "target_face": torch.cat(data["target_face"]),
        "episode_id": torch.cat(data["episode_id"]),
        "step_index": torch.cat(data["step_index"]),
        "env_id": torch.cat(data.get("env_id", [torch.zeros_like(x) for x in data["episode_id"]])),
        "model": args.model,
        "source": args.data,
    }
    torch.save(result, output)
    report = {"output": str(output), "samples": int(result["actions"].shape[0]),
              "image_dim": int(result["image_features"].shape[-1]),
              "language_dim": int(result["language_features"].shape[-1]), "model": args.model}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
