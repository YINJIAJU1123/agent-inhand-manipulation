"""Text-only goal grounding. No simulator state or color parser enters inference."""
from __future__ import annotations
import torch

FACE_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan"]
TRAIN_TEMPLATES = [
    "show the {color} marker", "reorient to the {color} marker",
    "turn toward the {color} marker", "bring the {color} marker to the top",
    "move the {color} marker into view", "place the {color} marker upward",
    "rotate the cube to show its {color} face", "make the {color} side face up",
    "I want to see the {color} surface", "please expose the {color} patch",
    "the goal is the {color} face", "orient the {color} side upward",
    "show me {color}", "target the {color} surface",
    "please turn the block until {color} is visible", "bring {color} into view",
    "the requested color is {color}", "position the {color} face on top",
]
# Fixed before training; never used for optimization or checkpoint selection.
HELDOUT_TEMPLATES = [
    "Could you present the {color} side of the cube?",
    "Adjust your grip so that the {color} patch ends up facing upwards.",
    "Let me have a look at the {color} face.",
    "Your next task: expose the surface colored {color}.",
    "Turn the object in your hand to reveal {color}.",
    "Keep rotating until you can show me the {color} marking.",
    "I'd like the upward facing side to be {color}.",
    "Please bring the side painted {color} to the top.",
    "Can you display the cube's {color} panel?",
    "Reposition this block with its {color} region uppermost.",
    "The surface I am asking for is {color}.",
    "Make {color} the visible face by turning the cube.",
]

def encode_texts(model, processor, texts, device, max_length=64, batch_size=32):
    """Fixed padding makes features independent of other instructions in a batch."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            inputs = processor(text=texts[start:start + batch_size], return_tensors="pt",
                               padding="max_length", max_length=max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            features = model.get_text_features(**inputs)
            outputs.append(torch.nn.functional.normalize(features.float(), dim=-1))
    return torch.cat(outputs)

def ground_features(features, adapter):
    """Return controller embeddings and predictions, using ONLY text features."""
    if adapter.get("version") != "siglip2_text_classifier_v2":
        raise ValueError("expected text-only classifier v2")
    weights = adapter["weight"].to(features)
    bias = adapter["bias"].to(features)
    logits = torch.nn.functional.linear(features, weights, bias)
    predicted = logits.argmax(-1)
    canonical = adapter["canonical_embeddings"].to(features)
    return canonical[predicted], predicted, logits.softmax(-1).amax(-1)
