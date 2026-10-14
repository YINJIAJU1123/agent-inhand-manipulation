"""Small VLM hidden features -> learned continuous goal tensor.

Only RGB and actual instruction enter the VLM. The controller consumes the
predicted tensor directly; nearest prototypes are diagnostics, not a lookup.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

from agent_observation_v4 import rgb_crop_box

PROMPT = '''Identify the SMALL colored sticker matching the user's instruction from the scene and its enlarged crop. The sticker has a black geometric mark outlined in white. Ignore large face backgrounds and letters. Symbols can be star, circle, triangle, cross (plus sign), diamond (rhombus), or bars (two parallel bars). Do not assume a fixed layout. If visual evidence is insufficient, preserve uncertainty. Only upward goals are supported.
User instruction: {instruction}
Summarize the grounded target and whether current visual evidence supports it.'''


def visual_inputs(processor, rgb, instruction, device):
    frame = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    box = rgb_crop_box(rgb)
    crop = ImageOps.pad(frame.crop(box), (336, 336), color=(32, 32, 32))
    content = [{'type': 'image', 'image': frame}, {'type': 'image', 'image': crop},
               {'type': 'text', 'text': PROMPT.format(instruction=instruction)}]
    inputs = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
                                          add_generation_prompt=True, return_dict=True, return_tensors='pt')
    return inputs.to(device), box


class GoalHead(nn.Module):
    def __init__(self, hidden_size, canonical):
        super().__init__()
        self.register_buffer('canonical', canonical.detach().float())
        self.trunk = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 256), nn.SiLU())
        self.latent = nn.Linear(256, canonical.shape[1])
        self.readable = nn.Linear(256, 1)

    def forward(self, features):
        belief = self.trunk(features.float())
        z = F.normalize(self.latent(belief), dim=-1) * self.canonical.norm(dim=-1).mean()
        logits = 15 * F.normalize(z, dim=-1) @ F.normalize(self.canonical, dim=-1).T
        return {'goal_embedding': z, 'belief': belief, 'readable_logit': self.readable(belief).squeeze(-1),
                'skill_logits': logits}


def load_backbone(model_path, adapter_path=None, training=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    base = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    base.config.use_cache = False
    if training:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant': False})
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=training)
    elif training:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                              target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',
                                              task_type='CAUSAL_LM', bias='none'))
    else:
        model = base
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=128 * 128, max_pixels=384 * 384)
    return model, processor


def hidden_features(model, inputs):
    # Call the multimodal backbone without computing 150k vocabulary logits.
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    output = base.model(**inputs, use_cache=False, return_dict=True)
    return output.last_hidden_state[:, -1]


class LatentPredictor:
    def __init__(self, checkpoint):
        root = Path(checkpoint)
        pack = torch.load(root / 'goal_head.pt', map_location='cpu', weights_only=False)
        if pack['format'] != 'qwen_hidden_goal_v4':
            raise ValueError('Unsupported latent checkpoint')
        self.model, self.processor = load_backbone(pack['model_path'], root / 'lora')
        self.model.eval()
        self.head = GoalHead(pack['hidden_size'], pack['head']['canonical']).cuda().eval()
        self.head.load_state_dict(pack['head'])
        self.threshold = pack.get('readable_threshold', .5)
        self.checkpoint_sha256 = hashlib.sha256((root / 'goal_head.pt').read_bytes()).hexdigest()

    @torch.inference_mode()
    def predict(self, rgb, instruction):
        inputs, box = visual_inputs(self.processor, rgb, instruction, 'cuda:0')
        result = self.head(hidden_features(self.model, inputs))
        probability = float(result['readable_logit'].sigmoid()[0])
        return {'goal_embedding': result['goal_embedding'][0].cpu().numpy(),
                'skill': int(result['skill_logits'][0].argmax()), 'readable_probability': probability,
                'accepted': probability >= self.threshold, 'crop_box': box,
                'image_sha256': hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()}
