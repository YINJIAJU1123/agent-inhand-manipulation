"""Original symbolic prompts/router with the same RGB crop as the latent agent."""
import hashlib
import json
import time
import urllib.request
import numpy as np
from PIL import Image, ImageOps
from semantic_agent_v3 import SemanticVLMClient
from agent_observation_v4 import rgb_crop_box
from visual_agent import COLORS, encode_png


class SharedCropSemanticClient(SemanticVLMClient):
    def ask(self, rgb, prompt, max_tokens=256):
        frame = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        box = rgb_crop_box(rgb)
        crop = ImageOps.pad(frame.crop(box), (336, 336), color=(32, 32, 32))
        payload = {'images': [encode_png(frame), encode_png(crop)], 'prompt': prompt, 'max_tokens': max_tokens}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.load(response)['text']
        parsed = None
        decoder = json.JSONDecoder()
        for i, character in enumerate(raw):
            if character in '{[':
                try:
                    parsed = decoder.raw_decode(raw[i:])[0]
                    break
                except json.JSONDecodeError:
                    pass
        if isinstance(parsed, list):
            parsed = {'stickers': parsed}
        if not isinstance(parsed, dict):
            parsed = {}
        color = parsed.get('color', 'unknown')
        return {'color': color if color in COLORS else 'unknown', 'visible': parsed.get('visible') is True,
                'raw': raw, 'parsed': parsed, 'elapsed_s': time.monotonic()-start, 'crop_box': box,
                'image_sha256': hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()}
