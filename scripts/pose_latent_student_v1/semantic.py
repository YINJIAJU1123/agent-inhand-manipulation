"""RSLCube semantic input contract shared by the two high-level model tracks.

Astra supplies structured tool calls, mapped locally to the frozen student bank.
The fine-tuned VLM supplies a learned continuous vector in the same bank's span.
Neither interface exposes a target quaternion to the deployed student.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

from goals import COLORS, SYMBOLS, load_bank

SYSTEM = '''You supervise a Revo3 dexterous hand reorienting a lettered cube.
The task-relevant marks are the printed capital letters T, E, R, L, S, H on
colored faces. Read the letters from the current RGB image and its enlarged
crop. The available color skills are red, green, blue, yellow, magenta, cyan.
Here magenta denotes the purple face and cyan denotes the pale-blue face.
Do not confuse the pale-blue face with the darker blue face. Do not infer that
an occluded letter has been visually observed. Preserve uncertainty and use
observations accumulated within this episode.
Each low-level color skill brings that face upward. Use inspect_face for
additional evidence, reorient_target for the grounded target, verify_target
for a fresh observation while maintaining the target, stop to declare visual
completion, or abstain without claiming success. Only upward goals are
supported. Stillness alone does not establish completion. You have no true
object orientation or success sensor. One tool call per response, no joint
commands. Completion is scored independently and your declaration can be wrong.'''

VLM_PROMPT = '''Ground the requested lettered cube face using the image and crop.
Task-relevant printed letters are T, E, R, L, S, H. The face-color skill names
are red, green, blue, yellow, magenta (purple), and cyan (pale blue).
Only upward goals are supported. Determine which color contains the requested
letter and whether it is actually readable in this observation. Do not claim
that an occluded letter is visible. Preserve uncertainty.
Instruction: {instruction}
Summarize the target and the available visual evidence.'''


def tools_schema():
    color = {'type': 'string', 'enum': list(COLORS)}
    symbol = {'type': 'string', 'enum': list(SYMBOLS)}
    reason = {'type': 'string'}
    duration = {'type': 'integer', 'minimum': 10, 'maximum': 90}
    definitions = [
        ('inspect_face', 'Obtain another view of the cube.', {'color': color, 'steps': duration, 'evidence': reason}),
        ('reorient_target', 'Execute the visually grounded upward goal.',
         {'goal_symbol': symbol, 'color': color, 'steps': duration, 'evidence': reason}),
        ('verify_target', 'Maintain the goal and obtain fresh evidence.',
         {'goal_symbol': symbol, 'color': color, 'evidence': reason}),
        ('stop', 'Declare completion from visual evidence.',
         {'goal_symbol': symbol, 'color': color, 'evidence': reason}),
        ('abstain', 'Stop without declaring success.', {'evidence': reason}),
    ]
    return [dict(type='function', name=name, description=description, strict=True,
                 parameters=dict(type='object', properties=properties,
                                 required=list(properties), additionalProperties=False))
            for name, description, properties in definitions]


def validate_action(name, arguments):
    definition = next((tool for tool in tools_schema() if tool['name'] == name), None)
    if definition is None or not isinstance(arguments, dict):
        raise ValueError('Unknown or malformed RSLCube tool call')
    properties = definition['parameters']['properties']
    if set(arguments) != set(properties):
        raise ValueError('Unexpected RSLCube tool arguments')
    for key, spec in properties.items():
        value = arguments[key]
        if spec['type'] == 'integer':
            if type(value) is not int or not spec['minimum'] <= value <= spec['maximum']:
                raise ValueError('Invalid control duration')
        elif not isinstance(value, str) or ('enum' in spec and value not in spec['enum']):
            raise ValueError('Invalid RSLCube tool value')
    # A wrong letter/color association remains a model error, not an oracle fix.
    return arguments


def visual_inputs(processor, rgb, instruction, device):
    """Image-only crop; no pose, visibility label, goal index or layout input."""
    agent_dir = str(Path(__file__).resolve().parents[1] / 'agentic')
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    from agent_observation_v4 import rgb_crop_box

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected RGB pixels')
    frame = Image.fromarray(rgb)
    box = rgb_crop_box(rgb)
    crop = ImageOps.pad(frame.crop(box), (336, 336), color=(32, 32, 32))
    content = [{'type': 'image', 'image': frame}, {'type': 'image', 'image': crop},
               {'type': 'text', 'text': VLM_PROMPT.format(instruction=instruction)}]
    inputs = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt')
    return inputs.to(device), box


class StudentGoalBank:
    def __init__(self, cache, original_asset, device='cpu'):
        self.vectors, _, self.manifest = load_bank(cache, original_asset, device)

    def from_astra_tool(self, name, arguments):
        arguments = validate_action(name, arguments)
        if name == 'abstain':
            return None
        return self.vectors[COLORS.index(arguments['color'])].clone()

    def check_vlm_head(self, head):
        torch.testing.assert_close(head.canonical.detach().cpu(), self.vectors.cpu(), atol=0, rtol=0)


class RSLCubeRulePredictor:
    """Frozen base VLM text generation followed by a deterministic skill lookup."""
    def __init__(self,url):
        import requests
        if not url.startswith('http://127.0.0.1:'):raise ValueError('Loopback required')
        self.url=url
        response=requests.get(url+'/health',timeout=10)
        response.raise_for_status()
        self.metadata=response.json()

    def predict(self,rgb,instruction):
        import hashlib
        import json
        import requests
        from agent_observation_v4 import rgb_crop_box
        from visual_agent import encode_png
        rgb=np.asarray(rgb,dtype=np.uint8)
        frame=Image.fromarray(rgb)
        box=rgb_crop_box(rgb)
        crop=ImageOps.pad(frame.crop(box),(336,336),color=(32,32,32))
        prompt=VLM_PROMPT.format(instruction=instruction)+"""
Return only one JSON object with these fields:
{"color":"red|green|blue|yellow|magenta|cyan|unknown","visible":true|false,"evidence":"brief observed detail"}
The color must belong to the face containing the requested printed letter.
Use unknown and visible=false when that letter is not readable. No fixed
letter-to-color mapping is supplied. A partially seen different letter is
not sufficient evidence for the requested one."""
        response=requests.post(self.url,json=dict(images=[encode_png(frame),encode_png(crop)],
            prompt=prompt,max_tokens=128),timeout=(10,180),allow_redirects=False)
        response.raise_for_status()
        raw=response.json()['text']
        decoder=json.JSONDecoder()
        parsed=None
        for index,char in enumerate(raw):
            if char=='{':
                try:
                    parsed=decoder.raw_decode(raw[index:])[0]
                    break
                except json.JSONDecodeError:
                    continue
        valid=isinstance(parsed,dict) and parsed.get('color') in COLORS and type(parsed.get('visible')) is bool
        accepted=valid and parsed['visible']
        return dict(accepted=bool(accepted),skill=COLORS.index(parsed['color']) if accepted else None,
            raw=raw,parsed=parsed,format_valid=bool(valid),crop_box=box,
            image_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
