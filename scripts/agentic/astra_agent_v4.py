"""Astra tool-using supervisor. Public inputs: RGB, instruction, proprioception.

No simulator labels, file names, layout, pose, or scored success enter requests.
API availability and real model performance require a separately configured key.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageOps

from semantic_agent_v3 import ObservationFeedback, SYMBOLS
from visual_agent import COLORS, Decision, encode_png
from agent_observation_v4 import rgb_crop_box


SYSTEM = '''You supervise a dexterous hand reorienting a cube to satisfy the user's language goal.
The SMALL colored stickers carry black geometric symbols with white outlines. Large cube-face backgrounds and printed letters are irrelevant. Sticker colors are red, green, blue, yellow, magenta, cyan. Symbols are star, circle, triangle, cross (plus), diamond (rhombus), bars (two parallel bars). Assignments vary; never infer a hidden symbol from a fixed layout.
The available low-level skill brings the selected sticker COLOR upward. You choose bounded observation or execution actions and receive actual images afterward. Track evidence across views and correct mistakes. The second image is an RGB-derived enlarged crop of the first.
Use inspect_face to obtain a better view, reorient_target when visual evidence identifies the requested target, verify_target to inspect the result while maintaining the selected skill, stop only when visual evidence supports completion, or abstain for unsupported/ambiguous instructions. Only upward goals are supported. Stillness alone is not completion. You have no access to true object orientation or a success sensor.
One tool call per response. Never output joint commands. A statement of completion is evaluated independently and may be wrong.'''


def tools_schema():
    color = {'type': 'string', 'enum': list(COLORS)}
    symbol = {'type': 'string', 'enum': list(SYMBOLS)}
    duration = {'type': 'integer', 'minimum': 10, 'maximum': 90}
    reason = {'type': 'string'}
    definitions = [
        ('inspect_face', 'Bring a sticker upward to obtain visual evidence.', {'color': color, 'steps': duration, 'evidence': reason}),
        ('reorient_target', 'Execute the visually grounded language goal.', {'goal_symbol': symbol, 'color': color, 'steps': duration, 'evidence': reason}),
        ('verify_target', 'Maintain a grounded target for 20 steps and request a fresh view.', {'goal_symbol': symbol, 'color': color, 'evidence': reason}),
        ('stop', 'Declare completion from visual evidence; this can be scored as false completion.', {'goal_symbol': symbol, 'color': color, 'evidence': reason}),
        ('abstain', 'Stop without claiming completion.', {'evidence': reason}),
    ]
    return [{'type': 'function', 'name': name, 'description': desc, 'strict': True,
             'parameters': {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}}
            for name, desc, props in definitions]


def validate_action(name, args):
    definition = next((x for x in tools_schema() if x['name'] == name), None)
    if definition is None or not isinstance(args, dict):
        raise ValueError('Unknown or malformed robot tool call')
    props = definition['parameters']['properties']
    if set(args) != set(props):
        raise ValueError('Unexpected robot tool arguments')
    for key, spec in props.items():
        value = args[key]
        if spec['type'] == 'integer':
            if type(value) is not int or not spec['minimum'] <= value <= spec['maximum']:
                raise ValueError('Invalid action duration')
        elif not isinstance(value, str) or ('enum' in spec and value not in spec['enum']):
            raise ValueError('Invalid robot tool value')
    return args


class AstraClient:
    def __init__(self, model='gpt-6-astra', key_env='OPENAI_API_KEY', base_url=None):
        self.model = model
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f'{key_env} is not configured; no Astra evaluation was performed')
        self.base_url = (base_url or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')
        if not self.base_url.startswith('https://'):
            raise ValueError('API credentials require an HTTPS endpoint')

    def ask(self, instruction, rgb, feedback, previous_response_id=None, previous_call_id=None):
        frame = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        box = rgb_crop_box(rgb)
        crop = ImageOps.pad(frame.crop(box), (336, 336), color=(32, 32, 32))
        messages = []
        if previous_call_id:
            messages.append({'type': 'function_call_output', 'call_id': previous_call_id,
                             'output': json.dumps({'measured_feedback': feedback, 'new_observation_follows': True})})
        prompt = json.dumps({'instruction': instruction, 'feedback': feedback}, ensure_ascii=False)
        content = [{'type': 'input_text', 'text': prompt}]
        content += [{'type': 'input_image', 'image_url': 'data:image/png;base64,' + encode_png(im), 'detail': 'high'} for im in (frame, crop)]
        messages.append({'role': 'user', 'content': content})
        payload = {'model': self.model, 'instructions': SYSTEM, 'input': messages,
                   'tools': tools_schema(), 'tool_choice': 'required', 'parallel_tool_calls': False,
                   'reasoning': {'effort': 'low'}, 'max_output_tokens': 1024, 'store': True}
        if previous_response_id:
            payload['previous_response_id'] = previous_response_id
        request = urllib.request.Request(self.base_url + '/responses', data=json.dumps(payload).encode(),
                                        headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            # Neither credentials nor the full request is included in error text.
            raise RuntimeError(f'Astra API HTTP {error.code}; check account/model access') from None
        calls = [x for x in result.get('output', []) if x.get('type') == 'function_call']
        if len(calls) != 1 or result.get('status') != 'completed':
            raise RuntimeError('Astra did not return exactly one completed tool call')
        call = calls[0]
        arguments = validate_action(call['name'], json.loads(call['arguments']))
        return {'tool': call['name'], 'arguments': arguments, 'call_id': call['call_id'],
                'response_id': result['id'], 'model': result.get('model'), 'usage': result.get('usage'),
                'elapsed_s': time.monotonic() - start, 'instruction': instruction, 'prompt': prompt,
                'system_prompt': SYSTEM, 'image_sha256': hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
                'crop_box': box, 'raw_tool_call': call}


class AstraAgent:
    def __init__(self, client, instruction, budget=6):
        self.client, self.instruction, self.budget = client, instruction, budget
        self.calls = 0
        self.trace, self.feedback_trace, self.transitions = [], [], []
        self.goal = self.goal_symbol = self.goal_direction = None
        self.goal_embedding = None
        self.monitor = ObservationFeedback()
        self.next_step = 0
        self.response_id = self.call_id = None
        self.last_decision = Decision(None, 'initial', '')
        self.declared_success = False
        self.recovery_used = False

    def tick(self, rgb, proprio, step):
        measured = self.monitor.update(rgb, proprio, step)
        feedback = dict(step=int(step), previous_skill=self.last_decision.color,
                        previous_state=self.last_decision.state, **measured)
        self.feedback_trace.append(feedback)
        if step < self.next_step or self.declared_success:
            return self.last_decision
        if self.calls >= self.budget:
            return self.last_decision
        result = self.client.ask(self.instruction, rgb, feedback, self.response_id, self.call_id)
        self.calls += 1
        self.response_id, self.call_id = result['response_id'], result['call_id']
        name, args = result['tool'], result['arguments']
        old = self.last_decision
        if name == 'abstain':
            decision = Decision(None, 'abstain', args['evidence'])
        else:
            color = COLORS.index(args['color'])
            if name != 'inspect_face':
                self.goal, self.goal_symbol, self.goal_direction = color, args['goal_symbol'], 'up'
            state = {'inspect_face': 'inspect', 'reorient_target': 'execute', 'verify_target': 'verify', 'stop': 'stop'}[name]
            decision = Decision(color, state, args['evidence'])
        self.declared_success = name == 'stop'
        self.next_step = step + args.get('steps', 20)
        self.last_decision = decision
        self.trace.append(dict(step=int(step), operation='astra_tool', accepted_goal=self.goal, **result))
        self.transitions.append({'step': step, 'from': vars(old), 'to': vars(decision)})
        return decision
