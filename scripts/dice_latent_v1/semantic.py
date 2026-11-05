"""Dice RGB/language contract. Simulator pose and true target never enter the VLM."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps
import torch

VLM_PROMPT = 'Ground the requested dotted face of the wooden die from the current image and enlarged crop.\nThere are six faces, with one through six black dots. Count the dots on each visible face.\nDetermine the requested face number and whether that face is actually readable in this view.\nA partially seen different face does not establish visibility of the requested face.\nDo not infer that an occluded face has been observed. Only upward goals are supported.\nInstruction: {instruction}\nSummarize the requested face and the visual evidence.'

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


SYSTEM = '''You supervise a Revo3 dexterous hand reorienting a wooden die.
Faces have one through six black dots. Use the current RGB image and enlarged
crop to count visible dots. Only upward goals are supported. A numbered skill
brings that dotted face upward. Do not claim to have observed an occluded face.
Use inspect_face to obtain another view, reorient_target for a visually grounded
goal, verify_target to maintain it and request new evidence, stop to declare
visual completion, or abstain without claiming success. Preserve uncertainty.
Each control step is 0.05 seconds. Stillness alone does not prove completion.
You may receive noisy local orientation-error estimates; they can be wrong and
are not a success sensor. No true object pose or task success is provided.
One tool call per response, no joint commands. Your stop is independently scored.'''

def tools_schema():
    number={'type':'integer','minimum':1,'maximum':6}
    reason={'type':'string'};duration={'type':'integer','minimum':10,'maximum':90}
    entries=[('inspect_face','Obtain a new view.',{'number':number,'steps':duration,'evidence':reason}),
      ('reorient_target','Execute the visually grounded upward goal.',{'goal_number':number,'number':number,'steps':duration,'evidence':reason}),
      ('verify_target','Maintain the target and gather fresh evidence.',{'goal_number':number,'number':number,'evidence':reason}),
      ('stop','Declare visual completion.',{'goal_number':number,'number':number,'evidence':reason}),
      ('abstain','Stop without success.',{'evidence':reason})]
    return [dict(type='function',name=name,description=description,strict=True,
      parameters=dict(type='object',properties=properties,required=list(properties),additionalProperties=False))
      for name,description,properties in entries]

def validate_action(name,arguments):
    spec=next((x for x in tools_schema() if x['name']==name),None)
    if spec is None or not isinstance(arguments,dict):raise ValueError('Malformed dice tool')
    props=spec['parameters']['properties']
    if set(arguments)!=set(props):raise ValueError('Unexpected dice tool arguments')
    for key,rule in props.items():
        value=arguments[key]
        if rule['type']=='integer':
            if type(value) is not int or not rule['minimum']<=value<=rule['maximum']:raise ValueError('Invalid dice integer')
        elif not isinstance(value,str):raise ValueError('Invalid evidence')
    return arguments

RULE_SUFFIX = '''
Return only one JSON object:
{"number":1,"visible":true,"evidence":"brief visible detail"}
number must be an integer from 1 to 6 identifying the requested dotted face,
or null when evidence is insufficient. visible must be a JSON boolean.
Use null and visible=false if the requested face is not readable. Never infer
visibility from the requested number alone.'''
