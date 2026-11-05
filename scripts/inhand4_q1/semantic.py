"""Visible letters are grounded to fixed colored physical motor regions."""
import importlib.util
from pathlib import Path

_path=Path(__file__).resolve().parents[1]/'dice_latent_v1/semantic.py'
_spec=importlib.util.spec_from_file_location('_shared_image_contract',_path)
_native=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_native)
tools_schema=_native.tools_schema
validate_action=_native.validate_action
LETTERS='ABCDEF'
INSTRUCTIONS=tuple(f'Reorient the object so that the surface marked with letter {x} faces upward.' for x in LETTERS)
CONTRACT='''The object carries six colored surface patches with letters A through F.
The fixed motor skill IDs are: 1 red, 2 green, 3 blue, 4 yellow, 5 cyan, 6 purple.
Each skill brings that colored surface region upward in its canonical pose.
Letters may be on different colors from training. Read the actual letter-color
association from RGB. Never assume A means red or skill 1, or use alphabetical
order as a motor mapping. An occluded letter has not been visually observed.
The color-to-motor dictionary above is fixed and shared by every method.
Only upward goals are supported. No current letter-to-color layout is provided.'''
VLM_PROMPT='''Ground the requested letter on the marked object from the current RGB and enlarged crop.
'''+CONTRACT+'''
Instruction: {instruction}
Summarize the requested letter, the visible color and whether the letter is readable.'''
SYSTEM='''You supervise a Revo3 dexterous hand performing semantic in-hand reorientation.
'''+CONTRACT+'''
Use inspect_face to try a physical skill and obtain another view, reorient_target
for a visually grounded goal, verify_target to maintain it and request new
evidence, stop to declare completion, or abstain without success.
Tool field number is a motor skill ID. Tool field goal_number encodes the
requested letter only: A=1, B=2, C=3, D=4, E=5, F=6. These two fields need not
be equal. current_number_skill and bound_number are motor IDs.
Each step lasts 0.05 seconds. Local orientation-error estimates are noisy,
not a success sensor. Stillness does not prove completion. No true object pose
or task success is supplied. One tool call per response, no joint commands.
Your stop is independently scored.'''
RULE_SUFFIX='''
Return only one JSON object:
{"number":1,"visible":true,"evidence":"visible letter and patch color"}
number is the physical motor skill ID for the requested letter's visible color,
not its alphabetical index. Use null and visible=false if the association is
not readable. visible must be a JSON boolean. Never infer visibility from text.'''

def visual_inputs(processor,rgb,instruction,device):
    _native.VLM_PROMPT=VLM_PROMPT
    return _native.visual_inputs(processor,rgb,instruction,device)
