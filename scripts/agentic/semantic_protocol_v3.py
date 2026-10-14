"""Scorer-only scenario generation. Ground-truth symbols/layout never enter agent."""
import numpy as np
from semantic_agent_v3 import SYMBOLS

PARAPHRASES = (
    'Please bring the sticker with five pointed tips onto the upper face of the cube.',
    'Reorient the cube until its round-symbol sticker faces upward.',
    'Make the sticker marked with a three-cornered shape face the ceiling.',
    'Rotate the cube to put the plus-sign sticker on top.',
    'I want the sticker carrying a rhombus to face upward.',
    'Turn the object until the sticker with two parallel bars is on the top face.',
)

def make_cases(seed, instruction_offset=0, template_set='canonical'):
    rng = np.random.default_rng(seed + 30000)
    cases = []
    for i in range(6):
        layout = rng.permutation(6).tolist()
        target = (i + instruction_offset) % 6
        symbol = SYMBOLS[target]
        instruction = (f'Turn the cube so the {symbol} sticker faces upward.' if template_set == 'canonical'
                       else PARAPHRASES[target])
        cases.append({'case': i, 'layout': layout, 'target_symbol': symbol,
                      'target_color_index': layout.index(target), 'instruction': instruction})
    return cases
