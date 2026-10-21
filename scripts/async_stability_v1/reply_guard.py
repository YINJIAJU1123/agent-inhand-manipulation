"""Optional local handoff ablations; no privileged state or goal labels.

Keep the frozen waiting controller, and change only known-skill execution when
no request is pending. This module does not select semantic targets or stop.
The default is the original controller. Experimental modes require opt-in.
"""
import torch


def guard_action(mode, proposal, current, controlled, pending, known, step, last_delivery,
                 ramp_steps=15, max_delta=.12):
    if mode == 'v3':
        return controlled
    if mode == 'direct':
        candidate = proposal
    elif mode == 'ramp':
        fraction = ((step-last_delivery).float()/ramp_steps).clamp(0,1)
        rate = .15 + .85*fraction
        candidate = current + rate[:,None]*(proposal-current)
    elif mode == 'cap':
        candidate = current + (controlled-current).clamp(-max_delta,max_delta)
    else:
        raise ValueError(f'Unknown handoff mode: {mode}')
    return torch.where((known & ~pending)[:,None],candidate,controlled)
