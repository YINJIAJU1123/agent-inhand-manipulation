"""Bounded same-goal reply intervention, using command history only."""
import torch


def eligible_reply(command, previous_command, windows, active):
    """A reply may preserve a hold only if it actually preserves the goal."""
    if not command['delivered'] or command['phase'] not in ('inspect', 'execute'):
        return torch.zeros_like(active)
    return torch.tensor([
        bool(active[i]) and color is not None and previous_command['colors'][i] == color
        and window.goal == color and window.rate == 0. and window.start is not None
        for i, (color, window) in enumerate(zip(command['colors'], windows))
    ], dtype=torch.bool, device=active.device)


def extend_hold(action, current, *, step, branch_step, duration, selected, active, colors, initial_colors):
    result=action.clone()
    if branch_step is None or not 0 <= step-branch_step < duration:
        return result
    unchanged=torch.tensor([new is not None and new == old for new,old in zip(colors,initial_colors)],
                           dtype=torch.bool,device=active.device)
    mask=selected & active & unchanged
    result[mask]=current[mask]
    return result
