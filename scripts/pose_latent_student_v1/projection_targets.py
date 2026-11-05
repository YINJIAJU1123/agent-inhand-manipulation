"""Training-only teacher projection into the bounded execution action family."""
import torch


@torch.no_grad()
def teacher_speed_projection(current, proposal, target, regularizer=.01,
                             residual_limit=.35, min_speed=.05, iterations=32):
    """Minimize mean((action-target)^2 + regularizer*residual^2) per state.

    The scalar speed is solved by convex bisection after eliminating the bounded
    residual analytically. This uses teacher labels, never deployed observations.
    Positive residual regularization resolves flat action-error decompositions.
    States with no proposal motion have no identifiable speed and are masked.
    """
    if current.ndim != 2 or current.shape != proposal.shape or current.shape != target.shape:
        raise ValueError('Current, proposal and target must have matching (N, action_dim) shapes')
    if not regularizer > 0 or not residual_limit > 0 or not 0 < min_speed <= 1 or iterations < 1:
        raise ValueError('Invalid projection bounds or regularizer')
    if not all(torch.isfinite(x).all() for x in (current, proposal, target)):
        raise ValueError('Nonfinite projection inputs')
    # Near an optimum, subtracting target and projected action can cancel in
    # float32. Solve labels in float64, then restore the model input dtype.
    dtype = current.dtype
    c = current.double().clamp(-1, 1)
    d = proposal.double().clamp(-1, 1)-c
    t = target.double().clamp(-1, 1)
    low = c.new_full((len(c),), min_speed)
    high = torch.ones_like(low)
    for _ in range(iterations):
        speed = (low+high)*.5
        base = c+speed[:, None]*d
        residual = ((t-base)/(1+regularizer)).clamp(-residual_limit, residual_limit)
        derivative = ((base+residual-t)*d).sum(-1)
        low = torch.where(derivative < 0, speed, low)
        high = torch.where(derivative >= 0, speed, high)
    identifiable = d.square().mean(-1) > 1e-8
    speed = torch.where(identifiable, (low+high)*.5, torch.ones_like(low))
    base = c+speed[:, None]*d
    residual = ((t-base)/(1+regularizer)).clamp(-residual_limit, residual_limit)
    action = (base+residual).clamp(-1, 1)
    return dict(speed=speed.to(dtype), residual=residual.to(dtype), action=action.to(dtype),
                identifiable=identifiable,
                objective=((action-t).square()+regularizer*residual.square()).mean(-1).to(dtype))
