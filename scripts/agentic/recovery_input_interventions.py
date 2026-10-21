"""Direct-input interventions for an already trained recovery policy.

The frozen student still receives correct live RGB-D and language. Its proposal,
proprioception and action history can carry indirect modality information. These
are test-time sensitivity interventions, not retrained no-modality baselines.
"""
import torch

VARIANTS=('full','frozen_visual','shifted_goal','frozen_visual_shifted_goal')


def controller_inputs(obs,goal,request_visual,shifted_goal,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    if obs.ndim!=2 or request_visual.ndim!=2 or request_visual.shape[0]!=len(obs):
        raise ValueError('Expected batched flat observations and request visual history')
    visual_dim=request_visual.shape[1]
    if not 0<visual_dim<obs.shape[1] or shifted_goal.shape!=goal.shape or len(goal)!=len(obs):
        raise ValueError('Incompatible intervention shapes')
    x=obs
    if variant in ('frozen_visual','frozen_visual_shifted_goal'):
        x=obs.clone();x[:,:visual_dim]=request_visual
    g=shifted_goal if variant in ('shifted_goal','frozen_visual_shifted_goal') else goal
    return x,g


@torch.no_grad()
def probe_actions(model,obs,goal,request_visual,shifted_goal,proposal,current,fraction,pending):
    """All alternatives share the same observed state; no simulator branches."""
    from recovery_controller_v3 import residual_command,fixed_command
    actions={};details={}
    for variant in VARIANTS:
        x,g=controller_inputs(obs,goal,request_visual,shifted_goal,variant)
        raw=model.distribution(x,g).mean
        action,rate,residual=residual_command(proposal,current,raw,fraction,pending)
        actions[variant]=action
        details[variant]=dict(rate=rate,residual_l2=residual.norm(dim=-1),raw=raw)
    actions['fixed_slow']=fixed_command('slow',proposal,current,current,pending,fraction)
    for name,action in actions.items():
        details.setdefault(name,{})
        details[name].update(action=action,action_l2_from_full=(action-actions['full']).norm(dim=-1),
                             action_max_abs_from_full=(action-actions['full']).abs().amax(dim=-1))
    return {name:{k:v.cpu().tolist() for k,v in values.items()} for name,values in details.items()}
