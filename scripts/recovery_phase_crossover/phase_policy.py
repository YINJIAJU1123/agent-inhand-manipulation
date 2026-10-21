"""Frozen phase selection. No fitting, observation reset or goal substitution."""
import torch
from recovery_controller_v3 import RecoveryController


def load_recovery_actor(path,device,obs_dim,manifest):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck['obs_dim']!=obs_dim:raise ValueError('Observation dimension differs')
    for key in ('student_sha256','adapter_sha256'):
        if ck['manifest'][key]!=manifest[key]:raise ValueError('Frozen dependency differs: '+key)
    devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        model=RecoveryController(obs_dim).to(device)
    model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
    return model


def select_controller(wait,recovery,step,wait_steps):
    if step<0 or wait_steps<=0:raise ValueError('Positive delay and nonnegative step required')
    return wait if step<wait_steps else recovery
