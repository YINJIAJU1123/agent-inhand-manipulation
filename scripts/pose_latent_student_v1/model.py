"""Direct latent-conditioned adaptation of the released ViserDex belief student.

Import with the pinned colleague repository's source/viserdex on PYTHONPATH.
Only current measured pose enters this policy. Goals enter as continuous vectors.
"""

from dataclasses import replace

import torch
from torch import nn

from viserdex.distillation.config import StudentCfg
from viserdex.distillation.model import BeliefStudent


PROP_KEEP = list(range(105)) + list(range(109, 113))


class LatentStudent(nn.Module):
    def __init__(self, original_cfg, latent_dim=768, goal_width=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.goal_width = goal_width
        self.original_cfg = original_cfg
        self.goal_encoder = nn.Linear(latent_dim, goal_width)
        self.core = BeliefStudent(replace(original_cfg, proprio_dim=109 + goal_width, extero_dim=7))

    def initial_state(self, batch_size, device):
        return self.core.initial_state(batch_size, device)

    def features(self, proprio, latent):
        if proprio.shape[-1] != 109 or latent.shape[-1] != self.latent_dim:
            raise ValueError('Expected target-free proprio109 and continuous latent768')
        return torch.cat((proprio, self.goal_encoder(latent).tanh()), -1)

    def forward(self, proprio, pose, latent, hidden):
        if pose.shape[-1] != 7:
            raise ValueError('Expected current position3 and XYZW orientation4 only')
        return self.core(self.features(proprio, latent), pose, hidden)

    def sequence(self, proprio, pose, latent, hidden, reset_before):
        return self.core.sequence(self.features(proprio, latent), pose, hidden, reset_before)


@torch.no_grad()
def migrate(payload, seed=182801):
    """Copy compatible coefficients, remapping columns rather than dropping layers."""
    cfg = StudentCfg(**payload['student_cfg'])
    if (cfg.proprio_dim, cfg.extero_dim, cfg.num_actions) != (113, 11, 21):
        raise ValueError('Migration requires released Revo3 observation/action contract')
    torch.manual_seed(seed)
    model = LatentStudent(cfg)
    old = payload['student']
    new = model.core.state_dict()
    copied = 0
    parameter_names = dict(model.core.named_parameters())
    special = {
        'extero_encoder.0.weight', 'rnn.weight_ih_l0', 'actor.0.weight',
        'extero_decoder.0.weight', 'extero_decoder.4.weight', 'extero_decoder.4.bias',
    }
    for name, value in new.items():
        if name.startswith(('prop_norm.', 'extero_norm.', 'target_norm.')) or name in special:
            continue
        value.copy_(old[name])
        if name in parameter_names:
            copied += value.numel()
    new['extero_encoder.0.weight'].copy_(old['extero_encoder.0.weight'][:, :7])
    copied += new['extero_encoder.0.weight'].numel()
    for name, tail in [('rnn.weight_ih_l0', cfg.extero_latent),
                       ('actor.0.weight', cfg.extero_latent + cfg.belief_dim)]:
        new[name].zero_()
        new[name][:, :109].copy_(old[name][:, PROP_KEEP])
        new[name][:, 109 + model.goal_width:].copy_(old[name][:, 113:113 + tail])
        copied += new[name].shape[0] * (109 + tail)
    keep_decoder = list(range(cfg.belief_dim + 7))
    new['extero_decoder.0.weight'].copy_(old['extero_decoder.0.weight'][:, keep_decoder])
    new['extero_decoder.4.weight'].copy_(old['extero_decoder.4.weight'][:7])
    new['extero_decoder.4.bias'].copy_(old['extero_decoder.4.bias'][:7])
    copied += sum(new[k].numel() for k in special if k.startswith('extero_decoder.'))
    # Normalize physical channels with the original statistics. Latent channels
    # have fixed zero/one statistics; learned features are bounded by tanh.
    for name in ('mean', 'var'):
        new['prop_norm.' + name][:109].copy_(old['prop_norm.' + name][PROP_KEEP])
        new['extero_norm.' + name].copy_(old['extero_norm.' + name][:7])
        keep_target = list(range(7)) + list(range(11, 11 + cfg.privileged_dim))
        new['target_norm.' + name].copy_(old['target_norm.' + name][keep_target])
    for norm in ('prop_norm', 'extero_norm', 'target_norm'):
        new[norm + '.count'].copy_(old[norm + '.count'])
    model.core.load_state_dict(new, strict=True)
    original_count = sum(v.numel() for k, v in old.items() if k in parameter_names)
    return model, {
        'original_parameters': original_count,
        'copied_parameters': copied,
        'new_parameters': sum(p.numel() for p in model.parameters()),
        'student_inputs': ['proprio109', 'current_predicted_pose7', 'continuous_goal_latent768'],
        'goal_quaternion_input': False,
        'relative_goal_quaternion_input': False,
        'copy_ratio_is_not_performance_retention': True,
    }


def configure_training(model, stage):
    """Warmup trains new coefficients; adaptation unfreezes the inherited core."""
    hooks = []
    if stage not in ('interface', 'adapt'):
        raise ValueError('Unknown migration stage')
    for parameter in model.parameters():
        parameter.requires_grad_(stage == 'adapt')
    if stage == 'interface':
        for parameter in model.goal_encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in (model.core.rnn.weight_ih_l0, model.core.actor[0].weight):
            parameter.requires_grad_(True)
            mask = torch.zeros_like(parameter)
            mask[:, 109:109 + model.goal_width] = 1
            hooks.append(parameter.register_hook(lambda grad, mask=mask: grad * mask))
    return hooks
