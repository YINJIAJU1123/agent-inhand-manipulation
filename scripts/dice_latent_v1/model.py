"""Direct-latent migration of the dice SuperMimic actor.

The public policy accepts physical history and continuous goal history only.
Source target/relative-quaternion channels are discarded before this boundary.
"""
from dataclasses import asdict
import torch
from torch import nn
from viserdex.supermimic.networks import Actor, RunningNorm, mlp, run_mlp
from viserdex.supermimic.core import Config

PHYSICAL_COLUMNS = tuple(range(105)) + tuple(range(109, 120))
TARGET_COLUMNS = (105, 106, 107, 108, 120, 121, 122, 123)
FORMAT = 'dice_supermimic_direct_latent_v2'


def physical_history(source_history):
    if source_history.ndim != 3 or source_history.shape[-1] != 124:
        raise ValueError('Expected original [batch, history, 124] observation')
    return source_history[..., list(PHYSICAL_COLUMNS)]


class DiceLatentActor(nn.Module):
    def __init__(self, config, latent_dim=768, goal_width=64):
        super().__init__()
        self.config = config
        self.latent_dim, self.goal_width = latent_dim, goal_width
        self.amp = config.amp
        self.obs_norm = RunningNorm(len(PHYSICAL_COLUMNS))
        self.register_buffer('goal_center', torch.zeros(latent_dim))
        self.register_buffer('goal_scale', torch.ones(latent_dim))
        self.goal_encoder = nn.Linear(latent_dim, goal_width)
        self.net = mlp((len(PHYSICAL_COLUMNS) + goal_width) * config.history,
                       config.actor_hidden, 21)

    @torch.no_grad()
    def configure_goal_statistics(self, bank):
        if bank.shape != (6, self.latent_dim):
            raise ValueError('Expected frozen six-goal latent bank')
        self.goal_center.copy_(bank.mean(0))
        self.goal_scale.copy_((bank-bank.mean(0)).square().mean(0).sqrt().clamp_min(.005))

    def forward(self, history, goal_history, return_features=False):
        if history.ndim != 3 or history.shape[1:] != (self.config.history, len(PHYSICAL_COLUMNS)):
            raise ValueError('Physical history has an incompatible shape')
        if goal_history.shape != (*history.shape[:2], self.latent_dim):
            raise ValueError('One continuous latent per actual historical observation is required')
        x = torch.cat((self.obs_norm(history), self.goal_encoder(((goal_history-self.goal_center)/self.goal_scale).clamp(-8,8)).tanh()), -1).flatten(1)
        if return_features:
            features = run_mlp(self.net[:-1], x, self.amp)
            logits = run_mlp(self.net[-1:], features, self.amp)
            return dict(action=logits.tanh(), features=features)
        return run_mlp(self.net, x, self.amp).tanh()


@torch.no_grad()
def migrate(payload, seed=186201, goal_width=64):
    if payload.get('version') != 17 or payload['metadata']['task_id'] != 'Isaac-ViserDex-Repose-Revo3-SuperMimic-RoundedDie-WujiOffset-HandPose-Hold-v1':
        raise ValueError('Expected the published dice v17 task')
    cfg = Config(**payload['config'])
    if cfg.history != 10 or cfg.action_mode != 'home_offset':
        raise ValueError('Unexpected dice policy/control contract')
    source = Actor(cfg)
    source.load_state_dict(payload['actor'], strict=True)
    torch.manual_seed(seed)
    model = DiceLatentActor(cfg, goal_width=goal_width)
    old = source.state_dict()
    new = model.state_dict()
    for name in ('mean', 'variance'):
        new['obs_norm.' + name].copy_(old['obs_norm.' + name][list(PHYSICAL_COLUMNS)])
    new['obs_norm.count'].copy_(old['obs_norm.count'])
    copied = 0
    for name, value in source.net.state_dict().items():
        if name == '0.weight':
            target = new['net.0.weight'].reshape(value.shape[0], cfg.history, -1)
            target.zero_()
            columns = value.reshape(value.shape[0], cfg.history, 124)[..., list(PHYSICAL_COLUMNS)]
            target[..., :len(PHYSICAL_COLUMNS)].copy_(columns)
            copied += columns.numel()
        else:
            new['net.' + name].copy_(value)
            copied += value.numel()
    model.load_state_dict(new, strict=True)
    model.requires_grad_(False)
    source.eval().requires_grad_(False)
    return model, source, dict(
        format=FORMAT, original_config=asdict(cfg), copied_parameters=copied,
        original_actor_parameters=sum(p.numel() for p in source.parameters()),
        migrated_actor_parameters=sum(p.numel() for p in model.parameters()),
        inputs=['physical_history_10x116', 'continuous_goal_history_10x768'],
        target_quaternion_input=False, relative_goal_quaternion_input=False,
        goal_history_protocol='append the latent active when each observation was acquired; repeat on reset; retain history on skill changes',
        copy_ratio_is_not_performance_retention=True,
        goal_preprocessing='frozen bank mean/std, std floor .005, clamp +/-8; no target quaternion decoder',
    )


def configure_training(model, stage):
    if stage not in ('interface', 'adapt'):
        raise ValueError('Unknown migration stage')
    model.requires_grad_(stage == 'adapt')
    handles = []
    if stage == 'interface':
        model.goal_encoder.requires_grad_(True)
        first = model.net[0].weight
        first.requires_grad_(True)
        mask = torch.zeros_like(first).reshape(first.shape[0], model.config.history, -1)
        mask[..., len(PHYSICAL_COLUMNS):] = 1
        handles.append(first.register_hook(lambda grad: grad * mask.reshape_as(grad)))
    return handles
