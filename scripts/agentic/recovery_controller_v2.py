"""Goal-conditioned residual control during a wait and its recovery transition.

The auxiliary head predicts success under the collecting policy followed by the
frozen student. It is an on-policy outcome predictor, not an expert/oracle
recoverability label or a calibrated safety certificate.
"""
import math
import torch
from torch import nn
from torch.distributions import Independent, Normal


def observation(visual_history, proprio, proposal, current, anchor, previous_action,
                age, pending, response_age):
    """Only available sensors and request clocks; no remaining delay or scorer."""
    clocks = torch.stack((age / 20., pending.float(), response_age / 2.,
                          (response_age / 2.).clamp(0, 1)), -1)
    x = torch.cat((visual_history.flatten(1), proprio, proposal, current, anchor,
                   previous_action, clocks), -1)
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite actor observation')
    return x.clamp(-10, 10)


def bridge_fraction(response_age, pending, bridge_seconds=2.):
    return torch.where(pending, torch.zeros_like(response_age),
                       (response_age / bridge_seconds).clamp(0, 1))


def residual_command(proposal, current, raw_action, fraction, residual_bound=.04):
    """Zero residual exactly matches slow + a two-second smooth rejoin.

Residual influence vanishes continuously at the end of the bridge. The
environment's normalized action scale remains exactly one.
"""
    base = .15 + .85 * fraction
    base_logit = torch.logit(base.clamp(1e-6, 1-1e-6))
    speed = torch.sigmoid(base_logit + 4 * raw_action[:, 0].tanh() * (1-fraction))
    speed = torch.where(fraction >= 1, torch.ones_like(speed), speed)
    residual = residual_bound * raw_action[:, 1:].tanh() * (1-fraction[:, None])
    action = current + speed[:, None] * (proposal-current) + residual
    return action.clamp(-1, 1), speed, residual


def fixed_command(mode, proposal, current, anchor, pending, fraction):
    if mode == 'continue':
        return proposal
    wait = current + .15 * (proposal-current) if mode.startswith('slow') else anchor
    if mode.endswith('_smooth'):
        start = .15 if mode.startswith('slow') else 0.
        rate = start + (1-start)*fraction
        resume = current + rate[:, None] * (proposal-current)
    else:
        resume = proposal
    return torch.where(pending[:, None], wait, resume).clamp(-1, 1)


class RecoveryController(nn.Module):
    def __init__(self, obs_dim, goal_dim=768, action_dim=21, privileged_dim=8):
        super().__init__()
        self.obs_dim, self.goal_dim = obs_dim, goal_dim
        self.action_dim, self.privileged_dim = action_dim, privileged_dim
        self.sensors = nn.Sequential(nn.Linear(obs_dim, 192), nn.LayerNorm(192), nn.Tanh())
        self.goal = nn.Sequential(nn.Linear(goal_dim, 64), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim+1)
        self.outcome = nn.Linear(128, 1)
        self.critic = nn.Sequential(nn.Linear(128+privileged_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((action_dim+1,), -2.3))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def latent(self, obs, goal):
        goal = torch.nn.functional.normalize(goal, dim=-1)
        return self.fusion(torch.cat((self.sensors(obs), self.goal(goal)), -1))

    def distribution(self, obs, goal):
        z = self.latent(obs, goal)
        return Independent(Normal(self.actor(z), self.log_std.clamp(-4, -.7).exp()), 1)

    def outcome_logit(self, obs, goal):
        return self.outcome(self.latent(obs, goal)).squeeze(-1)

    def value(self, obs, goal, privileged):
        return self.critic(torch.cat((self.latent(obs, goal), privileged), -1)).squeeze(-1)


def advantages(rewards, values, done, valid, gamma=.999, lam=.995):
    out = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else torch.zeros_like(carry)
        next_valid = valid[t+1] if t+1 < len(values) else torch.zeros_like(valid[t])
        live = (~done[t] & next_valid).float()
        delta = rewards[t] + gamma * next_value * live - values[t]
        carry = (delta + gamma * lam * live * carry) * valid[t]
        out[t] = carry
    return out


def update(model, optimizer, rollout, success, *, objective='completion', epochs=3):
    keys = ('obs','goal','priv','action','logprob','value','reward','done','valid')
    x, g, z, act, oldlog, values, rewards, done, valid = [
        torch.stack([r[k] for r in rollout]) for k in keys]
    adv = advantages(rewards, values, done, valid)
    returns = (adv + values)[valid]
    targets = success.float()[None].expand_as(valid)[valid]
    # Equal episode weighting for prediction; long waits must not dominate it.
    episode_weight = (1 / valid.sum(0).clamp_min(1).float())[None].expand_as(values)[valid]
    episode_weight /= episode_weight.mean()
    x,g,z,act,oldlog,adv = [t[valid] for t in (x,g,z,act,oldlog,adv)]
    if len(x) < 2:
        raise RuntimeError('Too few valid controller decisions')
    adv = (adv-adv.mean()) / adv.std().clamp_min(1e-6)
    stats=[]
    for _ in range(epochs):
        for ids in torch.randperm(len(x),device=x.device).split(256):
            dist=model.distribution(x[ids],g[ids]); log=dist.log_prob(act[ids])
            logratio=log-oldlog[ids];ratio=logratio.exp()
            policy=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
            value=(model.value(x[ids],g[ids],z[ids])-returns[ids]).square().mean()
            pred=model.outcome_logit(x[ids],g[ids])
            outcome=(torch.nn.functional.binary_cross_entropy_with_logits(pred,targets[ids],reduction='none')*episode_weight[ids]).mean()
            entropy=dist.entropy().mean()
            loss=policy+.25*value-.001*entropy+(.2*outcome if objective=='completion' else 0.)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
            optimizer.zero_grad();loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            kl=((ratio-1)-logratio).mean()
            stats.append([float(v.detach()) for v in (policy,value,outcome,entropy,kl,norm)])
    means=torch.tensor(stats).mean(0).tolist()
    return dict(samples=len(x),**dict(zip(('policy_loss','value_loss','outcome_bce','entropy','approx_kl','grad_norm'),means)))
