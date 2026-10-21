"""Sensor-only PPO supervisor for a frozen student during delayed verification.

Modes: continue, interpolate position target, latch and hold position target.
The hold primitive is prescribed, not a learned stabilizing policy.
"""
import torch
from torch import nn
from torch.distributions import Categorical

MODES = ('continue', 'slow', 'hold')


def normalized_target(raw):
    ids = raw.actuated_dof_indices
    lo, hi = raw.hand_dof_lower_limits[:, ids], raw.hand_dof_upper_limits[:, ids]
    return ((2 * raw.prev_targets[:, ids] - lo - hi) / (hi - lo)).clamp(-1, 1)


def command(student, current, anchor, mode, previous_mode, slow_rate=.15):
    entering = (mode == 2) & (previous_mode != 2)
    anchor = torch.where(entering[:, None], current, anchor)
    slow = current + slow_rate * (student - current)
    action = torch.where((mode == 0)[:, None], student,
                         torch.where((mode == 1)[:, None], slow, anchor))
    return action.clamp(-1, 1), anchor


def sensor_observation(visual, previous_visual, proprio, proposal, current, anchor, mode, age):
    """No object state, scorer, target label, or remaining delay is accepted."""
    age_features = torch.stack((age / 16., torch.ones_like(age)), -1)
    obs = torch.cat((visual, visual - previous_visual, proprio,
                     proposal, current, anchor,
                     torch.nn.functional.one_hot(mode, 3).float(), age_features), -1)
    return torch.nan_to_num(obs, nan=0., posinf=10., neginf=-10.).clamp(-10, 10)


class Supervisor(nn.Module):
    def __init__(self, obs_dim, privileged_dim=8):
        super().__init__()
        self.obs_dim, self.privileged_dim = obs_dim, privileged_dim
        self.actor = nn.Sequential(nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.Tanh(),
                                   nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 3))
        self.critic = nn.Sequential(nn.Linear(obs_dim + privileged_dim, 128), nn.Tanh(),
                                    nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        with torch.no_grad(): self.actor[-1].bias.copy_(torch.tensor([.3, 0., 0.]))

    def distribution(self, obs): return Categorical(logits=self.actor(obs))
    def value(self, obs, privileged): return self.critic(torch.cat((obs, privileged), -1)).squeeze(-1)


def ppo_update(model, optimizer, rollout, epochs=5):
    obs, priv, actions, oldlog, values, rewards, done, valid = [torch.stack([r[k] for r in rollout]) for k in
        ('obs', 'priv', 'action', 'logprob', 'value', 'reward', 'done', 'valid')]
    advantages = torch.zeros_like(rewards); carry = torch.zeros_like(rewards[0])
    # One decision spans 0.5 s. Last reward includes a frozen-student recovery rollout.
    for t in reversed(range(len(rollout))):
        next_value = values[t+1] if t+1 < len(rollout) else torch.zeros_like(carry)
        delta = rewards[t] + .995 * next_value * (~done[t]) - values[t]
        carry = delta + .995 * .95 * (~done[t]) * carry
        advantages[t] = carry
    returns = (advantages + values)[valid]
    x, z, act, log0, adv = obs[valid], priv[valid], actions[valid], oldlog[valid], advantages[valid]
    if len(x) < 2: raise RuntimeError('Too few surviving request states for training')
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    stats = []
    for _ in range(epochs):
        for ids in torch.randperm(len(x), device=x.device).split(256):
            dist = model.distribution(x[ids]); log = dist.log_prob(act[ids]); ratio = (log-log0[ids]).exp()
            policy = -torch.minimum(ratio*adv[ids], ratio.clamp(.8, 1.2)*adv[ids]).mean()
            value = (model.value(x[ids], z[ids])-returns[ids]).square().mean()
            entropy = dist.entropy().mean()
            loss = policy + .25*value - .02*entropy
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            stats.append([float(loss.detach()), float(entropy.detach()), float((log0[ids]-log).mean().detach())])
    return {'samples':len(x), 'loss_entropy_approxkl':torch.tensor(stats).mean(0).tolist()}
