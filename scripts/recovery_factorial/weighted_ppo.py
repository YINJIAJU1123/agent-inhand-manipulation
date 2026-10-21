"""Explicit weighted PPO without duplicating episodes or increasing passes.

All arms have the same 3-pass rule and batch size. Valid-decision counts and
KL-triggered early stopping can still differ and are reported. No claim of equal
accepted optimizer steps is made.
"""
import copy
import torch
from torch import nn
from torch.distributions import Independent,Normal,kl_divergence
from recovery_controller_v3 import update as ordinary_update
from recovery_controller_v2 import advantages


def weighted_mean(value,weight):
 return (value*weight).sum()/weight.sum().clamp_min(1e-8)


def update(model,optimizers,rollout,success,episode_weights,epochs=3,target_kl=.02):
 weights=torch.as_tensor(episode_weights,dtype=rollout[0]['obs'].dtype,device=rollout[0]['obs'].device)
 if weights.shape!=rollout[0]['valid'].shape or (weights<=0).any() or not torch.isfinite(weights).all():raise ValueError('Invalid episode weights')
 if torch.all(weights==weights[0]):
  stats=ordinary_update(model,optimizers,rollout,success,epochs=epochs,target_kl=target_kl,replay=None)
  stats.update(weighted_objective=False,unique_samples=stats['samples'],effective_samples=float(stats['samples']),episode_weight_min=1.,episode_weight_max=1.)
  return stats
 keys=('obs','goal','priv','action','logprob','value','reward','done','valid','mean','std')
 x,g,z,act,oldlog,values,rewards,done,valid,oldmean,oldstd=[torch.stack([r[k] for r in rollout]) for k in keys]
 adv=advantages(rewards,values,done,valid);returns=(adv+values)[valid]
 w=weights[None].expand_as(valid)[valid]
 x,g,z,act,oldlog,oldmean,oldstd,adv=[v[valid] for v in (x,g,z,act,oldlog,oldmean,oldstd,adv)]
 if len(x)<2:raise RuntimeError('Too few controller decisions')
 mean=weighted_mean(adv,w);variance=weighted_mean((adv-mean).square(),w)
 correction=1.-w.square().sum()/w.sum().square()
 adv=(adv-mean)/(variance/correction.clamp_min(1e-8)).sqrt().clamp_min(1e-6)
 value_losses=[]
 for _ in range(epochs):
  for ids in torch.randperm(len(x),device=x.device).split(256):
   loss=weighted_mean(((model.value(x[ids],g[ids],z[ids])-returns[ids])/12.).square(),w[ids])
   optimizers['value'].zero_grad();loss.backward()
   nn.utils.clip_grad_norm_(model.parameters_for('value'),1.,error_if_nonfinite=True);optimizers['value'].step()
   value_losses.append(float(loss.detach()))
 policy_losses=[];kls=[];rolled_back=False;olddist=Independent(Normal(oldmean,oldstd),1);params=model.parameters_for('policy')
 for _ in range(epochs):
  if rolled_back:break
  for ids in torch.randperm(len(x),device=x.device).split(256):
   dist=model.distribution(x[ids],g[ids]);ratio=(dist.log_prob(act[ids])-oldlog[ids]).exp()
   loss=weighted_mean(-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids])-.001*dist.entropy(),w[ids])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite policy loss')
   before=[p.detach().clone() for p in params];state=copy.deepcopy(optimizers['policy'].state_dict())
   optimizers['policy'].zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizers['policy'].step()
   with torch.no_grad():kl=float(kl_divergence(olddist,model.distribution(x,g)).mean())
   if not torch.isfinite(torch.tensor(kl)) or kl>target_kl:
    with torch.no_grad():
     for p,v in zip(params,before):p.copy_(v)
    optimizers['policy'].load_state_dict(state);rolled_back=True;break
   policy_losses.append(float(loss.detach()));kls.append(kl)
 with torch.no_grad():final=float(kl_divergence(olddist,model.distribution(x,g)).mean())
 if final>target_kl+1e-6:raise RuntimeError('KL guard failed')
 return dict(samples=len(x),unique_samples=len(x),effective_samples=float(w.sum().square()/w.square().sum()),weighted_objective=True,
  episode_weight_min=float(weights.min()),episode_weight_max=float(weights.max()),policy_loss=sum(policy_losses)/max(len(policy_losses),1),
  value_normalized_loss=sum(value_losses)/len(value_losses),value_optimizer_steps=len(value_losses),
  final_exact_kl=final,max_accepted_kl=max(kls,default=0.),accepted_policy_steps=len(kls),kl_rollback=rolled_back)
