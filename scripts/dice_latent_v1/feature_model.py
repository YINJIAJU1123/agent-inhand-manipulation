"""Direct latent actor with learned teacher-feature distillation.

The deployed interface is physical history plus continuous latent history.
A small neural adapter reconstructs eight normalized conditioning features.
No goal index, target quaternion, lookup or relative-goal oracle is supplied.
"""
import torch
from torch import nn
from viserdex.supermimic.networks import Actor,mlp,run_mlp
from viserdex.supermimic.core import Config
from model import PHYSICAL_COLUMNS,TARGET_COLUMNS

FORMAT='dice_supermimic_feature_latent_v3'

class FeatureLatentActor(nn.Module):
    def __init__(self,config,latent_dim=768):
        super().__init__();self.config=config;self.latent_dim=latent_dim;self.amp=config.amp
        reference=Actor(config);self.obs_norm=reference.obs_norm;self.net=reference.net
        self.register_buffer('goal_center',torch.zeros(latent_dim));self.register_buffer('goal_scale',torch.ones(latent_dim))
        self.goal_encoder=nn.Sequential(nn.Linear(latent_dim,32),nn.SiLU())
        self.relation=nn.Sequential(nn.Linear(32+4+32*4,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,8))
    @torch.no_grad()
    def configure_goal_statistics(self,bank):
        self.goal_center.copy_(bank.mean(0));self.goal_scale.copy_((bank-bank.mean(0)).square().mean(0).sqrt().clamp_min(.005))
    def conditioning_features(self,physical,goal):
        g=self.goal_encoder(((goal-self.goal_center)/self.goal_scale).clamp(-8,8))
        q=physical[...,112:116]
        interaction=(g.unsqueeze(-1)*q.unsqueeze(-2)).flatten(-2)
        return self.relation(torch.cat((g,q,interaction),-1))
    def normalized_physical(self,physical):
        return (physical-self.obs_norm.mean[list(PHYSICAL_COLUMNS)])/self.obs_norm.std[list(PHYSICAL_COLUMNS)]
    def action_with_features(self,physical,conditioning,return_features=False):
        full=physical.new_empty((*physical.shape[:-1],124))
        full[...,list(PHYSICAL_COLUMNS)]=self.normalized_physical(physical)
        full[...,list(TARGET_COLUMNS)]=conditioning
        x=full.flatten(1)
        if return_features:
            features=run_mlp(self.net[:-1],x,self.amp)
            return dict(action=run_mlp(self.net[-1:],features,self.amp).tanh(),features=features)
        return run_mlp(self.net,x,self.amp).tanh()
    def forward(self,physical,goal,return_features=False):
        if physical.shape[1:]!=(10,116) or goal.shape!=(*physical.shape[:2],self.latent_dim):raise ValueError('History shape mismatch')
        return self.action_with_features(physical,self.conditioning_features(physical,goal),return_features)

@torch.no_grad()
def migrate(payload,bank,seed=186205):
    if payload['version']!=17 or payload['config']['history']!=10:raise ValueError('Expected published v17 actor')
    torch.manual_seed(seed);cfg=Config(**payload['config']);model=FeatureLatentActor(cfg)
    reference=Actor(cfg);reference.load_state_dict(payload['actor'],strict=True)
    model.net.load_state_dict(reference.net.state_dict(),strict=True)
    model.obs_norm.load_state_dict(reference.obs_norm.state_dict(),strict=True)
    model.configure_goal_statistics(bank)
    model.requires_grad_(False);model.goal_encoder.requires_grad_(True);model.relation.requires_grad_(True)
    return model,dict(format=FORMAT,original_actor_parameters=sum(p.numel() for p in reference.parameters()),
        copied_actor_parameters=sum(p.numel() for p in reference.parameters()),
        adaptation='learned normalized conditioning-feature reconstruction, source actor frozen',
        inputs=['physical_history_10x116','continuous_goal_history_10x768'],
        inference_goal_index=False,inference_target_quaternion=False,inference_relative_quaternion=False,
        discrete_latent_lookup=False)
