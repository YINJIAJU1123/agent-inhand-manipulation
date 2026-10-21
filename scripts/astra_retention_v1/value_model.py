"""Learned action values for retaining a visually grounded target.

Inputs are frozen RGB-D policy/evidence features. Ground-truth pose, target
labels, seed, and future response never enter forward().
"""
import torch
from torch import nn
RATES=(1.,.25,0.)

class RetentionValue(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',torch.zeros(125));self.register_buffer('scale',torch.ones(125))
  self.net=nn.Sequential(nn.Linear(125,32),nn.LayerNorm(32),nn.SiLU(),nn.Dropout(.15),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,6))
 def forward(self,x):
  if x.shape[-1]!=125:raise ValueError('Expected measured frozen features')
  y=self.net(((x-self.mean)/self.scale).clamp(-8,8));return dict(value=y[...,:3],drop_logit=y[...,3:])

@torch.inference_mode()
def frozen_features(model,data,goal):
 captured=[];hook=model.query.register_forward_hook(lambda module,args,value:captured.append(value))
 try:
  output=model(data['x'],data['roi_rgb'],data['roi_depth'],data['roi_box'],data['roi_found'])
 finally:hook.remove()
 batch=torch.arange(len(goal),device=goal.device)
 result=torch.cat((captured[0][batch,goal],output['ordinal'][batch,goal],output['log_error'][batch,goal,None],data['x'][batch,goal,533:554]),-1)
 if result.shape[-1]!=125 or not torch.isfinite(result).all():raise RuntimeError('Invalid measured features')
 return result
