"""Separate readability from symbol identity using measured query-conditioned crops."""
import torch
from torch import nn
from torch.nn import functional as F

class SymbolEvidence(nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('colors',torch.tensor([[.85,.05,.05],[.05,.75,.15],[.05,.25,.90],[.95,.75,.05],[.80,.05,.75],[.05,.80,.85]]))
  layers=[]
  for n,m in [(5,16),(16,24),(24,32),(32,48)]:layers.extend([nn.Conv2d(n,m,3,stride=2,padding=1),nn.GroupNorm(4,m),nn.SiLU()])
  self.encoder=nn.Sequential(*layers,nn.Flatten(),nn.Linear(48*3*3,96),nn.SiLU())
  self.symbol=nn.Sequential(nn.Linear(107,96),nn.SiLU(),nn.Dropout(.15),nn.Linear(96,6))
  self.readability=nn.Sequential(nn.Linear(107,32),nn.SiLU(),nn.Linear(32,1))
 def forward(self,patches,geometry,colors):
  if patches.shape[1:]!=(3,48,48) or geometry.shape[-1]!=5:raise ValueError('Wrong sensor shape')
  image=patches.float();distance=(image-self.colors[colors,:,None,None]).square().mean(1,keepdim=True);match=torch.exp(-distance/.035)
  saturation=image.max(1,keepdim=True).values-image.min(1,keepdim=True).values;ink=(1-image.max(1,keepdim=True).values)*(1-saturation)
  z=torch.cat((self.encoder(torch.cat((image,match,ink),1)),geometry,F.one_hot(colors,6).float()),-1)
  visible=self.readability(z);symbol=self.symbol(z)
  return torch.cat((F.logsigmoid(visible)+F.log_softmax(symbol,-1),F.logsigmoid(-visible)),-1)
