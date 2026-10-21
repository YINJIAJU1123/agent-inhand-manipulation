"""Query-conditioned visual evidence, with no simulator inputs."""
import torch
from torch import nn

class SymbolEvidence(nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('colors',torch.tensor([[.85,.05,.05],[.05,.75,.15],[.05,.25,.90],[.95,.75,.05],[.80,.05,.75],[.05,.80,.85]]))
  layers=[]
  for n,m in [(5,16),(16,24),(24,32),(32,48)]:layers.extend([nn.Conv2d(n,m,3,stride=2,padding=1),nn.GroupNorm(4,m),nn.SiLU()])
  self.encoder=nn.Sequential(*layers,nn.Flatten(),nn.Linear(48*3*3,96),nn.SiLU())
  self.head=nn.Sequential(nn.Linear(107,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,7))
 def forward(self,patches,geometry,colors):
  if patches.shape[1:]!=(3,48,48) or geometry.shape[-1]!=5:raise ValueError('Wrong sensor shape')
  if colors.dtype!=torch.long or colors.min()<0 or colors.max()>5:raise ValueError('Wrong public color query')
  image=patches.float();distance=(image-self.colors[colors,:,None,None]).square().mean(1,keepdim=True)
  match=torch.exp(-distance/.035)
  ink=(1-image.max(1,keepdim=True).values)*(1-(image.max(1,keepdim=True).values-image.min(1,keepdim=True).values))
  encoded=self.encoder(torch.cat((image,match,ink),1))
  return self.head(torch.cat((encoded,geometry,torch.nn.functional.one_hot(colors,6).float()),-1))
