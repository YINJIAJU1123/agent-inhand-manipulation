"""A learned completion/progress head conditioned on the agent's goal hypothesis."""
import torch
from torch import nn
THRESHOLDS=(.08,.12,.16,.22,.32,.5,1.)
class CompletionModel(nn.Module):
    def __init__(self,dim=731,kind="mlp"):
        super().__init__();self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.kind=kind
        self.body=nn.Identity() if kind=='linear' else nn.Sequential(nn.Linear(dim,192),nn.LayerNorm(192),nn.SiLU(),nn.Dropout(.1),nn.Linear(192,96),nn.SiLU())
        width=dim if kind=='linear' else 96
        self.ordinal=nn.Linear(width,len(THRESHOLDS));self.error=nn.Linear(width,1);self.progress=nn.Linear(width,1)
    def forward(self,x):
        z=self.body(((x-self.mean)/self.scale).clamp(-8,8))
        return dict(ordinal=self.ordinal(z),log_error=self.error(z).squeeze(-1),progress=self.progress(z).squeeze(-1))
def load_model(path,device):
    p=torch.load(path,map_location=device,weights_only=True);m=CompletionModel(kind=p.get("kind","mlp")).to(device);m.load_state_dict(p['model']);m.eval().requires_grad_(False);return m,p
