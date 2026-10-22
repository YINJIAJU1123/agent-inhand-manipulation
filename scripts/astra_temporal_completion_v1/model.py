"""Goal-conditioned completion from measured dense features and their causal history."""
import torch
from torch import nn


class TemporalCompletion(nn.Module):
    def __init__(self, temporal=True):
        super().__init__();self.temporal=temporal
        self.register_buffer('mean',torch.zeros(125));self.register_buffer('scale',torch.ones(125))
        self.encoder=nn.Sequential(nn.Linear(125,48),nn.LayerNorm(48),nn.SiLU())
        if temporal:self.recurrent=nn.GRU(48,32,batch_first=True)
        else:self.static=nn.Sequential(nn.Linear(48,32),nn.SiLU())
        self.ready=nn.Linear(32,1);self.error=nn.Linear(32,1)

    def forward(self,x):
        # [batch, causal time, candidate goal, measured feature]
        if x.ndim!=4 or x.shape[2:]!=(6,125):raise ValueError('Expected B,T,6,125 measured features')
        b,t=x.shape[:2];z=self.encoder(((x-self.mean)/self.scale).clamp(-8,8))
        if self.temporal:
            z=z.permute(0,2,1,3).reshape(b*6,t,48)
            _,h=self.recurrent(z);h=h[-1].reshape(b,6,32)
        else:h=self.static(z[:,-1])
        return dict(ready=self.ready(h).squeeze(-1),log_error=self.error(h).squeeze(-1))
