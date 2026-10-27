"""Frozen temporal stop architecture from the completed paper study."""
import torch
from torch import nn
FRAME_DIM=125
WINDOW=3
BASE_LOGIT=98

class RichStop(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(FRAME_DIM*WINDOW))
        self.register_buffer('scale',torch.ones(FRAME_DIM*WINDOW))
        self.net=nn.Sequential(nn.Linear(FRAME_DIM*WINDOW,96),nn.LayerNorm(96),nn.SiLU(),
                               nn.Linear(96,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,features):
        if features.shape[-1]!=FRAME_DIM*WINDOW:raise ValueError('Invalid history width')
        base=features[..., (WINDOW-1)*FRAME_DIM+BASE_LOGIT]
        correction=self.net(((features-self.mean)/self.scale).clamp(-8,8)).squeeze(-1)
        return base+4*torch.tanh(correction/4)
