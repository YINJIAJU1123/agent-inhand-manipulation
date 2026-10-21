"""Sensor-only temporal representation and bounded correction to frozen V3.

The latent belongs to this local model, not Astra. Empirical outcome predictions
are diagnostic outputs and never an autonomous-stop or safety certificate.
"""
import torch
from torch import nn
from recovery_controller_v3 import RecoveryController

class TemporalState(nn.Module):
    def __init__(self,canonical):
        super().__init__()
        c=nn.functional.normalize(canonical.float(),dim=-1)
        center=c.mean(0);_,s,v=torch.linalg.svd(c-center,full_matrices=False)
        self.register_buffer('goal_center',center)
        self.register_buffer('goal_basis',v[:5].T/s[:5].clamp_min(1e-4))
        self.register_buffer('obs_mean',torch.zeros(616))
        self.register_buffer('obs_scale',torch.ones(616))
        self.frame=nn.Sequential(nn.Linear(100,64),nn.LayerNorm(64),nn.Tanh())
        self.temporal=nn.GRU(64,64,batch_first=True)
        self.state=nn.Sequential(nn.Linear(216,96),nn.LayerNorm(96),nn.Tanh())
        self.goal=nn.Sequential(nn.Linear(5,32),nn.Tanh())
        self.fusion=nn.Sequential(nn.Linear(192,128),nn.LayerNorm(128),nn.Tanh())
        self.outcome=nn.Linear(128,3)
        self.value=nn.Linear(128,1)
    def forward(self,obs,goal):
        if obs.shape[-1]!=616 or goal.shape[-1]!=768:raise ValueError('Wrong sensor/goal shape')
        x=((obs-self.obs_mean)/self.obs_scale).clamp(-8,8)
        h=self.temporal(self.frame(x[:,:400].reshape(-1,4,100)))[1][-1]
        g=(nn.functional.normalize(goal,dim=-1)-self.goal_center)@self.goal_basis
        return self.fusion(torch.cat((h,self.state(x[:,400:]),self.goal(g)),-1))
    def predictions(self,obs,goal):
        z=self(obs,goal)
        return self.outcome(z),self.value(z).squeeze(-1)

class LearnedController(nn.Module):
    def __init__(self,canonical):
        super().__init__()
        self.reference=RecoveryController(616)
        self.encoder=TemporalState(canonical)
        self.correction=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,22))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
        self.reference.requires_grad_(False)
    def correction_from_latent(self,z):return .1*self.correction(z).tanh()
    def raw_action(self,obs,goal):
        base=self.reference.distribution(obs,goal).mean
        return base+self.correction_from_latent(self.encoder(obs,goal))
    def diagnostic(self,obs,goal):return self.encoder.predictions(obs,goal)[0].softmax(-1)

def load_controller(path,device):
    pack=torch.load(path,map_location=device,weights_only=True)
    model=LearnedController(pack['canonical'].to(device)).to(device)
    model.load_state_dict(pack['model']);model.eval().requires_grad_(False)
    return model,pack
