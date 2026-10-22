"""Frozen dense goal features and temporally extended continuation actions."""
import copy
import torch
from torch import nn
from torch.distributions import Categorical,kl_divergence
from recovery_controller_v3 import Encoder
from recovery_controller_v2 import advantages

class EvidenceEncoder(nn.Module):
    def __init__(self, goal_dim):
        super().__init__()
        self.base = Encoder(616, goal_dim)
        self.evidence = nn.Sequential(nn.Linear(125, 96), nn.LayerNorm(96), nn.Tanh(), nn.Linear(96, 128))
        nn.init.zeros_(self.evidence[-1].weight)
        nn.init.zeros_(self.evidence[-1].bias)
    def forward(self, x, goal):
        if x.shape[-1] != 741:
            raise ValueError('Expected 616 measured control features and 125 dense target features')
        return self.base(x[:, :616], goal) + self.evidence(x[:, 616:])

class ModeController(nn.Module):
    def __init__(self,obs_dim=741,goal_dim=768):
        super().__init__();self.policy_encoder=EvidenceEncoder(goal_dim);self.mode=nn.Linear(128,3)
        self.value_encoder=EvidenceEncoder(goal_dim);self.critic=nn.Sequential(nn.Linear(136,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.zeros_(self.mode.weight)
        with torch.no_grad():self.mode.bias.copy_(torch.tensor([.8,.1,.1]).log())
    def initialize(self,old):
        selected={}
        for k,v in old.items():
            if k.startswith(('policy_encoder.','value_encoder.')):
                part,rest=k.split('.',1);selected[part+'.base.'+rest]=v
            elif k.startswith('critic.'):selected[k]=v
        missing,unexpected=self.load_state_dict(selected,strict=False)
        assert all(k in {'mode.weight','mode.bias'} or '.evidence.' in k for k in missing) and not unexpected
    def distribution(self,x,g):return Categorical(logits=self.mode(self.policy_encoder(x,g)))
    def value(self,x,g,z):return 12*self.critic(torch.cat((self.value_encoder(x,g),z),-1)).squeeze(-1)
    def parameters_for(self,part):
        return list(self.policy_encoder.parameters())+list(self.mode.parameters()) if part=='policy' else list(self.value_encoder.parameters())+list(self.critic.parameters())

def optimizers(model):
    groups={k:model.parameters_for(k) for k in ['policy','value']}
    assert not {id(p) for p in groups['policy']}&{id(p) for p in groups['value']}
    return {k:torch.optim.Adam(v,lr=1e-4 if k=='policy' else 3e-4) for k,v in groups.items()}

def update(model,opts,rollout,epochs=3,max_kl=.03):
    keys=('obs','goal','priv','mode','logprob','logits','value','reward','done','valid')
    x,g,z,mode,oldlog,logits,value,reward,done,valid=[torch.stack([r[k] for r in rollout]) for k in keys]
    adv=advantages(reward,value,done,valid);returns=(adv+value)[valid]
    x,g,z,mode,oldlog,logits,adv=[t[valid] for t in (x,g,z,mode,oldlog,logits,adv)]
    if len(x)<2:raise RuntimeError('Insufficient valid continuation states')
    adv=(adv-adv.mean())/adv.std().clamp_min(1e-6);values=[];policies=[];kls=[];rollback=False
    for _ in range(epochs):
        for ids in torch.randperm(len(x),device=x.device).split(256):
            loss=((model.value(x[ids],g[ids],z[ids])-returns[ids])/12).square().mean()
            opts['value'].zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters_for('value'),1.,error_if_nonfinite=True);opts['value'].step();values.append(float(loss.detach()))
    old=Categorical(logits=logits);params=model.parameters_for('policy')
    for _ in range(epochs):
        if rollback:break
        for ids in torch.randperm(len(x),device=x.device).split(256):
            dist=model.distribution(x[ids],g[ids]);ratio=(dist.log_prob(mode[ids])-oldlog[ids]).exp()
            loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()-.003*dist.entropy().mean()
            before=[p.detach().clone() for p in params];state=copy.deepcopy(opts['policy'].state_dict())
            opts['policy'].zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opts['policy'].step()
            with torch.no_grad():kl=float(kl_divergence(old,model.distribution(x,g)).mean())
            if not torch.isfinite(torch.tensor(kl)) or kl>max_kl:
                with torch.no_grad():
                    for p,v in zip(params,before):p.copy_(v)
                opts['policy'].load_state_dict(state);rollback=True;break
            policies.append(float(loss.detach()));kls.append(kl)
    with torch.no_grad():final=float(kl_divergence(old,model.distribution(x,g)).mean())
    assert final<=max_kl+1e-6
    return dict(samples=len(x),value_loss=sum(values)/len(values),policy_loss=sum(policies)/max(1,len(policies)),accepted_policy_steps=len(kls),final_kl=final,kl_rollback=rollback)

def command(mode,baseline,proposal,current):
    # Mode 0 preserves the frozen V3/student baseline bit for bit.
    slow=(current+.15*(proposal-current)).clamp(-1,1)
    return torch.where((mode==0)[:,None],baseline,torch.where((mode==1)[:,None],slow,current))
