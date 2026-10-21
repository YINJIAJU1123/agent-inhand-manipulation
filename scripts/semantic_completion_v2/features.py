"""Target-conditioned frozen policy features. Accepts measured sensors only."""
import importlib.util,sys
import torch
from torch import nn
from assets import ROOT,STUDENT,ADAPTER,sha
path=ROOT/'source/BrainCo_DexHand/BrainCo_DexHand/algo/agentic/visual_student.py'
spec=importlib.util.spec_from_file_location('completion_frozen_student',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

class FrozenFeatures(nn.Module):
    dimension=731
    def __init__(self,device='cpu'):
        super().__init__()
        pack=torch.load(STUDENT,map_location='cpu',weights_only=True)
        self.history=pack['history']
        self.policy=module.VisualLanguageStudent(pack['rgb_dim'],pack['language_dim'],pack['proprio_dim'],pack['action_dim'],memory_mode=pack['memory_mode'])
        self.policy.load_state_dict(pack['model']);self.policy.eval().requires_grad_(False)
        text=torch.load(ADAPTER,map_location='cpu',weights_only=True)['canonical_embeddings'].float()
        self.register_buffer('text',text)
        norm=nn.functional.normalize(text,dim=-1);center=norm.mean(0);_,s,v=torch.linalg.svd(norm-center,full_matrices=False)
        self.register_buffer('query',(norm-center)@v[:5].T/s[:5].clamp_min(1e-5))
        self.to(device);self.requires_grad_(False)
    @torch.no_grad()
    def forward(self,rgb_history,proprio_history,colors):
        if rgb_history.shape[-1]!=1060 or proprio_history.shape[-1]!=128:raise ValueError('Wrong sensor contract')
        if rgb_history.shape[1]!=self.history:raise ValueError('Frozen policy history mismatch')
        output=self.policy(module.VisualStudentBatch(rgb_history,self.text[colors],proprio_history))
        m=output['memory'];a=output['action'];last=proprio_history[:,-1]
        markers=rgb_history[...,-24:].reshape(-1,self.history,6,4)
        ids=torch.arange(len(colors),device=colors.device)
        target=markers[ids,-1,colors];delta=target-markers[ids,0,colors]
        base=torch.cat((rgb_history[:,-1,:12],rgb_history[:,-1,-24:]),-1)
        x=torch.cat((m[:,-1],m[:,-1]-m[:,0],a,a-last[:,:21],last,base,self.query[colors],target,delta),-1)
        assert x.shape[-1]==self.dimension
        return x
    @torch.no_grad()
    def candidates(self,rgb_history,proprio_history):
        n,t,_=rgb_history.shape
        rgb=rgb_history[:,None].expand(-1,6,-1,-1).reshape(n*6,t,-1)
        prop=proprio_history[:,None].expand(-1,6,-1,-1).reshape(n*6,t,-1)
        colors=torch.arange(6,device=rgb.device).repeat(n)
        return self(rgb,prop,colors).reshape(n,6,-1)
