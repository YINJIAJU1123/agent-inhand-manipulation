"""Sensor-only learned assistance around the frozen Astra tool supervisor."""
import numpy as np
import torch
from common import COLORS,SYMBOLS,color_patches
from model import load_model
from astra_agent_v4 import AstraAgent
from visual_agent import Decision


def language_symbol(instruction):
    # Text-only interpretation of the closed benchmark vocabulary, never a
    # case index, layout or target-color lookup. Unknown language stays unknown.
    aliases=(("star","five pointed"),("circle","round-symbol"),
             ("triangle","three-cornered"),("cross","plus-sign"),
             ("diamond","rhombus"),("bars","two parallel"))
    text=instruction.lower()
    found=[i for i,words in enumerate(aliases) if any(w in text for w in words)]
    return found[0] if len(found)==1 else None


class FeedbackClient:
    def __init__(self,base):
        self.base=base
        self.context={}
    def ask(self,instruction,rgb,feedback,*history):
        merged=dict(feedback,**self.context)
        return self.base.ask(instruction,rgb,merged,*history)


class EvidenceRuntime:
    def __init__(self,path,device,n):
        self.model,self.pack=load_model(path,device)
        self.ph=self.gh=None
        self.bank=torch.zeros(n,6,6,device=device)
        self.visits=torch.zeros(n,6,device=device)
        self.ready_threshold=self.pack['thresholds']['ready']
        self.symbol_threshold=self.pack['thresholds']['symbol']
        self.last=None
    @torch.inference_mode()
    def update(self,rgb,features,proprio):
        patches,geo=color_patches(rgb)
        if self.ph is None:
            self.ph=patches[:,None].repeat(1,4,1,1,1,1)
            self.gh=geo[:,None].repeat(1,4,1,1)
        else:
            self.ph=torch.cat((self.ph[:,1:],patches[:,None]),1)
            self.gh=torch.cat((self.gh[:,1:],geo[:,None]),1)
        self.last=self.model(self.ph,self.gh,features[:,-4:],proprio[:,-4:])
        prob=self.last['symbol'].softmax(-1)
        # Retain the latest confident observation. Repeated views do not
        # multiply confidence. Contradictory new evidence can replace memory.
        reliable=(prob[...,:6].max(-1).values>=self.symbol_threshold)
        self.bank[reliable]=prob[...,:6][reliable]
    def can_stop(self,i,color,symbol,instruction):
        requested=language_symbol(instruction)
        if color is None or requested is None or symbol!=SYMBOLS[requested]:return False
        binding=float(self.bank[i,color,requested])
        ready=float(self.last['ready'][i,color].sigmoid())
        return binding>=self.symbol_threshold and ready>=self.ready_threshold
    def inspection(self,i,instruction):
        target=language_symbol(instruction)
        if target is None:return None,None
        belief=self.bank[i]
        target_mass=belief[:,target]+(1-belief.sum(-1)).clamp_min(0)/6
        target_mass=target_mass/target_mass.sum().clamp_min(1e-6)
        future=self.last['future_visible'][i].sigmoid()
        survival=self.last['survival'][i].sigmoid()
        unseen=(1-belief.max(-1).values).clamp(0,1)
        gain=(future*(unseen*target_mass)[None]).sum(-1)
        values=gain-.25*(1-survival)-.02*self.visits[i].clamp(max=3)
        return int(values.argmax()),values.cpu().tolist()
    def summary(self,i,verification,observation):
        d={}
        if verification:
            d['local_verifier']={
                'symbol_probabilities_by_color':self.bank[i].cpu().tolist(),
                'pose_completion_probability_by_color':self.last['ready'][i].sigmoid().cpu().tolist(),
                'warning':'Learned sensor estimates, not a success sensor; stale or occluded views can be uncertain.'}
        if observation:
            d['observation_memory']={
                COLORS[c]:{'symbol':SYMBOLS[int(self.bank[i,c].argmax())] if float(self.bank[i,c].max())>=self.symbol_threshold else 'unknown',
                           'confidence':float(self.bank[i,c].max()),'executed_inspections':int(self.visits[i,c])}
                for c in range(6)}
        return d


def reject_stop(wrapper,step):
    """Only called by the physics owner after Future has been consumed."""
    assert wrapper.future is None
    color=wrapper.snapshot['decision']['color']
    decision=Decision(color,'execute','learned verifier requests more current evidence')
    wrapper.snapshot['decision']=vars(decision).copy()
    wrapper.agent.last_decision=decision
    wrapper.agent.declared_success=False
    wrapper.agent.next_step=step+30
    wrapper.due=step+30


def override_inspection(wrapper,color,step):
    assert wrapper.future is None
    decision=Decision(color,'inspect','learned 90-step observation-value selection')
    wrapper.snapshot['decision']=vars(decision).copy()
    wrapper.agent.last_decision=decision
    wrapper.agent.next_step=step+90
    wrapper.due=step+90
