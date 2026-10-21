"""15 Hz inference, immediate refresh at asynchronous phase/goal boundaries."""
import torch
from recovery_controller_v2 import observation,bridge_fraction
from recovery_controller_v3 import residual_command

class ControllerRuntime:
    def __init__(self,model,initial,learned):
        self.model,self.learned=model,learned
        n=len(initial);device=initial.device
        self.anchor=initial.clone();self.previous=initial.clone()
        self.history=torch.zeros(n,4,100,device=device)
        self.delivery=torch.full((n,),-100000,device=device,dtype=torch.long)
        self.wait_age=torch.zeros(n,device=device)
        self.was_pending=torch.zeros(n,device=device,dtype=torch.bool)
        self.raw=torch.zeros(n,22,device=device)
        self.last_goal=None;self.initialized=False
    def update_visual(self,compact):
        if not self.initialized:self.history[:]=compact[:,None];self.initialized=True
        else:self.history=torch.cat((self.history[:,1:],compact[:,None]),1)
    def action(self,step,goal,prop,proposal,current,pending,known,delivered,request_age):
        delivered=torch.as_tensor(delivered,dtype=torch.bool,device=current.device)
        self.delivery[delivered]=step
        started=pending & ~self.was_pending
        self.anchor[started]=current[started]
        self.wait_age[delivered]=request_age[delivered]
        age=torch.where(pending,request_age,self.wait_age)
        response_age=torch.where(pending,0.,(step-self.delivery).float()/30).clamp_min(0)
        eligible=known & (pending | (response_age<2))
        obs=observation(self.history,prop,proposal,current,self.anchor,self.previous,age,pending,response_age.clamp(max=2))
        changed=delivered|started
        if self.last_goal is not None:changed|=(goal!=self.last_goal).any(-1)
        refresh=eligible & (changed | (step%2==0))
        if refresh.any():
            self.raw[refresh]=self.model.raw_action(obs[refresh],goal[refresh]) if self.learned else self.model.distribution(obs[refresh],goal[refresh]).mean
        action,rate,residual=residual_command(proposal,current,self.raw,bridge_fraction(response_age,pending),pending)
        action=torch.where(eligible[:,None],action,proposal)
        diagnostic=dict(step=step,controller_active=eligible.cpu().tolist(),pending=pending.cpu().tolist(),known=known.cpu().tolist(),
            response_age=response_age.cpu().tolist(),rate=rate.cpu().tolist(),raw_refreshed=refresh.cpu().tolist())
        if self.learned and step%30==0:diagnostic['empirical_outcome_probability']=self.model.diagnostic(obs,goal).cpu().tolist()
        self.was_pending=pending.clone();self.last_goal=goal.clone()
        return action,diagnostic
    def commit(self,actions):self.previous=actions.clone()
